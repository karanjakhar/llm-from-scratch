"""Stage 05 experiments. docs/05_transformer_block.md §10.

    uv run python stages/05_block/run.py grads            # 10a, seconds
    uv run python stages/05_block/run.py depth            # 10b-c, pre/post/no-residual at 4 and 24 layers
    uv run python stages/05_block/run.py sweep            # 10d, norm / activation / MLP / dropout, 3 seeds
    uv run python stages/05_block/run.py speed            # 10e, LayerNorm vs RMSNorm timing
    uv run python stages/05_block/run.py plot             # figures from the saved JSON

Numbers -> results/05_block/*.json. Figures -> docs/visuals/05_*.png.
Training recipe is Stage 04's exactly, so the 4-layer baseline is the Stage 04
`d64` model rebuilt from this stage's parts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from block import LayerNorm, RMSNorm  # noqa: E402
from model import Config, TinyLM  # noqa: E402

from llmfs.data import MixedLoader, open_bin  # noqa: E402

VISUALS = REPO / "docs" / "visuals"
RESULTS = REPO / "results" / "05_block"
BINS = REPO / "data" / "bins" / "split_8192"

# name -> (Config overrides, peak lr, warmup steps)
DEPTH = {
    "pre_L4":       (dict(n_layer=4), 1e-3, 100),
    "post_L4":      (dict(n_layer=4, prenorm=False), 1e-3, 100),
    "nores_L4":     (dict(n_layer=4, residual=False), 1e-3, 100),
    "pre_L24":      (dict(n_layer=24), 1e-3, 100),
    "post_L24":     (dict(n_layer=24, prenorm=False), 1e-3, 100),
    "nores_L24":    (dict(n_layer=24, residual=False), 1e-3, 100),
    # Added after post_L4 stalled at 6.86: is it the learning rate, or our
    # tiny embedding (rms 0.028) that the 2017 recipe scaled up by √C?
    "post_L4_lr3e-4":   (dict(n_layer=4, prenorm=False), 3e-4, 100),
    "post_L4_lr1e-4":   (dict(n_layer=4, prenorm=False), 1e-4, 100),
    "post_L4_embscale": (dict(n_layer=4, prenorm=False, emb_scale=True), 1e-3, 100),
    "pre_L4_embscale":  (dict(n_layer=4, emb_scale=True), 1e-3, 100),
    # run.py diagnose showed √C embeddings keep post-norm's edits small. So the
    # fair depth test is post-norm WITH the 2017 recipe's √C, at 24 layers,
    # with and without warmup (the condition Xiong et al. analysed).
    "pre_L24_embscale":         (dict(n_layer=24, emb_scale=True), 1e-3, 100),
    "post_L24_embscale":        (dict(n_layer=24, prenorm=False, emb_scale=True), 1e-3, 100),
    "post_L24_embscale_nowarm": (dict(n_layer=24, prenorm=False, emb_scale=True), 1e-3, 0),
}

SWEEP = {
    "baseline":   dict(),                     # pre-norm, LayerNorm, GELU, no dropout
    "rmsnorm":    dict(norm="rms"),
    "relu":       dict(act="relu"),
    "swiglu":     dict(act="swiglu"),
    "no_mlp":     dict(act="none"),
    "dropout0.1": dict(dropout=0.1),
    # Added after no_mlp: is the loss from losing the MLP, or from losing
    # its 2.1M parameters? 12 attention-only layers ≈ the baseline's
    # non-embedding parameter count (3.16M vs 3.16M).
    "no_mlp_L12": dict(act="none", n_layer=12),
}


def write_json(name: str, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    """Stage 04's schedule, with warmup as a knob (0 = start at full lr)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def val_windows(T: int, n: int) -> torch.Tensor:
    data = open_bin(BINS / "val_code.bin")
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    return torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts]))


@torch.no_grad()
def evaluate(model, windows, device, bs=32) -> float:
    model.eval()
    losses = []
    for i in range(0, len(windows), bs):
        w = windows[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(w[:, :-1], w[:, 1:])
        losses.append(loss.item())
    model.train()
    return round(float(np.mean(losses)), 4)


def block_grad_norms(model) -> list[float]:
    """Gradient norm of each block's weights, bottom block first."""
    return [math.sqrt(sum(p.grad.float().pow(2).sum().item()
                          for p in b.parameters() if p.grad is not None))
            for b in model.blocks]


def train_one(name: str, cfg_kw: dict, lr: float, warmup: int, seed: int, args) -> dict:
    torch.manual_seed(seed)
    cfg = Config(max_len=args.T, **cfg_kw)
    device = "cuda"
    model = TinyLM(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                          "en": (BINS / "train_en.bin", 0.15)},
                         B=args.B, T=args.T, device=device, seed=seed)

    curve, layer_grads = [], []
    diverged_at = None
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, lr, warmup)
        x, y = loader.next_batch()                               # (B, T) each
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            diverged_at = step
            print(f"  {name} seed {seed}: loss not finite at step {step}, stopping", flush=True)
            break
        if step % 50 == 0 or step == args.steps - 1:
            # Total grad norm before clipping: how hard the clip is working.
            gn = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in model.parameters()
                                if p.grad is not None)).item()
            curve.append((step, round(loss.item(), 4), float(f"{gn:.4g}")))
            if step % 500 == 0 or step == args.steps - 1:
                layer_grads.append((step, [float(f"{g:.4g}") for g in block_grad_norms(model)]))
                print(f"  {name:16s} seed {seed} step {step:5d} loss {loss.item():.4f} |g| {gn:.3g}",
                      flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    train_s = time.time() - t0
    val = evaluate(model, val_windows(args.T, args.n_eval), device) if diverged_at is None else None
    out = {"variant": name, "cfg": cfg.__dict__, "lr": lr, "warmup": warmup, "seed": seed,
           "steps": args.steps, "B": args.B, "T": args.T, "tokens": args.steps * args.B * args.T,
           "params": n_params(model), "train_seconds": round(train_s, 1),
           "val_code_loss": val, "diverged_at": diverged_at,
           "final_train_loss": curve[-1][1] if curve else None,
           "curve": curve, "layer_grads": layer_grads}
    print(f"  {name} seed {seed}: val {val}  params {n_params(model):,}  {train_s:.0f}s", flush=True)
    return out


def run_set(table: dict, fname: str, names, seeds, args):
    path = RESULTS / fname
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    for seed in seeds:
        for name in names:
            spec = table[name]
            cfg_kw, lr, warmup = spec if isinstance(spec, tuple) else (spec, args.lr, 100)
            r = train_one(name, cfg_kw, lr, warmup, seed, args)
            runs = [x for x in runs if not (x["variant"] == name and x["seed"] == seed)] + [r]
            write_json(fname, {"device": torch.cuda.get_device_name(0), "runs": runs})
    summarize(runs, table)


def summarize(runs, table):
    print(f"\n{'variant':18s} {'seeds':>5s} {'params':>11s} {'val loss':>17s} {'final train':>11s} {'s/run':>6s}")
    for name in table:
        rs = [r for r in runs if r["variant"] == name]
        if not rs:
            continue
        v = np.array([r["val_code_loss"] if r["val_code_loss"] is not None else np.nan for r in rs])
        sd = f"± {np.nanstd(v, ddof=1):.4f}" if len(v) > 1 else ""
        div = [r["diverged_at"] for r in rs if r["diverged_at"] is not None]
        tr = np.mean([r["final_train_loss"] for r in rs if r["final_train_loss"] is not None])
        print(f"{name:18s} {len(rs):5d} {rs[0]['params']:11,d} {np.nanmean(v):8.4f} {sd:8s} "
              f"{tr:11.4f} {np.mean([r['train_seconds'] for r in rs]):6.0f}"
              + (f"  diverged at {div}" if div else ""))


# ---------------------------------------------------------------------------
# 10a. gradients and stream size at initialisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def token_similarity(x: torch.Tensor) -> float:
    """Mean cosine similarity between DIFFERENT positions of the same sequence.
    Near 0: tokens are still told apart. Near 1: every position carries the
    same vector, and the head cannot tell `for` from `return` (rank collapse)."""
    u = torch.nn.functional.normalize(x.float(), dim=-1)        # (B, T, C), unit rows
    sim = u @ u.transpose(1, 2)                                  # (B, T, T)
    T = sim.shape[-1]
    off = sim.sum((1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(-1)  # drop self-similarity
    return (off / (T * (T - 1))).mean().item()


def grads(args):
    """24-layer models, untrained. One real batch through each; record the
    gradient reaching every block and the size of the residual stream after
    every block. Averaged over 5 seeds (fresh init + fresh batch each)."""
    device = "cuda"
    variants = {"pre-norm": dict(), "post-norm": dict(prenorm=False),
                "pre-norm, no residual": dict(residual=False)}
    out = {}
    for label, kw in variants.items():
        g_all, s_all, c_all = [], [], []
        for seed in range(5):
            torch.manual_seed(seed)
            model = TinyLM(Config(n_layer=24, max_len=args.T, **kw)).to(device)
            loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                                  "en": (BINS / "train_en.bin", 0.15)},
                                 B=args.B, T=args.T, device=device, seed=seed)
            x, y = loader.next_batch()
            _, loss, stream = model(x, y, keep_stream=True)      # fp32: no autocast, exact sizes
            loss.backward()
            g_all.append(block_grad_norms(model))
            s_all.append([s.pow(2).mean().sqrt().item() for s in stream])   # rms of (B, T, C)
            c_all.append([token_similarity(s) for s in stream])
            del model
        g = np.array(g_all).mean(0)
        s = np.array(s_all).mean(0)
        c = np.array(c_all).mean(0)
        out[label] = {"block_grad_norm": [float(f"{v:.4g}") for v in g],
                      "stream_rms": [float(f"{v:.4g}") for v in s],
                      "token_cosine": [float(f"{v:.4g}") for v in c]}
        print(f"{label:22s} grad block 1 {g[0]:.3g}  block 24 {g[-1]:.3g}  ratio {g[0] / g[-1]:.3g}   "
              f"stream rms in {s[0]:.3g} out {s[-1]:.3g}   token cosine in {c[0]:.3f} out {c[-1]:.3f}")
    write_json("grads.json", {"n_layer": 24, "seeds": 5, "B": args.B, "T": args.T, "variants": out})
    plot_grads()


# ---------------------------------------------------------------------------
# 10b diagnosis: how big is each sublayer's edit next to the highway?
# ---------------------------------------------------------------------------

def diagnose(args):
    """Train 4-layer pre- and post-norm models for a few hundred steps (same
    recipe) and log, every 25 steps, rms(f(x)) / rms(x) for every sublayer and
    the gradient reaching every block. If post-norm fails because the edits
    outgrow the highway, the ratio climbs and the low-block gradients fall."""
    device = "cuda"
    out = {}
    for name, kw in (("pre_L4", dict()), ("post_L4", dict(prenorm=False)),
                     ("post_L4_embscale", dict(prenorm=False, emb_scale=True))):
        torch.manual_seed(0)
        model = TinyLM(Config(n_layer=4, max_len=args.T, **kw)).to(device)
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                 {"params": no_decay, "weight_decay": 0.0}],
                                lr=args.lr, betas=(0.9, 0.95), fused=True)
        loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                              "en": (BINS / "train_en.bin", 0.15)},
                             B=args.B, T=args.T, device=device, seed=0)
        log = []
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, 4000, args.lr, 100)        # the 4000-step schedule's start
            probe = step % 25 == 0
            for b in model.blocks:
                b.probe, b.ratios = probe, []
            x, y = loader.next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if probe:
                ratios = [r for b in model.blocks for r in b.ratios]   # attn₁, mlp₁, attn₂, ...
                log.append({"step": step, "loss": round(loss.item(), 4),
                            "edit_ratio": [float(f"{r:.4g}") for r in ratios],
                            "block_grad": [float(f"{g:.4g}") for g in block_grad_norms(model)]})
                if step % 100 == 0:
                    print(f"  {name:16s} step {step:4d} loss {loss.item():.3f} "
                          f"|f|/|x| {[round(r, 2) for r in ratios]} "
                          f"grad {[float(f'{g:.2g}') for g in block_grad_norms(model)]}", flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        out[name] = log
    write_json("diagnose.json", {"steps": args.steps, "log": out})
    plot_diagnose()


# ---------------------------------------------------------------------------
# 10e. LayerNorm vs RMSNorm speed
# ---------------------------------------------------------------------------

def _time(fn, x, iters=200) -> float:
    for _ in range(20):
        fn(x).sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x).sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3              # ms per fwd+bwd


def speed(args):
    """Forward + backward of one norm on (B·T, C) bf16 activations.
    Four implementations each: ours in eager PyTorch, ours under
    torch.compile, and PyTorch's built-in module."""
    device = "cuda"
    rows = []
    for C in (256, 768, 2048, 4096):
        x = torch.randn(8 * 1024, C, device=device, dtype=torch.bfloat16, requires_grad=True)  # 8 × T=1024
        impls = {
            "LayerNorm (ours)": LayerNorm(C).to(device, torch.bfloat16),
            "RMSNorm (ours)": RMSNorm(C).to(device, torch.bfloat16),
            "nn.LayerNorm": torch.nn.LayerNorm(C).to(device, torch.bfloat16),
            "nn.RMSNorm": torch.nn.RMSNorm(C, eps=1e-6).to(device, torch.bfloat16),
        }
        impls["LayerNorm (ours, compiled)"] = torch.compile(LayerNorm(C).to(device, torch.bfloat16))
        impls["RMSNorm (ours, compiled)"] = torch.compile(RMSNorm(C).to(device, torch.bfloat16))
        row = {"C": C, "rows": x.shape[0]}
        for k, m in impls.items():
            row[k] = round(_time(m, x), 4)
        # A matmul of the same width for scale: one (C, C) linear layer, fwd+bwd.
        lin = torch.nn.Linear(C, C, bias=False).to(device, torch.bfloat16)
        row["Linear C×C (for scale)"] = round(_time(lin, x), 4)
        rows.append(row)
        print(row, flush=True)
    write_json("speed.json", {"device": torch.cuda.get_device_name(0), "unit": "ms fwd+bwd", "rows": rows})


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_grads():
    d = json.loads((RESULTS / "grads.json").read_text())["variants"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for label, v in d.items():
        ax[0].plot(range(1, 25), v["block_grad_norm"], marker="o", ms=3, label=label)
        ax[1].plot(range(0, 25), v["stream_rms"], marker="o", ms=3, label=label)
        ax[2].plot(range(0, 25), v["token_cosine"], marker="o", ms=3, label=label)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("block (1 = nearest the input)")
    ax[0].set_ylabel("gradient norm of the block's weights")
    ax[0].set_title("Gradient reaching each block, untrained 24-layer model")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("after block (0 = embedding)")
    ax[1].set_ylabel("rms of the residual stream")
    ax[1].set_title("Size of the stream going up the stack")
    ax[2].set_xlabel("after block (0 = embedding)")
    ax[2].set_ylabel("mean cosine between different positions")
    ax[2].set_title("Can the tokens still be told apart?")
    ax[2].set_ylim(-0.05, 1.05)
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "05_init_grads.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '05_init_grads.png'}")


def plot_depth():
    runs = json.loads((RESULTS / "depth.json").read_text())["runs"]
    by = {r["variant"]: r for r in runs if r["seed"] == 0}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    groups = [("4 layers", ["pre_L4", "post_L4", "nores_L4", "post_L4_lr3e-4", "post_L4_lr1e-4",
                            "post_L4_embscale", "pre_L4_embscale"]),
              ("24 layers", ["pre_L24", "post_L24", "nores_L24", "pre_L24_embscale",
                             "post_L24_embscale", "post_L24_embscale_nowarm"])]
    for a, (title, names) in zip(ax, groups):
        for n in names:
            if n not in by:
                continue
            c = np.array([(s, l) for s, l, _ in by[n]["curve"]])
            # One colour per idea, same in both panels. post and nores land on the
            # same curve, so nores is dashed to keep both visible.
            col = {"pre": "C0", "post": "C1", "nores": "C2"}[n.split("_")[0]]
            if "embscale" in n:
                col = "C9" if n.startswith("pre") else "C3"
            if "lr" in n:
                col = "C4" if "3e-4" in n else "C5"
            ls = "--" if n.startswith("nores") or "nowarm" in n else "-"
            a.plot(c[:, 0], c[:, 1], label=n, lw=1.3, ls=ls, color=col)
        a.set_title(title)
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    ax[0].set_ylabel("training loss")
    ax[0].set_ylim(2.2, 8)
    fig.tight_layout()
    fig.savefig(VISUALS / "05_depth.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '05_depth.png'}")


def plot_sweep():
    runs = json.loads((RESULTS / "sweep.json").read_text())["runs"]
    names = [n for n in SWEEP if any(r["variant"] == n for r in runs)]
    mean = [np.mean([r["val_code_loss"] for r in runs if r["variant"] == n]) for n in names]
    sd = [np.std([r["val_code_loss"] for r in runs if r["variant"] == n], ddof=1) for n in names]
    base = mean[names.index("baseline")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#888" if n == "baseline" else ("#c44" if m > base else "#4a4") for n, m in zip(names, mean)]
    ax.bar(names, mean, yerr=sd, color=colors, capsize=4)
    ax.axhline(base, color="#888", ls="--", lw=1)
    for i, (m, e) in enumerate(zip(mean, sd)):
        ax.text(i, m + e + 0.008, f"{m:.3f}\n({m - base:+.3f})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(mean) - 0.1, max(m + e for m, e in zip(mean, sd)) + 0.08)
    ax.set_ylabel("held-out code loss (3 seeds)")
    ax.set_title("One change at a time, 4-layer model (grey = baseline: pre-norm, LayerNorm, GELU)")
    fig.tight_layout()
    fig.savefig(VISUALS / "05_sweep.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '05_sweep.png'}")


def plot_diagnose():
    log = json.loads((RESULTS / "diagnose.json").read_text())["log"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    colors = {"pre_L4": "C0", "post_L4": "C1", "post_L4_embscale": "C2"}
    for name, rows in log.items():
        st = [r["step"] for r in rows]
        ratio = np.array([r["edit_ratio"] for r in rows])          # (n_log, 8 sublayers)
        grad = np.array([r["block_grad"] for r in rows])           # (n_log, 4 blocks)
        ax[0].plot(st, ratio.max(1), color=colors[name], label=name)
        ax[1].plot(st, grad[:, 0], color=colors[name], label=name)
        ax[2].plot(st, [r["loss"] for r in rows], color=colors[name], label=name)
    ax[0].set_yscale("log")
    ax[0].axhline(1, color="#888", ls=":", lw=1)
    ax[0].set_title("Largest edit vs highway, rms(f(x)) / rms(x)")
    ax[1].set_yscale("log")
    ax[1].set_title("Gradient reaching block 1")
    ax[2].set_title("Training loss")
    for a in ax:
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "05_postnorm_diagnosis.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '05_postnorm_diagnosis.png'}")


def plot():
    for f, fn in (("diagnose.json", plot_diagnose), ("grads.json", plot_grads), ("depth.json", plot_depth), ("sweep.json", plot_sweep)):
        if (RESULTS / f).exists():
            fn()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("grads", "depth", "sweep", "speed", "plot", "summary", "diagnose"):
        a = sub.add_parser(name)
        a.add_argument("--steps", type=int, default=4000)
        a.add_argument("--B", type=int, default=32)
        a.add_argument("--T", type=int, default=256)
        a.add_argument("--lr", type=float, default=1e-3)
        a.add_argument("--n-eval", type=int, default=512)
        if name == "depth":
            a.add_argument("--variants", nargs="+", default=list(DEPTH), choices=list(DEPTH))
            a.add_argument("--seeds", nargs="+", type=int, default=[0])
        if name == "sweep":
            a.add_argument("--variants", nargs="+", default=list(SWEEP), choices=list(SWEEP))
            a.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()
    if args.cmd == "depth":
        run_set(DEPTH, "depth.json", args.variants, args.seeds, args)
        plot_depth()
    elif args.cmd == "sweep":
        run_set(SWEEP, "sweep.json", args.variants, args.seeds, args)
        plot_sweep()
    elif args.cmd == "summary":
        for f, t in (("depth.json", DEPTH), ("sweep.json", SWEEP)):
            if (RESULTS / f).exists():
                print(f"\n== {f}")
                summarize(json.loads((RESULTS / f).read_text())["runs"], t)
    else:
        {"grads": grads, "speed": speed, "plot": plot, "diagnose": diagnose}[args.cmd](args) if args.cmd != "plot" else plot()


if __name__ == "__main__":
    main()
