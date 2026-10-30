"""Stage 04 experiments. docs/04_attention.md §9.

    uv run python stages/04_attention/run.py variance      # 9a, seconds
    uv run python stages/04_attention/run.py cost          # 9b, ~1 min
    uv run python stages/04_attention/run.py ablation      # 9c, ~20 min
    uv run python stages/04_attention/run.py maps          # 9d, needs the d64 seed-0 checkpoint

Numbers -> results/04_attention/*.json. Figures -> docs/visuals/04_*.png.
The d64 seed-0 model is saved to checkpoints/04_attention/ (gitignored) for maps.
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attention import attention
from tiny_lm import Config, TinyLM

from llmfs.data import MixedLoader, open_bin
from llmfs.tokenizer import Tokenizer

VISUALS = REPO / "docs" / "visuals"
RESULTS = REPO / "results" / "04_attention"
CKPT = REPO / "checkpoints" / "04_attention"
BINS = REPO / "data" / "bins" / "split_8192"
TOKENIZER = REPO / "data" / "tokenizers" / "split_8192.json"

# name -> (n_head, scale). C = 256, so d = 256 / n_head.
VARIANTS = {
    "d64":          (4, True),
    "d64_noscale":  (4, False),
    "d256":         (1, True),
    "d256_noscale": (1, False),
}


def write_json(name: str, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def entropy(w: torch.Tensor) -> torch.Tensor:
    """Entropy of each attention row, in nats. 0 = all weight on one token."""
    return -(w * w.clamp_min(1e-30).log()).sum(-1)


# ---------------------------------------------------------------------------
# 9a. saturation with random vectors
# ---------------------------------------------------------------------------

def variance():
    """Unit-variance q, k, v; 256 keys; no mask. How big are the scores, how
    peaked is the softmax, and how much gradient reaches q?"""
    torch.manual_seed(0)
    n_keys, n_q = 256, 512
    rows = []
    for d in (16, 32, 64, 128, 256, 512, 1024):
        q = torch.randn(n_q, 1, d, dtype=torch.float64)          # (n_q, 1, d): one query per batch row
        k = torch.randn(n_q, n_keys, d, dtype=torch.float64)     # (n_q, n_keys, d)
        v = torch.randn(n_q, n_keys, d, dtype=torch.float64)
        g = torch.randn(n_q, 1, d, dtype=torch.float64)          # fixed upstream gradient
        row = {"d": d}
        for tag, scale in (("scaled", True), ("unscaled", False)):
            qq = q.clone().requires_grad_()
            # Written out instead of calling attention(), so the gradient at
            # the scores can be read. That is where saturation bites: softmax's
            # slope goes to 0. The gradient at q is that times k, and k's size
            # grows with √d, which partly hides the effect (docs §9a).
            s = qq @ k.transpose(-2, -1) / (math.sqrt(d) if scale else 1)   # (n_q, 1, n_keys)
            s.retain_grad()
            w = s.softmax(-1)
            out = w @ v                                          # (n_q, 1, d)
            (out * g).sum().backward()
            gs = s.grad.norm(dim=-1).flatten()                   # (n_q,) per-query score gradient
            row[tag] = {
                "score_std": round(s.std().item(), 3),
                "max_weight": round(w.max(-1).values.mean().item(), 4),
                "entropy_nats": round(entropy(w).mean().item(), 3),
                "grad_scores_median": float(f"{gs.median().item():.4g}"),
                "grad_scores_dead_frac": round((gs < 1e-3).double().mean().item(), 4),
                "grad_q_norm": float(f"{qq.grad.norm(dim=-1).mean().item():.4g}"),
            }
        rows.append(row)
        print(f"d {d:5d}  " + "   ".join(
            f"{tag}: std {r['score_std']:6.2f} max {r['max_weight']:.3f} H {r['entropy_nats']:.2f} "
            f"g_s {r['grad_scores_median']:.3g} dead {r['grad_scores_dead_frac']:.2f} g_q {r['grad_q_norm']:.3g}"
            for tag, r in ((t, row[t]) for t in ("scaled", "unscaled"))))
    write_json("variance.json", {"n_keys": n_keys, "n_queries": n_q,
                                 "max_entropy_nats": round(math.log(n_keys), 3), "rows": rows})

    ds = [r["d"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for tag, style in (("scaled", "o-"), ("unscaled", "s--")):
        ax[0].plot(ds, [r[tag]["score_std"] for r in rows], style, label=tag)
        ax[1].plot(ds, [r[tag]["max_weight"] for r in rows], style, label=tag)
        ax[2].plot(ds, [r[tag]["grad_scores_median"] for r in rows], style, label=tag)
    ax[0].set(title="Size of the scores (std)", xlabel="head size d", xscale="log", yscale="log")
    ax[1].set(title=f"Largest softmax weight ({n_keys} keys)\n1.0 = a hard lookup",
              xlabel="head size d", xscale="log", ylim=(0, 1.05))
    ax[2].set(title="Gradient at the scores (median per query)\nwhat saturation kills", xlabel="head size d",
              xscale="log", yscale="log")
    for a in ax:
        a.set_xticks(ds, [str(x) for x in ds])
        a.legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "04_sqrt_d.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '04_sqrt_d.png'}")


# ---------------------------------------------------------------------------
# 9b. the cost of T × T
# ---------------------------------------------------------------------------

def cost():
    dev = "cuda"
    B, H, d = 1, 8, 64
    rows = []
    # One throwaway call so the first measured T is not charged for cuBLAS and
    # kernel start-up (the first run charged T=256 more time than T=512).
    w = torch.randn(B, H, 256, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    attention(w, w, w).sum().backward()
    F.scaled_dot_product_attention(w, w, w, is_causal=True).sum().backward()
    for T in (256, 512, 1024, 2048, 4096, 8192, 16384):
        row = {"T": T}
        for tag, fn in (("ours", lambda q, k, v: attention(q, k, v, causal=True)),
                        ("sdpa", lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True))):
            torch.cuda.empty_cache()
            try:
                q, k, v = (torch.randn(B, H, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
                           for _ in range(3))
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                fn(q, k, v).sum().backward()                     # warm-up, and the memory measurement
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                n = 10 if T <= 4096 else 3
                t0 = time.perf_counter()
                for _ in range(n):
                    fn(q, k, v).sum().backward()
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / n * 1000
                row[tag] = {"peak_mb": round(peak / 2**20, 1), "ms": round(ms, 2)}
            except torch.OutOfMemoryError:
                row[tag] = "OOM"
            finally:
                q = k = v = None
        rows.append(row)
        print(T, row, flush=True)
    write_json("cost.json", {"device": torch.cuda.get_device_name(0), "B": B, "H": H, "d": d,
                             "dtype": "bf16", "measures": "forward + backward, causal",
                             "rows": rows})
    plot_cost()


def plot_cost():
    """Redraws from cost.json, so the figure can be remade without the GPU."""
    rows = json.loads((RESULTS / "cost.json").read_text())["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for tag, label, style in (("ours", "ours: stores the T × T matrix", "o-"),
                              ("sdpa", "PyTorch fused (FlashAttention-style)", "s-")):
        ok = [r for r in rows if isinstance(r[tag], dict)]
        Ts = [r["T"] for r in ok]
        ax[0].plot(Ts, [r[tag]["peak_mb"] for r in ok], style, label=label)
        ax[1].plot(Ts, [r[tag]["ms"] for r in ok], style, label=label)
        oom = [r["T"] for r in rows if r[tag] == "OOM"]
        for T in oom:
            ax[0].plot([T], [7797], "x", color="C0", ms=12, mew=3)
            ax[0].annotate("ours: out of memory", (T, 7797), xytext=(T, 2500), color="C0",
                           ha="right", arrowprops={"arrowstyle": "->", "color": "C0"})
    ax[0].axhline(7797, color="gray", ls="--", lw=1)
    ax[0].text(256, 7797 * 0.62, "whole GPU: 7,797 MB", color="gray")
    ax[0].set(title="Memory for one attention call (fwd + bwd)\nB=1, 8 heads, d=64, bf16",
              xlabel="sequence length T", ylabel="peak MB", xscale="log", yscale="log")
    ax[1].set(title="Time per call (fwd + bwd)", xlabel="sequence length T", ylabel="ms",
              xscale="log", yscale="log")
    for a in ax:
        a.set_xticks([r["T"] for r in rows], [str(r["T"]) for r in rows], rotation=45)
        a.legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "04_cost.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '04_cost.png'}")


# ---------------------------------------------------------------------------
# 9c. training without √d
# ---------------------------------------------------------------------------

def lr_at(step: int, steps: int, peak: float, warmup: int = 100) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def val_windows(T: int, n: int) -> torch.Tensor:
    data = open_bin(BINS / "val_code.bin")
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    return torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts]))


@torch.no_grad()
def evaluate(model, windows, device, bs=32) -> dict:
    model.eval()
    losses, ents, maxw = [], [], []
    model.keep_attention(True)
    for i in range(0, len(windows), bs):
        w = windows[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(w[:, :-1], w[:, 1:])
        losses.append(loss.item())
        if i < 4 * bs:                                           # attention stats on 128 windows
            A = torch.stack([a.float() for a in model.attention_weights()])  # (L, b, H, T, T)
            # Skip the first 32 rows: row t has only t+1 choices, so early rows
            # have low entropy for reasons that have nothing to do with scaling.
            A = A[..., 32:, :]
            ents.append(entropy(A).mean(dim=(1, 3)).cpu())       # (L, H)
            maxw.append(A.max(-1).values.mean(dim=(1, 3)).cpu())
    model.keep_attention(False)
    model.train()
    ent = torch.stack(ents).mean(0)                              # (L, H)
    return {"val_code_loss": round(float(np.mean(losses)), 4),
            "attn_entropy_by_layer_head": [[round(x, 3) for x in r] for r in ent.tolist()],
            "attn_entropy_mean": round(ent.mean().item(), 3),
            "attn_max_weight_mean": round(torch.stack(maxw).mean().item(), 4)}


def train_one(name: str, seed: int, args) -> dict:
    n_head, scale = VARIANTS[name]
    torch.manual_seed(seed)
    cfg = Config(n_head=n_head, scale=scale, max_len=args.T)
    device = "cuda"
    model = TinyLM(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                          "en": (BINS / "train_en.bin", 0.15)},
                         B=args.B, T=args.T, device=device, seed=seed)
    qkv_params = [b.attn.qkv.weight for b in model.blocks]

    curve = []
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        x, y = loader.next_batch()                               # (B, T) each
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % 50 == 0 or step == args.steps - 1:
            # Gradient reaching W_q/W_k/W_v, measured before clipping:
            # the "gradients vanish" claim, checked in a real model.
            gq = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in qkv_params)).item()
            curve.append((step, round(loss.item(), 4), float(f"{gq:.4g}")))
            print(f"  {name:13s} seed {seed} step {step:5d} loss {loss.item():.4f} |g_qkv| {gq:.3g}",
                  flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    train_s = time.time() - t0

    ev = evaluate(model, val_windows(args.T, args.n_eval), device)
    if name == "d64" and seed == 0:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, CKPT / "d64_seed0.pt")
    out = {"variant": name, "n_head": n_head, "head_dim": 256 // n_head, "scale": scale,
           "seed": seed, "steps": args.steps, "B": args.B, "T": args.T, "lr": args.lr,
           "tokens": args.steps * args.B * args.T, "train_seconds": round(train_s, 1),
           **ev, "curve": curve}
    print(f"  {name} seed {seed}: val {ev['val_code_loss']:.4f}  entropy {ev['attn_entropy_mean']:.3f}  "
          f"max w {ev['attn_max_weight_mean']:.3f}  {train_s:.0f}s", flush=True)
    return out


def ablation(args):
    path = RESULTS / "ablation.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    for seed in args.seeds:
        for name in args.variants:
            r = train_one(name, seed, args)
            runs = [x for x in runs if not (x["variant"] == name and x["seed"] == seed)] + [r]
            write_json("ablation.json", {"device": torch.cuda.get_device_name(0), "runs": runs})
    summarize(runs)


def summarize(runs):
    print(f"\n{'variant':14s} {'seeds':>5s} {'val loss':>17s} {'entropy':>8s} {'max w':>7s}")
    for name in VARIANTS:
        rs = [r for r in runs if r["variant"] == name]
        if not rs:
            continue
        v = np.array([r["val_code_loss"] for r in rs])
        sd = f"± {v.std(ddof=1):.4f}" if len(v) > 1 else ""
        print(f"{name:14s} {len(rs):5d} {v.mean():8.4f} {sd:8s} "
              f"{np.mean([r['attn_entropy_mean'] for r in rs]):8.3f} "
              f"{np.mean([r['attn_max_weight_mean'] for r in rs]):7.3f}")


# ---------------------------------------------------------------------------
# 9d. attention maps
# ---------------------------------------------------------------------------

SNIPPET = '''def total_price(items, tax):
    total = 0
    for item in items:
        total += item.price * item.qty
    return total * (1 + tax)
'''


@torch.no_grad()
def maps():
    device = "cuda"
    ck = torch.load(CKPT / "d64_seed0.pt", map_location=device)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    model.keep_attention(True)
    L, H = model.cfg.n_layer, model.cfg.n_head

    # Head statistics over held-out code: where does each head put its weight?
    W = val_windows(model.cfg.max_len, 64).to(device)
    model(W[:, :-1])
    stats = []
    for li, A in enumerate(model.attention_weights()):          # (b, H, T, T)
        A = A.float()[..., 32:, :]                               # rows 32.. (enough choices)
        T = A.shape[-1]
        rows = torch.arange(32, T, device=device)
        for h in range(H):
            a = A[:, h]                                          # (b, T-32, T)
            stats.append({
                "layer": li, "head": h,
                "on_previous_token": round(a[:, torch.arange(len(rows)), rows - 1].mean().item(), 3),
                "on_self": round(a[:, torch.arange(len(rows)), rows].mean().item(), 3),
                "on_first_token": round(a[..., 0].mean().item(), 3),
                "entropy_nats": round(entropy(a).mean().item(), 3),
            })
    for s in stats:
        print(s)

    # The snippet: one small map per head, then the two most distinct heads, labelled.
    tok = Tokenizer.load(str(TOKENIZER))
    ids = tok.encode_ordinary(SNIPPET)
    labels = [tok.decode([i]).replace("\n", "↵").replace(" ", "·") for i in ids]
    model(torch.tensor([ids], device=device))
    A = [a[0].float().cpu() for a in model.attention_weights()]  # L × (H, T, T)

    fig, ax = plt.subplots(L, H, figsize=(3 * H, 3 * L))
    for li in range(L):
        for h in range(H):
            s = next(x for x in stats if x["layer"] == li and x["head"] == h)
            ax[li, h].imshow(A[li][h], cmap="magma", vmin=0, vmax=1)
            ax[li, h].set_title(f"L{li} H{h}  prev {s['on_previous_token']:.2f}  "
                                f"first {s['on_first_token']:.2f}", fontsize=8)
            ax[li, h].set_xticks([])
            ax[li, h].set_yticks([])
    fig.suptitle("Every head on the same Python snippet (row = token looking, column = token looked at)")
    fig.tight_layout()
    fig.savefig(VISUALS / "04_attention_maps.png", dpi=110)
    plt.close(fig)

    prev = max(stats, key=lambda s: s["on_previous_token"])
    # The second panel, for contrast: among heads that are not sinks, the one
    # with the least weight on the previous token and itself. On our model this
    # picks a diffuse head (layer 0) that spreads weight thinly over everything.
    other = min((s for s in stats if s["on_first_token"] < 0.3),
                key=lambda s: s["on_previous_token"] + s["on_self"])
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a, s in zip(ax, (prev, other)):
        a.imshow(A[s["layer"]][s["head"]], cmap="magma", vmin=0, vmax=1)
        a.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        a.set_yticks(range(len(labels)), labels, fontsize=6)
        a.set_title(f"layer {s['layer']} head {s['head']}: {s['on_previous_token']:.0%} on previous "
                    f"token, {s['on_self']:.0%} on self, {s['on_first_token']:.0%} on first")
    fig.tight_layout()
    fig.savefig(VISUALS / "04_attention_heads.png", dpi=130)
    plt.close(fig)
    write_json("heads.json", {"snippet": SNIPPET, "n_tokens": len(ids), "heads": stats,
                              "shown": [prev, other]})
    print(f"wrote {VISUALS / '04_attention_maps.png'}\nwrote {VISUALS / '04_attention_heads.png'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("variance")
    sub.add_parser("cost")
    sub.add_parser("plot-cost")
    a = sub.add_parser("ablation")
    a.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    a.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    a.add_argument("--steps", type=int, default=4000)
    a.add_argument("--B", type=int, default=32)
    a.add_argument("--T", type=int, default=256)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--n-eval", type=int, default=512)
    sub.add_parser("maps")
    sub.add_parser("summary")
    args = p.parse_args()
    if args.cmd == "summary":
        summarize(json.loads((RESULTS / "ablation.json").read_text())["runs"])
    else:
        {"variance": variance, "cost": cost, "plot-cost": plot_cost, "maps": maps}.get(args.cmd, lambda: ablation(args))()


if __name__ == "__main__":
    main()
