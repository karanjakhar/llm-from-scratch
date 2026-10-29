"""Stage 03 experiments. docs/03_embeddings_position.md §7 (plots) and §8 (ablation).

    uv run python stages/03_embeddings/run.py plots
    uv run python stages/03_embeddings/run.py ablation                  # all variants, seed 0
    uv run python stages/03_embeddings/run.py ablation --variants rope none --seeds 1 2

Figures go to docs/visuals/03_*.png (they are book figures). Numbers go to
results/03_embeddings/ablation.json; each finished run is merged into that file
immediately, so an interrupted sweep keeps what it finished.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from embeddings import rope_angles, rope_interleaved, sinusoidal_table
from tiny_lm import Config, TinyLM

from llmfs.data import MixedLoader, open_bin

VISUALS = REPO / "docs" / "visuals"
RESULTS = REPO / "results" / "03_embeddings"
BINS = REPO / "data" / "bins" / "split_8192"

# Variant name -> (position scheme, tied head, token-embedding scale)
VARIANTS = {
    "none":              ("none", True, 1.0),
    "learned":           ("learned", True, 1.0),
    "sinusoidal":        ("sinusoidal", True, 1.0),
    # Added after the first sweep: the 2017 recipe scales token vectors by
    # sqrt(C) so the sinusoids do not drown them. docs §8.
    "sinusoidal_scaled": ("sinusoidal", True, 16.0),
    "rope":              ("rope", True, 1.0),
    "rope_untied":       ("rope", False, 1.0),
}


# ---------------------------------------------------------------------------
# plots (docs §7)
# ---------------------------------------------------------------------------

def plots():
    VISUALS.mkdir(parents=True, exist_ok=True)

    # Figure 1: the sinusoidal table and its similarity matrix.
    T, C = 128, 64
    pe = sinusoidal_table(T, C)                                  # (T, C)
    sim = pe @ pe.T                                              # (T, T)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im = ax[0].imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set(title="Sinusoidal positions: one row per position",
              xlabel="dimension (left = fast clocks, right = slow)", ylabel="position")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    im = ax[1].imshow(sim, cmap="viridis")
    ax[1].set(title="PE[p] · PE[q]: constant along each diagonal\n(depends only on p - q)",
              xlabel="position q", ylabel="position p")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(VISUALS / "03_sinusoidal.png", dpi=130)
    plt.close(fig)

    # Figure 2: RoPE scores depend only on offset.
    torch.manual_seed(0)
    T, d = 128, 64
    q, k = torch.randn(d).double(), torch.randn(d).double()
    ang = rope_angles(T, d, dtype=torch.float64)
    qs = rope_interleaved(q.expand(T, d), ang)                   # (T, d): q placed at each m
    ks = rope_interleaved(k.expand(T, d), ang)                   # (T, d): k placed at each n
    scores = (qs @ ks.T).numpy()                                 # (T, T): scores[m, n]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im = ax[0].imshow(scores, cmap="RdBu_r")
    ax[0].set(title="RoPE score for the same q, k at every (m, n)",
              xlabel="key position n", ylabel="query position m")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    offsets = np.arange(0, 60)
    for start, style in ((0, "-"), (30, "--"), (60, ":")):
        ax[1].plot(offsets, [scores[start + o, start] for o in offsets], style, lw=2,
                   label=f"key at n = {start}")
    ax[1].set(title="Score vs offset m - n: three starting points,\none curve",
              xlabel="offset m - n (tokens)", ylabel="q · k after rotation")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "03_rope_offset.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '03_sinusoidal.png'}\nwrote {VISUALS / '03_rope_offset.png'}")


# ---------------------------------------------------------------------------
# ablation (docs §8)
# ---------------------------------------------------------------------------

def eval_windows(T: int, n: int) -> torch.Tensor:
    """n fixed, evenly spaced windows of T+1 tokens from held-out code.
    Same windows for every variant, so differences are the model's alone."""
    data = open_bin(BINS / "val_code.bin")
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    return torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts]))


@torch.no_grad()
def per_position_loss(model, windows: torch.Tensor, device, bs: int = 32) -> np.ndarray:
    model.eval()
    total = torch.zeros(windows.shape[1] - 1, dtype=torch.float64)
    for i in range(0, len(windows), bs):
        w = windows[i:i + bs].to(device)                          # (b, T+1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(w[:, :-1], w[:, 1:], reduction="none")    # (b, T)
        total += loss.double().sum(0).cpu()
    model.train()
    return (total / len(windows)).numpy()                         # (T,)


def lr_at(step: int, steps: int, peak: float, warmup: int = 100) -> float:
    # Linear warmup, then cosine down to 10% of peak. Stage 08 explains why.
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train_one(name: str, seed: int, args) -> dict:
    pos, tie, scale = VARIANTS[name]
    torch.manual_seed(seed)
    cfg = Config(pos=pos, tie=tie, max_len=args.T, emb_scale=scale)
    device = "cuda"
    model = TinyLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    # The data order is seeded separately from the model so that every variant
    # with the same seed sees exactly the same batches.
    loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                          "en": (BINS / "train_en.bin", 0.15)},
                         B=args.B, T=args.T, device=device, seed=seed)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            curve.append((step, round(loss.item(), 4)))
            print(f"  {name:12s} seed {seed} step {step:5d} loss {loss.item():.4f}", flush=True)
    torch.cuda.synchronize()
    train_s = time.time() - t0

    pp = per_position_loss(model, eval_windows(args.T, args.n_eval), device)
    out = {"variant": name, "pos": pos, "tied": tie, "emb_scale": scale, "seed": seed,
           "params": n_params,
           "steps": args.steps, "B": args.B, "T": args.T, "lr": args.lr,
           "tokens": args.steps * args.B * args.T, "train_seconds": round(train_s, 1),
           "val_code_loss": round(float(pp.mean()), 4),
           "val_loss_by_position": [round(float(v), 4) for v in pp],
           "curve": curve}

    # Past the edge: windows twice the training length. Learned positions
    # cannot even run (docs §4), which is itself the result.
    try:
        pp2 = per_position_loss(model, eval_windows(2 * args.T, args.n_eval // 2), device)
        out["long_eval"] = {"T": 2 * args.T,
                            "loss_first_half": round(float(pp2[:args.T].mean()), 4),
                            "loss_second_half": round(float(pp2[args.T:].mean()), 4),
                            "loss_by_position": [round(float(v), 4) for v in pp2]}
    except ValueError as e:
        out["long_eval"] = {"T": 2 * args.T, "error": str(e)}
    print(f"  {name} seed {seed}: val {out['val_code_loss']:.4f}  "
          f"long {out['long_eval'].get('loss_second_half', 'n/a')}  {train_s:.0f}s", flush=True)
    return out


def ablation(args):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "ablation.json"
    runs = json.loads(path.read_text())["runs"] if path.exists() else []
    for seed in args.seeds:
        for name in args.variants:
            r = train_one(name, seed, args)
            runs = [x for x in runs if not (x["variant"] == name and x["seed"] == seed)] + [r]
            path.write_text(json.dumps({"device": torch.cuda.get_device_name(0),
                                        "runs": runs}, indent=1))
    summarize(runs)


def summarize(runs: list[dict]):
    by = {}
    for r in runs:
        by.setdefault(r["variant"], []).append(r)
    print(f"\n{'variant':12s} {'params':>9s} {'seeds':>5s} {'val loss':>16s} {'pos 256-511':>12s}")
    for name in VARIANTS:
        if name not in by:
            continue
        rs = by[name]
        v = np.array([r["val_code_loss"] for r in rs])
        sd = f"± {v.std(ddof=1):.4f}" if len(v) > 1 else ""
        longs = [r["long_eval"].get("loss_second_half") for r in rs]
        lg = f"{np.mean(longs):.3f}" if None not in longs else "cannot run"
        print(f"{name:12s} {rs[0]['params']:9,d} {len(rs):5d} {v.mean():8.4f} {sd:7s} {lg:>12s}")
    plot_ablation(by)


def plot_ablation(by: dict):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # One fixed colour per variant, so both panels read the same.
    colors = dict(zip(VARIANTS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for name in VARIANTS:
        if name not in by:
            continue
        rs = by[name]
        r = next(x for x in rs if x["seed"] == min(y["seed"] for y in rs))
        steps, losses = zip(*r["curve"])
        ax[0].plot(steps, losses, label=name, lw=1.5, color=colors[name])
        les = [x["long_eval"] for x in rs if "loss_by_position" in x["long_eval"]]
        if les:
            pp = np.mean([le["loss_by_position"] for le in les], axis=0)   # mean over seeds
            k = 16                                               # smooth: mean over 16-token buckets
            xs = np.arange(len(pp) // k) * k + k / 2
            ax[1].plot(xs, pp[: len(xs) * k].reshape(-1, k).mean(1), label=name, lw=1.5,
                       color=colors[name])
    ax[0].set(title="Training loss (seed 0, identical batches)", xlabel="step", ylabel="loss",
              ylim=(None, 6))
    ax[0].legend()
    T = next(iter(by.values()))[0]["T"]
    ax[1].axvline(T, color="gray", ls="--", lw=1)
    ax[1].text(T + 5, 2.05, "← trained up to here", color="gray")
    ax[1].set(title=f"Held-out code loss by position, windows of {2 * T} (mean of seeds)\n"
                    "learned positions cannot run past their table",
              xlabel="position in window", ylabel="loss", ylim=(2, 8))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "03_ablation.png", dpi=130)
    plt.close(fig)
    print(f"wrote {VISUALS / '03_ablation.png'}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plots")
    a = sub.add_parser("ablation")
    a.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    a.add_argument("--seeds", nargs="+", type=int, default=[0])
    a.add_argument("--steps", type=int, default=2000)
    a.add_argument("--B", type=int, default=32)
    a.add_argument("--T", type=int, default=256)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--n-eval", type=int, default=512)
    sub.add_parser("summary")
    args = p.parse_args()
    if args.cmd == "plots":
        plots()
    elif args.cmd == "ablation":
        ablation(args)
    else:
        summarize(json.loads((RESULTS / "ablation.json").read_text())["runs"])


if __name__ == "__main__":
    main()
