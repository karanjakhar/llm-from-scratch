"""Stage 10 experiments. docs/10_scaling_laws.md §8.

    uv run python stages/10_scaling/run.py isoflop --budget 3e15
    uv run python stages/10_scaling/run.py isoflop --budget 1e16
    uv run python stages/10_scaling/run.py overtrain     # 8c
    uv run python stages/10_scaling/run.py mix           # 8d
    uv run python stages/10_scaling/run.py repeat        # 8e
    uv run python stages/10_scaling/run.py fit           # parabola fits + plot

Compute is counted honestly per token as 6N + 6CV + 12LTC (Stage 09 §1): the
non-embedding matmuls, the output head (a real matmul even when tied), and
attention scores. Every run's cosine schedule is set to its own length
(docs §2: the Kaplan/Chinchilla lesson).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llmfs.model.gpt import GPTConfig  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "10_scaling"
VISUALS = REPO / "docs" / "visuals"
BINS = REPO / "data" / "bins" / "split_8192"
V, T, B = 8196, 512, 16                           # 8,192 tokens per step

# (n_layer, n_embd); head size 64, C/L ≈ 64 (docs §4)
SIZES = {
    "L2_C128":  (2, 128),     # added after 3e15 showed the smallest size winning: bracket the optimum
    "L3_C192":  (3, 192),
    "L4_C256":  (4, 256),
    "L6_C384":  (6, 384),
    "L8_C512":  (8, 512),
    "L10_C640": (10, 640),
    "L12_C768": (12, 768),
}


def n_nonemb(L, C):
    return L * (12 * C * C + 13 * C)


def n_total(L, C):
    return n_nonemb(L, C) + V * C + T * C + 2 * C


def flops_per_token(L, C):
    return 6 * n_nonemb(L, C) + 6 * C * V + 12 * L * T * C


def lr_for(C):
    # Wider models want smaller steps. A mild rule anchored at our tested
    # 1e-3 for C=256 and GPT-2's 6e-4 for C=768.
    return round(1e-3 * (256 / C) ** 0.46, 6)


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def read_json(name, default):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default


def run_one(name, L, C, tokens, sources=None, seed=0, extra=None):
    steps = max(200, int(tokens // (B * T)))
    # qk_norm: the v1 sweep without it diverged at L4 (docs §8f, diagnose.py)
    kw = dict(model=dict(n_layer=L, n_head=C // 64, n_embd=C, block_size=T, qk_norm=True), B=B, T=T, steps=steps,
              lr=lr_for(C), warmup=min(200, steps // 10), eval_every=max(500, steps // 4),
              eval_windows=256, name=name, seed=seed, compile=True)
    if sources:
        kw["sources"] = sources
    kw.update(extra or {})
    torch.backends.cuda.matmul.allow_tf32 = True
    r = train(TrainConfig(**kw))
    tok = steps * B * T
    rec = dict(name=name, L=L, C=C, N_nonemb=n_nonemb(L, C), N_total=n_total(L, C), steps=steps,
               tokens=tok, flops=flops_per_token(L, C) * tok, tokens_per_param=round(tok / n_total(L, C), 1),
               val=r["final"]["val"], seconds=r["seconds"], lr=kw["lr"])
    del r
    torch.cuda.empty_cache()
    print(rec, flush=True)
    return rec


def cmd_isoflop(args):
    budget = float(args.budget)
    key = f"isoflop_{args.budget}.json"
    runs = read_json(key, [])
    done = {r["name"] for r in runs}
    for name in args.sizes.split(","):
        L, C = SIZES[name]
        key_name = name if args.seed == 0 else f"{name}_s{args.seed}"
        if key_name in done:
            continue
        tokens = budget / flops_per_token(L, C)
        runs.append(run_one(f"{key_name}@{args.budget}", L, C, tokens, seed=args.seed))
        runs[-1]["name"] = key_name
        runs[-1]["seed"] = args.seed
        write_json(key, runs)


def cmd_overtrain(args):
    L, C = SIZES[args.size]
    opt_tokens = 20 * n_total(L, C)
    runs = read_json("overtrain.json", [])
    for mult in (1, 2, 5, 10):
        if any(r["mult"] == mult for r in runs):
            continue
        r = run_one(f"{args.size}_x{mult}", L, C, mult * opt_tokens)
        r["mult"] = mult
        runs.append(r)
        write_json("overtrain.json", runs)


def cmd_mix(args):
    L, C = SIZES[args.size]
    tokens = 20 * n_total(L, C)
    runs = read_json("mix.json", [])
    for label, w in (("code100", (1.0, 0.0)), ("code85_en15", (0.85, 0.15)), ("code70_en30", (0.7, 0.3))):
        if any(r["mix"] == label for r in runs):
            continue
        src = {"code": (str(BINS / "train_code.bin"), w[0]), "en": (str(BINS / "train_en.bin"), w[1])}
        r = run_one(f"{args.size}_{label}", L, C, tokens, sources=src)
        r["mix"] = label
        runs.append(r)
        write_json("mix.json", runs)


def cmd_repeat(args):
    """Same budget: fresh data vs a 1/4 slice seen 4 times vs a 1/16 slice seen 16 times."""
    from llmfs.data import open_bin
    L, C = SIZES[args.size]
    tokens = int(20 * n_total(L, C))
    full = open_bin(BINS / "train_code.bin")
    runs = read_json("repeat.json", [])
    for epochs in (1, 4, 16):
        if any(r["epochs"] == epochs for r in runs):
            continue
        n = tokens // epochs + T + 1
        path = REPO / "data" / "bins" / "scratch" / f"code_slice_{epochs}ep.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        if epochs > 1:
            np.asarray(full[:n]).tofile(path)                    # the first n tokens, repeated by the sampler
            src = {"code": (str(path), 1.0)}
        else:
            src = {"code": (str(BINS / "train_code.bin"), 1.0)}
        r = run_one(f"{args.size}_{epochs}ep", L, C, tokens, sources=src)
        r["epochs"] = epochs
        r["unique_tokens"] = int(min(n, len(full)))
        runs.append(r)
        write_json("repeat.json", runs)
        if epochs > 1:
            path.unlink()


def cmd_fit(args):
    import matplotlib.ticker as mt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    fits = {}
    for i, f in enumerate(sorted(RESULTS.glob("isoflop_*.json"))):
        runs = json.loads(f.read_text())
        budget = f.stem.split("_")[1]
        N = np.array([r["N_total"] for r in runs], dtype=float)
        loss = np.array([r["val"]["code"] for r in runs])
        order = np.argsort(N)
        N, loss = N[order], loss[order]
        a, b, c = np.polyfit(np.log(N), loss, 2)
        n_opt = float(np.exp(-b / (2 * a)))
        d_opt = float(budget) / (6 * n_opt)
        fits[budget] = dict(N_opt=round(n_opt), D_opt_6N=round(d_opt), tokens_per_param=round(d_opt / n_opt, 1),
                            loss_at_opt=round(float(a * np.log(n_opt) ** 2 + b * np.log(n_opt) + c), 4),
                            chinchilla_N=round(math.sqrt(float(budget) / 120)))
        ax[0].plot(N, loss, "o", color=f"C{i}", label=f"C = {budget}")
        xs = np.linspace(np.log(N[0]) - 0.2, np.log(N[-1]) + 0.2, 100)
        ax[0].plot(np.exp(xs), a * xs ** 2 + b * xs + c, "-", color=f"C{i}", alpha=0.6)
        ax[0].axvline(n_opt, color=f"C{i}", ls=":")
        for r in runs:
            tp = r["tokens_per_param"]
            ax[0].annotate(f"{tp:.0f} tok/p" if tp >= 10 else f"{tp:.1f} tok/p", (r["N_total"], r["val"]["code"]),
                           fontsize=7, textcoords="offset points", xytext=(3, 3))
    ax[0].set(xscale="log", xlabel="parameters (total)", ylabel="val loss (code)",
              title="IsoFLOP curves (dotted: fitted optimum)")
    ax[0].xaxis.set_minor_formatter(mt.NullFormatter())
    ax[0].set_xticks([2e6, 5e6, 1e7, 2e7, 5e7], ["2M", "5M", "10M", "20M", "50M"])
    ax[0].legend()
    ot = read_json("overtrain.json", [])
    if ot:
        ot = sorted(ot, key=lambda r: r["mult"])
        ax[1].plot([r["tokens"] for r in ot], [r["val"]["code"] for r in ot], "o-")
        for r in ot:
            ax[1].annotate(f"{r['mult']}×", (r["tokens"], r["val"]["code"]), textcoords="offset points",
                           xytext=(4, 4))
        ax[1].set(xscale="log", xlabel="training tokens", ylabel="val loss (code)",
                  title=f"Past optimal: {ot[0]['name'].split('_x')[0]}")
        ax[1].xaxis.set_minor_formatter(mt.NullFormatter())
        ax[1].set_xticks([6e7, 1.2e8, 3e8, 6e8], ["60M", "120M", "300M", "600M"])
    mix = read_json("mix.json", [])
    rep = read_json("repeat.json", [])
    labels, code, en = [], [], []
    for r in mix:
        labels.append(r["mix"])
        code.append(r["val"]["code"])
        en.append(r["val"]["en"])
    for r in sorted(rep, key=lambda r: r["epochs"]):
        labels.append(f"{r['epochs']} epoch{'s' if r['epochs'] > 1 else ''}")
        code.append(r["val"]["code"])
        en.append(r["val"]["en"])
    x = np.arange(len(labels))
    ax[2].bar(x - 0.2, code, 0.4, label="val code")
    ax[2].bar(x + 0.2, en, 0.4, label="val English")
    ax[2].set_xticks(x, labels, rotation=30, fontsize=8)
    lo = min(code + en) if code else 0
    ax[2].set(ylim=(lo - 0.3, max(code + en) + 0.2) if code else None, ylabel="val loss",
              title="Mix ratio, and repeated data")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "10_scaling.png", dpi=130)
    write_json("fits.json", fits)
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["isoflop", "overtrain", "mix", "repeat", "fit"])
    p.add_argument("--budget", default="1e16")
    p.add_argument("--sizes", default="L3_C192,L4_C256,L6_C384,L8_C512,L10_C640")
    p.add_argument("--size", default="L4_C256")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
