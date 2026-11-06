"""Stage 12 experiments. docs/12_modern_arch.md §9.

    uv run python stages/12_modern_arch/run.py ladder        # 9a
    uv run python stages/12_modern_arch/run.py gqa           # 9b
    uv run python stages/12_modern_arch/run.py window        # 9c
    uv run python stages/12_modern_arch/run.py rope_scale    # 9d (uses the ladder's final model)
    uv run python stages/12_modern_arch/run.py sinks         # 9e
    uv run python stages/12_modern_arch/run.py plot

All runs: 6 layers, C=384, 8 heads (d=48), T=512, B=16, same steps, lr, seed
and data. Only the switch named in the row changes.
"""

from __future__ import annotations

import argparse
import json
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

from llmfs.data import open_bin  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.model.llama import LlamaConfig, kv_bytes_per_token  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "12_modern_arch"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "12_modern_arch"
BINS = REPO / "data" / "bins" / "split_8192"
BASE = dict(arch="llama", n_layer=6, n_embd=384, n_head=8)
GPT2_LIKE = dict(norm="layer", pos="learned", mlp="gelu", bias=True, n_kv_head=8)
STEPS = 4000

LADDER = [
    ("GPT-2 style",        dict(GPT2_LIKE)),
    ("+ RMSNorm",          dict(GPT2_LIKE, norm="rms")),
    ("+ RoPE",             dict(GPT2_LIKE, norm="rms", pos="rope")),
    ("+ SwiGLU",           dict(GPT2_LIKE, norm="rms", pos="rope", mlp="swiglu")),
    ("+ no bias",          dict(GPT2_LIKE, norm="rms", pos="rope", mlp="swiglu", bias=False)),
    ("+ GQA (8q / 2kv)",   dict(GPT2_LIKE, norm="rms", pos="rope", mlp="swiglu", bias=False, n_kv_head=2)),
    ("+ QK-norm",          dict(GPT2_LIKE, norm="rms", pos="rope", mlp="swiglu", bias=False, n_kv_head=2,
                                qk_norm=True)),
]
LLAMA = dict(norm="rms", pos="rope", mlp="swiglu", bias=False)


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def read_json(name, default):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default


def run(name, model_kw, steps=STEPS, seed=0, save=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    c = TrainConfig(model={**BASE, **model_kw}, B=16, T=512, steps=steps, lr=1e-3, warmup=200,
                    eval_every=1000, name=name, seed=seed, compile=True)
    r = train(c)
    m = r["model"]
    cfg = m.cfg
    rec = dict(name=name, kw=model_kw, seed=seed, val=r["final"]["val"], params=r["params"],
               seconds=r["seconds"], kv_bytes_per_token_bf16=kv_bytes_per_token(cfg),
               curve=[[h["step"], h["loss"]] for h in r["hist"]])
    if save:
        CKPT.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CKPT / save, m)
    del r, m
    torch.cuda.empty_cache()
    print({k: v for k, v in rec.items() if k != "curve"}, flush=True)
    return rec


def cmd_ladder(args):
    out = read_json("ladder.json", [])
    for i, (name, kw) in enumerate(LADDER):
        for seed in range(args.seeds):
            if any(r["name"] == name and r["seed"] == seed for r in out):
                continue
            out.append(run(name, kw, seed=seed,
                           save="llama_final.pt" if (i == len(LADDER) - 1 and seed == 0) else None))
            write_json("ladder.json", out)


def cmd_gqa(args):
    out = read_json("gqa.json", [])
    rows = [(f"kv={k}", dict(LLAMA, n_kv_head=k)) for k in (8, 4, 2, 1)]
    # parameter-matched MQA: give the saved k/v weights back to the MLP
    base = LlamaConfig(**{k: v for k, v in BASE.items() if k != "arch"}, **LLAMA)
    saved = 2 * base.n_embd * (base.n_embd - base.head_dim)      # per layer, kv=8 -> kv=1
    extra = saved // (3 * base.n_embd)
    rows.append(("kv=1 + bigger MLP", dict(LLAMA, n_kv_head=1, mlp_hidden=base.hidden + extra)))
    for name, kw in rows:
        if any(r["name"] == name for r in out):
            continue
        out.append(run(name, kw))
        write_json("gqa.json", out)


def cmd_window(args):
    out = read_json("window.json", [])
    for w in (0, 128, 64, 16):
        name = "full" if w == 0 else f"window={w}"
        if any(r["name"] == name for r in out):
            continue
        rec = run(name, dict(LLAMA, n_kv_head=2, window=w), save=f"window{w}.pt")
        rec["by_position"] = loss_by_position(CKPT / f"window{w}.pt", 512)
        out.append(rec)
        write_json("window.json", out)


@torch.no_grad()
def loss_by_position(path, T, n=64, **over):
    model, _ = load_model(path, device="cuda", **over)
    model.eval()
    data = open_bin(BINS / "val_code.bin")
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    w = torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts])).cuda()
    tot = torch.zeros(T, device="cuda")
    for i in range(0, n, 8):
        b = w[i:i + 8]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(b[:, :-1])
        lp = torch.log_softmax(logits.float(), -1).gather(2, b[:, 1:, None])[..., 0]
        tot += -lp.sum(0)
    per = (tot / n).cpu().numpy()
    buckets = [0, 16, 64, 128, 256, 512, 1024, 2048, 4096]
    return {f"{a}-{b}": round(float(per[a:b].mean()), 4) for a, b in zip(buckets, buckets[1:]) if a < T}


def cmd_rope_scale(args):
    """The ladder's final RoPE model, trained at 512, evaluated at 512-2048."""
    path = CKPT / "llama_final.pt"
    out = {}
    for T in (512, 1024, 2048):
        for s in (1.0, 2.0, 4.0):
            if T == 512 and s != 1.0:
                continue
            key = f"T{T}_scale{s}"
            out[key] = loss_by_position(path, T, n=32, block_size=T, rope_scale=s)
            out[key]["mean"] = round(float(np.mean(list(out[key].values()))), 4)
            print(key, out[key], flush=True)
    write_json("rope_scale.json", out)


@torch.no_grad()
def cmd_sinks(args):
    """Share of attention on position 0, per layer, in the trained Llama model."""
    model, _ = load_model(CKPT / "llama_final.pt", device="cuda")
    model.eval()
    data = open_bin(BINS / "val_code.bin")
    T = 512
    starts = np.linspace(0, len(data) - T - 2, 16).astype(np.int64)
    x = torch.from_numpy(np.stack([data[s:s + T].astype(np.int64) for s in starts])).cuda()
    shares = []
    from llmfs.model.llama import apply_rope
    h = model.embed_tokens(x)
    rope = model.rope(0, T, x.device)
    for blk in model.layers:
        a = blk.self_attn
        cfg = a.cfg
        z = blk.input_layernorm(h)
        B = z.shape[0]
        q = a.q_proj(z).view(B, T, cfg.n_head, cfg.head_dim).transpose(1, 2)
        k = a.k_proj(z).view(B, T, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
        if cfg.qk_norm:
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        k = k.repeat_interleave(cfg.n_head // cfg.kv_heads, 1)
        att = (q @ k.transpose(-1, -2)) / cfg.head_dim ** 0.5
        att = att.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        att = att.softmax(-1)                                    # (B, H, T, T)
        # mean weight on key 0 for queries past position 32 (so it is not just "few choices")
        s0 = att[:, :, 32:, 0].mean((0, 2))                      # (H,)
        shares.append([round(v, 3) for v in s0.tolist()])
        h = blk(h, rope)
    uniform = float(np.mean([1 / (t + 1) for t in range(32, T)]))
    out = dict(share_on_token0=shares, uniform_baseline=round(uniform, 4))
    print(json.dumps(out, indent=1))
    write_json("sinks.json", out)


def cmd_plot(args):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    lad = read_json("ladder.json", [])
    names = [n for n, _ in LADDER if any(r["name"] == n for r in lad)]
    means = [np.mean([r["val"]["code"] for r in lad if r["name"] == n]) for n in names]
    stds = [np.std([r["val"]["code"] for r in lad if r["name"] == n]) for n in names]
    ax[0, 0].errorbar(range(len(names)), means, yerr=stds, fmt="o-", capsize=4)
    for i, (m, n) in enumerate(zip(means, names)):
        ax[0, 0].annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax[0, 0].set_xticks(range(len(names)), names, rotation=25, fontsize=8)
    ax[0, 0].set(ylabel="val loss (code)", title="GPT-2 → Llama, one change at a time")

    g = read_json("gqa.json", [])
    if g:
        ax[0, 1].plot([r["kv_bytes_per_token_bf16"] / 1024 for r in g], [r["val"]["code"] for r in g], "o")
        for r in g:
            ax[0, 1].annotate(f"{r['name']} ({r['params'] / 1e6:.1f}M)",
                              (r["kv_bytes_per_token_bf16"] / 1024, r["val"]["code"]),
                              textcoords="offset points", xytext=(5, 4), fontsize=8)
        ax[0, 1].set(xlabel="KV cache per token (KB, bf16)", ylabel="val loss (code)",
                     title="MHA → GQA → MQA")
    w = read_json("window.json", [])
    for i, r in enumerate(w):
        bp = r.get("by_position", {})
        ax[1, 0].plot(list(bp.keys()), list(bp.values()), "o-", color=f"C{i}",
                      label=f"{r['name']}: {r['val']['code']}")
    ax[1, 0].set(xlabel="position bucket", ylabel="val loss", title="Sliding window: loss by position")
    ax[1, 0].legend(fontsize=8)
    rs = read_json("rope_scale.json", {})
    for i, (k, v) in enumerate(rs.items()):
        items = [(kk, vv) for kk, vv in v.items() if kk != "mean"]
        ax[1, 1].plot([a for a, _ in items], [b for _, b in items], "o-", color=f"C{i}", label=k)
    ax[1, 1].set(xlabel="position bucket", ylabel="loss", title="Past the training length (512)")
    ax[1, 1].tick_params(axis="x", rotation=30)
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "12_modern_arch.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ladder", "gqa", "window", "rope_scale", "sinks", "plot"])
    p.add_argument("--seeds", type=int, default=2)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
