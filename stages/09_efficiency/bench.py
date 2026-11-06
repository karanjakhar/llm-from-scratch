"""Stage 09: the throughput table. docs/09_efficiency.md.

    uv run python stages/09_efficiency/bench.py peak      # 9a: best bf16 / fp32 / tf32 matmul FLOP/s
    uv run python stages/09_efficiency/bench.py stack     # 9b: one optimisation at a time
    uv run python stages/09_efficiency/bench.py profile   # 9c: top kernels in one step
    uv run python stages/09_efficiency/bench.py plot

Model: the workhorse shape, 8 layers, C=512, 8 heads, T=512, our 8,196 vocab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llmfs.data import MixedLoader  # noqa: E402
from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.train.trainer import _Ckpt  # noqa: E402

RESULTS = REPO / "results" / "09_efficiency"
VISUALS = REPO / "docs" / "visuals"
BINS = REPO / "data" / "bins" / "split_8192"
MODEL = dict(vocab_size=8196, n_layer=8, n_head=8, n_embd=512)
T = 512


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def flops_per_token(cfg: GPTConfig, T: int) -> float:
    """6N + 12·L·T·C, N = non-embedding params, plus the lm_head matmul (docs §1).

    The output head is a real (C, V) matmul even though its weight is tied
    to the embedding table, so it counts: 6·C·V per token.
    """
    C, L, V = cfg.n_embd, cfg.n_layer, cfg.vocab_size
    N = L * (12 * C * C + 13 * C)
    return 6 * N + 6 * C * V + 12 * L * T * C


def cmd_peak(args):
    res = {}
    n = 4096
    for name, dtype, tf32 in (("fp32", torch.float32, False), ("tf32", torch.float32, True),
                              ("bf16", torch.bfloat16, False), ("fp16", torch.float16, False)):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        reps = 20
        t = time.perf_counter()
        for _ in range(reps):
            a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
        res[name] = round(2 * n ** 3 / dt / 1e12, 2)                 # TFLOP/s
        print(name, res[name], "TFLOP/s", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    res["gpu"] = torch.cuda.get_device_name()
    write_json("peak.json", res)


# Each row adds ONE thing to the previous row.
STACK = [
    ("fp32 baseline",          dict()),
    ("+ TF32 matmuls",         dict(tf32=True)),
    ("+ bf16 autocast",        dict(tf32=True, bf16=True)),
    ("+ fused attention",      dict(tf32=True, bf16=True, flash=True)),
    ("+ fused AdamW",          dict(tf32=True, bf16=True, flash=True, fused_opt=True)),
    ("+ torch.compile",        dict(tf32=True, bf16=True, flash=True, fused_opt=True, compile=True)),
    ("+ act. checkpointing",   dict(tf32=True, bf16=True, flash=True, fused_opt=True, compile=True,
                                    ckpt=True)),
    ("+ batch 8 -> 32",        dict(tf32=True, bf16=True, flash=True, fused_opt=True, compile=True,
                                    ckpt=True, B=32)),
    ("compile, no ckpt, B=16", dict(tf32=True, bf16=True, flash=True, fused_opt=True, compile=True,
                                    B=16)),
]


def run_config(opts: dict, steps: int = 30, warmup: int = 10) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = opts.get("tf32", False)
    torch.backends.cudnn.allow_tf32 = opts.get("tf32", False)
    torch.manual_seed(0)
    B = opts.get("B", 8)
    cfg = GPTConfig(block_size=T, flash=opts.get("flash", False), **MODEL)
    model = GPT(cfg).cuda()
    if opts.get("ckpt"):
        model.transformer.h = torch.nn.ModuleList(_Ckpt(b) for b in model.transformer.h)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}]
    if opts.get("fused_opt"):
        opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95), fused=True)
    else:
        opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95), foreach=False)   # one kernel set per tensor
    fwd = torch.compile(model) if opts.get("compile") else model
    loader = MixedLoader({"code": (BINS / "train_code.bin", 1.0)}, B=B, T=T, device="cuda", seed=0)
    batches = [loader.next_batch() for _ in range(steps + warmup)]
    bf16 = opts.get("bf16", False)

    def step(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            _, loss = fwd(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    t_compile = time.perf_counter()
    for i in range(warmup):
        step(*batches[i])
    torch.cuda.synchronize()
    t_compile = time.perf_counter() - t_compile
    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.perf_counter()
    for i in range(warmup, warmup + steps):
        losses.append(step(*batches[i]))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_s = steps * B * T / dt
    return dict(B=B, tokens_per_s=round(tok_s), step_ms=round(dt / steps * 1000, 1),
                peak_mb=round(torch.cuda.max_memory_allocated() / 2**20),
                warmup_s=round(t_compile, 1), loss_last=round(losses[-1].item(), 4),
                tflops=round(tok_s * flops_per_token(cfg, T) / 1e12, 2))


def cmd_stack(args):
    peak = json.loads((RESULTS / "peak.json").read_text())
    rows = []
    for name, opts in STACK:
        try:
            r = run_config(opts)
        except torch.OutOfMemoryError:
            r = dict(oom=True)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        if not r.get("oom"):
            r["mfu_vs_measured_bf16"] = round(r["tflops"] / peak["bf16"], 3)
            r["speedup"] = round(r["tokens_per_s"] / rows[0]["tokens_per_s"], 2) if rows else 1.0
        rows.append(dict(name=name, opts=opts, **r))
        print(rows[-1], flush=True)
        write_json("stack.json", rows)


def cmd_profile(args):
    from torch.profiler import ProfilerActivity, profile
    out = {}
    for label, opts in (("baseline", dict()),
                        ("bf16+flash+fused", dict(tf32=True, bf16=True, flash=True, fused_opt=True))):
        torch.backends.cuda.matmul.allow_tf32 = opts.get("tf32", False)
        torch.manual_seed(0)
        model = GPT(GPTConfig(block_size=T, flash=opts.get("flash", False), **MODEL)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=opts.get("fused_opt", False),
                                foreach=None if opts.get("fused_opt") else False)
        x, y = MixedLoader({"code": (BINS / "train_code.bin", 1.0)}, B=8, T=T,
                           device="cuda", seed=0).next_batch()

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=opts.get("bf16", False)):
                _, loss = model(x, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
        evs = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
        tot = sum(e.self_device_time_total for e in evs)

        def bucket(n: str) -> str:
            n = n.lower()
            # attention first: fused-attention kernels are built on cutlass too
            if "flash" in n or "fmha" in n or "attention" in n or "efficient" in n:
                return "fused attention"
            if "copy" in n:
                return "dtype casts / copies"
            if any(k in n for k in ("gemm", "cutlass", "sm80", "sm89", "ampere", "matmul", "xmma")):
                return "matmul"
            if "softmax" in n:
                return "softmax"
            if "adam" in n or "multi_tensor" in n:
                return "optimizer"
            if "layer_norm" in n or "layernorm" in n:
                return "layernorm"
            if "reduce" in n or "sum" in n:
                return "reductions"
            return "other elementwise"

        buckets: dict[str, float] = {}
        for e in evs:
            b = bucket(e.key)
            buckets[b] = buckets.get(b, 0) + e.self_device_time_total
        top = sorted(evs, key=lambda e: -e.self_device_time_total)[:10]
        out[label] = dict(
            share={k: round(v / tot, 3) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
            top=[dict(name=e.key[:90], pct=round(e.self_device_time_total / tot, 3), kind=bucket(e.key)) for e in top],
            kernels=len(evs))
        print(label, out[label]["share"], flush=True)
        del model, opt
        torch.cuda.empty_cache()
    write_json("profile.json", out)


def cmd_plot(args):
    rows = [r for r in json.loads((RESULTS / "stack.json").read_text()) if not r.get("oom")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [r["name"] for r in rows]
    y = range(len(rows))
    ax[0].barh(y, [r["tokens_per_s"] / 1000 for r in rows], color="C0")
    for i, r in enumerate(rows):
        ax[0].text(r["tokens_per_s"] / 1000, i, f"  {r['speedup']}×  MFU {r['mfu_vs_measured_bf16']:.0%}",
                   va="center", fontsize=8)
    ax[0].set_yticks(list(y), names)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="thousand tokens / s", title="Throughput, one change per row")
    ax[0].set_xlim(0, max(r["tokens_per_s"] for r in rows) / 1000 * 1.45)
    ax[1].barh(y, [r["peak_mb"] for r in rows], color="C1")
    for i, r in enumerate(rows):
        ax[1].text(r["peak_mb"], i, f"  {r['peak_mb']} MB (B={r['B']})", va="center", fontsize=8)
    ax[1].set_yticks(list(y), [""] * len(rows))
    ax[1].invert_yaxis()
    ax[1].set(xlabel="peak MB allocated", title="Memory")
    ax[1].set_xlim(0, max(r["peak_mb"] for r in rows) * 1.4)
    fig.tight_layout()
    fig.savefig(VISUALS / "09_stack.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["peak", "stack", "profile", "plot"])
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
