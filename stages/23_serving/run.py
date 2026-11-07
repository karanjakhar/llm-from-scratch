"""Stage 23 experiments. docs/23_serving.md §7.

    uv run python stages/23_serving/run.py curve  --model checkpoints/11_pretrain/workhorse/final.pt
    uv run python stages/23_serving/run.py bursty --model ...
    uv run python stages/23_serving/run.py hf     --model ...   # HF generate baseline on the same weights
    uv run python stages/23_serving/run.py plot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
from llmfs.serve.engine import Engine, Request, serve  # noqa: E402

RESULTS = REPO / "results" / "23_serving"
VISUALS = REPO / "docs" / "visuals"
BINS = REPO / "data" / "bins" / "split_8192"


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def make_trace(n, rate, seed=0, bursty=False, max_len=1024):
    """Prompts are real val code windows (32-256 tokens); outputs 16-256 tokens, heavy-tailed."""
    rng = random.Random(seed)
    data = open_bin(BINS / "val_code.bin")
    t, reqs = 0.0, []
    for i in range(n):
        if bursty and rng.random() < 0.15:
            gap = 0.0                                            # bursts: several requests at once
        else:
            gap = rng.expovariate(rate)
        t += gap
        plen = rng.randint(32, 256)
        s = rng.randrange(0, len(data) - plen)
        out = int(min(256, 16 + rng.paretovariate(1.2) * 16))
        reqs.append(Request(i, data[s:s + plen].astype(np.int64).tolist(), out, arrival=t))
    return reqs


def fresh(trace):
    return [Request(r.rid, list(r.prompt), r.max_tokens, arrival=r.arrival) for r in trace]


def cmd_curve(a):
    model, _ = load_model(a.model, device="cuda")
    model = model.eval().to(torch.bfloat16)
    rows = []
    for slots in (1, 4, 16, 32):
        for rate in (2, 5, 10, 20, 40):
            eng = Engine(model, slots=slots, max_len=model.cfg.block_size)
            r = serve(eng, make_trace(a.n, rate, seed=1), "continuous")
            r.update(slots=slots, offered_rps=rate)
            rows.append(r)
            print(r, flush=True)
            write_json(f"curve_{a.tag}.json", rows)


def cmd_bursty(a):
    model, _ = load_model(a.model, device="cuda")
    model = model.eval().to(torch.bfloat16)
    trace = make_trace(a.n, a.rate, seed=2, bursty=True)
    out = {}
    for policy in ("static", "continuous"):
        eng = Engine(model, slots=16, max_len=model.cfg.block_size)
        reqs = fresh(trace)
        r = serve(eng, reqs, policy)
        used = np.mean([len(q.prompt) + len(q.out) for q in reqs])
        r["cache_used_fraction"] = round(float(used / eng.Tmax), 3)
        r["latencies"] = sorted(round(q.t_done - q.arrival, 3) for q in reqs)
        out[policy] = r
        print({k: v for k, v in r.items() if k != "latencies"}, flush=True)
    out["trace"] = dict(n=a.n, mean_rate=a.rate, out_tokens_mean=float(np.mean([q.max_tokens for q in trace])),
                        out_tokens_max=max(q.max_tokens for q in trace))
    write_json(f"bursty_{a.tag}.json", out)


def cmd_hf(a):
    """Hugging Face generate on the same weights (exported to GPT-2 layout)."""
    from transformers import GPT2Config, GPT2LMHeadModel
    model, _ = load_model(a.model, device="cpu")
    c = model.cfg
    hf = GPT2LMHeadModel(GPT2Config(vocab_size=c.vocab_size, n_positions=c.block_size, n_embd=c.n_embd,
                                    n_layer=c.n_layer, n_head=c.n_head, activation_function="gelu_new"))
    sd = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            continue
        name = k.removeprefix("transformer.")
        if name.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            v = v.T.contiguous()                                 # back to Conv1D layout
        sd["transformer." + name] = v
    missing = hf.load_state_dict(sd, strict=False)
    hf = hf.cuda().eval().to(torch.bfloat16)
    ours = model.cuda().eval().to(torch.bfloat16)
    x = torch.randint(0, 8192, (1, 64), device="cuda")
    with torch.no_grad():
        d = (hf(x).logits.float() - ours(x)[0].float()).abs().max().item()
    rows = dict(max_logit_diff_bf16=round(d, 4), missing=[k for k in missing.missing_keys if "attn.bias" not in k])
    for B in (1, 16):
        ids = torch.randint(0, 8192, (B, 128), device="cuda")
        hf.generate(ids[:, :8], max_new_tokens=8, do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        t = time.perf_counter()
        hf.generate(ids, max_new_tokens=128, min_new_tokens=128, do_sample=False, pad_token_id=0)
        torch.cuda.synchronize()
        rows[f"hf_tok_s_B{B}"] = round(B * 128 / (time.perf_counter() - t), 1)
        eng = Engine(ours, slots=B, max_len=c.block_size)
        reqs = [Request(i, ids[i].tolist(), 128) for i in range(B)]
        torch.cuda.synchronize()
        t = time.perf_counter()
        serve(eng, reqs, "continuous")
        rows[f"ours_tok_s_B{B}"] = round(B * 128 / (time.perf_counter() - t), 1)
    print(rows)
    write_json(f"hf_baseline_{a.tag}.json", rows)


def cmd_plot(a):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = json.loads((RESULTS / f"curve_{a.tag}.json").read_text())
    for i, s in enumerate(sorted({r["slots"] for r in rows})):
        rr = [r for r in rows if r["slots"] == s]
        ax[0].plot([r["tokens_per_s"] for r in rr], [r["latency_p50"] for r in rr], "o-", color=f"C{i}",
                   label=f"{s} slots")
        for r in rr:
            ax[0].annotate(f"{r['offered_rps']}/s", (r["tokens_per_s"], r["latency_p50"]), fontsize=6,
                           textcoords="offset points", xytext=(3, 3))
    ax[0].set(xlabel="throughput (tokens/s)", ylabel="median request latency (s)", yscale="log",
              title="Throughput vs latency, continuous batching")
    ax[0].legend()
    b = json.loads((RESULTS / f"bursty_{a.tag}.json").read_text())
    for i, pol in enumerate(("static", "continuous")):
        lat = b[pol]["latencies"]
        ax[1].plot(lat, np.linspace(0, 1, len(lat)), color=f"C{i + 3}",
                   label=f"{pol}: p50 {b[pol]['latency_p50']}s, p99 {b[pol]['latency_p99']}s")
    ax[1].set(xlabel="request latency (s)", ylabel="fraction of requests finished", xscale="log",
              title="Bursty load, 16 slots")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "23_serving.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["curve", "bursty", "hf", "plot"])
    p.add_argument("--model", default=str(REPO / "checkpoints" / "11_pretrain" / "workhorse" / "final.pt"))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--rate", type=float, default=10.0)
    p.add_argument("--tag", default="workhorse")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
