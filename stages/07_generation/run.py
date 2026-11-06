"""Stage 07 experiments. docs/07_generation.md §8.

    uv run python stages/07_generation/run.py speed    # 8a-b: naive vs KV cache, time + memory
    uv run python stages/07_generation/run.py temps    # 8c: τ 0.1 / 1.0 / 2.0 and other knobs
    uv run python stages/07_generation/run.py plot

speed uses randomly initialised models: speed and token identity do not depend
on training. temps uses the Stage 06 checkpoint (4 layers, 12k steps).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sample import generate_cached, generate_naive  # noqa: E402

from llmfs.model.gpt import GPT, GPTConfig, load_checkpoint  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

RESULTS = REPO / "results" / "07_generation"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "06_gpt" / "tied_12000.pt"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"

SIZES = {"L4_C256": (4, 4, 256), "L8_C512": (8, 8, 512), "L12_C768": (12, 12, 768)}


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def cmd_speed(args):
    rows = []
    prompt_len = 16
    for name, (L, H, C) in SIZES.items():
        torch.manual_seed(0)
        m = GPT(GPTConfig(vocab_size=8196, block_size=1024, n_layer=L, n_head=H, n_embd=C)).cuda().eval()
        ids = torch.randint(0, 8196, (1, prompt_len), device="cuda")
        for total in (128, 256, 512, 1024):
            n = total - prompt_len
            generate_cached(m, ids, 8, temperature=0.0)                      # warm-up
            a, t_naive = timed(lambda: generate_naive(m, ids, n, temperature=0.0))
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            b, t_cache = timed(lambda: generate_cached(m, ids, n, temperature=0.0))
            peak_cached = torch.cuda.max_memory_allocated() - base
            same_greedy = bool(torch.equal(a, b))
            s1 = generate_naive(m, ids, min(n, 200), seed=7, temperature=1.0, top_p=0.95)
            s2 = generate_cached(m, ids, min(n, 200), seed=7, temperature=1.0, top_p=0.95)
            row = dict(model=name, total_len=total, new_tokens=n,
                       naive_s=round(t_naive, 3), cached_s=round(t_cache, 3),
                       speedup=round(t_naive / t_cache, 2),
                       naive_tok_s=round(n / t_naive, 1), cached_tok_s=round(n / t_cache, 1),
                       cache_bytes_theory=2 * L * C * 4 * total,          # fp32 model here
                       peak_extra_bytes_cached=int(peak_cached),
                       identical_greedy=same_greedy,
                       identical_sampled=bool(torch.equal(s1, s2)))
            rows.append(row)
            print(row, flush=True)
        del m
        torch.cuda.empty_cache()
    write_json("speed.json", rows)


# -- 8c: what the knobs do to text ------------------------------------------

PROMPTS = [
    "def is_prime(n):\n",
    "import os\nimport sys\n\n\ndef main():\n",
    "class Stack:\n    def __init__(self):\n",
    "The",
]

SETTINGS = {
    "greedy":        dict(temperature=0.0),
    "tau0.1":        dict(temperature=0.1),
    "tau0.7":        dict(temperature=0.7),
    "tau1.0":        dict(temperature=1.0),
    "tau1.5":        dict(temperature=1.5),
    "tau2.0":        dict(temperature=2.0),
    "tau1.0_topk40": dict(temperature=1.0, top_k=40),
    "tau1.0_topp0.9": dict(temperature=1.0, top_p=0.9),
    "tau0.7_rep1.3": dict(temperature=0.7, repetition_penalty=1.3),
    "greedy_rep1.3": dict(temperature=0.0, repetition_penalty=1.3),
}


def text_stats(ids: list[int], model, tok) -> dict:
    """Numbers that separate 'loops' from 'noise' (docs §8c)."""
    n = len(ids)
    bigrams = list(zip(ids, ids[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))          # 1 = never repeats a pair
    # Mean log-prob of the text under the model at τ=1: how "plausible"
    # the model itself finds what it wrote. Low = noise, very high = loops.
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        logits, _ = model(x)
    lp = torch.log_softmax(logits[0, :-1].float(), -1).gather(1, x[0, 1:, None]).mean().item()
    try:
        import ast
        ast.parse(tok.decode(ids))
        parses = True
    except Exception:
        parses = False
    return dict(distinct2=round(distinct2, 3), mean_logprob=round(lp, 3), parses=parses, tokens=n)


def cmd_temps(args):
    tok = Tokenizer.load(str(TOK))
    model, _ = load_checkpoint(CKPT, device="cuda", flash=True)
    model.eval()
    out = {}
    for sname, knobs in SETTINGS.items():
        rows = []
        for p in PROMPTS:
            for seed in range(5 if knobs.get("temperature", 0) > 0 else 1):
                ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
                g = generate_cached(model, ids, 200 - ids.shape[1], eot=tok.eot_id, seed=seed, **knobs)
                gen = g[0].tolist()
                st = text_stats(gen[ids.shape[1]:], model, tok)
                rows.append(dict(prompt=p, seed=seed, text=tok.decode(gen), **st))
        agg = {k: round(sum(r[k] for r in rows) / len(rows), 3)
               for k in ("distinct2", "mean_logprob", "parses")}
        out[sname] = dict(knobs=knobs, mean=agg, samples=rows)
        print(sname, agg, flush=True)
    write_json("temps.json", out)


def cmd_plot(args):
    VISUALS.mkdir(parents=True, exist_ok=True)
    sp = json.loads((RESULTS / "speed.json").read_text())
    tp = json.loads((RESULTS / "temps.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, name in enumerate(SIZES):
        r = [x for x in sp if x["model"] == name]
        ax[0].plot([x["total_len"] for x in r], [x["speedup"] for x in r], "o-", color=f"C{i}",
                   label=name)
    ax[0].set(xscale="log", xlabel="sequence length (prompt 16 + generated)",
              ylabel="speedup: naive time / cached time", title="KV cache speedup (greedy, fp32)")
    ax[0].set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].axhline(1, color="gray", ls=":")
    ax[0].legend()
    names = [k for k in tp if k.startswith(("greedy", "tau")) and "_" not in k] or list(tp)
    order = ["greedy", "tau0.1", "tau0.7", "tau1.0", "tau1.5", "tau2.0"]
    names = [k for k in order if k in tp]
    d2 = [tp[k]["mean"]["distinct2"] for k in names]
    lp = [tp[k]["mean"]["mean_logprob"] for k in names]
    ax[1].plot(names, d2, "o-", color="C0", label="distinct bigrams (1 = no repeats)")
    ax[1].set_ylabel("distinct-2", color="C0")
    ax2 = ax[1].twinx()
    ax2.plot(names, lp, "s--", color="C3", label="mean log-prob under the model")
    ax2.set_ylabel("mean log-prob per token", color="C3")
    ax[1].set_title("Temperature: loops at one end, noise at the other")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "07_generation.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["speed", "temps", "plot"])
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
