"""Stage 15 experiments. docs/15_speculative.md §6.

    uv run python stages/15_speculative/run.py sweep     # speedup + acceptance vs γ and temperature
    uv run python stages/15_speculative/run.py plot

Target: the Stage 11 base model. Draft: the Stage 06 4-layer model. Both use
the split_8192 tokenizer, which is the one hard requirement for a draft.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stages" / "07_generation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sample import generate_cached  # noqa: E402
from speculative import speculative_generate  # noqa: E402

from llmfs.model import load_model  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

RESULTS = REPO / "results" / "15_speculative"
VISUALS = REPO / "docs" / "visuals"
PROMPTS = ["def is_prime(n):\n", "import numpy as np\n\n\ndef normalize(x):\n",
           "class LinkedList:\n    def __init__(self):\n", "def read_json(path):\n    \"\"\""]


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, time.perf_counter() - t


def cmd_sweep(args):
    tok = Tokenizer.load(str(REPO / "data" / "tokenizers" / "split_8192.json"))
    target, _ = load_model(args.target, device="cuda")
    draft, _ = load_model(args.draft, device="cuda", flash=True)
    target, draft = target.eval().to(torch.bfloat16), draft.eval().to(torch.bfloat16)
    n = 160
    rows = []
    for temp in (0.0, 0.7, 1.0):
        base_t = []
        for i, p in enumerate(PROMPTS):
            ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
            generate_cached(target, ids, 8, temperature=temp, seed=0)            # warm
            _, t = timed(lambda: generate_cached(target, ids, n, temperature=temp, seed=i))
            base_t.append(t)
        base = float(np.mean(base_t))
        for gamma in (1, 2, 3, 4, 6, 8):
            ts, acc, tpc = [], [], []
            same = True
            for i, p in enumerate(PROMPTS):
                ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
                (out, st), t = timed(lambda: speculative_generate(target, draft, ids, n, gamma=gamma,
                                                                  temperature=temp, seed=i))
                ts.append(t)
                acc.append(st["acceptance_rate"])
                tpc.append(st["tokens_per_target_call"])
                if temp == 0:
                    ref = generate_cached(target, ids, n, temperature=0.0)
                    same &= bool(torch.equal(ref, out))
            row = dict(temperature=temp, gamma=gamma, acceptance=round(float(np.mean(acc)), 3),
                       tokens_per_target_call=round(float(np.mean(tpc)), 2),
                       speedup=round(base / float(np.mean(ts)), 2),
                       baseline_tok_s=round(n / base, 1), spec_tok_s=round(n / float(np.mean(ts)), 1))
            if temp == 0:
                row["identical_to_greedy"] = same
            rows.append(row)
            print(row, flush=True)
    # cost ratio: one draft step vs one target step
    x = torch.tensor([tok.encode_ordinary(PROMPTS[0])], device="cuda")
    _, tt = timed(lambda: generate_cached(target, x, 100, temperature=0.0))
    _, td = timed(lambda: generate_cached(draft, x, 100, temperature=0.0))
    write_json(f"sweep_{args.tag}.json", dict(rows=rows, target=str(args.target), draft=str(args.draft),
                                  draft_step_cost_ratio=round(td / tt, 3),
                                  target_params=sum(p.numel() for p in target.parameters()),
                                  draft_params=sum(p.numel() for p in draft.parameters())))


def cmd_plot(args):
    d = json.loads((RESULTS / f"sweep_{args.tag}.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, temp in enumerate((0.0, 0.7, 1.0)):
        r = [x for x in d["rows"] if x["temperature"] == temp]
        ax[0].plot([x["gamma"] for x in r], [x["speedup"] for x in r], "o-", color=f"C{i}", label=f"τ={temp}")
        ax[1].plot([x["gamma"] for x in r], [x["acceptance"] for x in r], "o-", color=f"C{i}", label=f"τ={temp}")
    ax[0].axhline(1, color="gray", ls=":")
    ax[0].set(xlabel="γ (draft tokens per round)", ylabel="speedup vs target alone",
              title=f"Speculative decoding (draft/target step cost {d['draft_step_cost_ratio']})")
    ax[1].set(xlabel="γ", ylabel="acceptance rate", title="How often the target agrees")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(VISUALS / f"15_speculative_{args.tag}.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sweep", "plot"])
    p.add_argument("--target", default=str(REPO / "checkpoints" / "11_pretrain" / "workhorse" / "final.pt"))
    p.add_argument("--draft", default=str(REPO / "checkpoints" / "06_gpt" / "tied_12000.pt"))
    p.add_argument("--tag", default="workhorse")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
