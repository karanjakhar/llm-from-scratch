"""Stage 18 experiments. docs/18_reward_model.md §8.

    uv run python stages/18_reward/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse
    uv run python stages/18_reward/run.py ... --flip 0.2       # the label-noise ablation
    uv run python stages/18_reward/run.py --plot --tag workhorse
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
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reward import RewardModel, csn_pairs, mbpp_pairs, score, train_rm  # noqa: E402

from llmfs.model import load_model  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

RESULTS = REPO / "results" / "18_reward"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "18_reward"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


def calibration(rc, rr, bins=5):
    m = np.array(rc) - np.array(rr)
    p = 1 / (1 + np.exp(-m))
    conf = np.maximum(p, 1 - p)                 # confidence in the predicted direction
    right = (m > 0).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    out = []
    for a, b in zip(edges, edges[1:]):
        sel = (conf >= a) & (conf < b + 1e-9)
        if sel.sum():
            out.append(dict(bin=f"{a:.1f}-{b:.1f}", n=int(sel.sum()), mean_conf=round(float(conf[sel].mean()), 3),
                            accuracy=round(float(right[sel].mean()), 3)))
    return out


def main(a):
    tok = Tokenizer.load(str(TOK))
    train = csn_pairs("train", limit=40000)
    held = csn_pairs("validation", limit=1000)
    mb = mbpp_pairs(("test", "validation"))
    print(f"pairs: train {len(train)} csn-val {len(held)} mbpp-verified {len(mb)}", flush=True)
    backbone, _ = load_model(a.sft, device="cuda")
    backbone.grad_checkpointing = a.act_ckpt
    rm = RewardModel(backbone).cuda()
    before = dict(csn_val=score(rm, tok, held)["accuracy"], mbpp=score(rm, tok, mb)["accuracy"])
    hist = train_rm(rm, tok, train, steps=a.steps, flip=a.flip,
                    evals={"csn_val": held[:300], "mbpp": mb[:300]})
    sh, sm = score(rm, tok, held), score(rm, tok, mb)
    res = dict(sft=a.sft, flip=a.flip, n_train=len(train), before=before, hist=hist,
               csn_val_acc=sh["accuracy"], mbpp_acc=sm["accuracy"],
               mbpp_scores=dict(chosen=sm["chosen"], rejected=sm["rejected"]),
               calibration_mbpp=calibration(sm["chosen"], sm["rejected"]),
               calibration_csn=calibration(sh["chosen"], sh["rejected"]))
    # accuracy by mutation kind on MBPP: which bugs are easy to see?
    import difflib
    kinds = {}
    for p, c, r in zip(mb, sm["chosen"], sm["rejected"]):
        d = [l for l in difflib.ndiff(p["chosen"].split(), p["rejected"].split()) if l[:2] in ("- ", "+ ")]
        k = " → ".join(x[2:] for x in d[:2])[:30] if d else "?"
        key = ("return None" if "None" in k else "comparison" if any(o in k for o in ("<", ">", "==", "!=", " in"))
               else "and/or" if ("and" in k or "or" in k) else "arith" if any(o in k for o in "+-*/%") else "const/other")
        kinds.setdefault(key, []).append(c > r)
    res["mbpp_acc_by_kind"] = {k: dict(n=len(v), acc=round(float(np.mean(v)), 3)) for k, v in kinds.items()}
    tag = a.tag + (f"_flip{a.flip}" if a.flip else "")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=1))
    if not a.flip:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"config": backbone.cfg.to_dict(), "model": rm.state_dict()}, CKPT / f"{a.tag}.pt")
    print({k: v for k, v in res.items() if k not in ("mbpp_scores", "hist")}, flush=True)


def plot(a):
    base = json.loads((RESULTS / f"{a.tag}.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    s = base["mbpp_scores"]
    ax[0].hist(s["chosen"], bins=40, alpha=0.6, label="chosen (passes tests)", color="C2")
    ax[0].hist(s["rejected"], bins=40, alpha=0.6, label="rejected (fails tests)", color="C3")
    ax[0].set(xlabel="reward", ylabel="pairs", title=f"MBPP verified pairs: accuracy {base['mbpp_acc']:.1%}")
    ax[0].legend()
    for name, color in ((a.tag, "C0"), (f"{a.tag}_flip0.2", "C1"), (f"{a.tag}_flip0.4", "C3")):
        p = RESULTS / f"{name}.json"
        if p.exists():
            h = json.loads(p.read_text())["hist"]
            ax[1].plot([x["step"] for x in h], [x["mbpp"] for x in h], "o-", color=color,
                       label=f"{name.replace(a.tag + '_', '') if '_flip' in name else 'clean labels'}")
    ax[1].axhline(0.5, color="gray", ls=":")
    ax[1].set(xlabel="step", ylabel="MBPP pairwise accuracy", title="Label noise sets a ceiling")
    ax[1].legend()
    cal = base["calibration_mbpp"]
    ax[2].plot([c["mean_conf"] for c in cal], [c["accuracy"] for c in cal], "o-", label="reward model")
    ax[2].plot([0.5, 1], [0.5, 1], ":", color="gray", label="perfect calibration")
    ax[2].set(xlabel="predicted P(chosen better)", ylabel="actual accuracy", title="Calibration (MBPP)")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "18_reward.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--act-ckpt", action="store_true", help="activation checkpointing (needed for the 92M capstone on 8 GB)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--flip", type=float, default=0.0)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    plot(a) if a.plot else main(a)
