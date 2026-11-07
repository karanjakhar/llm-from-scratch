"""Stage 18: a reward model. docs/18_reward_model.md.

    RewardModel = pretrained/SFT backbone + a linear head C -> 1 on the last token.
    loss = -log σ(r(chosen) - r(rejected))          Bradley-Terry (docs §2)

Pairs (llmfs/post/prefs.py): correct function vs one-mutation bug.
Train: CodeSearchNet mutants. Test: MBPP mutants verified to fail the tests.
"""

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "16_sft"))
from sft import IGNORE, render, split_function  # noqa: E402

from llmfs.eval.codeeval import load_mbpp  # noqa: E402
from llmfs.eval.execution import run_many  # noqa: E402
from llmfs.post.prefs import mutate  # noqa: E402

CSN = REPO / "data" / "raw" / "codesearchnet" / "python"


def header_of(fn_src: str) -> str | None:
    """Signature + docstring of an (unparsed) function, as in Stage 16."""
    sp = split_function(fn_src)
    return sp[0] if sp else None


def csn_pairs(split="train", limit=None, seed=0, max_chars=1500) -> list[dict]:
    rng = random.Random(seed)
    t = pq.read_table(CSN / f"{split}-00000-of-00001.parquet", columns=["whole_func_string"])
    out = []
    for src in t.column("whole_func_string").to_pylist():
        sp = split_function(src)
        if sp is None or len(sp[1]) > max_chars:
            continue
        m = mutate(sp[1], rng)
        if m is None:
            continue
        good, bad = m
        hg, hb = header_of(good + "\n"), header_of(bad + "\n")
        if hg is None or hg != hb:                     # the bug must be in the body, not the prompt
            continue
        out.append(dict(prompt=hg, chosen=good + "\n", rejected=bad + "\n", source="csn"))
        if limit and len(out) >= limit:
            break
    return out


def mbpp_pairs(splits=("test",), seed=0, per_problem=3) -> list[dict]:
    """Reference solution vs mutants that FAIL the tests (verified)."""
    rng = random.Random(seed)
    cands = []
    for sp in splits:
        for p in load_mbpp(sp):
            for k in range(per_problem):
                m = mutate(p["reference"], random.Random(rng.random()))
                if m is None:
                    continue
                cands.append((p, m[0], m[1]))
    progs = []
    for p, good, bad in cands:
        progs += [good + "\n\n" + p["test"], bad + "\n\n" + p["test"]]
    res = run_many(progs, workers=16)
    out, seen = [], set()
    for i, (p, good, bad) in enumerate(cands):
        ok_good, ok_bad = res[2 * i][0], res[2 * i + 1][0]
        if ok_good and not ok_bad and (p["task_id"], bad) not in seen:
            seen.add((p["task_id"], bad))
            out.append(dict(prompt=p["prompt"], chosen=good + "\n", rejected=bad + "\n",
                            source="mbpp", task_id=p["task_id"]))
    return out


class RewardModel(nn.Module):
    """Backbone hidden states -> scalar at the last real token (docs §3)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        C = backbone.cfg.n_embd
        self.head = nn.Linear(C, 1)
        nn.init.normal_(self.head.weight, std=1 / (C + 1) ** 0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, ids, lengths):
        h = self.backbone.hidden(ids)                              # (B, T, C)
        last = h[torch.arange(len(ids), device=ids.device), lengths - 1]   # (B, C)
        return self.head(last.float())[:, 0]                       # (B,)


def encode_pair(tok, pr: dict, max_len=512):
    max_len = min(max_len, 640)    # 2 × 16 × 1,024-token sequences OOM'd on 8 GB (journal)
    """Chat format, so the RM reads exactly what a policy produces (prompt + reply)."""
    a = render(tok, pr["prompt"], pr["chosen"])[0][:max_len]
    b = render(tok, pr["prompt"], pr["rejected"])[0][:max_len]
    return a, b


def batch_ids(seqs, pad, device="cuda"):
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device), torch.tensor([len(s) for s in seqs], device=device)


def bt_loss(r_chosen, r_rejected):
    """-log σ(rc - rr). Derived in docs §2. logsigmoid is the stable form."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


@torch.no_grad()
def score(rm, tok, pairs, B=16):
    rm.eval()
    pad = tok.special_tokens["<|pad|>"]
    rc, rr = [], []
    for i in range(0, len(pairs), B):
        enc = [encode_pair(tok, p, rm.backbone.cfg.block_size) for p in pairs[i:i + B]]
        xa, la = batch_ids([a for a, _ in enc], pad)
        xb, lb = batch_ids([b for _, b in enc], pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rc += rm(xa, la).tolist()
            rr += rm(xb, lb).tolist()
    rm.train()
    acc = sum(a > b for a, b in zip(rc, rr)) / len(rc)
    return dict(accuracy=round(acc, 4), chosen=rc, rejected=rr)


def train_rm(rm, tok, pairs, steps=1500, B=16, lr=2e-5, flip=0.0, seed=0, log=print, evals=None):
    """flip: fraction of pairs whose labels are swapped (the noise ablation)."""
    import math
    rng = random.Random(seed)
    pairs = [dict(p) for p in pairs]
    n_flip = int(flip * len(pairs))
    for p in rng.sample(pairs, n_flip):
        p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
    pad = tok.special_tokens["<|pad|>"]
    opt = torch.optim.AdamW([{"params": rm.backbone.parameters(), "lr": lr},
                             {"params": rm.head.parameters(), "lr": lr * 10}], weight_decay=0.0)
    hist = []
    for step in range(steps):
        f = 0.5 * (1 + math.cos(math.pi * step / steps))
        for g, base in zip(opt.param_groups, (lr, lr * 10)):
            g["lr"] = base * (min(1.0, (step + 1) / 50)) * (0.1 + 0.9 * f)
        batch = rng.sample(pairs, B)
        opt.zero_grad(set_to_none=True)
        micro = 8                                                # accumulate to B (Stage 08 §6)
        for i in range(0, B, micro):
            enc = [encode_pair(tok, p, rm.backbone.cfg.block_size) for p in batch[i:i + micro]]
            xa, la = batch_ids([a for a, _ in enc], pad)
            xb, lb = batch_ids([b for _, b in enc], pad)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = bt_loss(rm(xa, la), rm(xb, lb))
            (loss * len(enc) / B).backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()
        if evals and (step % 250 == 0 or step == steps - 1):
            rec = dict(step=step, loss=round(loss.item(), 4))
            for name, ev in evals.items():
                rec[name] = score(rm, tok, ev)["accuracy"]
            hist.append(rec)
            log(rec)
    return hist
