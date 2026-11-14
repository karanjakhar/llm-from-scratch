"""Stage 22: one harness, every checkpoint. docs/22_evaluation.md §8.

    uv run python stages/22_evaluation/run.py table           # 8a
    uv run python stages/22_evaluation/run.py contamination   # 8b
    uv run python stages/22_evaluation/run.py mc              # 8c
    uv run python stages/22_evaluation/run.py plot            # docs/visuals/22_evaluation.png
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "16_sft"))

from sft import render  # noqa: E402

from llmfs.data import open_bin  # noqa: E402
from llmfs.eval.codeeval import evaluate, load_humaneval, load_mbpp  # noqa: E402
from llmfs.eval.execution import run_many  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.post.prefs import mutate  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import evaluate as lm_eval, make_val  # noqa: E402

RESULTS = REPO / "results" / "22_evaluation"
CK = REPO / "checkpoints"
BINS = REPO / "data" / "bins" / "split_8192"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"

# (label, path, chat?) — whatever exists is evaluated
CANDIDATES = [
    ("Stage 06 GPT 4L (5M)", CK / "06_gpt" / "tied_12000.pt", False),
    ("Stage 12 Llama 6L (14M)", CK / "12_modern_arch" / "llama_final.pt", False),
    ("Stage 13 MoE 8e top-2", CK / "13_moe" / "moe_8e_top2.pt", False),
    ("workhorse base", CK / "11_pretrain" / "workhorse" / "final.pt", False),
    ("workhorse SFT", CK / "16_sft" / "workhorse.pt", True),
    ("workhorse DPO", CK / "19_dpo" / "workhorse.pt", True),
    ("workhorse GRPO", CK / "20_grpo" / "workhorse.pt", True),
    ("capstone base", CK / "11_pretrain" / "capstone" / "final.pt", False),
    ("capstone SFT", CK / "16_sft" / "capstone.pt", True),
    ("capstone DPO", CK / "19_dpo" / "capstone.pt", True),
    ("capstone GRPO", CK / "20_grpo" / "capstone.pt", True),
]


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def bits_per_byte(loss_nats, split):
    """Our tokens average this many bytes on val (Stage 06 oracle measured on the same text)."""
    o = json.loads((REPO / "results" / "06_gpt" / "oracle.json").read_text())
    return round(loss_nats / math.log(2) / o[split]["our_bytes_per_token"], 4)


def cmd_table(a):
    tok = Tokenizer.load(str(TOK))
    he, mb = load_humaneval(), load_mbpp("test")
    rows = []
    prev = RESULTS / "table.json"
    done = {r["model"]: r for r in json.loads(prev.read_text())} if prev.exists() and not a.redo else {}
    for label, path, chat in CANDIDATES:
        if not path.exists():
            continue
        if label in done:
            rows.append(done[label])
            continue
        model, _ = load_model(path, device="cuda")
        model.eval()
        T = min(512, model.cfg.block_size)
        v = lm_eval(model, make_val({"code": str(BINS / "val_code.bin"), "en": str(BINS / "val_en.bin")}, T, 256))
        cf = (lambda u: render(tok, u)[0]) if chat else None
        mx = min(300, model.cfg.block_size - 200)
        r_he = evaluate(model, tok, he, n=1, temperature=0.0, chat=cf, max_new=mx)
        r_mb = evaluate(model, tok, mb, n=1, temperature=0.0, mbpp=True, chat=cf, max_new=mx)
        row = dict(model=label, params=sum(p.numel() for p in model.parameters()), chat=chat,
                   val_code=v["code"], val_en=v["en"], bpb_code=bits_per_byte(v["code"], "val_code"),
                   humaneval_pass1=round(r_he["pass@1"], 4), mbpp_pass1=round(r_mb["pass@1"], 4),
                   humaneval_solved=r_he["solved"], mbpp_solved=r_mb["solved"])
        if a.pass10 and ("workhorse" in label or "capstone" in label):
            r10 = evaluate(model, tok, he, n=10, temperature=0.8, ks=(1, 10), chat=cf, max_new=mx)
            m10 = evaluate(model, tok, mb, n=10, temperature=0.8, ks=(1, 10), mbpp=True, chat=cf, max_new=mx)
            row.update(humaneval_pass10=round(r10["pass@10"], 4), humaneval_pass1_t08=round(r10["pass@1"], 4),
                       mbpp_pass10=round(m10["pass@10"], 4), mbpp_pass1_t08=round(m10["pass@1"], 4))
        rows.append(row)
        print({k: v for k, v in row.items() if "solved" not in k}, flush=True)
        write_json("table.json", rows)
        del model
        torch.cuda.empty_cache()
    write_json("table.json", rows)


def ngram_hashes(ids, n=13):
    ids = np.asarray(ids, dtype=np.uint64)
    if len(ids) < n:
        return set()
    P = np.uint64(1000003)
    h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * P + ids[k:len(ids) - n + 1 + k]
    return set(h.tolist())


def cmd_contamination(a):
    tok = Tokenizer.load(str(TOK))
    import pyarrow.parquet as pq
    he_rows = pq.read_table(REPO / "data/raw/humaneval/openai_humaneval/test-00000-of-00001.parquet").to_pylist()
    bench = {}
    for r in he_rows:
        bench[r["task_id"]] = ngram_hashes(tok.encode_ordinary(r["prompt"] + r["canonical_solution"]))
    for p in load_mbpp("test"):
        bench[p["task_id"]] = ngram_hashes(tok.encode_ordinary(p["reference"]))
    want = set().union(*bench.values())
    data = open_bin(BINS / "train_code.bin")
    n, chunk = 13, 50_000_000
    found = set()
    P = np.uint64(1000003)
    for s in range(0, len(data) - n, chunk):
        ids = np.asarray(data[s:s + chunk + n - 1], dtype=np.uint64)
        h = np.zeros(len(ids) - n + 1, dtype=np.uint64)
        for k in range(n):
            h = h * P + ids[k:len(ids) - n + 1 + k]
        hit = np.isin(h, np.fromiter(want, dtype=np.uint64))
        found |= set(h[hit].tolist())
        print(f"scanned {min(s + chunk, len(data)) / 1e6:.0f}M tokens, {len(found)} benchmark 13-grams seen", flush=True)
    flagged = {t: round(len(hs & found) / max(1, len(hs)), 3) for t, hs in bench.items() if hs & found}
    # "any shared 13-gram" catches boilerplate (`from typing import List\n\n\ndef`).
    # Report counts by the FRACTION of a problem's 13-grams seen in training too.
    th = {f">={t}": dict(humaneval=sum(v >= t for k, v in flagged.items() if k.startswith("Human")),
                         mbpp=sum(v >= t for k, v in flagged.items() if k.startswith("mbpp")))
          for t in (0.0, 0.05, 0.2, 0.5)}
    out = dict(train_tokens=int(len(data)), n=n, flagged=flagged, by_fraction=th,
               humaneval_flagged=sum(t.startswith("HumanEval") for t in flagged),
               mbpp_flagged=sum(t.startswith("mbpp") for t in flagged))
    tab = RESULTS / "table.json"
    if tab.exists():
        split = {}
        for r in json.loads(tab.read_text()):
            solved = set(r["humaneval_solved"] + r["mbpp_solved"])
            fl = {t for t, v in flagged.items() if v >= 0.2}       # "heavily overlapping"
            split[r["model"]] = dict(solved_flagged=len(solved & fl), solved_unflagged=len(solved - fl))
        out["solved_split"] = split
    write_json("contamination.json", out)
    print({k: v for k, v in out.items() if k != "flagged"})


@torch.no_grad()
def cmd_mc(a):
    """MBPP: reference vs 3 failing mutants; pick by summed vs mean log-prob."""
    tok = Tokenizer.load(str(TOK))
    rng = random.Random(0)
    items = []
    for p in load_mbpp("test"):
        muts = []
        for k in range(12):
            m = mutate(p["reference"], random.Random(rng.random()))
            if m and m[1] not in muts:
                good = m[0]
                muts.append(m[1])
        if len(muts) < 3:
            continue
        res = run_many([good + "\n\n" + p["test"]] + [x + "\n\n" + p["test"] for x in muts], workers=16)
        if not res[0][0]:
            continue
        bad = [x for x, (ok, _) in zip(muts, res[1:]) if not ok][:3]
        if len(bad) == 3:
            items.append(dict(prompt=p["prompt"], options=[good] + bad))
    print(f"{len(items)} items", flush=True)
    out = {}
    for label, path, chat in CANDIDATES:
        if not path.exists():
            continue
        model, _ = load_model(path, device="cuda")
        model.eval()
        right_sum = right_norm = 0
        lens = []
        for it in items:
            sums, means = [], []
            for opt in it["options"]:
                if chat:
                    ids, labels = render(tok, it["prompt"], opt + "\n")
                    start = next(i for i, l in enumerate(labels) if l != -100)
                else:
                    pre = tok.encode_ordinary(it["prompt"])
                    ids = pre + tok.encode_ordinary(opt + "\n")
                    start = len(pre)
                x = torch.tensor([ids[:model.cfg.block_size]], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)
                lp = torch.log_softmax(logits[0, :-1].float(), -1).gather(1, x[0, 1:, None])[:, 0]
                comp = lp[start - 1:]
                sums.append(comp.sum().item())
                means.append(comp.mean().item())
                lens.append(len(comp))
            right_sum += int(np.argmax(sums) == 0)
            right_norm += int(np.argmax(means) == 0)
        out[label] = dict(acc_sum=round(right_sum / len(items), 4), acc_norm=round(right_norm / len(items), 4))
        print(label, out[label], flush=True)
        del model
        torch.cuda.empty_cache()
    rank_sum = sorted(out, key=lambda k: -out[k]["acc_sum"])
    rank_norm = sorted(out, key=lambda k: -out[k]["acc_norm"])
    write_json("mc.json", dict(items=len(items), chance=0.25, scores=out, rank_sum=rank_sum, rank_norm=rank_norm))


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = json.loads((RESULTS / "table.json").read_text())
    mc = json.loads((RESULTS / "mc.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    # (a) HumanEval as problem counts: at this scale a percentage hides how few problems it is
    names = [r["model"] for r in tab][::-1]
    g = [round(r["humaneval_pass1"] * 164) for r in tab][::-1]
    p10 = [r.get("humaneval_pass10", 0) * 164 for r in tab][::-1]
    y = np.arange(len(names))
    ax[0].barh(y + 0.2, g, 0.38, color="C0", label="greedy pass@1 (problems solved)")
    ax[0].barh(y - 0.2, p10, 0.38, color="C1", label="pass@10 × 164 (τ = 0.8)")
    ax[0].set(yticks=y, yticklabels=names, xlabel="HumanEval problems (of 164)",
              title="HumanEval: the untuned capstone is as good as anything after it")
    ax[0].legend(loc="upper right", fontsize=8)          # the small models' rows are empty there
    ax[0].grid(axis="x", alpha=0.3)
    # (b) the same models under two scoring rules; a line joins each model's two scores
    sc = mc["scores"]
    names_b = list(sc)[::-1]
    yb = np.arange(len(names_b))
    for i, k in enumerate(names_b):
        ax[1].plot([sc[k]["acc_sum"] * 100, sc[k]["acc_norm"] * 100], [i, i], color="gray", lw=1.5, zorder=1)
    ax[1].scatter([sc[k]["acc_sum"] * 100 for k in names_b], yb, s=50, color="C3", label="summed log-prob", zorder=2)
    ax[1].scatter([sc[k]["acc_norm"] * 100 for k in names_b], yb, s=50, color="C0", label="per-token log-prob", zorder=2)
    ax[1].axvline(25, color="gray", ls=":", lw=1)
    ax[1].text(25.5, len(names_b) - 0.7, "chance", fontsize=8, color="gray")
    ax[1].set(yticks=yb, yticklabels=names_b, xlim=(20, 85), xlabel="% of 352 MBPP items: working program picked over 3 mutants",
              title="Length normalisation: +4-7 points, except for DPO")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPO / "docs" / "visuals" / "22_evaluation.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["table", "contamination", "mc", "plot"])
    p.add_argument("--pass10", action="store_true")
    p.add_argument("--redo", action="store_true")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
