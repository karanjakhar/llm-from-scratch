"""Stage 16 experiments. docs/16_sft.md §8.

    uv run python stages/16_sft/run.py --base checkpoints/11_pretrain/workhorse/final.pt --tag workhorse
    uv run python stages/16_sft/run.py --base ... --tag workhorse --prompt-loss     # the ablation

Writes results/16_sft/<tag>[_promptloss].json and checkpoints/16_sft/<tag>[...].pt.
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
sys.path.insert(0, str(HERE))
from sft import build_pairs, render, sft_val_loss, train_sft  # noqa: E402

from llmfs.eval.codeeval import evaluate, load_humaneval, load_mbpp  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import evaluate as lm_eval, make_val  # noqa: E402

RESULTS = REPO / "results" / "16_sft"
CKPT = REPO / "checkpoints" / "16_sft"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


def chat_fn(tok):
    return lambda user: render(tok, user)[0]


def code_scores(model, tok, chat):
    he = evaluate(model, tok, load_humaneval(), n=1, temperature=0.0, chat=chat, max_new=300)
    mb = evaluate(model, tok, load_mbpp("test"), n=1, temperature=0.0, mbpp=True, chat=chat, max_new=300)
    return dict(humaneval_pass1=round(he["pass@1"], 4), mbpp_pass1=round(mb["pass@1"], 4),
                humaneval_solved=he["solved"], mbpp_solved=mb["solved"],
                humaneval_failures=he["failure_kinds"], examples=he["examples"])


def main(a):
    tok = Tokenizer.load(str(TOK))
    model, _ = load_model(a.base, device="cuda")
    model.grad_checkpointing = a.act_ckpt
    val_lm = make_val({"code": str(REPO / "data/bins/split_8192/val_code.bin"),
                       "en": str(REPO / "data/bins/split_8192/val_en.bin")}, 512, 256)
    t0 = time.time()
    train_ex = build_pairs(tok, "train", max_tokens=min(512, model.cfg.block_size))
    val_ex = build_pairs(tok, "validation", max_tokens=min(512, model.cfg.block_size), limit=1000)
    print(f"pairs: train {len(train_ex)} val {len(val_ex)} ({time.time() - t0:.0f}s)", flush=True)
    res = dict(base=a.base, train_pairs=len(train_ex), val_pairs=len(val_ex), prompt_loss=a.prompt_loss)
    if not a.prompt_loss:
        res["before"] = dict(lm_val=lm_eval(model, val_lm), sft_val=sft_val_loss(model, val_ex, tok),
                             base_mode=code_scores(model, tok, None),
                             chat_mode=code_scores(model, tok, chat_fn(tok)))
        print("before", {k: v for k, v in res["before"].items()}, flush=True)
    hist = train_sft(model, train_ex, val_ex, tok, steps=a.steps, B=16, micro=a.micro, lr=a.lr,
                     include_prompt_loss=a.prompt_loss)
    res["hist"] = hist
    res["after"] = dict(lm_val=lm_eval(model, val_lm), sft_val=sft_val_loss(model, val_ex, tok),
                        chat_mode=code_scores(model, tok, chat_fn(tok)))
    # side by side, three fixed HumanEval prompts
    probs = load_humaneval()[:3]
    res["side_by_side"] = []
    sys.path.insert(0, str(REPO / "stages" / "07_generation"))
    from sample import generate_cached
    for p in probs:
        ids = torch.tensor([render(tok, p["prompt"])[0]], device="cuda")
        out = generate_cached(model, ids, 200, eot=tok.special_tokens["<|im_end|>"], temperature=0.0)
        res["side_by_side"].append(dict(task=p["task_id"], sft=tok.decode(out[0, ids.shape[1]:].tolist())))
    tag = a.tag + ("_promptloss" if a.prompt_loss else "")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(json.dumps(res, indent=1))
    CKPT.mkdir(parents=True, exist_ok=True)
    save_checkpoint(CKPT / f"{tag}.pt", model)
    print("after", {k: v for k, v in res["after"].items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--act-ckpt", action="store_true", help="activation checkpointing (needed for the 92M capstone on 8 GB)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--prompt-loss", action="store_true")
    p.add_argument("--micro", type=int, default=None, help="micro-batch size (gradient accumulation)")
    main(p.parse_args())
