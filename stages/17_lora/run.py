"""Stage 17 experiments. docs/17_peft_lora.md §8.

    uv run python stages/17_lora/run.py --base checkpoints/11_pretrain/workhorse/final.pt --tag workhorse

Same data, steps and batch as Stage 16. Rows: full fine-tune, LoRA r ∈ {1,4,16,64},
QLoRA r=16 (NF4 base). Each: trainable params, peak memory, ms/step, completion
val loss, HumanEval/MBPP pass@1 (greedy, chat mode). The r=16 adapter is merged
and the merged model re-scored, to show merging is free.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "16_sft"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("sft_run", REPO / "stages" / "16_sft" / "run.py")
_sft_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sft_run)
code_scores, chat_fn = _sft_run.code_scores, _sft_run.chat_fn   # Stage 16's scorer
from sft import build_pairs, sft_val_loss, train_sft  # noqa: E402

from llmfs.model import load_model  # noqa: E402
from llmfs.post.lora import add_lora, merge_lora, quantize_base_nf4, trainable  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

RESULTS = REPO / "results" / "17_lora"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


def main(a):
    tok = Tokenizer.load(str(TOK))
    base, _ = load_model(a.base, device="cuda")
    train_ex = build_pairs(tok, "train", max_tokens=512)
    val_ex = build_pairs(tok, "validation", max_tokens=512, limit=1000)
    rows = []
    out = RESULTS / f"{a.tag}.json"
    if out.exists():
        rows = json.loads(out.read_text())
    setups = [("full", None, False), ("lora_r1", 1, False), ("lora_r4", 4, False),
              ("lora_r16", 16, False), ("lora_r64", 64, False), ("qlora_r16", 16, True)]
    for name, r, q in setups:
        if any(x["name"] == name for x in rows):
            continue
        model = copy.deepcopy(base)
        lr = a.lr
        if r is not None:
            add_lora(model, r=r, alpha=2 * r)
            lr = a.lora_lr
            if q:
                quantize_base_nf4(model)
        n_train = trainable(model) if r is not None else sum(p.numel() for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        t0 = time.time()
        hist = train_sft(model, train_ex, val_ex, tok, steps=a.steps, B=16, lr=lr, eval_every=1000)
        dt = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() - base_mem) / 2**20
        rec = dict(name=name, r=r, qlora=q, trainable=n_train, peak_train_mb_above_weights=round(peak),
                   ms_per_step=round(1000 * dt / a.steps, 1), sft_val=sft_val_loss(model, val_ex, tok),
                   hist=hist)
        sc = code_scores(model, tok, chat_fn(tok))
        rec.update(humaneval_pass1=sc["humaneval_pass1"], mbpp_pass1=sc["mbpp_pass1"])
        if name == "lora_r16":
            merge_lora(model)
            rec["merged_sft_val"] = sft_val_loss(model, val_ex, tok)
        rows.append(rec)
        print({k: v for k, v in rec.items() if k != "hist"}, flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-lr", type=float, default=1e-3)
    main(p.parse_args())
