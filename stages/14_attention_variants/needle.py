"""Needle in a haystack, for code. docs/14_attention_variants.md §5.

A haystack of real Python from the val set. At a chosen depth we insert

    SECRET_KEY = "48213"

and the context ends with

    assert SECRET_KEY == "

The model passes if its greedy continuation is the 5 digits. Our tokenizer
splits every digit (Stage 01), so retrieval means copying 5 tokens from far
back: exactly the long-range look-up that average loss hides.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from llmfs.data import open_bin


def make_case(tok, data, length: int, depth: float, rng: random.Random):
    key = "".join(rng.choice("0123456789") for _ in range(5))
    needle = tok.encode_ordinary(f'\nSECRET_KEY = "{key}"\n')
    query = tok.encode_ordinary('\nassert SECRET_KEY == "')
    n_hay = length - len(needle) - len(query)
    start = rng.randrange(0, len(data) - n_hay - 1)
    hay = data[start:start + n_hay].astype(np.int64).tolist()
    cut = int(depth * n_hay)
    ids = hay[:cut] + needle + hay[cut:] + query
    answer = tok.encode_ordinary(key)
    return ids, answer


@torch.no_grad()
def run_needle(model, tok, val_path, lengths, depths, n=20, seed=0, device="cuda"):
    data = open_bin(val_path)
    rng = random.Random(seed)
    grid = {}
    for L in lengths:
        for d in depths:
            ok = 0
            for _ in range(n):
                ids, ans = make_case(tok, data, L, d, rng)
                x = torch.tensor([ids], device=device)
                got = []
                for _ in range(len(ans)):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(x, last_only=True)
                    nxt = logits[:, -1].argmax(-1, keepdim=True)
                    got.append(int(nxt))
                    x = torch.cat([x, nxt], 1)
                ok += got == ans
            grid[f"{L}@{d}"] = ok / n
    return grid
