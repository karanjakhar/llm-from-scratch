"""Tokenize TinyStories (2 of 4 train shards + validation) for the Stage 08 smoke test.

    uv run python stages/08_training/prepare_tinystories.py

TinyStories is a test fixture, never a target (CLAUDE.md). Same tokenizer as
everything else (split_8192), same encoder and binary format as Stage 02, so
the smoke test exercises exactly the code path the real runs use.
-> data/bins/tinystories/{train,val}.bin
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "02_data"))
from prepare import tokenize_to_bin  # noqa: E402

RAW = REPO / "data" / "raw" / "tinystories" / "data"
OUT = REPO / "data" / "bins" / "tinystories"
TOK = str(REPO / "data" / "tokenizers" / "split_8192.json")


def docs(pattern):
    for f in sorted(RAW.glob(pattern)):
        for i, t in enumerate(pq.read_table(f, columns=["text"]).column("text").to_pylist()):
            if t and t.strip():
                yield SimpleNamespace(uid=f"{f.name}:{i}", text=t.strip())


if __name__ == "__main__":
    for split, pat in (("val", "validation-*.parquet"), ("train", "train-*.parquet")):
        st = tokenize_to_bin(docs(pat), TOK, OUT / f"{split}.bin", workers=12)
        print(split, st, flush=True)
