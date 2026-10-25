"""Stage 02, step 2 — tokenize the corpus once into flat uint16 binaries.

    uv run python stages/02_data/prepare.py --tokenizer data/tokenizers/grouped_8192.json

Writes to data/bins/<tokenizer-name>/:

    train_code.bin  val_code.bin  train_en.bin  val_en.bin    raw uint16, no header
    manifest.json                                              everything needed to reproduce them

Each document is encoded with specials DISABLED and followed by exactly one
<|endoftext|> (docs/02_data.md §7). Code and English are kept in separate
files so the mixing ratio is a runtime knob, not a re-tokenization (§9).

Exact dedup uses one hash set shared across splits, with train processed
first: a val file that is a byte-identical copy of a train file (a fork under
another repo name) is dropped from val. The count of such drops is leakage the
repository-level split cannot catch on its own (§4, §5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import islice
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llmfs.data.corpus import (  # noqa: E402
    PERMISSIVE_LICENSES, VAL_PER_10K, Doc, code_shards, en_shards, iter_code_docs, iter_en_docs,
)
from llmfs.tokenizer import Tokenizer  # noqa: E402

RAW = REPO / "data" / "raw"
DTYPE = np.uint16

# -- worker side --------------------------------------------------------------

_TOK: Tokenizer | None = None


def _init_worker(tok_path: str):
    global _TOK
    _TOK = Tokenizer.load(tok_path)        # once per process, then its cache warms


def _encode_batch(texts: list[str]) -> bytes:
    """Encode a batch of documents into one contiguous uint16 byte string."""
    ids: list[int] = []
    for t in texts:
        ids.extend(_TOK.encode_document(t))
    return np.asarray(ids, dtype=DTYPE).tobytes()


# -- main side ----------------------------------------------------------------

def _batched(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


def drop_near_dups(docs: Iterable[Doc], drop: set[str], stats: dict) -> Iterator[Doc]:
    """Skip docs that stages/02_data/dedup.py marked as near-duplicates."""
    for d in docs:
        if d.uid in drop:
            stats["near_dup_docs"] += 1
            continue
        yield d


class Deduper:
    """Exact dedup by content hash. 16-byte digests: ~100 bytes/doc in a set."""

    def __init__(self):
        self.seen: set[bytes] = set()

    def __call__(self, docs: Iterable[Doc], stats: dict) -> Iterator[Doc]:
        for d in docs:
            h = hashlib.blake2b(d.text.encode("utf-8"), digest_size=16).digest()
            if h in self.seen:
                stats["dup_docs"] += 1
                continue
            self.seen.add(h)
            yield d


def tokenize_to_bin(docs: Iterable[Doc], tok_path: str, out_path: Path,
                    workers: int, batch_docs: int = 64) -> dict:
    """Stream docs -> parallel encode -> append to out_path. Returns stats.

    Pool.imap is ORDERED, so the output is byte-identical for any worker count
    (test_data.py asserts it). imap_unordered would be marginally faster and
    would make the bins irreproducible.
    """
    stats = {"docs": 0, "bytes": 0, "tokens": 0}

    def texts():
        for batch in _batched(docs, batch_docs):
            stats["docs"] += len(batch)
            stats["bytes"] += sum(len(d.text.encode("utf-8")) for d in batch)
            yield [d.text for d in batch]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    last = t0
    # "spawn", not the Linux default "fork": by the time we get here torch has
    # been imported (llmfs.data pulls in the loader) and torch starts threads.
    # fork() copies only the calling thread, so a lock held by any other thread
    # is copied locked forever -- a rare, silent deadlock in the child. Spawn
    # starts clean interpreters; the cost is ~1 s of startup per pool.
    ctx = mp.get_context("spawn")
    with open(out_path, "wb") as f, ctx.Pool(workers, _init_worker, (tok_path,)) as pool:
        for blob in pool.imap(_encode_batch, texts(), chunksize=4):
            f.write(blob)
            stats["tokens"] += len(blob) // DTYPE().itemsize
            now = time.perf_counter()
            if now - last > 15:
                print(f"    {out_path.name}: {stats['tokens']/1e6:,.0f}M tokens, "
                      f"{stats['bytes']/1e6/(now-t0):.1f} MB/s", flush=True)
                last = now
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="per (source, split); for quick end-to-end checks")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--near-dups", type=str, default=str(REPO / "data" / "dedup" / "near_dups.json"),
                    help="drop list from stages/02_data/dedup.py")
    ap.add_argument("--no-near-dedup", action="store_true",
                    help="skip near-dup removal (only for measuring what it buys)")
    args = ap.parse_args()

    tok_path = str(Path(args.tokenizer).resolve())
    tok = Tokenizer.load(tok_path)
    assert tok.vocab_size <= np.iinfo(DTYPE).max + 1, "vocab no longer fits in uint16"
    assert "<|endoftext|>" in tok.special_tokens, "tokenizer needs <|endoftext|>"

    out_dir = Path(args.out) if args.out else REPO / "data" / "bins" / Path(tok_path).stem
    cs, es = code_shards(RAW), en_shards(RAW)
    print(f"tokenizer {Path(tok_path).name}: vocab_size {tok.vocab_size}, eot {tok.eot_id}")
    print(f"code shards {len(cs)}, English files {len(es)} -> {out_dir}")

    near: set[str] = set()
    near_meta = None
    if not args.no_near_dedup:
        nd = Path(args.near_dups)
        if not nd.exists():
            raise SystemExit(f"{nd} missing: run stages/02_data/dedup.py, or pass --no-near-dedup")
        payload = json.loads(nd.read_text())
        near = set(payload["drop"])
        near_meta = {"file": str(nd.relative_to(REPO)), "sha256": sha256(nd),
                     "params": payload["params"], "uids": len(near)}
        print(f"near-dup drop list: {len(near):,} code docs")

    dedup = Deduper()
    plan = [  # train before val: val copies of train docs get dropped, not the reverse
        ("train_code", lambda: iter_code_docs(cs, "train")),
        ("train_en",   lambda: iter_en_docs(es, "train")),
        ("val_code",   lambda: iter_code_docs(cs, "val")),
        ("val_en",     lambda: iter_en_docs(es, "val")),
    ]
    bins = {}
    for name, make in plan:
        st = {"dup_docs": 0, "near_dup_docs": 0}
        docs = dedup(drop_near_dups(make(), near, st), st)
        if args.limit_docs:
            docs = islice(docs, args.limit_docs)
        print(f"  {name} ...", flush=True)
        stats = tokenize_to_bin(docs, tok_path, out_dir / f"{name}.bin", args.workers)
        stats["dup_docs_dropped"] = st["dup_docs"]
        stats["near_dup_docs_dropped"] = st["near_dup_docs"]
        stats["bytes_per_token"] = round(stats["bytes"] / max(1, stats["tokens"] - stats["docs"]), 4)
        stats["sha256"] = sha256(out_dir / f"{name}.bin")
        bins[name] = stats
        print(f"  {name}: {stats['docs']:,} docs, {stats['tokens']/1e6:,.1f}M tokens, "
              f"{stats['bytes_per_token']} B/tok, {stats['dup_docs_dropped']:,} dups dropped, "
              f"{stats['seconds']}s")

    sources = RAW / "sources.json"
    manifest = {
        "sources": json.loads(sources.read_text()) if sources.exists() else None,
        "code_shards_used": [p.name for p in cs],
        "en_files_used": [str(p.relative_to(RAW)) for p in es],
        "license_allowlist": sorted(PERMISSIVE_LICENSES),
        "val_per_10k": VAL_PER_10K,
        "near_dedup": near_meta,
        "tokenizer": {"file": Path(tok_path).name, "sha256": sha256(Path(tok_path)),
                      "vocab_size": tok.vocab_size, "eot_id": tok.eot_id},
        "dtype": "uint16",
        "limit_docs": args.limit_docs,
        "bins": bins,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
