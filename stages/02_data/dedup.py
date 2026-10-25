"""Stage 02, step 1b — near-duplicate dedup of the code corpus (MinHash + LSH).

    uv run python stages/02_data/dedup.py

Built because the leakage ablation (run.py leakage) measured that, even with a
repository-level split, 34% of rare val lines and the majority of lines in 43%
of val files appear verbatim in train: forks and vendored code under other
repo names. Exact dedup cannot see those. docs/02_data.md §4.

Writes data/dedup/near_dups.json (uids to drop; prepare.py reads it) and
results/02_data/near_dedup.json (stats, committed).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llmfs.data import dedup as D  # noqa: E402
from llmfs.data.corpus import code_shards, iter_code_docs, split_of  # noqa: E402

OUT = REPO / "data" / "dedup" / "near_dups.json"
RESULTS = REPO / "results" / "02_data" / "near_dedup.json"

_PERMS = None


def _init():
    global _PERMS
    _PERMS = D.make_perms()


def _sigs(texts: list[str]) -> list[np.ndarray | None]:
    a, b = _PERMS
    out = []
    for t in texts:
        sh = D.shingles(t)
        out.append(D.signature(sh, a, b) if len(sh) else None)
    return out


def _batches(docs, n=256):
    it = iter(docs)
    while batch := list(islice(it, n)):
        yield batch


def main():
    shards = code_shards(REPO / "data" / "raw")
    print(f"near-dedup over {len(shards)} shards (permissive licenses, both splits)")

    uids, is_val, sigs = [], [], []
    t0 = time.perf_counter()

    def texts():
        for batch in _batches(iter_code_docs(shards)):
            for d in batch:
                uids.append(d.uid)
                is_val.append(split_of(d.key) == "val")
            yield [d.text for d in batch]

    with mp.get_context("spawn").Pool(18, _init) as pool:
        for out in pool.imap(_sigs, texts(), chunksize=2):
            sigs.extend(out)
            if len(sigs) % 51200 < 256:
                print(f"  {len(sigs):,} signatures, {time.perf_counter()-t0:.0f}s", flush=True)
    t_sig = time.perf_counter() - t0
    print(f"  {len(sigs):,} docs signed in {t_sig:.0f}s "
          f"({sum(s is None for s in sigs):,} too short to shingle, never dropped)")

    t1 = time.perf_counter()
    uf, cst = D.cluster(sigs)
    drop, dst = D.choose_drops(uf, is_val)
    t_clu = time.perf_counter() - t1

    n_val = sum(is_val)
    n_train = len(is_val) - n_val
    stats = {
        "params": {"ngram": D.NGRAM, "num_perm": D.NUM_PERM, "bands": D.BANDS,
                   "rows": D.ROWS, "threshold": D.THRESHOLD},
        "docs": len(sigs), "train_docs": n_train, "val_docs": n_val,
        **cst, **dst,
        "train_drop_rate": dst["train_dropped"] / max(1, n_train),
        "val_drop_rate": dst["val_dropped"] / max(1, n_val),
        "seconds_signatures": round(t_sig, 1), "seconds_clustering": round(t_clu, 1),
    }
    print()
    print(f"  candidates {cst['candidates']:,}, verified >= {D.THRESHOLD}: {cst['verified']:,}")
    print(f"  clusters containing duplicates: {dst['clusters_with_dups']:,}")
    print(f"  train: {dst['train_dropped']:,} / {n_train:,} dropped "
          f"({stats['train_drop_rate']:.1%})")
    print(f"  val:   {dst['val_dropped']:,} / {n_val:,} dropped ({stats['val_drop_rate']:.1%}), "
          f"of which {dst['val_dropped_due_to_train_copy']:,} had a near-copy in TRAIN")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"params": stats["params"],
                               "drop": sorted(uids[i] for i in drop)}))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(stats, indent=1))
    print(f"\n  wrote {OUT} and {RESULTS}")


if __name__ == "__main__":
    main()
