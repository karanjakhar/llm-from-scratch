"""Stage 02 experiments. docs/02_data.md §5 (leakage ablation) and §11 (throughput).

    uv run python stages/02_data/run.py leakage              # CPU, ~minutes, no tokenizer needed
    uv run python stages/02_data/run.py throughput --bins data/bins/<name>
    uv run python stages/02_data/run.py summary   --bins data/bins/<name>

Results are written to results/02_data/*.json so they are committed alongside
the code that produced them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from llmfs.data.corpus import VAL_PER_10K, code_shards, iter_code_docs, split_of  # noqa: E402

RESULTS = REPO / "results" / "02_data"
MIN_LINE = 20          # shorter lines ("    pass", "else:") are noise, not evidence
RARE = 5               # a line in <= 5 files corpus-wide is not boilerplate


def file_level_split(repo: str, path: str) -> str:
    """The WRONG split, for comparison: same val fraction, decided per file."""
    h = int.from_bytes(hashlib.blake2b(f"{repo}/{path}".encode(), digest_size=8).digest(), "little")
    return "val" if (h % 10_000) < VAL_PER_10K else "train"


def lines_of(text: str) -> set[int]:
    # hash() is salted per process but consistent within this run, which is
    # all a membership test needs -- and it is ~10x faster than blake2b.
    return {hash(s) for ln in text.splitlines() if len(s := ln.strip()) >= MIN_LINE}


def leakage(args):
    import gzip
    import json as _json
    shards = code_shards(REPO / "data" / "raw")[: args.shards]
    drop: set[str] = set()
    if args.drop:
        drop = set(json.loads(Path(args.drop).read_text())["drop"])
    print(f"leakage ablation on {len(shards)} shard(s), permissive files only"
          + (f", excluding {len(drop):,} near-dup uids" if drop else ""))

    # Pass 1: for every line, in how many files does it occur (globally)?
    # Record per-file line sets for files that land in val under EITHER split.
    t0 = time.perf_counter()
    global_count: Counter = Counter()
    val_files = {"file": [], "repo": []}       # list of line-sets
    n_files = 0
    for path in shards:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for lineno, raw in enumerate(f):
                r = _json.loads(raw)
                if r["license"] not in {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause",
                                        "unlicense", "isc", "cc0-1.0"}:
                    continue
                if f"{path.name}:{lineno}" in drop:
                    continue
                ls = lines_of(r["content"])
                global_count.update(ls)
                n_files += 1
                if file_level_split(r["repo_name"], r["path"]) == "val":
                    val_files["file"].append(ls)
                if split_of(r["repo_name"]) == "val":
                    val_files["repo"].append(ls)
    print(f"  {n_files:,} files, {len(global_count):,} distinct lines, "
          f"{time.perf_counter()-t0:.0f}s")

    # A val line is 'in train' iff it occurs in more files than the val side
    # accounts for: global_count - val_count > 0. One global Counter serves
    # both splits, instead of two near-identical train sets in RAM.
    out = {}
    for split, files in val_files.items():
        val_count = Counter()
        for ls in files:
            val_count.update(ls)
        tot = leaked = rare_tot = rare_leaked = 0
        per_file = []
        for ls in files:
            fl = 0
            for h in ls:
                in_train = global_count[h] - val_count[h] > 0
                tot += 1
                leaked += in_train
                fl += in_train
                if global_count[h] <= RARE:
                    rare_tot += 1
                    rare_leaked += in_train
            if ls:
                per_file.append(fl / len(ls))
        mostly = sum(p > 0.5 for p in per_file) / max(1, len(per_file))
        out[split] = {
            "val_files": len(files),
            "line_leak_rate": leaked / max(1, tot),
            "rare_line_leak_rate": rare_leaked / max(1, rare_tot),
            "val_files_over_half_leaked": mostly,
        }

    print()
    print(f"  {'split':<6} {'val files':>9}  {'lines in train':>14}  "
          f"{'RARE lines in train':>19}  {'files >50% leaked':>17}")
    for k, v in out.items():
        print(f"  {k:<6} {v['val_files']:>9,}  {v['line_leak_rate']:>13.1%}  "
              f"{v['rare_line_leak_rate']:>18.1%}  {v['val_files_over_half_leaked']:>16.1%}")
    f, r = out["file"], out["repo"]
    print()
    print(f"  all-lines gap (file - repo):  {(f['line_leak_rate'] - r['line_leak_rate'])*100:+.1f} pts")
    print(f"  rare-lines gap (file - repo): {(f['rare_line_leak_rate'] - r['rare_line_leak_rate'])*100:+.1f} pts")
    print()
    print("  Reading it: the all-lines rate is mostly boilerplate (license headers,")
    print("  common imports) that is in train under ANY split. The rare-lines rate")
    print("  is the real signal -- a line seen in <=5 files that is on both sides is")
    print("  a copy. Whatever remains under the repo split is cross-repo copying")
    print("  (forks, vendoring), which only near-duplicate dedup can remove.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"shards": [p.name for p in shards], "files": n_files, "near_dedup": bool(drop),
               "min_line_chars": MIN_LINE, "rare_threshold": RARE, "results": out}
    name = "leakage_after_near_dedup.json" if drop else "leakage.json"
    (RESULTS / name).write_text(json.dumps(payload, indent=1))
    print(f"\n  saved {RESULTS / name}")


def throughput(args):
    import torch

    from llmfs.data import MixedLoader
    bins = Path(args.bins)
    srcs = {"code": (bins / "train_code.bin", 0.85), "en": (bins / "train_en.bin", 0.15)}
    B, T = args.B, args.T
    need = 80_000          # upper end of the workhorse's tokens/s (docs/domain_choice.md)
    rows = []
    for device, pin in (("cpu", False), ("cuda", False), ("cuda", True)):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        ld = MixedLoader(srcs, B=B, T=T, device=device, pin=pin, seed=0)
        for _ in range(20):
            ld.next_batch()                          # warm the page cache and CUDA
        if device == "cuda":
            torch.cuda.synchronize()
        n, t0 = 200, time.perf_counter()
        for _ in range(n):
            x, y = ld.next_batch()
        if device == "cuda":
            torch.cuda.synchronize()
        tps = n * B * T / (time.perf_counter() - t0)
        rows.append({"device": device, "pinned": pin, "tokens_per_s": tps,
                     "x_model_need": tps / need})
        print(f"  {device:<5} pinned={str(pin):<5}  {tps/1e6:6.2f}M tokens/s  "
              f"= {tps/need:5.0f}x the workhorse's ~{need//1000}k tok/s")
    print("\n  If every row is >> 10x, the loader is not a bottleneck and a multi-worker")
    print("  DataLoader would be complexity with no payoff. docs/02_data.md §11.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "throughput.json").write_text(json.dumps({"B": B, "T": T, "rows": rows}, indent=1))


def summary(args):
    m = json.loads((Path(args.bins) / "manifest.json").read_text())
    print(f"tokenizer {m['tokenizer']['file']}  vocab_size {m['tokenizer']['vocab_size']}")
    tot = 0
    print(f"  {'bin':<11} {'docs':>9}  {'tokens':>10}  {'text MB':>9}  {'B/tok':>6}  "
          f"{'near-dups':>9}  {'exact dups':>10}")
    for name, b in m["bins"].items():
        tot += b["tokens"]
        print(f"  {name:<11} {b['docs']:>9,}  {b['tokens']/1e6:>9,.1f}M  "
              f"{b['bytes']/1e6:>9,.0f}  {b['bytes_per_token']:>6.3f}  "
              f"{b.get('near_dup_docs_dropped', 0):>9,}  {b['dup_docs_dropped']:>10,}")
    print(f"  total {tot/1e6:,.1f}M tokens = {tot*2/1e6:,.0f} MB on disk")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leakage")
    p.add_argument("--shards", type=int, default=2)
    p.add_argument("--drop", type=str, default=None,
                   help="near-dup drop list; measures leakage AFTER near-dedup")
    p = sub.add_parser("throughput")
    p.add_argument("--bins", required=True)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=1024)
    p = sub.add_parser("summary")
    p.add_argument("--bins", required=True)
    args = ap.parse_args()
    {"leakage": leakage, "throughput": throughput, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
