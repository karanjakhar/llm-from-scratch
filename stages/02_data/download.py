"""Stage 02, step 1 — fetch the raw corpus at PINNED revisions.

Idempotent: files already present are not re-downloaded. Writes
data/raw/sources.json recording exactly what was fetched, which prepare.py
copies into the manifest. docs/02_data.md §12.

    uv run python stages/02_data/download.py                 # 12 code shards + 1 English
    uv run python stages/02_data/download.py --code-shards 20

Sizes (verified 2026-10-25): each codeparrot shard is ~246 MB gzipped,
100k files, ~1.02 GB of text, ~58% permissive by bytes. The fineweb-edu file
is 2.15 GB, 726k documents, ~0.7B tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

CODE = {
    "repo": "codeparrot/codeparrot-clean",
    "revision": "35a59fb025bc0a102f7d96eac09d145b896d487b",
    "files": lambda n: [f"file-{i:012d}.json.gz" for i in range(1, n + 1)],
    "dir": RAW / "codeparrot",
}
ENGLISH = {
    "repo": "HuggingFaceFW/fineweb-edu",
    "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "files": lambda n: [f"sample/10BT/{i:03d}_00000.parquet" for i in range(n)],
    "dir": RAW / "fineweb",
}


def fetch(spec: dict, n: int) -> list[str]:
    files = spec["files"](n)
    for fn in files:
        dest = spec["dir"] / fn
        if dest.exists():
            print(f"  have {spec['repo']}/{fn}")
            continue
        print(f"  get  {spec['repo']}/{fn}", flush=True)
        hf_hub_download(spec["repo"], fn, repo_type="dataset",
                        revision=spec["revision"], local_dir=spec["dir"])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-shards", type=int, default=12,
                    help="~0.6 GB permissive text each; 12 covers the 125M capstone")
    ap.add_argument("--en-files", type=int, default=1,
                    help="~0.7B tokens each; 1 covers the 15%% English share")
    args = ap.parse_args()

    sources = {}
    for name, spec, n in (("code", CODE, args.code_shards), ("en", ENGLISH, args.en_files)):
        files = fetch(spec, n)
        sources[name] = {"repo": spec["repo"], "revision": spec["revision"], "files": files}

    (RAW / "sources.json").write_text(json.dumps(sources, indent=1))
    print(f"wrote {RAW / 'sources.json'}")


if __name__ == "__main__":
    main()
