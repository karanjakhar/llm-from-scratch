"""Stage 02, step 1c — fetch the small datasets later stages read, at PINNED revisions.

    uv run python stages/02_data/download_extras.py

download.py fetches the pretraining corpus. These four were fetched by hand
during the project and only recorded in docs/datasets.md, so a reader starting
from a clean clone had no command for them (found while writing the book's
reproduction appendix, journal 2026-11-14). The revisions below are the ones
Hugging Face's download cache recorded for the files the results were made
with, so this reproduces those exact bytes.

    HumanEval      Stages 16-24 eval            164 problems         ~120 KB
    MBPP (full)    Stages 16-24 eval, RM/DPO     974 problems         ~290 KB
    CodeSearchNet  Stage 16 SFT pairs           412k pairs (python)  ~555 MB
    TinyStories    Stage 08 smoke test          2 of 4 train shards  ~484 MB

Idempotent: files already present are skipped.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

EXTRAS = [
    dict(repo="openai/openai_humaneval", revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
         dir="humaneval", files=["openai_humaneval/test-00000-of-00001.parquet"]),
    dict(repo="google-research-datasets/mbpp", revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
         dir="mbpp", files=[f"full/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test", "prompt")]),
    dict(repo="code-search-net/code_search_net", revision="bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
         dir="codesearchnet", files=[f"python/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test")]),
    # Only two of the four train shards: the 10M smoke test needs ~0.2B tokens, not all 2.1M stories.
    dict(repo="roneneldan/TinyStories", revision="f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
         dir="tinystories", files=["data/train-00000-of-00004-2d5a1467fff1081b.parquet",
                                   "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
                                   "data/validation-00000-of-00001-869c898b519ad725.parquet"]),
]


def main():
    for spec in EXTRAS:
        local = RAW / spec["dir"]
        for fn in spec["files"]:
            if (local / fn).exists():
                print(f"  have {spec['repo']}/{fn}")
                continue
            print(f"  get  {spec['repo']}/{fn}", flush=True)
            hf_hub_download(spec["repo"], fn, repo_type="dataset", revision=spec["revision"], local_dir=local)


if __name__ == "__main__":
    main()
