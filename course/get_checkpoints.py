"""Download the book's trained models, so later modules need not wait hours for pretraining.

    python course/get_checkpoints.py                 # workhorse base + its SFT model
    python course/get_checkpoints.py --capstone      # also the 92M capstone base (370 MB)

docs/COURSE.md §5. Files land exactly where the stage scripts look for them
(checkpoints/...), so a module that says "the workhorse checkpoint" works the
same whether you trained it or downloaded it.

The Hub repository is created when the course is published; until then this
script stops with a message instead of guessing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HUB_REPO = "karanjakhar/llm-from-scratch"        # model repo on huggingface.co
REVISION = None                                  # pinned to a commit hash once published

FILES = {   # path in the Hub repo -> path in this repo
    "workhorse/final.pt": "checkpoints/11_pretrain/workhorse/final.pt",       # 30M base, 120 MB
    "workhorse_sft/final.pt": "checkpoints/16_sft/workhorse.pt",              # 30M after module 11's SFT
}
CAPSTONE = {"capstone/final.pt": "checkpoints/11_pretrain/capstone/final.pt"}  # 92M base, 369 MB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capstone", action="store_true")
    a = ap.parse_args()
    if REVISION is None:
        print(f"The checkpoints are not published yet ({HUB_REPO}). Train the workhorse yourself "
              "(module 9) or check the course page for news.")
        return 1
    from huggingface_hub import hf_hub_download
    for src, dst in {**FILES, **(CAPSTONE if a.capstone else {})}.items():
        out = ROOT / dst
        if out.exists():
            print(f"  have {dst}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"  get  {src}", flush=True)
        shutil.copy2(hf_hub_download(HUB_REPO, src, revision=REVISION), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
