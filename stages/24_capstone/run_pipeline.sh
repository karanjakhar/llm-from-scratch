#!/bin/bash
# Stage 11 capstone (resumes from checkpoints/11_pretrain/capstone/ckpt.pt), then the
# capstone post-training / eval chain and Stage 24 assembly. Run from the repo root:
#   nohup bash stages/24_capstone/run_pipeline.sh > capstone_pipeline.log 2>&1 &
cd "$(dirname "$0")/../.."
set -ex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=checkpoints/11_pretrain/capstone/final.pt
uv run python stages/11_pretrain/run.py capstone
uv run python stages/11_pretrain/run.py samples --which capstone
uv run python stages/11_pretrain/run.py plot
uv run python stages/15_speculative/run.py sweep --target $C --draft checkpoints/11_pretrain/workhorse/final.pt --tag capstone
uv run python stages/15_speculative/run.py plot --tag capstone
bash stages/24_capstone/run_post.sh
