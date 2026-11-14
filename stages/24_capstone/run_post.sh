#!/bin/bash
# The capstone post-training / eval chain (SFT onward) and Stage 24 assembly.
#   nohup bash stages/24_capstone/run_post.sh > capstone_post.log 2>&1 &
# set -e: the first version of this chain ran on after SFT ran out of memory,
# and every later step then had no checkpoint to load (journal, Session 10).
# --act-ckpt: activation checkpointing; the 92M model OOMs at B=16 without it.
cd "$(dirname "$0")/../.."
set -ex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=checkpoints/11_pretrain/capstone/final.pt
S=checkpoints/16_sft/capstone.pt
uv run python stages/16_sft/run.py --base $C --tag capstone --act-ckpt
uv run python stages/18_reward/run.py --sft $S --tag capstone --act-ckpt
uv run python stages/19_dpo/run.py --sft $S --tag capstone --beta 1.0 --save --act-ckpt
uv run python stages/20_grpo/run.py --sft $S --tag capstone --tasks easy --steps 60 --max-tries 16 --rollout-temp 0.8 --eval-every 15 --save --act-ckpt
uv run python stages/21_distill_quant/run.py quant --model $C --tag capstone
uv run python stages/21_distill_quant/run.py plot --tag capstone
uv run python stages/23_serving/run.py bursty --model $C --tag capstone
uv run python stages/22_evaluation/run.py table --pass10
uv run python stages/22_evaluation/run.py mc
uv run python stages/22_evaluation/run.py contamination
uv run python stages/24_capstone/assemble.py
echo CHAIN5_DONE
