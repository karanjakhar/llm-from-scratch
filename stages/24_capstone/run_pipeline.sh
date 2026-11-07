#!/bin/bash
# Stage 11 capstone (resumes from checkpoints/11_pretrain/capstone/ckpt.pt), then the
# capstone post-training / eval chain and Stage 24 assembly. Run from the repo root:
#   nohup bash stages/24_capstone/run_pipeline.sh > capstone_pipeline.log 2>&1 &
cd "$(dirname "$0")/../.."
set -x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=checkpoints/11_pretrain/capstone/final.pt
uv run python stages/11_pretrain/run.py capstone
uv run python stages/11_pretrain/run.py samples --which capstone
uv run python stages/11_pretrain/run.py plot
uv run python stages/15_speculative/run.py sweep --target $C --draft checkpoints/11_pretrain/workhorse/final.pt --tag capstone
uv run python stages/15_speculative/run.py plot --tag capstone
uv run python stages/16_sft/run.py --base $C --tag capstone
uv run python stages/18_reward/run.py --sft checkpoints/16_sft/capstone.pt --tag capstone
uv run python stages/19_dpo/run.py --sft checkpoints/16_sft/capstone.pt --tag capstone --beta 1.0 --save
uv run python stages/20_grpo/run.py --sft checkpoints/16_sft/capstone.pt --tag capstone --tasks easy --steps 60 --max-tries 16 --rollout-temp 0.8 --eval-every 15 --save
uv run python stages/21_distill_quant/run.py quant --model $C --tag capstone
uv run python stages/21_distill_quant/run.py plot --tag capstone
uv run python stages/23_serving/run.py bursty --model $C --tag capstone
uv run python stages/22_evaluation/run.py table --pass10
uv run python stages/22_evaluation/run.py mc
uv run python stages/22_evaluation/run.py contamination
uv run python stages/24_capstone/assemble.py
echo CHAIN5_DONE
