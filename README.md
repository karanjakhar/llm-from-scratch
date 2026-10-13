# Building an LLM From Scratch

A from-first-principles build of a modern language model. Every component is
implemented by hand in plain PyTorch before any library shortcut is allowed.
Dense GPT first, then Mixture-of-Experts, then pretraining, then the whole
post-training stack (SFT, LoRA, reward models, DPO/GRPO, distillation,
quantization), then evaluation and serving.

The goal is **understanding**, not a leaderboard score. Each stage answers
"why does this exist and what breaks without it?" before it answers "how do I
code it?"

## How this repo is organized

| Path | What lives there |
|---|---|
| `CURRICULUM.md` | The full 25-stage plan. The map. Read this first. |
| `PROGRESS.md` | What is done, what is next, session-by-session log. |
| `CLAUDE.md` | Instructions so a fresh AI session can resume with zero context. |
| `GLOSSARY.md` | Every term defined once, in plain language. |
| `docs/NN_*.md` | The lesson for stage NN: theory, math, intuition, pitfalls. |
| `stages/NN_*/` | Runnable code + experiments for stage NN. Self-contained. |
| `llmfs/` | The library that accumulates. Stages promote their good code here. |
| `data/` | Datasets and tokenized binaries. Gitignored. |
| `checkpoints/` | Model weights. Gitignored. |

Two layers on purpose: `stages/` is the scratchpad where things are written
the slow, explicit, educational way. `llmfs/` is the clean reusable version
that later stages import. Learning twice is the point.

## Hardware this was built on

- NVIDIA RTX 4060 Laptop, **8 GB VRAM** (7797 MB actually addressable)
- ~98 GB free disk
- Python 3.12, PyTorch 2.10 + CUDA 12.8

8 GB is a real constraint and it is treated as a feature, not a problem. It
forces honest engineering: gradient accumulation, mixed precision, activation
checkpointing, and an understanding of exactly where every byte of memory
goes. Models in the 10M-150M parameter range train locally. One optional
stage rents a cloud GPU for a larger run.

## Quickstart

```bash
uv sync                      # create .venv, install deps
uv run python stages/00_setup/verify_env.py
```

Then open `CURRICULUM.md` and start at Stage 00.

## Status

Early. See `PROGRESS.md` for the live state.

## Running the stages

```bash
# Stage 00 — environment + VRAM budget check
python3 stages/00_setup/verify_env.py

# Stage 01 — tokenizer: tests, then the real experiment
python3 -m pytest stages/01_tokenizer/ -q
python3 stages/01_tokenizer/run.py --vocab 8192          # a few minutes
python3 stages/01_tokenizer/run.py --vocab 2048 --skip-ablation   # fast
```

`uv sync` gives a clean isolated environment; the commands above also work in
any environment that has `torch`, `regex`, `tiktoken`, and `pytest`.
