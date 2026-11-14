# Appendix C: reproducing everything

Every number in this book comes from a command in the companion repository.
This appendix lists those commands in the order they depend on each other,
with what each one costs on the machine the book was written on: an RTX 4060
Laptop GPU (8 GB), Python 3.12, PyTorch 2.11 with CUDA 12.8.

The three longest runs take about **30 hours of GPU time** between them: about
20 for the 92M capstone pretraining, about 7 for its post-training and
evaluation chain, and 2 h 18 min for the 30M workhorse. The other experiments
run from seconds to an hour or two each. The pretraining runs can be stopped
and resumed by running the same command again.

## C.1 What you need

| Resource | Amount | Where it goes |
|---|---|---|
| NVIDIA GPU | 8 GB or more, CUDA 12.8 driver | everything trains on it |
| Disk | about 25 GB | Python environment 7.1 GB, raw data 5.8 GB, tokenized data 5.0 GB, checkpoints 5.0 GB |
| Network | about 12 GB of downloads | the Python environment (~6 GB) and the datasets (~6 GB) |
| Tools | `git`, [`uv`](https://docs.astral.sh/uv/) | nothing else is installed by hand |

A smaller GPU runs the early chapters (stages 00-08 train models of 10M
parameters or less). The 92M capstone needs the full 8 GB, with the memory
tricks from chapter 11.

## C.2 The environment

```bash
git clone https://github.com/karanjakhar/llm-from-scratch
cd llm-from-scratch
uv sync --extra dev                                # ~7 GB, mostly CUDA libraries
uv run python stages/00_setup/verify_env.py        # GPU, memory, bf16, the 16-bytes-per-parameter check
uv run pytest -q                                   # 295 tests, about 50 s
```

`pyproject.toml` and `uv.lock` pin every package, and point PyTorch at the
CUDA 12.8 wheel index so `uv` cannot quietly install a CPU-only build. Always
run code through `uv run`. During the project a separately managed Python
environment disappeared between sessions, and the project kept working only
because these two files define the environment (journal, 2026-10-25).

## C.3 The data

Every dataset is downloaded from Hugging Face at a pinned revision, so you get
the same bytes the book's numbers were made from.

```bash
uv run python stages/02_data/download.py           # 12 codeparrot shards + 1 fineweb-edu file, ~4.9 GB
uv run python stages/02_data/download_extras.py    # HumanEval, MBPP, CodeSearchNet, TinyStories, ~1 GB
uv run python stages/02_data/dedup.py              # near-duplicate removal, ~2-3 min
uv run python stages/01_tokenizer/train_production.py   # the production tokenizer, ~1 h
uv run python stages/02_data/prepare.py --tokenizer data/tokenizers/split_8192.json   # ~5 min
uv run python stages/08_training/prepare_tinystories.py # the smoke-test data
```

What comes out:

| File | Contents |
|---|---|
| `data/tokenizers/split_8192.json` | 8,192 learned tokens + 4 specials, digits split one per token |
| `data/bins/split_8192/train_code.bin` | 1,330M tokens of permissive-licence Python, 521,423 files |
| `data/bins/split_8192/train_en.bin` | 1,017M tokens of English, 720,855 documents |
| `data/bins/split_8192/val_*.bin` | 4.9M code / 5.2M English tokens from held-out repositories |
| `data/bins/split_8192/manifest.json` | sources, revisions, counts: enough to rebuild the bins |

The tokenizer must be trained before `prepare.py`, and every checkpoint is
tied to it. Retraining it with a different setting silently invalidates every
model trained afterwards.

## C.4 The stages, in order

Each stage has the same three files: `<topic>.py` (the implementation),
`test_<topic>.py` (its acceptance tests) and `run.py` (the experiments). Each
`run.py` lists its experiments, and what they cost, in its docstring. Numbers
go to `results/NN_*/` as JSON, figures to `docs/visuals/`. The `results/`
files from the book's runs are committed, so you can compare yours against
them line by line.

**Foundations (stages 01-08), each minutes to about an hour:**

```bash
uv run python stages/01_tokenizer/run.py --vocab 8192
uv run python stages/02_data/run.py leakage
uv run python stages/03_embeddings/run.py ablation
uv run python stages/04_attention/run.py ablation  &&  uv run python stages/04_attention/run.py maps
uv run python stages/05_block/run.py depth  &&  uv run python stages/05_block/run.py sweep
uv run python stages/06_gpt/run.py oracle          # downloads OpenAI GPT-2 and matches Hugging Face's outputs
uv run python stages/06_gpt/run.py tie             # also saves the 4-layer model chapter 17 uses as a draft
uv run python stages/07_generation/run.py speed
uv run python stages/08_training/run.py tinystories # 10M smoke test, ~20 min
```

**Efficiency and scaling (stages 09-10):**

```bash
uv run python stages/09_efficiency/bench.py stack  # 14.3k -> 73.7k tokens/s, one optimisation at a time
uv run python stages/10_scaling/run.py isoflop --budget 3e15
uv run python stages/10_scaling/run.py isoflop --budget 1e16
uv run python stages/10_scaling/run.py fit
```

**The pretraining runs (stage 11):**

```bash
uv run python stages/11_pretrain/run.py workhorse  # 30M parameters, 590M tokens, 2 h 18 min
uv run python stages/11_pretrain/run.py capstone   # 92M parameters, 1.80B tokens, ~20 h
```

Both resume from `checkpoints/11_pretrain/<name>/ckpt.pt` if stopped. Expect a
final validation code loss of 1.554 (workhorse) and 1.295 (capstone).

**Architecture experiments (stages 12-15)** train their own small models from
the same data, except stage 15, which needs the stage 11 and stage 06
checkpoints:

```bash
uv run python stages/12_modern_arch/run.py ladder
uv run python stages/13_moe/run.py train
uv run python stages/14_attention_variants/run.py mla
uv run python stages/15_speculative/run.py sweep
```

**Post-training on the workhorse (stages 16-21, 23).** Each step reads the
checkpoint the previous one wrote:

```bash
W=checkpoints/11_pretrain/workhorse/final.pt
uv run python stages/16_sft/run.py --base $W --tag workhorse
uv run python stages/17_lora/run.py --base $W --tag workhorse
uv run python stages/18_reward/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse
uv run python stages/19_dpo/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse --beta 0.1
uv run python stages/20_grpo/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse
uv run python stages/21_distill_quant/run.py quant --model $W --tag workhorse
uv run python stages/23_serving/run.py bursty --model $W
```

**The capstone chain (stages 16-24 on the 92M model), about 7 hours:**

```bash
bash stages/24_capstone/run_post.sh                # SFT, reward model, DPO, GRPO, quantize, serve, evaluate
```

It stops at the first failure, and runs with activation checkpointing because
the 92M model does not fit SFT's batch otherwise (chapter 26).

**Evaluation (stage 22)** scores every checkpoint above with one harness:

```bash
uv run python stages/22_evaluation/run.py table --pass10
uv run python stages/22_evaluation/run.py contamination
uv run python stages/22_evaluation/run.py mc
```

**A caution about evaluation.** HumanEval, MBPP and GRPO's reward all run code
the model wrote. Each program runs in a separate Python process in a temporary
directory, with a time limit and a memory cap (`llmfs/eval/execution.py`).
That stops runaway loops and memory blow-ups. It is not a security sandbox: a
program could still read or delete files your user can. A 92M model is very
unlikely to write anything harmful, but if you change the prompts or use a
larger model, run evaluation inside a container or a throwaway VM.

## C.5 What will differ on your machine

- **Speed.** Tokens per second, and so every wall-clock time above, depend on
  the GPU. The losses do not.
- **The last digit.** bf16 arithmetic and some CUDA kernels are not
  bit-for-bit deterministic. Expect losses within about 0.01 of the book's.
- **Single problems.** One HumanEval problem is 0.61 percentage points, and
  the same checkpoint scored 0 and 1 of 164 in two evaluations with the same
  settings (chapter 24). Differences of one or two problems are noise.
- **Dataset revisions** cannot drift: every download names its commit hash.
  If a dataset is removed from Hugging Face upstream, the hash will fail
  loudly rather than fetch something different.
