# Model card: llmfs-capstone-92M

A small Python-code language model built from scratch, as the end point of
the curriculum in this repository. It exists to be understood, not used.
Every number below comes from a results file in `results/`, named next to it.

## What it is

| | |
|---|---|
| Architecture | GPT-2-style decoder: 12 layers, width 768, 12 heads, learned positions, pre-norm LayerNorm, GELU MLP, tied embeddings, **plus QK-norm** (Stage 10 §8f) |
| Parameters | 92.1M (85.1M non-embedding) |
| Context | 1,024 tokens |
| Tokenizer | our byte-level BPE, 8,192 merges + 4 specials, every digit split (`data/tokenizers/split_8192.json`) |
| Pretraining data | 85% permissively-licensed Python (codeparrot-clean, deduplicated) / 15% FineWeb-Edu English |
| Pretraining tokens | 1,802,240,000 (27,500 steps × 65,536), about 20 per parameter; ~20 GPU-hours, resumed once across a 5-day pause (`results/11_pretrain/capstone.json`) |
| Hardware | one RTX 4060 Laptop GPU, 8 GB |
| Variants | base; SFT (docstring → function, CodeSearchNet); DPO (correct vs mutated code); GRPO (unit-test reward on an easy function-writing curriculum) |

## Results

All from `results/24_capstone/pipeline.json`, which copies them from the
stage result files listed in its `sources`.

**Language modelling** (base model). Validation loss on held-out code
**1.295** nats per token, English **2.898**, on Stage
11's validation set. Stage 22's harness, which uses 512-token windows,
gives 1.423 and 2.950 (0.66 bits per byte of code); the rows below use that harness.

**Code benchmarks** (Stage 22 harness; greedy pass@1, pass@10 at τ = 0.8 from
10 samples):

| variant | val code | HumanEval pass@1 | pass@10 | MBPP pass@1 | pass@10 |
|---|---|---|---|---|---|
| base (prompt continued as code) | 1.423 | **3.05%** (5/164) | **5.49%** | 0.2% | 0.4% |
| SFT (chat) | 1.582 | 1.22% (2/164) | 4.88% | 0.4% | 0.4% |
| DPO, β = 1 (chat) | 1.632 | 1.22% | 1.83% | 0.4% | 0.6% |
| GRPO (chat) | 1.582 | **3.05%** | 4.88% | 0.4% | 0.6% |

For reference, published codegen-350M-mono scores 12.8% HumanEval pass@1.

**Other measurements**

| | |
|---|---|
| Picking the working program among 4 (352 MBPP items, chance 25%) | base 76.7%, DPO 77.8% (per-token scoring) |
| Reward model on test-verified MBPP pairs | 86.5% |
| GRPO curriculum, held-out | 27.3% → 73.3% in 60 steps |
| Quantization damage (val code) | int8 per-channel +0.000; NF4 +0.013 (176 → 57 MB); int4 per-tensor +3.22 (broken) |
| Serving, bursty trace, 16 slots | 720 tok/s; time to first token p50 8 ms (continuous batching) vs 440 ms (static) |

## Intended use

Reading, running and breaking, to learn how a language model is built. It is
a teaching artifact with an objective score, not a coding assistant.

## Limitations, stated plainly

- At 92M parameters and 1.8B tokens it has **no world knowledge** to speak of,
  no multi-step reasoning, and no general instruction following.
- It writes Python that is often syntactically plausible and usually wrong.
  It solves 5 of 164 HumanEval problems, each a one- to four-line body, and
  2 of 500 MBPP problems. Greedy decoding falls into repetition loops.
- Its English is fluent at the sentence level and unreliable beyond it.
- It has seen no safety training. It will continue any prompt.
- Its training code comes from public GitHub repositories (permissive
  licences only); it can reproduce fragments of that code.

## How it was evaluated

HumanEval (164) and MBPP test (500) with our own harness, verified on the
reference solutions (164/164 pass), greedy pass@1 and sampled pass@10
(unbiased estimator), sandboxed execution with a sentinel against early-exit
hacks. A 13-gram contamination scan of all 1.33B training code tokens found no
benchmark problem with ≥ 50% of its 13-grams in training, and **no 13-gram at
all** from any problem the model solves (`results/22_evaluation/contamination.json`).
Post-training made HumanEval worse before GRPO restored it: compare variants
in their own best prompt format (`docs/22_evaluation.md` §8a).

## Reproducing

`README.md` → "Running the stages". The pretraining run is resumable:
`uv run python stages/11_pretrain/run.py capstone`.
