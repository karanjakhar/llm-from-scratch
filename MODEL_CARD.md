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
| Pretraining tokens | *(filled from results/11_pretrain/capstone.json)* |
| Hardware | one RTX 4060 Laptop GPU, 8 GB |
| Variants | base; SFT (docstring → function, CodeSearchNet); DPO (correct vs mutated code); GRPO (MBPP unit tests) |

## Results

*(filled from results/24_capstone/pipeline.json)*

## Intended use

Reading, running and breaking, to learn how a language model is built. It is
a teaching artifact with an objective score, not a coding assistant.

## Limitations, stated plainly

- At ~92M parameters and ~2B tokens it has **no world knowledge** to speak of,
  no multi-step reasoning, and no general instruction following.
- It writes Python that is often syntactically plausible and usually wrong.
  Its HumanEval and MBPP pass rates are in the low single digits at best.
- Its English is fluent at the sentence level and unreliable beyond it.
- It has seen no safety training. It will continue any prompt.
- Its training code comes from public GitHub repositories (permissive
  licences only); it can reproduce fragments of that code.

## How it was evaluated

HumanEval (164) and MBPP test (500) with our own harness, verified on the
reference solutions (164/164 pass), greedy pass@1 and sampled pass@10
(unbiased estimator), sandboxed execution with a sentinel against early-exit
hacks. A 13-gram contamination scan of the training data found no benchmark
problem with ≥50% of its 13-grams in training (`results/22_evaluation/contamination.json`).

## Reproducing

`README.md` → "Running the stages". The pretraining run is resumable:
`uv run python stages/11_pretrain/run.py capstone`.
