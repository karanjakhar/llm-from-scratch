# Progress

Live state of the project. Update this at the end of every working session.
A new AI session reads `CLAUDE.md` then this file and knows exactly where to pick up.

## Current position

**Stage 01 — Tokenization** · code complete, full-scale run pending
**Next action:** read `docs/01_tokenizer.md` end to end, then run the vocab-8192
experiment and the regex ablation:

```bash
python3 stages/01_tokenizer/run.py --vocab 8192      # several minutes
```

Record its compression numbers below. Then Stage 02 (data pipeline).

## Stage status

| # | Stage | Status | Notes / result |
|---|---|---|---|
| 00 | Setup & mental model | `[x]` | env verified, VRAM model holds to 8.5% |
| 01 | Tokenization | `[~]` | BasicBPE + RegexBPE done, 57 tests pass; 8192 run pending |
| 02 | Data pipeline | `[ ]` | |
| 03 | Embeddings & position | `[ ]` | |
| 04 | Attention | `[ ]` | |
| 05 | Transformer block | `[ ]` | |
| 06 | GPT assembly | `[ ]` | |
| 07 | Generation & decoding | `[ ]` | |
| 08 | Training loop | `[ ]` | |
| 09 | Efficiency | `[ ]` | |
| 10 | Scaling laws | `[ ]` | |
| 11 | Pretraining run | `[ ]` | |
| 12 | Modern architecture | `[ ]` | |
| 13 | Mixture of Experts | `[ ]` | |
| 14 | Attention variants | `[ ]` | |
| 15 | Speculative decoding | `[ ]` | optional |
| 16 | SFT | `[ ]` | |
| 17 | LoRA / QLoRA | `[ ]` | |
| 18 | Reward modeling | `[ ]` | |
| 19 | DPO / RLHF | `[ ]` | |
| 20 | Reasoning / GRPO | `[ ]` | |
| 21 | Distill & quantize | `[ ]` | |
| 22 | Evaluation | `[ ]` | |
| 23 | Serving | `[ ]` | |
| 24 | Capstone | `[ ]` | |

## Results log

Numbers only. Every experiment that produced a measurement goes here so trends
stay visible across months.

| Date | Stage | Experiment | Metric | Value |
|---|---|---|---|---|
| 2026-10-14 | 00 | 12.6M-param toy model, fp32 + AdamW | predicted vs measured fixed VRAM | 192.0 MB vs 208.3 MB (**8.5% error**) |
| 2026-10-14 | 00 | same | peak allocated incl. activations | 256.3 MB |
| 2026-10-14 | 01 | RegexBPE on tinyshakespeare (1.0M chars train) | held-out bytes/token @ vocab 512 | 1.878 |
| 2026-10-14 | 01 | same | held-out bytes/token @ vocab 1024 | 2.257 (+20.2% over 512) |
| 2026-10-14 | 01 | our encoder vs tiktoken, 100k chars | encode throughput | 0.53 MB/s vs 14.4 MB/s (**27x slower**) |
| 2026-10-14 | 01 | gpt2 / cl100k / o200k on same held-out text | bytes/token | 3.107 / 3.557 / 3.614 |
| 2026-10-14 | 01 | weighted-dedup training vs naive | merge lists identical, suite runtime | 8.8s → 3.25s |

## Open questions

Things noticed but not yet understood. Carried forward until answered.

- Our encoder is ~27x slower than tiktoken. How much of that is the O(n·m)
  rescan in `_encode_chunk` vs Python interpreter overhead? Worth implementing
  the linked-list + heap version once, to find out.
- At vocab 1024 on Shakespeare, `héllo` and `你好` tokenize to raw bytes with no
  merges at all. The corpus is pure ASCII, so no multi-byte merges were ever
  learned. Open: how much multilingual data does a corpus need before the
  tokenizer stops penalizing non-English text?
- Compression gain from 512→1024 was +20%. Does the curve flatten the way the
  reference numbers in docs §6 suggest? The vocab-8192 run answers this.

## Session log

### 2026-10-14 — Session 1
- Repo initialized. Scaffold, `README.md`, `CURRICULUM.md` (25 stages),
  `CLAUDE.md`, `GLOSSARY.md`, `PROGRESS.md`.
- **Stage 00 complete.** `verify_env.py` runs; the 16-bytes-per-parameter
  training-memory model predicted within 8.5% of what the allocator reported.
  Found that the 4060 exposes 7797 MB, not 8192 — docs corrected.
- **Stage 01 code complete.** `BasicBPE` (no pre-tokenization) and `RegexBPE`
  (GPT-2 pattern), special tokens with a non-forgeable default, JSON
  save/load, compression metric. 57 tests pass. `run.py` produces the
  compression table, the regex ablation, the tiktoken comparison, and a
  learned-token inspection.
- Three things learned while building, now folded into the docs and tests:
  1. A requested vocab size is an **upper bound**. BPE stops early when no
     mergeable pairs remain, so the embedding table must be sized from the
     tokenizer that came out — not the config. Silent-dead-parameter bug
     otherwise.
  2. Training on **unique chunks weighted by frequency** gives provably
     identical merges to the naive per-occurrence version, far faster.
     Asserted by `test_weighted_dedup_training_equals_naive`.
  3. All-whitespace tokens legitimately contain interior spaces (the `\s+`
     branch of the pattern), so the "no boundary-spanning tokens" check has to
     exclude them. First version of that test was wrong.
- Decisions recorded in `CLAUDE.md` → Decisions, including complete-working-code
  over exercise skeletons.
- Next: vocab-8192 run + ablation, then Stage 02 (data pipeline).
