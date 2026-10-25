# Progress

Live state of the project. Update this at the end of every working session.
A new AI session reads `CLAUDE.md` then this file and knows exactly where to pick up.

## Current position

**Stage 02 — Data pipeline** · code done, tested, near-dedup run; full tokenization pending
**Stage 01 — production tokenizer** · training on the real corpus (3 digit-rule variants)

**Next actions, in order:**

1. When `stages/01_tokenizer/train_production.py` finishes, read
   `data/tokenizers/results.json` and pick the tokenizer (vocab 8192 vs 16384,
   digit rule grouped / cap3 / split). Record the choice + reason in `CLAUDE.md`.
2. Full tokenization:
   ```bash
   uv run python stages/02_data/prepare.py --tokenizer data/tokenizers/<chosen>.json
   uv run python stages/02_data/run.py summary    --bins data/bins/<chosen>
   uv run python stages/02_data/run.py throughput --bins data/bins/<chosen>
   ```
3. Check the token budget. Near-dedup removed 28% of train, so 12 shards may
   now fall short of the ~2.1B code tokens the 125M capstone wants.
   `download.py --code-shards N` + re-run `dedup.py` if so.
4. Then Stage 03.

## Stage status

| # | Stage | Status | Notes / result |
|---|---|---|---|
| 00 | Setup & mental model | `[x]` | env verified, VRAM model holds to 8.5% |
| 01 | Tokenization | `[~]` | algorithm done (58 tests); production tokenizer training on real corpus |
| 02 | Data pipeline | `[~]` | pipeline + near-dedup done (34 tests); full tokenization pending |
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
| 2026-10-25 | 01 | stage-01 trainer on 20 MB real Python | time per merge | 0.22 s (4.9M chunks → 110k unique) |
| 2026-10-25 | 02 | codeparrot-clean shard profile | files / text / permissive bytes | 100k / 1.02 GB / 57.7% per shard |
| 2026-10-25 | 02 | leakage, 2 shards, **file** split, no near-dedup | rare val lines in train / files >50% leaked | 43.9% / 48.6% |
| 2026-10-25 | 02 | leakage, 2 shards, **repo** split, no near-dedup | rare val lines in train / files >50% leaked | 34.4% / 43.3% |
| 2026-10-25 | 02 | near-dedup, 12 shards, 729,865 files, 149 s | train dropped / val dropped | **28.2%** / **39.5%** (1,258 of 1,264 had a train copy) |
| 2026-10-25 | 02 | leakage, **file** split, after near-dedup | rare val lines in train / files >50% leaked | 21.7% / 27.4% |
| 2026-10-25 | 02 | leakage, **repo** split, after near-dedup | rare val lines in train / files >50% leaked | **8.1%** / **13.8%** |
| 2026-10-25 | 02 | exact dedup check on real shards | exact duplicates found | 0 (confirms upstream claim) |

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
- Is 8192 vocab too small for Python? Identifiers, keywords and indentation runs
  all compete for slots. Measure 8192 vs 16384 bytes/token on the real corpus
  before committing the embedding table.
- We have ~2.1B tokens of code+English but want ~2.5B for a 125M model, so
  ~1.2 epochs. How much does repeating code data actually cost? Stage 10.
- After near-dedup, 8.1% of rare val lines still appear in train. Believed to
  be partial copies (one vendored function inside an original file) that
  whole-file Jaccard cannot see. Would line- or function-level dedup close it,
  and does it matter for val loss in practice?
- Near-dedup left ~1,900 code val files (~5M tokens?). Enough for a stable val
  loss? Rough estimate says yes (per-file correlation → effective n ≈ files,
  std of mean ≈ 0.01), but measure it once a model exists.
- `codeparrot-clean` is less well filtered than the gated `bigcode/*` corpora.
  Does that show up as measurable quality loss, or is 5 GB of lightly-filtered
  Python fine at this scale?

## Session log

### 2026-10-25 — Session 3
- **Environment broke and was rebuilt.** The conda env sessions 1-2 used had
  vanished; system python is 3.10 with no torch. Rebuilt with `uv sync` from
  `pyproject.toml`, committed `uv.lock`, pinned torch to the cu128 index. Found
  `regex` had never been declared as a dependency. Rule now in `CLAUDE.md`:
  always `uv run`.
- **Corrected a session-2 number.** "codeparrot-clean = 477k files / 5 GB" came
  from the datasets-server, which converts only the first ~5 GB of big datasets.
  Real size: 54 shards, 12.8 GB gzipped, ~53 GB of text. Downloaded 12 shards.
- **Corrected a doc error**: Llama 3 does *not* split digits individually (it uses
  GPT-4's `\p{N}{1,3}`); Llama 1/2 and Qwen2 do.
- **Profiled before filtering.** Upstream already removed autogenerated / long-line
  / low-alpha files, so those planned filters would have been no-ops. Wrote
  `docs/02_data.md` around the measured profile.
- Built Stage 02: corpus streaming, permissive-license filter, repo-hash split,
  parallel tokenization to uint16 bins with a reproducibility manifest,
  random + shuffled-window samplers, runtime code/English mixing.
- Built a fast cached encoder (`llmfs/tokenizer`), proven identical to stage 01.
- **The big finding: GitHub near-duplicates.** The leakage ablation showed that even
  with a repo-level split, 43% of val files were mostly copies of train files
  (forks, vendoring). By the doc's own decision rule, near-dedup got built
  (MinHash + LSH). It removed 28% of train and 40% of val, and cut rare-line
  leakage 34% → 8%. Repo split and near-dedup turned out to be complementary;
  neither alone is enough.
- Fixed a latent deadlock: worker pools forked after torch had started threads.
  Switched to `spawn`.
- In progress at session end: production tokenizer training (3 variants to 16k).

### 2026-10-14 — Session 2
- **Domain decided: Python code + ~15% high-quality English.** Written up in
  `docs/domain_choice.md` (7 options weighed) and threaded through
  `CURRICULUM.md` stages 01, 02, 08, 10, 11, 16, 20, 22.
- Deciding criterion was **verifiability**, not usefulness: code gives Stage 20
  a free objective RL reward (unit tests pass) and Stage 22 objective eval
  (pass@k). Every unverifiable domain collapses those stages into
  LLM-as-judge, which at 125M params measures noise.
- Compute arithmetic done: the 4060 ceiling is ~125M params / 2.5B tokens /
  36-60 h at 15-25% MFU. 350M is out of reach at 280-470 h.
- **Verified every candidate dataset against the HF API rather than trusting
  recall.** Two traps found that would have cost a session each:
  1. `HuggingFaceTB/python-edu` is a *config* of `smollm-corpus`, and it ships
     **metadata only** — `blob_id`/`repo_name`/`score`, no source code. Getting
     the code means 477k+ authenticated S3 fetches from Software Heritage.
  2. Every `bigcode/*` corpus (`starcoderdata`, `the-stack-v2`, even
     `the-stack-smol`) is gated and needs terms accepted + a token.
  Landed on `codeparrot/codeparrot-clean` (477k files, 5.0 GB, ungated, has a
  per-file `license` column) + `fineweb-edu` `sample-10BT` + CodeSearchNet for
  SFT + HumanEval/MBPP for eval. Under 10 GB total, nothing gated.
- Recorded in `docs/datasets.md` with the verification date, since dataset
  repos get gated and restructured without notice.

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
