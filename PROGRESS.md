# Progress

Live state of the project. Update this at the end of every working session.
A new AI session reads `CLAUDE.md` then this file and knows exactly where to pick up.

## Current position

**All 25 stages complete (2026-11-14).** The capstone (92M, 1.80B tokens)
went through the whole chain: SFT → reward model → DPO → GRPO → quantize →
serve → evaluate. Stage 24 is assembled in `results/24_capstone/pipeline.json`,
written up in `docs/24_capstone.md`, and summarised in `MODEL_CARD.md`.

Headline: capstone base HumanEval 3.05% pass@1 (5/164), 5.49% pass@10; no
post-training stage beat the base model on HumanEval (GRPO tied it); every
solved problem has zero 13-gram overlap with training.

**Published (2026-11-16):** the book at karanjakhar.dev/book, the course at
karanjakhar.dev/course (12 core modules, `docs/COURSE.md`), its starter
repo github.com/karanjakhar/llm-from-scratch-course, and the checkpoints at
huggingface.co/karanjakhar/llm-from-scratch (pinned in
`course/get_checkpoints.py`). MIT licence. A change here reaches learners
only after regenerating the starter and redeploying the site (COURSE.md §6).

What is left is optional: the open questions below, the book's revision
pass (`docs/BOOK.md`: chapters are `draft`), and the course's electives.

To rerun the capstone chain from the pretrained checkpoint:

    nohup bash stages/24_capstone/run_post.sh > capstone_post.log 2>&1 &

Ready to use:
- Tokenizer: `data/tokenizers/split_8192.json` (8,196 embedding rows, eot 8192)
- Token files: `data/bins/split_8192/{train,val}_{code,en}.bin`, 2.36B tokens
- TinyStories bins (Stage 08 fixture): `data/bins/tinystories/`
- Eval data: `data/raw/{humaneval,mbpp,codesearchnet}` (harness verified: 164/164 references pass)
- Models: `llmfs.model.gpt.GPT` (GPT-2 names, loads OpenAI weights, KV cache,
  optional QK-norm), `llmfs.model.llama.Llama` (RMSNorm/RoPE/GQA/SwiGLU/
  QK-norm/window/MoE/MLA switches; equals HF Llama). `llmfs.model.load_model`
  rebuilds either from a checkpoint.
- Trainer: `llmfs.train.trainer.train(TrainConfig(...))`.

Deferred, with a trigger:
- Before Stage 11 (original note): download ~8 more code shards. Decided
  against: disk is at 26 GB free; 1.8B tokens at 85/15 = 1.15 code epochs.
- Not done: a full second `prepare.py` run to confirm byte-identical SHA-256s.

## Stage status

| # | Stage | Status | Notes / result |
|---|---|---|---|
| 00 | Setup & mental model | `[x]` | env verified, VRAM model holds to 8.5% |
| 01 | Tokenization | `[x]` | production tokenizer `split_8192`; beats GPT-2 on code 3.38 vs 2.22 B/tok |
| 02 | Data pipeline | `[x]` | 2.36B tokens; leakage 34%→8%; loader 700×+ faster than needed |
| 03 | Embeddings & position | `[x]` | RoPE 2.52 < sinusoidal 2.63 < learned 2.69 < none 3.35; tying helps |
| 04 | Attention | `[x]` | = SDPA to 1e-5; no √d: −0.04 at d=64, +0.95 at d=256; naive OOM at T=16k |
| 05 | Transformer block | `[x]` | = HF GPT2Block; post-norm 2.69 @4L / 6.70 @24L; no residual = unigram 6.86; SwiGLU −0.038 |
| 06 | GPT assembly | `[x]` | = HF GPT-2 (fp64 2e-11); 124,439,808 by hand; untied penalty 0.076→0.003 with 3× steps |
| 07 | Generation & decoding | `[x]` | cache exact in 36/36; speedup @512 1.3× (5M) / 3.3× (30M) / 4.5× (92M): overhead-bound |
| 08 | Training loop | `[x]` | `llmfs/train/trainer.py`; TinyStories 10M → coherent in 11.5 min; no warmup @3e-3 +1.84; sequential order +1.25 |
| 09 | Efficiency | `[x]` | 5.16× (14.3k → 73.7k tok/s, MFU 63%); compile 2.1× alone; capstone 25.4k tok/s |
| 10 | Scaling laws | `[x]` | optimum 4.2M @3e15 / 8.6M @1e16 = 22-28 tok/param (Chinchilla ±15%); QK-norm needed (v1 diverged) |
| 11 | Pretraining run | `[x]` | workhorse 30M 1.554; capstone 92M / 1.80B tokens / ~20 GPU-h: val code 1.295, Eng 2.898; resume continuous across a 5-day pause |
| 12 | Modern architecture | `[x]` | GPT-2→Llama 2.518→2.274; RoPE −0.230; MQA +0.03; window64 +0.24; NTK s=4 recovers 2.56→2.19 |
| 13 | Mixture of Experts | `[x]` | MoE ≈ dense at 33M tokens (2.325 vs 2.320), 16×top-4 2.303; no aux → 4/8 experts dead, loss unchanged |
| 14 | Attention variants | `[x]` | MLA r128 2.414 vs GQA-2 2.315 (MLA loses at this scale); window-128 0% needles beyond window at +0.11 loss |
| 15 | Speculative decoding | `[x]` | exact (fp32); greedy acceptance 0.83 @γ4, best speedup 1.20× (draft costs 0.43 of target) |
| 16 | SFT | `[x]` | format learned in 250 steps; HumanEval 0→1.2%; forgetting +0.15 code / +0.31 Eng; prompt-loss ablation a tie |
| 17 | LoRA / QLoRA | `[x]` | = peft; r=1 (0.22%) matches full SFT (1.0318 vs 1.0328); QLoRA +0.004; saves 510 MB = optimizer state |
| 18 | Reward modeling | `[x]` | 85.2% on test-verified MBPP pairs (56.8% untrained); 20%/40% flips → 76.6%/71.6% |
| 19 | DPO / RLHF | `[x]` | pair acc 0.50→0.87 (β=1); β≤0.1 over-optimises (RM 9.56→7.8, chosen logp −53); pass@1 unchanged |
| 20 | Reasoning / GRPO | `[x]` | MBPP: 0/576 rollouts pass, no signal; easy curriculum 13.3%→38.0% held-out, HumanEval 1.2→2.4%; format reward worse |
| 21 | Distill & quantize | `[x]` | distilled 2.438 vs scratch 2.518 (same tokens); int8 +0.0004, NF4 +0.016, int4 per-tensor +2.39 vs group +0.035 |
| 22 | Evaluation | `[x]` | 11 checkpoints, one harness; capstone base best (HumanEval 3.05% / pass@10 5.49%); solved problems 0 13-gram overlap; length norm +4-7 pts, reorders DPO |
| 23 | Serving | `[x]` | continuous batching: TTFT 377→8 ms, p50 0.78→0.27 s; 1→32 slots 295→1,304 tok/s; 21% of cache used |
| 24 | Capstone | `[x]` | 92M through SFT/RM/DPO/GRPO/quant/serve; SFT −3 HumanEval problems, GRPO +3; GRPO curriculum 27→73%; `MODEL_CARD.md` |

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
| 2026-10-25 | 01 | held-out code B/tok @ 8192: grouped / cap3 / split | compression | 3.381 / 3.376 / 3.337 |
| 2026-10-25 | 01 | held-out code B/tok @ 16384, grouped | compression | 3.586 (+6.1% over 8192) |
| 2026-10-25 | 01 | GPT-2 (50k) / cl100k (100k) on same held-out code | compression | 2.222 / 4.244 |
| 2026-10-25 | 01 | llmfs cached encoder vs stage-01, 1 core, real code | throughput | 8.14 vs 0.66 MB/s (12×), 98.2% cache hits |
| 2026-10-25 | 02 | full tokenization, 18 workers | tokens / time | 2,357M / ~4.5 min (~36 MB/s) |
| 2026-10-25 | 02 | train_code / train_en / val_code / val_en | tokens | 1,330.1M / 1,016.9M / 4.9M / 5.2M |
| 2026-10-25 | 02 | fineweb-edu exact duplicates | dropped | 1,476 train, 13 val |
| 2026-10-25 | 02 | MixedLoader B=16 T=1024, cpu / cuda / cuda+pinned | tokens/s | 56M / 125M / 140M (700-1750× need) |
| 2026-10-29 | 03 | tiny LM (4L, C=256, 32.8M tok, 3 seeds), **no position** | held-out code loss | 3.350 ± 0.013 |
| 2026-10-29 | 03 | same, learned absolute positions | held-out code loss | 2.689 ± 0.012 |
| 2026-10-29 | 03 | same, sinusoidal **without** √C token scaling | held-out code loss | 4.723 ± 0.080 (position 35× token size) |
| 2026-10-29 | 03 | same, sinusoidal with √C token scaling | held-out code loss | 2.633 ± 0.001 |
| 2026-10-29 | 03 | same, RoPE, tied head | held-out code loss | **2.523 ± 0.010** |
| 2026-10-29 | 03 | same, RoPE, untied head (+2.10M params) | held-out code loss | 2.598 ± 0.009 |
| 2026-10-29 | 03 | trained T=256, eval positions 256-511: none / sin-scaled / RoPE / RoPE-untied | loss (vs 0-255) | 3.51 (3.30) / 4.94 (2.61) / 3.29 (2.51) / 3.19 (2.58) |
| 2026-10-30 | 04 | random q,k,v, 256 keys, no √d, d=64 / 1024 | score-gradient median; queries with ~none | 1.23 / 0.022; 2% / 41% |
| 2026-10-30 | 04 | same, gradient at q, scaled vs unscaled, d=1024 | norm | 3.1 vs 63.1 (grows, not vanishes) |
| 2026-10-30 | 04 | naive vs fused attention, B=1 H=8 d=64 bf16 fwd+bwd, T=8192 | peak MB / ms | 4,168 / 148 vs 64.5 / 10.7; naive OOM at 16,384 |
| 2026-10-30 | 04 | tiny LM with our attention (= Stage 03 `learned`), seed 0 | held-out code loss | 2.6856 (Stage 03 with SDPA: 2.6849) |
| 2026-10-30 | 04 | 4 heads × 64, scaled / unscaled, 3 seeds | held-out code loss | 2.689 ± 0.012 / **2.652 ± 0.006** |
| 2026-10-30 | 04 | 1 head × 256, scaled / unscaled, 3 seeds | held-out code loss | 2.705 ± 0.003 / **3.655 ± 0.018** |
| 2026-10-30 | 04 | same, attention entropy / top weight | nats / weight | 2.78 / 0.29 vs 0.92 / 0.78 |
| 2026-10-30 | 04 | trained d=64 model, head statistics | prev-token head; sinks | L2H2 68%; L3H0 24%, L3H1 37% on first token |
| 2026-11-06 | 05 | untrained 24L, block-1 / block-24 grad norm: pre / post / no-residual | grad | 2.21/0.38 · 1.32/0.12 · 3.65/2.63 |
| 2026-11-06 | 05 | same, mean cosine between positions after block 24 | cosine | 0.69 · 0.03 · **1.000** (no residual, from block 12) |
| 2026-11-06 | 05 | unigram (token-frequency) cross-entropy, 512 val code windows | loss | **6.860** |
| 2026-11-06 | 05 | pre-norm 4L / 24L, small embeddings, seed 0 | held-out code loss | 2.685 / 2.567 |
| 2026-11-06 | 05 | post-norm 4L / 24L, small embeddings | held-out code loss | 6.856 / 6.857 (frequency floor) |
| 2026-11-06 | 05 | no residual 4L / 24L | held-out code loss | 6.856 / 6.856; block-1 grad 3.8e-19 @ step 1000 (24L) |
| 2026-11-06 | 05 | post-norm 4L, edit size rms(f)/rms(x) at step 500: small emb / √C emb | ratio | 2-24 / 0.3-0.6 |
| 2026-11-06 | 05 | √C embeddings: pre 4L / post 4L / pre 24L / post 24L / post 24L no warmup | held-out code loss | 2.685 / **2.693** / **2.469** / **6.700** / 6.856 |
| 2026-11-06 | 05 | post-norm 4L small emb, lr 3e-4 / 1e-4 | held-out code loss | 3.822 / 4.698 |
| 2026-11-06 | 05 | 4L sweep, 3 seeds: baseline / RMSNorm / ReLU / SwiGLU | held-out code loss | 2.689±.012 / 2.712±.027 / 2.714±.002 / **2.651±.012** |
| 2026-11-06 | 05 | no MLP 4L / no MLP 12L (param-matched) / dropout 0.1 | held-out code loss | 2.959±.028 / 2.909±.022 / 2.775±.024 |
| 2026-11-06 | 05 | norm fwd+bwd, 8192×C bf16, C=256: LN ours / RMS ours / nn.LN / nn.RMS | ms | 1.18 / 0.60 / 0.42 / 0.30 (C=4096 fused: 3.92 / 3.91; Linear C×C 36.8) |
| 2026-11-06 | 06 | our GPT with OpenAI GPT-2 weights vs HF, 200 val windows, GPU fp32 / fp64 | max logit diff | 0.055 / 2e-11 (residual stream max ≈3,000) |
| 2026-11-06 | 06 | GPT-2 small parameter count | by hand = numel | 124,439,808 |
| 2026-11-06 | 06 | GPT-2 124M on our val (650-tok windows) | bits per byte code / English | 1.394 / 1.064 |
| 2026-11-06 | 06 | 256-tok windows: GPT-2 124M vs ours 5.3M (12k steps) | bits/byte code; English | 1.522 vs **1.066**; **1.122** vs 1.774 |
| 2026-11-06 | 06 | stream rms after last block at init, plain vs scaled, 12L / 48L | rms (embed 0.028) | 0.474 vs 0.090 / 0.933 vs 0.089 |
| 2026-11-06 | 06 | 12L, 4k steps, 2 seeds: plain vs scaled residual init | held-out code loss | 2.581 vs 2.568 (tie) |
| 2026-11-06 | 06 | no shift (target = input), 1k steps | train / next-token val | 0.0001 / 21.86 (shift: 4.09) |
| 2026-11-06 | 06 | tied vs untied, 4L, 4k / 12k steps | held-out code loss | 2.6814 vs 2.7576 / 2.3577 vs 2.3608 |
| 2026-11-06 | 07 | KV cache vs naive, batch 1, fp32, length 512 / 1024 | speedup 5M / 30M / 92M | 1.27 / 3.32 / 4.47 ; 2.28 / 5.51 / 7.23 |
| 2026-11-06 | 07 | cached vs naive tokens, 36 settings | identical | 36/36 (greedy and sampled) |
| 2026-11-06 | 07 | Stage 06 model, greedy / τ0.7 / τ1.0 / τ2.0 | distinct-2 ; mean log-prob | 0.07/0.32/0.79/1.00 ; −0.24/−0.95/−3.19/−9.51 |
| 2026-11-06 | 08 | hand AdamW vs torch, 25 steps, fp32 / fp64 | max weight diff | 2.3e-4 / 4.2e-13 |
| 2026-11-06 | 08 | 4L, 1.5k steps: SGD / +momentum / Adam / AdamW / Adam+L2 (0.1) | held-out code loss | 5.615 / 4.310 / 3.609 / 3.614 / 6.695 |
| 2026-11-06 | 08 | overfit one batch (8×256) | steps to loss < 1 / 0.1 / 0.01 | 72 / 96 / 129 (spike ×1,600 at step 259) |
| 2026-11-06 | 08 | TinyStories 10.1M, 49M tokens | val loss / minutes | 1.510 / 11.5 |
| 2026-11-06 | 08 | 6L/384 lr 3e-3: base / no clip / no warmup / neither | held-out code loss | 2.946 / 3.028 / 4.781 / 5.057 |
| 2026-11-06 | 08 | 4L 4k steps: sequential / random / shuffled windows | held-out code loss | 3.937 / 2.695 / 2.683 |
| 2026-11-06 | 09 | 4096² matmul fp32 / TF32 / bf16 / fp16 | TFLOP/s | 5.48 / 10.43 / 23.63 / 24.76 |
| 2026-11-06 | 09 | 8L/512 T=512 B=8: fp32 → TF32 → bf16 → SDPA → fused Adam → compile | tokens/s | 14,271 → 18,536 → 23,142 → 34,490 → 35,098 → 73,656 (5.16×, MFU 63%) |
| 2026-11-06 | 09 | + activation checkpointing | tokens/s / peak MB | 60,955 / 718 (from 1,221) |
| 2026-11-06 | 09 | T=1024 full stack: workhorse B16 / capstone B8 | tokens/s (MFU) | 70,200 (67%) / 25,389 (71%) |
| 2026-11-06 | 10 | 4L/256, 13k steps, plain GPT-2 recipe | max attention score / val | 2,191,156 / 2.656 (diverged) |
| 2026-11-06 | 10 | same: lr 5e-4 / QK-norm lr 1e-3 | val code | 2.431 / **2.280** |
| 2026-11-07 | 10 | IsoFLOP 3e15 (L2..L8) | val code | 2.480 / 2.329 / **2.279** / 2.481 / 3.667 |
| 2026-11-07 | 10 | IsoFLOP 1e16 (L3..L10) | val code | 2.184 / 2.065 / **2.055** / 2.241 / 2.622 |
| 2026-11-07 | 10 | fitted optimum 3e15 / 1e16 | N (tokens/param) | 4.2M (28) / 8.6M (22.5); Chinchilla 5.0M / 9.1M |
| 2026-11-07 | 10 | 4L, 108M tokens: code100 / 85-15 / 70-30 | val code ; English | 2.182 / 2.204 / 2.260 ; 5.308 / 4.046 / 3.844 |
| 2026-11-07 | 10 | 4L code, 1 / 4 / 16 epochs | val code | 2.176 / 2.288 / 2.803 |
| 2026-11-07 | 10 | 3L, 1× / 2× / 5× / 10× optimal tokens | val code | 2.534 / 2.365 / 2.228 / 2.178 |
| 2026-11-07 | 11 | workhorse 30M, 590M tokens, 2h18m | val code / English | 1.554 / 3.293 (max grad norm after step 1k: 1.49) |
| 2026-11-07 | 12 | ladder (2 seeds): GPT-2 / +RMS / +RoPE / +SwiGLU / +no bias / +GQA2 / +QK-norm | val code | 2.518 / 2.551 / 2.321 / 2.295 / 2.304 / 2.329 / 2.274 |
| 2026-11-07 | 12 | kv heads 8 / 4 / 2 / 1 / 1+MLP | val code | 2.306 / 2.305 / 2.315 / 2.335 / 2.328 |
| 2026-11-07 | 12 | window full / 128 / 64 / 16 | val code | 2.314 / 2.424 / 2.559 / 3.141 |
| 2026-11-07 | 12 | trained @512, eval @2048, NTK s=1 / 2 / 4 | loss at pos 1024-2048 | 2.557 / 2.232 / 2.187 |
| 2026-11-07 | 13 | dense / MoE 8×top2 / no aux / 7+1 shared top1 / 16×top4 / dense same total | val code | 2.320 / 2.325 / 2.326 / 2.369 / 2.303 / 2.354 |
| 2026-11-07 | 13 | busiest expert share of top-2 assignments, with / without aux | fraction (balanced 0.125) | 0.23 / 0.41 (4 of 8 dead mid-layer) |
| 2026-11-07 | 14 | MLA r=128 / 64 / 32 / 16 / 4 | val code (cache B/token) | 2.414 (1728) / 2.476 (960) / 2.765 (576) / 2.801 (384) / 3.198 (240) |
| 2026-11-07 | 14 | needle mean retrieval: full / win128 / win64 / MLA64 / MLA16 | fraction | 0.51 / 0.31 / 0.09 / 0.58 / 0.04 |
| 2026-11-07 | 11 | resume at 600: resumed / fresh Adam | final val (uninterrupted 3.9158) | 3.9158 (bit-identical) / 3.9323 |
| 2026-11-07 | 15 | spec. decoding, 30M target / 5M draft, greedy γ=1/4/6/8 | acceptance ; speedup | 0.93/0.83/0.77/0.73 ; 0.86/1.17/1.20/1.15 |
| 2026-11-07 | 15 | same, τ=1.0, γ=1 / 8 | acceptance ; speedup | 0.67 / 0.26 ; 0.82 / 0.25 |
| 2026-11-07 | 16 | SFT workhorse, 3k steps: completion val ; HumanEval / MBPP pass@1 (chat) | loss ; fraction | 1.168→1.033 ; 0.0122 / 0.0 |
| 2026-11-07 | 16 | SFT forgetting: pretraining val code / English | loss before → after | 1.684→1.832 / 3.341→3.646 |
| 2026-11-07 | 18 | RM on SFT workhorse: CSN held-out / MBPP verified | pairwise acc | 0.826 / 0.852 (untrained 0.507 / 0.568) |
| 2026-11-07 | 18 | labels flipped 0 / 20 / 40% | MBPP acc | 0.852 / 0.766 / 0.716 |
| 2026-11-07 | 19 | DPO β 0.01/0.05/0.1/0.3/1.0: MBPP pair acc | fraction | 0.816 / 0.835 / 0.842 / 0.861 / 0.873 |
| 2026-11-07 | 19 | same: RM score of own samples (SFT 9.56) ; KL | score ; nats | 7.89/7.64/7.84/9.10/11.13 ; 0.173/0.185/0.160/0.071/0.020 |
| 2026-11-07 | 20 | GRPO on MBPP, SFT workhorse, 3 steps | rollouts passing | 0 / 576 |
| 2026-11-07 | 20 | easy tasks held-out pass@1, steps 0/15/30/45/60, outcome only | fraction | 0.133 / 0.140 / 0.247 / 0.307 / 0.380 |
| 2026-11-07 | 20 | same, outcome + format | fraction | 0.133 / 0.167 / 0.200 / 0.247 / 0.267 |
| 2026-11-07 | 20 | after GRPO (outcome only): HumanEval / MBPP pass@1 | fraction | 0.0244 / 0.002 (SFT 0.0122 / 0) |
| 2026-11-07 | 21 | 5.4M student, 6k steps: scratch / distilled from workhorse | val code | 2.518 / 2.438 (380 s / 1,126 s) |
| 2026-11-07 | 21 | workhorse quant: int8-ch / NF4-64 / int4-g32 / int4-g128 / int4-ch / int4-tensor / int3-g32 | Δ val code | +0.0004 / +0.016 / +0.020 / +0.035 / +0.059 / +2.39 / +0.13 |
| 2026-11-07 | 23 | serving ceiling, 1 / 4 / 16 / 32 slots | tokens/s | 295 / 782 / 1,189 / 1,304 |
| 2026-11-07 | 23 | bursty 10 req/s, static vs continuous | p50 / p99 / TTFT p50 (s) | 0.777 / 1.953 / 0.377 vs 0.272 / 1.563 / 0.008 |
| 2026-11-07 | 17 | SFT full / LoRA r1 / r4 / r16 / r64 / QLoRA r16 | completion val loss | 1.0328 / 1.0318 / 1.0278 / 1.0281 / 1.0328 / 1.0321 |
| 2026-11-07 | 17 | peak training MB, full vs LoRA r16 | MB | 4,615 vs 4,104 |
| 2026-11-14 | 11 | capstone 92M, 1.80B tokens, ~20.4 GPU-h | val code / English | 1.295 / 2.898 (max grad norm after warm-up 1.03) |
| 2026-11-13 | 11 | capstone resume after 5-day pause, steps 10,500 / 10,620 / 10,700 | train loss before vs after | 1.5527=1.5527 / 1.8381 vs 1.8383 / 1.5945=1.5945 |
| 2026-11-14 | 15 | capstone target, workhorse draft: greedy acceptance γ1 / γ4 / γ8 ; best speedup | fraction ; × | 0.972 / 0.891 / 0.791 ; 0.98 (draft costs 0.671 of target) |
| 2026-11-14 | 16 | capstone SFT: completion val loss before / step 250 / after | nats | 0.998 / 0.888 / 0.886 |
| 2026-11-14 | 16 | capstone forgetting, code / English | Δ nats | +0.159 / +0.311 |
| 2026-11-14 | 16 | capstone HumanEval: base as code vs SFT as chat | pass@1 | 0.0305 vs 0.0122 |
| 2026-11-14 | 18 | capstone RM: held-out CSN / MBPP verified | accuracy | 0.871 / 0.865 (workhorse 0.826 / 0.852) |
| 2026-11-14 | 19 | capstone DPO β=1: pair acc ; RM own samples ; KL | | 0.563→0.895 ; 20.41→25.37 ; 0.027; pass@1 unchanged |
| 2026-11-14 | 20 | capstone GRPO easy held-out, steps 0/15/30/45/60 | pass@1 | 0.273 / 0.380 / 0.420 / 0.533 / 0.733 |
| 2026-11-14 | 20 | capstone GRPO: HumanEval / MBPP | pass@1 | 0.0305 / 0.004 (SFT 0.0122 / 0.004) |
| 2026-11-14 | 21 | capstone quant: int8-ch / NF4-64 / int4-g128 / int4-ch / int4-tensor | Δ val code | +0.0000 / +0.0134 / +0.0254 / +0.0456 / +3.2174 |
| 2026-11-14 | 23 | capstone bursty: static vs continuous | tok/s ; TTFT p50 (s) | 707 vs 720 ; 0.440 vs 0.008 |
| 2026-11-14 | 22 | HumanEval greedy / pass@10: workhorse base, SFT, DPO, GRPO | fraction | 0/.012, .012/.024, 0/.006, .024/.018 |
| 2026-11-14 | 22 | HumanEval greedy / pass@10: capstone base, SFT, DPO, GRPO | fraction | .0305/.0549, .0122/.0488, .0122/.0183, .0305/.0488 |
| 2026-11-14 | 22 | 5M / 12M / 34M-MoE models, HumanEval and MBPP | pass@1 | 0 / 0 / 0 |
| 2026-11-14 | 22 | MC (352 items, 4 options): summed vs per-token, capstone base / DPO, workhorse base / DPO | accuracy | .716/.767, .767/.778, .594/.668, .653/.651 |
| 2026-11-14 | 22 | solved problems with any training 13-gram | count | 0 of 9 distinct |

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
- ~~How much does repeating code data cost?~~ **Partly answered (Stage 10):**
  4 epochs of a contiguous quarter +0.11, 16 epochs +0.63; confounded by the
  slice's lower diversity. A randomly sampled slice would separate the two.
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
- ~~Does the untied-head penalty shrink with longer training?~~ **Answered
  (Stage 06):** yes. +0.076 at 4k steps, +0.003 at 12k. Slow learning.
- NoPE degrades least past the training length (+0.21) but starts 0.83 worse.
  Does the in-range gap shrink with depth and more tokens, as the NoPE papers
  suggest? Not worth a dedicated run; note if Stage 14 revisits it.
- Without √d at d=256, attention saturated and loss rose 0.95, yet the
  gradient reaching W_qkv was *larger*, not smaller. What exactly makes a
  saturated model learn worse? Candidates: gradient concentrated on few
  queries, effective learning rate too high for the scores. Unproven.
- At d=64 the √d-scaled model started with score std 0.10 (near-uniform
  attention) and lost to unscaled by 0.04. Would a larger q/k init close it?
  Worth a check when Stage 05 revisits initialisation.
- Pre-norm's positions grow alike with depth at init and √C embeddings
  helped 24L pre-norm by 0.10 (one seed). Stage 06's residual-scaled init
  keeps the stream from swamping the embeddings at init (0.474 → 0.090 rms
  at 12 layers) but changed final loss by only 0.013; the √C question at 24
  layers was not re-run.
- Post-norm 24L with √C still failed at lr 1e-3 + 100 warmup. Would a long
  warmup or lr 3e-4 rescue it, and at what cost vs pre-norm? Not run; it
  would take ~1 h of GPU and does not change any decision.
- ~~RMSNorm's +0.023: a tie?~~ **Answered (Stage 12):** +0.033 at 6L/384, 2
  seeds (spread 0.01): a small, consistent cost, kept for speed.

- Small models' gradient norms creep up over long runs even with QK-norm
  (3L/192: 1.3 → 2.5 over 73k steps). Embedding norm, logit growth, or
  Adam's ε? (Stage 10)
- MoE tied dense and MLA lost to GQA at 33M tokens / 12M params. Do they win
  at the workhorse's 590M tokens? (Stages 13-14)
- DPO on mutation pairs improved ranking (0.50 → 0.87) but never pass@1.
  Can pairwise preference move generation at all at this scale? (Stage 19)
  **Still open after the capstone:** 0.895 pair accuracy, identical solved
  set; 3× the parameters changed nothing here.
- GRPO on the capstone was still climbing at step 60 (53% → 73% between
  steps 45 and 60). Where does it saturate, and does more of it ever reach
  HumanEval beyond the five one-liners? (Stage 20)
- Would a shorter SFT (250 steps, where the loss flattened) keep the base
  model's 5 HumanEval problems? (Stage 16 §8d)
- SFT taught one prompt format: prose tasks 0%, code-shaped 7.5% (Stage 20).
  Would a mixed-format SFT set lift MBPP from 0%?

## Session log

### 2026-11-15 — Session 13
- Owner chose to publish without a Colab pass. Starter repo public:
  github.com/karanjakhar/llm-from-scratch-course (`be3993f`, generated by
  `course/make_starter.py`, scanned for secrets and large files first).
  /course live at karanjakhar.dev/course (site commit `dd8db75`, deployed
  from a clean worktree so the owner's uncommitted mobile edits stayed local).
- Checkpoint upload to the HF Hub failed: the local token is read-only (403).
  Model card drafted; needs a write token, then pin `REVISION` in
  `course/get_checkpoints.py`.
- 2026-11-16: with the owner's new write token, uploaded the three
  checkpoints (Hub commit `e09d40e`), pinned it in `get_checkpoints.py`,
  verified a fresh download by SHA-256, updated module 9's lesson, the
  starter repo and /course.
- 2026-11-16: MIT licence added (this repo, starter, Hub card). Owner marked
  the Colab pass done.
- 2026-11-16 pre-final audit: published starter equals a fresh build; 302
  tests pass; fixed web-page links to gitignored files (book + course);
  README now points to the book, course, starter and checkpoints.
- 2026-11-16: the owner's hand-made phone layout for the book page moved
  into `book/template.html`; book rebuilt without its 5 dead links; the
  site's nav has Book and Course; the LLM project page links the course and
  says 92M, not ~125M.
- 2026-11-16 privacy scan: removed 13 local paths from committed files
  (results configs now saved repo-relative); repo-local git email set to
  the owner's Gmail in all three repos.
- Noted: the repo has no LICENSE file (owner's call).

### 2026-11-15 — Session 12
- Owner decided to turn the project into a course (site + starter repo,
  implement-and-tests-grade, Colab baseline, 12 core modules + electives).
  Plan: `docs/COURSE.md`.
- Built the course machinery and module 1 as a pilot: `course/exercises.py`
  (what to blank), `course/make_starter.py` (blanks via `ast`, no stage file
  edited), `course/check.py` (grader), `course/make_notebooks.py` (Colab),
  `course/build_web.py` (the /course page, reusing the book's template),
  `course/modules/01_tokenizer.md`, `course/STARTER_README.md`.
- Verified on a generated starter: 0 → 5 → 58 of 58 tests as functions are
  written; a clean Py 3.12 venv with the notebook's pip list passes too.
- Then all 12 core modules: `course/exercises.py` (58 exercises),
  `course/modules/00-11_*.md`, `course/colab/*.ipynb` (Drive-backed),
  `course/verify_starter.py`. Evaluation (10) now precedes SFT (11). New
  tests: `course/module00/test_budget.py` (5), `llmfs/train/test_resume.py`
  (2); suite 302 passed. The starter ships `split_8192.json`.
- Verified: every module blank-fails and solution-passes; each of the 58
  exercises blanked alone fails a test. Every experiment rerun on a learner-
  sized subset (2 code shards: 275M code tokens, 1.0M val code tokens) and
  its numbers put in the lessons beside the book's.
- Fixed in lessons before publishing: six wrong claims (journal), Stage 05
  `grads` needs `--B 8` (default 32 OOMs at 24L fp32), eval `table` needs
  `--redo`, Stage 08's `overfit`/`tinystories` use torch AdamW not the
  learner's.
- Not yet: a pass through the notebooks on real Colab; the starter repo on
  GitHub, checkpoints on the HF Hub, the /course deploy (all need the
  owner's go).

### 2026-11-14 — Session 11
- Published: pushed the 41 commits to GitHub and rebuilt the web edition at
  karanjakhar.dev/book (site repo commit `3c8c9c5`).
- Wrote the last two book parts: `docs/preface.md` and
  `docs/appendix_reproduce.md` (Appendix C). The book is 30/30 parts drafted,
  82,709 words. Refreshed `docs/book_intro.md`, which still said "six of 26".
- Found that HumanEval, MBPP, CodeSearchNet and TinyStories had no download
  command; added `stages/02_data/download_extras.py` with the revisions the
  HF cache recorded, checked against the Hub (all 11 files exist).
- `book/build_web.py` no longer copies `.py` files from `docs/visuals/` into
  the site.

### 2026-11-13/14 — Session 10
- Resumed the paused capstone (step 10,500, bit-identical continuation) and
  finished it: 1.80B tokens, val code 1.295, English 2.898, max grad norm 1.03.
- Capstone chain: Stage 15 sweep (acceptance up to 0.97, no speedup), SFT,
  RM, DPO, GRPO, quant, serving, the Stage 22 table, MC and contamination.
- **SFT OOM'd on the 92M model at B = 16** and the unguarded chain ran on.
  Added `GPT.grad_checkpointing` (no parameter renaming), `--act-ckpt` in
  stages 16/18/19/20, `train_sft(micro=…)`, two tests, and
  `stages/24_capstone/run_post.sh` with `set -e`.
- Findings: SFT cost the capstone 3 HumanEval problems; GRPO won them back
  and took its curriculum 27% → 73%; DPO never changed a solved problem;
  length normalisation reorders DPO; nothing solved is contaminated.
- Stage 21's capstone plot overwrote the workhorse figure; restored.
- Stages 22 and 24 written, `MODEL_CARD.md` filled, README status, two new
  figures (`22_evaluation.png`, `24_capstone.png`), BOOK rows. All stages done.
- Not pushed (owner pushes on request).

### 2026-11-06/08 — Session 9
- Stages 06-21 and 23 built, tested and measured on the workhorse (see the
  status table and results log); Stage 10 needed QK-norm after the first
  sweep diverged; workhorse pretrained (30M, val code 1.554 on Stage 11's
  set); capstone started and paused at the owner's request at step 10,500.

### 2026-11-06 — Session 8
- **Stage 05 complete.** `docs/05_transformer_block.md`; `stages/05_block/`
  with 23 tests (norms vs `nn.LayerNorm` / `nn.RMSNorm` / `LlamaRMSNorm`,
  GELU/SiLU, SwiGLU vs `LlamaMLP`, our block vs HF `GPT2Block`, the
  Conv1D-transpose trap, residual identity, causality, param count).
- Experiments: init gradients/stream/similarity at 24L; pre/post/no-residual
  at 4 and 24 layers; a post-norm diagnosis; √C-embedding fair depth test;
  6-variant × 3-seed sweep + parameter-matched no-MLP; norm speed.
  ~5 h of GPU in total.
- **Corrections to the textbook story, all measured:** without residuals the
  gradients did *not* vanish at init; tokens merged first (rank collapse),
  and gradients died only after the model fell back to predicting token
  frequencies. Failure was a silent plateau at the unigram loss (6.86), not
  divergence. Post-norm failed at 4 layers too, because of our small
  embeddings; with the 2017 √C multiplier it matched pre-norm at 4 layers and
  failed at 24, the curriculum's claim.
- Caught a test mistake: targets = inputs in a tied model gives 8.46 at init,
  not ~9.0. Kept as its own test.
- 9 journal entries, 12 glossary terms, 4 figures (`docs/visuals/05_*.png`).
- Not pushed (owner pushes on request).

### 2026-10-31 — Session 7
- **Web edition of the book**: `book/build_web.py` turns the drafted docs
  (chapter map read from `docs/BOOK.md` §3) into one page plus figures.
  Built into the personal site at `public/book/`, served at
  karanjakhar.dev/book/ once that site is deployed. Project page links to it.
- Figures are now embedded in `docs/03_*.md` and `docs/04_*.md`, not only
  named by path.
- `markdown-it-py` added as a direct dependency (was only transitive via rich).
- Quarto considered for PDF/EPUB; install not approved, dropped for now.
- Stage 05 not started.

### 2026-10-30 — Session 6
- **Stage 04 complete.** `docs/04_attention.md`; `stages/04_attention/` with 12
  tests (SDPA outputs + gradients to 1e-5, `nn.MultiheadAttention`, causality,
  the wrong-reshape trap, the NaN-row trap, score variance).
- Experiments: √d saturation on random vectors; naive vs fused memory/time up
  to T=16,384; √d training ablation (4 variants × 3 seeds); attention maps.
- **Two corrections to the textbook story**, both measured: the gradient at q
  grows without √d (only the score gradient vanishes), and at d=64 dropping
  √d slightly *helped* because small init made scaled scores start flat.
- At d=256 dropping √d broke the model (+0.95), as the curriculum predicted.
- 6 journal entries, 6 glossary terms, 4 figures (`docs/visuals/04_*.png`).
- Not pushed (owner pushes on request).

### 2026-10-29 — Session 5
- Pushed the session 3-4 commits (pre-push checks: no data/, no co-author
  lines, no credentials).
- **Stage 03 complete.** `docs/03_embeddings_position.md` (book voice, worked
  examples), `stages/03_embeddings/` with 22 tests: embedding = one-hot matmul
  (outputs and gradients), permutation equivariance, sinusoid properties, RoPE
  offset-only scores, interleaved = complex, half-split = HF Llama exactly,
  tied head shares storage.
- Ablation, 6 variants × 3 seeds (18 runs, ~40 min): RoPE best (2.523), no
  position worst among working schemes (3.350). See results log.
- **Caught a silent bug by comparison:** sinusoids without √C token scaling
  scored 4.72, worse than no position. Position was 35× the token's size. Fixed
  as a separate variant so both numbers stay in the record.
- Test failure from float32 RoPE angles: test asked for more precision than
  the input had. Fixed the test, added a `dtype` argument, documented.
- 6 journal entries, 6 glossary terms, 3 figures (`docs/visuals/03_*.png`).

### 2026-10-29 — Session 4
- **The project will become a book** (owner's decision). Set up the system so
  findings are written beside the work, not reconstructed later:
  - `docs/JOURNAL.md`: lab notebook, backfilled with 33 dated entries from
    sessions 1-3 (results, surprises, mistakes, traps, decisions).
  - `docs/BOOK.md`: pitch, reader, a fixed 10-part chapter arc, voice guide,
    chapter map (26 chapters + 3 appendices), figure inventory, upkeep rules.
  - "What we found" sections added to `docs/00`, `01`, `02`.
  - Visual build map saved into the repo at
    `docs/visuals/stages_00-02_build_map.html` (only copy was in a temp dir).
  - Rules 9-10 added to `CLAUDE.md` so every session keeps the journal current.
- No stage work this session.

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
- **Production tokenizer chosen: `split_8192`.** 16k vocab bought only +6.1%
  on code. Splitting digits costs 1.3% but gives stable digit ids. Our 8k
  tokenizer beats GPT-2's 50k on code (3.38 vs 2.22 B/tok): GPT-2 has no
  whitespace-run tokens, so indentation is ruinous for it.
- Fixed `train_production.py` saving tokenizers without special tokens.
- **Full corpus tokenized: 2.36B tokens, 4.4 GB, ~4.5 min.** All acceptance
  checks pass on the real files. The val_en decode check "failed" once, but the
  cause was a verbatim train/val duplicate article that dedup had correctly
  removed; the check was wrong.
- Loader is 700-1,750× faster than the model needs. No DataLoader workers.
- Published to GitHub (personal account) after scrubbing a work-account name
  from history. Rule recorded: no Co-Authored-By in commits.
- Built a visual artifact of stages 00-02 with a live BPE lab, verified
  merge-for-merge against `RegexBPE`.

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
