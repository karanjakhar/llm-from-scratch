# Stage 24 — Capstone

One model, taken the whole way:

```
tokenizer (01) → data (02) → pretrain (11) → SFT (16) → DPO (19) → GRPO (20)
             → quantize (21) → evaluate (22) → serve (23)
```

Every step below uses code built and tested in its own stage. Nothing here is
new machinery; this stage is about putting it on one page, with one set of
numbers, and being honest about what the result can and cannot do. The model
card is `MODEL_CARD.md` at the repository root; every number in it comes
from `results/24_capstone/pipeline.json`, which is assembled from the stage
result files without recomputing anything.

---

## 1. The pipeline, with its numbers

*(filled in from results/24_capstone/pipeline.json once the capstone chain
finishes)*

---

## 2. What the model can and cannot do

*(filled in after the evaluation)*

---

## 3. What was learned, part by part

**Foundations (Stages 00-07).** The tokenizer decided more than expected: a
code-specific 8,192-token vocabulary compresses our Python 1.5× better than
GPT-2's, and at small model sizes the embedding table is a large share of
the parameters (39% of our 5M model, 31% of GPT-2 small, 7% of the 92M
capstone with our vocabulary; it would be 30% with GPT-2's), so vocabulary
size is a model-size decision (Stages 01, 06). Data work
was most of the work: near-duplicate removal cut train/val leakage from 34%
to 8% (Stage 02). Every architectural piece has a failure you can produce on
purpose and measure: no positions (+0.83), no √d at width 256 (+0.95), no
residual (the model learns only unigram statistics, 6.86), post-norm at
depth (fails at 24 layers), no next-token shift (train loss 0, real loss
21.9). Loading OpenAI's GPT-2 weights into our own code and matching Hugging
Face's logits (float64 difference 2e-11) was the moment the model became
trustworthy (Stage 06).

**Training (Stages 08-11).** A training loop fails quietly. The one-batch
overfit test caught nothing because nothing was wrong, which is the point of
running it; the resume test caught a real checkpoint bug on its first run
(Stage 11). The biggest single speedup was not mixed precision but kernel
fusion (2.1×), because after bf16 the step was dominated by elementwise
operations (Stage 09). Scaling laws transferred to our scale: our IsoFLOP
optima sit within 15% of Chinchilla's 20 tokens per parameter (Stage 10).
But the first sweep diverged: attention scores grew to two million over long
runs of a small model, and QK-norm, borrowed early from Stage 12's list,
fixed it (Stage 10).

**Architecture (Stages 12-15).** Unbundled, GPT-2 → Llama is mostly one
change: RoPE is −0.230 of the −0.244 (Stage 12). GQA is free down to 2 kv
heads. MoE, MLA and speculative decoding all depend on scale or data we do
not have: MoE ties a dense model at 33M tokens, MLA loses to GQA at 12M
parameters, and speculative decoding reaches 1.2× because a 5M draft step
costs 43% of a 30M target step. Each is still worth building, because the
measurement says exactly which precondition is missing.

**Post-training (Stages 16-21).** SFT teaches a format in 250 steps and then
only forgets (Stage 16). LoRA at rank 1 matches full fine-tuning on that
format (Stage 17). A 30M model that cannot write code can still judge it: 85%
accuracy telling working functions from their one-mutation twins (Stage 18).
DPO learned that ranking and did not improve generation; at small β it
over-optimised, lowering the reward model's own opinion of its samples (Stage
19). GRPO with unit tests as the reward could not start on MBPP (0 of 576
rollouts passed) and nearly tripled pass@1 on a curriculum the model could
sometimes solve (Stage 20). Quantization granularity matters more than bit
width between 4 and 8 bits (Stage 21).

**Evaluation and serving (Stages 22-23).** The harness had to be tested
before the models: reference solutions must pass, stubs must fail, and an
early `os._exit(0)` must not count. A trailing newline in HumanEval's prompts
emptied every base-model completion until we read one raw (Stage 22).
Continuous batching changed who waits, not how fast the GPU runs: 47× faster
first token at the same throughput (Stage 23).

---

## 4. What is still fuzzy

- **Why small models' gradient norms creep upward over long runs** even with
  QK-norm (the 3-layer model's grew from 1.3 to 2.5 over 73k steps). Is it
  the embedding norm, the output logits, or Adam's `ε`?
- **Whether MoE and MLA win at a scale we can reach.** Both lost or tied at
  33M tokens; neither was tried at the workhorse's 590M.
- **What DPO on synthetic bug pairs would do with more data or a larger
  model**: whether pairwise preference can ever move pass@1, or only ranking.
- **How much of the HumanEval gap is format.** The SFT model scored 0% on
  prose-phrased tasks it solved 7.5% of the time when phrased as code. A
  mixed-format SFT set would answer it.
- **Repetition.** Four epochs cost 0.11 nats, more than published results;
  our repeated slice was also less diverse. Untangling those two needs a
  randomly sampled slice.

---

## 5. What to learn next

- **Multi-GPU training**: data, tensor and pipeline parallelism, and the
  communication that dominates them. Everything here fit on one GPU, which
  hid the hardest part of training large models.
- **Fused kernels**: a Triton kernel for the quantized matmul (Stage 21's
  missing speedup) and for paged attention (Stage 23's missing memory saving).
- **Data quality at scale**: classifier-based filtering (as FineWeb-Edu did)
  applied to code, and measured with the Stage 22 harness.
- **Reasoning that needs steps**: an RLVR curriculum where thinking helps
  (multi-step arithmetic, small algorithmic tasks), to see whether response
  length grows when it pays.
