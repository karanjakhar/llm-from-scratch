# Glossary

Every term, defined once, in plain language. Alphabetical. Add a term the
first time it appears in any stage.

**Activation checkpointing** — Don't store intermediate activations during the
forward pass; recompute them during the backward pass instead. Trades roughly
30% more compute for a large cut in memory. Stage 09.

**Activations** — The intermediate tensors produced by the forward pass. They
must be kept alive until the backward pass uses them, which is why they
dominate memory at long sequence lengths.

**Attention** — A mechanism where each position builds its output as a weighted
average of all (allowed) positions' values, with weights computed from
query-key similarity. Stage 04.

**Autoregressive** — Generating a sequence one token at a time, each token
conditioned on all previous ones. Formally: `P(x) = Π P(x_t | x_<t)`.

**bf16 (bfloat16)** — 16-bit float with the same exponent range as fp32 but
fewer mantissa bits. Range matters more than precision for training, which is
why bf16 needs no loss scaling while fp16 does. Stage 09.

**BPE (Byte Pair Encoding)** — A tokenizer training algorithm: repeatedly find
the most frequent adjacent pair of symbols in the corpus and merge it into a
new symbol. Stage 01.

**Causal mask** — Setting attention scores to `-inf` for future positions, so
position t can only attend to positions ≤ t. What makes the model
autoregressive. Stage 04.

**Chinchilla** — The 2022 result that compute-optimal training uses roughly 20
training tokens per parameter — far more data than earlier practice. Stage 10.

**Context length** — The maximum number of tokens the model can attend over in
one forward pass. Also called block size or sequence length.

**d_model / n_embd** — The width of the residual stream; the embedding
dimension that every block reads from and writes to.

**DPO (Direct Preference Optimization)** — Fine-tuning on preference pairs via a
closed-form loss that provably optimizes the same objective as RLHF, with no
reward model and no RL loop. Stage 19.

**Embedding** — A lookup table mapping each token id to a learned vector.
Mathematically a one-hot vector times a matrix. Stage 03.

**Expert** — One of several parallel FFN sub-networks in an MoE layer. Each token
is routed to only a few of them. Stage 13.

**FFN / MLP** — The per-position feed-forward network inside a transformer
block, usually expanding width 4x and projecting back. Where most parameters
live. Stage 05.

**FlashAttention** — An attention implementation that tiles the computation so
the n×n attention matrix is never written to memory. Same math, far less memory
traffic. Stage 09.

**GQA (Grouped-Query Attention)** — Query heads are grouped and each group shares
one key/value head, shrinking the KV cache. MHA and MQA are its two extremes.
Stage 12.

**Gradient accumulation** — Summing gradients over several small forward/backward
passes before stepping the optimizer, to simulate a large batch on small memory.
Stage 08.

**GRPO (Group Relative Policy Optimization)** — An RL method that samples a group
of responses per prompt and uses their relative rewards as advantages, removing
the value network. Stage 20.

**Head** — One attention subspace. Multi-head attention splits `d_model` into
`n_head` heads of size `d_model / n_head`, each attending independently.

**KV cache** — Stored keys and values from previous decoding steps, so each new
token's attention does not recompute them. Turns O(n²) decoding work into O(n).
Stage 07.

**Logits** — The raw, unnormalized scores over the vocabulary that come out of
the final linear layer, before softmax.

**LoRA (Low-Rank Adaptation)** — Freeze the base weights and train a low-rank
update `B·A` instead, cutting trainable parameters by orders of magnitude.
Stage 17.

**MFU (Model FLOPs Utilization)** — Achieved FLOP/s divided by the GPU's peak
FLOP/s. The honest measure of training efficiency. Stage 09.

**MoE (Mixture of Experts)** — Replacing one FFN with many, and routing each
token to a few, so total parameters grow without per-token FLOPs growing.
Stage 13.

**Perplexity** — `exp(cross-entropy loss)`. Intuitively, how many tokens the
model is effectively choosing between at each step. Lower is better.

**Pre-norm** — Normalizing the input to each sublayer rather than its output.
Required for stable training at depth. Stage 05.

**Residual stream** — The running sum that each block reads from and adds to.
The "highway" that lets gradients reach early layers.

**RMSNorm** — Normalization by root-mean-square only, without subtracting the
mean. Cheaper than LayerNorm with equal quality. Stage 05.

**RoPE (Rotary Position Embedding)** — Encoding position by rotating query and
key vectors by an angle proportional to position, making attention scores depend
on relative distance. Stage 03.

**Router / gate** — The small network in an MoE layer that decides which experts
each token goes to. Stage 13.

**SFT (Supervised Fine-Tuning)** — Training a base model on
instruction/response pairs so it answers instead of merely continuing text.
Stage 16.

**Softmax** — Turns a vector of scores into a probability distribution:
`exp(x_i) / Σ exp(x_j)`.

**SwiGLU** — A gated feed-forward variant: `(W1·x ⊙ silu(W3·x))·W2`. Standard in
modern models. Stage 05.

**Temperature** — Dividing logits before softmax. <1 sharpens the distribution,
>1 flattens it. Stage 07.

**Token** — The unit the model actually sees: a sub-word chunk of bytes produced
by the tokenizer. Not a word, not a character. Stage 01.

**Top-k / top-p sampling** — Truncating the next-token distribution to the k most
likely tokens, or to the smallest set whose probability sums to p, before
sampling. Stage 07.

**Warmup** — Ramping the learning rate up from ~0 over the first few hundred
steps, because Adam's variance estimates are unreliable early. Stage 08.

**Weight tying** — Sharing one matrix between the input embedding and the output
projection. Saves `vocab × d_model` parameters. Stage 06.
