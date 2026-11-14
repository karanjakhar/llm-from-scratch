# Glossary

Every term, defined once, in plain language. Alphabetical. Add a term the
first time it appears in any stage.

**Acceptance rate** — In speculative decoding, the fraction of draft tokens
the target model accepts. Decides whether speculation pays off. Stage 15.

**Activation checkpointing** — Don't store intermediate activations during the
forward pass; recompute them during the backward pass instead. Trades roughly
30% more compute for a large cut in memory. Stage 09.

**Activations** — The intermediate tensors produced by the forward pass. They
must be kept alive until the backward pass uses them, which is why they
dominate memory at long sequence lengths.

**Active parameters** — The parameters actually used for one token. In an
MoE, everything shared plus `k` of the `E` experts per layer. Compute per
token scales with active parameters; memory with total. Stage 13.

**Adam / AdamW** — Optimizer keeping two running averages per weight: the
gradient (`m`, momentum) and the squared gradient (`v`), stepping by
`m/√v` so every weight moves on the same scale. AdamW applies weight decay
directly to the weights instead of through the gradient. Stage 08.

**Advantage** — How much better an action turned out than expected. PPO
estimates it with a value model (GAE); GRPO with the group mean. Stage 19.

**Attention** — A mechanism where each position builds its output as a weighted
average of all (allowed) positions' values, with weights computed from
query-key similarity. Stage 04.

**Attention map** — The `(T, T)` matrix of attention weights for one head:
row i shows how token i splits its attention over earlier tokens. Shows where a
head looked, not what it used. Stage 04.

**Attention sink** — A position (usually the first token) that an attention
head dumps weight on when it has nothing useful to read, because softmax
weights must sum to 1. The weight is large but carries little information.
Stage 04.

**Autocast** — PyTorch mode that runs matmuls in a low-precision format
(bf16) while keeping reductions, softmax and the weights themselves in fp32.
Stage 09.

**Autoregressive** — Generating a sequence one token at a time, each token
conditioned on all previous ones. Formally: `P(x) = Π P(x_t | x_<t)`.

**bf16 (bfloat16)** — 16-bit float with the same exponent range as fp32 but
fewer mantissa bits. Range matters more than precision for training, which is
why bf16 needs no loss scaling while fp16 does. Stage 09.

**Bias correction** — Dividing Adam's `m` and `v` by `1 − βᵗ` to undo their
zero initialisation. Makes step 1 move every weight by exactly ±lr. Stage 08.

**Bits per byte (bpb)** — Loss converted to bits and divided by the number of
bytes of text predicted, instead of tokens. Comparable across tokenizers,
which loss per token is not. Stage 06.

**BPE (Byte Pair Encoding)** — A tokenizer training algorithm: repeatedly find
the most frequent adjacent pair of symbols in the corpus and merge it into a
new symbol. Stage 01.

**Bradley-Terry model** — `P(a beats b) = σ(r_a − r_b)`: preferences explained
by hidden scores whose differences matter. The reward-model loss is its
negative log-likelihood. Stage 18.

**Capacity factor** — In MoE, the per-expert token buffer as a multiple of the
balanced share. Tokens over capacity are dropped (skip the layer). Stage 13.

**Catastrophic forgetting** — Fine-tuning on narrow data degrading what the
model knew before. Measured as pretraining val loss before and after.
Stage 16.

**Causal mask** — Setting attention scores to `-inf` for future positions, so
position t can only attend to positions ≤ t. What makes the model
autoregressive. Stage 04.

**Chat template** — The fixed format that turns a conversation into one token
sequence, with special tokens marking roles (ours: ChatML,
`<|im_start|>` / `<|im_end|>`). Stage 16.

**Chinchilla** — The 2022 result that compute-optimal training uses roughly 20
training tokens per parameter — far more data than earlier practice. Stage 10.

**Compute-optimal** — For a fixed compute budget, the model size and token
count that give the lowest loss. Chinchilla: about 20 tokens per parameter.
Stage 10.

**Contamination** — Benchmark problems or solutions present in the training
data, so a score measures memory rather than skill. Screened with long
n-gram overlap. Stage 22.

**Context length** — The maximum number of tokens the model can attend over in
one forward pass. Also called block size or sequence length.

**Continuous batching** — Rebuilding the decode batch every step, so finished
requests leave and new ones join immediately. Stage 23.

**Cosine schedule** — Learning rate that falls from its peak along half a
cosine curve to a floor (10% of peak here). Most of a run's final loss
improvement comes during this decay. Stage 08.

**Cross-entropy loss** — `−log p(correct token)`, averaged over positions. At
uniform guessing over `V` tokens it equals `ln V` (9.01 for our 8,196). The
training objective of every language model here. Stage 06.

**d_model / n_embd** — The width of the residual stream; the embedding
dimension that every block reads from and writes to.

**Decode (phase)** — Generating one token per step after the prompt is
processed. Reads every weight and the KV cache for little arithmetic, so it is
limited by memory bandwidth. Stage 07, Stage 23.

**Distillation** — Training a smaller student to match a larger teacher's
output distributions (soft targets), not just the true next token. Stage 21.

**Document masking** — A block-diagonal attention mask that stops tokens of one
packed document from attending to the previous document. The fix for packing's
cross-document contamination; used in long-context training and SFT. Stage 02.

**DPO (Direct Preference Optimization)** — Fine-tuning on preference pairs via a
closed-form loss that provably optimizes the same objective as RLHF, with no
reward model and no RL loop. Stage 19.

**Dropout** — Zeroing a random fraction `p` of activations during training
(survivors scaled by `1/(1-p)`), off at inference. Fights overfitting, so it
only helps when data repeats. Modern pretraining runs use `p = 0`. Stage 05.

**Elo** — A rating system from chess: each pairwise comparison moves both
players' ratings by an amount that depends on how surprising the result was.
Chatbot Arena ranks models this way from human votes. Stage 22.

**Embedding** — A lookup table mapping each token id to a learned vector.
Mathematically a one-hot vector times a matrix. Stage 03.

**Exact dedup** — Removing byte-identical copies by hashing content. Cheap, and
blind to a fork that changed one character. Stage 02.

**Expert** — One of several parallel FFN sub-networks in an MoE layer. Each token
is routed to only a few of them. Stage 13.

**Expert collapse** — In MoE training, the router sending almost all tokens
to one or two experts, which then get all the gradient while the rest stay
untrained. Stage 13.

**FFN / MLP** — The per-position feed-forward network inside a transformer
block, usually expanding width 4x and projecting back. Where most parameters
live. Stage 05.

**FlashAttention** — An attention implementation that tiles the computation so
the n×n attention matrix is never written to memory. Same math, far less memory
traffic. Stage 09.

**Format reward** — Partial reward for easy-to-check structure (it parses,
it defines the right function, it stops). Helps when correct answers are
rare. Stage 20.

**fp16 / loss scaling** — 16-bit float with a small range (≈6e-5 to 65,504).
Small gradients underflow to zero, so fp16 training multiplies the loss by a
large factor before backward and divides it out after. bf16 does not need
this. Stage 09.

**Gated linear unit (GLU)** — An MLP that makes two projections of the input
and multiplies them elementwise, one passed through an activation to act as a
per-unit gate on the other. SwiGLU is the SiLU-gated version. Stage 05.

**GELU** — `x · Φ(x)`, a smooth version of ReLU with a small dip below zero.
GPT-2 uses a tanh approximation of it (`gelu_new`). Stage 05.

**GQA (Grouped-Query Attention)** — Query heads are grouped and each group shares
one key/value head, shrinking the KV cache. MHA and MQA are its two extremes.
Stage 12.

**Gradient accumulation** — Summing gradients over several small forward/backward
passes before stepping the optimizer, to simulate a large batch on small memory.
Stage 08.

**Gradient clipping (global norm)** — If the norm of all gradients together,
viewed as one vector, exceeds a threshold (1.0), scale them all down to the
threshold. Keeps the direction, limits the step after a rare huge gradient.
Stage 08.

**Greedy decoding** — Always picking the highest-probability next token.
Deterministic, and prone to repetition loops. Stage 07.

**Group-wise quantization** — One scale per group of `g` consecutive
weights (32-128), so one outlier only damages its own group. Stage 21.

**GRPO (Group Relative Policy Optimization)** — An RL method that samples a group
of responses per prompt and uses their relative rewards as advantages, removing
the value network. Stage 20.

**Head** — One attention subspace. Multi-head attention splits `d_model` into
`n_head` heads of size `d_model / n_head`, each attending independently.

**Implicit reward** — In DPO, `β · log(π(y|x)/π_ref(y|x))`: the reward a
policy implies, by DPO's derivation. Stage 19.

**IsoFLOP curve** — Final loss against model size for runs that all use the
same compute. U-shaped; its minimum is the compute-optimal size. Stage 10.

**Jaccard similarity** — Overlap of two sets: `|A ∩ B| / |A ∪ B|`, from 0 (disjoint)
to 1 (identical). Near-dedup measures it over sets of word 5-grams. Stage 02.

**Jacobian** — The matrix of all partial derivatives of a vector function's
outputs with respect to its inputs. The backward pass multiplies gradients by
one per layer. Stage 05.

**Kernel fusion** — Combining several GPU operations into one kernel so the
data is read from memory once instead of once per operation. What
`torch.compile` does for chains of elementwise operations. Stage 09.

**KL divergence** — `Σ p log(p/q)`: how different distribution `p` is from
`q`. Zero when equal. Used to keep a fine-tuned policy near its reference.
Stage 19.

**KV cache** — Stored keys and values from previous decoding steps, so each new
token's attention does not recompute them. Turns O(n²) decoding work into O(n).
Stage 07.

**LayerNorm** — Normalising each token's vector on its own: subtract its mean,
divide by its standard deviation, then apply a learned gain `γ` and bias `β`.
Keeps a vector's shape, discards its offset and size. GPT-2. Stage 05.

**Leakage (data)** — Validation text that the model effectively saw during
training, so validation loss measures memorization rather than generalization.
For code, mostly forks and vendored copies under different repo names. Stage 02.

**Learned absolute positions** — A lookup table with one learned vector per
slot number (0, 1, 2, ...), added to the token vectors. GPT-2 uses 1,024 rows
and cannot run on longer input: there is no row for position 1,024. Stage 03.

**Length normalisation** — In log-likelihood scoring of multiple choices,
dividing a candidate's total log-probability by its length so short answers
are not favoured. Stage 22.

**Linear attention** — Attention with `exp(q·k)` replaced by `φ(q)·φ(k)`, so
the past can be kept as a fixed-size running sum: an RNN in disguise.
Stage 14.

**LLM-as-judge** — Using a strong model to compare or grade outputs. Biased
by position, length and self-preference. Stage 22.

**Load-balancing loss** — The MoE auxiliary loss `E · Σ f_i P_i` (fraction of
tokens routed to expert `i` times its mean router probability). Equals 1 when
balanced, `E` when collapsed. Stage 13.

**Logits** — The raw, unnormalized scores over the vocabulary that come out of
the final linear layer, before softmax.

**LoRA (Low-Rank Adaptation)** — Freeze the base weights and train a low-rank
update `B·A` instead, cutting trainable parameters by orders of magnitude.
Stage 17.

**Loss masking** — Setting the targets of prompt tokens to -100 so only the
response is trained on. Stage 16.

**LSH (Locality-Sensitive Hashing)** — Splitting a MinHash signature into bands
and bucketing documents by each band, so only documents likely to be similar are
ever compared. Turns an O(n²) all-pairs search into roughly O(n). Stage 02.

**Massive activations** — A few hidden dimensions in a trained transformer's
residual stream with values hundreds or thousands of times larger than the
rest (≈3,000 in GPT-2 small). They make float32 rounding errors visible and
break naive int8 quantization of activations. Stage 06, Stage 21.

**Memmap (memory-mapped file)** — A file mapped into the address space so the OS
loads pages only when touched. A 5 GB token array costs almost no RAM until read.
Stage 02.

**MFU (Model FLOPs Utilization)** — Achieved FLOP/s divided by the GPU's peak
FLOP/s. The honest measure of training efficiency. Stage 09.

**MinHash** — A fixed-size fingerprint of a set: for each of P hash functions,
keep the minimum hash over the set's elements. The fraction of positions where two
fingerprints agree is an unbiased estimate of their Jaccard similarity. Stage 02.

**MLA (Multi-head Latent Attention)** — DeepSeek's attention that caches a
small latent vector per token and rebuilds each head's keys and values from
it, plus a small shared RoPE key. Stage 14.

**Model card** — A short document published with a model: what it is, how it
was trained, how it scores, and what it should not be used for. Ours is
`MODEL_CARD.md`. Stage 24.

**MoE (Mixture of Experts)** — Replacing one FFN with many, and routing each
token to a few, so total parameters grow without per-token FLOPs growing.
Stage 13.

**Momentum** — A running average of recent gradients, followed instead of the
latest one. Cancels zigzags, accumulates consistent directions. Adam's `m`.
Stage 08.

**MQA (Multi-Query Attention)** — Attention where all query heads share one
key head and one value head. The smallest KV cache; some quality loss. GQA
with one group. Stage 12.

**Multi-head attention** — Running H smaller attentions side by side, each on
a `d = C / H` slice, then mixing their outputs with one `C × C` projection.
Same parameter count as one big head; H attention patterns instead of one.
Stage 04.

**Near-duplicate dedup** — Removing documents that are almost, but not exactly,
the same (forks, reformatted copies). Done with MinHash + LSH. Stage 02.

**Needle in a haystack** — A long-context test: hide a fact deep in a long
input and ask for it at the end. Shows retrieval failures that average loss
hides. Stage 14.

**NF4 (4-bit NormalFloat)** — A 4-bit format whose 16 levels sit at quantiles
of a normal distribution, matching how trained weights are distributed, with
one scale per block of 64. Stage 17.

**Non-embedding parameters** — Parameter count excluding the token and
position tables. Scaling laws use it because a table lookup costs memory but
almost no compute per token. GPT-2 small: 85.1M of 124.4M. Stage 06.

**NoPE (no positional encoding)** — A causal language model given no position
information at all. It still learns some sense of order, because the causal
mask lets token t see exactly t+1 tokens. Stage 03 measures how much it loses.

**NTK-aware scaling** — Extending RoPE to longer contexts by raising its base
`θ`, which slows the slow rotation pairs by the stretch factor while barely
changing the fast ones. Stage 12.

**One-hot vector** — A vector of zeros with a single 1, at the index of the
item it represents. Multiplying a one-hot by a matrix selects one row, which is
why an embedding layer is a linear layer on one-hot inputs. Stage 03.

**Output head / unembedding** — The final `C × V` matrix that turns each
position's hidden vector into one score (logit) per vocabulary token. Often
shares its weights with the embedding table (see weight tying). Stage 03.

**Overfitting** — Fitting the training data in ways that do not carry over
to new data, usually by memorising examples seen many times. Shows as
training loss falling while validation loss stalls or rises.

**Packing** — Concatenating documents into one stream separated by an
end-of-text token and cutting fixed-length windows from it, instead of padding
each document to length. No wasted compute on padding. Stage 02.

**PagedAttention** — Storing the KV cache in fixed-size blocks allocated on
demand through a block table, so cache memory is not reserved and wasted.
vLLM. Stage 23.

**pass@k** — The probability that at least one of `k` sampled programs
passes the tests, estimated without bias from `n ≥ k` samples as
`1 − C(n−c, k)/C(n, k)`. Stage 22.

**Permutation equivariance** — Shuffling the input shuffles the output the
same way and changes nothing else. Attention with no mask and no position
information has this property, so it cannot tell `a = b` from `b = a`. Stage 03.

**Perplexity** — `exp(cross-entropy loss)`. Intuitively, how many tokens the
model is effectively choosing between at each step. Lower is better.

**Position bias** — An LLM judge's tendency to prefer whichever answer it is
shown first (or second), regardless of quality. Controlled by judging each
pair in both orders. Stage 22.

**Post-norm** — Normalising *after* the residual add: `Norm(x + f(x))`. The
2017 transformer and BERT. Puts a norm on the residual path itself, which makes
deep stacks harder to train. Stage 05.

**PPO (Proximal Policy Optimization)** — RL algorithm with a clipped
objective that limits how far one batch can move the policy. RLHF's original
workhorse, with policy, reference, reward and value models. Stage 19.

**Pre-norm** — Normalizing the input to each sublayer rather than its output.
Required for stable training at depth. Stage 05.

**Prefill (phase)** — Processing the whole prompt in one forward pass to fill
the KV cache. Many tokens per weight read, so it is compute-bound. Stage 07.

**Previous-token head** — An attention head that puts most of its weight on
the token just before. Shows as a bright line one step below the diagonal of
its attention map. Stage 04.

**PTQ / QAT** — Post-training quantization (quantize trained weights; e.g.
GPTQ, AWQ) vs quantization-aware training (simulate quantization while
training). Stage 21.

**QK-norm** — RMSNorm applied to each head's queries and keys before the dot
product, keeping attention scores in a bounded range for stability.
Stage 12.

**QLoRA** — LoRA on top of a base model stored in 4-bit NF4, dequantized on
the fly. Stage 17.

**Quantization** — Storing numbers as low-bit integers plus a scale (and
maybe a zero-point): `w ≈ (q − z) · s`. Stage 21.

**Query, key, value** — The three vectors each token makes for attention: the
query is what it is looking for, the key is what it matches on, the value is
what it hands over when picked. Each is the token vector times its own learned
matrix. Stage 04.

**Rank collapse** — Every position's vector becoming the same as you go up a
stack of attention layers, so tokens can no longer be told apart. What actually
breaks when residual connections are removed (measured: cosine 1.000 between
positions by block 12). Stage 05.

**ReLU** — `max(0, x)`. Cheap; its gradient is exactly 0 for negative inputs,
so a unit can stop learning ("die"). Stage 05.

**Repetition penalty** — Lowering the logits of tokens already in the context
(divide positive logits, multiply negative ones, by ~1.1-1.3). A blunt fix
for loops that also discourages legitimate repetition, such as variable names
in code. Stage 07.

**Residual connection** — Adding a layer's output to its input, `x + f(x)`,
instead of replacing it. The layer learns a correction, and the gradient gets a
direct path past the layer. Stage 05.

**Residual stream** — The running sum that each block reads from and adds to.
The "highway" that lets gradients reach early layers.

**Residual-scaled init** — Initialising the layers that write into the
residual stream (`c_proj`) with std `0.02/√(2L)`, so the summed contribution
of all `2L` sublayers at init has the size of one. GPT-2's choice. Stage 06.

**Reward hacking** — A policy reaching high reward through flaws in the
reward (model or verifier) rather than by doing the task. Stage 18.

**Reward model** — A language model with a scalar head that scores a
(prompt, response) pair; trained on preference pairs. Stage 18.

**RLVR (RL with verifiable rewards)** — Reinforcement learning where the
reward is computed by a program (tests, answer checker) instead of a learned
model. Stage 20.

**RMSNorm** — Normalization by root-mean-square only, without subtracting the
mean. Cheaper than LayerNorm with equal quality. Stage 05.

**RoPE (Rotary Position Embedding)** — Encoding position by rotating query and
key vectors by an angle proportional to position, making attention scores depend
on relative distance. Stage 03.

**Router / gate** — The small network in an MoE layer that decides which experts
each token goes to. Stage 13.

**Router z-loss** — Penalty on the square of `logsumexp(router logits)` that
keeps router logits small and the softmax numerically stable. Stage 13.

**SFT (Supervised Fine-Tuning)** — Training a base model on
instruction/response pairs so it answers instead of merely continuing text.
Stage 16.

**SGD (stochastic gradient descent)** — `θ ← θ − lr · g`, with `g` estimated
on one batch. The baseline optimizer. Stage 08.

**Shared expert** — An MoE expert every token always uses, alongside its
routed experts. Holds common knowledge once. DeepSeek-MoE. Stage 13.

**Shift (next-token targets)** — Targets are the inputs moved left by one:
input `ids[:, :-1]`, target `ids[:, 1:]`. Together with the causal mask it
turns one window of `T+1` tokens into `T` next-token training examples.
Stage 06.

**Shingle** — An overlapping window of n consecutive words (we use n = 5). A
document's set of shingles is what near-dedup compares. Stage 02.

**SiLU (swish)** — `x · sigmoid(x)`. Smooth, with a dip below zero; the gate
in SwiGLU. Stage 05.

**Sinusoidal positions** — Position encoded as the sines and cosines of
`p·ω_i` at many speeds `ω_i`, like a row of clocks running at different rates.
No parameters; moving by k positions is a fixed rotation. From the 2017
transformer paper. Stage 03.

**Sliding-window attention** — Each token attends only to the last `W`
tokens. Bounded cache and compute; information still travels further across
layers. Stage 12.

**Softmax** — Turns a vector of scores into a probability distribution:
`exp(x_i) / Σ exp(x_j)`.

**Softmax saturation** — When the inputs to softmax are far apart, one weight
approaches 1 and the rest 0. Softmax's slope is then near zero everywhere, so
almost no gradient flows back. Dividing attention scores by `√d` prevents it.
Stage 04.

**Speculative decoding** — A small draft model proposes several tokens; the
big model checks them all in one pass; a rejection-sampling rule keeps the
output distribution exactly the big model's. Stage 15.

**State-space model (SSM)** — A layer that carries a hidden state through a
learned linear recurrence `h_t = a·h_{t−1} + b·x_t`, computable step by step
or all at once by a scan. S4, Mamba. Stage 14.

**SwiGLU** — A gated feed-forward variant: `(W1·x ⊙ silu(W3·x))·W2`. Standard in
modern models. Stage 05.

**Temperature** — Dividing logits before softmax. <1 sharpens the distribution,
>1 flattens it. Stage 07.

**TF32** — A tensor-core mode that multiplies fp32 inputs using only 10
mantissa bits. Faster matmuls with one line of code, slightly less precise.
Stage 09.

**Token** — The unit the model actually sees: a sub-word chunk of bytes produced
by the tokenizer. Not a word, not a character. Stage 01.

**Top-k / top-p sampling** — Truncating the next-token distribution to the k most
likely tokens, or to the smallest set whose probability sums to p, before
sampling. Stage 07.

**TTFT / TPOT** — Time to first token (queueing + prefill) and time per
output token (one decode step). The two latencies users feel. Stage 23.

**Unigram loss** — The cross-entropy of predicting every token from its overall
frequency alone, ignoring all context. The loss of a model that learned
nothing about context: 6.86 on our code validation set. Stage 05.

**Warmup** — Ramping the learning rate up from ~0 over the first few hundred
steps, because Adam's variance estimates are unreliable early. Stage 08.

**Weight decay** — Shrinking every weight slightly towards zero each step.
Applied to matrices, not to biases or norm gains. Stage 08.

**Weight tying** — Sharing one matrix between the input embedding and the output
projection. Saves `vocab × d_model` parameters. Stage 03 (measured), Stage 06.
