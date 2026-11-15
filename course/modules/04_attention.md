# Module 4: Attention, from one head up

**You will build** scaled dot-product attention, the causal mask that stops a
token reading the future, and multi-head attention with the reshape that most
attention bugs hide in.

**Time:** 3-4 hours. **Hardware:** any for the tests and the first experiment;
GPU for the training one. **Book chapter:** [6, Attention, from one head up](../../docs/04_attention.md).

---

## 1. Why

Attention is how a token gathers information from the tokens before it. Each
token asks a question (its query), every earlier token offers a label (its
key) and a payload (its value), and the answer is a weighted average of
payloads, weighted by how well each label matches the question. It is a
dictionary lookup that is allowed to be fuzzy.

What breaks without the details: drop the division by √d and, with a 256-wide
head, the book's model collapsed onto about 2.5 tokens per lookup and its loss
rose from 2.70 to 3.64, a bigger loss than removing position information
entirely. Get the head reshape wrong and the output has the right shape and
the wrong contents; no error is raised.

## 2. Read

Chapter 6 of the book:

- §1 *The idea*, with the worked example in small numbers.
- §2 *The formula* and §3 *Why divide by √d*.
- §4 *The causal mask*: why −∞ and not 0.
- §5 *Many heads instead of one*: the reshape, drawn.

## 3. Implement

Open `stages/04_attention/attention.py`. Three blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `causal_mask` | which keys each query may see |
| 2 | `attention` | softmax(q kᵀ / √d) v, with the mask |
| 3 | `MultiHeadAttention.forward` | split into heads, attend, merge back |

## 4. Check

```bash
uv run python course/check.py 4
```

12 tests. Your attention must equal PyTorch's fused
`F.scaled_dot_product_attention` in outputs **and** gradients, your multi-head
layer must equal `nn.MultiheadAttention` with the same weights, and changing a
future token must leave every earlier output unchanged. One test builds the
wrong reshape on purpose and checks it gives the right shape and the wrong
answer, so you can see the trap.

## 5. Run

```bash
uv run python stages/04_attention/run.py variance
```

About 10 seconds; no training, no data. Random queries and keys with head
size d from 16 to 1,024, with and without the √d. The book's numbers:

| d | score std, scaled / unscaled | top weight, unscaled | queries with ~no gradient, unscaled |
|---|---|---|---|
| 16 | 1.00 / 4.00 | 0.494 | 0% |
| 64 | 1.00 / 7.98 | 0.740 | 2% |
| 256 | 1.00 / 16.03 | 0.881 | 14% |
| 1024 | 1.00 / 32.04 | 0.940 | 41% |

These are from random numbers with a fixed seed, so they should match to the
digits shown. Without the √d, the scores grow like √d, softmax piles almost
all the weight on one key, and at d = 1,024 two in five queries get no
gradient to learn from.

## 6. Break it

Train it, on your data:

```bash
uv run python stages/04_attention/run.py ablation --variants d256 d256_noscale --seeds 0
```

One head of size 256, with and without √d, 4,000 steps each, about
140 seconds per model on the book's GPU.

| run | held-out code loss, book (seed 0) |
|---|---|
| `d256`, scaled | 2.702 |
| `d256_noscale` | 3.637 |

On the two-shard data, run while writing this course: 2.716 and 3.659,
within 0.02 of the book's. Then run `run.py ablation --variants d64 d64_noscale --seeds 0`.
The book found that with small heads, dropping √d *helped* slightly (2.655
against 2.686). Chapter §9c explains why, and it is the most surprising
result in the chapter.

## 7. Check yourself

<details><summary>1. Why is the mask applied with −∞ rather than by setting the future scores to 0?</summary>

Softmax exponentiates. e^0 = 1, so a zero score still gets weight; e^−∞ = 0
exactly, so the future gets none and the remaining weights still sum to 1.
</details>

<details><summary>2. Dot products of two random d-dimensional vectors with unit-variance entries have what standard deviation, and what does dividing by √d do to it?</summary>

√d: the dot product sums d independent products of variance 1. Dividing by
√d brings it back to 1 for every head size, which is the "scaled std 1.00"
column above.
</details>

<details><summary>3. 41% of queries got no gradient at the scores at d = 1,024, yet the gradient reaching the query was larger unscaled (63 against 3.1). How?</summary>

The gradient at q is the score gradient multiplied by k. Unscaled, k enters
at full size instead of divided by √d, and that factor outweighs the collapse
for the queries that are not saturated. The gradient is not vanishing; it is
concentrated on a few queries while the rest learn nothing (chapter §9a).
</details>

<details><summary>4. A wrong reshape <code>view(B, H, T, d)</code> instead of <code>view(B, T, H, d).transpose(1, 2)</code> gives the right shape. What does it compute?</summary>

It cuts the flat memory in the wrong order, so a "head" mixes pieces of
different tokens. Every shape check passes; only a numerical comparison with
a reference catches it.
</details>

---

**Stuck?** Solution:
[`stages/04_attention/attention.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/04_attention/attention.py).

**Next:** Module 5, the transformer block: attention plus an MLP, normalised
and stacked with a residual connection.
