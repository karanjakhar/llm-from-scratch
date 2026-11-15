# Module 3: Turning ids into vectors, and telling the model where they are

**You will build** the token embedding table, the three ways of telling a
model where each token sits (learned positions, sinusoidal positions and
RoPE), and the two layouts of RoPE that real model weights use.

**Time:** 3-4 hours, plus about 6 minutes of training on the book's GPU. **Hardware:** GPU.
**Book chapter:** [5, Turning ids into vectors and position](../../docs/03_embeddings_position.md).

---

## 1. Why

The tokenizer hands the model integers. An integer has no meaning a network
can compute with, so each id looks up its own row of learned numbers: its
embedding. That is half the job. The other half is order: attention, which
you build in module 4, treats its input as a set. Without position
information, "a = b" and "b = a" look identical to it.

What breaks without it: the book trained the same small model five ways. With
no position information at all, held-out code loss was 3.36; with RoPE, 2.52.
Add sinusoidal positions without scaling the token embeddings and it is
worse than having no positions: 4.71, because the position signal is 35 times
the size of the token signal and drowns it.

## 2. Read

Chapter 5 of the book:

- §1 *From ids to vectors*: the lookup, and why it is a matrix product with a
  one-hot vector.
- §3 *Why position is needed at all*.
- §4-§6: learned, sinusoidal and RoPE. The RoPE section has a worked example
  with small numbers; do it by hand.
- §6 *Two layouts, one idea (a trap)*: why the same rotation can be stored two
  ways, and why loading weights with the wrong one fails silently.

## 3. Implement

Open `stages/03_embeddings/embeddings.py`. Seven blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `TokenEmbedding.forward` | id → its row |
| 2 | `LearnedPosition.forward` | the first T rows of a position table, with a hard wall |
| 3 | `frequencies` | the speed of each sin/cos "clock" |
| 4 | `sinusoidal_table` | the 2017 transformer's fixed positions |
| 5 | `rope_angles` | position × speed, for RoPE |
| 6 | `rope_interleaved` | rotate pairs (0,1), (2,3), … (Meta's layout) |
| 7 | `rope_half` | rotate pairs (i, i + d/2) (Hugging Face's layout) |

`rope_complex` is left complete: it is the same rotation written with complex
numbers, and a test checks your `rope_interleaved` equals it exactly.

## 4. Check

```bash
uv run python course/check.py 3
```

22 tests. Among them: your lookup equals `nn.Embedding` and the one-hot
product; only the rows of tokens in the batch receive a gradient; attention
without position cannot see order (shuffle the input and the output is
shuffled the same way, nothing else changes); with RoPE the
attention score between two tokens depends only on how far apart they are;
your `rope_half` matches Hugging Face's Llama implementation.

## 5. Run

```bash
uv run python stages/03_embeddings/run.py ablation --variants none learned sinusoidal sinusoidal_scaled rope
```

Five small models (4 layers, 256 wide), 2,000 steps each, identical except
for position. About 75 seconds per model on the book's GPU. The book's
numbers (seed 0, on the full 12-shard data):

| position scheme | held-out code loss, book |
|---|---|
| none | 3.363 |
| learned (GPT-2) | 2.685 |
| sinusoidal, unscaled | 4.708 |
| sinusoidal, tokens scaled by √C | 2.633 |
| RoPE | 2.523 |

Your data is the two-shard subset from module 2, with its own validation
set, so your losses will not equal these. On the two-shard data, run while writing this course,
`none` scored 3.777 and `rope` 2.905: both about 0.4 above the book, because
the validation set is a different slice of code, but the gap between
them was 0.87 against the book's 0.84. The ranking is what to
compare: RoPE best, unscaled sinusoidal worst, no position far behind the
rest. Your run overwrites `results/03_embeddings/ablation.json`;
`git diff results/03_embeddings/` shows your numbers beside the book's.

## 6. Break it

The ablation above is the break: the `none` row is this module's component
removed. Two more to try:

1. Train with `--variants rope_untied`. The output layer gets its own table
   instead of sharing the embedding (2.10M more parameters). The book: worse,
   2.606 against 2.523. Chapter §2 explains why sharing helps a small model.
2. In `rope_half`, swap the signs (`a*cos + b*sin`, `-a*sin + b*cos`) and run
   the tests. One rotation direction is as good as the other for training, so
   which test fails, and why?

## 7. Check yourself

<details><summary>1. An embedding lookup is a matrix product with a one-hot vector. Why does that mean only some rows get a gradient each step?</summary>

The one-hot vector is zero everywhere except the token's own position, so the
gradient with respect to every other row is zero. Rows of tokens absent from
the batch do not move.
</details>

<details><summary>2. Learned positions have a hard wall at <code>max_len</code>. What happens to sinusoidal positions past the training length, and to RoPE?</summary>

Sinusoidal and RoPE can compute any position, so there is no wall. But the
model never saw those angles in training: the book measured loss on
positions 256-511 after training at 256 and every scheme got worse there
(chapter §8). "Can compute" is not "has learned".
</details>

<details><summary>3. With RoPE, why does the score between query position m and key position n depend only on m − n?</summary>

Rotating q by angle mθ and k by nθ, their dot product equals the dot product
of the unrotated vectors with a rotation of (m − n)θ between them. The
absolute positions cancel.
</details>

<details><summary>4. Why did unscaled sinusoidal positions do worse than no positions at all?</summary>

The token embeddings start with standard deviation 0.02; the sinusoidal
values are between −1 and 1. Added together, position is about 35 times
larger than token identity, so the model starts out barely able to see which
token is which. Scaling the tokens by √C fixes it (2.633).
</details>

---

**Stuck?** Solution:
[`stages/03_embeddings/embeddings.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/03_embeddings/embeddings.py).

**Next:** Module 4, attention: how each token reads the others.
