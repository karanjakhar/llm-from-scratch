# Module 5: The transformer block

**You will build** the pieces a transformer block is made of: LayerNorm and
RMSNorm, the GELU and SiLU activations, the classic MLP and Llama's SwiGLU.
The block itself, with every design choice as a switch, is provided, so you
can turn each piece off and measure what it was doing.

**Time:** 3 hours, plus about 10 minutes of training on the book's GPU. **Hardware:** GPU.
**Book chapter:** [7, The transformer block](../../docs/05_transformer_block.md).

---

## 1. Why

A block is attention (tokens exchange information) followed by an MLP (each
token processes what it gathered on its own), each wrapped in a
normalisation and a residual connection. Stack a few dozen and you have a
GPT. The wrapping matters as much as the contents: it is what lets the
signal and the gradient survive 24 or 96 layers.

What breaks without it: remove the residual connection and the book's 24-layer
model learned nothing. Its loss stayed at 6.856, the loss of a model that
only knows how often each token appears (6.860). By layer 12 every position
held the same vector: cosine similarity 1.000.

## 2. Read

Chapter 7 of the book:

- §1 *The picture*.
- §2 *The residual connection*: why the gradient needs a road.
- §3 *Normalisation*: LayerNorm, then RMSNorm.
- §4 *Pre-norm vs post-norm*.
- §5-§6 *The MLP* and *Activation functions*.

## 3. Implement

Open `stages/05_block/block.py`. Seven blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `LayerNorm.forward` | per token: mean 0, variance 1, then scale and shift |
| 2 | `RMSNorm.forward` | per token: divide by the root mean square, then scale |
| 3 | `gelu_tanh` | GPT-2's GELU |
| 4 | `gelu_exact` | GELU through erf |
| 5 | `silu` | x · sigmoid(x) |
| 6 | `MLP.forward` | C → 4C → C |
| 7 | `SwiGLU.forward` | Llama's gated MLP |

Read `Block` below them before you start: `_sublayer` is the whole
difference between pre-norm and post-norm, in four lines.

## 4. Check

```bash
uv run python course/check.py 5
```

23 tests. Your norms must match `nn.LayerNorm`, `nn.RMSNorm` and Hugging
Face's `LlamaRMSNorm` in outputs and gradients; your SwiGLU must match
`LlamaMLP`; the whole block must match Hugging Face's `GPT2Block` with the
same weights. One test shows that a LayerNorm using `x.var()` (which divides
by C − 1) is wrong by a small amount that only a reference comparison catches.

## 5. Run

```bash
uv run python stages/05_block/run.py grads --B 8
```

About 6 seconds. `--B 8` matters: it is the batch the book used, and the
default of 32 runs out of memory on an 8 GB GPU in full fp32 at 24 layers. No training: it builds untrained 24-layer stacks three ways
(pre-norm, post-norm, no residual) and measures the gradient at the first and
last block and how alike the token vectors become. The book's numbers:

| | grad norm, block 1 / block 24 | cosine between positions after block 24 |
|---|---|---|
| pre-norm | 2.21 / 0.38 | 0.69 |
| post-norm | 1.32 / 0.12 | 0.03 |
| no residual | 3.65 / 2.63 | **1.000** |

A rerun while writing this course gave 2.27 / 0.385, 1.28 / 0.123 and
3.7 / 2.7, with cosines 0.697, 0.036 and 1.000. Without the residual, the
gradient is fine; it is the signal that dies: every
position ends up holding the same vector, so there is nothing left to learn
from. The chapter's §10a calls this the surprise of the stage: the book
expected vanishing gradients and found merging tokens instead.

## 6. Break it

Train it:

```bash
uv run python stages/05_block/run.py depth --variants pre_L4 nores_L4
```

| run | held-out code loss, book |
|---|---|
| `pre_L4` (4 pre-norm blocks) | 2.685 |
| `nores_L4` (no residual) | 6.856 |

On the two-shard data, run while writing this course: 2.698 and 6.852,
about 4 minutes per model on the book's GPU. 6.86 is the frequency floor: the model learned how common each
token is and nothing else. If you have time, add `post_L4`: post-norm also
sat at 6.856 in the book, and chapter §10b is a detective story about why
(the answer turned out to be the size of the embeddings, not post-norm
itself).

## 7. Check yourself

<details><summary>1. LayerNorm normalises over which dimension, and why not over the batch?</summary>

Over the C features of each token separately. Normalising over the batch
(BatchNorm) would make a token's output depend on the other sequences in the
batch, and behave differently with batch size 1 at inference.
</details>

<details><summary>2. RMSNorm skips the mean subtraction and the bias. What does that buy, and what did the book measure for it?</summary>

Fewer operations and parameters. In the book's 4-layer sweep it scored
2.712 ± 0.027 against LayerNorm's 2.689 ± 0.012, a tie within noise; its
speed advantage disappears once the norm is a fused kernel (chapter §10e).
</details>

<details><summary>3. In pre-norm, <code>x = x + f(norm(x))</code>. Why does that keep a 24-layer stack trainable at initialisation?</summary>

The x term passes straight through every block, so the gradient has a path
from the loss to the first layer that does not go through any f. Each block
only adds an edit to the stream instead of replacing it.
</details>

<details><summary>4. SwiGLU has three weight matrices, the GELU MLP two. How does Llama keep the parameter count the same?</summary>

It shrinks the hidden width from 4C to 8C/3: three matrices of C × 8C/3 cost
what two of C × 4C cost.
</details>

---

**Stuck?** Solution:
[`stages/05_block/block.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/05_block/block.py).

**Next:** Module 6, GPT: stack the blocks, add an output layer, and load
OpenAI's real GPT-2 weights into your model.
