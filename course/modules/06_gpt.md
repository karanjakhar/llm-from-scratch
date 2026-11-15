# Module 6: Assembling GPT, and loading OpenAI's weights into it

**You will build** the whole GPT-2 model from your parts, its parameter count
by hand, its initialisation, and a loader that copies OpenAI's released GPT-2
weights into it. Then you prove your model is GPT-2 by matching Hugging
Face's implementation number for number.

**Time:** 3-4 hours. **Hardware:** GPU recommended (the comparison downloads
GPT-2, 550 MB). **Book chapter:** [8, Assembling GPT](../../docs/06_gpt_model.md).

---

## 1. Why

Everything so far was a part. This module puts them together: embeddings,
a stack of blocks, a final norm and an output layer that scores every token
in the vocabulary. The test of whether you understood the parts is
unforgiving: OpenAI's 124M-parameter GPT-2 weights must load into your model
and produce the same outputs as the reference implementation.

What breaks without the key idea: train without shifting the targets by one
position and the model learns to copy its input. The book's run reached a
training loss of 0.0001 and a next-token validation loss of 21.86, against
4.09 for the same model trained correctly. A perfect score on the wrong task.

## 2. Read

Chapter 8 of the book:

- §1 *The picture* and §2 *The shift*.
- §3 *Counting parameters by hand*: GPT-2 small is 124,439,808, and you
  will get there with arithmetic.
- §4 *Initialisation*, including why the residual-writing layers start smaller.
- §5 *Names, so real weights fit*: the three traps in GPT-2's checkpoint.

## 3. Implement

Open `stages/06_gpt/gpt.py`. Six blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `CausalSelfAttention.forward` | module 4's attention, under GPT-2's names |
| 2 | `Block.forward` | pre-norm: attention, then MLP, each added to the stream |
| 3 | `GPT._scale_residual_init` | shrink the layers that write to the stream |
| 4 | `GPT.forward` | ids → logits, and the loss if targets are given |
| 5 | `load_gpt2_state` | copy OpenAI's tensors in, refusing anything that does not fit |
| 6 | `count_params_by_hand` | the parameter count, from the config alone |

This file imports your `attention` from module 4 and your `LayerNorm` and
GELU from module 5.

## 4. Check

```bash
uv run python course/check.py 6
```

20 tests. Your hand count must equal PyTorch's for several shapes and give
124,439,808 for GPT-2 small; an untrained model's loss must start at ln V;
the tied output layer must be the same tensor as the embedding; your loader
must fail loudly on a missing or extra tensor; and, the big one, GPT-2 loaded
into your model must match Hugging Face's `GPT2LMHeadModel`.

## 5. Run

```bash
uv run python stages/06_gpt/run.py oracle
```

About 45 seconds. It loads GPT-2 into your model and into Hugging Face's,
runs both on 200 validation windows and reports the largest difference in any
output score, then GPT-2's compression of our code and English (bits per
byte: lower is better).

| | book |
|---|---|
| largest logit difference, fp32 (code) | 0.055 |
| GPT-2 bits per byte, code / English | 1.394 / 1.064 |

On the two-shard data, run while writing this course: 0.048, and
1.318 / 1.057 bits per byte. The difference is not zero and not the
book's to the digit, because the 200 windows come from your validation set;
what must hold is that it stays around 0.05 rather than 1 or 10. A difference of 0.055 sounds like a bug and is not: GPT-2's
residual stream reaches values near 3,000, and float32 keeps about 7
significant digits, so 0.055 is rounding. In float64 the book measured
2 × 10⁻¹¹. Chapter §6 walks through that check; it is a good habit to copy.

## 6. Break it

```bash
uv run python stages/06_gpt/run.py noshift
```

About 2.5 minutes: the 4-layer model trained with targets equal to inputs
instead of shifted by one, and the same model trained correctly for
comparison.

| | book |
|---|---|
| training loss after 1,000 steps (copy task) | 0.0001 |
| validation loss on real next-token prediction | 21.86 |
| the same model trained with the shift | 4.09 |

On the two-shard data, run while writing this course: 0.0006, 21.41 and
4.13. It also prints what the model writes after `def add(a, b):`. Compare the
two samples.

## 7. Check yourself

<details><summary>1. GPT-2 small has 124,439,808 parameters. Roughly what share is the token embedding, and why does that matter for a small model?</summary>

50,257 × 768 = 38.6M, about 31%. In a small model the vocabulary table is a
large fraction of the budget, which is why this project uses an 8,192-token
vocabulary and ties the output layer to the embedding.
</details>

<details><summary>2. Why does an untrained model's loss start at ln V, and what does it tell you if yours starts at 30?</summary>

Small initial weights make every logit near zero, so the predicted
distribution is uniform over V tokens, and the cross-entropy of a uniform
guess is ln V (9.01 for V = 8,196). A much larger starting loss means the
initial outputs are far from uniform: an initialisation bug.
</details>

<details><summary>3. GPT-2's checkpoint stores some weights transposed. Why, and why would loading them untransposed not raise an error for every layer?</summary>

OpenAI used a Conv1D layer that stores (in, out), where nn.Linear stores
(out, in). For square matrices (C × C) the shapes match either way, so only
a numerical comparison catches it; for the 3C and 4C layers the shape check
does.
</details>

<details><summary>4. The residual-writing layers start with std 0.02 / √(2L). What would happen at 48 layers without that?</summary>

2L layers each add a random edit to the stream, so its size grows with depth.
The book measured the stream's size after the last block at initialisation:
0.93 at 48 layers with plain init, 0.09 with the scaled init.
</details>

---

**Stuck?** Solution:
[`stages/06_gpt/gpt.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/06_gpt/gpt.py).

**Next:** Module 7, making it talk: sampling and the KV cache.
