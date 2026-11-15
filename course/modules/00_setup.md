# Module 0: Where the memory goes

**You will build** the arithmetic that decides what you can train: how many
parameters a layer has, how many bytes training them costs, the largest model
your GPU can hold, and a function that measures it on the real allocator.

**Time:** 1-2 hours. **Hardware:** any; the measurement needs a CUDA GPU
(Colab's T4 works). **Book chapter:** [1, What a language model actually computes](../../docs/00_setup.md).

---

## 1. Why

Every later module runs into the same wall: GPU memory. A model that does not
fit does not train, and the error arrives after the data is loaded and the
first batch is half done. The rule that predicts it is simple enough to do in
your head, and this module makes you prove it with the allocator before you
trust it.

What breaks without it: forget that Adam keeps two extra numbers per weight
and your estimate is half the real cost. On the book's GPU, that is the
difference between "a 350M model fits" and "it does not".

## 2. Read

Chapter 1 of the book:

- §1 *What a language model actually is*: the one idea everything else
  serves, predicting the next token.
- §2 *Where the VRAM goes*, including the worked 50M example.
- §3 *PyTorch essentials*: skim if you have used PyTorch; read carefully if
  not. Every later module uses these operations.

## 3. Implement

Open `course/module00/budget.py`. Four functions are blank:

| # | Function | What it does |
|---|---|---|
| 1 | `linear_params` | parameters in one `nn.Linear` |
| 2 | `training_bytes` | weights + gradients + optimizer state, in bytes |
| 3 | `max_params` | the largest model whose fixed cost fits, after a reserve |
| 4 | `measure_training_bytes` | the same thing, read from the CUDA allocator |

## 4. Check

```bash
uv run python course/check.py 0        # Colab: !python course/check.py 0
```

Five tests. They check your counts against PyTorch's own, the book's 192 MB
toy model, the cost of three optimizers, the largest model on the book's
7,797 MB GPU, and (on a GPU) that your measurement agrees with your
prediction to within 20%.

## 5. Run

```bash
uv run python stages/00_setup/verify_env.py
```

A few seconds. It trains one step of a 12-layer, 12.6M-parameter toy model
in fp32 with AdamW and reads the allocator at each point. The book's run:

| | book (RTX 4060 Laptop) |
|---|---|
| predicted fixed cost | 192.0 MB |
| measured total | 208.3 MB (8.5% over) |
| weights | 48.0 MB |
| gradients + activations | 64.3 MB |
| AdamW state | 96.0 MB |
| peak, including activations | 256.3 MB |

The weights and AdamW rows are pure arithmetic and should match exactly on
any GPU. The gradient row is 48 MB of gradients plus a few MB of activations
still held, so it can move slightly. The second table it prints, "what fits
on this GPU", depends on your card: on a 16 GB T4 a 350M model's fixed cost
is about a third of memory, on the book's 8 GB card about two thirds.

## 6. Break it

The script prints the broken estimate itself: "forgot AdamW?" predicts 96 MB
for a run that really takes 208 MB. Then make it break for real: in a Python
shell, build a model your `max_params` says is just too big for your GPU and
call `measure_training_bytes` on it. Read the `OutOfMemoryError` message: it
tells you how much was asked for and how much was free, which is the same
arithmetic you just wrote.

## 7. Check yourself

<details><summary>1. Training a 50M-parameter model in fp32 with AdamW. What is the fixed memory cost, before activations?</summary>

50M × 16 bytes = 800 MB (about 763 MiB): 200 MB of weights, 200 MB of
gradients, 400 MB of Adam state.
</details>

<details><summary>2. Why does the AdamW state not appear until after the first <code>opt.step()</code>?</summary>

PyTorch's Adam creates its two state tensors lazily, the first time it
updates a parameter. Before that step the memory looks like plain SGD's.
</details>

<details><summary>3. The book's "8 GB" GPU reports 7,797 MB. Where did the rest go, and why reserve another 600 MB?</summary>

The card has 8,192 MiB, but about 390 MB is reserved before PyTorch sees any
of it, so `total_memory` reports 7,797. Then the CUDA context, the kernels
and the allocator's slack take several hundred MB more before your first
tensor. Budget against what the runtime reports, not what the box says.
</details>

<details><summary>4. Your activations grow with which two numbers, and why does the 16-bytes rule ignore them?</summary>

Batch size and sequence length (and the model's width and depth). The rule
counts what exists for every parameter regardless of the batch; activations
are the part you control by changing the batch, which is why later modules
use gradient accumulation and activation checkpointing.
</details>

---

**Stuck?** The solution is
[`course/module00/budget.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/course/module00/budget.py)
in the main repository; the book's version of the same arithmetic is inline in
[`stages/00_setup/verify_env.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/00_setup/verify_env.py).

**Next:** Module 1, the tokenizer: how text becomes the integers a model reads.
