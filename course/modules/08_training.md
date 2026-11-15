# Module 8: The training loop

**You will build** the optimizers (SGD, momentum, and AdamW with decoupled
weight decay), the warmup-then-cosine learning-rate schedule, and gradient
clipping, then train a 10M-parameter model on children's stories to prove the
loop works.

**Time:** 3 hours, plus about 20 minutes of training on the book's GPU. **Hardware:** GPU.
**Book chapter:** [10, The training loop](../../docs/08_training_loop.md).

---

## 1. Why

The model is a function with millions of knobs. Training is turning them,
a little at a time, in whatever direction lowers the loss on the next batch.
The direction comes from the gradient; how far to turn each knob is the
optimizer's job, and it is where most training failures start.

What breaks without the details: on the same small model and data, the
book's plain SGD reached a loss of 5.62 in 1,500 steps, Adam 3.61, and Adam
with the old way of doing weight decay (folded into the loss) 6.70, barely
better than a model that only knows how often each token appears (6.86). Same gradients each time; only the update
rule changed.

## 2. Read

Chapter 10 of the book:

- §1 *The loop in one picture*.
- §2 *From SGD to Adam*, a-d. AdamW's "decoupled" weight decay is the
  subtle part.
- §4 *The learning-rate schedule* and §5 *Gradient clipping*.
- §7 *Sanity checks before trusting a run*, especially overfitting one batch.

## 3. Implement

Open `stages/08_training/optim.py`. Four blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `SGD.step` | step downhill, with optional momentum |
| 2 | `AdamW.step` | a per-weight step size, bias correction, decoupled decay |
| 3 | `lr_schedule` | linear warmup, then cosine down to 10% |
| 4 | `clip_grad_norm` | cap the length of the whole gradient, keep its direction |

## 4. Check

```bash
uv run python course/check.py 8
```

10 tests. Your AdamW must track `torch.optim.AdamW` over 25 steps (the book
measured a largest difference of 2.3 × 10⁻⁴ in float32 and 4.2 × 10⁻¹³ in
float64: the float32 gap is rounding, as the float64 run proves); its first
step must move every weight by exactly the learning rate; L2-in-the-loss must
differ from decoupled decay; your clipping must match PyTorch's; gradient
accumulation over 4 micro-batches must equal one batch 4 times larger; and a
model must be able to overfit a single batch.

## 5. Run

Your optimizers, racing on the same model and data:

```bash
uv run python stages/08_training/run.py optimizers
```

A 4-layer model trained for 1,500 steps five times, once per update rule,
all using your `SGD`, `AdamW` and `lr_schedule`. About 7.5 minutes on the book's GPU.

| update rule | held-out code loss, book |
|---|---|
| SGD | 5.615 |
| SGD + momentum | 4.310 |
| Adam | 3.609 |
| AdamW (decoupled decay) | 3.614 |
| Adam + L2 decay in the loss | 6.695 |

On the two-shard data, run while writing this course: 5.653, 4.408,
3.731, 3.738 and 6.691: each 0.04-0.12 above the book except the L2 run,
and the same order. Same gradients every time; only the update changed.

From here on, the course trains with the library's loop
(`llmfs/train/trainer.py`), which uses PyTorch's fused AdamW: the algorithm
you just wrote and matched, made fast. Two checks of that loop:

```bash
uv run python stages/08_training/run.py overfit
```

About 10 seconds: one batch of 8 × 256 tokens, trained until memorised. The
book's model went below a loss of 1 at step 72, below 0.1 at 96 and below
0.01 at 129, and a rerun while writing this course gave the same three
numbers. The batch comes from your data, so the
counts depend on which 2,048 tokens it holds: on the two-shard data it took
53, 74 and 111 steps. Any loop that works gets below 0.01 within a few
hundred steps. A loop that cannot memorise 2,048 tokens cannot
learn from 49 million: never skip this check.

Then the smoke test. It needs the TinyStories data (once):

```bash
uv run python stages/02_data/download_extras.py      # if you have not already
uv run python stages/08_training/prepare_tinystories.py
uv run python stages/08_training/run.py tinystories
```

A 10.1M-parameter model trained on 49M tokens of children's stories. The book
took 11.5 minutes and reached a validation loss of 1.510. A rerun while writing this course: 1.5104 in 10.9 minutes. Read the
samples it prints at each checkpoint: at the start they are noise, then
words, then sentences. TinyStories is used here because a broken loop shows
in the samples within minutes; it is a test, not a goal.

## 6. Break it

```bash
uv run python stages/08_training/run.py ablate
```

The same 6-layer model at a high learning rate (3 × 10⁻³), with and without
warmup and clipping:

| run | held-out code loss, book |
|---|---|
| warmup + clipping | 2.946 |
| no clipping | 3.028 |
| no warmup | 4.781 |
| neither | 5.057 |

The book expected a loss spike without warmup and got something else: the
loss fell to 7.0 within 10 steps and then sat there, at the level of a model
that only knows token frequencies, for about 40 steps. Chapter §8d has the
plot. At the default learning rate (10⁻³) both safety rails looked
unnecessary; they only show their value when they carry load.

## 7. Check yourself

<details><summary>1. Adam's first step moves every weight by almost exactly the learning rate, whatever its gradient. Why?</summary>

After bias correction, the first update is m̂ / √v̂ = g / |g| = ±1 per weight
(plus a tiny ε), so the step is lr in the sign of the gradient. Large and
small gradients move the same distance.
</details>

<details><summary>2. Why is "weight decay in the loss" (L2) different from AdamW's decoupled decay?</summary>

In the loss, the decay term joins the gradient and is then divided by √v
like everything else, so weights with large gradients are barely decayed.
Decoupled decay shrinks every weight by the same fraction, outside Adam's
per-weight scaling.
</details>

<details><summary>3. Gradient clipping rescales the whole gradient, not each element. What does that preserve?</summary>

The direction. Every element is multiplied by the same factor, so the step
points the same way, just shorter.
</details>

<details><summary>4. With gradient accumulation over 4 micro-batches, why must each loss be divided by 4?</summary>

Gradients add up across backward passes. Without the division the
accumulated gradient is 4 times the big-batch gradient, which is the same as
training at 4 times the learning rate. One of the tests checks exactly this.
</details>

---

**Stuck?** Solution:
[`stages/08_training/optim.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/08_training/optim.py).

**Next:** Module 9, pretraining: the checkpoint that makes a long run survive
a crash, and your own base model.
