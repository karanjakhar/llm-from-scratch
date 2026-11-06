# Stage 10 — Scaling Laws and Choosing Hyperparameters

We have a correct model (Stage 06), a correct loop (Stage 08) and a fast one
(Stage 09). Stage 11 will spend about two days of GPU time on one training
run. Before spending it, we need to answer a question that sounds simple:
**for a fixed amount of compute, how big should the model be, and how many
tokens should it see?**

A bigger model learns more from each token but costs more per token. With a
fixed budget, every parameter added means fewer tokens. Somewhere between "a
huge model that sees almost nothing" and "a tiny model that sees everything"
is a best size. Scaling laws are the measured answer to where.

What is in this stage:

- the compute formula **`C ≈ 6ND`**;
- the **Kaplan** (2020) and **Chinchilla** (2022) results, and why they
  disagreed;
- the **~20 tokens per parameter** rule, and why most small models are
  undertrained;
- the **IsoFLOP** method, run for real on our GPU;
- **width vs depth**;
- why **inference cost** moves the optimum (over-training small models on
  purpose);
- the domain questions for our project: **100% code vs 85/15 code/English**,
  and **what repeating data costs**.

---

## 1. `C ≈ 6ND`

Stage 09 §1 counted the training cost: about `6N` FLOPs per token for a model
with `N` non-embedding parameters (2 forward, 4 backward). A run over `D`
tokens therefore costs

```
C ≈ 6 · N · D        FLOPs
```

Worked example: GPT-2 small's 85M non-embedding parameters on 2.5B tokens is
`6 × 85e6 × 2.5e9 = 1.3e18` FLOPs. At 20 TFLOP/s sustained, that is 64,000
seconds, about 18 hours.

This formula is the budget. Every choice in this stage is a way of spending
a fixed `C` between `N` and `D`.

---

## 2. Two answers to the same question

**Kaplan et al. (OpenAI, 2020)** trained many models and fitted power laws:
loss falls smoothly as a power of `N`, of `D`, and of `C`. Their recommendation:
when compute grows 10×, grow the model about 5.5× and the data only about
1.8×. Models should be big, and trained on relatively few tokens. GPT-3 (175B
parameters, 300B tokens, under 2 tokens per parameter) followed this advice.

**Hoffmann et al. (DeepMind, "Chinchilla", 2022)** redid the experiment and
got a different answer: **grow `N` and `D` equally**. When compute grows 10×,
grow both by about √10 ≈ 3.2×. The best model for a budget sees about **20
tokens per parameter**. They proved it by training Chinchilla, 70B parameters
on 1.4T tokens, with the same compute as the 280B-parameter Gopher. Chinchilla
beat Gopher on almost everything.

Why did the two disagree? Mostly the **learning-rate schedule**. Kaplan used
one cosine schedule length for all runs and read losses off the middle of
training. A model stopped before its schedule finished has not had its
learning rate decayed (Stage 08 §4), so its loss is worse than it would be in
a run of that length. That penalised long training, i.e. many tokens, and made
big-model, few-token runs look better than they were. Chinchilla tuned the
schedule to each run's length.

The lesson for us is practical: **every run in a scaling study must have its
schedule set to its own length.** Our IsoFLOP runs do.

---

## 3. The IsoFLOP method

Chinchilla's cleanest approach is the **IsoFLOP curve**:

1. Fix a compute budget `C`.
2. Pick several model sizes `N₁ < N₂ < ... < N₅`.
3. For each, set `D = C / (6N)` and train with a full schedule.
4. Plot final loss against `N`. The curve is U-shaped: too small a model
   cannot learn enough; too big a model sees too few tokens.
5. The bottom of the U is the best size for that budget.

Repeat at a few budgets and the bottoms trace out the **compute-optimal
frontier**: how `N_opt` grows with `C`.

```
loss
 │ ╲                       ╱
 │  ╲                    ╱
 │   ╲                 ╱
 │    ╲    ●        ╱        one compute budget
 │      ╲__________╱
 │           ↑
 │         N_opt
 └──────────────────────────── N (log scale)
```

We can afford this at small scale: budgets around `1e16` FLOPs, models from
~4M to ~60M non-embedding parameters, each run 10-30 minutes on the 4060.

---

## 4. Width vs depth

For a fixed `N`, should the model be wide and shallow, or narrow and deep?
`N ≈ 12·L·C²` lets many `(L, C)` pairs give the same `N`.

Kaplan found that loss depends mostly on `N`, and only weakly on the shape,
over a wide range of aspect ratios (`C/L` from ~20 to ~200). Outside that
range, things go wrong:

- **too shallow**: few layers means few rounds of tokens exchanging
  information. Multi-step patterns need depth.
- **too narrow**: tiny head size, and the per-token vector cannot hold much.
- **too deep for the width**: on our GPU, also slow. Each layer is a sequence
  of small matmuls, and small matmuls use the GPU poorly (Stage 09).

For the speed reason, at small scale we keep `C/L` around 64 (GPT-2 small is
768/12 = 64) and head size 64.

---

## 5. Why inference changes the answer

Chinchilla minimises **training** compute for a given loss. But a model that
is going to be used a lot also costs compute every time it runs, and
inference cost scales with `N`, not `D`.

A model 2× smaller than Chinchilla-optimal, trained on 4× more tokens, can
reach the same loss for somewhat more training compute and is then **half the
price to run, forever**. That is why Llama 1 7B saw 1T tokens (~140 per
parameter), and Llama 3 8B saw 15T (~1,900 per parameter). They are
deliberately "over-trained" by Chinchilla's standard.

The cost of over-training is diminishing returns: each doubling of `D` gives a
smaller loss drop. §8c measures one model trained to 10× its Chinchilla
token count.

---

## 6. Our two domain questions

**Mix ratio.** Our data is 85% Python and 15% English (Stage 02). Does the
English cost code performance at a fixed budget, or help it? Code loss is the
number that matters for HumanEval. English might help (natural-language
docstrings and comments are part of code) or might just take budget away.
§8d trains both at the same compute.

**Repeating data.** We have about 2.36B tokens. A 92M-parameter model at 20
tokens per parameter wants 1.8B; at our planned 2.5B, that is about 1.06
epochs. How much does seeing data a second time cost? Muennighoff et al.
(2023) found that up to ~4 epochs, repeated data is worth almost as much as
fresh data. §8e measures it at our scale: the same budget, on fresh data vs a
quarter of the data repeated four times.

---

## 7. Build order and acceptance criteria

`stages/10_scaling/run.py` uses `llmfs.train.trainer` unchanged (with the
Stage 09 settings). Per run: model shape, `N`, `D`, `C`, final val loss on
code and English, wall time.

- IsoFLOP at two budgets, five sizes each, every schedule set to its own length;
- a fitted parabola in `log N` per budget, its minimum reported with
  tokens-per-parameter;
- the measured optimum compared with Chinchilla's 20 tokens per parameter;
- an over-trained run: one size, 10× its optimal tokens;
- mix 100/0 vs 85/15 at one size and budget;
- fresh vs 4-epoch data at one size and budget;
- the capstone config chosen from these numbers, with the reasoning written
  down.

---

## 8. Experiments

Every run: GPT with QK-norm (§8f), T=512, 16 sequences per step, AdamW,
warmup then cosine set to the run's own length, lr scaled mildly with width
(1.1e-3 at C=192 down to 6.5e-4 at C=640). Compute per token counted
honestly as `6N + 6CV + 12LTC` (Stage 09 §1), so small models, whose output
head and attention are a large share of their FLOPs, are not flattered.

### 8a. IsoFLOP curves

| model | params (total) | `3e15` FLOPs: tokens/param, val code | `1e16`: tokens/param, val code |
|---|---|---|---|
| 2 layers, C=128 | 1.5M | 194, 2.480 | — |
| 3 layers, C=192 | 3.0M | 48, 2.329 | 158, 2.184 |
| 4 layers, C=256 | 5.4M | 15, **2.279** | 49, 2.065 |
| 6 layers, C=384 | 14.0M | 2.2, 2.481 | 7.4, **2.055** |
| 8 layers, C=512 | 29.7M | 0.5, 3.667 | 1.7, 2.241 |
| 10 layers, C=640 | 54.8M | — | 0.5, 2.622 |

Both budgets give the U of §3: too small a model saturates (the 1.5M model
at 194 tokens per parameter), too big a model barely starts (the 30M model
at half a token per parameter is still at 3.67). Each run is 4-20 minutes;
the two sweeps together took about 3 hours.

### 8b. The frontier and the capstone choice

A parabola in `log N` through each budget's points, its minimum, and
Chinchilla's prediction `N = √(C/120)` (which is `D = 20N` with `C = 6ND`):

| budget | fitted optimal N | tokens per parameter | Chinchilla's N |
|---|---|---|---|
| `3e15` | **4.2M** | **28** | 5.0M |
| `1e16` | **8.6M** | **22.5** | 9.1M |

**At our scale, on our data, with our tokenizer, the Chinchilla rule holds
to within 15%.** The optimum grows with compute (3.3× more compute, 2.0×
larger model; Chinchilla says √3.3 = 1.8×), and the tokens-per-parameter
ratio lands at 22-28 against Chinchilla's 20. Two budgets and five points
each is a thin basis for a law, and the fitted minima are sensitive to
single runs (the v1 sweep's diverged runs would have moved them a lot).
What it does support is the order of magnitude: on this data, a model
should see a few tens of tokens per parameter.

**For the capstone** (Stage 11), the frontier says: at 92M parameters, about
1.8-2.3B tokens. That is what 20 hours of the 4060 buys at 25.4k tokens/s
(Stage 09), and it is what our data supports (§8e: 1.15 epochs of code). The
capstone is planned at **1.8B tokens**, 20 per parameter, on the
compute-optimal line rather than past it.

### 8c. Ten times past optimal

The 3-layer, 3.0M model, trained to 1×, 2×, 5× and 10× its Chinchilla
token count (20 per parameter = 60M tokens):

| tokens | ×optimal | val code | gain over previous |
|---|---|---|---|
| 60M | 1 | 2.534 | — |
| 120M | 2 | 2.365 | −0.169 |
| 301M | 5 | 2.228 | −0.137 (for 2.5×) |
| 601M | 10 | 2.178 | −0.050 (for 2×) |

Training 10× longer improves the loss by 0.36 nats, but the last doubling
of compute buys only 0.05. Compare the IsoFLOP table: the 10× run used
`1.25e16` FLOPs (`20.9M × 601M`) and reached 2.178; the best model at `1e16`,
with 20% less compute, reached **2.055**. Spending the compute on a bigger
model would have been far better for loss. Over-training is worth it only
when the model's size is fixed by something else, such as inference cost
(§5) or, for us, GPU memory.

### 8d. Code only vs 85/15

The 4-layer model, 5.4M parameters, 108M tokens (20 per parameter), same
compute, three mixes:

| mix | val code | val English |
|---|---|---|
| 100% code | **2.182** | 5.308 |
| 85% code / 15% English (ours) | 2.204 (+0.022) | **4.046** |
| 70% / 30% | 2.260 (+0.079) | 3.844 |

Taking 15% of the budget away from code costs **0.022 nats** of code loss and
buys **1.26 nats** of English. At 30% the code cost is 3.6× larger. The
85/15 choice from Stage 02 holds up: the English is nearly free for code and
transforms the model's English (5.31 is barely above unigram-level
English; 4.05 writes sentences). Whether that English helps follow
instructions after SFT (the curriculum's second question) is for Stage 16.

### 8e. Fresh vs repeated data

Same model and budget (108M tokens, code only), with fewer unique tokens
seen more times:

| unique code tokens | epochs | val code | cost |
|---|---|---|---|
| 108M | 1 | 2.176 | — |
| 27M | 4 | 2.288 | +0.112 |
| 6.7M | 16 | 2.803 | +0.627 |

Four epochs cost 0.11 nats, 5% of the loss; sixteen cost 0.63. This is a
larger penalty than Muennighoff et al. (2023) report for 4 epochs ("almost as
good as fresh"), and there is a confound we did not remove: the repeated
slice is the **first** 27M (or 6.7M) tokens of the file, a contiguous run of
repositories, so it is less varied as well as smaller. Part of the cost is
diversity, not repetition. The Stage 11 capstone sees 1.15 epochs of code:
far closer to the fresh-data end, where we expect the cost to be small, but
we did not measure 1.15 epochs directly.

### 8f. The sweep that diverged, and QK-norm

The first IsoFLOP sweep used the plain GPT-2 recipe that had trained cleanly
in every stage so far. The 4-layer, C=256 model failed in all four of its
long runs: gradient norms climbed into the tens and thousands after about
6,000 steps, and the 100% code run ended at a code loss of **3.52**, worse
than the same model reaches in 2,000 steps. A probe on one fixed batch every
250 steps found the cause. The largest attention score `q·k/√d` in block 1:

```
step       0     1000   3000   5000   6000   6500    7000     8000      12000
max score  0.6   35     62     70     218    575     14,207   262,641   1,998,983
grad norm  3.9   1.06   1.59   1.16   1.52   2.30    3.40     18.7      32.5
```

The score grew slowly for 5,000 steps, then ran away: a score of 14,000 makes
the softmax exactly one-hot, and the gradient through a saturated softmax is
large and erratic (Stage 04 §3). The largest gradient in the network sat in
that block's LayerNorm. **The attention score moved 1,000 steps before the
gradient norm did, and 6,000 steps before the loss.** This is the
attention-logit-growth instability (Dehghani et al. 2023; Wortsman et al.
2023), usually discussed for billion-parameter runs, here in a 5M model, by
running it 3-4× longer than any earlier stage did.

The same run, 13,000 steps, one change at a time:

| variant | code val | max grad norm | max attention score |
|---|---|---|---|
| as in the sweep (lr 1e-3) | 2.656 | 269 | 2,191,156 |
| lr 5e-4 | 2.431 | 3.9 | 117 |
| **QK-norm, lr 1e-3** | **2.280** | 3.8 | 37 |
| B=32, T=256 (Stage 06's shape) | 2.476 | 3.8 | 78 |

QK-norm (normalise each head's `q` and `k` to rms 1 times a learned gain,
Stage 12 §5) bounds the score by construction and keeps the higher learning
rate, which is worth another 0.15 nats over simply halving it. **Every run in
§8a-e uses QK-norm.** The diverged first sweep is kept in
`results/10_scaling/v1_no_qknorm/`.

---

## 9. What we found

| Question | Answer, measured |
|---|---|
| Does the plain GPT-2 recipe survive long runs? | No: the 4L/256 model diverged in all 4 long runs. Attention scores grew to 2M; QK-norm fixed it and was 0.15 better than halving lr |
| Compute-optimal size, `3e15` FLOPs | 4.2M parameters, 28 tokens/param (Chinchilla: 5.0M) |
| Compute-optimal size, `1e16` FLOPs | 8.6M parameters, 22.5 tokens/param (Chinchilla: 9.1M) |
| 10× past optimal? | −0.36 nats in total; the last doubling only −0.05. A bigger model with less compute did better |
| Code-only vs 85/15? | 85/15 costs 0.022 code nats and gains 1.26 English nats |
| Repeated data? | 4 epochs +0.11, 16 epochs +0.63 (with a diversity confound) |
| Capstone | 92M parameters, 1.8B tokens (20/param): on the line, and what 20 GPU-hours buy |

Two habits from this stage:

- **Log the leading indicator, not just the loss.** The attention-score
  maximum moved 6,000 steps before the loss did. A scaling sweep is many
  long runs, and every one of them can fail quietly.
- **Rerun the whole sweep when the recipe changes.** Patching only the
  failed points would mix two recipes in one curve.
