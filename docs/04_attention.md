# Stage 04 — Attention, From One Head Up

After Stage 03, every token is a vector that knows *what* it is and *where* it
is. But each vector still knows nothing about its neighbours. In

```python
total = price * qty
return total
```

the vector for the second `total` has no idea that `total` was just assigned,
or what from. To predict what comes after `return total`, it needs
information from earlier tokens. **Attention is how a token gathers
information from other tokens.** It is the only place in a transformer where
tokens talk to each other. Everything else (the MLP, the norms) works on one
token at a time.

This stage builds attention from one head to many, checks it against
PyTorch's own `F.scaled_dot_product_attention`, measures what it costs, and
breaks the `√d` scaling on purpose.

Shape letters, as in Stage 03: `B` batch, `T` tokens, `C` model width,
`H` heads, `d = C / H` head size.

---

## 1. The idea: a dictionary lookup that is allowed to be fuzzy

### A Python dict, the strict version

```python
memory = {"price": 9.99, "qty": 3, "total": 29.97}
memory["qty"]          # → 3
```

You bring a **key**, it is compared with every stored key, exactly one matches,
you get its **value**. Exact match, one answer, nothing in between.

### Attention, the soft version

Attention does the same three things with vectors, and allows partial
matches:

1. Each token asks a question: its **query** `q`.
2. Each token advertises what it has: its **key** `k`.
3. Each token offers some content: its **value** `v`.

A token compares its query with *every* key (a dot product: large when the two
vectors point the same way). The scores become weights that add up to 1. The
result is a **weighted average of all the values**. A strong match gets most of
the weight, a weak match a little, never exactly zero.

```
            query of "return total"
                     │
        compare with every key (dot product)
       ┌─────────────┼─────────────────┬───────────────┐
       ▼             ▼                 ▼               ▼
   key "total"   key "="          key "price"      key "qty"
   score 4.0     score 0.1        score 1.2        score 0.9
       └─────────────┴──────┬──────────┴───────────────┘
                     softmax: scores → weights summing to 1
                            │
             0.80·v_total + 0.05·v_= + 0.09·v_price + 0.06·v_qty
                            │
                 new vector for "return total"
```

(Illustration: the scores are invented to show the flow.)

Why soft instead of exact? **Because it has a gradient.** A hard lookup
either picks an entry or does not, and you cannot nudge it a little. A
weighted average changes smoothly when the scores change, so training can
adjust the queries and keys bit by bit until the right tokens get the weight.

Where do `q`, `k` and `v` come from? Three learned matrices, applied to each
token's vector:

```
q = x @ W_q      k = x @ W_k      v = x @ W_v      each (C,) @ (C, d) → (d,)
```

The same token produces three different vectors for three different jobs:
what am I looking for, what do I match on, what do I hand over if picked.

### Worked example (small numbers, computed)

One query, three tokens, head size `d = 2`:

```
q  = [2, 0]
k1 = [2, 0]   v1 = [10,  0]
k2 = [0, 2]   v2 = [ 0, 10]
k3 = [1, 1]   v3 = [ 5,  5]
```

| step | token 1 | token 2 | token 3 |
|---|---|---|---|
| score `q · k` | 4.0 | 0.0 | 2.0 |
| divide by `√d = 1.414` | 2.828 | 0.000 | 1.414 |
| softmax (weights) | 0.768 | 0.045 | 0.187 |

Output = `0.768·v1 + 0.045·v2 + 0.187·v3 = [8.61, 1.39]`. Mostly token 1,
some token 3 (a partial match), a trace of token 2 (no match, still not zero).

---

## 2. The formula

For all tokens at once, stack the queries, keys and values into matrices:

```
Q = X W_q          (T, C) @ (C, d) → (T, d)
K = X W_k          (T, d)
V = X W_v          (T, d)

scores  = Q Kᵀ / √d                  (T, d) @ (d, T) → (T, T)
weights = softmax(scores, each row)  (T, T), every row sums to 1
out     = weights V                  (T, T) @ (T, d) → (T, d)
```

`scores[i, j]` is "how much token `i`'s question matches token `j`'s key". Row
`i` of `weights` is token `i`'s attention: how it splits its attention across
all tokens. That `(T, T)` matrix is the **attention map**, and §8 shows real
ones.

Two pieces need explaining: the `√d`, and how to stop tokens from looking at
the future.

---

## 3. Why divide by √d

### The problem: dot products grow with the length of the vectors

A dot product is a sum of `d` products: `q·k = q₁k₁ + q₂k₂ + ... + q_d k_d`.
If each `qᵢ` and `kᵢ` is a random number with mean 0 and variance 1, each
product also has mean 0 and variance 1. Adding `d` independent ones gives
variance `d`, so the typical size (standard deviation) is `√d`.

| head size `d` | typical size of `q·k` |
|---|---|
| 4 | 2 |
| 64 | 8 |
| 512 | 22.6 |

### Why large scores are bad: softmax saturates

Softmax turns differences between scores into ratios: a score larger by 1
gets e ≈ 2.7× the weight, larger by 10 gets e¹⁰ ≈ 22,000×. When scores are
spread over ±20, the largest one takes almost all the weight. With 8 random
keys and a random query (computed):

| `d` | max weight, no scaling | max weight, divide by √d |
|---|---|---|
| 4 | 0.43 | 0.31 |
| 64 | 0.81 | 0.27 |
| 512 | **1.000** | 0.41 |

A weight of 1.000 means the soft lookup has become a hard one. And a hard
lookup has no gradient, which was the whole point of making it soft. The
softmax's slope is `p(1 - p)`, which is 0 at `p = 1` and at `p = 0`. **Every
weight is now either 0 or 1, so almost no gradient flows back to the queries
and keys.** They stop learning.

### The fix

Divide the scores by `√d`. Variance `d` divided by `(√d)² = d` gives variance
1, whatever the head size. One line, and the softmax starts in its useful
range at any width.

This argument assumes `q` and `k` have entries of size about 1. With small
initial weights they are smaller, so the scaling matters less at the start
and more as training grows the weights. §9 measures both: the clean
random-vector case, and a real model trained with and without the scaling.

---

## 4. The causal mask: no peeking

A language model predicts token `t + 1` from tokens `0..t`. In training, all
positions are computed in parallel from one sequence. If position 5 could
attend to position 6, it could read the answer it is being asked to predict.
The loss would drop to near zero and the model would learn nothing usable.

The fix: before the softmax, set every score where `j > i` (the key is in the
future) to `-∞`.

```
scores (T = 4)                after mask                 softmax rows
 [ s00 s01 s02 s03 ]          [ s00  -∞   -∞   -∞ ]       [ 1    0    0    0  ]
 [ s10 s11 s12 s13 ]   ──▶    [ s10 s11   -∞   -∞ ]  ──▶  [ .4   .6   0    0  ]
 [ s20 s21 s22 s23 ]          [ s20 s21  s22   -∞ ]       [ .2   .5   .3   0  ]
 [ s30 s31 s32 s33 ]          [ s30 s31  s32  s33 ]       [ .1   .3   .4   .2 ]
                                                       (weights illustrative)
```

Why `-∞` and not 0? Softmax computes `e^score`. `e^0 = 1`, which is a real
weight. `e^(-∞) = 0` exactly, so a future token gets exactly zero weight and
the remaining weights still add up to 1.

Why mask *before* the softmax and not zero the weights after? Zeroing after
leaves rows that sum to less than 1. Re-normalizing fixes that, but it is
two steps where one would do.

**The trap.** If a row is masked *everywhere* (this happens with padding,
not with a plain causal mask, because every token can see itself), softmax
computes `0 / 0` and returns NaN. One NaN spreads through every later layer.
Our causal-only setup cannot hit it; Stage 16 (SFT with padding) can.

---

## 5. Many heads instead of one

One head computes one attention pattern per token. But a token often needs
several kinds of information at once: the previous token (for syntax), the
matching `(` (for brackets), the definition of a variable (for meaning).

**Multi-head attention** runs `H` smaller attentions side by side, each with
its own `W_q, W_k, W_v` of width `d = C / H`, and joins their outputs:

```
x (B, T, C)
  │  one big matmul makes Q, K, V for all heads at once
  ▼
q, k, v  (B, T, C) each
  │  split C into H heads of size d:      view   → (B, T, H, d)
  │  move heads next to batch:            transpose(1, 2) → (B, H, T, d)
  ▼
attention inside each head (independently)   (B, H, T, T) weights
  ▼
out (B, H, T, d)
  │  move heads back:                     transpose(1, 2) → (B, T, H, d)
  │  glue the heads side by side:         reshape → (B, T, C)
  ▼
output projection W_o (C, C)  ── mixes what the heads found
  ▼
(B, T, C)
```

The `view` + `transpose` part is where most attention bugs live, because a
wrong reshape gives the right *shape* and wrong *contents*. No error is
raised. The test that compares against PyTorch catches it.

**Cost compared with one big head:** the same. `W_q, W_k, W_v, W_o` are four
`C × C` matrices either way (`4C²` parameters, 262,144 at `C = 256`). Heads
split the width, they do not add to it. What changes is that there are `H`
attention maps instead of one, each looking in a `d`-dimensional slice.

---

## 6. What attention costs: the T × T matrix

The `scores` matrix has one entry per pair of tokens: `T × T` per head. It is
the reason long context is expensive.

| `T` | entries per head | bytes per layer at `B = 1`, `H = 8`, fp32 |
|---|---|---|
| 1,024 | 1.0M | 32 MB |
| 4,096 | 16.8M | 512 MB |
| 16,384 | 268M | 8.6 GB, more than our whole GPU |

Double the context, four times the memory and compute. A naive
implementation also keeps the `(T, T)` weights for the backward pass.

**FlashAttention** (Stage 09) computes the same result in tiles without ever
storing the full matrix, so memory grows with `T` instead of `T²`. PyTorch's
`F.scaled_dot_product_attention` uses it when it can. §9 measures our
naive version against it on the 4060.

---

## 7. Checking ourselves against PyTorch

The rule for this project: write it, then match the library. Our attention
must equal `F.scaled_dot_product_attention` to 1e-5, in outputs **and**
gradients, with and without the causal mask. The multi-head module, with its
weights copied in, must equal `nn.MultiheadAttention`.

---

## 8. Reading attention maps

A trained model's attention weights can be plotted: rows are the token doing
the looking, columns the tokens looked at. Some heads do recognisable jobs.
The best-known is the **previous-token head**, which puts most of its weight
on the token just before. It shows as a bright line one step below the
diagonal.

A warning before reading too much into maps: a weight says where a head
*looked*, not what it *used*. If the value vectors at that spot are near zero,
a large weight contributes nothing. Maps are a hint, not an explanation.

§9 shows maps from our trained tiny model on real Python, and counts how many
heads are previous-token heads.

---

## 9. Experiments

All produced by `stages/04_attention/run.py`.

### 9a. Saturation with random vectors (`run.py variance`)

Random `q`, `k`, `v` with unit-variance entries, 256 keys, `d` from 16 to 1024:
the size of the scores, the largest softmax weight, the entropy of the
weights (in nats; spreading evenly over 256 keys gives ln 256 = 5.55, all weight
on one key gives 0), and the gradient at two places: at the **scores** (where
softmax's slope lives) and at the **query**. 512 queries per row.

| `d` | score std, scaled / unscaled | max weight, scaled / unscaled | entropy, unscaled | score gradient, scaled / unscaled | queries with ~no score gradient, unscaled | gradient at q, scaled / unscaled |
|---|---|---|---|---|---|---|
| 16 | 1.00 / 4.00 | 0.046 / 0.494 | 1.90 | 0.356 / 0.829 | 0% | 0.38 / 3.7 |
| 64 | 1.00 / 7.98 | 0.042 / 0.740 | 0.78 | 0.744 / 1.23 | 2% | 0.78 / 13.2 |
| 256 | 1.00 / 16.03 | 0.043 / 0.881 | 0.33 | 1.50 / 0.328 | 14% | 1.58 / 33.9 |
| 1024 | 1.00 / 32.04 | 0.043 / 0.940 | 0.15 | 2.99 / **0.022** | **41%** | 3.1 / 63.1 |

"~no gradient" = score-gradient norm below 1e-3. Source:
`results/04_attention/variance.json`, figure `docs/visuals/04_sqrt_d.png`.

What it shows:
- **Scaled scores have std 1.00 at every `d`.** The fix does exactly what §3
  says. Unscaled std is `√d`: 4, 8, 16, 32.
- **Unscaled softmax saturates as `d` grows.** At `d = 1024` the top key gets
  94% of the weight on average, and entropy is 0.15 nats: the "soft" lookup is
  choosing about one key.
- **The gradient at the scores collapses**: from 1.23 at `d = 64` to 0.022 at
  `d = 1024`, and 41% of queries get essentially none. That is the vanishing
  gradient the curriculum predicted.
- **But the gradient at `q` does not vanish. It grows** (63 vs 3.1). The
  gradient at `q` is the score gradient multiplied by `k`, and unscaled, `k`
  enters at full size `√d` instead of `√d / √d`. The two effects partly cancel,
  and in this setup the `k` factor wins. So "gradients vanish" is true at the
  softmax, and misleading if you measure at the weights. What is really broken
  is that the gradient is concentrated on a few queries (the ones not yet
  saturated) while 41% contribute nothing. §9c checks a real model.

### 9b. The cost of T × T (`run.py cost`)

Our attention vs PyTorch's fused kernel, `B = 1`, `H = 8`, `d = 64`, causal,
forward + backward, on the RTX 4060 Laptop: peak memory and time per call as
`T` grows.

| `T` | ours: peak MB | ours: ms | fused: peak MB | fused: ms |
|---|---|---|---|---|
| 256 | 4.3 | 0.40 | 2.0 | 0.14 |
| 1,024 | 66 | 1.5 | 8.1 | 0.31 |
| 2,048 | 262 | 8.8 | 16.1 | 0.92 |
| 4,096 | 1,044 | 36 | 32.3 | 2.9 |
| 8,192 | 4,168 | 148 | 64.5 | 10.7 |
| 16,384 | **out of memory** | — | 129 | 41 |

Source: `results/04_attention/cost.json`, figure `docs/visuals/04_cost.png`.

- **Ours grows 4× per doubling of `T`** (262 → 1,044 → 4,168 MB): the `T²` from
  §6, measured. The fused kernel grows 2× per doubling: linear in `T`.
- **Ours keeps about four `T × T` matrices alive**, not one. At `T = 1,024` one
  bf16 `T × T` matrix across 8 heads is 16.8 MB, and we peak at 66 MB: the
  scores, the masked scores, the softmax weights kept for backward, and their
  gradient. The §6 table counted one copy, so a naive implementation is about
  4× worse than that table suggests.
- At 16,384 tokens, one attention call would need about 17 GB. Ours cannot run
  at all. The fused kernel takes 129 MB and 41 ms. That is the whole case for
  FlashAttention (Stage 09).
- At `T = 8,192` the fused kernel is also **14× faster**. Memory traffic, not
  arithmetic, is what the naive version wastes time on.

### 9c. Training without √d (`run.py ablation`)

The Stage 03 tiny model (4 layers, `C = 256`, learned positions, tied head,
32.8M tokens), now using **our** attention. Four variants, 2 seeds each:

| run | heads × head size | scaling |
|---|---|---|
| `d64` | 4 × 64 | `÷√d` |
| `d64_noscale` | 4 × 64 | none |
| `d256` | 1 × 256 | `÷√d` |
| `d256_noscale` | 1 × 256 | none |

Measured: held-out code loss, and the attention of the trained model on 128
held-out windows (entropy and top weight, rows 32+ only, so early rows with few
choices do not dominate).

| run | val loss (3 seeds) | attention entropy | top weight | score std at init | `‖∇W_qkv‖` at end (seed 0) |
|---|---|---|---|---|---|
| `d64` | 2.689 ± 0.012 | 2.75 | 0.31 | 0.10 | 0.56 |
| `d64_noscale` | **2.652 ± 0.006** | 2.16 | 0.42 | 0.81 | 0.63 |
| `d256` | 2.705 ± 0.003 | 2.78 | 0.29 | 0.10 | 0.51 |
| `d256_noscale` | **3.655 ± 0.018** | **0.92** | **0.78** | 1.61 | 0.97 |

Source: `results/04_attention/ablation.json`.

**Cross-check first.** `d64` is the Stage 03 `learned` model with our attention
swapped in for PyTorch's. Stage 03 seed 0 scored 2.6849, this one 2.6856. Our
attention trains the same as PyTorch's.

**At `d = 256`, dropping `√d` is a disaster**: +0.95 loss, more than
removing position entirely cost in Stage 03 (+0.66 against learned positions). The trained model's
attention is saturated: the top key gets 78% of the weight, and entropy 0.92
nats means each token effectively reads from e^0.92 ≈ 2.5 tokens. The
scaled model spreads over e^2.78 ≈ 16.

**At `d = 64`, dropping `√d` *helped*, slightly but on all 3 seeds** (2.652 vs
2.689). This was not expected. The reason is the starting point. Our weights
start small (std 0.02), so queries and keys start small too: measured score
std at initialisation is 0.81 unscaled and only **0.10** scaled. With scores
that small, softmax is almost uniform, so the scaled model starts out unable to
focus, and spends early training growing its weights. Unscaled, it starts with
scores in a healthy range. At `d = 256` the unscaled start is 1.61, still fine,
but the scores keep growing as training grows the weights, and by the end the
softmax has saturated.

**The weight gradients did not vanish.** In `d256_noscale` the gradient
reaching `W_q, W_k, W_v` is *larger* at the end (0.97 vs 0.51), consistent with
§9a: the vanishing happens at the softmax, and the `k` factor hides it at the
weights. So we can say what broke (attention collapsed onto ~2.5 tokens and
loss rose by 0.95). We have not shown exactly *why* a saturated model then
learns worse; "gradients vanish" is the textbook story, and our measurement
does not support it in that form.

**Lesson.** `√d` is not a magic constant. It keeps the scores' scale fixed as
`d` changes, so one learning rate and one initialisation work for any head
size. At our small `d = 64` and small init, it was not needed, and even cost a
little. At `d = 256` it was the difference between a working model and a
broken one. Real models use `d = 64-128` with much larger widths, and train for
far longer than our 32.8M tokens, which gives the scores more time to grow.

### 9d. Attention maps (`run.py maps`)

The trained `d64` model (seed 0) on a hand-written Python snippet
(`run.py` `SNIPPET`, 42 tokens), plus statistics for every head over 64
held-out windows. Figures: `docs/visuals/04_attention_maps.png` (all 16
heads), `docs/visuals/04_attention_heads.png` (two heads, labelled). Source:
`results/04_attention/heads.json`.

What we found among the 16 heads:

- **One clear previous-token head: layer 2, head 2.** 68% of its weight on the
  token just before, on held-out code. Its map is a bright line one step below
  the diagonal. A model of 5M parameters, trained for about 3 minutes, grew it
  without being told to.
- **Two "attention sink" heads: layer 3, heads 0 and 1** put 24% and 37% of
  their weight on the **first token of the window**. Our windows start at a
  random point in the data, so that first token means nothing in particular.
  This is a known effect (Xiao et al., 2023): softmax must put weight
  *somewhere*, and a head that has nothing useful to read parks it on a spot
  every token can see. This is why a weight is not evidence of use (§8).
- **Column heads: layer 1, head 0** shows vertical stripes: many later tokens
  all look at the same few earlier tokens.
- **Diffuse heads in layer 0** (entropy 3.4-4.2 nats): weight spread thinly
  over everything, close to an average of recent tokens.
- The snippet also shows our tokenizer at work: `tax` is two tokens (`·t`,
  `ax`) and `qty` is two (`q`, `ty`).

---

## 10. Build order and acceptance criteria

1. `stages/04_attention/attention.py`: `attention()` (functional core),
   `SingleHeadAttention`, `MultiHeadAttention`.
2. `stages/04_attention/test_attention.py`.
3. `stages/04_attention/tiny_lm.py`: the Stage 03 preview model with our
   attention in place of PyTorch's.
4. `stages/04_attention/run.py`: `variance`, `cost`, `ablation`, `maps`.

Acceptance criteria (as tests):

- Our attention equals `F.scaled_dot_product_attention` within 1e-5, causal and
  not, outputs and gradients.
- `MultiHeadAttention` with copied weights equals `nn.MultiheadAttention`.
- Multi-head with `H = 1` equals single-head with the same weights.
- Causality: changing a future token changes no earlier output, exactly.
- Every row of weights sums to 1; masked weights are exactly 0.
- Unscaled scores of unit-variance vectors have variance ≈ `d`; scaled ≈ 1.

---

## 11. What we found

Dated entries in `docs/JOURNAL.md` (2026-10-30). The short version:

| What | Number | Lesson |
|---|---|---|
| Our attention vs PyTorch | equal to 1e-5, outputs and gradients; same training loss (2.6856 vs 2.6849) | Match the reference twice: numerically, and in a real run. |
| Wrong reshape | right shape, wrong answer, no error | Only a reference comparison catches it. |
| √d at random vectors | score gradient 0.022 at d = 1024, 41% of queries dead | Saturation is real and gets worse with `d`. |
| Gradient at `q` without √d | *larger*, 63 vs 3.1 | "Gradients vanish" depends on where you measure. |
| No √d, d = 256, real model | loss +0.95, attention on ~2.5 tokens | The ablation the curriculum predicted. |
| No √d, d = 64, real model | loss −0.04, on all seeds | Small init made scaled scores start too flat (std 0.10). |
| Memory, naive vs fused | 4× vs 2× per doubling; OOM at 16k vs 129 MB | `T²` is real; ~4 copies of the matrix are alive, not 1. |
| Trained heads | one previous-token head (68%), two sinks (24%, 37%) | Structure appears on its own. Weight ≠ use. |

**The habit this chapter teaches: a textbook reason is a prediction, not a
fact.** "Without `√d`, gradients vanish" turned out to be right at the softmax,
wrong at the weights, harmless at `d = 64`, and devastating at `d = 256`. Each
part was only visible because we measured instead of assuming.
