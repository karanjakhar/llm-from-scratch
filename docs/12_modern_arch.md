# Stage 12 — Architecture Upgrades (Llama-class)

GPT-2 is from 2019. Almost every open model since 2023 (Llama 2 and 3,
Mistral, Qwen, Gemma) uses the same handful of changes on top of it. This
stage makes each change on our own model, **one at a time**, and measures
what each one buys. They usually arrive as a bundle. Taken apart, each one
gets its own number.

| Change | GPT-2 | Llama | Covered in |
|---|---|---|---|
| normalisation | LayerNorm | RMSNorm | Stage 05 §3 |
| positions | learned table | RoPE | Stage 03 §4 |
| MLP | GELU, 4C | SwiGLU, ≈8C/3 | Stage 05 §6 |
| biases | everywhere | none | §6 here |
| attention heads | MHA | **GQA** | §2 here |
| (optional) | | QK-norm, sliding window | §4-§5 here |

The new ideas in this stage are the ones about **attention's memory**: GQA and
MQA, sliding windows, RoPE scaling for longer context, QK-norm and attention
sinks. The older ones were explained in Stages 03 and 05 and are only
combined here.

The model is `llmfs/model/llama.py`. Every change is a config field, and with
all of them switched on it matches Hugging Face's `LlamaForCausalLM` to 1e-4
with copied weights (`test_llama.py`).

---

## 1. The upgraded block

```
            x  (B, T, C)
            │
     ┌──────┤
     │      ▼
     │   RMSNorm                         was LayerNorm
     │      ▼
     │   q = x Wq  (H heads)
     │   k = x Wk  (Hk heads)  ← fewer    was H heads
     │   v = x Wv  (Hk heads)  ← fewer
     │   rotate q, k by position (RoPE)   was: add a position vector at the bottom
     │   attention, each group of H/Hk query heads sharing one k, v
     │      ▼
     │   o_proj  (no bias)
     └─────(+)
            │
     ┌──────┤
     │      ▼
     │   RMSNorm
     │      ▼
     │   SwiGLU: down( silu(gate x) ⊙ up x )   was: proj(gelu(fc x))
     └─────(+)
```

---

## 2. GQA and MQA: sharing keys and values

### 2a. The problem: the cache

Stage 07 §6d measured the KV cache: for every token, every layer stores a key
and a value vector of size `C`. For a 7B model at 32k tokens, that is 16 GB,
more than the weights. During decoding, every step reads the whole cache from
memory (Stage 07 §6c), so the cache's size also sets the decode speed.

### 2b. The idea: fewer key/value heads

In multi-head attention (MHA), each of the `H` heads has its own query, key
and value projections. **Multi-query attention** (MQA, Shazeer 2019) keeps
`H` query heads but gives them **one shared** key head and value head.
**Grouped-query attention** (GQA, Ainslie et al. 2023) sits between: `Hk` key/
value heads, each shared by a group of `H/Hk` query heads.

```
MHA   H=8, Hk=8:   q q q q q q q q        each q has its own k, v
                   k k k k k k k k
GQA   H=8, Hk=2:   q q q q | q q q q      each group of 4 shares one k, v
                   k       | k
MQA   H=8, Hk=1:   q q q q q q q q        all 8 share one k, v
                   k
```

The cache shrinks by `H/Hk`. Llama 2 70B uses `H = 64, Hk = 8`: an 8× smaller
cache. Llama 3 8B uses `H = 32, Hk = 8`.

### 2c. The cost

Each query head still asks its own question (its own `q`), but heads in a
group must look at the **same keys** and read the **same values**. They can
weight positions differently, but they all choose among the same summaries of
the past. With `Hk = 1`, every head reads the same `v` vectors. That is less
expressive, and quality drops. The question GQA answers is how few kv heads
you can get away with. Ainslie et al. found that 8 groups recovers almost all
of MHA's quality at large scale.

Parameters shrink a little too: `Wk` and `Wv` go from `C × C` to `C × Hk·d`.
At matched parameter count, the savings can go back into the MLP. §9b compares
both ways: at the same architecture, and parameter-matched.

### 2d. The implementation detail

The cache stores `Hk` heads. At attention time, each kv head is
**repeated** `H/Hk` times so the shapes line up with the queries:

```
k: (B, Hk, T, d) → repeat_interleave(H/Hk, dim=1) → (B, H, T, d)
```

`repeat_interleave`, not `repeat`: heads 0-3 must share kv head 0, heads 4-7
kv head 1. `repeat` would give heads 0, 2, 4, 6 kv head 0, which is a
different (also valid, but HF-incompatible) grouping. A test against HF's
Llama catches the difference.

---

## 3. RoPE and longer context

RoPE (Stage 03 §4) rotates each pair of dimensions in `q` and `k` by an angle
proportional to the position, at a different speed per pair: the fastest pair
turns by 1 radian per token, the slowest by `1/10000` per token (`θ = 10000`).

A model trained on length 512 has seen rotations up to angle `512 × speed` in
each pair. Beyond that, the slow pairs reach angles never seen in training,
and quality collapses (Stage 03 §10 measured it for learned and sinusoidal
positions).

Two families of fixes, both one-line changes to the angles:

- **Position interpolation** (Chen et al. 2023): divide every position by `s`.
  Position 2,048 now looks like 512. But the fast pairs, which tell neighbours
  apart, are now squashed by `s` too: adjacent tokens look `s×` closer than
  in training.
- **NTK-aware scaling** (bloc97, 2023): increase the base `θ` instead,
  `θ' = θ · s^(d/(d−2))`. That stretches the **slow** pairs by almost exactly
  `s` and leaves the **fast** pairs almost untouched. Neighbours still look
  like neighbours, and long distances map into the trained range.

Llama 3 raised `θ` to 500,000 at training time, which builds the stretch in
from the start. (`test_ntk_scaling_keeps_fast_clocks` checks the two
properties.) §9d measures loss past the training length with and without
NTK scaling, with no fine-tuning.

---

## 4. Sliding-window attention

Full causal attention lets token `t` see all `t` earlier tokens: compute and
cache grow with `T`. **Sliding-window attention** (Longformer; used in Mistral
7B with window 4,096) lets each token see only the last `W` tokens.

```
full                     window W = 3
1 . . . . .              1 . . . . .
1 1 . . . .              1 1 . . . .
1 1 1 . . .              1 1 1 . . .
1 1 1 1 . .              . 1 1 1 . .
1 1 1 1 1 .              . . 1 1 1 .
1 1 1 1 1 1              . . . 1 1 1
```

The cache only needs the last `W` tokens: bounded memory, however long the
text. Information still travels further than `W`, layer by layer: after `L`
layers, a token can be influenced by one `L·W` back, the way a stack of
small convolutions has a large receptive field.

For code, what is lost is the direct look-up of something far away (an import
at the top of the file, a function defined 2,000 tokens earlier). §9c measures
the loss for a window of 64 and 128 at training length 512.

---

## 5. QK-norm and attention sinks

**QK-norm** (Dehghani et al. 2023, used in OLMo 2, Gemma 3): apply an RMSNorm
to each head's `q` and `k` before the dot product. Without it, `q · k` can
grow without limit as weights grow, the softmax saturates, and the gradients
through it vanish or explode (Stage 04 §3 showed what a saturated softmax does
to gradients). This is a known cause of loss spikes in large runs. With
QK-norm, each `q` and `k` has rms 1 (times a learned gain), so scores stay in
a controlled range. At our scale it should matter little for loss; it is a
stability measure.

**Attention sinks** (Xiao et al. 2023). Stage 04 found heads that put most of
their attention on the first token, whatever it is. Softmax must give out a
total weight of 1; a head with nothing useful to read puts the weight on a
token whose value vector does little, and the first token, visible to
everyone, becomes that dumping ground. It matters for sliding windows and
streaming: if the first token falls out of the window, those heads lose their
sink and quality drops sharply. StreamingLLM's fix keeps the first few tokens
in the cache forever. We measure how much attention our trained model sends to
position 0 (§9e).

---

## 6. No biases

Llama has no biases in any linear layer, and RMSNorm has no shift. The
argument: with pre-norm and enough width, biases add parameters and memory
traffic and buy nothing measurable; the norms already centre (or do not
need to centre) the signal. §9a measures it as its own step.

---

## 7. Parameter counts

Per block, with `C` width, `d` head size, `Hk` kv heads, `Hd` MLP hidden size:

```
attention:  C·C (q) + 2·C·(Hk·d) (k, v) + C·C (o)
SwiGLU:     3·C·Hd
norms:      2·C (RMSNorm: gain only)
```

SwiGLU with `Hd = 8C/3` has `3 · C · 8C/3 = 8C²`, the same as GELU's
`2 · C · 4C`. We round `Hd` up to a multiple of 64 for GPU-friendly shapes,
as Llama does (`multiple_of`), which adds a fraction of a percent.

---

## 8. Build order and acceptance criteria

- `llmfs/model/llama.py` with the switches in the module docstring;
- `test_llama.py`:
  - all switches on: logits match HF `LlamaForCausalLM` within 1e-4, for
    MHA, GQA and MQA;
  - KV-cache decoding matches full recompute, for every switch combination,
    including multi-token chunks after the cache is filled;
  - the sliding window really hides tokens outside it;
  - GQA's cache and parameter savings match the formulas;
  - NTK scaling leaves the fastest pair alone and slows the slowest by `s`.
- `run.py ladder`: GPT-2 baseline, then each change added in turn;
- `run.py gqa`: `Hk` ∈ {8, 4, 2, 1} at `H = 8`, plus a parameter-matched
  MQA; KV bytes per token next to loss;
- `run.py window`, `run.py rope_scale`, `run.py sinks`.

---

## 9. Experiments

Every run: 6 layers, C=384, 8 query heads of 48, T=512, B=16, 4,000 steps
(33M tokens of the 85/15 mix), lr 1e-3, same seed and data order.

### 9a. The ladder: one change at a time

Two seeds per row; each row adds one change to the row above.

| change | val code (seed 0 / 1) | mean | Δ | params |
|---|---|---|---|---|
| GPT-2 style (LayerNorm, learned positions, GELU, biases, MHA) | 2.510 / 2.526 | 2.518 | — | 13.99M |
| + RMSNorm | 2.556 / 2.546 | 2.551 | +0.033 | 13.99M |
| + **RoPE** | 2.320 / 2.321 | **2.321** | **−0.230** | 13.79M |
| + SwiGLU | 2.297 / 2.294 | 2.295 | −0.026 | 13.79M |
| + no biases | 2.305 / 2.303 | 2.304 | +0.009 | 13.77M |
| + GQA, 2 kv heads | 2.319 / 2.340 | 2.329 | +0.026 | 12.44M |
| + **QK-norm** | 2.271 / 2.277 | **2.274** | **−0.055** | 12.44M |

From GPT-2 to the full Llama-style block: **2.518 → 2.274, −0.244**, with 11%
fewer parameters and a 4× smaller KV cache. Almost all of it comes from two
rows:

- **RoPE is worth 0.23 nats**, by far the largest change. Stage 03 measured
  RoPE vs learned positions at 0.17 on a smaller model; here the gap is
  larger. Learned positions must learn every position's vector separately
  from data; RoPE gives relative position for free.
- **QK-norm is worth 0.055** at this learning rate, more than SwiGLU. Stage 10
  showed why: it lets a high learning rate train without attention scores
  running away.
- **RMSNorm costs 0.03**, the same small, consistent penalty as Stage 05
  (+0.023). Its case is speed (Stage 05 §10e), not quality.
- **No biases**: +0.009, inside the seed spread. Llama's claim, "biases buy
  nothing", holds.
- **SwiGLU**: −0.026, close to Stage 05's −0.038.

The seed spread is 0.001-0.022 per row. Anything under ~0.02 is a tie.

### 9b. MHA → GQA → MQA

Same Llama block (RMSNorm, RoPE, SwiGLU, no bias, no QK-norm), one seed,
varying the number of key/value heads for 8 query heads:

| kv heads | KV cache per token | params | val code | Δ vs MHA |
|---|---|---|---|---|
| 8 (MHA) | 9,216 B | 13.77M | 2.306 | — |
| 4 | 4,608 B | 12.88M | 2.305 | −0.002 |
| 2 | 2,304 B | 12.44M | 2.315 | +0.009 |
| 1 (MQA) | 1,152 B | 12.22M | 2.335 | +0.029 |
| 1 + params back into the MLP | 1,152 B | 13.77M | 2.328 | +0.022 |

Halving the cache (8 → 4 kv heads) costs **nothing**. Quartering it costs
0.009, inside seed noise (the ladder's two GQA seeds differ by 0.022). Only
**MQA**, an 8× smaller cache, shows a cost, 0.03 nats, and giving MQA's
saved parameters to the MLP buys back only a quarter of it: the loss is about
sharing, not parameter count. That is the GQA paper's picture at small
scale: a few kv groups recover nearly all of MHA's quality. **Where quality
breaks**: between 2 kv heads and 1.

### 9c. Sliding windows

Llama block with 2 kv heads, trained and evaluated at T=512, with each token
allowed to see only its last `W` tokens. Overall val loss, and loss by
position in the window (64 val windows):

| W | val code | pos 0-16 | 16-64 | 64-128 | 128-256 | 256-512 |
|---|---|---|---|---|---|---|
| full | **2.314** | 3.82 | 2.80 | 2.45 | 2.20 | **1.97** |
| 128 | 2.424 (+0.11) | 3.78 | 2.75 | **2.42** | 2.31 | 2.25 |
| 64 | 2.559 (+0.24) | 3.78 | **2.73** | 2.55 | 2.57 | 2.48 |
| 16 | 3.141 (+0.83) | 3.79 | 3.14 | 3.03 | 3.21 | 3.08 |

The full-attention curve keeps falling with position: tokens 256-512 are
predicted 0.5 nats better than tokens 16-64, because more of the file is
visible. A windowed model's curve **goes flat** once position passes `W`:
beyond that, it sees no more than it did at position `W`. So the cost of a
window is concentrated in long files, which is exactly where code needs it
(imports, earlier definitions).

Two details worth noticing. Within their window, the windowed models are
slightly **better** than full attention (window 64 at positions 16-64: 2.73
vs 2.80): a small model with fewer positions to attend over spends less
capacity on far context. And six layers × 64 tokens is a 384-token
"receptive field" in principle, yet window-64 loses 0.5 nats at positions
256-512: information passed hop by hop through layers is a poor substitute
for a direct look.

### 9d. Longer than trained: NTK scaling

The final ladder model (RoPE, trained at T=512 only), evaluated on longer
windows with the RoPE base stretched by `s` (§3), no fine-tuning (32
windows per length):

| context | `s` | loss at 256-512 | at 512-1024 | at 1024-2048 |
|---|---|---|---|---|
| 1,024 | 1 | 1.96 | 2.20 | — |
| 1,024 | **2** | 1.98 | **2.00** | — |
| 1,024 | 4 | 2.03 | 2.01 | — |
| 2,048 | 1 | 1.94 | 2.09 | 2.56 |
| 2,048 | 2 | 1.96 | **1.87** | 2.23 |
| 2,048 | 4 | 2.03 | 1.86 | **2.19** |

Without scaling, positions past 512 get worse the further out they are
(2.20 at 512-1024, 2.56 at 1024-2048), but RoPE does not collapse the way
learned positions did in Stage 03: relative rotations beyond the trained
range are unfamiliar, not meaningless. With NTK scaling `s = 2`, positions
512-1024 are as good as positions 256-512 were at training length, and with
`s = 4`, positions 1024-2048 recover 0.37 of the 0.6 lost. The cost: a
small loss at short range (0.02-0.07 at positions 256-512), because the
slow clocks no longer turn at the speed the model learned. A short
fine-tune at the new length usually removes it; we did not.

### 9e. Attention sinks

Mean attention weight on token 0, for queries past position 32, in the
final ladder model (QK-norm on). Uniform attention over the visible prefix
would give 0.0057:

```
layer 1   0.005 0.005 0.005 0.005 0.006 0.006 0.005 0.005
layer 2   0.005 0.003 0.002 0.005 0.002 0.002 0.002 0.001
layer 3   0.001 0.002 0.001 0.000 0.000 0.000 0.000 0.000
layer 4   0.002 0.002 0.001 0.000 0.000 0.001 0.000 0.002
layer 5   0.002 0.002 0.001 0.000 0.002 0.000 0.001 0.006
layer 6   0.019 0.005 0.010 0.047 0.007 0.051 0.058 0.009
```

**No strong sinks.** Most heads put *less* than uniform weight on token 0;
the strongest, in the last layer, put 5-6%, ten times uniform but nothing
like the sink heads Stage 04 found in a GPT-2-style model (which sent most
of their weight to position 0). One plausible reason: QK-norm bounds every
score, so no single key can win a softmax by an enormous margin, which is
how a sink forms. We did not run the matching model without QK-norm, so
this is a hypothesis, not a measurement.

---

## 10. What we found

| Question | Answer, measured |
|---|---|
| GPT-2 → full Llama block | 2.518 → 2.274 (−0.244), 11% fewer parameters, 4× smaller cache |
| Biggest single change | RoPE, −0.230 |
| QK-norm | −0.055: worth more than SwiGLU (−0.026) at lr 1e-3 |
| RMSNorm, no biases | +0.033 and +0.009: costs within noise, kept for speed and simplicity |
| How few kv heads? | 4 and 2 heads: free (±0.01); 1 head (MQA): +0.03, and extra MLP width buys back only a quarter |
| Sliding window | +0.11 (W=128), +0.24 (W=64), +0.83 (W=16); loss stops improving with position past `W` |
| Longer than trained | RoPE degrades gently; NTK `s=2` at 2× length recovers fully, `s=4` at 4× recovers 0.37 of 0.6 |
| Attention sinks | none strong with QK-norm (max 5.8% on token 0) |

One habit from this stage: **when a change arrives in a bundle, unbundle it
before believing any part of it.** "Llama is better than GPT-2" is true here,
but 94% of it is one change (RoPE), and two of the bundle's changes cost a
little.
