# Stage 17 — Parameter-Efficient Fine-Tuning: LoRA and QLoRA

Stage 16 fine-tuned every weight of the model. For our 30-100M models that is
fine. For a 7B model it means 7B gradients and 14B Adam state numbers: about
**112 GB** of training memory (16 bytes per parameter, Stage 00), for a model
whose weights are 14 GB. Most people fine-tuning a 7B model have one GPU
with 24 GB or less.

**LoRA** (Hu et al. 2021) freezes the original weights and trains a tiny
add-on instead. **QLoRA** (Dettmers et al. 2023) also stores the frozen
weights in 4 bits. Together they fit a 65B fine-tune on a single 48 GB GPU.

This stage builds both from scratch, checks LoRA against the `peft` library
number for number, and measures what is lost.

---

## 1. The low-rank hypothesis

Fine-tuning changes each weight matrix `W` by some update `ΔW`. For a
`768 × 768` matrix, `ΔW` has 589,824 numbers. But fine-tuning only nudges a
model towards a new format or task; it does not rebuild it. Aghajanyan et al.
(2020) measured that the useful part of fine-tuning updates lives in a
surprisingly small number of directions: the updates have low **intrinsic
rank**.

**Rank**, in plain words: a matrix of rank `r` can be written as the product
of a tall thin matrix and a short wide one:

```
ΔW   =   B   ·   A
(d×k)   (d×r)  (r×k)        r ≪ d, k
```

At `d = k = 768`, `r = 8`: `B` and `A` hold `768·8 + 8·768 = 12,288` numbers,
2% of `ΔW`'s 589,824. If the true update is close to rank 8, we lose almost
nothing by learning `B` and `A` instead of `ΔW`.

---

## 2. LoRA

```
            x  (.., k)
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
   W x           A x        (.., r)     A: r × k, trained
 (frozen)          ▼
     │           B (A x)    (.., d)     B: d × r, trained, starts at ZERO
     │             │ × α/r
     └────(+)──────┘
           │
           ▼  y = W x + (α/r) B A x
```

Three details, each with a reason:

1. **`B` starts at zero.** Then `BA = 0` at step 0, and the model is exactly
   the pretrained model. Fine-tuning starts from where pretraining ended, not
   from a randomly perturbed version. (`test_zero_init_changes_nothing`.)
2. **`A` starts random**, not zero. If both were zero, the gradient of `B`
   (which is proportional to `A x`) and the gradient of `A` (proportional to
   `Bᵀ`) would both be zero forever. One of them must be non-zero to break the
   symmetry.
3. **The `α/r` scale.** As `r` grows, `BA` sums more terms and its size grows
   with it. Dividing by `r` keeps the update's scale roughly independent of
   `r`, so a learning rate tuned at `r = 8` still works at `r = 64`. `α` is
   then a single knob for "how strong is the adapter" (commonly `α = 2r`).
   rsLoRA (Kalajievski 2023) argues for `α/√r` instead, for the same reason
   done more carefully; the difference shows up at large `r`.

**Which matrices?** The original paper adapted only the attention `q` and `v`
projections. Later work (QLoRA) found that adapting **every** linear layer,
MLP included, at a smaller rank works better for the same parameter count. We
target `c_attn`, `c_proj` and `c_fc`: every linear in the block.

**Merging.** After training, `W' = W + (α/r) B A` is an ordinary matrix of
the original shape. Fold it in and the adapter costs nothing at inference.
(`test_merge_equals_unmerged`.) Or keep it separate, and serve **many
adapters on one base model**: each customer's fine-tune is a few MB, swapped
per request.

---

## 3. The memory arithmetic

For full fine-tuning in bf16 compute with fp32 master weights and AdamW
(Stage 00's accounting), per parameter: 4 (weights) + 4 (grads) + 8 (Adam) =
16 bytes. With LoRA, the frozen weights need no gradients and no optimizer
state: 2 bytes each in bf16. Only the adapter pays 16 bytes per parameter.

| | 7B full FT | 7B LoRA r=16, all linears | 7B QLoRA |
|---|---|---|---|
| frozen weights | — | 14 GB (bf16) | 3.5 GB (4-bit) |
| trainable params | 7B | ~40M | ~40M |
| grads + Adam + master | 112 GB | 0.6 GB | 0.6 GB |
| **total (excl. activations)** | **~112 GB** | **~15 GB** | **~4 GB** |

Activations still need memory and are not shrunk by LoRA. §8 measures our
own version on our model.

---

## 4. Why fewer parameters can still learn the task

Intuition: the pretrained model already contains the skills. Fine-tuning
mostly needs to *select and reweight* existing features: "after
`<|im_start|>assistant`, behave like the code-writing mode, and stop at the
end". Such a change is a small number of directions in weight space, which is
what a low-rank update can express. Teaching genuinely new knowledge (a new
programming language, a new fact base) is less low-rank, and that is where
LoRA falls behind full fine-tuning (Biderman et al. 2024, "LoRA learns less
and forgets less"). The **forgets less** half is the other side of the same
coin: the frozen weights are untouched, so forgetting (Stage 16 §6) is
limited.

---

## 5. QLoRA and NF4

QLoRA keeps LoRA's adapters in bf16 and stores the frozen base in **4 bits**,
dequantizing each weight block on the fly in the forward pass. Gradients flow
through the dequantized weights into the adapters; the 4-bit codes never
change.

**NF4** ("4-bit NormalFloat"). 4 bits give 16 levels. Where should they go?
Trained weights are roughly normally distributed: most are near zero, few are
large. Evenly spaced levels (int4) waste most of their codes on the rare
large values. NF4 places the 16 levels at the **quantiles of a normal
distribution**, so each level is used about equally often:

```
int4 levels (scaled):   -1  -.86 -.71 -.57 -.43 -.29 -.14  0  .14 .29 .43 .57 .71 .86  1
NF4 levels:             -1  -.70 -.53 -.39 -.28 -.18 -.09  0  .08 .16 .25 .34 .44 .56 .72 1
                                        └──── crowded near 0, where the weights are ────┘
```

Each block of 64 weights is scaled by its own absolute maximum (stored in
bf16), so `level × absmax` reconstructs the weight. Storage: 4 bits per weight
plus 16 bits per 64 weights = **4.25 bits per weight**. QLoRA's **double
quantization** quantizes those absmax values too (to 8 bits, in blocks of
256), bringing it to about 4.13 bits; we skip that refinement. **Paged
optimizers** move optimizer state to CPU memory during memory spikes; on one
small model we do not need them.

`test_nf4_levels_match_paper` checks our 16 levels against the paper's table.

---

## 6. Variants, briefly

- **DoRA** (Liu et al. 2024): split each weight into a magnitude (one number
  per column) and a direction, and apply LoRA to the direction only. Closer to
  full fine-tuning at the same rank.
- **rsLoRA**: scale by `α/√r` (§2).
- **LoRA+**: a larger learning rate for `B` than for `A`.

---

## 7. Acceptance criteria

`llmfs/post/lora.py`, `stages/17_lora/test_lora.py`:

- zero-initialised adapter leaves the model's output bit-identical;
- only adapter parameters are trainable, and their count is `r·(in+out)` per
  layer;
- outputs match `peft`'s LoRA with the same `A`, `B` to 1e-5;
- merged model equals unmerged model to 1e-5; `BA` has rank `r`;
- NF4 levels match the paper; 4-bit storage is 0.5 bytes per weight plus
  scales;
- `run.py`: LoRA SFT vs full SFT (Stage 16 data and steps) on the same base:
  trainable params, peak memory, step time, completion val loss, pass@1;
- rank sweep `r` ∈ {1, 4, 16, 64}; QLoRA at `r = 16`; a merged adapter.

---

## 8. Experiments

Base: the Stage 11 workhorse (30M). Data, steps and batch: Stage 16's
(119k docstring → function pairs, 3,000 steps of 16, completion-only loss).
Learning rate 1e-4 for full fine-tuning, 1e-3 for the adapters (adapters
start at zero and need larger steps). LoRA on every linear layer of every
block (`c_attn`, `c_proj`, `c_fc`), `α = 2r`.

| method | trainable params | peak MB (training, above weights) | ms / step | completion val loss | HumanEval |
|---|---|---|---|---|---|
| full fine-tune | 29,941,760 (100%) | 4,615 | 253 | 1.0328 | 1.2% |
| LoRA r = 1 | 65,536 (0.22%) | 4,105 | 261 | 1.0318 | 0.0% |
| LoRA r = 4 | 262,144 (0.88%) | 4,091 | 263 | **1.0278** | 0.6% |
| LoRA r = 16 | 1,048,576 (3.5%) | 4,104 | 263 | 1.0281 | 0.6% |
| LoRA r = 16, merged | — | — | — | 1.0280 | — |
| LoRA r = 64 | 4,194,304 (14%) | 4,158 | 266 | 1.0328 | 0.6% |
| QLoRA r = 16 (NF4 base) | 1,048,576 | 4,104 | 267 | 1.0321 | 0.6% |

**Quality saturates at rank 1.** With 0.22% of the parameters trainable, LoRA
matches full fine-tuning (1.0318 vs 1.0328), and ranks 4-16 are slightly
*better* (1.028). The low-rank hypothesis of §1 holds strongly here: this SFT
is learning a format (the chat template, the stop token, "write the whole
function"), and a format is a handful of directions. More capacity does not
help, and r = 64 and full fine-tuning, with the most capacity, are a
little worse: closer to fitting the training pairs' specifics. HumanEval
moves between 0 and 2 problems of 164 across all rows, which is noise.

**Merging is free**: the r = 16 adapter folded into the weights gives
1.0280 against 1.0281 unmerged (§2).

**QLoRA costs 0.004** over LoRA at the same rank (1.0321 vs 1.0281): the
base model's weights are now 4-bit NF4, and the adapter trains around the
quantization error.

**The memory savings are small at this size, and the table shows why.**
LoRA saves 510 MB of a 4.6 GB training peak (−11%). That is almost exactly
the 16-bytes-per-parameter accounting of §3 for the gradients and Adam state
LoRA no longer keeps: 30M × 16 B = 480 MB. The rest of the 4.1 GB is
activations (16 sequences × 512 tokens, and the `16 × 512 × 8,196` logits in
fp32), which LoRA does not touch. For a 7B model, where the 112 GB of
optimizer state dwarfs the activations, the same mechanism is the
difference between impossible and possible on one GPU. QLoRA saves nothing
visible here either: its 4-bit base is 22 MB instead of 57 MB, against a
4 GB peak. Speed is also unchanged (253-267 ms/step): the forward and
backward passes still run through the whole frozen network.

---

## 9. What we found

| Question | Answer, measured |
|---|---|
| Does our LoRA equal `peft`'s? | Yes, to 1e-5 with the same A, B (test) |
| Where does quality saturate? | At r = 1 (0.22% of params): 1.0318 vs full 1.0328; r = 4-16 best at 1.028 |
| Merge | 1.0280 merged vs 1.0281 unmerged |
| QLoRA cost | +0.004 over LoRA r = 16 |
| Memory saved | 510 MB of 4.6 GB (= the 30M × 16 B optimizer accounting); activations dominate at 30M |
| Speed | unchanged (~260 ms/step) |

One habit from this stage: **work out where the memory goes before choosing
a memory technique.** LoRA attacks optimizer state; at 30M parameters the
optimizer state was 11% of the peak.
