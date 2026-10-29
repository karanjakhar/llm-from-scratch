# Stage 00 — Setup and the Mental Model

Before any code: what is the thing we are building, and what does the hardware
let us build?

---

## 1. What a language model actually is

Strip away the hype. A language model is a function that takes a sequence of
tokens and returns a probability distribution over which token comes next.

```
input:  [15496, 11, 616, 1438, 318]        # "Hello, my name is"
output: a vector of length vocab_size, summing to 1
        → P(next = " John")  = 0.031
        → P(next = " a")     = 0.027
        → P(next = " not")   = 0.019
        ... one number for every token in the vocabulary
```

That is the whole contract. Everything else — chat, reasoning, code, agents —
is built on top of repeatedly sampling from that distribution and feeding the
result back in.

### The autoregressive factorization

Why next-token prediction is enough to model *any* text. The probability of a
whole sequence factorizes by the chain rule of probability:

```
P(x₁, x₂, ..., x_n) = P(x₁) · P(x₂|x₁) · P(x₃|x₁,x₂) · ... · P(x_n|x₁...x_{n-1})
                    = Π_t P(x_t | x_<t)
```

No approximation. This is exact. So a model that is good at every conditional
`P(x_t | x_<t)` is a complete model of the text distribution. Learning to
predict the next token is not a proxy task — it is the whole task.

### Training is one trick: shift the labels

Given a sequence of T tokens, the inputs are positions `0..T-2` and the targets
are positions `1..T-1`. Same array, offset by one:

```
tokens:   [  the,  cat,  sat,   on,  the,  mat ]
inputs:   [  the,  cat,  sat,   on,  the ]
targets:  [       cat,  sat,   on,  the,  mat ]
```

The model produces a prediction at *every* position in parallel, and we compute
cross-entropy loss at every position. One forward pass gives T-1 training
signals. This is why transformers train so efficiently — and the causal mask
(Stage 04) is what makes it legal, by stopping position 2 from peeking at
position 3's answer.

### The train/inference asymmetry

This asymmetry causes most of the engineering in this repo:

| | Training | Inference (decode) |
|---|---|---|
| Sequence processing | all T positions at once | one token at a time |
| Bottleneck | compute (matmul FLOPs) | memory bandwidth |
| Needs activations stored | yes, for backward | no |
| Parallelism | across sequence + batch | across batch only |

Training is embarrassingly parallel over positions. Generation is inherently
serial — token t+1 cannot start before token t exists. That single fact is why
the KV cache (Stage 07), speculative decoding (Stage 15), and continuous
batching (Stage 23) all exist.

---

## 2. Where the VRAM goes

8 GB is the binding constraint on this project, so this arithmetic is not
optional. There are four consumers of GPU memory during training.

Let `N` = number of parameters, in **bf16 mixed precision with AdamW** (the
standard setup):

| Consumer | Bytes per parameter | Why |
|---|---|---|
| Parameters | 2 (bf16) or 4 (fp32 master) | the weights themselves |
| Gradients | 2-4 | one per parameter |
| AdamW state | 8 | two fp32 buffers: momentum + variance |
| **Total, fixed** | **~12-16** | independent of batch size |

Plus **activations**, which scale with batch size and sequence length and are
the part you actually control at runtime.

### Worked example: a 50M parameter model

```
parameters   50M × 2 bytes (bf16)   = 100 MB
gradients    50M × 2 bytes          = 100 MB
AdamW m,v    50M × 8 bytes (fp32)   = 400 MB
fp32 master  50M × 4 bytes          = 200 MB
                                     --------
fixed cost                          ≈ 800 MB
```

800 MB out of 7797 MB usable. Comfortable. The remaining ~7 GB is activations, CUDA
context (~500 MB), and fragmentation.

A rough activation estimate per layer, per sequence, is
`~ s · (several × d_model + n_head · s)` elements — the `n_head · s²` attention
matrix term is the one that explodes. With FlashAttention/SDPA that term
disappears, which is why Stage 09 matters so much on 8 GB.

### The scaling reality

| Params | Fixed cost (bf16 + AdamW) | Fits in 8 GB? |
|---|---|---|
| 10M | ~160 MB | easily |
| 125M (GPT-2 small) | ~2.0 GB | yes, with modest batch |
| 350M | ~5.6 GB | tight, needs checkpointing + small batch |
| 1B | ~16 GB | no — needs LoRA or offload |
| 7B | ~112 GB | no — inference only, quantized |

**Rule of thumb: full fine-tuning needs ~16 bytes/param; inference in bf16
needs ~2 bytes/param; inference in int4 needs ~0.5 bytes/param.** That one
line explains why a 7B model you can chat with locally cannot be trained
locally.

### Why this is the Stage 00 exercise

The ablation for this stage is: compute a budget that *omits* the AdamW states
(a very common mistake — it is 8 of the ~16 bytes), predict that a model will
fit, then watch it OOM. Being wrong on purpose, once, in a controlled way, is
cheaper than being confused about it for months.

---

## 3. PyTorch essentials actually used in this repo

Not a tutorial — a checklist of what shows up constantly.

**Shape convention.** Every tensor in this project is commented with its shape.
The standard names:
- `B` — batch size
- `T` — time / sequence length (number of tokens)
- `C` — channels / `d_model` / embedding dimension
- `nh` — number of heads, `hs` — head size (`C / nh`)
- `V` — vocabulary size

**Broadcasting.** PyTorch aligns shapes from the right and stretches dims of
size 1. `(B,1,T,T) + (1,nh,T,T)` works and gives `(B,nh,T,T)`. Most attention
bugs are silent broadcasting that produces the right shape with the wrong math.
This is why shape comments are mandatory.

**`view` vs `reshape` vs `transpose`.** `transpose` makes a tensor
non-contiguous; `view` then fails and you need `.contiguous()` or `.reshape()`.
Multi-head attention is three of these in a row and it is where everyone's
first implementation breaks.

**Autograd.** Every op on a tensor with `requires_grad=True` records itself in a
graph. `loss.backward()` walks it backwards accumulating `.grad`. Gradients
*accumulate* — they are not reset — which is a bug source (`zero_grad()`) and a
feature (gradient accumulation, Stage 08).

**`@torch.no_grad()`** for evaluation and generation: skips graph construction,
saving memory and time.

**`nn.Module`.** Holds parameters, supports `.to(device)`, `.train()`/`.eval()`,
`state_dict()`. We subclass it constantly and use almost nothing else from it.

---

## 4. Environment

```bash
cd /mnt/extra_ssd/kj/building_llm_model_from_scratch
uv sync                                          # creates .venv from pyproject.toml
uv run python stages/00_setup/verify_env.py      # checks CUDA, prints memory budget
```

`verify_env.py` prints the actual measured memory cost of a toy model and
compares it to the formula above. If the prediction and the measurement
disagree by more than ~15%, something in the mental model is wrong and that is
worth chasing before Stage 01.

---

## 5. Checklist before moving on

- [ ] Can state what a language model outputs, exactly, including the shape.
- [ ] Can explain why next-token prediction models the full joint distribution.
- [ ] Can explain the label shift and why one forward pass gives T-1 signals.
- [ ] Can name the four consumers of training memory and the bytes each costs.
- [ ] Can predict, within 15%, the fixed memory cost of an N-parameter model.
- [ ] Ran `verify_env.py`; prediction matched measurement.
- [ ] Know why `transpose` then `view` fails.

## 6. What we found

Measured on the RTX 4060 Laptop, 2026-10-14. Full entries in `docs/JOURNAL.md`.

**The 16-bytes-per-parameter rule held to 8.5%.** A 12.58M-parameter model in
fp32 with AdamW:

| | predicted | measured |
|---|---|---|
| weights | 48.0 MB (4 B/param) | 48.0 MB |
| gradients (+ activations) | 48.0 MB | 64.3 MB |
| AdamW state | 96.0 MB (8 B/param) | 96.0 MB |
| **total** | **192.0 MB** | **208.3 MB** |

The 16 MB difference is activations held for the backward pass, which the rule
leaves out on purpose. AdamW's two buffers are the largest slice. A budget that
forgets them predicts 96 MB, off by more than 2×.

**An "8 GB" GPU gives PyTorch 7,797 MB.** About 390 MB is reserved before any
tensor exists. Budget against what the runtime reports.

**What that means for the project.** At 16 bytes per parameter plus
activations, ~125M parameters is the largest model this GPU can train
comfortably. That one number sets the scale of everything that follows.

## Further reading

- Karpathy, "Let's build GPT: from scratch" — the canonical video walkthrough.
- The Illustrated Transformer (Jay Alammar) — for the pictures.
- PyTorch autograd mechanics docs — read once, properly.
