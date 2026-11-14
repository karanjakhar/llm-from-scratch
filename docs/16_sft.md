# Stage 16 — Supervised Fine-Tuning

The pretrained model from Stage 11 is a **base model**: it continues text.
Give it a docstring and it writes something that looks like what follows
docstrings in its training data: sometimes a function body, sometimes another
docstring, sometimes a test file. It has no idea it is being *asked*
anything.

**Supervised fine-tuning** (SFT) teaches a format: a request, then an answer,
then *stop*. It is the same next-token training as pretraining, on a small,
curated set of request/answer pairs, with one important change to the loss.

What is in this stage:

- base vs instruct behaviour, measured;
- **chat templates** and the special tokens that mark who is speaking;
- **loss masking**: train on the answer only;
- **padding vs packing**, and the attention-mask trap;
- **catastrophic forgetting**;
- how much data SFT needs (LIMA's answer);
- the ablation: train on the prompt tokens too.

---

## 1. Base model vs instruct model

The difference is behaviour, not knowledge. SFT on a few thousand examples
cannot teach a model Python; it took billions of tokens to do that. What SFT
changes is **what the model does with a prompt**:

```
base model, given "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n"
  → "    return a + b\n\n\ndef sub(a, b):\n    \"\"\"Subtract..."     keeps going
instruct model, given the same as a user turn
  → "def add(a, b):\n    ...\n    return a + b\n<|im_end|>"           answers, stops
```

The base model's habit of continuing is not a bug. It is exactly what it was
trained to do: predict the next token of a file. After a function there is
usually another function.

---

## 2. Chat templates

A **chat template** turns a conversation into one token sequence. Ours is
ChatML, using the special tokens reserved since Stage 01:

```
<|im_start|>user
def add(a, b):
    """Add two numbers."""
<|im_end|>
<|im_start|>assistant
def add(a, b):
    """Add two numbers."""
    return a + b
<|im_end|>
```

`<|im_start|>` (id 8194) and `<|im_end|>` (id 8195) can never be produced by
encoding ordinary text: our tokenizer only emits them when asked (Stage 01 §5).
So a user cannot fake a turn boundary by typing the string `<|im_end|>`.

These embedding rows were never trained during pretraining: the tokens never
appeared in the data. SFT is the first time the model sees them. This is
normal, and one reason a new chat template needs some SFT data to "take".

At inference time we render the user turn, append `<|im_start|>assistant\n`,
and let the model write until it produces `<|im_end|>`. **Learning to emit
`<|im_end|>`** is half the point of SFT: a model that answers correctly and
then keeps writing is not usable.

---

## 3. Loss masking

In pretraining every token is a target. In SFT, the prompt is given, not
generated. If the loss includes prompt tokens, the model spends capacity
learning to *predict user questions*, which it will never need to do, and the
gradient from the (often longer) prompt drowns the gradient from the answer.

So the targets for prompt tokens are set to `-100`, which PyTorch's
`cross_entropy(..., ignore_index=-100)` skips:

```
tokens:  <s> user \n def add ... """ <e> \n <s> assistant \n def add ... return a + b <e>
labels:  -100 -100 ...                              -100     def add ... return a + b <e>
                    └──── prompt: no loss ────┘           └──── reply: loss ────┘
```

The closing `<|im_end|>` **is** a target: that is how the model learns to stop.
The test `test_loss_mask_covers_exactly_the_reply` checks that the unmasked
tokens decode to exactly the reply, plus the stop token.

**What breaks without the mask?** §8c trains with the prompt included in the
loss and compares.

---

## 4. Padding, packing, and the mask trap

Examples have different lengths. Two ways to batch them:

**Padding.** Put each example in its own row and fill the rest with
`<|pad|>`. The pad positions get label `-100`. With **right** padding (pads
after the text), the ordinary causal mask is already correct: a real token
can only attend to earlier positions, and all pads come later. The test
`test_right_padding_does_not_change_real_logits` checks it. The cost is
wasted compute on pads.

**Packing.** Concatenate several examples into one full row with no pads.
Efficient, but now example 2 can attend to example 1. Unless the attention
mask is changed to **block-diagonal** (each example sees only itself), the
model learns from, and is conditioned on, unrelated neighbours. This is the
**attention-mask correctness trap**: it raises no error and silently changes
what the model learns. Pretraining tolerates it (documents separated by
`<|endoftext|>`, Stage 02); for SFT, where each example is a separate
conversation, it is a real leak.

We use right padding. SFT data is small, so the wasted compute is affordable,
and the correctness argument is one line.

---

## 5. The data: docstring → function

CodeSearchNet Python has 412k functions with docstrings from GitHub. We keep
free functions (no `self`/`cls`) that parse on their own, have a docstring and
a body, and fit in 512 tokens. That leaves roughly a third.

Each one becomes:

- **user**: the signature and docstring, exactly the shape of a HumanEval
  prompt;
- **assistant**: the whole function.

No annotation is needed. The data is "free" because GitHub authors wrote
docstrings. The cost: docstrings are often poor specifications ("Helper
function.") and the functions use libraries the model has never seen. It is
real-world data, not curated data.

**How much is needed?** LIMA (Zhou et al. 2023) fine-tuned a 65B model on
1,000 carefully chosen examples and matched models tuned on 50× more. The
lesson is that SFT mostly teaches *format and style*, which needs few
examples; the knowledge was already there. A 30M-100M model has much less
knowledge to unlock, so we use more data, and measure val loss on held-out
pairs as it trains.

---

## 6. Catastrophic forgetting

Fine-tuning on a narrow dataset pushes the weights towards that dataset and
away from everything else. Train long enough on docstring→function and the
model's general code modelling gets worse. We measure it directly: the
pretraining val loss (code and English) before and after SFT. A small,
stable learning rate (1e-4, a tenth of pretraining) and a short run keep the
damage small.

---

## 7. Acceptance criteria

`stages/16_sft/sft.py`, `test_sft.py`:

- the loss mask covers exactly the reply plus `<|im_end|>`;
- collation pads on the right, never targets a pad, and shifts correctly;
- right padding does not change any real position's logits;
- the prompt-loss ablation trains on every non-pad token;
- `run.py`: SFT the Stage 11 base model; report completion val loss,
  pretraining val loss before/after (forgetting), HumanEval and MBPP pass@1
  before vs after, and side-by-side completions;
- the ablation: same run with the prompt in the loss.

---

## 8. Experiments

Base: the Stage 11 workhorse (30M). Data: **119,178** CodeSearchNet pairs
survived the filters (of 412,178 functions), 1,000 held out. 3,000 steps of
16 examples (48k examples, 0.4 epoch), lr 1e-4, warmup 50, cosine. Base-model
HumanEval/MBPP numbers come from the Stage 22 table, where the prompt
trailing-newline trap (journal) is handled; this run's own "base mode"
measurement predates that fix and is not used.

### 8a. Base vs SFT

| | completion val loss | HumanEval pass@1 | MBPP pass@1 |
|---|---|---|---|
| base model, prompted as chat (it has never seen the template) | 1.168 | 0.0% | 0.0% |
| **after SFT** | **1.033** | **1.2%** (2/164) | 0.0% |

The loss on assistant tokens falls from 1.168 to 1.036 **in the first 250
steps** and then barely moves (1.033 at step 3,000): the format is learned
almost immediately, LIMA's observation at our scale. The two solved problems
are `strlen` and `concatenate`.

The behavioural change is easiest to see in the outputs. Given HumanEval/0
as a user turn, the SFT model repeats the signature and docstring exactly
(as its training targets did), writes a body, and **stops with `<|im_end|>`**:

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    ...
    """
    if len(numbers) == 2:
        return True
    if len(threshold) == 2:
        return False
```

The format is right; the logic is not. A 30M model has learned what a
function looks like, not what this one should do. Failures on HumanEval are
59 wrong answers (`AssertionError`), 56 syntax errors and 11 indentation
errors, the latter two mostly from the tab/space mixing Stage 11 noticed.

### 8b. Forgetting

Pretraining val loss on the same windows, before and after SFT:

| | before SFT | after SFT | change |
|---|---|---|---|
| code | 1.684 | 1.832 | **+0.148** |
| English | 3.341 | 3.646 | **+0.305** |

0.4 epoch of a narrow dataset at a tenth of the pretraining learning rate
costs 0.15 nats of general code modelling and twice that on English, which
the SFT data contains almost none of. The completion loss stopped improving
after step 250, so 92% of these steps bought forgetting and nothing else. A
shorter run would be strictly better here.

### 8c. Training on the prompt too

Same run with every non-pad token in the loss:

| | completion-only loss (ours) | prompt + completion |
|---|---|---|
| completion val loss | 1.0328 | 1.0295 |
| HumanEval | 1.2% | 1.2% (the same two problems) |
| pretraining val, code / English | 1.832 / 3.646 | 1.843 / 3.654 |

**Almost no difference at this scale.** The prompt here (signature +
docstring) is itself the start of the same function, so learning to predict
it is not wasted effort; and with 119k examples there is no shortage of
completion signal for it to drown out. The prompt-trained run forgets
slightly more (+0.011 code, +0.008 English) because it trains on more tokens.
Loss masking matters most where prompts are long and unlike the answers
(chat, retrieval-augmented prompts); for docstring → function, the two are
nearly the same task.

### 8d. The capstone (92M)

Same data, steps and learning rate, on the Stage 11 capstone (activation
checkpointing on: B = 16 does not fit in 8 GB otherwise, journal Session 10):

| | 30M workhorse | 92M capstone |
|---|---|---|
| completion val loss, before → after | 1.168 → 1.033 | 0.998 → **0.886** |
| step at which it flattened | 250 | 250 (0.888) |
| HumanEval, base model prompted as plain code (Stage 22 harness) | see Stage 22 | **3.05%** (5/164) |
| HumanEval, after SFT, prompted as chat | 1.2% (2/164) | **1.22%** (2/164) |
| MBPP, after SFT | 0.0% | 0.4% (2/500) |
| forgetting, code / English | +0.148 / +0.305 | +0.159 / +0.311 |

The bigger model starts lower and ends lower on the completion loss, and the
same 250-step story repeats: 0.971 at step 0, 0.888 at step 250, 0.886 at
step 3,000. Forgetting is almost identical in size.

The surprise is the HumanEval row. **SFT made the capstone worse at
HumanEval**: the base model, given the prompt as a file to continue, solves 5
problems; the SFT model, given the same prompt as a chat turn, solves 2, and
only one of them (`HumanEval/28`, `concatenate`) is in both sets. HumanEval's
prompts are already "signature + docstring, now write the body": exactly
what a base model continues well. SFT bought the chat format, a habit of
stopping, and 0.16 nats of forgetting; for a benchmark that needs none of
those, the trade was a loss. A bigger model gains more from pretraining
alone, so it has more to lose.

---

## 9. What we found

| Question | Answer, measured |
|---|---|
| Does SFT teach the format? | Yes: completion loss 1.168 → 1.036 within 250 steps; the model stops with `<|im_end|>` |
| Does it teach coding? | Barely: HumanEval 0 → 1.2% (2 problems), MBPP 0% |
| Forgetting | +0.148 code, +0.305 English pretraining loss, mostly after the format was learned |
| Prompt in the loss | A tie here: 1.0295 vs 1.0328, same problems solved |
| On the 92M capstone | Same shape: flat after 250 steps, +0.16 / +0.31 forgetting, and HumanEval 3.05% (base, as code) → 1.22% (SFT, as chat) |
| How much data? | The format took ~4,000 examples (250 steps × 16); the rest cost forgetting |

One habit from this stage: **stop SFT when the completion loss flattens.**
Every step after that is paid for in forgetting.
