# Module 11: Teaching it to follow instructions

**You will build** the chat template that turns a request and a reply into
one token sequence, the loss mask that trains only on the reply, and the
batching that pads examples without corrupting them. Then you fine-tune your
base model on 119,000 docstring-to-function pairs and measure, with module
10's harness, what it gained and what it forgot.

**Time:** 2-3 hours, plus about 40 minutes of training on the book's GPU. **Hardware:** GPU.
**Book chapter:** [18, Teaching it to follow instructions](../../docs/16_sft.md).

---

## 1. Why

A base model continues text. Ask it "write a function that reverses a list"
and it may continue with another request, because that is what came next in
its training data. Supervised fine-tuning (SFT) shows it thousands of
request → answer pairs in a fixed format, and trains it to produce only the
answer part.

What breaks without the loss mask: train on the prompt as well as the reply,
and the model spends its updates learning to predict docstrings it will
always be given. Chapter §8c measures the cost.

## 2. Read

Chapter 18 of the book:

- §1 *Base model vs instruct model*.
- §2 *Chat templates* and §3 *Loss masking*.
- §4 *Padding, packing, and the mask trap*.
- §6 *Catastrophic forgetting*: what fine-tuning takes away.

## 3. Implement

Open `stages/16_sft/sft.py`. Three blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `render` | user text + reply → token ids and labels, in the chat format |
| 2 | `split_function` | a Python function → (signature + docstring, whole function) |
| 3 | `collate` | pad a batch on the right and shift for next-token prediction |

`split_function` builds the training data: CodeSearchNet's 412,178 Python
functions, filtered to free functions with a docstring and a body. The
request is the signature and docstring; the reply is the whole function.

## 4. Check

```bash
uv run python course/check.py 11
```

7 tests. The loss mask must cover exactly the reply and its closing
`<|im_end|>` (the model must learn to stop); padding on the right must not
change the logits of the real tokens; methods with `self` must be rejected;
accumulating gradients over micro-batches must match one full batch.

## 5. Run

Needs the workhorse base model from module 9 and CodeSearchNet (from
`download_extras.py`):

```bash
uv run python stages/16_sft/run.py --base checkpoints/11_pretrain/workhorse/final.pt --tag mine
```

3,000 steps; 37 minutes on the book's GPU, including evaluation. The book's run on the 30M workhorse:

| | before SFT | after SFT |
|---|---|---|
| completion loss on held-out pairs | 1.168 | 1.033 |
| HumanEval pass@1 (asked as a chat turn) | 0 | 1.22% (2 of 164) |
| pretraining validation loss, code | 1.684 | 1.832 |
| pretraining validation loss, English | 3.341 | 3.646 |

Run while writing this course on the book's workhorse: the completion loss
went 1.1675 → 1.0328 and HumanEval reached 1.22%, identical to the book,
because the fine-tuning pairs and the base model are the same. On the
two-shard validation set the forgetting read 1.671 → 1.826 (code) and
3.359 → 3.662 (English). The first row is what SFT trains. The last two rows are the
forgetting: the model got worse at plain code and English while learning the
format. On the book's 92M model the trade was worse still: SFT cost it 3
HumanEval problems (5 → 2), which reinforcement learning with unit tests
later won back (chapter 22).

Then add your SFT model to module 10's table:

```bash
uv run python stages/22_evaluation/run.py table --redo
```

(The table looks for `checkpoints/16_sft/workhorse.pt`; copy or rename
yours, `checkpoints/16_sft/mine.pt`, to that name first.)

## 6. Break it

```bash
uv run python stages/16_sft/run.py --base checkpoints/11_pretrain/workhorse/final.pt --tag mine --prompt-loss
```

The same fine-tune, but with the loss on the prompt too. Compare its
completion loss and its forgetting with your first run, then read chapter
§8c for the book's result.

## 7. Check yourself

<details><summary>1. Why must the closing <code>&lt;|im_end|&gt;</code> be inside the loss mask?</summary>

It is how the model learns to stop. If the end token is never a target, the
model never learns to produce it and keeps generating past its answer.
</details>

<details><summary>2. Padding goes on the right and the pad positions get label IGNORE. Why does right padding leave the real tokens' logits unchanged?</summary>

The model is causal: a position only attends to positions before it. Pads
placed after the real tokens are never seen by them, so their outputs are
exactly what they would be without padding.
</details>

<details><summary>3. SFT lowered the model's skill at plain code (validation loss up by 0.15). Why train it at all?</summary>

Because the format is the point: a chat model must answer the request it is
given and stop. The book measured the price honestly; whether it is worth it
depends on how the model will be used. The base model, prompted as plain
code, stayed the best HumanEval scorer at 92M.
</details>

<details><summary>4. Why keep only free functions (no <code>self</code>) from CodeSearchNet?</summary>

A method depends on a class the request does not show, so the reply would
refer to attributes the model cannot know. Free functions with a docstring
are self-contained: the request fully describes the answer.
</details>

---

**Stuck?** Solution:
[`stages/16_sft/sft.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/16_sft/sft.py).

**After the core path.** You have built a tokenizer, a GPT, a training loop,
a base model, an evaluation harness and an instruction-tuned model, each
checked against a reference. The electives continue from here: LoRA (fine-tune
1% of the parameters), reward models, DPO, reinforcement learning with unit
tests, quantization, serving. Each is a chapter of the book with its code and
tests in the repository.
