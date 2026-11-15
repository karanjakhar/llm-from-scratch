# Build a Language Model From Scratch: the course

You will write, by hand, every part of a small GPT that writes Python: the
tokenizer, attention, the transformer, the training loop, fine-tuning and
evaluation. Each part is checked by the same tests the original project
used, and each module ends with a number you can compare against the book.

- **Lessons:** `course/modules/`, one page per module (also at
  karanjakhar.dev/course).
- **The book:** `docs/`, one chapter per stage; each lesson tells you which
  sections to read. Also at [karanjakhar.dev/book](https://karanjakhar.dev/book/).
- **Solutions:** the complete project, at
  [github.com/karanjakhar/llm-from-scratch](https://github.com/karanjakhar/llm-from-scratch).

## How a module works

1. **Why**: what this part solves, and what breaks without it.
2. **Read** the marked sections of the book chapter.
3. **Implement**: the module's functions have their bodies replaced by a hint
   and `raise NotImplementedError`. Write them.
4. **Check**: `uv run python course/check.py <module>` runs the tests.
5. **Run** the experiment and compare your number with the book's.
6. **Break it**: remove what you built and measure the damage.
7. **Check yourself**: a few questions, answers folded away.

## Setup

**On your own machine** (NVIDIA GPU with 6 GB or more for the later modules;
module 1 needs only a CPU):

```bash
git clone https://github.com/karanjakhar/llm-from-scratch-course
cd llm-from-scratch-course
uv sync --extra dev          # installs Python 3.12, PyTorch with CUDA, and the rest (~7 GB)
uv run python course/check.py 1
```

[`uv`](https://docs.astral.sh/uv/) is the only thing you install by hand.

**On Google Colab** (free T4 GPU): open the module's notebook from
`course/colab/` and run the first cell, which clones this repository and
installs what Colab does not already have.

## Modules

| # | Module | Status |
|---|---|---|
| 1 | Cutting text into tokens | ready |
| 0, 2-11 | setup, data, embeddings, attention, the block, GPT, generation, training, pretraining, fine-tuning, evaluation | coming |

## Honour system

Everything is in the open: the solutions repository, and the `llmfs/`
library here, which later modules import and which contains finished versions
of earlier modules' code. Nothing stops you from copying. The tests can only
tell you your code works; whether you understand it is between you and the
"Break it" step.
