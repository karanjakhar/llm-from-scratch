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
`course/colab/` and run its first cell. It mounts your Google Drive and keeps
this repository there (`MyDrive/llm-course`), so your code and the data you
build in module 2 survive when Colab resets the machine.

## Modules

Do them in order: each one's experiment runs on the code you wrote in the earlier ones.

| # | Module | Lesson | Colab notebook |
|---|---|---|---|
| 0 | Where the memory goes | [`course/modules/00_setup.md`](course/modules/00_setup.md) | [`course/colab/00_setup.ipynb`](course/colab/00_setup.ipynb) |
| 1 | Cutting text into tokens | [`course/modules/01_tokenizer.md`](course/modules/01_tokenizer.md) | [`course/colab/01_tokenizer.ipynb`](course/colab/01_tokenizer.ipynb) |
| 2 | The data is most of the model | [`course/modules/02_data.md`](course/modules/02_data.md) | [`course/colab/02_data.ipynb`](course/colab/02_data.ipynb) |
| 3 | Embeddings and position | [`course/modules/03_embeddings.md`](course/modules/03_embeddings.md) | [`course/colab/03_embeddings.ipynb`](course/colab/03_embeddings.ipynb) |
| 4 | Attention | [`course/modules/04_attention.md`](course/modules/04_attention.md) | [`course/colab/04_attention.ipynb`](course/colab/04_attention.ipynb) |
| 5 | The transformer block | [`course/modules/05_block.md`](course/modules/05_block.md) | [`course/colab/05_block.ipynb`](course/colab/05_block.ipynb) |
| 6 | Assembling GPT, and loading GPT-2 | [`course/modules/06_gpt.md`](course/modules/06_gpt.md) | [`course/colab/06_gpt.ipynb`](course/colab/06_gpt.ipynb) |
| 7 | Making it talk | [`course/modules/07_generation.md`](course/modules/07_generation.md) | [`course/colab/07_generation.ipynb`](course/colab/07_generation.ipynb) |
| 8 | The training loop | [`course/modules/08_training.md`](course/modules/08_training.md) | [`course/colab/08_training.ipynb`](course/colab/08_training.ipynb) |
| 9 | Pretraining, and the checkpoint | [`course/modules/09_pretraining.md`](course/modules/09_pretraining.md) | [`course/colab/09_pretraining.ipynb`](course/colab/09_pretraining.ipynb) |
| 10 | Measuring what matters | [`course/modules/10_evaluation.md`](course/modules/10_evaluation.md) | [`course/colab/10_evaluation.ipynb`](course/colab/10_evaluation.ipynb) |
| 11 | Teaching it to follow instructions | [`course/modules/11_sft.md`](course/modules/11_sft.md) | [`course/colab/11_sft.ipynb`](course/colab/11_sft.ipynb) |

After module 11, the book's later chapters (LoRA, reward models, DPO, RL with unit tests, quantization, serving) are the electives: each has its code and tests in this repository.

## Honour system

Everything is in the open: the solutions repository, and the `llmfs/`
library here, which later modules import and which contains finished versions
of earlier modules' code. Nothing stops you from copying. The tests can only
tell you your code works; whether you understand it is between you and the
"Break it" step.

## License

MIT, see [LICENSE](LICENSE). Your solutions are yours.
