# About this draft

This is a book being written while the project it describes is being built.
A reader who knows Python but not machine learning follows one real project,
from raw GitHub files to a small working model that writes Python code, and
understands every part because every part is built by hand, measured, and
broken on purpose.

**How far it has got.** Six of the planned 26 chapters are drafted, one per
finished stage of the [25-stage curriculum](https://github.com/karanjakhar/llm-from-scratch/blob/main/CURRICULUM.md).
They are first drafts: the measurements and findings are in, but some
sections still read like engineering notes and will be rewritten for readers.
New chapters appear as stages finish.

**What makes it different.**

1. **Everything is measured.** Every claim comes with a number from a real run
   on one RTX 4060 Laptop GPU with 8 GB of memory. A change either helped by a
   measured amount or it did not.
2. **The mistakes stay in.** Wrong tests, datasets that turned out to hold no
   data, a leak that put 40% of the exam into the textbook. Each chapter shows
   what went wrong and how it was caught.
3. **Every component gets broken on purpose.** Each chapter removes the thing
   it just built and records the damage. That is where the understanding lands.
4. **Honest about scale.** A 125M-parameter model will not reason or know
   facts. The book is about understanding the pipeline, not beating benchmarks.

**Code.** Every number traces to code and results in the
[companion repository](https://github.com/karanjakhar/llm-from-scratch). Paths
like `stages/04_attention/run.py` in the text refer to files there.

**Interactive figures** for chapters 1 to 4 (the data pipeline, the tokenizer
training loop, a step-by-step BPE lab) are on one page:
[the build map](visuals/stages_00-02_build_map.html).
