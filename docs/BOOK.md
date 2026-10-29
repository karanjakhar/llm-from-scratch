# The Book

This project ends in a book. Everything in `docs/` is written as a draft of it,
and `docs/JOURNAL.md` holds the raw story. This file is the plan: what the book
is, how it is written, which chapter comes from which source, and how far each
one has got.

**Working title:** *From Bytes to Code: Building a Language Model From Scratch
on One Laptop GPU* (placeholder; the owner decides the final title)

---

## 1. What the book is

**One-sentence pitch.** A reader who knows Python but not machine learning
follows one real project, from raw GitHub files to a small working code model,
and understands every part because every part is built, measured and broken
on purpose.

**What makes it different from other "build an LLM" books:**

1. **Everything is measured.** Every claim comes with a number from a real run
   on modest hardware (RTX 4060 Laptop, 8 GB). No "this should help". It
   either helped by a measured amount or it did not.
2. **The mistakes stay in.** The wrong tests, the datasets that turned out to
   contain no data, the leak that put 40% of the exam into the textbook. Each
   chapter shows what went wrong and how it was caught, because that is what
   readers will meet in their own work.
3. **Every component gets broken on purpose.** Each chapter removes the thing
   it just built and shows the damage. That ablation is where the
   understanding lands.
4. **Honest about scale.** A 125M-parameter model will not reason or know
   facts. The book says so up front and is about understanding the pipeline,
   not beating benchmarks.

**Reader.** Comfortable with Python and basic high-school math. Has heard of
ChatGPT, has never trained a model. Often a visual learner, so every mechanism
gets a picture.

---

## 2. How every chapter is written

Each chapter follows the same arc, so readers always know where they are:

1. **The question.** What problem does this component solve? What breaks
   without it? One paragraph, plain words.
2. **The picture.** A diagram of the mechanism before any formula.
3. **The idea, plainly.** Intuition first, with small worked numbers.
4. **The idea, formally.** The math, with tensor shapes.
5. **The build.** The code, walked through. Shapes on every tensor line.
6. **The measurement.** The numbers from our run, in a table or chart.
7. **What surprised us.** From `JOURNAL.md`: what we expected, what happened.
8. **What we got wrong.** Mistakes and how they were caught.
9. **Breaking it on purpose.** The ablation and the damage it caused.
10. **Takeaways.** Three to five rules the reader keeps.

### Voice and style

- **Plain language first.** Explain as you would to a smart friend across a
  table. Define every term before using it, then add it to `GLOSSARY.md`.
- Short sentences. Active voice. No hype. No "simply", "just", "obviously".
- **Real numbers only.** Every number in the book traces to a results file, a
  journal entry, or a command the reader can run. No invented examples where a
  real one exists. When a small made-up example is clearer (like the
  "low lower lowest" BPE walkthrough), label it as an illustration.
- Analogies are welcome when they are accurate (the exam and the textbook for
  train/val). Drop any analogy that breaks under a second look.
- Say what is simplified, and what real systems do differently.
- When something is genuinely hard, say so.

---

## 3. Chapter map

Status: `outline` · `draft` (stage doc written, findings in) · `revised` (rewritten
in book voice) · `final`

| # | Chapter (working) | Stages | Sources | Status |
|---|---|---|---|---|
| — | Preface: why one laptop, why code | — | `README.md`, `docs/domain_choice.md` | outline |
| 1 | What a language model actually computes | 00 | `docs/00_setup.md` | draft |
| 2 | Choosing what to teach it | — | `docs/domain_choice.md`, `docs/datasets.md` | draft |
| 3 | Cutting text into tokens | 01 | `docs/01_tokenizer.md` | draft |
| 4 | The data is most of the model | 02 | `docs/02_data.md` | draft |
| 5 | Turning ids into vectors and position | 03 | `docs/03_*` | not started |
| 6 | Attention, from one head up | 04 | | not started |
| 7 | The transformer block | 05 | | not started |
| 8 | Assembling GPT | 06 | | not started |
| 9 | Making it talk: generation and decoding | 07 | | not started |
| 10 | The training loop | 08 | | not started |
| 11 | Making it fast on 8 GB | 09 | | not started |
| 12 | How big, how long: scaling laws | 10 | | not started |
| 13 | The real pretraining run | 11 | | not started |
| 14 | Modern architecture: from GPT-2 to Llama | 12 | | not started |
| 15 | Mixture of Experts | 13 | | not started |
| 16 | Other kinds of attention, longer context | 14 | | not started |
| 17 | Faster decoding | 15 | | not started |
| 18 | Teaching it to follow instructions | 16 | | not started |
| 19 | Fine-tuning with 1% of the parameters: LoRA | 17 | | not started |
| 20 | Learning what people prefer: reward models | 18 | | not started |
| 21 | RLHF, DPO and GRPO | 19 | | not started |
| 22 | Rewards you can verify: unit tests | 20 | | not started |
| 23 | Smaller and faster: distillation and quantization | 21 | | not started |
| 24 | Measuring what matters | 22 | | not started |
| 25 | Serving it | 23 | | not started |
| 26 | The whole pipeline on one page | 24 | | not started |
| A | Appendix: glossary | — | `GLOSSARY.md` | draft |
| B | Appendix: every result in one table | — | `PROGRESS.md` results log | draft |
| C | Appendix: reproducing everything | — | `README.md`, `pyproject.toml`, manifests | outline |

The chapter order may change; the stage numbers will not.

---

## 4. Figures

Every mechanism gets a picture. Figures live in `docs/visuals/` (interactive
HTML) and, for print, will need static versions.

| Figure | Chapter | Shows | Source | Print-ready? |
|---|---|---|---|---|
| Training memory bar | 1 | weights 48 / grads+acts 64 / Adam 96 MB vs 192 predicted | `visuals/stages_00-02_build_map.html` | no |
| 25-stage track | Preface | where the project is | same | no |
| Data pipeline | 4 | GitHub → license → split → dedup → tokenizer → bins → loader | same | no |
| Leakage bars | 4 | 43.9 / 34.4 / 21.7 / 8.1% rare-line leakage | same | no |
| BPE training loop | 3 | bytes → cut → count → merge → record → repeat | same | no |
| BPE lab (interactive) | 3 | merges happening step by step | same | needs static frames |
| Encoding replays merges | 3 | a new word encoded, merge by merge | same | needs static frames |
| Three digit rules | 3 | grouped / cap3 / split on real input | same | no |
| Compression vs GPT-2/GPT-4 | 3 | 2.22 / 3.34 / 3.38 / 3.59 / 4.24 bytes per token | same | no |

**Wanted, not yet made:** a MinHash/LSH diagram (chapter 4), a picture of
packing documents with `<|endoftext|>` (chapter 4), and the random-vs-shuffled
sampler coverage plot showing the 37% never seen (chapter 4).

**For print:** the interactive pages will need exporting to static SVG/PNG
before typesetting. Keep interactive versions as the book's companion website.

---

## 5. How the book is kept up to date (every session)

1. **Write findings the same session they happen** in `docs/JOURNAL.md`, using
   its entry format, with real numbers. Never reconstruct later.
2. **Every stage doc ends with a "What we found" section**, beside the
   theory: the measurements, the surprises, the mistakes. Link to journal
   entries rather than duplicating them at length.
3. **New terms go in `GLOSSARY.md`** the first time they appear.
4. **Every number gets a source**: a `results/` JSON, a manifest, or a
   `PROGRESS.md` results-log row.
5. **New figures go in `docs/visuals/`** and get a row in §4 above.
6. **Update the chapter status** in §3 when a stage doc reaches draft.

---

## 6. Open questions for the book

- Final title, and whether the book follows stage order strictly or groups
  some stages (e.g. 14 and 15) into one chapter.
- Format and publisher: self-published (Leanpub, print-on-demand) or
  traditional. This affects how the figures need to be produced.
- Whether the companion repo is tagged per chapter, so readers can check out
  "the code as of chapter 5".
