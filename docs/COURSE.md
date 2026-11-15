# The Course

This project becomes a course as well as a book (owner's decision, 2026-11-15).
The book is for reading; the course is for doing. A learner who finishes the
core path has written a tokenizer, attention, a GPT, a training loop and a
fine-tuning step themselves, checked each one against the same tests this
project used, and trained a small model that writes Python.

This file is the plan: who it is for, how a module works, what is in the core
path, how learners get the code and the compute, and what still has to be
built.

---

## 1. Decisions (2026-11-15)

| Question | Decision | Why |
|---|---|---|
| Where it lives | Lesson pages at karanjakhar.dev/course, built from this repo like the book; learners clone a starter repo from GitHub | free, owned, one build pipeline, and the book chapters are one click away |
| What learners do | Implement the key functions themselves; the existing tests grade them; complete solutions are this repo | writing the code is where it sticks, and ~295 tests already exist as acceptance criteria |
| Hardware assumed | A free Colab T4 (16 GB) or any local NVIDIA GPU with 6 GB+; the long training runs are optional, with our checkpoints published to download instead | most learners do not have a spare GPU for 20 hours |
| Length | 12 core modules (tokenizer to an evaluated, fine-tuned model) plus 13 electives | finishable in a few weeks of evenings; the rest stays available |

Note the difference from how this project was built. The owner learned from
complete working code (CLAUDE.md). The course gives learners blanks to fill,
because they do not have the build history: writing the function is their
version of building it.

---

## 2. Who it is for

Same reader as the book (`docs/BOOK.md` §1): comfortable with Python and
high-school algebra, has never trained a model. Knows what a function, a
class and a list comprehension are. Does not need to know PyTorch beforehand;
module 0 covers the tensor operations the rest uses.

---

## 3. How one module works

Every module has the same seven steps, so learners always know where they are.

1. **Why.** One paragraph: what problem this part solves and what breaks
   without it.
2. **Read.** The matching book chapter, with the sections that matter for the
   exercise marked.
3. **Implement.** The starter repo has the module's file with chosen
   functions blanked: signature, docstring, shape comments and a hint stay;
   the body is `raise NotImplementedError`. Usually 3-6 functions.
4. **Check.** `uv run course/check.py NN` runs the module's tests and prints
   which functions pass. The tests are the ones this project wrote; nothing is
   weakened for the course.
5. **Run.** The module's experiment (`run.py`), producing a number. The
   learner compares it against the book's number from `results/`, with the
   tolerance stated (seed and hardware noise are real, `docs/appendix_reproduce.md` §C.5).
6. **Break it.** The ablation: remove what was just built, rerun, and see the
   damage the book measured.
7. **Check yourself.** Three to five questions whose answers are in the
   chapter or in what the learner just measured, answers folded away.

Stuck learners get, in order: the hint in the docstring, the chapter section,
then the solution in this repo (linked per function).

---

## 4. The core path

Every module was checked end to end on a generated starter
(`course/verify_starter.py`): with only that module blank, the checker lists
every exercise and fails; with the solution restored, it passes. Each of the
58 exercises was also blanked on its own, and every one fails at least one
test, so none is ungraded.

| # | Module | Stage | Learner implements | Tests | Experiment |
|---|---|---|---|---|---|
| 0 | Setup and the memory budget | 00 | parameter count, 16-bytes rule, what fits, a measurement | 5 (new, `course/module00/`) | `verify_env.py` |
| 1 | Tokenizer | 01 | BPE stats, merge, train, encode, decode; the regex variant | 58 | compression vs vocab size; regex ablation |
| 2 | Data | 02 | hash split, exact dedup, MinHash, shuffled sampler, mixed loader | 34 | leakage on 2 shards, before and after near-dedup; build the token files |
| 3 | Embeddings and position | 03 | token lookup, learned, sinusoidal, RoPE (both layouts) | 22 | position ablation |
| 4 | Attention | 04 | causal mask, scaled dot-product, multi-head | 12 | √d saturation; √d training ablation |
| 5 | The transformer block | 05 | LayerNorm, RMSNorm, GELU, SiLU, MLP, SwiGLU | 23 | gradients at init; no-residual ablation |
| 6 | GPT | 06 | attention, block, forward, scaled init, GPT-2 loader, hand count | 20 | match Hugging Face's GPT-2; no-shift ablation |
| 7 | Generation | 07 | temperature, top-k, top-p, repetition penalty, sampling, KV cache | 16 | cache speedup |
| 8 | Training loop | 08 | SGD, AdamW, schedule, clipping | 10 | optimizer race; overfit one batch; TinyStories |
| 9 | Pretraining | 11 | the checkpoint `save` | 2 (new, `llmfs/train/test_resume.py`) | resume test; the 30M workhorse, or download it |
| 10 | Evaluation | 22 | truncation, pass@k, test execution | 10 | HumanEval and MBPP for your checkpoints |
| 11 | Instruction tuning | 16 | chat template, loss mask, padding | 7 | SFT the workhorse; measure the forgetting |

Evaluation comes before instruction tuning on purpose. Module 11's
experiment scores the fine-tuned model with the harness module 10 builds, and
the book's own finding reads best in that order: build the ruler, measure the
base model, then measure what fine-tuning changed.

Learners train on a smaller data set than the book: two of the twelve code
shards and up to 100,000 English documents (module 2). Experiments that do
not touch that data (modules 0, 1, 4's saturation, 7, 8's overfit, 9's
resume test, 6's GPT-2 match) were rerun while writing the course and
reproduce the book's numbers. For those that do, each lesson states what the
smaller data changed, measured on the same subset.

**Electives**, each standalone after module 9: efficiency (09), scaling laws
(10), modern architecture (12), MoE (13), attention variants (14), speculative
decoding (15), LoRA (17), reward models (18), DPO (19), GRPO with unit tests
(20), distillation and quantization (21), serving (23), and the capstone (24).

---

## 5. Getting code and compute to learners

**Starter repo.** Generated, not hand-maintained: `course/make_starter.py`
copies this repo, then blanks the functions listed in `course/exercises.py`
using Python's syntax tree (so the docstring, signature and comments stay and
nothing in `stages/` has to be edited). It is published as a separate GitHub
repo, `llm-from-scratch-course`. Regenerating it after a fix in this repo is
one command.

**Solutions** are this repo. Earlier modules' answers are also in `llmfs/`,
which later modules import; that is accepted: the course runs on the honour
system, and a learner on module 6 has already written module 4.

**Colab.** One notebook per module (`course/make_notebooks.py`). It mounts
the learner's Google Drive and clones the starter repo there once, so code
and the module 2 data survive Colab's resets; installs the dependencies with
`pip` (Colab ships its own PyTorch); then runs the check and the experiment.
Learners with a local GPU use `uv` as the book does.

**The tokenizer** ships inside the starter (`data/tokenizers/split_8192.json`,
271 KB), so no learner spends an hour retraining it and every checkpoint
stays compatible.

**Checkpoints** (workhorse 30M base, workhorse SFT, capstone 92M base) are
published to the Hugging Face Hub so modules 10-11 and the electives can start
without the 2-20 hour pretraining runs.

---

## 6. Open work and risks

- **Nothing has run on Colab yet.** The notebooks' install list was checked
  in a clean Python 3.12 environment without PyTorch, and every number in the
  lessons comes from the book's RTX 4060 Laptop. Before publishing, one pass
  through the notebooks on a real Colab T4 is needed.
- **Colab's T4 has no bf16.** The training scripts use bf16 autocast, which
  a T4 can only emulate, so training there runs near fp32 speed: expect the
  GPU modules to take several times the stated minutes. An fp16 path with a
  loss scaler in `llmfs/train/trainer.py` would recover most of it; not built.
- **Reading data from Drive.** Training reads the token files with random
  access through Drive's network filesystem, which may be slow. If the Colab
  pass shows it is, the notebooks should copy `data/bins/` to the local disk
  at the start of each session.
- **The checkpoints are not published.** `course/get_checkpoints.py` stops
  with a message until the Hugging Face repo exists and its revision is
  pinned. Until then, module 9's workhorse has to be trained (2 h 18 min on
  the book's GPU).
- **Test granularity.** Some tests exercise several functions at once, so one
  blank can fail many tests. `check.py` lists the blank functions first and
  says which one to start with.
- **Publishing needs the owner.** Creating the `llm-from-scratch-course` repo,
  uploading checkpoints to the Hugging Face Hub, and deploying /course are
  outward-facing and use the owner's accounts; each is confirmed first.

---

## 7. Status

| Piece | Status |
|---|---|
| This plan | draft |
| Starter generator, `check.py`, `verify_starter.py` | built; all 12 modules verified |
| Lessons and notebooks, modules 0-11 | draft |
| Electives | not started (the book chapters and repo code stand in) |
| /course page on the site | built (`course/build_web.py`), not deployed |
| A pass through the notebooks on Colab | not done |
| Starter repo on GitHub | not created (owner confirms) |
| Checkpoints on the Hub | not uploaded (owner confirms) |
