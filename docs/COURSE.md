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

Times are for a learner, not for the compute alone. "Colab" means it runs on
a free T4 in the time shown.

| # | Module | Stage | Learner implements | Experiment | Colab |
|---|---|---|---|---|---|
| 0 | Setup and the memory budget | 00 | the 16-bytes-per-parameter estimate | measure it on your GPU | yes |
| 1 | Tokenizer | 01 | `get_stats`, `merge`, BPE `train`, `encode`, `decode`, the regex variant | compression vs vocab size; regex ablation | yes, CPU |
| 2 | Data | 02 | repo-level split, exact and near dedup, packing documents with an end-of-text token | the leakage measurement | yes, 1 shard |
| 3 | Embeddings and position | 03 | token + learned position, sinusoidal, RoPE | position ablation | yes |
| 4 | Attention | 04 | scaled dot-product, causal mask, multi-head | √d ablation, attention maps | yes |
| 5 | The transformer block | 05 | LayerNorm, RMSNorm, GELU MLP, pre-norm block | depth and residual ablation | yes |
| 6 | GPT | 06 | the full model; load OpenAI GPT-2 weights | match Hugging Face to 1e-4 | yes |
| 7 | Generation | 07 | temperature, top-k, top-p, KV cache | cache speedup; sampling knobs | yes |
| 8 | Training loop | 08 | AdamW, warmup + cosine schedule, clipping, the loop | 10M TinyStories smoke test | yes, ~1 h |
| 9 | Pretraining | 11 | (configures, does not re-implement) | train a small code model, or download ours | small run yes; 30M optional |
| 10 | Instruction tuning | 16 | prompt masking, the SFT loss | SFT the workhorse; measure forgetting | yes |
| 11 | Evaluation | 22 | pass@k estimator, test execution, contamination check | score your model on HumanEval | yes |

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

**Colab.** One notebook per module that clones the starter repo, installs
the dependencies with `pip` (Colab ships its own PyTorch), and runs the check
and experiment cells. Learners with a local GPU use `uv` as the book does.

**Checkpoints** (workhorse 30M base, workhorse SFT, capstone 92M base) are
published to the Hugging Face Hub so modules 10-11 and the electives can start
without the 2-20 hour pretraining runs.

---

## 6. Open work and risks

- **Colab's T4 has no bf16.** The trainer uses bf16 autocast. On a T4 that
  either falls back to fp32 (about 2-4× slower) or needs an fp16 path with a
  loss scaler. To measure in module 8 before writing its times down; adding
  fp16 to `llmfs/train/trainer.py` is likely needed.
- **Module 2 data size.** The full corpus is 4.9 GB of downloads and needs
  near-dedup over 12 shards; the module uses 1 shard, so its leakage numbers
  will differ from the book's. State the expected range, not the book's value.
- **Test granularity.** Some tests exercise several functions at once, so one
  blank can fail many tests. `check.py` maps tests to functions so the report
  says which function to fix.
- **Publishing needs the owner.** Creating the `llm-from-scratch-course` repo
  and uploading checkpoints to the Hugging Face Hub are outward-facing and use
  the owner's accounts; each is confirmed before it happens.

---

## 7. Status

| Piece | Status |
|---|---|
| This plan | draft |
| Starter generator + `check.py` | built, verified on module 1 |
| Module 1 lesson page + notebook | draft |
| Modules 0, 2-11 | not started |
| Electives | not started |
| /course pages on the site | built (`course/build_web.py`), not deployed |
| Starter repo on GitHub | not started (owner confirms) |
| Checkpoints on the Hub | not started (owner confirms) |
