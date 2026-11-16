# Module 9: Pretraining, and the checkpoint that saves it

**You will build** the checkpoint that lets a long training run survive a
crash and continue as if nothing happened, prove it bit-for-bit, and then
train (or download) the 30M-parameter base model the rest of the course
fine-tunes and evaluates.

**Time:** 1-2 hours of work. The base model itself is 2 h 18 min of GPU on
the book's machine, or a download. **Hardware:** GPU.
**Book chapter:** [13, The real pretraining run](../../docs/11_pretraining.md).

---

## 1. Why

Everything up to now trained in minutes. A real pretraining run takes hours
or days, and something always interrupts it: a crash, a power cut, a Colab
session ending. A run that cannot resume exactly is a run you cannot finish.
The book's 92M model trained for about 20 hours, was paused for five days in
the middle, and resumed with its next losses within 0.0002 of the values
logged before the pause.

What breaks without it: forget Adam's state in the checkpoint and the resumed
run jumps (+0.057 in loss at the first step after resuming, still +0.017 at
the end). Forget the data position and there is no visible mark at all: the
model silently re-reads data it has already seen, and the loss looks better
than it is.

## 2. Read

Chapter 13 of the book:

- §1 *Why two models*: a small workhorse to debug on, one big run at the end.
- §2 *What a checkpoint must hold*: the table of six things and what breaks
  without each.
- §5 *Determinism*.
- §8b *Resume, and resume done wrong*, including the bug the test caught.

## 3. Implement

One function, in the library trainer every later module uses:
`llmfs/train/trainer.py`, `save`. Read `train` below it first: the resume
code there is the other half of the contract, and it tells you exactly which
keys your checkpoint must contain.

## 4. Check

```bash
uv run python course/check.py 9
```

Two tests, a few seconds on a GPU. They train a tiny model on random tokens
for 20 steps straight through, then again with a stop at step 10 and a
resume, and require the two loss curves to be **identical**, not close. The
second test resumes without the optimizer state and requires the curve to
differ, so you can see the check has teeth.

## 5. Run

The same test on a real model:

```bash
uv run python stages/11_pretrain/run.py resume_test
```

About 4 minutes: a 4-layer model, 1,200 steps, trained three ways. The book's
validation losses, which a rerun while writing this course matched exactly:

| run | final validation code loss |
|---|---|
| uninterrupted | 3.9158 |
| stopped at step 600, resumed | 3.9158 (bit-identical) |
| resumed without Adam's state | 3.9323 |

Those are on the book's data. On the two-shard data the three runs gave
4.0124, 4.0124 and 4.0313: different numbers, the same verdict. Equality
between the first two is the result; the values themselves depend on the
data.

Then the base model. On your own GPU:

```bash
uv run python stages/11_pretrain/run.py workhorse      # resumable: rerun the same command after a stop
```

30M parameters, 590M tokens, 18,000 steps. The book's run took 2 h 18 min on
the RTX 4060 Laptop and finished at a validation loss of 1.554 on code and
3.293 on English. Yours trains on the two-shard data from module 2: 275M tokens of
unique code against the book's 1,330M, so the run's 500M tokens of code read
it nearly twice.
The book measured that cost directly (chapter 12, the repetition experiment):
at the same budget, 4 passes over the data instead of 1 raised the loss by
0.11. Expect your workhorse to land a little above 1.554 on its own
validation set.

On Colab, or to save the time, download the book's trained workhorse:

```bash
python course/get_checkpoints.py
```

About 240 MB from [huggingface.co/karanjakhar/llm-from-scratch](https://huggingface.co/karanjakhar/llm-from-scratch):
the book's workhorse base and its module 11 SFT model, placed where the
stage scripts look for them. Add `--capstone` for the 92M model (369 MB).
Module 9's test still grades your `save`; the download only replaces the
2-hour run.

## 6. Break it

Edit your `save` to leave out `"loader"`, and run `course/check.py 9` again.
Then put it back and leave out `"opt"`. Which omission does the test catch
with a different loss curve, and which would you only notice by reading the
code? Chapter §8b's last paragraph is the answer.

## 7. Check yourself

<details><summary>1. Why write the checkpoint to a .tmp file and rename it, instead of saving straight over the old one?</summary>

A rename is atomic: the file is either the old checkpoint or the new one. A
crash in the middle of <code>torch.save</code> straight over the old file leaves a
half-written file and no good checkpoint at all.
</details>

<details><summary>2. Losing Adam's state cost +0.057 at step 610 but only +0.017 at the end. Why so mild?</summary>

At step 600 of a cosine schedule the learning rate is already well down, so
the oversized first steps of a fresh Adam move the weights less than they
would early in training. The damage is a permanent offset, not a blow-up.
</details>

<details><summary>3. The first version of the resume test crashed with "RNG state must be a torch.ByteTensor". What was wrong?</summary>

The checkpoint was loaded with <code>map_location="cuda"</code>, which also moved the
saved CPU random-number state onto the GPU. Restoring needs it back on the
CPU. It would have crashed the 20-hour run's first restart.
</details>

<details><summary>4. Why debug every stage on a 30M model and train the 92M model only once?</summary>

A bug found at hour 2 of the workhorse costs 2 hours; the same bug at hour
12 of the capstone costs 12, and its fix needs a restart. The workhorse is
big enough to behave like the real thing and small enough to rerun.
</details>

---

**Stuck?** Solution:
[`llmfs/train/trainer.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/llmfs/train/trainer.py).

**Next:** Module 10, evaluation: does your model write Python that passes
unit tests?
