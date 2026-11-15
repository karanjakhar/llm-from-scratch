# Module 10: Measuring what matters

**You will build** the three pieces that turn "the loss went down" into "the
model solves this many programming problems": cutting a completion where the
function ends, running it against unit tests safely, and the unbiased pass@k
estimator. Then you score your base model on HumanEval and MBPP.

**Time:** 2-3 hours. **Hardware:** GPU. **Book chapter:** [24, Measuring what matters](../../docs/22_evaluation.md).

---

## 1. Why

Loss tells you a model predicts text better. It does not tell you whether the
model can do anything. For code there is a better question with an objective
answer: write the function, run the tests, count the passes. That is why
this project chose Python: every result in this module is a number nobody
has to judge.

What breaks without the details: the first version of this harness scored
the workhorse at exactly zero, and all 164 completions were empty. HumanEval
prompts end with a newline, which this tokenizer encodes as a bare `"\n"`
token; in training, a bare newline almost always meant "the next line starts
in column 0". So the model began a new top-level function instead of writing
the body, and truncation correctly cut it all away. The tests ran, failed
and reported a real-looking 0%. The fix, stripping the prompt's trailing
newline, is in `evaluate`, which you are not writing; the lesson is that
where a token boundary falls is part of the prompt.

## 2. Read

Chapter 24 of the book:

- §1 *Perplexity: necessary, not sufficient*.
- §3 *Generative benchmarks and pass@k*, including why the naive estimate of
  pass@k is biased.
- §5 *Contamination*: how to check that a solved problem was not in the
  training data.
- §8a *The table*: every model the book trained, through one harness.

## 3. Implement

Three blanks, in the evaluation library the later stages share:

| # | Function | File | What it does |
|---|---|---|---|
| 1 | `truncate` | `llmfs/eval/codeeval.py` | cut a completion where the function ends |
| 2 | `pass_at_k` | `llmfs/eval/codeeval.py` | the unbiased estimator from n samples with c correct |
| 3 | `run_program` | `llmfs/eval/execution.py` | run code with a timeout and a memory cap, and only count a pass if it reached the end |

Before the check, download HumanEval and MBPP (once), or three of the tests
are skipped:

```bash
uv run python stages/02_data/download_extras.py
```

## 4. Check

```bash
uv run python course/check.py 10
```

10 tests. Among them: pass@k against known values; a program that loops
forever times out; a program that calls `sys.exit(0)` before its asserts is
**not** a pass; all 164 HumanEval reference solutions pass under your
harness and wrong answers fail.

**A warning.** `run_program` executes code a model wrote. It runs in a
separate process with a time limit and a memory cap, which stops runaway
loops; it is not a security sandbox. At 30M parameters the risk is close to
nothing, but if you point this at a larger model or change the prompts, run
it inside a container or a throwaway VM.

## 5. Run

```bash
uv run python stages/22_evaluation/run.py table --redo
```

`--redo` matters: without it, the script reuses the book's committed rows
and evaluates nothing. It scores every checkpoint you have: the validation
loss, and greedy pass@1 on HumanEval's 164 problems and MBPP's 500.
About 9 minutes per model on the book's GPU, most of it generating and
running 664 programs.

| model | val code loss | HumanEval pass@1 | MBPP pass@1 |
|---|---|---|---|
| workhorse base (book) | 1.684 | 0 | 0 |
| capstone base, 92M (book) | 1.423 | 3.05% | 0.2% |

Run while writing this course on the book's workhorse with the two-shard
validation set: loss 1.671, and zero on both benchmarks, as in the book. Zero is the honest answer for a 30M model choosing its single most
likely completion. Add `--pass10` to draw 10 samples per problem at
temperature 0.8: the book's workhorse then solved 1.22% (2 of 164).
Run while writing this course, it took 21 minutes for the one model and
reproduced the 1.22%.

## 6. Break it

1. In `run_program`, count any exit code 0 as a pass (drop the sentinel
   check). Run the tests. Then think about what a model trained with "tests
   pass" as its reward (chapter 22) would learn to do.
2. Replace `pass_at_k` with the naive "fraction of problems where at least
   one of the first k samples passed" and compare on n = 10, c = 1, k = 5.
   The unbiased estimator gives 0.5; what does the naive one give on average,
   and why is it noisier?

## 7. Check yourself

<details><summary>1. Why is validation loss alone not enough to compare these models?</summary>

The book's three small models differed by up to 0.085 in code loss and all
scored exactly zero on both benchmarks. Loss sees differences that the tests
cannot yet, and says nothing about whether any program runs.
</details>

<details><summary>2. One HumanEval problem is 0.61 percentage points. The same checkpoint scored 0 and then 1 of 164 in two runs with the same settings. How should you read a difference of one problem?</summary>

As a tie. Greedy decoding in bf16 can flip at near-ties when the batch
changes, so one or two problems either way is noise at this scale.
</details>

<details><summary>3. What does a 13-gram contamination check look for, and what did the book find?</summary>

Any 13-token run of a benchmark's reference solution that also appears in
the training data. None of the 9 distinct problems the book's models solved
had a single 13-gram in the 1.33B training tokens: the scores are small and
clean.
</details>

<details><summary>4. pass@10 came out lower than pass@1 for one of the book's models. Is that a bug?</summary>

No: they come from different decoding. Greedy pass@1 takes the single most
likely completion; pass@10 samples at temperature 0.8, which spends its tries
elsewhere. Only the same decoding makes them comparable.
</details>

---

**Stuck?** Solutions:
[`llmfs/eval/codeeval.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/llmfs/eval/codeeval.py)
and [`llmfs/eval/execution.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/llmfs/eval/execution.py).

**Next:** Module 11, instruction tuning: teach the base model to answer a
request, then measure what that cost with the harness you just built.
