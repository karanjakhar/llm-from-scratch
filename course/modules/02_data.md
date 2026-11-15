# Module 2: The data is most of the model

**You will build** the parts of a data pipeline that decide whether a model's
numbers can be trusted: a train/validation split that keeps copies of the
same code on one side, exact and near-duplicate removal, and the loader that
feeds the GPU windows of tokens in a shuffled order. Then you build the
training data every later module uses.

**Time:** 3-4 hours, plus the download (2.6 GB) and about 5 minutes of processing.
**Hardware:** CPU; about 5 GB of disk. **Book chapter:** [4, The data is most of the model](../../docs/02_data.md).

---

## 1. Why

A model is a compressed copy of its training data, and its validation score
is only as honest as the split. Code is full of copies: forks, vendored
libraries, files pasted between projects. Split it carelessly and the "unseen"
test set is mostly seen.

What breaks without it: split by file, and 43.9% of the rare lines in the
validation set (lines that occur in at most five files anywhere) also appear
in training. Split by repository and it is still 34.4%, because forks live
under other repositories' names. Only near-duplicate detection catches those.
You will measure all of this yourself.

## 2. Read

Chapter 4 of the book:

- §2 *What is actually in the corpus*: measured, not assumed.
- §4 *Deduplication*: exact, then near-duplicate with MinHash.
- §5 *The train/val split*: by repository, by hash, and the ablation.
- §7 *Packing*: documents end to end, separated by `<|endoftext|>`.
- §10 *Sampling windows from the stream*: why random offsets leave 37% of
  the data unseen after one epoch.

## 3. Implement

Six blanks, in the stage's script and the library it uses:

| # | Function | File | What it does |
|---|---|---|---|
| 1 | `split_of` | `llmfs/data/corpus.py` | train or validation, from a stable hash of the repository name |
| 2 | `Deduper.__call__` | `stages/02_data/prepare.py` | drop byte-identical documents |
| 3 | `signature` | `llmfs/data/dedup.py` | MinHash: a 120-number fingerprint of a file's 5-word phrases |
| 4 | `estimated_jaccard` | `llmfs/data/dedup.py` | how similar two fingerprints say two files are |
| 5 | `ShuffledSampler.next` | `llmfs/data/loader.py` | the next window, each visited once per epoch |
| 6 | `MixedLoader.next_batch` | `llmfs/data/loader.py` | a batch mixing code and English, shifted into inputs and targets |

Every later module trains through your sampler and loader.

## 4. Check

```bash
uv run python course/check.py 2
```

34 tests. The split must not depend on Python's per-process hash seed; no
repository may land on both sides; the deduplicator must drop the validation
copy of a training file, never the reverse; MinHash's estimate must track the
true similarity; the shuffled sampler must visit every window exactly once
per epoch, and a resumed loader must continue exactly where it stopped.

## 5. Run

Download two of the twelve code shards the book used, and the English file
(about 2.6 GB):

```bash
uv run python stages/02_data/download.py --code-shards 2
```

Then measure leakage before removing anything:

```bash
uv run python stages/02_data/run.py leakage --shards 2
```

About 25 seconds. The book ran exactly this, on exactly these two shards, so
your numbers should match to the digit:

| split | rare validation lines also in train | validation files more than half leaked |
|---|---|---|
| by file | 43.9% | 48.6% |
| by repository | 34.4% | 43.3% |

## 6. Break it, then fix it

The table above is the broken version: no near-duplicate removal. Now run
the fix and measure again:

```bash
uv run python stages/02_data/dedup.py
uv run python stages/02_data/run.py leakage --shards 2 --drop data/dedup/near_dups.json
```

Deduplication took 69 seconds on two CPU cores while this course was being
written. On two shards it found and removed 16.7% of training files and 29.2%
of validation files, 156 of the 158 dropped validation files having a near
copy in training. Leakage after it:

| split | rare lines leaked, before → after | files >50% leaked, before → after |
|---|---|---|
| by file | 43.9% → 23.3% | 48.6% → 29.2% |
| by repository | 34.4% → 17.2% | 43.3% → 20.1% |

The book ran deduplication over all twelve shards (it can only find a copy
whose original is in the data), removed 28.2% of training files, and got
repository-split leakage down to 8.1% and 13.8%. More data, more copies
found.

Finally, build the token files every later module trains on:

```bash
uv run python stages/02_data/prepare.py --tokenizer data/tokenizers/split_8192.json --limit-docs 100000
```

`--limit-docs 100000` caps each source at 100,000 documents, which keeps
English to about the 15% share the book used without tokenizing the whole
2 GB file. While writing this course it took 174 seconds on two CPU cores and
produced 275M tokens of training code from 98,598 files, 142M tokens of
English, and about 810 MB of files. That is about a fifth of the book's
1,330M tokens of code, enough for every core module. The validation code is
smaller too: 1.0M tokens from 383 files, against the book's 4.9M, so your
validation losses will move more from run to run than the book's did.

## 7. Check yourself

<details><summary>1. Why split by a hash of the repository name rather than by a random draw?</summary>

A hash gives the same answer on every run and every machine, whatever order
the files arrive in and however many shards are added later. A random draw
would put a repository on a different side each time the data is rebuilt.
</details>

<details><summary>2. Why not use Python's built-in <code>hash()</code> for it?</summary>

It is salted per process (PYTHONHASHSEED): the same string hashes
differently in every run, so the split would change every time. One of the
tests runs it under two different seeds to catch exactly this.
</details>

<details><summary>3. Splitting by repository already keeps a repository's files together. What does near-duplicate removal catch that the split cannot?</summary>

Copies of the same code under different repository names: forks, vendored
libraries, files pasted between projects. The split sees two repositories;
MinHash sees one file.
</details>

<details><summary>4. Drawing random window offsets leaves about 37% of the data unseen after one epoch's worth of draws. Where does 37% come from?</summary>

Each draw misses a given window with probability 1 − 1/N; after N draws the
chance it was never picked is (1 − 1/N)^N ≈ 1/e ≈ 0.37. A shuffled order of
non-overlapping windows visits every one exactly once per epoch instead.
</details>

---

**Stuck?** Solutions: [`llmfs/data/`](https://github.com/karanjakhar/llm-from-scratch/tree/main/llmfs/data)
and [`stages/02_data/prepare.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/02_data/prepare.py).

**Next:** Module 3, embeddings and position: the first layer of the model.
