# Lab Journal

The raw story of this project: every surprise, mistake, correction, trap and
decision, written down **the same session it happened**, with the real numbers.
This is source material for the book (`docs/BOOK.md`). The stage docs explain
the concepts. This file records what actually happened when we built them.

**Why keep it separately.** Six months from now nobody remembers that a test
was wrong, or which "obvious" dataset turned out to ship no data. Those moments
are the most useful thing a book like this can contain, and they are the first
thing to be lost.

### Entry format

```
### YYYY-MM-DD · Stage NN · short title                         [kind]
**Expected:**  what we thought would happen
**Found:**     what actually happened, with numbers
**Why:**       the cause, once understood
**Lesson:**    the general rule a reader can take away
**Where:**     files / doc sections / results JSON
```

`[kind]` is one of:
- **result**: a measured number worth reporting
- **surprise**: reality differed from the expectation
- **mistake**: we got something wrong and caught it
- **trap**: something outside our code that would have bitten anyone
- **decision**: a choice made, and the evidence behind it

Newest entries go at the bottom of each session. Never edit an old entry to
make it look better; add a correction entry instead.

---

## Session 1 — 2026-10-14 · setup and the tokenizer algorithm

### 2026-10-14 · Stage 00 · The 16-bytes-per-parameter rule holds                [result]
**Expected:** Training memory ≈ 16 bytes per parameter: 4 weights + 4 gradients
+ 8 for Adam's two state buffers.
**Found:** A 12.58M-parameter toy model: predicted 192.0 MB, measured 208.3 MB.
Error **8.5%**. Breakdown from the allocator: weights 48.0 MB, gradients plus
activations 64.3 MB, AdamW state 96.0 MB.
**Why:** The extra 16 MB is activations kept alive for the backward pass,
which the rule ignores on purpose.
**Lesson:** Optimizer state is the single largest slice of training memory. A
budget that forgets it is off by 2×, which is the difference between "fits"
and "out of memory".
**Where:** `stages/00_setup/verify_env.py`, `docs/00_setup.md` §2

### 2026-10-14 · Stage 00 · An "8 GB" GPU has 7,797 MB                        [surprise]
**Expected:** 8,192 MB usable.
**Found:** `torch.cuda.get_device_properties(0).total_memory` reports
**7,797 MB**. About 390 MB is reserved before PyTorch gets any.
**Lesson:** Budget against what the runtime reports, not the box.
**Where:** `docs/00_setup.md` §2

### 2026-10-14 · Stage 01 · BPE ran out of things to merge                      [mistake]
**Expected:** Asking for a 512-token vocab gives a 512-token vocab.
**Found:** Training stopped at **127 merges** and two tests failed. The test
corpus repeated about 30 distinct words, and once each word had collapsed
into a single token there were no adjacent pairs left.
**Why:** BPE can only merge pairs that exist. A corpus with few distinct words
supports few merges, however often they repeat.
**Lesson:** The requested vocab size is an upper bound, not a promise. Size the
model's embedding table from the tokenizer that came out, or you get rows that
never receive a gradient: dead parameters and a silent bug.
**Where:** `docs/01_tokenizer.md` §3, `test_training_stops_when_pairs_exhausted`

### 2026-10-14 · Stage 01 · Our own test of the regex split was wrong           [mistake]
**Expected:** With regex pre-tokenization, no learned token ever contains a
space in its middle.
**Found:** The test failed. Tokens like `"\n\t  "` contain interior spaces.
**Why:** The pattern's `\s+` branch deliberately makes whitespace runs their own
chunk, so merging inside them is correct. The test was over-strict.
**Lesson:** When a test fails, check the test's claim before "fixing" the code.
The corrected test excludes all-whitespace tokens and still catches the real
defect (words glued to their neighbours).
**Where:** `stages/01_tokenizer/test_bpe.py`

### 2026-10-14 · Stage 01 · Training on unique chunks: same merges, far faster   [result]
**Expected:** The textbook BPE loop over the whole corpus would be fine on 1 MB.
**Found:** The first experiment run did not finish in over 4 minutes. Rewritten
to count each **unique** chunk once, weighted by how often it occurs, it
produces a **provably identical** merge list (a test compares it against the
naive version), and the test suite went from 8.8 s to 3.25 s.
**Why:** Pre-tokenized chunks repeat enormously. Pair counts over the corpus
equal the sum over unique chunks of (frequency × pairs in the chunk).
**Lesson:** Look for the repetition in your data before optimizing code. This
is the same trick production tokenizer libraries use.
**Where:** `RegexBPE.train`, `test_weighted_dedup_training_equals_naive`

### 2026-10-14 · Stage 01 · Pure Python is 27× slower than tiktoken              [result]
**Found:** Encoding 100k characters: ours 0.53 MB/s, tiktoken 14.4 MB/s.
**Why:** Partly the interpreter, partly the algorithm: our encoder rescans for
the best pair every step, while tiktoken uses a linked list and a heap.
**Lesson:** Production tokenizers are never written in Python, and the reason
is algorithmic as much as linguistic.
**Where:** `stages/01_tokenizer/run.py` section 3

### 2026-10-14 · Stage 01 · A tokenizer trained on English punishes other languages [surprise]
**Found:** Trained on Shakespeare at vocab 1,024, `héllo` and `你好` encoded as
raw bytes with no merges at all. `你好` became 6 tokens.
**Why:** The corpus is pure ASCII, so no multi-byte character ever got merged.
**Lesson:** A tokenizer's training data decides who pays more per word. Users of
under-represented languages get less context for the same price. This is
visible at layer zero, before any model exists.
**Where:** `docs/01_tokenizer.md` §2, §7

### 2026-10-14 · Tooling · `pkill -f` killed the command that ran it            [trap]
**Found:** `pkill -f "stages/01_tokenizer/run.py"` killed the shell executing
it, so the edits queued after it never ran.
**Why:** `-f` matches anywhere in a process's full command line, and the
shell's own command line contained that string.
**Lesson:** Kill by PID, or anchor the pattern so it cannot match itself.
**Where:** session log

### 2026-10-14 · Repo · `data/` in .gitignore also ignored `llmfs/data/`         [trap]
**Found:** The package directory `llmfs/data/` was silently missing from the
first commit.
**Why:** An unanchored `data/` pattern matches a directory with that name at
any depth.
**Lesson:** Anchor ignore rules meant for the repo root: `/data/`.
**Where:** `.gitignore`

---

## Session 2 — 2026-10-14 · choosing the domain

### 2026-10-14 · Domain · Choose by verifiability, not usefulness              [decision]
**Expected:** Pick the most *useful* domain.
**Found:** The deciding property turned out to be whether outputs can be
**checked by a machine**. With Python code, the RL reward in Stage 20 is "do the
unit tests pass" and the eval in Stage 22 is pass@k. Stories, prose, or a
low-resource language would push those stages onto an LLM judge, which at
125M parameters measures noise.
**Lesson:** For a learning project, the domain with a verifier makes half the
curriculum into real experiments.
**Where:** `docs/domain_choice.md`

### 2026-10-14 · Hardware · The ceiling is ~125M parameters                     [result]
**Found:** With `C ≈ 6ND`, Chinchilla's ~20 tokens per parameter, and 15-25%
utilization of the 4060's ~58 TFLOPS bf16: 30M params ≈ 2-3.5 h,
125M ≈ 36-60 h, 350M ≈ 280-470 h.
**Lesson:** Do the compute arithmetic before choosing a model size. It turns
"how big can we go?" into a table.
**Where:** `docs/domain_choice.md` §1

### 2026-10-14 · Data · The obvious Python dataset contains no Python            [trap]
**Expected:** `HuggingFaceTB/python-edu`, quality-scored Python, is the best
starting corpus.
**Found:** Its schema is `blob_id, repo_name, path, length_bytes, score`:
metadata only. The code itself has to be fetched from Software Heritage's S3
storage by blob id, ~500k authenticated requests. It is also not its own repo
but a config of `HuggingFaceTB/smollm-corpus`.
**Lesson:** Read the schema, not the dataset card's prose.
**Where:** `docs/datasets.md`

### 2026-10-14 · Data · Every BigCode corpus is gated                           [trap]
**Found:** `bigcode/starcoderdata`, `the-stack-v2`, even `the-stack-smol` all
require accepting terms and a token.
**Lesson:** Check access before building a plan on a dataset.
**Where:** `docs/datasets.md`

---

## Session 3 — 2026-10-25 · the data pipeline

### 2026-10-25 · Environment · The Python environment disappeared                [trap]
**Found:** The conda environment used in sessions 1-2 was gone. `python3` was
now system Python 3.10 with no torch. Rebuilding exposed a second problem:
`regex`, which Stage 01 imports, had never been declared as a dependency. It
had only worked because conda happened to ship it.
**Lesson:** If the project's environment is not defined in files it controls
(`pyproject.toml` + `uv.lock`), it is not defined. Always run through `uv run`.
**Where:** `pyproject.toml`, `CLAUDE.md`

### 2026-10-25 · Data · The dataset was 2.5× bigger than reported               [mistake]
**Expected:** `codeparrot-clean` = 477k files, 5.0 GB (recorded in session 2).
**Found:** The real dataset is **54 shards, 12.8 GB gzipped, ~53 GB of text,
~5.4M files**. The session-2 number came from the HF datasets-server, which
converts only the first ~5 GB of large datasets (its path says
`partial-train`).
**Lesson:** Know where a number came from. An API that summarizes a dataset may
be summarizing part of it.
**Where:** `docs/datasets.md`, `docs/02_data.md` §2

### 2026-10-25 · Stage 01 · Llama 3 does not split digits                        [mistake]
**Found:** The Stage 01 doc said Llama 3 splits every digit. It uses GPT-4's
`\p{N}{1,3}`. Per-digit splitting is Llama 1/2 (SentencePiece `split_digits`)
and Qwen2. Corrected in four files.
**Lesson:** Facts about other people's models go stale or are misremembered.
Check before writing them into teaching material.
**Where:** `docs/01_tokenizer.md` §4

### 2026-10-25 · Stage 02 · Profile the data before writing filters             [surprise]
**Expected:** Filter out auto-generated files, very long lines, and
low-alphabetic files.
**Found:** In shard 1 (100k files): `autogenerated` was False for all 100,000,
`line_max ≤ 999` for all, `alpha_frac ≥ 0.25` for all. Upstream had already
applied exactly these filters.
**Lesson:** Filters copied onto already-cleaned data do nothing. Filters copied
onto data with a different distribution do damage. Measure first.
**Where:** `docs/02_data.md` §2

### 2026-10-25 · Stage 02 · Repositories are scattered across shards            [surprise]
**Found:** 47,232 distinct repos in 100,000 files (median 1 file per repo per
shard). Files of one repository land in many shards.
**Lesson:** A split by shard leaks whole repositories into both train and val.
The split has to hash the repository name.
**Where:** `docs/02_data.md` §2, §5

### 2026-10-25 · Stage 02 · 58% of the code is permissively licensed           [result]
**Found:** By bytes: apache-2.0 23.1%, mit 16.4%, bsd-3 16.0%, others 2.2% =
**57.7% permissive**. GPL family and other copyleft make up the rest.
**Decision:** Keep permissive only, so the trained weights are publishable
without an asterisk. At ~53 GB available against ~6 GB needed, it costs
nothing we would have used.
**Where:** `docs/02_data.md` §3

### 2026-10-25 · Stage 01 · The "too slow" trainer was fast enough             [surprise]
**Expected:** The Stage 01 trainer would need rewriting for real data.
**Found:** On 20 MB of real Python: 4.9M chunks but only **110k unique**, so
**0.22 s per merge**, ~30 min for an 8k vocab. No rewrite needed.
**Lesson:** Measure before rewriting. The dedup trick from session 1 had
already solved the problem.
**Where:** `PROGRESS.md` results log

### 2026-10-25 · Stage 01 · Merge lists are prefixes of each other              [result]
**Found:** Training is greedy and deterministic, so the first k merges of a
16k run *are* the k-merge tokenizer. One run per variant gives 2k, 4k, 8k and
16k. Now asserted by a test.
**Lesson:** Look for structure that lets one experiment answer many questions.
**Where:** `test_merges_are_a_prefix_sequence`

### 2026-10-25 · Stage 02 · 43% of the exam was copied from the textbook         [surprise]
**Expected:** A repository-level train/val split prevents leakage.
**Found:** Even with the repo split, **34.4%** of rare val lines (lines in ≤ 5
files corpus-wide) appear verbatim in train, and **43.3%** of val files have
most of their lines in train. A file-level split was worse: 43.9% and 48.6%.
**Why:** Forks, vendored libraries and copy-pasted files live under other repo
names, where no repo-level split can see them. Upstream had removed *exact*
duplicates only.
**Lesson:** "Deduplicated" can be true and still not mean what you need. Measure
leakage directly. Line overlap, restricted to rare lines to exclude
boilerplate, is a cheap way to do it without training anything.
**Where:** `stages/02_data/run.py leakage`, `results/02_data/leakage.json`

### 2026-10-25 · Stage 02 · Near-dedup removed 28% of the training code          [result]
**Found:** MinHash (120 permutations, word 5-grams) + LSH (20 bands × 6 rows) +
union-find over 729,865 files in **149 s**: **28.2%** of train files dropped
(205,241), **39.5%** of val (1,264), and **1,258 of those 1,264** had a
near-copy in train. Leakage after near-dedup, repo split: rare lines 34.4% →
**8.1%**, majority-leaked val files 43.3% → **13.8%**.
**Lesson:** More than a quarter of "clean" GitHub Python is near-duplicate.
Without this step, ~40% of the code validation set would have tested
memorization.
**Where:** `stages/02_data/dedup.py`, `results/02_data/near_dedup.json`

### 2026-10-25 · Stage 02 · The repo split and near-dedup catch different leaks [surprise]
**Expected:** After near-dedup, the split strategy would matter less.
**Found:** The gap between file-level and repo-level splitting *grew*: +9.5 →
**+13.6** points of rare-line leakage. Near-dedup with a file split still left
21.7%.
**Why:** Near-dedup removes cross-repo copies. What remains is files *within*
one repository sharing helpers and code, which only the repo split catches.
**Lesson:** Two defenses can look redundant and be complementary. Measure each
one alone.
**Where:** `docs/02_data.md` §5

### 2026-10-25 · Stage 02 · The 8.1% that remains                                [result]
**Found:** After both defenses, 8.1% of rare val lines still appear in train.
**Why (believed, not proven):** Partial copies, e.g. one vendored function
inside an otherwise original file. Whole-file Jaccard cannot detect them.
**Decision:** Document it rather than chase it. Val loss will be slightly
optimistic, and the HumanEval/MBPP numbers in Stage 22 do not depend on this
split at all.
**Where:** `docs/02_data.md` §5, `PROGRESS.md` open questions

### 2026-10-25 · Stage 02 · Python's `hash()` would have broken dedup silently   [trap]
**Found:** Signatures are computed in 18 worker processes. Python's built-in
`hash` is salted per process (`PYTHONHASHSEED`), so each worker's signatures
would be incomparable with every other's, and nothing would ever match. No
error, just zero duplicates found.
**Lesson:** Anything compared across processes needs a stable hash (`zlib.crc32`,
`blake2b`). A test now computes signatures in spawned processes and asserts
they are identical.
**Where:** `llmfs/data/dedup.py`, `test_signatures_identical_across_processes`

### 2026-10-25 · Stage 02 · fork() after importing torch can deadlock            [trap]
**Found:** A `DeprecationWarning` in the test output: "This process is
multi-threaded, use of fork() may lead to deadlocks in the child."
**Why:** Importing the loader imports torch, which starts threads. `fork()`
copies only the calling thread, so a lock held by another thread stays locked
forever in the child.
**Lesson:** Read warnings. Worker pools now use `spawn`, and the suite runs with
that warning promoted to an error.
**Where:** `stages/02_data/prepare.py`

### 2026-10-25 · Stage 01 · An hour-long job with no progress output            [mistake]
**Found:** The production tokenizer run (3 × 16k merges, ~1 h) printed nothing
between start and finish. A crash and a slow run looked identical.
**Lesson:** Any job longer than a few minutes prints progress, at least a line
every few minutes.

### 2026-10-25 · Stage 01 · Our 8k tokenizer beats GPT-2's 50k on code           [surprise]
**Found:** Held-out Python, bytes per token: GPT-2 (50,257 vocab) **2.222**,
ours at 8,192 **3.381** (grouped digits) / **3.337** (split digits), GPT-4's
cl100k (100,277) 4.244. On English GPT-2 still wins: 4.570 vs our 3.369.
**Why:** GPT-2 was trained on web text and never learned tokens for runs of
spaces, so every space of Python indentation costs it a whole token.
**Lesson:** For a tokenizer, the training distribution matters more than the
vocabulary size. This is why code models add whitespace tokens.
**Where:** `docs/01_tokenizer.md` §9, `results/01_tokenizer/production.json`

### 2026-10-25 · Stage 01 · Production tokenizer: split digits, 8,192 tokens     [decision]
**Found:** 8k → 16k vocab: +6.1% on code for +4.2M embedding parameters (d=512,
about 17% of a 50M model). Digit rules at 8k: grouped 3.381, cap-at-3 3.376,
split 3.337 bytes/token.
**Decision:** `split_8192`. Splitting costs 1.3% and gives every digit a stable
id, so `range(100)` is `range ( 1 0 0`. Reversible by re-running `prepare.py`.
**Where:** `docs/01_tokenizer.md` §9, `CLAUDE.md`

### 2026-10-25 · Stage 01 · Saved tokenizers were missing their special tokens    [mistake]
**Found:** `train_production.py` saved all 12 tokenizers without
`<|endoftext|>`. `prepare.py`'s check refused to run on them.
**Lesson:** Assert your preconditions at the boundary. That one-line assert
turned a silent bug (documents with no boundaries) into an immediate error.
Fixed in the script and patched into the saved files rather than retraining
for an hour.

### 2026-10-25 · Stage 02 · A cache makes the encoder 12× faster                 [result]
**Found:** Caching chunk → token ids: **8.14 MB/s vs 0.66 MB/s** on one core,
**98.2%** cache hit rate from a cold start, identical ids on real files (test).
**Lesson:** Zipf's law is an optimization opportunity. A few thousand chunks
(`"    "`, `" self"`, `" return"`) make up almost everything.
**Where:** `llmfs/tokenizer/bpe.py`

### 2026-10-25 · Stage 02 · 2.36 billion tokens in 4.5 minutes                   [result]
**Found:** `train_code` 1,330.1M, `train_en` 1,016.9M, `val_code` 4.9M,
`val_en` 5.2M tokens, 4.4 GB on disk, ~36 MB/s across 18 workers. That is
about 4× below the workers' combined capacity. The suspected bottleneck is
the single main process reading gzipped JSON; this has not been profiled.
**Where:** `results/02_data/manifest_split_8192.json`

### 2026-10-25 · Stage 02 · The "failed" check that found a duplicate article    [mistake]
**Found:** Decoding the first 200 `val_en` documents and comparing them with
the source failed at document 174. That document was a **verbatim copy** of an
article already in `train_en` (a children's-book worksheet published under
two IDs). The shared deduper had correctly removed it from val. The check
ignored exact dedup, so everything after it was off by one.
**Also found:** fineweb-edu has exact duplicates: 1,476 dropped from train,
13 from val.
**Lesson:** When a verification fails, prove which side is wrong before fixing
anything. Here it was the check.
**Where:** `docs/02_data.md` acceptance criteria

### 2026-10-25 · Stage 02 · The loader is 700-1,750× faster than needed          [result]
**Found:** `MixedLoader`, B=16, T=1024: 56M tokens/s (CPU), 125M (GPU), 140M
(GPU, pinned memory), against a model need of ~80k tokens/s.
**Caveat:** The CPU row ran first, while the file was not yet in the OS page
cache, so the rows are not cleanly comparable. All clear the bar by two orders
of magnitude.
**Decision:** No multi-worker `DataLoader`; it would be complexity with no payoff.
**Where:** `results/02_data/throughput.json`

### 2026-10-25 · Figures · The interactive tokenizer lab is the real algorithm    [result]
**Found:** The browser BPE in the visual build map produced the same first 40
merges as the Python `RegexBPE` on the same text.
**Lesson:** A teaching visual should be checked against the real code. A
lookalike teaches the wrong thing convincingly.
**Where:** `docs/visuals/stages_00-02_build_map.html`
