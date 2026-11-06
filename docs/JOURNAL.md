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

---

## Session 5 — 2026-10-29 · embeddings and position (Stage 03)

### 2026-10-29 · Stage 03 · The RoPE test failed on float32, not on maths        [mistake]
**Expected:** The score of a rotated `q` and `k` depends only on `m - n`, so
every diagonal of the 128 × 128 score matrix is constant to 1e-5.
**Found:** Two tests failed with a drift of **1.37e-5** on scores of size ~12.4
(about 1e-6 relative). The test converted angles to float64, but only *after*
`rope_angles` had already rounded them to float32.
**Why:** The fastest RoPE pair turns 1 radian per token, so its angle *is* the
position. float32 keeps ~7 significant digits, so at position 127 the angle is
off by up to ~4e-6 rad, and at position 100,000 by ~4e-3.
**Lesson:** Prove which side is wrong before fixing. The code was right, the test
asked for more precision than the input had. Also a real effect worth knowing:
RoPE angle precision degrades linearly with position, which matters for
long-context models.
**Where:** `stages/03_embeddings/embeddings.py` `rope_angles(dtype=...)`

### 2026-10-29 · Stage 03 · Our RoPE matches HuggingFace Llama exactly            [result]
**Found:** The half-split layout equals `LlamaRotaryEmbedding` +
`apply_rotary_pos_emb` from transformers 5.20.2 to 1e-5. The interleaved layout
equals the complex-number form. The two layouts differ only by a fixed shuffle
of dimensions; without the shuffle they disagree.
**Lesson:** There are two "correct" RoPEs and loading weights trained with one
into the other fails silently. Any port of real weights has to check this.
**Where:** `stages/03_embeddings/test_embeddings.py`

### 2026-10-29 · Stage 03 · Sinusoidal positions were worse than none at all     [mistake]
**Expected:** Sinusoids roughly match learned positions (val loss ~2.7).
**Found:** **4.723 ± 0.080** over 3 seeds, worse than no position at all
(3.350). Training loss fell every step, with no error or NaN.
**Why:** A sinusoid vector has length √(C/2) = **11.3** at C = 256. Our token
rows start with length **0.32** (std 0.02). Position outweighed the token
**35:1**. The 2017 paper multiplies token embeddings by √C before adding; we
had left that out. With it: **2.633 ± 0.001**, better than learned (2.689).
**Lesson:** When two signals are added, check they are the same size. The
failure was silent, and only the comparison against other variants exposed it.
One line from the paper was worth 2.1 nats.
**Where:** `docs/03_embeddings_position.md` §8, `stages/03_embeddings/tiny_lm.py` `emb_scale`

### 2026-10-29 · Stage 03 · Position is worth 0.83 nats; RoPE is best             [result]
**Found:** 4 layers, C = 256, 32.8M tokens, 3 seeds, held-out code loss:
none 3.350, learned 2.689, sinusoidal (scaled) 2.633, **RoPE 2.523**. Seed
noise ±0.01. RoPE wins with zero position parameters; the learned table spends
65,536 and loses by 0.17.
**Lesson:** A causal model without position does learn *some* order through
the mask (it beats broken sinusoids by 1.4), but nowhere near enough. The
curriculum's "cannot learn word order" is too strong; "learns it badly" is
what we measured.
**Where:** `results/03_embeddings/ablation.json`, `docs/visuals/03_ablation.png`

### 2026-10-29 · Stage 03 · Untying the head: +40% params, worse loss             [surprise]
**Expected:** A separate output head adds capacity, so equal or slightly better.
**Found:** Untied RoPE model: 7.36M params vs 5.26M, loss **2.598 vs 2.523**,
consistent across all 3 seeds.
**Why (believed):** Tied, the table gets a gradient on every row every step
from the output softmax. Untied, input rows learn only from tokens present in
the batch, so 2.1M parameters learn slowly at 32.8M tokens of training.
Not tested whether the gap closes with longer training.
**Lesson:** At small scale, weight tying is free quality, not only saved memory.
**Where:** `results/03_embeddings/ablation.json`

### 2026-10-29 · Stage 03 · "RoPE extrapolates" means "no wall", not "works"     [surprise]
**Expected:** RoPE degrades gracefully past the training length; sinusoids less so.
**Found:** On 512-token windows after training on 256: RoPE holds for ~40
tokens (2.23 at positions 256-271), then climbs to 3.89 at 496-511. Scaled
sinusoids jump straight from 2.44 to 3.56 and reach 5.28. The no-position model
degrades least (+0.21) but from a much worse base. Learned cannot run.
**Why:** RoPE only sees distances. Just past the edge, most attention is to
nearby tokens at distances it trained on; the share at unseen distances grows
with position.
**Lesson:** Test claims about long context by measuring loss by position.
Averages hide where the cliff is.
**Where:** `docs/03_embeddings_position.md` §8, `docs/visuals/03_ablation.png`

---

## Session 6 — 2026-10-30 · attention (Stage 04)

### 2026-10-30 · Stage 04 · Our attention matches PyTorch, twice                  [result]
**Found:** Outputs and gradients equal `F.scaled_dot_product_attention` to 1e-5,
causal and not; the multi-head module equals `nn.MultiheadAttention` with copied
weights. All 12 tests passed first run. In training, the Stage 03 model with
our attention in place of PyTorch's scored **2.6856** vs **2.6849** (seed 0).
**Lesson:** Check against the reference numerically, then once more in a real
training run. The second check catches what unit tests cannot (dtype, autocast,
masking at real sizes).
**Where:** `stages/04_attention/test_attention.py`, `results/04_attention/ablation.json`

### 2026-10-30 · Stage 04 · "Gradients vanish" depends on where you look           [surprise]
**Expected:** Without √d, gradients shrink as `d` grows.
**Found:** Random unit-variance q, k, v, 256 keys. Without √d, the gradient at
the **scores** falls from 1.23 (d = 64) to **0.022** (d = 1024), with **41%** of
queries getting < 1e-3. But the gradient at **q** grows: 13.2 → **63.1**
(scaled: 0.78 → 3.1).
**Why:** grad(q) = grad(scores) × k. Unscaled, k enters at full size √d. The
softmax's vanishing slope and k's growing size partly cancel.
**Lesson:** A claim about gradients must say *which* gradient. Measure at the
place the mechanism acts.
**Where:** `results/04_attention/variance.json`, `docs/04_attention.md` §9a

### 2026-10-30 · Stage 04 · Dropping √d: harmless at d=64, broken at d=256        [result]
**Found:** 4-layer C=256 model, 32.8M tokens, 3 seeds, held-out code loss:
d=64 scaled 2.689, unscaled **2.652**; d=256 scaled 2.705, unscaled **3.655**.
The broken model's attention: top weight 0.78, entropy 0.92 nats (≈2.5 tokens
read per position, vs ≈16 scaled).
**Also found:** the weight gradient ‖∇W_qkv‖ at the end was *larger* in the
broken run (0.97 vs 0.51). The damage is real; the textbook mechanism
(vanishing weight gradients) is not what we measured. Why saturation then
hurts learning is still open.
**Where:** `results/04_attention/ablation.json`, `docs/04_attention.md` §9c

### 2026-10-30 · Stage 04 · √d made our small model start too flat               [surprise]
**Expected:** Scaling never hurts.
**Found:** At d=64, unscaled beat scaled on all 3 seeds by 0.037 (noise ~0.01).
Score std at initialisation: **0.10** scaled, **0.81** unscaled, **1.61** at
d=256 unscaled. With std-0.02 weights, scaled scores start so small that
attention is nearly uniform.
**Lesson:** √d keeps the score scale constant across head sizes; it does not
choose a good scale. The right scale also depends on initialisation and
training length. Real models (larger init after LayerNorm, longer training)
grow their scores further, which is where the guard matters.
**Where:** `docs/04_attention.md` §9c

### 2026-10-30 · Stage 04 · Naive attention keeps four T×T matrices alive         [result]
**Found:** B=1, H=8, d=64, bf16, forward+backward: ours 66 MB at T=1,024 →
4,168 MB at 8,192 → **out of memory at 16,384** (~17 GB needed). Fused kernel:
8.1 → 64.5 → 129 MB, and 14× faster at 8,192. One bf16 T×T matrix at T=1,024
is 16.8 MB, so ours holds ~4 copies (scores, masked scores, weights, their
gradient).
**Lesson:** The doc's back-of-envelope (one copy) was 4× optimistic. Measure
peak memory, do not count tensors in your head.
**Where:** `results/04_attention/cost.json`, `docs/visuals/04_cost.png`

### 2026-10-30 · Stage 04 · A previous-token head and two sinks, unasked          [surprise]
**Found:** In the trained d=64 model (5.3M params, ~3 minutes of training),
layer 2 head 2 puts **68%** of its weight on the previous token. Layer 3
heads 0 and 1 put **24%** and **37%** on the first token of the window, which
is an arbitrary token because windows start at random offsets.
**Why:** The sinks are the known "attention sink" effect (Xiao et al., 2023):
softmax weights must sum to 1, so a head with nothing to read parks weight on
a position every token can see.
**Lesson:** Attention maps show recognisable structure, and also show weight
that is not used. Weight is not evidence of use.
**Where:** `results/04_attention/heads.json`, `docs/visuals/04_attention_*.png`

---

## Session 7 — 2026-10-31 · the book gets a web edition

### 2026-10-31 · Book · A web edition, built from the stage docs in place        [decision]
**Expected:** Use a book toolchain (Quarto: one source to web, PDF and EPUB).
**Found:** A first preview, stitched straight from `docs/*.md` into one
reading page, was what the owner wanted on karanjakhar.dev. Quarto was not
installed and its install was not approved this session, so it was dropped
before anything was rendered with it. The web edition is now
`book/build_web.py`: 8 of 30 planned parts drafted, 21,740 words, 7 figures.
The chapter list comes from the `docs/BOOK.md` §3 table, so a chapter appears
on the site as soon as its status there says `draft`.
**Why:** The drafts already are the book. A builder that reads them unchanged
means there is no second copy to keep in sync.
**Lesson:** Building the preview showed a gap no review had: the stage docs
named their figures by path (`docs/visuals/04_cost.png`) but never embedded
them. A reader saw text about a chart and no chart. Seeing the whole thing
laid out end to end catches what reading one file at a time does not.
**Where:** `book/build_web.py`, `book/template.html`, `docs/book_intro.md`;
figure embeds added to `docs/03_embeddings_position.md` §7-8 and
`docs/04_attention.md` §9

---

## Session 8 — 2026-11-06 · the transformer block

### 2026-11-06 · Stage 05 · Our block equals Hugging Face's GPT2Block                [result]
**Expected:** A pre-norm LayerNorm/GELU block with GPT-2's weights copied in
reproduces `GPT2Block` to 1e-5.
**Found:** It does, after two adjustments on the *test* side. (1) GPT-2's
linear layers are `Conv1D`, stored `(in, out)`; every weight needs a `.T`.
Copying the square `c_proj` without it raises no error and moves the output
by more than 1e-2 (kept as `test_block_without_transpose_is_wrong`). (2) HF's
eager GPT-2 attention applies no causal mask when the block is called on its
own; the model builds the mask and passes it in. The test has to pass an
additive `-inf` mask itself. GPT-2's GELU is the tanh approximation, which
differs from the exact one by up to ~5e-4, so that one is needed too.
**Lesson:** A reference block is only a reference for the inputs it is given.
Read what the reference expects from its caller before trusting a mismatch,
or a match.
**Where:** `stages/05_block/test_block.py`, `docs/05_transformer_block.md` §9

### 2026-11-06 · Stage 05 · An untrained tied model "predicts" its own input           [mistake]
**Expected:** An untrained model's loss is about `ln(8196) = 9.01`.
**Found:** 8.46. The test passed the input ids as the targets. With random
targets, 9.07.
**Why:** With a tied head, the logit for token `i` contains `⟨E[i], E[i]⟩ =
|E[i]|² > 0` because the token's own embedding is in the stream. An untrained
model already leans towards repeating its input. Our mistake was the test;
the effect is real and is now its own test.
**Lesson:** A bug that copies inputs to targets can look like early learning
in a tied model. Check the loss at step 0 against `ln V` with real
next-token targets.
**Where:** `stages/05_block/test_block.py::test_tied_head_at_init_predicts_its_own_input`

### 2026-11-06 · Stage 05 · Without residuals, gradients did not vanish; tokens merged   [surprise]
**Expected:** (docs §2) In a 24-layer model without residual connections, the
gradient reaching block 1 is a tiny fraction of block 24's (the `0.8^24`
story).
**Found:** Untrained, 5 seeds, one real batch each. Block-1 gradient norm
**3.65**, block-24 **2.63**: nearly flat, and larger than pre-norm *with*
residuals (2.21 / 0.38). What broke instead: the mean cosine similarity
between different positions' vectors went 0.012 (embedding) → **0.75** after
one block → 0.989 after block 8 → **1.000** from block 12. Every position
carried the same vector.
**Why:** The LayerNorm in front of each sublayer rescales the signal back to
size 1 on the way up and divides the gradient by the input rms on the way
down, so a shrinking product never forms. Attention at init is close to a
plain causal average, and averages of overlapping sets look alike. Without a
residual, that average replaces the token's own vector at every layer. This
is rank collapse (Dong et al., 2021).
**Lesson:** "Residuals stop gradients vanishing" is the textbook reason, but
in a normalised network the measurable failure is different: the tokens
become indistinguishable, and the large gradients carry no per-token
information. Measure the representation, not only the gradient.
**Where:** `results/05_block/grads.json`, `docs/visuals/05_init_grads.png`,
`docs/05_transformer_block.md` §10a

### 2026-11-06 · Stage 05 · Post-norm starts out calm at our init                  [surprise]
**Expected:** (docs §4, after Xiong et al. 2020) Post-norm gradients misbehave
at initialisation, worst at the top layers.
**Found:** 24-layer post-norm, untrained: every block from 2 to 24 gets
gradient norm 0.119 to 0.122, essentially flat. Block 1 gets 1.32 (it reads
the raw, tiny embedding). Positions stay distinct (cosine 0.03 at the top).
**Why:** With weights at std 0.02, each sublayer's output is small next to
the normalised stream (rms 1), so `Norm(x + f(x)) ≈ x`. The model begins as
a near-identity chain. Xiong et al.'s analysis assumes an init where `f(x)` is
as large as `x`.
**Lesson:** Stability claims come with an initialisation attached. The
interesting question is whether post-norm stays calm once training grows the
weights (§10b).
**Where:** `results/05_block/grads.json`, `docs/05_transformer_block.md` §10a

### 2026-11-06 · Stage 05 · Post-norm and no-residual both learned exactly nothing about context   [result]
**Expected:** (curriculum) Pre-norm trains at 24 layers where post-norm
diverges. Removing the residual makes training fail.
**Found:** Same recipe as Stage 04, seed 0. Pre-norm: 2.685 (4 layers),
2.567 (24 layers). Post-norm: **6.856** (4 layers), **6.857** (24 layers).
No residual: **6.856** / **6.856**. Nothing diverged (no NaN). All four failed
runs reached ~6.9 by step 50 and stayed there for 3,950 steps. The cross-entropy
of plain token frequencies (85/15 train mix) on the same 512 val windows is
**6.860**. The failed models predict how common each token is, and ignore
their input, even the current token. Gradient to block 1 by step 1000:
2.7e-9 (post, 4L), 1.2e-12 (post, 24L), 3.8e-19 (no residual, 24L), against
0.45-0.61 for pre-norm.
**Why:** The frequency answer is a constant function of the input. Once the
top of the network outputs a near-constant, its derivative with respect to
its input is ~0, which cuts the gradient to every block below. Nothing below
can then learn to use context. A trap that, once entered, closes itself.
**Lesson:** "Diverges" was the wrong prediction: the failure is a silent
plateau at a recognisable number. Compute the unigram loss of your data
once; a run sitting at it has learned nothing about context. And post-norm
failed at 4 layers too, so in our setup this was not about depth.
**Where:** `results/05_block/depth.json`, `docs/visuals/05_depth.png`,
`docs/05_transformer_block.md` §10b-c

### 2026-11-06 · Stage 05 · Post-norm failed because our embeddings were small        [surprise]
**Expected:** If post-norm fails, it fails because of depth or learning rate.
**Found:** `run.py diagnose` logged rms(f(x)) / rms(x), the size of each
sublayer's edit next to the residual path, every 25 steps. Post-norm 4L:
0.02-0.11 at init, **2-24 by step 500**; gradient to block 1 fell to 1.5e-5.
The same model with token embeddings multiplied by √C (the 2017 paper's
recipe; our embedding rms is 0.028 otherwise): edits peaked at 2.4 near step
50, came back to **0.3-0.6**, block-1 gradient stayed 0.13-0.24, and loss at
step 500 was 4.65, a little ahead of pre-norm (4.74). Both post-norm runs
touched the frequency plateau at step ~50; only the √C one climbed out, by
step 100. Pre-norm's edits are *larger* (up to 16× the stream at block 1)
and do no harm.
**Why:** In post-norm every gradient passes through `Norm(x + f(x))`, which
divides it by the size of `x + f(x)`. Once the edit dwarfs `x`, the path
back to the earlier layers shrinks by that factor at every sublayer. A tiny
embedding makes `x` small from the start, so the edits win the race. In
pre-norm no norm sits on the residual path, so edit size does not matter.
**Lesson:** The √C multiplier in the 2017 paper, which Stage 03 met as a way
to balance token and position, also kept post-norm alive. Recipes are
bundles: take one part out (small embeddings) and another part (post-norm)
can stop working. Pre-norm is the design that does not care.
**Where:** `results/05_block/diagnose.json`, `docs/visuals/05_postnorm_diagnosis.png`,
`docs/05_transformer_block.md` §10b

### 2026-11-06 · Stage 05 · One change at a time: SwiGLU helps, dropout hurts, MLP matters   [result]
**Expected:** (docs §3, §6, §7) RMSNorm ≈ LayerNorm; SwiGLU ≥ GELU; ReLU a
little worse; dropout worse on single-pass data; no MLP clearly worse.
**Found:** 4 layers, 3 seeds each, held-out code loss. Baseline (pre-norm,
LayerNorm, GELU) **2.689 ± 0.012**, identical to Stage 04's `d64`
(2.689 ± 0.012), a cross-check that the hand-written LayerNorm trains the
same as `nn.LayerNorm`. RMSNorm 2.712 ± 0.027 (+0.023). ReLU 2.714 ± 0.002
(+0.025). **SwiGLU 2.651 ± 0.012 (−0.038)** with 7k fewer parameters.
No MLP 2.959 ± 0.028 (+0.270). Dropout 0.1: 2.775 ± 0.024 (+0.086), and its
train loss is worse too (3.03 vs 2.86).
**Why:** Dropout: nothing to overfit in 32.8M tokens seen once, so it only
adds noise. RMSNorm's +0.023 is within about one standard deviation of the
seed spread (0.027); we read it as "no measurable difference", not a loss.
**Lesson:** Of the modern changes, SwiGLU is the one that pays at our scale.
RMSNorm is adopted for cost, not quality.
**Where:** `results/05_block/sweep.json`, `docs/visuals/05_sweep.png`

### 2026-11-06 · Stage 05 · The fair depth test: post-norm trains at 4 layers, fails at 24   [result]
**Expected:** (curriculum) Pre-norm trains at 24 layers where post-norm fails.
**Found:** With `√C` embeddings for both (the 2017 recipe), one seed each:
post-norm **2.693** at 4 layers (pre-norm 2.685), **6.700** at 24 layers
(pre-norm **2.469**). Post-norm 24L's block-1 gradient was 7.9e-12 from step
500 on. Without warmup, 24L post-norm sat at 6.856, the frequency floor.
Lowering the learning rate for small-embedding post-norm at 4 layers let it
escape but slowly: 3e-4 → 3.822, 1e-4 → 4.698. `√C` also helped pre-norm at
24 layers (2.469 vs 2.567) but not at 4 (2.685 vs 2.685).
**Why:** Every post-norm sublayer is one more place where the gradient is
divided by the size of `x + f(x)`. Forty-eight of them are too many. In deep
pre-norm, a small embedding is swamped by 48 additions (positions start too
alike, §10a), which `√C` counters.
**Lesson:** The curriculum's prediction held only once the comparison was
fair. Without the diagnosis, we would have reported "post-norm fails at 4
layers too", which is true for our recipe and misleading about post-norm.
**Where:** `results/05_block/depth.json`, `docs/visuals/05_depth.png`

### 2026-11-06 · Stage 05 · The MLP is not just parameters; RMSNorm is not much faster   [result]
**Expected:** Removing the MLP hurts partly because it removes 2.1M
parameters. RMSNorm is cheaper than LayerNorm.
**Found:** A 12-layer attention-only model with the baseline's non-embedding
parameter count (3.17M vs 3.16M): 2.909 ± 0.022, still +0.220 worse than the
baseline (4-layer no-MLP: +0.270). Speed, fwd+bwd, 8,192 rows, bf16: written
out in PyTorch, RMSNorm is ~1.8× faster than LayerNorm (0.60 vs 1.18 ms at
C=256), which cost our LayerNorm runs 9% of wall time (250 s vs 228 s).
Fused (`nn.*`): 0.30 vs 0.42 ms at C=256, 3.91 vs 3.92 at C=4096. At C=4096
either norm is ~10% of one C×C matmul (36.8 ms).
**Lesson:** Parameter-matched controls separate "this part matters" from
"these parameters matter". Speed claims depend on whether the operation is
fused; measure the version you will actually run.
**Where:** `results/05_block/sweep.json`, `results/05_block/speed.json`

## Session 9 — 2026-11-06 · GPT assembled, and the rest of the curriculum built

### 2026-11-06 · Stage 06 · Our GPT loads OpenAI's weights and matches Hugging Face   [result]
**Expected:** `GPT.from_pretrained("gpt2")` reproduces `GPT2LMHeadModel`'s
logits within 1e-3.
**Found:** On the test sentence (CPU, fp32): yes, same argmax everywhere, same
loss to 1e-4. Hand-counted parameters: 124,439,808, equal to `sum(numel)`
for GPT-2 small and 9 other shapes, tied and untied. The loader refuses a
state dict with a missing or an extra tensor.
**Lesson:** Using the reference's own parameter names in our module made the
loader 30 lines with no mapping table; the only special cases are the three
traps (Conv1D transpose, mask buffers, tied head).
**Where:** `stages/06_gpt/gpt.py`, `test_gpt.py`, `docs/06_gpt_model.md` §5, §8a

### 2026-11-06 · Stage 06 · A 0.055 logit gap that was not a bug               [surprise]
**Expected:** On 200 windows of our val code (~650 GPT-2 tokens each, GPU),
ours and HF agree within the test's 1e-3.
**Found:** Worst window: **0.055**. Same window in float64: **2e-11**. Each
implementation is 0.015 from its own float64 answer in float32. GPT-2's
residual stream holds values near **3,000** from block 3 on (max per block:
4.5, 114, 604, 2479, ... 3023), so float32's ~7 digits leave errors near
3e-4 that compound over 12 blocks.
**Why:** Massive activations: a few dimensions a trained transformer uses as
large constant offsets.
**Lesson:** When two implementations disagree beyond tolerance on long real
inputs, run both in float64 before hunting a bug. Identical to 1e-11 there
means the arithmetic is the same and the gap is rounding.
**Where:** `docs/06_gpt_model.md` §8a, `results/06_gpt/oracle.json`

### 2026-11-06 · Stage 06 · Scaled residual init: big at step 0, small at the end  [result]
**Expected:** (docs §4b) a visible change in the starting stream; little
effect on final loss at 12 layers.
**Found:** Stream rms after the last block at init: plain 0.474 (12 layers)
and 0.933 (48 layers), against 0.028 for the embeddings. Scaled: 0.090 and
0.089, so the growth no longer depends on depth. Final val loss (12 layers,
4k steps, 2 seeds): plain 2.5927 / 2.5698, scaled 2.5632 / 2.5734. Means
2.581 vs 2.568: a 0.013 gain, inside the seed spread.
**Lesson:** An init fix can transform the starting state and barely move the
end result in a pre-norm model at moderate depth. Keep it for depth, and
report it as a tie at 12 layers.
**Where:** `results/06_gpt/init.json`, `docs/visuals/06_init.png`

### 2026-11-06 · Stage 06 · Without the shift: train loss 0, real loss 21.9      [result]
**Expected:** target = input makes training trivial and the model useless.
**Found:** Train loss 7.90 → 0.0001 by step 700. Next-token val loss **21.86**,
2.4× worse than uniform guessing (9.01). Greedy output: the prompt's last token
repeated 40 times. The normal model at the same 1,000 steps: val 4.09.
**Lesson:** A loss going to zero is not evidence of learning. It can mean the
task leaked into the input. A val loss above `ln V` means the model is
confidently wrong, which is worse than knowing nothing.
**Where:** `results/06_gpt/noshift.json`, `docs/06_gpt_model.md` §8c

### 2026-11-06 · Stage 06 · A number quoted from memory was wrong                [mistake]
**Expected:** The draft of docs/06 §4c said Stage 05's √C embedding run moved
pre-norm loss "2.685 → 2.680".
**Found:** `results/05_block/depth.json` says 2.6852 → 2.6850. The draft had
invented a 0.005 gap.
**Lesson:** Quote earlier stages' numbers from their results JSON, never from
memory. Even "a tiny effect" was overstated by 25×.
**Where:** `docs/06_gpt_model.md` §4c

### 2026-11-06 · Stage 06 · The untied-head penalty was slow learning              [result]
**Expected:** Stage 03 left open whether the untied head's +0.075 penalty is
lasting or slow learning.
**Found:** 4L, C=256. At 4,000 steps: tied 2.6814, untied 2.7576 (+0.076,
the Stage 03 gap again). At 12,000 steps: tied 2.3577, untied 2.3608
(+0.003, seed noise).
**Lesson:** An untied output matrix must learn from scratch what a tied one
inherits from the input embeddings. It catches up, and no further. For small
models on short budgets, tying wins outright; for long runs it is a wash in
quality and still saves V·C parameters.
**Where:** `results/06_gpt/tie.json`, `docs/06_gpt_model.md` §8d

### 2026-11-06 · Stage 06 · A 5M code model beats GPT-2 124M on code             [result]
**Expected:** GPT-2, 23× larger, does better on everything.
**Found:** Bits per byte on identical 256-token windows of our val split:
code: ours **1.066**, GPT-2 1.522. English: ours 1.774, GPT-2 **1.122**.
**Why:** In-domain training. GPT-2's WebText had little Python; ours is 85%
drawn from this code distribution.
**Lesson:** Compare across tokenizers in bits per byte, and remember that an
in-domain loss win says nothing about general knowledge.
**Where:** `results/06_gpt/bpb.json`, `docs/06_gpt_model.md` §8d

### 2026-11-06 · Stage 07 · The KV cache was exact, and barely faster for small models  [surprise]
**Expected:** (curriculum) 5-20× speedup at length 512, identical outputs.
**Found:** Identical tokens in all 36 settings (greedy and sampled). Speedup
at 512: **1.27×** (4L, 5M), 3.32× (8L, 30M), 4.47× (12L, 92M); at 1,024:
2.28× / 5.51× / 7.23×. The 4L model's cached decoding runs at ~1,420
tokens/s at every length: a fixed ~0.7 ms per step.
**Why:** At batch 1 and this size, a decode step costs kernel launches and
Python, not arithmetic. Recomputing 512 positions of a 5M model is almost free
on an otherwise idle GPU, so the cache removes work that was barely costing
anything.
**Lesson:** FLOP counts predict speedups only when FLOPs are the bottleneck.
Measure the per-step time across lengths: if it is flat, the step is
overhead-bound.
**Where:** `results/07_generation/speed.json`, `docs/07_generation.md` §8a

### 2026-11-06 · Stage 07 · The greedy loop is the most "valid" Python             [surprise]
**Expected:** Lower temperature, more syntactically valid code.
**Found:** Greedy output parses 25% of the time, τ=0.7 5%, τ≥1.0 never. But
greedy's distinct-2 is 0.07: it is `def is_prime(n): return n` repeated.
At τ=2.0 the model's mean log-prob of its own text is −9.51, worse than
uniform (−9.01).
**Lesson:** "Parses" rewards repetition. Any cheap proxy for code quality
needs a second number that catches degenerate output.
**Where:** `results/07_generation/temps.json`, `docs/07_generation.md` §8c

### 2026-11-06 · Stage 08 · Our AdamW drifts from PyTorch's in float32             [trap]
**Expected:** Hand-written AdamW equals `torch.optim.AdamW` to 1e-5 after 25 steps.
**Found:** 2.3e-4 in float32; 4.2e-13 in float64.
**Why:** Same arithmetic, different order; Adam divides by `√v`, so weights
with near-zero gradients take `±lr` steps decided by rounding noise, and
training compounds it.
**Lesson:** Compare optimizers in float64. A float32 tolerance loose enough to
pass hides real bugs too.
**Where:** `stages/08_training/test_training.py::test_adamw_matches_torch`

### 2026-11-06 · Stage 08 · L2-in-the-loss with Adam crushed the model           [result]
**Expected:** Adam + L2 (0.1) vs AdamW (0.1): a small difference.
**Found:** 1,500 steps, 4L: AdamW val 3.614, weight norm 98.7; Adam + L2 val
**6.695** (unigram level), weight norm **1.41**. SGD 5.615, SGD+momentum 4.310,
Adam without decay 3.609.
**Why:** The L2 gradient `0.1·θ` keeps the same sign step after step, and
Adam's normalisation turns any consistent gradient into a step of ~lr. Every
weight marches to zero.
**Lesson:** With adaptive optimizers, "weight decay" and "L2 penalty" are
different algorithms. The same coefficient means wildly different strength.
**Where:** `results/08_training/optimizers.json`, `docs/08_training_loop.md` §8a

### 2026-11-06 · Stage 08 · A loss spike from memorising one batch                [surprise]
**Expected:** The one-batch overfit test decreases monotonically to ~0.
**Found:** < 0.01 by step 129; then at step 259, 0.00036 → 0.579 in two steps
(1,600×), recovered by step ~290. Same batch every step.
**Why:** Near-zero, noisy gradients make Adam's `v` tiny, so `m/√v` steps
are ±lr in noise-chosen directions. Constant lr, no clipping.
**Lesson:** Spikes do not need bad data. Adam can produce them when gradients
vanish; schedules and clipping are what damp them.
**Where:** `results/08_training/overfit.json`, `docs/08_training_loop.md` §8b

### 2026-11-06 · Stage 08 · No warmup stalls at the unigram plateau, it does not spike   [surprise]
**Expected:** (curriculum) "record the loss spike" from removing warmup.
**Found:** 6L/384 at lr 3e-3, 2,000 steps: warmup+clip 2.946, no clip 3.028,
no warmup **4.781**, neither 5.057. Without warmup the loss fell to 7.0 by step
10 and sat there ~40 steps (unigram ≈ 6.86); grad norm > 1 on 67% of steps
afterwards (7% with warmup).
**Lesson:** At small scale, the cost of skipping warmup is getting stuck, not
blowing up. Test safety rails at a learning rate where they carry load:
at lr 1e-3 both looked unnecessary.
**Where:** `results/08_training/ablate.json`, `docs/08_training_loop.md` §8d

### 2026-11-06 · Stage 08 · File-order batches cost 1.25 nats                      [result]
**Expected:** Shuffling matters; random vs shuffled differ slightly.
**Found:** 4L, 4k steps: sequential 3.937, random 2.695, shuffled 2.683. Train
loss std over the last 1k steps: 0.74 sequential vs 0.16.
**Why:** Sequential batches are 32 consecutive windows, i.e. one or two files;
each update chases one repository. Random vs shuffled cannot differ at 2.5% of
an epoch: both are effectively fresh data.
**Lesson:** Stage 02's 37%-unseen argument for shuffled windows only bites
near one epoch. Order within an epoch matters far more than coverage early on.
**Where:** `results/08_training/sampling.json`, `docs/08_training_loop.md` §8e

### 2026-11-06 · Stage 08 · TinyStories smoke test passes in 11.5 minutes         [result]
**Expected:** A 10M model writes coherent short stories in ~20 minutes.
**Found:** 10,103,040 params (6L, C=320), 49M tokens, 691 s, val 1.510.
Step 200: broken grammar; step 1000: grammatical, no tracking of who is
who; step 6000: named character carried across sentences, quoted dialogue.
**Lesson:** The loop is correct end to end. Samples at fixed steps make a
cheap, readable progress check for every later run.
**Where:** `results/08_training/tinystories.json`, `docs/08_training_loop.md` §8c

### 2026-11-06 · Stage 09 · torch.compile doubled speed because matmuls stopped mattering   [result]
**Expected:** (curriculum) 3-5× total from stacking mixed precision, fused
attention, fused AdamW, compile and checkpointing; most of it from bf16.
**Found:** 8L/512, T=512, B=8: fp32 14.3k tok/s → TF32 18.5k → bf16 23.1k →
SDPA 34.5k → fused AdamW 35.1k → compile **73.7k** (5.16×, MFU 12% → 63% of
our measured 23.6 TFLOP/s bf16 peak). bf16 alone gave only 1.25× against a
4.3× peak ratio. Profiler before compile: matmul 29%, other elementwise 45%,
dtype casts 14%, fused attention 4%.
**Why:** Once matmuls are fast, the memory-bound work dominates: our
hand-written tanh-GELU (7-8 kernels) and autocast's per-step weight casts.
Fusion removes exactly that.
**Lesson:** The FLOP count tells you what the GPU *should* spend time on; the
profiler tells you what it *does*. They diverged by 2× here.
**Where:** `results/09_efficiency/{peak,stack,profile}.json`, `docs/09_efficiency.md` §9

### 2026-11-06 · Stage 09 · A profiler bucket hid the attention kernels           [mistake]
**Expected:** The kernel-kind breakdown shows fused attention's share.
**Found:** First version: no "fused attention" bucket at all. The matmul test
(`"cutlass" in name`) ran before the attention test, and the fused attention
kernels are cutlass kernels. Reordered: fused attention is 4% of step time.
**Lesson:** When categorising by name, test the most specific category
first, and print the top kernels with their assigned category so a misfile
is visible.
**Where:** `stages/09_efficiency/bench.py::cmd_profile`

### 2026-11-06 · Stage 09 · Activation checkpointing buys memory, not speed, here   [result]
**Expected:** Checkpointing (+33% compute) pays for itself by allowing a
bigger batch.
**Found:** −17% throughput, −41% memory (1,221 → 718 MB). With 4× the batch:
same 60.8k tok/s. Capstone 12L/768 at T=1024: B=8 25.4k tok/s, B=16 OOM,
B=16 + checkpointing 21.8k.
**Lesson:** Bigger batches only help until the GPU is full; 4,096 tokens per
micro-batch already fills a 4060 at these sizes. Use checkpointing to fit,
not to speed up.
**Where:** `results/09_efficiency/{stack,shapes}.json`

### 2026-11-06 · Stage 10 · The IsoFLOP runs diverged: attention logits reached 2 million   [surprise]
**Expected:** Small GPT-2-style models at lr 1e-3 (stable through Stages
04-08) train cleanly in the IsoFLOP sweep (T=512, B=16, up to 58k steps).
**Found:** The 4-layer, C=256 model failed every long run: grad norm rose
from ~1.5 to 30-3,500 after ~6k steps, val loss rose from 2.45 to 2.61
(3e15), and the 100% code / 85-15 mix runs ended at 3.52 / 2.61. A probe
on one fixed batch found the cause: the largest attention score q·k/√d in
block 1 went 35 (step 1k) → 218 (6k) → 14,206 (7k) → 2.0 million (12k).
The largest gradient sat in that block's LayerNorm. Compiled and eager runs
were identical to 4 decimals, so not a compiler bug. 6-10 layer models kept
grad norm near 1.
**Why:** Attention-logit growth (Dehghani et al. 2023; Wortsman et al. 2023,
"small-scale proxies for large-scale instabilities"): nothing bounds q·k,
the softmax saturates to one-hot, and its gradient becomes huge and erratic.
Earlier stages' runs were 3-4× shorter.
**Lesson:** "It trained fine at 4k steps" says little about 40k steps. Log
the attention-score maximum along with the loss; it moved 6,000 steps before
the loss did.
**Where:** `stages/10_scaling/diagnose.py`, `results/10_scaling/diagnose.json`

### 2026-11-06 · Stage 10 · QK-norm goes into our GPT, from measurement           [decision]
**Expected:** Fix the divergence by lowering the learning rate.
**Found:** Same 4L recipe, 13k steps, one change each: lr 1e-3 as before
2.656 (diverged, attention max 2.2M); lr 5e-4 **2.431** (max 117); QK-norm at
lr 1e-3 **2.280** (max 37); B32×T256 at lr 1e-3 2.476 (max 78, not
comparable windows). QK-norm both removes the instability and keeps the
higher learning rate, which is worth 0.15 nats here.
**Decision:** Stage 10 is rerun with `qk_norm=True`, and the Stage 11
workhorse and capstone use it. This is a departure from pure GPT-2,
borrowed from Stage 12's list on evidence; `from_pretrained("gpt2")` keeps it
off. The diverged v1 sweep is kept in `results/10_scaling/v1_no_qknorm/`.
**Where:** `llmfs/model/gpt.py` (`qk_norm`), `results/10_scaling/diagnose.json`

### 2026-11-07 · Stage 10 · Our IsoFLOP optimum lands on Chinchilla's line          [result]
**Expected:** A small sweep would be too noisy to say anything about the
20-tokens-per-parameter rule.
**Found:** With QK-norm, two budgets × five sizes, compute counted as
6N + 6CV + 12LTC. Fitted minima: 3e15 → 4.2M params at 28 tokens/param
(Chinchilla √(C/120) = 5.0M); 1e16 → 8.6M at 22.5 tokens/param (9.1M).
Best points: 4L/256 2.279 at 3e15, 6L/384 2.055 at 1e16.
**Lesson:** The rule transfers to 1-50M-parameter code models with an 8k
vocabulary, within ~15%. It needed the instability fixed first: the diverged
v1 points would have dragged the minima.
**Where:** `results/10_scaling/{isoflop_3e15,isoflop_1e16,fits}.json`, `docs/visuals/10_scaling.png`

### 2026-11-07 · Stage 10 · English is nearly free; repetition is not            [result]
**Found:** 4L/256 at 108M tokens. Mix: code-only 2.182 code / 5.308 English;
85/15 2.204 / 4.046; 70/30 2.260 / 3.844. Repetition (code only): fresh
2.176; 4 epochs of 27M 2.288; 16 epochs of 6.7M 2.803. Over-training 3L
(3.0M): 1× 2.534, 2× 2.365, 5× 2.228, 10× 2.178; the 10× run used 1.26e16
FLOPs, more than the 1e16 optimum that reached 2.055.
**Lesson:** 15% English costs 1% of code loss. Four epochs cost 5% here,
more than published small-epoch results, but our repeated slice is a
contiguous (less diverse) block, a confound we left in.
**Where:** `results/10_scaling/{mix,repeat,overtrain}.json`, `docs/10_scaling_laws.md` §8c-e

### 2026-11-07 · Stage 11 · The resume test caught a checkpoint bug on its first run   [mistake]
**Expected:** Stop at step 600, resume, continue the same curve.
**Found:** The resume crashed: `TypeError: RNG state must be a
torch.ByteTensor`. The trainer loads checkpoints with
`map_location="cuda"`, which also moved the saved CPU random-number state to
the GPU; `torch.set_rng_state` only accepts a CPU tensor.
**Lesson:** This is why the resume ablation runs before the long run: the
bug cost 1 minute here instead of a crashed resume at hour 12 of the
capstone. Anything saved for restoring state needs to be restored on the
device it came from.
**Where:** `llmfs/train/trainer.py` (resume), `stages/11_pretrain/run.py resume_test`

### 2026-11-07 · Stage 12 · GPT-2 → Llama, unbundled: RoPE is 94% of it            [result]
**Expected:** Each Llama change helps a little.
**Found:** 6L/384, 33M tokens, 2 seeds per row: GPT-2 2.518; +RMSNorm 2.551
(+0.033); +RoPE 2.321 (−0.230); +SwiGLU 2.295; +no bias 2.304; +GQA(2 kv)
2.329; +QK-norm 2.274. KV heads 8/4/2/1: 2.306/2.305/2.315/2.335; MQA with
the saved params in the MLP: 2.328.
**Lesson:** Of the −0.244 total, RoPE is −0.230. RMSNorm and no-biases cost a
little and are kept for speed. GQA is free down to 2 kv heads here.
**Where:** `results/12_modern_arch/{ladder,gqa}.json`, `docs/12_modern_arch.md` §9a-b

### 2026-11-07 · Stage 12 · Windowed attention is better nearby, worse far away    [surprise]
**Found:** Loss at positions 16-64: window-64 2.73, full 2.80. At 256-512:
window-64 2.48, full 1.97. Overall +0.24. NTK RoPE scaling, no fine-tune:
positions 1024-2048 of a model trained at 512 go 2.56 → 2.19 with s=4.
QK-norm model: no head puts more than 5.8% of attention on token 0.
**Lesson:** A window trades long-range loss for a small short-range gain;
the average hides both. Stacked layers did not recover what the window cut.
**Where:** `results/12_modern_arch/{window,rope_scale,sinks}.json`

### 2026-11-07 · Stage 13 · Without the aux loss half the experts died, and the loss did not notice   [surprise]
**Expected:** Aux loss off → router collapse → visible quality loss.
**Found:** 6L/384, 33M tokens. Collapse: yes. Busiest expert 0.41-0.43 of
top-2 assignments from step 100 (balanced 0.125; with aux 0.23), aux value
2.46 vs 1.09; 4 of 8 experts get zero first-choice tokens in the middle
layer. Quality: 2.326 vs 2.325. MoE vs dense at equal active params: 2.325
vs 2.320; fine-grained 16×top-4 2.303; same-total-params dense 2.354.
**Why:** At 33M tokens, each of 8 experts sees too little data to be worth
more than two well-trained ones. The collapse wastes capacity that this
budget could not use anyway.
**Lesson:** Collapse is a capacity problem first and a loss problem later.
Watch the utilisation histogram; the loss curve will not warn you.
**Where:** `results/13_moe/{train,analyse}.json`, `docs/visuals/13_moe.png`

### 2026-11-07 · Stage 13 · The router learned "English vs code" on its own       [result]
**Found:** Middle MoE layer, first-choice expert: expert 0 gets 96% English
tokens, expert 7 83%; experts 2 and 4 are 99% code (`(`, ` =`, `,`,
indentation); expert 3 takes identifiers and digits.
**Lesson:** Experts specialise by surface token type, not by topic.
**Where:** `results/13_moe/analyse.json`

### 2026-11-07 · Stage 14 · MLA lost to GQA at our scale                          [surprise]
**Expected:** (DeepSeek-V2) MLA matches or beats MHA with a much smaller cache.
**Found:** 6L/384, 33M tokens. MLA r=128 (1,728 B/token) 2.414; r=64 2.476;
r=32 2.765; r=16 2.801; r=4 3.198. GQA-2 (2,304 B) 2.315, MQA (1,152 B)
2.335, MHA 2.306.
**Lesson:** A low-rank cache needs a model with redundancy to compress and
enough data to learn the up-projections. At 12M parameters and 33M tokens,
neither holds. Reported as measured, untuned.
**Where:** `results/14_attention_variants/mla.json`

### 2026-11-07 · Stage 14 · A 0.11-nat loss gap hid 0% retrieval                 [result]
**Found:** Needle (5 digits, 30 cases per cell): full attention 51% mean;
window-128 0% in every cell with the needle beyond 128 tokens, 50-67% inside;
window-64 9% mean; MLA-64 58% with worse loss than GQA; MLA-16 4%.
**Lesson:** Average loss and long-range retrieval measure different things;
they can disagree in either direction.
**Where:** `results/14_attention_variants/needle.json`, `docs/visuals/14_attention_variants.png`

### 2026-11-07 · Stage 11 · Resume is bit-identical; dropping Adam's state is not   [result]
**Found:** (after the RNG fix) 4L, 1,200 steps, stop at 600. Resumed run:
every logged loss and grad norm equal to the uninterrupted run; final val
3.9158 in both. Resumed with fresh Adam m, v: +0.057 at step 610, +0.04 at
660, final 3.9323 (+0.017). Workhorse 30M: 590M tokens, 2h18m, val code
1.554, English 3.293, grad norm ≤ 1.49 after step 1,000.
**Lesson:** The test that catches checkpoint bugs is cheap and decisive:
compare curves, not just "it resumed without crashing".
**Where:** `results/11_pretrain/{resume_test,workhorse}.json`

### 2026-11-07 · Stage 15 · Exact speculative decoding, inexact bf16              [trap]
**Expected:** Greedy speculative output equals plain greedy output (the CPU
test proves it in fp32).
**Found:** On the GPU in bf16, 3 of 4 prompts identical over 160 tokens; the
4th diverged at token 130, where the target's top-2 logits differed by
0.031. In fp32: 4 of 4 identical.
**Why:** Verification scores γ+1 tokens in one forward pass; plain decoding
scores one at a time. Different shapes, different bf16 summation order,
different rounding; a near-tie flips.
**Lesson:** "Distribution-exact" holds for the mathematics. In low precision,
check exactness in fp32 and expect rare near-tie flips in bf16.
**Where:** `stages/15_speculative/run.py`, `docs/15_speculative.md` §6
