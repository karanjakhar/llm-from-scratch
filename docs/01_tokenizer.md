# Stage 01 — Tokenization

The model never sees text. It sees integers. The tokenizer is the layer that
decides *which* integers, and that choice silently shapes everything
downstream: vocabulary size, embedding table size, effective context length,
arithmetic ability, and multilingual fairness.

It is also the one component that is **not** learned by gradient descent. It is
trained by a separate counting algorithm, frozen, and then everything else is
built on top of it. Change the tokenizer and every checkpoint you own becomes
garbage.

---

## 1. The design space

We need a map from text → list of integers. Three obvious options, all of which
fail, which is why BPE exists.

### Option A: characters

```
"tokenization" → ['t','o','k','e','n','i','z','a','t','i','o','n']   # 12 tokens
```

- Vocab: tiny (~128 for ASCII, ~150k for all of Unicode).
- Problem: **sequences get very long**. Attention is O(T²), so 4x longer
  sequences cost 16x more attention compute. Your context window, measured in
  actual text, collapses.
- Problem: the model must spend its early layers rediscovering that "t-h-e" is
  a unit. Wasted capacity.

### Option B: words

```
"tokenization is hard" → ['tokenization', 'is', 'hard']   # 3 tokens
```

- Vocab: huge (English alone has >1M word forms) and **open-ended**.
- Problem: any word not in the vocab becomes `<UNK>` and its meaning is gone
  forever. Typos, names, new words, code identifiers — all destroyed.
- Problem: the embedding table becomes most of the model. 1M words × 768 dims
  = 768M parameters of embedding alone.
- Problem: no morphology. "run", "running", "runner" are unrelated ids.

### Option C: sub-words — the actual answer

Common things get one token; rare things get decomposed into pieces.

```
"tokenization" → ['token', 'ization']       # 2 tokens
"Jakhar"       → ['Jak', 'har']             # rare name, still representable
```

- Vocab: controllable (8k-200k), chosen as a hyperparameter.
- No `<UNK>` ever, if the base alphabet covers all bytes.
- Frequent words stay single tokens, so sequences stay short.
- Morphology emerges for free from the merge statistics.

**BPE is the algorithm that finds a good sub-word vocabulary automatically,
from data.**

---

## 2. Bytes as the foundation

Before BPE: what is the base alphabet?

If the base alphabet is "characters", Unicode has ~150,000 code points and you
cannot include them all. Any you exclude becomes `<UNK>`.

Instead, work on **UTF-8 bytes**. Every possible string encodes to a sequence
of bytes, and there are exactly 256 byte values. So:

- Base vocab = 256. Fixed, small, complete.
- **Zero possibility of `<UNK>`.** Any text, any language, any emoji, any
  binary garbage, round-trips exactly.

```python
"héllo 🌍".encode("utf-8")
# b'h\xc3\xa9llo \xf0\x9f\x8c\x8d'
# → [104, 195, 169, 108, 108, 111, 32, 240, 159, 140, 141]   11 bytes, 7 chars
```

Note that non-ASCII characters cost multiple bytes. "é" is 2 bytes, "🌍" is 4.
This is the root of why non-English text costs more tokens, and why
non-English users get less effective context for the same price. A real
fairness issue, visible right here at the byte layer.

### GPT-2's byte↔unicode trick

A detail that confuses everyone reading GPT-2's tokenizer code. GPT-2 wanted the
vocabulary to be printable/debuggable, but bytes 0-32 and 127-160 are control
characters and whitespace that break when printed or stored in a JSON vocab
file. So GPT-2 maps those "unprintable" byte values to unused printable Unicode
code points (shifting by 256), producing the famous `Ġ` for a space (byte 32 →
U+0120).

It is a **cosmetic encoding for the vocab file**, not part of the algorithm.
`Ġthe` just means `" the"`. We will implement it so GPT-2 vocab files are
readable, and clearly mark it as cosmetic.

---

## 3. BPE: the algorithm

Byte Pair Encoding was a 1994 *compression* algorithm, repurposed for
tokenization in 2015 (Sennrich et al.) and used by GPT-2 onward.

**Training.** Repeat until the vocabulary reaches the target size:

1. Count every adjacent pair of symbols in the corpus.
2. Find the most frequent pair.
3. Mint a new symbol for it. Replace every occurrence.
4. Record the merge in an ordered list.

### Worked example, by hand

Corpus (tiny, for clarity): `"low low low lower lowest"`

Start as bytes — here shown as characters since it is all ASCII. `_` is a space.

```
l o w _ l o w _ l o w _ l o w e r _ l o w e s t
```

Pair counts:
```
(l,o): 5   (o,w): 5   (w,_): 3   (_,l): 4   (w,e): 2   (e,r): 1  ...
```

`(l,o)` and `(o,w)` tie at 5. Break the tie deterministically (first seen).

**Merge 1: `(l,o) → lo`**, new id 256.
```
lo w _ lo w _ lo w _ lo w e r _ lo w e s t
```

**Merge 2: `(lo,w) → low`**, count 5, new id 257.
```
low _ low _ low _ low e r _ low e s t
```

**Merge 3: `(low,_) → low_`**? count 3. Or `(_,low)` count 4 — pick that,
new id 258.
```
low _low _low _lowe r _lowe s t      ... and so on
```

After 3 merges, `"low"` is a single token instead of 3. The vocabulary learned
that "low" is a unit **purely from counting**, with no linguistic knowledge.

### Requested vocab size is an upper bound, not a promise

BPE cannot invent merges the corpus does not contain. Once every distinct
pre-token chunk has collapsed into a single id, there are no adjacent pairs
left and training stops early. A corpus with 30 distinct words supports only
about 130 merges no matter how many times those words repeat.

Practical consequence: **size the model's embedding table from the tokenizer
that actually came out, not from the vocab size you asked for.** Ask for 8192
on a small corpus and you may get 3400. Hard-coding 8192 into the model config
then produces an embedding table with thousands of rows that receive no
gradient — dead parameters, and a silent bug. `test_bpe.py` asserts this
behaviour explicitly.

### What the trained tokenizer consists of

Two artifacts, and that is all:

1. `merges` — an **ordered** list of pairs. Order is the whole point: it is the
   priority ranking used at encode time.
2. `vocab` — id → bytes. Derivable from merges, but cached for decoding.

### Encoding new text

Given the merge list, encode text by applying merges **in training order**:

```python
tokens = list(text.encode("utf-8"))
while len(tokens) >= 2:
    # find the adjacent pair with the lowest merge rank (= learned earliest)
    pair = min(all_adjacent_pairs(tokens), key=lambda p: ranks.get(p, inf))
    if pair not in ranks:
        break                      # no mergeable pair left
    tokens = merge(tokens, pair, ranks[pair])
```

Lowest rank first is **not** an arbitrary choice. It reproduces exactly the
state the training corpus was in, so encoding is consistent with training.
Applying merges greedily in a different order gives a different (worse,
inconsistent) tokenization.

Naive cost is O(n² ) per chunk in the worst case. Real implementations
(`tiktoken` in Rust, `tokenizers` in Rust) use a linked list with a priority
queue. We write the slow clear version first, measure it, then optimize — the
speed difference is itself a lesson about why these libraries are not in Python.

### Decoding

Trivial: look up each id's bytes, concatenate, `.decode("utf-8")`.

One trap: a single token's bytes may be an **incomplete UTF-8 sequence** (half
of an emoji). Decoding one token at a time therefore crashes or produces
replacement characters. This is why streaming generation must buffer bytes
until they form valid UTF-8, and why `errors="replace"` appears in every
decoder. Stage 07 hits this for real.

---

## 4. Regex pre-tokenization — the non-obvious part

Run naive BPE on raw text and you get merges like `" the"`, `"the."`,
`" the "`, `"dog."` — tokens that **span word and punctuation boundaries**.
This wastes vocabulary on near-duplicates and makes the tokenization of a word
depend on its punctuation context.

GPT-2's fix: split text into chunks with a regex **first**, run BPE
independently inside each chunk, and never allow a merge to cross a chunk
boundary.

GPT-2's pattern:
```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Read it piece by piece:
- `'s|'t|'re|'ve|'m|'ll|'d` — English contractions get their own tokens.
- ` ?\p{L}+` — an optional leading space plus letters. **This is why GPT tokens
  start with a space**: `" the"` is one token, and leading-space-attached is the
  canonical form.
- ` ?\p{N}+` — optional space plus digits.
- ` ?[^\s\p{L}\p{N}]+` — optional space plus punctuation, grouped.
- `\s+(?!\S)` — trailing whitespace, and `\s+` — any other whitespace run.

Note `\p{N}+` groups **all consecutive digits** into one chunk, so GPT-2 can
produce a single token for `"2024"` and a different one for `"202"`. This is
precisely why early LLMs were bad at arithmetic: the digit "5" has a different
id depending on which numbers it appeared inside during training. GPT-4's
pattern, which Llama 3 also uses, caps digit runs at 3 (`\p{N}{1,3}`). Llama 1
and 2 (SentencePiece with `split_digits`) and Qwen2 go further and split
**every digit separately**, which makes digit tokens consistent and is reported
to help arithmetic. A regex choice changes the model's math ability. That is
how deep tokenizer effects run.

**The ablation for this stage** is to train BPE with and without the regex and
inspect the merge lists side by side.

---

## 5. Special tokens

Tokens that are not learned from text but injected for control:

| Token | Purpose | Stage |
|---|---|---|
| `<|endoftext|>` | document boundary during pretraining | 02 |
| `<|pad|>` | padding in batches (always masked out of the loss) | 16 |
| `<|im_start|>`, `<|im_end|>` | chat role delimiters | 16 |

Two rules, both learned the hard way by the field:

1. **Reserve ids for tokens you do not need yet.** Adding a token later means
   resizing the embedding table and invalidating checkpoints. Leave a gap.
2. **Special tokens must never be producible from user text.** If a user can
   type the literal string `<|im_end|>` and have it tokenize to the control
   token, they can forge a conversation turn. The encoder must have a mode that
   treats specials as plain text. This is a real prompt-injection surface.

---

## 6. Quality metric: compression ratio

How do you know a tokenizer is good? Measure **bytes per token** on held-out
text of the kind you care about:

```
compression = total_bytes / total_tokens
```

Higher is better: more text per token means more content in the same context
window and fewer FLOPs per character. Reference points:

| Tokenizer | Vocab | ~bytes/token on English |
|---|---|---|
| raw bytes | 256 | 1.0 |
| GPT-2 | 50,257 | ~4.0 |
| GPT-4 (`cl100k_base`) | 100,277 | ~4.4 |
| Llama 3 | 128,256 | ~4.3 |

Returns diminish: doubling the vocab from 50k to 100k buys ~10% compression
while adding 50k × d_model embedding parameters. The tradeoff:

- **Bigger vocab** → shorter sequences (cheaper attention), bigger embedding
  and output layers, rarer tokens get less training signal each.
- **Smaller vocab** → longer sequences (costlier attention), compact model,
  every token well-trained.

**Our choice: 8192.** At 10-50M parameters, a 50k vocab would make the
embedding table larger than the whole transformer. 8192 × 384 dims ≈ 3.1M
params, which is reasonable. Recorded in `CLAUDE.md` → Decisions.

---

## 7. Why tokenizers cause famous LLM failures

Not trivia — each one is diagnostic of what the model can and cannot see.

**"How many r's in strawberry?"** The model sees `['str','aw','berry']`. It
never sees letters. Asking it to count characters is asking it to count
something that was destroyed before layer 1. The surprise is that it often
gets it right at all, having memorized spellings from text *about* spelling.

**Reversing a string** — same cause. Character-level operations on a
sub-word-level representation.

**Arithmetic** — inconsistent digit grouping, as in §4.

**Non-English cost** — more bytes per character, so more tokens per sentence,
so less context and higher price for the same content.

**Trailing whitespace weirdness** — `"Hello"` vs `"Hello "` tokenize
differently, and since tokens carry leading spaces, a trailing space puts the
model in a state it rarely saw in training. This is why API docs warn against
trailing spaces in prompts.

**SolidGoldMagikarp** — glitch tokens. Strings frequent enough in the
tokenizer's training corpus to earn a token, but near-absent from the LM's
training corpus. Their embeddings stayed near initialization, so the model
behaves erratically when it meets them. A direct consequence of training the
tokenizer and the model on different data.

---

## 8. Build order for this stage

1. `bpe.py` — `train()`, `encode()`, `decode()` in pure Python, no regex. Get
   round-tripping exact first.
2. Add regex pre-tokenization. Diff the merge lists against step 1. **This is
   the ablation.**
3. Add special-token handling, with the "treat as plain text" safe mode.
4. Save/load the tokenizer to JSON.
5. Train on a real corpus at vocab 8192. Measure bytes/token on held-out text.
6. Compare compression and speed against `tiktoken` gpt2. Explain both gaps.
7. Promote the clean version to `llmfs/tokenizer/`.

### Acceptance criteria

- [ ] `decode(encode(s)) == s` for 1000 random held-out strings, including
      emoji, CJK, and control characters.
- [ ] Never emits `<UNK>`; vocab covers all 256 bytes.
- [ ] Encoding is deterministic and rank-ordered (verify a known case by hand).
- [ ] Compression ratio measured and recorded in `PROGRESS.md`.
- [ ] Merge list with regex vs without compared, with the difference explained.
- [ ] Specials cannot be forged from user text in safe mode (test asserts it).

## 9. The production tokenizer — measured on the real corpus

Trained by `stages/01_tokenizer/train_production.py` on 36 MB of permissive
Python + fineweb-edu English (train split only), then measured on held-out code
and English from **validation repositories the tokenizer never saw**. Every
variant used GPT-2's pattern with only the digit rule changed. Each was trained
once to 16,384, with smaller vocabs read off as merge prefixes (~1 h per
variant, 3 in parallel).

**Held-out bytes per token** (higher = more text per token):

| digit rule | vocab | code | English |
|---|---|---|---|
| grouped `\p{N}+` (GPT-2) | 2,048 | 2.762 | 2.567 |
| | 4,096 | 3.099 | 2.942 |
| | **8,192** | **3.381** | 3.369 |
| | 16,384 | 3.586 | 3.796 |
| cap3 `\p{N}{1,3}` (GPT-4, Llama 3) | 8,192 | 3.376 | 3.365 |
| | 16,384 | 3.580 | 3.789 |
| split `\p{N}` (Llama 1/2, Qwen2) | 8,192 | 3.337 | 3.327 |
| | 16,384 | 3.529 | 3.734 |
| GPT-2's tokenizer | 50,257 | **2.222** | 4.570 |
| GPT-4's `cl100k_base` | 100,277 | 4.244 | 4.688 |

### What the table says

**Our 8k tokenizer beats GPT-2's 50k tokenizer on code by 52%.** GPT-2 was
trained on web text and never learned merges for runs of spaces, so every
space of Python indentation costs it a whole token. Vocabulary size is not
the main thing here. The training distribution is. This is the reason Codex
added dedicated whitespace tokens. On English, GPT-2 still wins easily, as it
should with 6× the vocab trained on English.

**Vocab returns diminish fast.** Each doubling buys less on code: +12.2%
(2k→4k), +9.1% (4k→8k), +6.1% (8k→16k).

**The digit rule is nearly free.** Splitting every digit costs 1.3% on code
versus grouping; capping at 3 costs 0.15%.

### Decision: `split_8192`

- **Vocab 8,192.** Going to 16,384 buys 6.1% on code at the cost of 4.2M more
  embedding parameters (d=512), about 17% of a 50M workhorse once the table is
  tied to the output head. Both models must share one tokenizer, and the
  workhorse is where every stage gets debugged.
- **Split digits.** It costs 1.3% compression, and in exchange every digit
  always has the same id. `range(100)` becomes `range`, `(`, `1`, `0`, `0`, so
  the model sees the 1 and the 0s as the same tokens wherever they appear. For a
  small model that has to learn what numeric literals *mean*, consistency is
  worth more than 1.3%.
- **Cheap to reverse.** Switching tokenizers means re-running
  `stages/02_data/prepare.py` (minutes), not retraining anything. Any model
  trained before a switch is invalidated, though. Revisit only with evidence,
  e.g. a Stage 10 ablation.

Final layout: ids 0-255 bytes, 256-8191 merges, 8192 `<|endoftext|>`,
8193 `<|pad|>`, 8194 `<|im_start|>`, 8195 `<|im_end|>` → **8,196 embedding rows**.

### Speed

The fast cached encoder in `llmfs/tokenizer` runs at **8.1 MB/s on one core
vs 0.66 MB/s** for the stage-01 encoder (12×), with a 98.2% chunk-cache hit
rate starting cold, and produces identical ids (tested on real GitHub files).

## Further reading

- Sennrich et al. 2015, "Neural Machine Translation of Rare Words with Subword
  Units" — the original BPE-for-NLP paper.
- Karpathy, `minbpe` — the reference minimal implementation. Read *after*
  writing your own.
- `tiktoken` source — see what the production version does differently.
