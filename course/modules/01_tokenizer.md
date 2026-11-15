# Module 1: Cutting text into tokens

**You will build** the byte-pair encoding (BPE) tokenizer that every GPT uses:
the training loop that learns which byte pairs to merge, the encoder that
applies those merges to new text, the decoder, and the regex pre-tokenizer
that keeps merges inside words.

**Time:** 3-5 hours. **Hardware:** CPU only; runs on Colab or any laptop.
**Book chapter:** [3, Cutting text into tokens](../../docs/01_tokenizer.md).

---

## 1. Why

A language model cannot read text. It reads lists of integers, and the
tokenizer decides what those integers mean. Too small a vocabulary (single
bytes) and every sentence becomes a long sequence the model must spend
compute on; too large and the embedding table eats the parameter budget. BPE
learns a vocabulary from data: start with the 256 bytes, then repeatedly glue
the most common neighbouring pair into a new token.

What breaks without it: with raw bytes, a 1,024-token context window holds
1,024 characters. With the tokenizer you are about to write, it holds about
2,300 characters of Shakespeare. With this project's production tokenizer,
about 3,400 characters of Python.

## 2. Read

Chapter 3 of the book, these sections before you start:

- §2 *Bytes as the foundation*: why the base vocabulary is 256 bytes and
  there is no "unknown" token.
- §3 *BPE: the algorithm*: the worked example by hand. Do it on paper too.
- §3 *Encoding new text*: why merges are applied in the order they were
  learned.
- §4 *Regex pre-tokenization*: what the regex split prevents.

Leave §9 (the production tokenizer) for after the module.

## 3. Implement

Open `stages/01_tokenizer/bpe.py`. Eight functions have their bodies replaced
by a hint and `raise NotImplementedError`. Write them in this order; each one
uses the ones before it.

| # | Function | What it does |
|---|---|---|
| 1 | `get_stats` | count every neighbouring pair of ids |
| 2 | `merge` | replace one pair with a new id, without overlaps |
| 3 | `BasicBPE._build_vocab` | turn the merge list into id → bytes |
| 4 | `BasicBPE.train` | the BPE loop on the raw byte stream |
| 5 | `BasicBPE._encode_chunk` | apply learned merges to new bytes |
| 6 | `BasicBPE.decode` | ids → text |
| 7 | `RegexBPE.train` | the same loop, inside regex chunks, counting each unique chunk once |
| 8 | `RegexBPE.encode_ordinary` | split, encode each chunk, join |

`BasicBPE()` calls `_build_vocab` when it is created, so almost every test
fails until function 3 works. That is expected: `get_stats` and `merge` have
their own tests that pass first.

## 4. Check

```bash
uv run python course/check.py 1        # Colab: !python course/check.py 1
```

It lists which of the eight are still blank, then runs the 58 tests this
project wrote for its own tokenizer. Among them: a round trip on 1,000 random
strings including emoji and invalid UTF-8; merges that never cross a word
boundary; and your fast `RegexBPE.train` producing exactly the same merges as
the slow version. The module is done when all of them pass.

## 5. Run

```bash
uv run python stages/01_tokenizer/run.py --vocab 1024
```

About 70 seconds. It downloads tinyshakespeare (1.1 MB), trains on the first
90% and measures compression on the last 10%. Compare section 1 of the
output with the book's run:

| vocab | held-out bytes/token, book |
|---|---|
| 256 | 1.000 |
| 512 | 1.878 |
| 1024 | 2.257 |

Training is deterministic, so a correct implementation matches these to
within ±0.01. A different tie-break when two pairs have the same count can
move the third decimal; anything larger is a bug. (Section 3 of the output
measures a different slice of text, so its 2.275 is not the number to
compare.)

Section 3 also times your encoder against OpenAI's `tiktoken`. The book's
pure-Python encoder was 27-28× slower; yours will be similar. The chapter
explains why that is an algorithm problem as much as a language one.

## 6. Break it

Section 2 of the same output is the ablation: the same corpus, vocabulary
2,048, trained with and without the regex split.

| | tokens that span a word boundary | held-out bytes/token |
|---|---|---|
| `BasicBPE` (no regex) | 778 of 1,792 | 2.788 |
| `RegexBPE` | 0 of 1,792 | 2.557 |

Without the split, almost half the learned vocabulary is fragments like
`'e '`, `'and '` and `' the '`. Note that `BasicBPE` compresses held-out
Shakespeare *better*. Read the chapter's §4 for why that is still the wrong
tokenizer, then try one more break of your own: in `_encode_chunk`, apply
merges longest-first instead of in learned order, and watch which tests fail.

## 7. Check yourself

<details><summary>1. Why does a byte-level BPE tokenizer never need an "unknown" token?</summary>

Every one of the 256 possible byte values has an id before training starts,
so any text, in any language, can be encoded as bytes in the worst case.
</details>

<details><summary>2. Why apply merges in the order they were learned, not longest match first?</summary>

Learned order reproduces the exact sequence of states the training text went
through, so new text is cut the same way the model saw during training.
Longest-match produces different cuts for the same string.
</details>

<details><summary>3. In section 3 of your output, your vocabulary-1024 tokenizer gets about 2.28 bytes per token and GPT-2's gets 3.11 on the same text. Name two reasons for the gap.</summary>

GPT-2 has 50,257 tokens to your 1,024, so it can hold whole common words as
single tokens; and it was trained on far more text. The cost is a 50k-row
embedding table, which at 10-50M parameters would be bigger than the rest of
the model.
</details>

<details><summary>4. Why does <code>encode</code> treat the string <code>&lt;|endoftext|&gt;</code> as ordinary text unless told otherwise?</summary>

If user text could produce control tokens, anyone could forge the end of a
document or a chat turn by typing it. Control tokens should come only from
code you wrote (chapter §5).
</details>

<details><summary>5. "strawberry" becomes <code>['st', 'ra', 'w', 'ber', 'ry']</code>. What does this tell you about asking a model to count the r's?</summary>

The model never sees individual letters for this word, only five tokens, and
no token is a single "r". Counting letters needs knowledge the tokenizer hid.
</details>

---

**Stuck?** The hint is in each function's docstring. The complete solution is
[`stages/01_tokenizer/bpe.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/01_tokenizer/bpe.py)
in the main repository.

**Next:** Module 2, the data pipeline: where the Python comes from, how much
of a test set leaks into training without anyone noticing, and how text
becomes the token files a model trains on.
