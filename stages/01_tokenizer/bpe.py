"""Stage 01 — Byte Pair Encoding, from scratch.

Reference implementation for docs/01_tokenizer.md. Nothing is imported that
does tokenization; only `regex` (for Unicode property classes like \\p{L},
which Python's stdlib `re` does not support).

Two tokenizers here, on purpose:

    BasicBPE  — BPE straight on the byte stream, no pre-tokenization.
    RegexBPE  — splits text with the GPT-2 regex first, so merges can never
                cross a word/punctuation boundary. This is what real
                tokenizers do.

Training both on the same corpus and diffing the merge lists is the ablation
for this stage: see run.py.

Shape/type conventions used throughout:
    text   : str
    ids    : list[int]              token ids
    merges : dict[(int,int) -> int] pair -> new id, insertion-ordered = rank
    vocab  : dict[int -> bytes]     id -> the bytes it expands to
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter

import regex as re

# ---------------------------------------------------------------------------
# Pre-tokenization patterns
# ---------------------------------------------------------------------------

# GPT-2's split pattern. Walked through in docs/01_tokenizer.md §4.
#   's|'t|...        English contractions as their own chunks
#   ?\p{L}+          optional leading space + letters  <- why tokens carry spaces
#   ?\p{N}+          optional leading space + digits   <- all digits grouped (bad for math)
#   ?[^\s\p{L}\p{N}]+ optional space + punctuation run
#   \s+(?!\S)        trailing whitespace
#   \s+              any other whitespace run
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# GPT-4's pattern. Two differences worth noticing:
#   - `\p{N}{1,3}` caps digit runs at 3, so numbers tokenize consistently
#   - case-insensitive contractions, and no space before \s+ handling
# Llama 3 goes further and splits every digit on its own.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


# ---------------------------------------------------------------------------
# Core BPE primitives
# ---------------------------------------------------------------------------

def get_stats(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count every adjacent pair in `ids`.

    get_stats([1, 2, 3, 2, 3]) -> {(1,2): 1, (2,3): 2, (3,2): 1}

    `counts` is passed in so we can accumulate across many chunks without
    building an intermediate list per chunk.
    """
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):          # (a,b), (b,c), (c,d), ...
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]

    Note the i += 2 on a hit: occurrences do not overlap, so [1,1,1] with
    pair (1,1) becomes [99,1], not [99,99]. BPE training counts overlapping
    pairs but merges non-overlapping ones; this asymmetry is standard and
    harmless because the next iteration re-counts from scratch.
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# byte <-> printable-unicode mapping (GPT-2's cosmetic trick)
# ---------------------------------------------------------------------------

def bytes_to_unicode() -> dict[int, str]:
    """Map all 256 byte values to printable unicode characters.

    Purely cosmetic: it exists so a vocab file is human-readable and JSON-safe.
    Bytes 0-32 and 127-160 are control chars/whitespace that break when
    printed, so they are shifted into the unused U+0100.. range. This is why
    GPT-2 vocab files show `Ġthe` -- `Ġ` (U+0120) is byte 32, a space.

    NOT part of the BPE algorithm. See docs/01_tokenizer.md §2.
    """
    # the byte values that are already safe to print as-is
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)      # park it in the private/unused range
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def render_token(t: bytes) -> str:
    """Make a token's bytes printable for debugging: escape control chars."""
    s = t.decode("utf-8", errors="replace")
    return "".join(
        ch if unicodedata.category(ch)[0] != "C" else f"\\u{ord(ch):04x}"
        for ch in s
    )


# ---------------------------------------------------------------------------
# BasicBPE — no pre-tokenization
# ---------------------------------------------------------------------------

class BasicBPE:
    """BPE applied directly to the whole byte stream.

    This is the version that produces merges spanning word boundaries
    (" the", "the.", " of the"). Kept as the ablation baseline -- it is what
    you get when you skip regex pre-tokenization.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}   # pair -> new id (rank order)
        self.vocab: dict[int, bytes] = self._build_vocab()
        self.special_tokens: dict[str, int] = {}

    # -- vocab ------------------------------------------------------------

    def _build_vocab(self) -> dict[int, bytes]:
        """Rebuild id -> bytes from the merge list.

        Base vocab is all 256 byte values, which is why there is no UNK:
        every possible byte already has an id.
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]     # merges are in rank order,
        return vocab                               # so parents always exist first

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # -- training ---------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`.

        The loop is the entire BPE algorithm:
            count pairs -> take the most frequent -> merge it -> record the rank
        """
        assert vocab_size >= 256, "vocab must include all 256 byte values"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))      # the only place text becomes bytes
        merges: dict[tuple[int, int], int] = {}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                print(f"  stopped early at {i} merges: nothing left to merge")
                break
            # max() breaks ties by first-encountered, which makes training
            # deterministic for a fixed corpus. Important for reproducibility.
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id
            if verbose and (i < 5 or i % 500 == 0):
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"(count {stats[pair]})")

        self.merges = merges
        self.vocab = self._build_vocab()

    # -- special tokens ---------------------------------------------------

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Attach control tokens, e.g. {"<|endoftext|>": 8192}.

        Ids should sit ABOVE the learned vocab, and it is worth leaving a gap:
        adding a token later means resizing the model's embedding table, which
        invalidates every checkpoint. See docs/01_tokenizer.md §5.
        """
        self.special_tokens = dict(special_tokens)
        self.inverse_special = {v: k for k, v in special_tokens.items()}

    # -- encode / decode --------------------------------------------------

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """Apply merges to one byte string, lowest rank (earliest learned) first.

        Rank order is not a stylistic choice: it reproduces the exact state the
        training corpus passed through, so encoding stays consistent with
        training. Greedy longest-match would give a different, worse result.

        Cost is O(n * m) here -- we rescan for the best pair every iteration.
        tiktoken does this with a linked list + heap in Rust. run.py measures
        how much that costs us.
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            # lowest merge rank among pairs that are actually mergeable
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                       # nothing left that we know how to merge
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode, treating any special-token string as ordinary text."""
        return self._encode_chunk(text.encode("utf-8"))

    def encode(self, text: str, allowed_special: str = "none") -> list[int]:
        """Encode text to ids.

        allowed_special:
            "none"       special-token strings are encoded as ordinary text.
                         SAFE DEFAULT -- user input must never be able to forge
                         a control token (see docs/01_tokenizer.md §5).
            "all"        special-token strings map to their control ids.
                         Use only for text you construct yourself, e.g. chat
                         templates in Stage 16.
        """
        if allowed_special == "none" or not self.special_tokens:
            return self.encode_ordinary(text)
        if allowed_special != "all":
            raise ValueError("allowed_special must be 'none' or 'all'")

        # Split on the special tokens, keeping them as delimiters.
        pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        """ids -> text.

        errors="replace" matters: a single token's bytes can be an incomplete
        UTF-8 sequence (half an emoji), so decoding token-by-token during
        streaming generation must either buffer or tolerate this. Stage 07.
        """
        parts: list[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in getattr(self, "inverse_special", {}):
                parts.append(self.inverse_special[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        """Write merges + specials as JSON.

        Merges are stored as an ordered list, because rank order IS the model.
        A dict with string keys would be fragile; a list is unambiguous.
        """
        b2u = bytes_to_unicode()
        payload = {
            "type": type(self).__name__,
            "pattern": getattr(self, "pattern", None),
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
            # readable view only -- never parsed back. Debug aid.
            "vocab_preview": {
                str(i): "".join(b2u[b] for b in self.vocab[i])
                for i in sorted(self.vocab)[:512]
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BasicBPE":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        pattern = payload.get("pattern")
        tok = RegexBPE(pattern) if pattern else BasicBPE()
        tok.merges = {(p0, p1): idx for p0, p1, idx in payload["merges"]}
        tok.vocab = tok._build_vocab()
        if payload.get("special_tokens"):
            tok.register_special_tokens(payload["special_tokens"])
        return tok


# ---------------------------------------------------------------------------
# RegexBPE — with pre-tokenization. The real thing.
# ---------------------------------------------------------------------------

class RegexBPE(BasicBPE):
    """BPE with regex pre-tokenization, GPT-2 style.

    The only difference from BasicBPE: text is split into chunks first, pair
    counts are accumulated per chunk, and merges are applied per chunk. A merge
    therefore can never span a chunk boundary -- so no `the.` or ` of the`
    tokens, and a word tokenizes the same way regardless of the punctuation
    next to it.
    """

    def __init__(self, pattern: str | None = None) -> None:
        super().__init__()
        self.pattern = pattern or GPT2_SPLIT_PATTERN
        self.compiled = re.compile(self.pattern)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train on UNIQUE chunks weighted by frequency.

        The naive version (see BasicBPE.train) walks every position in the
        corpus on every merge: O(corpus_length x num_merges). But pre-tokenized
        chunks repeat heavily -- 1M characters of Shakespeare contains only
        ~30k distinct chunks -- and the pair counts come out identical either
        way:

            count(pair) over corpus == sum over unique chunks of
                                       (chunk frequency x pairs in that chunk)

        So deduplicate first and carry a weight. Same merges, exactly, for far
        less work on real text. test_bpe.py asserts that equivalence against a
        naive reference implementation.

        HuggingFace `tokenizers` does the same thing. This dedup is why
        training a 50k-vocab tokenizer on gigabytes of text is feasible.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # chunk -> frequency. Insertion order follows first appearance in the
        # corpus, which keeps the max() tie-break deterministic.
        chunk_counts = Counter(self.compiled.findall(text))
        # parallel lists: ids per unique chunk, and that chunk's weight
        chunks: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
        weights: list[int] = list(chunk_counts.values())

        merges: dict[tuple[int, int], int] = {}

        for i in range(num_merges):
            stats: Counter = Counter()
            for chunk, w in zip(chunks, weights):
                for pair in zip(chunk, chunk[1:]):
                    stats[pair] += w          # weighted by chunk frequency
            if not stats:
                print(f"  stopped early at {i} merges: nothing left to merge")
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            # Only chunks that actually contain the pair need rebuilding.
            a, b = pair
            for j, chunk in enumerate(chunks):
                if any(chunk[k] == a and chunk[k + 1] == b
                       for k in range(len(chunk) - 1)):
                    chunks[j] = merge(chunk, pair, new_id)
            merges[pair] = new_id
            if verbose and (i < 5 or i % 500 == 0):
                print(f"  merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                      f"(count {stats[pair]})")

        self.merges = merges
        self.vocab = self._build_vocab()

    def encode_ordinary(self, text: str) -> list[int]:
        """Split, encode each chunk independently, concatenate."""
        ids: list[int] = []
        for chunk in self.compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids


# ---------------------------------------------------------------------------
# Metric: compression ratio
# ---------------------------------------------------------------------------

def compression_ratio(tokenizer, text: str) -> float:
    """Bytes of UTF-8 per token. Higher is better.

    This is the tokenizer quality metric that matters: more text per token
    means more content fits in the context window and fewer FLOPs per
    character. Reference points in docs/01_tokenizer.md §6.
    """
    n_bytes = len(text.encode("utf-8"))
    n_tokens = len(tokenizer.encode(text))
    return n_bytes / n_tokens
