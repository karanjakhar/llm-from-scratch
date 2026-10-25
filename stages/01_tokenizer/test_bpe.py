"""Stage 01 tests — the acceptance criteria from docs/01_tokenizer.md §8.

    uv run pytest stages/01_tokenizer/ -v
"""

import random
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from bpe import (  # noqa: E402
    BasicBPE,
    RegexBPE,
    compression_ratio,
    get_stats,
    merge,
)

def _make_corpus() -> str:
    """A corpus small enough to train in a test, varied enough to support ~800 merges.

    Variety matters more than size here. BPE runs out of pairs to merge once
    every distinct pre-token chunk has collapsed into a single id, so a corpus
    of 30 unique words supports only ~130 merges no matter how often they
    repeat. The synthetic vocabulary below supplies enough distinct words.
    """
    rng = random.Random(0)
    syl = ["ta", "ko", "ni", "za", "shun", "pre", "post", "ing", "ed", "ly",
           "tion", "ment", "able", "un", "re", "de", "con", "pro", "sub", "over"]
    words = ["".join(rng.choice(syl) for _ in range(rng.randint(1, 4)))
             for _ in range(400)]
    synthetic = " ".join(rng.choice(words) for _ in range(4000))
    prose = (
        "the quick brown fox jumps over the lazy dog. " * 20
        + "tokenization is the process of turning text into tokens. " * 20
        + "low lower lowest lowly slowly below. " * 20
        + "héllo wörld 🌍 你好世界 مرحبا بالعالم\n\t " * 10
    )
    return prose + " " + synthetic


CORPUS = _make_corpus()

# Strings that break naive tokenizers. Each one is here for a reason.
HARD_STRINGS = [
    "",                                  # empty
    " ",                                 # lone space
    "\n\n\t  ",                          # whitespace runs
    "a",                                 # single char
    "🌍",                                # 4-byte emoji, one char
    "👨‍👩‍👧‍👦",                              # ZWJ emoji sequence
    "héllo",                             # 2-byte latin
    "你好世界",                            # 3-byte CJK
    "مرحبا بالعالم",                      # RTL
    "\x00\x01\x02",                      # control bytes
    "é",                           # combining accent (NOT precomposed é)
    "a" * 500,                           # long repeat -> deep merge chain
    "<|endoftext|>",                     # special-token string as plain text
    "   trailing space   ",
    "x = [1, 2, 3]; print(f'{x!r}')",    # code + punctuation runs
    "1234567890 3.14159 -42",            # digits
]


@pytest.fixture(scope="module")
def basic():
    t = BasicBPE()
    t.train(CORPUS, vocab_size=512)
    return t


@pytest.fixture(scope="module")
def rbpe():
    t = RegexBPE()
    t.train(CORPUS, vocab_size=512)
    return t


# -- primitives -------------------------------------------------------------

def test_get_stats_counts_adjacent_pairs():
    assert get_stats([1, 2, 3, 2, 3]) == {(1, 2): 1, (2, 3): 2, (3, 2): 1}


def test_get_stats_empty():
    assert get_stats([]) == {}
    assert get_stats([7]) == {}


def test_merge_replaces_pair():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_is_non_overlapping():
    # [1,1,1] with pair (1,1): first two merge, the third is left alone.
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1]


def test_merge_no_match_is_identity():
    assert merge([1, 2, 3], (5, 6), 99) == [1, 2, 3]


# -- the worked example from docs/01_tokenizer.md §3 ------------------------

def test_known_merge_sequence():
    """Hand-checkable: 'low' repeated should merge l+o, then lo+w."""
    t = BasicBPE()
    t.train("low low low lower lowest", vocab_size=256 + 2)
    pairs = list(t.merges.keys())
    assert pairs[0] == (ord("l"), ord("o")), f"first merge should be (l,o), got {pairs[0]}"
    assert pairs[1] == (256, ord("w")), f"second merge should be (lo,w), got {pairs[1]}"
    # id 257 must expand back to the bytes b"low"
    assert t.vocab[257] == b"low"


# -- round-tripping: the core correctness property -------------------------

@pytest.mark.parametrize("s", HARD_STRINGS)
def test_roundtrip_hard_strings_basic(basic, s):
    assert basic.decode(basic.encode(s)) == s


@pytest.mark.parametrize("s", HARD_STRINGS)
def test_roundtrip_hard_strings_regex(rbpe, s):
    assert rbpe.decode(rbpe.encode(s)) == s


def test_roundtrip_1000_random_strings(rbpe):
    """Acceptance criterion: 1000 random held-out strings round-trip exactly."""
    rng = random.Random(1337)
    # Sample across the whole BMP so we hit multi-byte sequences and surrogates
    alphabet = [chr(c) for c in range(0x20, 0x2000)] + list("🌍😀你好héllo\n\t ")
    for _ in range(1000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert rbpe.decode(rbpe.encode(s)) == s, f"failed on {s!r}"


def test_roundtrip_raw_bytes_untrained():
    """With zero merges, encoding must be the identity on bytes -> no UNK ever."""
    t = BasicBPE()
    s = "anything at all 🌍 你好"
    assert t.encode(s) == list(s.encode("utf-8"))
    assert t.decode(t.encode(s)) == s


# -- vocab invariants -------------------------------------------------------

def test_all_256_bytes_have_ids(rbpe):
    """The no-UNK guarantee: every byte value is in the base vocab."""
    for b in range(256):
        assert rbpe.vocab[b] == bytes([b])


def test_vocab_size_matches_request(rbpe):
    assert len(rbpe.vocab) == 512


def test_training_stops_when_pairs_exhausted():
    """BPE cannot invent merges that the corpus does not contain.

    Ask for a 5000-token vocab from a corpus with a handful of distinct words
    and training stops early. Real consequence: a requested vocab size is an
    upper bound, not a guarantee, and the model's embedding table must be sized
    from the tokenizer that actually came out -- not from the config you asked
    for.
    """
    t = RegexBPE()
    t.train("low low lower lowest " * 50, vocab_size=5000)
    assert len(t.vocab) < 5000


def test_merge_parents_precede_children(rbpe):
    """A merged token's parents must have lower ids, or _build_vocab breaks."""
    for (p0, p1), idx in rbpe.merges.items():
        assert p0 < idx and p1 < idx


def test_decode_rejects_unknown_id(rbpe):
    with pytest.raises(ValueError):
        rbpe.decode([999_999])


# -- determinism ------------------------------------------------------------

def test_weighted_dedup_training_equals_naive():
    """RegexBPE dedups chunks and weights them. Prove that changes nothing.

    The naive reference below keeps one entry per occurrence and merges all of
    them every round. If the two merge lists differ, the weighting is wrong.
    """
    import regex as re_

    from bpe import GPT2_SPLIT_PATTERN

    text = CORPUS[:60_000]
    compiled = re_.compile(GPT2_SPLIT_PATTERN)

    chunks = [list(c.encode("utf-8")) for c in compiled.findall(text)]
    naive_merges: dict[tuple[int, int], int] = {}
    for i in range(300):
        stats: Counter = Counter()
        for ch in chunks:
            get_stats(ch, stats)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        chunks = [merge(ch, pair, 256 + i) for ch in chunks]
        naive_merges[pair] = 256 + i

    fast = RegexBPE()
    fast.train(text, 256 + 300)

    assert fast.merges == naive_merges


def test_merges_are_a_prefix_sequence():
    """Training to vocab V and truncating to v merges == training to v directly.

    train_production.py depends on this to read 2k/4k/8k tokenizers off one
    16k run. It holds because training is greedy and deterministic: merge k
    depends only on the state after merges 1..k-1.
    """
    long, short = RegexBPE(), RegexBPE()
    long.train(CORPUS, 800)
    short.train(CORPUS, 500)
    assert list(long.merges.items())[: 500 - 256] == list(short.merges.items())


def test_training_is_deterministic():
    a, b = RegexBPE(), RegexBPE()
    a.train(CORPUS, 400)
    b.train(CORPUS, 400)
    assert a.merges == b.merges


def test_encoding_is_deterministic(rbpe):
    s = "the quick brown fox tokenization"
    assert rbpe.encode(s) == rbpe.encode(s)


# -- special tokens: the security property ---------------------------------

def test_specials_not_forgeable_by_default(rbpe):
    """User text containing '<|endoftext|>' must NOT become the control id.

    Otherwise a user can forge a document/turn boundary. docs §5.
    """
    rbpe.register_special_tokens({"<|endoftext|>": 512})
    ids = rbpe.encode("hello <|endoftext|> world")        # default: allowed_special="none"
    assert 512 not in ids
    assert rbpe.decode(ids) == "hello <|endoftext|> world"


def test_specials_used_when_explicitly_allowed(rbpe):
    rbpe.register_special_tokens({"<|endoftext|>": 512})
    ids = rbpe.encode("a<|endoftext|>b", allowed_special="all")
    assert 512 in ids
    assert rbpe.decode(ids) == "a<|endoftext|>b"


def test_bad_allowed_special_raises(rbpe):
    rbpe.register_special_tokens({"<|endoftext|>": 512})
    with pytest.raises(ValueError):
        rbpe.encode("x", allowed_special="maybe")


# -- the regex pre-tokenization ablation -----------------------------------

def test_basic_bpe_merges_across_word_boundaries():
    """The ablation, asserted: without regex, merges span spaces.

    This is the defect regex pre-tokenization exists to prevent.
    """
    t = BasicBPE()
    t.train(CORPUS, vocab_size=800)
    spanning = [
        t.vocab[i] for i in t.vocab
        if i >= 256 and b" " in t.vocab[i][1:]   # a space in the middle, not just leading
    ]
    assert spanning, "expected BasicBPE to learn boundary-spanning tokens"


def test_regex_bpe_never_merges_across_word_boundaries():
    t = RegexBPE()
    t.train(CORPUS, vocab_size=800)
    for i, tok in t.vocab.items():
        if i < 256:
            continue
        # Two legitimate cases are excluded:
        #   - a LEADING space: " the" is the canonical form of a word (' ?\p{L}+')
        #   - an ALL-WHITESPACE token: the '\s+' branch makes whitespace runs
        #     their own chunk, so "\n\t  " merging into one token is correct
        if tok.strip() == b"":
            continue
        assert b" " not in tok[1:], f"token {i} = {tok!r} spans a word boundary"


def test_regex_tokens_carry_leading_space():
    """' ?\\p{L}+' means the canonical form of a word includes its leading space."""
    t = RegexBPE()
    t.train(CORPUS, vocab_size=800)
    assert any(tok.startswith(b" ") and len(tok) > 1
               for i, tok in t.vocab.items() if i >= 256)


# -- compression ------------------------------------------------------------

def test_compression_improves_with_vocab_size():
    ratios = []
    for v in (256, 512, 1024):
        t = RegexBPE()
        t.train(CORPUS, v)
        ratios.append(compression_ratio(t, CORPUS))
    assert ratios[0] == pytest.approx(1.0, abs=0.01), "vocab 256 = raw bytes = 1.0"
    assert ratios[1] > ratios[0] and ratios[2] > ratios[1]


# -- persistence ------------------------------------------------------------

def test_save_load_roundtrip(tmp_path, rbpe):
    p = tmp_path / "tok.json"
    rbpe.save(str(p))
    loaded = RegexBPE.load(str(p))
    assert loaded.merges == rbpe.merges
    s = "the quick brown 🌍 fox"
    assert loaded.encode(s) == rbpe.encode(s)
    assert loaded.decode(loaded.encode(s)) == s


def test_loaded_basic_stays_basic(tmp_path, basic):
    p = tmp_path / "basic.json"
    basic.save(str(p))
    loaded = BasicBPE.load(str(p))
    assert not hasattr(loaded, "compiled"), "BasicBPE must not gain a regex on load"
    assert loaded.encode("the lazy dog") == basic.encode("the lazy dog")
