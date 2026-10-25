"""The fast encoder must be the SAME tokenizer as stage 01, only faster.

    uv run pytest llmfs/tokenizer -q
"""

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stages" / "01_tokenizer"))
sys.path.insert(0, str(REPO))

from bpe import GPT4_SPLIT_PATTERN, RegexBPE  # noqa: E402  (stage 01 reference)

from llmfs.tokenizer import Tokenizer  # noqa: E402

SPECIALS = {"<|endoftext|>": 600, "<|pad|>": 601}


def _corpus() -> str:
    rng = random.Random(0)
    code = (
        "def forward(self, x):\n    return self.proj(x) + 1\n\n"
        "for i in range(10):\n    print(f'{i:04d}', i * 2024)\n"
        "class Foo(Bar):\n    '''Docstring.'''\n    pass\n"
    ) * 30
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 7)))
             for _ in range(300)]
    return code + " ".join(rng.choice(words) for _ in range(3000)) + " héllo 🌍 你好"


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """(stage-01 reference, fast llmfs) loaded from the SAME json file."""
    ref = RegexBPE()
    ref.train(_corpus(), 600)
    ref.register_special_tokens(SPECIALS)
    path = tmp_path_factory.mktemp("tok") / "t.json"
    ref.save(str(path))
    return RegexBPE.load(str(path)), Tokenizer.load(str(path))


SAMPLES = [
    "", " ", "\n\n\t  ", "def f(x):\n    return x+1\n", "🌍👨‍👩‍👧‍👦", "你好世界",
    "é", "\x00\x01", "x" * 300, "numbers 2024 1234567 3.14159",
    "<|endoftext|> literally in the text",
]


@pytest.mark.parametrize("s", SAMPLES)
def test_identical_to_stage01(pair, s):
    ref, fast = pair
    assert fast.encode_ordinary(s) == ref.encode_ordinary(s)


def test_identical_on_random_unicode(pair):
    ref, fast = pair
    rng = random.Random(7)
    alphabet = [chr(c) for c in range(0x20, 0x3000)] + list("\n\t    ")
    for _ in range(300):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        assert fast.encode_ordinary(s) == ref.encode_ordinary(s), repr(s)


def test_identical_with_a_different_pattern():
    ref = RegexBPE(GPT4_SPLIT_PATTERN)
    ref.train(_corpus(), 500)
    fast = Tokenizer(ref.merges, ref.pattern)
    s = _corpus()[:5000]
    assert fast.encode_ordinary(s) == ref.encode_ordinary(s)


def test_cache_flush_does_not_change_output(pair):
    """A tiny cache forces constant flushing; ids must not change."""
    ref, _ = pair
    tiny = Tokenizer(ref.merges, ref.pattern, SPECIALS, cache_size=3)
    s = _corpus()[:4000]
    assert tiny.encode_ordinary(s) == ref.encode_ordinary(s)


def test_cache_actually_hits(pair):
    _, fast = pair
    fast.cache_hits = fast.cache_misses = 0
    fast.encode_ordinary(_corpus())
    assert fast.cache_hits > 10 * fast.cache_misses, "Zipf head should dominate"


def test_roundtrip(pair):
    _, fast = pair
    for s in SAMPLES:
        assert fast.decode(fast.encode_ordinary(s)) == s


def test_encode_document_appends_exactly_one_eot(pair):
    _, fast = pair
    ids = fast.encode_document("x = 1\n")
    assert ids[-1] == fast.eot_id
    assert ids.count(fast.eot_id) == 1


def test_corpus_text_cannot_inject_eot(pair):
    """Tokenizer source code on GitHub contains the literal '<|endoftext|>'.

    It must encode as plain text, or a file could forge a document boundary
    in the middle of itself.
    """
    _, fast = pair
    ids = fast.encode_document('EOT = "<|endoftext|>"\n')
    assert ids.count(fast.eot_id) == 1 and ids[-1] == fast.eot_id
    assert fast.decode(ids[:-1]) == 'EOT = "<|endoftext|>"\n'


def test_specials_only_when_allowed(pair):
    _, fast = pair
    assert fast.eot_id not in fast.encode("a<|endoftext|>b")
    assert fast.eot_id in fast.encode("a<|endoftext|>b", allowed_special="all")


def test_vocab_size_covers_specials(pair):
    _, fast = pair
    assert fast.n_learned == 600
    assert fast.vocab_size == 602          # ids 0..601 -> 602 embedding rows


def test_load_rejects_basic_bpe(tmp_path):
    sys.path.insert(0, str(REPO / "stages" / "01_tokenizer"))
    from bpe import BasicBPE
    b = BasicBPE()
    b.train("aaa bbb", 260)
    b.save(str(tmp_path / "b.json"))
    with pytest.raises(ValueError):
        Tokenizer.load(str(tmp_path / "b.json"))


# -- on the real corpus, when it has been downloaded -------------------------

PROD = REPO / "data" / "tokenizers" / "grouped_2048.json"


@pytest.mark.skipif(not PROD.exists(), reason="production tokenizers not trained yet")
def test_identical_on_real_files():
    from llmfs.data.corpus import code_shards, en_shards, iter_code_docs, iter_en_docs
    import itertools
    ref = RegexBPE.load(str(PROD))
    fast = Tokenizer.load(str(PROD))
    raw = REPO / "data" / "raw"
    docs = list(itertools.islice(iter_code_docs(code_shards(raw)[:1], "val"), 40))
    docs += list(itertools.islice(iter_en_docs(en_shards(raw), "val"), 20))
    for d in docs:
        assert fast.encode_ordinary(d.text) == ref.encode_ordinary(d.text), d.key
