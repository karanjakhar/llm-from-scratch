"""Stage 02 acceptance criteria (docs/02_data.md §13), as tests.

    uv run pytest stages/02_data -q
"""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "01_tokenizer"))
sys.path.insert(0, str(Path(__file__).parent))

from bpe import RegexBPE  # noqa: E402
from prepare import Deduper, tokenize_to_bin  # noqa: E402

from llmfs.data import MixedLoader, RandomSampler, ShuffledSampler, open_bin  # noqa: E402
from llmfs.data.corpus import Doc, split_of  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402


# -- fixtures -----------------------------------------------------------------

def _docs(n=200, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        body = "".join(rng.choice(["def f(x):\n", "    return x\n", "import os\n",
                                   "# héllo 🌍\n", "y = 2024\n", "class A:\n    pass\n"])
                       for _ in range(rng.randint(1, 30)))
        out.append(Doc("code", f"user{i % 37}/repo{i % 37}", body))
    return out


@pytest.fixture(scope="module")
def tok_path(tmp_path_factory):
    t = RegexBPE()
    t.train("".join(d.text for d in _docs()), 400)
    t.register_special_tokens({"<|endoftext|>": 400, "<|pad|>": 401})
    p = tmp_path_factory.mktemp("tok") / "tok.json"
    t.save(str(p))
    return str(p)


# -- split ----------------------------------------------------------------------

def test_split_is_deterministic():
    assert all(split_of(f"r{i}") == split_of(f"r{i}") for i in range(1000))


def test_split_fraction_is_half_a_percent():
    keys = [f"owner{i}/repo{i}" for i in range(200_000)]
    frac = sum(split_of(k) == "val" for k in keys) / len(keys)
    assert 0.004 < frac < 0.006


def test_split_does_not_depend_on_python_hash_seed():
    """hash() is salted per process; the split must survive a fresh interpreter."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); from llmfs.data.corpus import split_of;"
            "print(''.join('v' if split_of(f'k{i}')=='val' else 't' for i in range(3000)))"
            % str(REPO))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": s}).stdout for s in ("1", "2")}
    assert len(outs) == 1


def test_no_repository_on_both_sides():
    docs = _docs(2000)
    train = {d.key for d in docs if split_of(d.key) == "train"}
    val = {d.key for d in docs if split_of(d.key) == "val"}
    assert not (train & val)


# -- dedup ----------------------------------------------------------------------

def test_dedup_drops_exact_copies_only():
    a, b = Doc("code", "r1", "x = 1\n"), Doc("code", "r2", "x = 1\n")
    c = Doc("code", "r3", "x = 1 \n")                        # one byte different: kept
    st = {"dup_docs": 0}
    kept = list(Deduper()([a, b, c], st))
    assert [d.key for d in kept] == ["r1", "r3"] and st["dup_docs"] == 1


def test_dedup_shared_across_splits_drops_val_copy_of_train():
    """prepare.py runs train first with one Deduper: a fork in val gets dropped."""
    dd = Deduper()
    train = [Doc("code", "orig/repo", "def f(): pass\n")]
    val = [Doc("code", "fork/repo", "def f(): pass\n"), Doc("code", "new/repo", "z=3\n")]
    s1, s2 = {"dup_docs": 0}, {"dup_docs": 0}
    list(dd(train, s1))
    kept_val = list(dd(val, s2))
    assert [d.key for d in kept_val] == ["new/repo"] and s2["dup_docs"] == 1


# -- tokenize_to_bin ----------------------------------------------------------------

def _bin(tmp_path, tok_path, docs, workers, name="t.bin"):
    out = tmp_path / name
    stats = tokenize_to_bin(iter(docs), tok_path, out, workers=workers, batch_docs=7)
    return out, stats


def test_bins_identical_for_any_worker_count(tmp_path, tok_path):
    docs = _docs(150)
    p1, _ = _bin(tmp_path, tok_path, docs, 1, "a.bin")
    p4, _ = _bin(tmp_path, tok_path, docs, 4, "b.bin")
    assert hashlib.sha256(p1.read_bytes()).digest() == hashlib.sha256(p4.read_bytes()).digest()


def test_every_document_ends_with_exactly_one_eot(tmp_path, tok_path):
    docs = _docs(80)
    p, stats = _bin(tmp_path, tok_path, docs, 2)
    arr = open_bin(p)
    tok = Tokenizer.load(tok_path)
    assert arr[-1] == tok.eot_id
    assert int((arr == tok.eot_id).sum()) == len(docs) == stats["docs"]


def test_bin_decodes_back_to_source_exactly(tmp_path, tok_path):
    docs = _docs(60)
    p, _ = _bin(tmp_path, tok_path, docs, 3)
    arr = np.asarray(open_bin(p))
    tok = Tokenizer.load(tok_path)
    cuts = np.flatnonzero(arr == tok.eot_id)
    starts = np.concatenate([[0], cuts[:-1] + 1])
    decoded = [tok.decode(arr[s:e]) for s, e in zip(starts, cuts)]
    assert decoded == [d.text for d in docs]


def test_literal_eot_in_source_is_not_a_boundary(tmp_path, tok_path):
    docs = [Doc("code", "r", 'EOT = "<|endoftext|>"\n'), Doc("code", "r", "x = 1\n")]
    p, _ = _bin(tmp_path, tok_path, docs, 1)
    tok = Tokenizer.load(tok_path)
    assert int((open_bin(p) == tok.eot_id).sum()) == 2


# -- samplers -----------------------------------------------------------------

def test_shuffled_sampler_covers_every_window_once_per_epoch():
    N, T = 10_001, 100
    s = ShuffledSampler(N, T, seed=3)
    starts = [s.next() for _ in range(s.n_windows)]
    assert sorted(starts) == [w * T for w in range(s.n_windows)]
    assert s.epoch == 0
    s.next()
    assert s.epoch == 1                                   # rolled over


def test_shuffled_sampler_reorders_each_epoch():
    s = ShuffledSampler(10_001, 100, seed=0)
    e0 = [s.next() for _ in range(s.n_windows)]
    e1 = [s.next() for _ in range(s.n_windows)]
    assert sorted(e0) == sorted(e1) and e0 != e1


def test_random_sampler_misses_about_37_percent():
    """docs §10: one epoch of with-replacement sampling covers ~1 - 1/e of tokens."""
    N, T = 200_000, 50
    s = RandomSampler(N, T, seed=0)
    covered = np.zeros(N, dtype=bool)
    for _ in range(N // T):                               # one epoch's worth of windows
        i = s.next()
        covered[i : i + T + 1] = True
    assert abs(covered.mean() - (1 - np.exp(-1))) < 0.02


@pytest.mark.parametrize("kind", ["random", "shuffled"])
def test_sampler_resume_is_exact(kind):
    cls = RandomSampler if kind == "random" else ShuffledSampler
    a = cls(5_001, 10, seed=9)
    for _ in range(777):
        a.next()
    b = cls(5_001, 10, seed=9)
    b.load_state_dict(a.state_dict())
    assert [a.next() for _ in range(900)] == [b.next() for _ in range(900)]


# -- MixedLoader ----------------------------------------------------------------

@pytest.fixture
def two_bins(tmp_path):
    code = np.arange(0, 50_000, dtype=np.uint16) % 30_000            # values < 30000
    en = (np.arange(0, 20_000, dtype=np.uint16) % 1_000) + 40_000    # values >= 40000
    pc, pe = tmp_path / "code.bin", tmp_path / "en.bin"
    code.tofile(pc)
    en.tofile(pe)
    return pc, pe


def test_loader_shapes_and_shift(two_bins):
    pc, pe = two_bins
    ld = MixedLoader({"code": (pc, 1.0)}, B=4, T=32)
    x, y = ld.next_batch()
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])               # y is x shifted by one


def test_loader_mix_ratio(two_bins):
    pc, pe = two_bins
    ld = MixedLoader({"code": (pc, 0.85), "en": (pe, 0.15)}, B=64, T=16, seed=1)
    en_rows = total = 0
    for _ in range(100):
        x, _ = ld.next_batch()
        en_rows += int((x[:, 0] >= 40_000).sum())
        total += x.shape[0]
    assert abs(en_rows / total - 0.15) < 0.02


def test_loader_zero_weight_excludes_source(two_bins):
    pc, pe = two_bins
    ld = MixedLoader({"code": (pc, 1.0), "en": (pe, 0.0)}, B=32, T=16)
    for _ in range(20):
        x, _ = ld.next_batch()
        assert int((x >= 40_000).sum()) == 0


def test_loader_resume_is_exact(two_bins):
    pc, pe = two_bins
    srcs = {"code": (pc, 0.7), "en": (pe, 0.3)}
    a = MixedLoader(srcs, B=8, T=16, seed=4)
    for _ in range(37):
        a.next_batch()
    b = MixedLoader(srcs, B=8, T=16, seed=4)
    b.load_state_dict(a.state_dict())
    for _ in range(20):
        (xa, _), (xb, _) = a.next_batch(), b.next_batch()
        assert torch.equal(xa, xb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_loader_to_cuda(two_bins):
    pc, _ = two_bins
    x, y = MixedLoader({"code": (pc, 1.0)}, B=4, T=16, device="cuda").next_batch()
    assert x.is_cuda and y.is_cuda
