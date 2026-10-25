"""MinHash/LSH near-dedup correctness. uv run pytest llmfs/data -q"""

import random

import numpy as np
import pytest

from llmfs.data.dedup import (
    UnionFind, choose_drops, cluster, estimated_jaccard, make_perms, shingles, signature,
    true_jaccard,
)

A, B = make_perms()


def _code(seed: int, n_lines: int = 60) -> str:
    rng = random.Random(seed)
    names = [f"var_{rng.randint(0, 10**6)}" for _ in range(40)]
    return "\n".join(
        f"def {rng.choice(names)}({rng.choice(names)}, {rng.choice(names)}):\n"
        f"    return {rng.choice(names)} + {rng.randint(0, 999)}"
        for _ in range(n_lines))


def sig(t):
    return signature(shingles(t), A, B)


def test_identical_documents_estimate_one():
    t = _code(1)
    assert estimated_jaccard(sig(t), sig(t)) == 1.0


def test_reformatting_does_not_change_the_fingerprint():
    """Only \\w+ runs are shingled, so whitespace/punctuation edits are invisible."""
    t = _code(2)
    reformatted = t.replace("    ", "\t").replace(", ", ",").replace("\n", "\n\n")
    assert estimated_jaccard(sig(t), sig(reformatted)) == 1.0


def test_small_edit_stays_similar():
    t = _code(3)
    lines = t.splitlines()
    lines[10] = "    return totally_new_identifier * 42"
    assert estimated_jaccard(sig(t), sig("\n".join(lines))) > 0.85


def test_unrelated_documents_are_dissimilar():
    assert estimated_jaccard(sig(_code(4)), sig(_code(5))) < 0.1


def test_estimate_tracks_true_jaccard():
    """Mean |est - true| over many pairs; per-pair std is <= 0.046 at P=120."""
    rng = random.Random(0)
    errs = []
    base = _code(6, 200).splitlines()
    for k in range(25):
        other = list(base)
        for i in rng.sample(range(len(other)), k * 8 % len(other)):
            other[i] = f"x_{rng.randint(0, 10**9)} = {rng.randint(0, 10**9)}"
        t1, t2 = "\n".join(base), "\n".join(other)
        errs.append(abs(estimated_jaccard(sig(t1), sig(t2)) - true_jaccard(t1, t2)))
    assert np.mean(errs) < 0.05 and max(errs) < 0.15


def test_short_documents_have_no_shingles():
    assert len(shingles("x = 1")) == 0


def test_shingles_are_a_set():
    t = "a b c d e " * 50                       # one distinct 5-gram... and its rotations
    assert len(shingles(t)) == len(np.unique(shingles(t))) <= 5


def test_signatures_identical_across_processes():
    """The whole reason built-in hash() is banned in this module."""
    import multiprocessing as mp
    t = _code(7)
    with mp.get_context("spawn").Pool(2) as pool:
        remote = pool.map(_remote_sig, [t, t])
    assert np.array_equal(remote[0], sig(t)) and np.array_equal(remote[1], sig(t))


def _remote_sig(t):
    from llmfs.data.dedup import make_perms, shingles, signature
    a, b = make_perms()
    return signature(shingles(t), a, b)


def test_cluster_groups_near_dups_only():
    base = _code(8)
    lines = base.splitlines()
    lines[3] = "    return edited_line + 1"
    docs = [base, _code(9), "\n".join(lines), _code(10), base]
    uf, _ = cluster([sig(d) for d in docs])
    roots = [uf.find(i) for i in range(len(docs))]
    assert roots[0] == roots[2] == roots[4]
    assert len({roots[0], roots[1], roots[3]}) == 3


def test_cluster_skips_none_signatures():
    uf, st = cluster([None, sig(_code(11)), None])
    assert st["candidates"] == 0 and [uf.find(i) for i in range(3)] == [0, 1, 2]


def test_choose_drops_prefers_train_and_drops_val_copies():
    uf = UnionFind(5)
    uf.union(0, 1); uf.union(1, 2)              # cluster {0,1,2}: val, train, val
    uf.union(3, 4)                              # cluster {3,4}: val, val
    is_val = [True, False, True, True, True]
    drop, st = choose_drops(uf, is_val)
    assert drop == {0, 2, 4}                    # keep 1 (the train copy) and 3
    assert st["val_dropped_due_to_train_copy"] == 2 and st["train_dropped"] == 0


@pytest.mark.parametrize("n", [1, 2, 50])
def test_union_find_is_deterministic(n):
    uf = UnionFind(n)
    for i in range(n - 1):
        uf.union(n - 1 - i, n - 2 - i)
    assert all(uf.find(i) == 0 for i in range(n))
