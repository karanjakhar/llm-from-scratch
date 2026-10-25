"""Near-duplicate detection: MinHash signatures, LSH banding, union-find.

docs/02_data.md §4. The pipeline:

    text -> word 5-gram shingles (a SET) -> MinHash signature (120 x uint32)
         -> LSH: 20 bands x 6 rows; docs sharing any band become candidates
         -> verify candidates by estimated Jaccard >= 0.7
         -> union-find into clusters -> keep one doc per cluster

Every hash here is deterministic across processes. Python's built-in hash()
is salted per interpreter (PYTHONHASHSEED), and signatures are computed in
spawned worker processes, so using it would silently make every worker's
signatures incomparable with every other's.
"""

from __future__ import annotations

import re
import zlib

import numpy as np

NGRAM = 5
NUM_PERM = 120
BANDS, ROWS = 20, 6            # BANDS * ROWS == NUM_PERM
THRESHOLD = 0.7
assert BANDS * ROWS == NUM_PERM

# Identifier/number/keyword runs only. Ignoring whitespace and punctuation
# makes the fingerprint robust to reformatting: a fork run through `black`
# still matches its original.
_WORD = re.compile(rb"\w+")
_ROLL = np.uint64(0x100000001B3)          # FNV-1a 64-bit prime, odd
_M32 = np.uint64(0xFFFFFFFF)


def make_perms(num_perm: int = NUM_PERM, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Parameters for num_perm independent hash functions h(x) = (a*x + b) >> 32.

    With 32-bit keys x, a random odd 64-bit `a`, random 64-bit `b`, and
    arithmetic mod 2^64, taking the top 32 bits is the multiply-add-shift
    family (Dietzfelbinger 1996) -- 2-independent, which is what MinHash needs.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)
    return a, b


def shingles(text: str) -> np.ndarray:
    """The SET of word-5-gram hashes, as unique uint32. Empty if < NGRAM words.

    Jaccard similarity is defined on sets, so a shingle that occurs 40 times
    counts once -- np.unique is load-bearing, not a micro-optimization.
    """
    words = _WORD.findall(text.encode("utf-8"))
    n = len(words) - NGRAM + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    w = np.fromiter((zlib.crc32(x) for x in words), dtype=np.uint64, count=len(words))
    h = np.zeros(n, dtype=np.uint64)
    for k in range(NGRAM):                         # polynomial roll over the window,
        h = h * _ROLL + w[k : k + n]               # wrapping mod 2^64 by design
    h = (h ^ (h >> np.uint64(32))) & _M32          # fold to 32-bit keys (see make_perms)
    return np.unique(h)


def signature(sh: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """MinHash: for each of P hash functions, the minimum over all shingles.

    P(sig_A[i] == sig_B[i]) == Jaccard(A, B), for each i independently. So the
    fraction of matching positions is an unbiased estimate of Jaccard with
    std sqrt(J(1-J)/P) -- at most 0.046 for P = 120.
    """
    sig = np.full(len(a), np.iinfo(np.uint64).max, dtype=np.uint64)
    for s in range(0, len(sh), chunk):             # chunked: a 900 KB file has ~100k shingles
        x = sh[s : s + chunk]
        hv = a[:, None] * x[None, :] + b[:, None]  # (P, n), wraps mod 2^64
        np.minimum(sig, hv.min(axis=1), out=sig)
    # >> 32 is monotone, so min-then-shift == shift-then-min.
    return (sig >> np.uint64(32)).astype(np.uint32)


def estimated_jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
    return float(np.mean(s1 == s2))


def true_jaccard(t1: str, t2: str) -> float:
    a, b = set(shingles(t1).tolist()), set(shingles(t2).tolist())
    return len(a & b) / max(1, len(a | b))


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:                         # path compression
            p[x], x = root, p[x]
        return int(root)

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)  # lower index stays root: deterministic


def cluster(sigs: list[np.ndarray | None], threshold: float = THRESHOLD) -> tuple[UnionFind, dict]:
    """LSH-band every signature, verify candidates, union the matches.

    Each band bucket remembers only the FIRST doc that landed in it, and later
    docs are compared against that one. This is an approximation: if the
    first doc fails verification against a newcomer, a later doc in the same
    bucket that would have passed is not tried. Real pipelines accept the same
    tradeoff to keep memory linear in the number of docs.
    """
    uf = UnionFind(len(sigs))
    tables: list[dict[bytes, int]] = [dict() for _ in range(BANDS)]
    stats = {"candidates": 0, "verified": 0}
    for i, s in enumerate(sigs):
        if s is None:
            continue
        cands = set()
        for band in range(BANDS):
            key = s[band * ROWS : (band + 1) * ROWS].tobytes()
            j = tables[band].setdefault(key, i)
            if j != i:
                cands.add(j)
        for j in cands:
            stats["candidates"] += 1
            if estimated_jaccard(s, sigs[j]) >= threshold:
                stats["verified"] += 1
                uf.union(i, j)
    return uf, stats


def choose_drops(uf: UnionFind, is_val: list[bool]) -> tuple[set[int], dict]:
    """Keep one doc per cluster, preferring TRAIN. Returns indices to drop.

    Preferring train is the point: a val file whose near-copy is in train is
    leakage, so it is the val copy that goes. A cluster entirely inside val
    keeps its first member, so val is deduplicated against itself too.
    """
    clusters: dict[int, list[int]] = {}
    for i in range(len(is_val)):
        clusters.setdefault(uf.find(i), []).append(i)
    drop: set[int] = set()
    st = {"clusters_with_dups": 0, "train_dropped": 0, "val_dropped": 0,
          "val_dropped_due_to_train_copy": 0}
    for members in clusters.values():
        if len(members) == 1:
            continue
        st["clusters_with_dups"] += 1
        train = [m for m in members if not is_val[m]]
        keep = train[0] if train else members[0]
        for m in members:
            if m == keep:
                continue
            drop.add(m)
            if is_val[m]:
                st["val_dropped"] += 1
                st["val_dropped_due_to_train_copy"] += bool(train)
            else:
                st["train_dropped"] += 1
    return drop, st
