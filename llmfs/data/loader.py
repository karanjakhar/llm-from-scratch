"""Batch sampling from flat uint16 token binaries. docs/02_data.md §9-§11.

Two samplers:

    RandomSampler    random offsets, with replacement. nanoGPT's approach.
                     Simple, but one epoch's worth of samples leaves ~37% of
                     tokens unseen (§10). Kept as the baseline.
    ShuffledSampler  non-overlapping windows, seeded permutation per epoch.
                     Every window exactly once per epoch. What we train with.

MixedLoader draws each sequence from one source (code / English) with
probability equal to its weight, so the mix ratio is a runtime argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def open_bin(path: str | Path) -> np.memmap:
    """Map a token binary without reading it. Pages load on first touch."""
    return np.memmap(path, dtype=np.uint16, mode="r")


class RandomSampler:
    """Window starts drawn uniformly from [0, N - T - 1]. Stateless apart from the RNG."""

    def __init__(self, n_tokens: int, T: int, seed: int = 0):
        assert n_tokens > T + 1, "binary shorter than one window"
        self.hi = n_tokens - T - 1
        self.rng = np.random.default_rng(seed)

    def next(self) -> int:
        return int(self.rng.integers(0, self.hi + 1))

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, s):
        self.rng.bit_generator.state = s["rng"]


class ShuffledSampler:
    """Non-overlapping windows of T+1 tokens' reach, visited in a seeded random order.

    Window w covers inputs [w*T, w*T + T) and targets shifted by one, so it
    needs w*T + T + 1 <= N. Consecutive windows share one token at the seam
    (the last target of w is the first input of w+1); that is the minimum
    possible overlap for next-token prediction and costs nothing.

    The permutation for epoch e is a pure function of (seed, e), so resuming
    needs only (epoch, pos) -- no index array saved to disk.
    """

    def __init__(self, n_tokens: int, T: int, seed: int = 0):
        self.T = T
        self.n_windows = (n_tokens - 1) // T
        assert self.n_windows >= 1, "binary shorter than one window"
        self.seed = seed
        self.epoch = 0
        self.pos = 0
        self._perm = self._make_perm(0)

    def _make_perm(self, epoch: int) -> np.ndarray:
        return np.random.default_rng([self.seed, epoch]).permutation(self.n_windows)

    def next(self) -> int:
        if self.pos == self.n_windows:
            self.epoch += 1
            self.pos = 0
            self._perm = self._make_perm(self.epoch)
        w = int(self._perm[self.pos])
        self.pos += 1
        return w * self.T

    def state_dict(self):
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, s):
        self.epoch, self.pos = s["epoch"], s["pos"]
        self._perm = self._make_perm(self.epoch)


SAMPLERS = {"random": RandomSampler, "shuffled": ShuffledSampler}


class MixedLoader:
    """Yields (x, y) int64 batches of shape (B, T) mixed across sources.

    sources: {"code": (path, 0.85), "en": (path, 0.15)}. A weight of 0 drops a
    source entirely; weights are renormalized, so (1, 0) means 100% code.
    """

    def __init__(self, sources: dict[str, tuple[str | Path, float]], B: int, T: int,
                 device: str = "cpu", sampler: str = "shuffled", seed: int = 0,
                 pin: bool = True):
        self.names = [n for n, (_, w) in sources.items() if w > 0]
        assert self.names, "all source weights are zero"
        w = np.array([sources[n][1] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.data = {n: open_bin(sources[n][0]) for n in self.names}
        # Each source gets its own sampler stream, seeded independently, so
        # changing the mix ratio does not change which windows a source yields.
        self.samplers = {n: SAMPLERS[sampler](len(self.data[n]), T, seed=seed * 1000 + i)
                         for i, n in enumerate(self.names)}
        self.mix_rng = np.random.default_rng([seed, 7])
        self.B, self.T = B, T
        self.device = torch.device(device)
        self.pin = pin and self.device.type == "cuda"
        self.counts = {n: 0 for n in self.names}   # sequences drawn per source

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        picks = self.mix_rng.choice(len(self.names), size=B, p=self.weights)
        buf = np.empty((B, T + 1), dtype=np.int64)               # (B, T+1)
        for row, k in enumerate(picks):
            name = self.names[k]
            i = self.samplers[name].next()
            # One contiguous T+1 read from the memmap; the uint16 -> int64 cast
            # happens here, on a few KB, never on the whole file. docs §6.
            buf[row] = self.data[name][i : i + T + 1]
            self.counts[name] += 1
        t = torch.from_numpy(buf)
        if self.pin:
            t = t.pin_memory()                                   # enables async DMA copy
        t = t.to(self.device, non_blocking=self.pin)
        return t[:, :-1], t[:, 1:]                               # x (B, T), y (B, T)

    def state_dict(self):
        return {"samplers": {n: s.state_dict() for n, s in self.samplers.items()},
                "mix_rng": self.mix_rng.bit_generator.state, "counts": dict(self.counts)}

    def load_state_dict(self, s):
        for n, st in s["samplers"].items():
            self.samplers[n].load_state_dict(st)
        self.mix_rng.bit_generator.state = s["mix_rng"]
        self.counts = dict(s["counts"])
