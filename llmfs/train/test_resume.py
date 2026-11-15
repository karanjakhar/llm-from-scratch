"""A stopped-and-resumed run must continue exactly where it left off.

Stage 11's resume_test checks this on a real model and real data (results/11_pretrain/
resume_test.json: bit-identical at 3.9158). This is the small version, on random tokens,
so it runs in seconds and can grade the course's module 9 (docs/COURSE.md §4).
Everything the next step depends on must be in the checkpoint: weights, Adam's
state, the loader's position and the random number generators.
"""

import numpy as np
import pytest
import torch

from llmfs.train.trainer import TrainConfig, train

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer runs on CUDA")


@pytest.fixture(scope="module")
def bins(tmp_path_factory):
    d = tmp_path_factory.mktemp("bins")
    rng = np.random.default_rng(0)
    for name, n in (("train", 200_000), ("val", 20_000)):
        rng.integers(0, 512, n, dtype=np.uint16).tofile(d / f"{name}.bin")
    return d


def _config(bins, out, **kw):
    return TrainConfig(model=dict(n_layer=2, n_head=2, n_embd=64, vocab_size=512),
                       sources={"x": (str(bins / "train.bin"), 1.0)}, val={"x": str(bins / "val.bin")},
                       B=4, T=32, steps=20, lr=1e-3, warmup=5, eval_every=0, eval_windows=4,
                       dtype="fp32", out_dir=str(out), ckpt_every=10, **kw)


def _losses(c):
    seen = {}
    train(c, on_step=lambda step, rec, model: seen.__setitem__(step, rec["loss"]))
    return seen


def test_resume_is_bit_identical(bins, tmp_path):
    straight = _losses(_config(bins, tmp_path / "a"))
    first = _losses(_config(bins, tmp_path / "b", stop_after=10))
    second = _losses(_config(bins, tmp_path / "b", resume=True))
    assert sorted(first) == list(range(10)) and sorted(second) == list(range(10, 20))
    assert {**first, **second} == straight


def test_resume_without_optimizer_state_differs(bins, tmp_path):
    straight = _losses(_config(bins, tmp_path / "a"))
    _losses(_config(bins, tmp_path / "b", stop_after=10))
    fresh = _losses(_config(bins, tmp_path / "b", resume=True, resume_drop_optimizer=True))
    assert any(fresh[s] != straight[s] for s in range(10, 20))
