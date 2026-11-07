"""Acceptance criteria for Stage 20 (docs/20_reasoning_rlvr.md §7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.post.grpo import group_advantages, grpo_loss  # noqa: E402


def test_advantages_zero_mean_and_zero_for_uniform_groups():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    a = group_advantages(r)
    assert torch.allclose(a.mean(1), torch.zeros(3), atol=1e-6)
    assert torch.equal(a[1], torch.zeros(4)) and torch.equal(a[2], torch.zeros(4))
    assert a[0, 0].item() == pytest.approx(1.0, abs=1e-3)


def test_loss_on_policy_with_no_kl_is_minus_advantage():
    lp = torch.log(torch.full((2, 5), 0.3))
    mask = torch.ones(2, 5)
    loss, st = grpo_loss(lp, lp, lp, mask, torch.tensor([1.0, -1.0]), beta=0.04)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)          # +1 and -1 cancel; kl 0
    loss, st = grpo_loss(lp, lp, lp, mask, torch.tensor([2.0, 2.0]))
    assert loss.item() == pytest.approx(-2.0) and st["kl"] == pytest.approx(0.0)


def test_clip_stops_large_ratio_gains():
    old = torch.zeros(1, 1)
    new = torch.log(torch.tensor([[2.0]]))                     # ratio 2
    loss, _ = grpo_loss(new, old, new, torch.ones(1, 1), torch.tensor([1.0]), clip=0.2, beta=0.0)
    assert loss.item() == pytest.approx(-1.2)                  # clipped at 1 + ε


def test_k3_kl_nonnegative():
    ref = torch.randn(3, 7)
    lp = torch.randn(3, 7)
    _, st = grpo_loss(lp, lp, ref, torch.ones(3, 7), torch.zeros(3), beta=1.0)
    assert st["kl"] >= 0


def test_easy_tasks_references_pass_and_split_differs():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from easy_tasks import HELDOUT, TRAIN
    from llmfs.eval.execution import run_many
    tr, ho = TRAIN(), HELDOUT()
    res = run_many([t["reference"] + "\n" + t["test"] for t in tr[:80] + ho[:40]], workers=16)
    assert all(ok for ok, _ in res)
    assert {t["prompt"] for t in tr}.isdisjoint({t["prompt"] for t in ho})
