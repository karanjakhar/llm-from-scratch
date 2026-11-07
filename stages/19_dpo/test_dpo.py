"""Acceptance criteria for Stage 19 (docs/19_rlhf_dpo.md §7)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.post.dpo import dpo_loss  # noqa: E402


def test_dpo_at_reference_is_ln2():
    z = torch.zeros(4)
    loss, rc, rr = dpo_loss(z, z, z, z, beta=0.1)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-6)


def test_dpo_gradient_direction():
    pi_c = torch.zeros(1, requires_grad=True)
    pi_r = torch.zeros(1, requires_grad=True)
    loss, _, _ = dpo_loss(pi_c, pi_r, torch.zeros(1), torch.zeros(1), beta=0.5)
    loss.backward()
    assert pi_c.grad.item() < 0 < pi_r.grad.item()      # raise chosen, lower rejected
    # magnitude: β·σ(-margin) = 0.5·0.5
    assert pi_c.grad.item() == pytest.approx(-0.25)


def test_dpo_only_relative_to_reference():
    """Shifting both policy and reference log-probs by the same amount changes nothing."""
    a = dpo_loss(torch.tensor([-5.0]), torch.tensor([-7.0]), torch.tensor([-6.0]), torch.tensor([-6.0]), 0.1)[0]
    b = dpo_loss(torch.tensor([-15.0]), torch.tensor([-17.0]), torch.tensor([-16.0]), torch.tensor([-16.0]), 0.1)[0]
    assert a.item() == pytest.approx(b.item())
