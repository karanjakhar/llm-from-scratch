"""Acceptance tests for course module 0. The numbers are the book's (docs/00_setup.md)."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget import MB, linear_params, max_params, measure_training_bytes, training_bytes  # noqa: E402


def test_linear_params_matches_pytorch():
    for d_in, d_out, bias in ((1024, 1024, False), (768, 3072, True), (5, 3, True)):
        layer = nn.Linear(d_in, d_out, bias=bias)
        assert linear_params(d_in, d_out, bias) == sum(p.numel() for p in layer.parameters())


def test_book_toy_model_is_192_mb():
    # stages/00_setup/verify_env.py: 12 bias-free 1024x1024 layers
    n = 12 * linear_params(1024, 1024, bias=False)
    assert n == 12_582_912
    assert training_bytes(n) / MB == 192.0


def test_optimizers_cost_different_amounts():
    assert training_bytes(1, "sgd") == 8
    assert training_bytes(1, "momentum") == 12
    assert training_bytes(1, "adamw") == 16
    assert training_bytes(1, "adamw", bytes_per_value=2) == 8      # a 16-bit everything run


def test_max_params_on_the_book_gpu():
    # 7,797 MB addressable on the "8 GB" RTX 4060 Laptop (journal, 2026-10-14)
    n = max_params(7797 * MB)
    assert training_bytes(n) <= (7797 - 600) * MB < training_bytes(n + 1)
    assert 471e6 < n < 472e6                    # fixed cost alone; activations make the real ceiling lower
    assert max_params(500 * MB) == 0            # less than the reserve: nothing fits


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_measurement_matches_the_rule_within_20_percent():
    model = nn.Sequential(*[nn.Linear(1024, 1024, bias=False) for _ in range(12)])
    m = measure_training_bytes(model, torch.randn(8, 1024, device="cuda"))
    predicted = training_bytes(12_582_912)
    assert m["weights"] == 12_582_912 * 4
    assert m["optimizer"] == 12_582_912 * 8     # Adam's two states, created on the first step
    assert abs(m["total"] - predicted) / predicted < 0.2
