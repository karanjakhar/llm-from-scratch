"""Acceptance criteria for Stage 14 (docs/14_attention_variants.md §6)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from variants import (linear_attention_parallel, linear_attention_recurrent,  # noqa: E402
                      ssm_parallel, ssm_recurrent)


def test_linear_attention_two_forms_agree():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 40, 8, dtype=torch.float64) for _ in range(3))
    assert torch.allclose(linear_attention_parallel(q, k, v), linear_attention_recurrent(q, k, v),
                          atol=1e-10)


def test_linear_attention_is_causal():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 20, 4) for _ in range(3))
    a = linear_attention_parallel(q, k, v)
    v2 = v.clone()
    v2[0, 10] += 5
    b = linear_attention_parallel(q, k, v2)
    assert torch.allclose(a[0, :10], b[0, :10])


def test_ssm_two_forms_agree():
    torch.manual_seed(0)
    B, T, D = 2, 50, 6
    x = torch.randn(B, T, D, dtype=torch.float64)
    a = torch.rand(B, T, D, dtype=torch.float64) * 0.5 + 0.45       # keep 45-95% per step
    b, c = torch.randn(B, T, D, dtype=torch.float64), torch.randn(B, T, D, dtype=torch.float64)
    assert torch.allclose(ssm_recurrent(x, a, b, c), ssm_parallel(x, a, b, c), atol=1e-8)


def test_needle_case_layout():
    from needle import make_case

    class Tok:                                      # bytes as tokens, enough for layout
        def encode_ordinary(self, s):
            return list(s.encode())
    data = torch.randint(97, 123, (5000,)).numpy().astype("uint16")
    ids, ans = make_case(Tok(), data, 300, 0.5, random.Random(0))
    assert len(ids) == 300
    text = bytes(ids).decode()
    assert text.endswith('assert SECRET_KEY == "')
    key = bytes(ans).decode()
    assert f'SECRET_KEY = "{key}"' in text
