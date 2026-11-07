"""Acceptance criteria for Stage 21 (docs/21_distill_quantize.md §8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.post.quant import QuantLinear, fake_quant, model_bytes, quantize, quantize_model  # noqa: E402


def test_worked_example_int8_symmetric():
    w = torch.tensor([[0.5, -1.27, 0.0, 1.27]])
    q, s, z = quantize(w, bits=8, granularity="tensor")
    assert s.item() == pytest.approx(0.01)
    assert q.tolist() == [[50, -127, 0, 127]]


def test_asymmetric_maps_range_exactly():
    w = torch.tensor([[0.0, 1.0, 2.0, 2.55]])
    q, s, z = quantize(w, bits=8, granularity="tensor", symmetric=False)
    assert q.min().item() == 0 and q.max().item() == 255
    assert torch.allclose(fake_quant(w, bits=8, granularity="tensor", symmetric=False), w, atol=s.item())


@pytest.mark.parametrize("bits,gran", [(8, "tensor"), (8, "channel"), (4, "channel"), (4, "group")])
def test_error_within_half_step(bits, gran):
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q, s, z = quantize(w, bits=bits, granularity=gran, group=64)
    back = fake_quant(w, bits=bits, granularity=gran, group=64)
    step = s.max().item()
    assert (back - w).abs().max().item() <= step / 2 + 1e-6


def test_outlier_hurts_per_tensor_not_per_group():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w[3, 7] = 2.0                                               # one outlier, 100x typical
    e_t = (fake_quant(w, bits=4, granularity="tensor") - w).pow(2).mean()
    e_g = (fake_quant(w, bits=4, granularity="group", group=64) - w).pow(2).mean()
    assert e_t > 50 * e_g


@pytest.mark.parametrize("bits", [8, 4])
def test_quantlinear_matches_fake_quant_and_is_small(bits):
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64)
    ql = QuantLinear(lin, bits=bits, granularity="group", group=32)
    x = torch.randn(3, 128)
    want = torch.nn.functional.linear(x, fake_quant(lin.weight, bits=bits, granularity="group", group=32), lin.bias)
    assert torch.allclose(ql(x), want, atol=1e-2)
    codes_bytes = 64 * 128 * bits // 8
    assert ql.codes.numel() * ql.codes.element_size() == codes_bytes


def test_quantize_model_runs_and_shrinks():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)).eval()
    before = model_bytes(m)
    x = torch.randint(0, 100, (2, 16))
    a, _ = m(x)
    quantize_model(m, bits=8, granularity="channel")
    b, _ = m(x)
    assert (a - b).abs().max() < 0.05
    assert model_bytes(m) < before
