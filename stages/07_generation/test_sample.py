"""Acceptance criteria for Stage 07 (docs/07_generation.md §7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample import (apply_repetition_penalty, apply_temperature, generate_cached,  # noqa: E402
                    generate_naive, sample_next, top_k_filter, top_p_filter)

from llmfs.model.gpt import GPT, GPTConfig, KVCache  # noqa: E402


@pytest.mark.parametrize("tau,want", [
    (0.5, [0.867, 0.117, 0.016]),
    (1.0, [0.665, 0.245, 0.090]),
    (2.0, [0.506, 0.307, 0.186]),
    (0.1, [1.000, 0.000, 0.000]),
])
def test_temperature_table(tau, want):
    p = apply_temperature(torch.tensor([[2.0, 1.0, 0.0]]), tau).softmax(-1)[0]
    assert torch.allclose(p, torch.tensor(want), atol=1e-3)


def test_top_k_keeps_k():
    logits = torch.randn(4, 100)
    out = top_k_filter(logits, 7)
    assert (out > float("-inf")).sum(-1).tolist() == [7] * 4
    assert torch.equal(out.argmax(-1), logits.argmax(-1))


def test_top_p_keeps_crossing_token():
    probs = torch.tensor([[0.62, 0.21, 0.09, 0.04, 0.02, 0.02]])
    out = top_p_filter(probs.log(), 0.9)
    assert (out > float("-inf")).tolist() == [[True, True, True, False, False, False]]


def test_top_p_single_dominant_token_survives():
    probs = torch.tensor([[0.95, 0.03, 0.02]])
    out = top_p_filter(probs.log(), 0.9)
    assert (out > float("-inf")).tolist() == [[True, False, False]]


def test_top_p_unsorted_input():
    probs = torch.tensor([[0.02, 0.62, 0.04, 0.21, 0.09, 0.02]])     # same as above, shuffled
    out = top_p_filter(probs.log(), 0.9)
    assert (out > float("-inf")).tolist() == [[False, True, False, True, True, False]]


def test_repetition_penalty_moves_both_signs_down():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1, 1]]), 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0]]


def test_greedy_is_argmax_and_temperature_zero():
    logits = torch.randn(3, 50)
    assert torch.equal(sample_next(logits, temperature=0.0)[:, 0], logits.argmax(-1))


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = GPTConfig(vocab_size=97, block_size=64, n_layer=3, n_head=4, n_embd=32, **kw)
    return GPT(cfg).eval()


@pytest.mark.parametrize("flash", [True, False])
def test_cached_logits_match_full_recompute(flash):
    m = _model(flash=flash)
    ids = torch.randint(0, 97, (2, 30))
    full, _ = m(ids)                                             # (2, 30, V)
    cache = KVCache(m.cfg, 2, 64, "cpu", torch.float32)
    prefill, _ = m(ids[:, :10], cache=cache)                     # 10 tokens at once
    assert (prefill - full[:, :10]).abs().max() < 1e-4
    for t in range(10, 30):
        step, _ = m(ids[:, t:t + 1], cache=cache)
        assert (step[:, 0] - full[:, t]).abs().max() < 1e-4, t
    assert cache.pos == 30


def test_multi_token_chunks_after_cache():
    """A second multi-token chunk must see the old keys plus its own causal prefix."""
    m = _model()
    ids = torch.randint(0, 97, (1, 24))
    full, _ = m(ids)
    cache = KVCache(m.cfg, 1, 64, "cpu", torch.float32)
    m(ids[:, :8], cache=cache)
    chunk, _ = m(ids[:, 8:16], cache=cache)
    assert (chunk - full[:, 8:16]).abs().max() < 1e-4


def test_greedy_naive_equals_cached():
    m = _model()
    ids = torch.randint(0, 97, (2, 5))
    a = generate_naive(m, ids, 40, temperature=0.0)
    b = generate_cached(m, ids, 40, temperature=0.0)
    assert torch.equal(a, b)


def test_sampled_naive_equals_cached_at_fixed_seed():
    m = _model()
    ids = torch.randint(0, 97, (1, 5))
    knobs = dict(temperature=0.9, top_k=50, top_p=0.95, repetition_penalty=1.1)
    a = generate_naive(m, ids, 40, seed=123, **knobs)
    b = generate_cached(m, ids, 40, seed=123, **knobs)
    assert torch.equal(a, b)


def test_stops_on_eot():
    m = _model()
    with torch.no_grad():                                        # force token 7 to win always
        m.lm_head.weight.zero_()
        m.lm_head.weight[7] = 1.0
        m.transformer.ln_f.weight.fill_(0.0)
        m.transformer.ln_f.bias.fill_(1.0)
    ids = torch.randint(0, 97, (1, 5))
    for gen in (generate_naive, generate_cached):
        out = gen(m, ids, 20, eot=7, temperature=0.0)
        assert out.shape[1] == 6 and out[0, -1] == 7
