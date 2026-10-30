"""Acceptance criteria for Stage 04 (docs/04_attention.md §10).

    uv run pytest stages/04_attention -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attention import MultiHeadAttention, SingleHeadAttention, attention, causal_mask

B, H, T, d = 2, 4, 33, 16      # odd T catches off-by-one mask bugs
C = H * d


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def qkv(requires_grad=False):
    return [torch.randn(B, H, T, d, dtype=torch.float32, requires_grad=requires_grad)
            for _ in range(3)]


# --- the core against PyTorch -----------------------------------------------

@pytest.mark.parametrize("causal", [True, False])
def test_matches_sdpa_outputs_and_gradients(causal):
    q, k, v = qkv(requires_grad=True)
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    g = torch.randn(B, H, T, d)                                  # upstream gradient

    ours = attention(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(q2, k2, v2, is_causal=causal)
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-5)

    (ours * g).sum().backward()
    (ref * g).sum().backward()
    for a, b in ((q, q2), (k, k2), (v, v2)):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_multihead_matches_nn_multiheadattention(causal):
    ours = MultiHeadAttention(C, H, causal=causal)
    ref = nn.MultiheadAttention(C, H, batch_first=True)
    with torch.no_grad():
        # Both pack [W_q; W_k; W_v] as one (3C, C) matrix in the same order.
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.proj.weight)
        ref.out_proj.bias.copy_(ours.proj.bias)
    x = torch.randn(B, T, C)
    mask = ~causal_mask(T) if causal else None                   # PyTorch: True = NOT allowed
    expected, _ = ref(x, x, x, attn_mask=mask, need_weights=False)
    torch.testing.assert_close(ours(x), expected, rtol=1e-5, atol=1e-5)


def test_one_head_multihead_equals_single_head():
    mha = MultiHeadAttention(C, 1, bias=False)
    sha = SingleHeadAttention(C, C)
    with torch.no_grad():
        wq, wk, wv = mha.qkv.weight.split(C, dim=0)
        sha.q.weight.copy_(wq)
        sha.k.weight.copy_(wk)
        sha.v.weight.copy_(wv)
        mha.proj.weight.copy_(torch.eye(C))                      # identity: nothing to mix
    x = torch.randn(B, T, C)
    torch.testing.assert_close(mha(x), sha(x), rtol=1e-5, atol=1e-5)


def test_wrong_reshape_gives_right_shape_wrong_answer():
    """The silent bug from docs §5: view straight to (B, H, T, d) without the
    transpose keeps the shape and scrambles which numbers belong to which head."""
    x = torch.randn(B, T, C)
    mha = MultiHeadAttention(C, H)
    q, k, v = mha.qkv(x).split(C, dim=-1)
    wrong = [t.reshape(B, H, T, d) for t in (q, k, v)]           # no error, wrong grouping
    out_wrong = attention(*wrong).transpose(1, 2).reshape(B, T, C)
    out_wrong = mha.proj(out_wrong)
    assert out_wrong.shape == mha(x).shape
    assert not torch.allclose(out_wrong, mha(x), atol=1e-3)


# --- the mask ----------------------------------------------------------------

def test_causal_future_tokens_cannot_change_the_past():
    mha = MultiHeadAttention(C, H)
    x = torch.randn(1, T, C)
    t = 20
    x2 = x.clone()
    x2[:, t + 1:] = torch.randn_like(x2[:, t + 1:]) * 100        # rewrite the future, wildly
    out, out2 = mha(x), mha(x2)
    assert torch.equal(out[:, :t + 1], out2[:, :t + 1])          # past: bit-identical
    assert not torch.allclose(out[:, t + 1:], out2[:, t + 1:])


def test_rows_sum_to_one_and_masked_weights_are_exactly_zero():
    q, k, v = qkv()
    _, w = attention(q, k, v, causal=True, return_weights=True)  # (B, H, T, T)
    torch.testing.assert_close(w.sum(-1), torch.ones(B, H, T))
    future = ~causal_mask(T)
    assert (w[..., future] == 0).all()                            # exactly 0, not tiny
    assert (w[..., ~future] > 0).all()                            # soft: never exactly 0 where allowed


def test_fully_masked_row_gives_nan():
    """The padding trap from docs §4: a row with every key masked is 0/0."""
    scores = torch.full((1, 4), float("-inf"))
    assert torch.isnan(scores.softmax(-1)).all()


# --- why √d ------------------------------------------------------------------

@pytest.mark.parametrize("dim", [16, 256, 1024])
def test_score_variance_grows_with_d_and_scaling_fixes_it(dim):
    q = torch.randn(4096, dim)
    k = torch.randn(4096, dim)
    raw = (q * k).sum(-1)                                        # 4096 independent dot products
    assert abs(raw.var().item() / dim - 1) < 0.1                 # variance ≈ d
    assert abs((raw / dim ** 0.5).var().item() - 1) < 0.1        # scaled: ≈ 1
