"""Acceptance criteria for Stage 03 (docs/03_embeddings_position.md §9).

    uv run pytest stages/03_embeddings -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import (
    LearnedPosition,
    OutputHead,
    SinusoidalPosition,
    TokenEmbedding,
    attention_no_mask,
    count_params,
    embed_onehot,
    interleaved_to_half,
    rope_angles,
    rope_complex,
    rope_half,
    rope_interleaved,
    sinusoidal_table,
)

V, C = 8196, 64          # our real vocab size; a small width keeps tests fast


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# --- token embedding --------------------------------------------------------

def test_indexing_equals_onehot_matmul_equals_nn_embedding():
    emb = TokenEmbedding(V, C)
    ref = nn.Embedding(V, C)
    ref.weight.data.copy_(emb.weight.data)
    ids = torch.randint(0, V, (2, 7))                            # (B, T)

    a = emb(ids)                                                 # (B, T, C)
    b = embed_onehot(ids, emb.weight)
    c = ref(ids)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)


def test_gradients_identical_and_only_used_rows_get_gradient():
    ids = torch.tensor([[5, 9, 5, 100]])                          # token 5 appears twice
    w1 = (torch.randn(V, C) * 0.02).requires_grad_()
    w2 = w1.detach().clone().requires_grad_()
    g = torch.randn(1, 4, C)                                     # an arbitrary upstream gradient

    w1[ids].backward(g)                                          # indexing
    embed_onehot(ids, w2).backward(g)                            # one-hot matmul
    torch.testing.assert_close(w1.grad, w2.grad)

    used = torch.zeros(V, dtype=torch.bool)
    used[[5, 9, 100]] = True
    assert w1.grad[~used].abs().sum() == 0                       # unused rows: exactly zero
    assert (w1.grad[used].abs().sum(dim=1) > 0).all()
    # A repeated token accumulates: row 5's gradient is the sum of both uses.
    torch.testing.assert_close(w1.grad[5], g[0, 0] + g[0, 2])


# --- why position is needed -------------------------------------------------

def test_attention_without_position_is_permutation_equivariant():
    x = torch.randn(1, 10, C)                                    # (B, T, C)
    perm = torch.randperm(10)
    out = attention_no_mask(x)
    out_shuffled = attention_no_mask(x[:, perm])
    # Shuffle in -> the same outputs, shuffled the same way. Order is invisible.
    torch.testing.assert_close(out_shuffled, out[:, perm], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scheme", ["learned", "sinusoidal"])
def test_added_position_breaks_equivariance(scheme):
    T = 10
    x = torch.randn(1, T, C)
    pos = LearnedPosition(T, C)(T) if scheme == "learned" else sinusoidal_table(T, C)
    perm = torch.randperm(T)
    out = attention_no_mask(x + pos)
    out_shuffled = attention_no_mask(x[:, perm] + pos)           # position stays with the slot
    assert not torch.allclose(out_shuffled, out[:, perm], atol=1e-3)


# --- learned positions -------------------------------------------------------

def test_learned_position_has_a_hard_wall():
    pos = LearnedPosition(256, C)
    assert pos(256).shape == (256, C)
    with pytest.raises(ValueError):
        pos(257)


# --- sinusoidal ------------------------------------------------------------

def test_sinusoidal_bounded_and_unique():
    pe = sinusoidal_table(2048, C)                               # (T, C)
    assert pe.abs().max() <= 1.0
    # No two positions share a vector: smallest pairwise distance is well above 0.
    d = torch.cdist(pe[:512], pe[:512])
    d.fill_diagonal_(float("inf"))
    assert d.min() > 0.1


def test_sinusoidal_similarity_depends_only_on_offset():
    pe = sinusoidal_table(300, C).double()
    sim = pe @ pe.T                                              # (T, T)
    for k in (0, 1, 7, 50):
        diag = torch.diagonal(sim, offset=k)                     # all pairs with q - p = k
        assert (diag - diag[0]).abs().max() < 1e-4


def test_sinusoidal_shift_is_a_fixed_rotation():
    """PE[p + k] = R_k PE[p], with R_k the same for every p (docs §5)."""
    k, T = 13, 200
    pe = sinusoidal_table(T + k, C).double()
    from embeddings import frequencies
    a = k * frequencies(C)                                       # (C/2,) rotation angle per pair
    s, c = pe[:T, 0::2], pe[:T, 1::2]                            # sin, cos of p*w
    # sin(x + a) = sin x cos a + cos x sin a ; cos(x + a) = cos x cos a - sin x sin a
    s_k = s * a.cos() + c * a.sin()
    c_k = c * a.cos() - s * a.sin()
    torch.testing.assert_close(s_k, pe[k:, 0::2])
    torch.testing.assert_close(c_k, pe[k:, 1::2])


def test_sinusoidal_module_any_length():
    assert SinusoidalPosition(C)(10_000).shape == (10_000, C)    # no wall


# --- RoPE -----------------------------------------------------------------

@pytest.mark.parametrize("rope", [rope_interleaved, rope_half, rope_complex])
def test_rope_preserves_length(rope):
    x = torch.randn(2, 4, 50, C)                                 # (B, H, T, d)
    y = rope(x, rope_angles(50, C))
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


@pytest.mark.parametrize("rope", [rope_interleaved, rope_half])
def test_rope_score_depends_only_on_offset(rope):
    """Place the same q and k at every (m, n). Score must equal score at (m-n, 0)."""
    T = 128
    q = torch.randn(C).double()
    k = torch.randn(C).double()
    # float64 angles: with float32 ones the diagonals drift by ~1e-5 because
    # the angle itself (= position, in radians) is rounded. See rope_angles.
    ang = rope_angles(T, C, dtype=torch.float64)                 # (T, d/2)
    qs = rope(q.expand(T, C), ang)                               # (T, d): q at every position
    ks = rope(k.expand(T, C), ang)
    scores = qs @ ks.T                                           # (T, T): scores[m, n]
    for off in (0, 1, 5, 40):
        diag = torch.diagonal(scores, offset=-off)               # all (m, n) with m - n = off
        assert (diag - diag[0]).abs().max() < 1e-5
    # and it is not trivially constant: different offsets give different scores
    assert (scores[10, 0] - scores[10, 9]).abs() > 1e-3


def test_rope_position_zero_is_identity():
    x = torch.randn(3, 1, C)
    torch.testing.assert_close(rope_interleaved(x, rope_angles(1, C)), x)


def test_rope_interleaved_equals_complex():
    x = torch.randn(2, 4, 64, C)
    ang = rope_angles(64, C)
    torch.testing.assert_close(rope_interleaved(x, ang), rope_complex(x, ang))


def test_rope_layouts_differ_only_by_dimension_shuffle():
    x = torch.randn(2, 4, 64, C)
    ang = rope_angles(64, C)
    lhs = rope_half(interleaved_to_half(x), ang)
    rhs = interleaved_to_half(rope_interleaved(x, ang))
    torch.testing.assert_close(lhs, rhs)
    # ...and without the shuffle they genuinely disagree: the silent-bug trap.
    assert not torch.allclose(rope_half(x, ang), rope_interleaved(x, ang), atol=1e-3)


def test_rope_half_matches_huggingface_llama():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    B, H, T, d = 2, 4, 64, 64
    cfg = LlamaConfig(hidden_size=H * d, num_attention_heads=H, max_position_embeddings=T)
    hf_rope = LlamaRotaryEmbedding(cfg)
    q = torch.randn(B, H, T, d)
    k = torch.randn(B, H, T, d)
    pos_ids = torch.arange(T)[None].expand(B, T)                 # (B, T)
    cos, sin = hf_rope(q, pos_ids)                               # (B, T, d) each
    q_hf, k_hf = apply_rotary_pos_emb(q, k, cos, sin)

    ang = rope_angles(T, d)                                      # base 10000, same as Llama default
    torch.testing.assert_close(rope_half(q, ang), q_hf, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rope_half(k, ang), k_hf, rtol=1e-5, atol=1e-5)


def test_rope_breaks_equivariance_without_adding_anything():
    """RoPE adds nothing to x, yet order becomes visible to attention."""
    T = 10
    x = torch.randn(1, T, C)
    perm = torch.randperm(T)
    ang = rope_angles(T, C)

    def attn(z):
        q = k = rope_interleaved(z, ang)                         # rotate q, k by slot
        w = (q @ k.transpose(-2, -1) / C ** 0.5).softmax(-1)     # (B, T, T)
        return w @ z                                             # v is NOT rotated

    assert not torch.allclose(attn(x[:, perm]), attn(x)[:, perm], atol=1e-3)


# --- weight tying -----------------------------------------------------------

def test_tied_head_shares_storage_and_saves_exactly_V_times_C():
    emb = TokenEmbedding(V, C)
    tied = nn.ModuleDict({"emb": emb, "head": OutputHead(C, V, tie_to=emb)})
    emb2 = TokenEmbedding(V, C)
    untied = nn.ModuleDict({"emb": emb2, "head": OutputHead(C, V)})

    assert tied["head"].weight is tied["emb"].weight
    assert tied["head"].weight.data_ptr() == tied["emb"].weight.data_ptr()
    assert count_params(untied) - count_params(tied) == V * C


def test_tied_gradient_is_sum_of_both_uses():
    emb = TokenEmbedding(50, 8)
    head = OutputHead(8, 50, tie_to=emb)
    ids = torch.tensor([[1, 2, 3]])
    logits = head(emb(ids))                                      # (1, 3, 50)
    logits.logsumexp(-1).sum().backward()

    # Recompute with two separate copies; their grads must add up to the tied grad.
    w_in = emb.weight.detach().clone().requires_grad_()
    w_out = emb.weight.detach().clone().requires_grad_()
    (w_in[ids] @ w_out.T).logsumexp(-1).sum().backward()
    torch.testing.assert_close(emb.weight.grad, w_in.grad + w_out.grad)
    # Every output row got gradient (softmax touches all V scores),
    # even rows whose token never appeared as input.
    assert (w_out.grad.abs().sum(dim=1) > 0).all()
