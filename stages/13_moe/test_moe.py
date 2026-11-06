"""Acceptance criteria for Stage 13 (docs/13_moe.md §8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.llama import MLP, Llama, LlamaConfig  # noqa: E402
from llmfs.model.moe import MoE  # noqa: E402

SMALL = dict(vocab_size=101, block_size=64, n_layer=2, n_head=4, n_embd=64)


def test_one_expert_top1_equals_dense_mlp():
    torch.manual_seed(0)
    cfg = LlamaConfig(**SMALL)
    dense = MLP(cfg)
    moe = MoE(64, n_experts=1, top_k=1, hidden=cfg.hidden)
    moe.experts[0].load_state_dict(dense.state_dict())
    x = torch.randn(3, 7, 64)
    assert torch.allclose(moe(x), dense(x), atol=1e-6)


def test_output_is_weighted_sum_of_chosen_experts():
    torch.manual_seed(0)
    m = MoE(16, n_experts=4, top_k=2, hidden=32)
    x = torch.randn(5, 16)
    probs = m.router(x).softmax(-1)
    w, idx = probs.topk(2, -1)
    w = w / w.sum(-1, keepdim=True)
    want = torch.stack([sum(w[i, j] * m.experts[idx[i, j]](x[i:i + 1])[0] for j in range(2))
                        for i in range(5)])
    assert torch.allclose(m(x), want, atol=1e-5)


def test_aux_loss_is_one_when_balanced_and_E_when_collapsed():
    m = MoE(8, n_experts=4, top_k=1, hidden=8)
    with torch.no_grad():
        m.router.weight.zero_()
    # uniform router: P_i = 1/4; topk ties go to expert 0, so f is collapsed: aux = 4·(1·1/4) = 1
    x = torch.randn(32, 8)
    m(x)
    assert m.aux_loss.item() == pytest.approx(1.0, abs=1e-5)
    # a router that puts all probability on expert 0: f_0 = 1, P_0 ≈ 1 -> aux ≈ E
    with torch.no_grad():
        m.router.weight[0].fill_(0.0)
        m.router.weight[0, 0] = 100.0
    x = torch.ones(32, 8)
    m(x)
    assert m.aux_loss.item() == pytest.approx(4.0, rel=1e-3)
    # perfectly balanced assignments with uniform-ish probs -> 1
    with torch.no_grad():
        m.router.weight.copy_(torch.eye(4, 8) * 5)
    x = torch.eye(8)[[0, 1, 2, 3] * 8]
    m(x)
    assert m.last_counts.tolist() == [8, 8, 8, 8]
    assert m.aux_loss.item() == pytest.approx(1.0, abs=1e-5)


def test_aux_loss_gradient_reaches_router():
    m = MoE(8, n_experts=4, top_k=2, hidden=8)
    m(torch.randn(16, 8))
    m.aux_loss.backward()
    assert m.router.weight.grad.abs().sum() > 0


def test_capacity_drops_overflow():
    m = MoE(8, n_experts=4, top_k=1, hidden=8, capacity_factor=1.0)
    with torch.no_grad():
        m.router.weight.zero_()
        m.router.weight[0, 0] = 100.0
    x = torch.ones(32, 8)
    m(x)                                     # all 32 want expert 0; capacity = 32·1/4 = 8
    assert m.last_dropped == pytest.approx(24 / 32)


def test_llama_moe_adds_losses_and_counts_active_params():
    torch.manual_seed(0)
    cfg = LlamaConfig(**SMALL, n_experts=8, top_k=2)
    m = Llama(cfg)
    dense = Llama(LlamaConfig(**SMALL))
    ids = torch.randint(0, 101, (2, 16))
    _, loss = m(ids, ids)
    assert loss.item() == pytest.approx(
        (m.last_lm_loss + cfg.aux_coef * m.last_aux + cfg.z_coef * m.last_z).item(), rel=1e-6)
    # experts have hidden/2 each, top-2 active: active MLP size == dense MLP size
    assert abs(m.active_params() - dense.num_params()) / dense.num_params() < 0.01
    assert m.num_params() > 3 * dense.num_params() - 2 * 101 * 64 * 3


def test_shared_expert_always_on():
    torch.manual_seed(0)
    m = MoE(8, n_experts=2, top_k=1, hidden=8, n_shared=1)
    x = torch.randn(4, 8)
    with torch.no_grad():
        for e in m.experts:
            e.down_proj.weight.zero_()
    assert torch.allclose(m(x), m.shared[0](x), atol=1e-6)
