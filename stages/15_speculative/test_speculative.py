"""Acceptance criteria for Stage 15 (docs/15_speculative.md §5)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample import generate_cached  # noqa: E402
from speculative import speculative_generate  # noqa: E402

from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402


def _pair(V=50):
    torch.manual_seed(0)
    target = GPT(GPTConfig(vocab_size=V, block_size=128, n_layer=3, n_head=2, n_embd=32)).eval()
    torch.manual_seed(1)
    draft = GPT(GPTConfig(vocab_size=V, block_size=128, n_layer=1, n_head=1, n_embd=16)).eval()
    # make them somewhat similar: draft shares the target's embedding table scaled down
    return target, draft


def test_greedy_equals_target_greedy():
    target, draft = _pair()
    ids = torch.randint(0, 50, (1, 6))
    want = generate_cached(target, ids, 40, temperature=0.0)
    for gamma in (1, 3, 6):
        got, st = speculative_generate(target, draft, ids, 40, gamma=gamma, temperature=0.0)
        assert torch.equal(got, want), gamma


def test_draft_equals_target_accepts_everything():
    target, _ = _pair()
    ids = torch.randint(0, 50, (1, 6))
    got, st = speculative_generate(target, target, ids, 30, gamma=4, temperature=1.0, seed=0)
    assert st["acceptance_rate"] == 1.0
    assert st["tokens_per_target_call"] > 4.5          # γ accepted + 1 bonus each round


def test_sampled_distribution_is_the_targets():
    """The first two generated tokens' joint distribution must be the target's."""
    V = 6
    target, draft = _pair(V)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        p1 = target(ids)[0][0, -1].softmax(-1)                          # (V,)
        exact = torch.zeros(V, V)
        for a in range(V):
            p2 = target(torch.cat([ids, torch.tensor([[a]])], 1))[0][0, -1].softmax(-1)
            exact[a] = p1[a] * p2
    n = 6000
    counts = torch.zeros(V, V)
    for s in range(n):
        out, _ = speculative_generate(target, draft, ids, 2, gamma=2, temperature=1.0, seed=s)
        counts[out[0, 3], out[0, 4]] += 1
    tv = 0.5 * (counts / n - exact).abs().sum().item()
    # 36 cells, 6000 draws: sampling noise alone gives TV ≈ 0.03
    assert tv < 0.05, tv
