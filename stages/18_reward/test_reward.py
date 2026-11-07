"""Acceptance criteria for Stage 18 (docs/18_reward_model.md §7)."""

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reward import RewardModel, batch_ids, bt_loss  # noqa: E402

from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.post.prefs import mutate  # noqa: E402

SRC = '''def clamp(x, lo, hi):
    """Clamp x to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
'''


def test_mutation_changes_behaviour_and_keeps_format():
    rng = random.Random(0)
    seen = set()
    for _ in range(30):
        good, bad = mutate(SRC, rng)
        assert good == ast.unparse(ast.parse(SRC))           # both sides canonicalised
        assert bad != good
        ns_g, ns_b = {}, {}
        exec(good, ns_g)
        exec(bad, ns_b)
        outs = [(ns_g["clamp"](x, 0, 10), ns_b["clamp"](x, 0, 10)) for x in (-5, 0, 5, 10, 15)]
        seen.add(any(a != b for a, b in outs))
    assert True in seen


def test_bt_loss_values():
    assert bt_loss(torch.tensor([0.0]), torch.tensor([0.0])).item() == pytest.approx(0.6931, abs=1e-4)
    assert bt_loss(torch.tensor([10.0]), torch.tensor([0.0])).item() < 1e-4
    # symmetric: P(chosen > rejected) = σ(rc - rr)
    assert bt_loss(torch.tensor([1.0]), torch.tensor([3.0])).item() == pytest.approx(
        -torch.nn.functional.logsigmoid(torch.tensor(-2.0)).item(), abs=1e-6)


def test_reward_reads_last_real_token():
    torch.manual_seed(0)
    rm = RewardModel(GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=1, n_head=1, n_embd=16))).eval()
    a = [1, 2, 3, 4]
    x, l = batch_ids([a, a + [5, 6]], pad=0, device="cpu")
    x2, l2 = batch_ids([a], pad=0, device="cpu")
    r = rm(x, l)
    assert torch.allclose(r[0], rm(x2, l2)[0], atol=1e-6)     # padding does not change the score
