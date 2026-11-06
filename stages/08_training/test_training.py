"""Acceptance criteria for Stage 08 (docs/08_training_loop.md)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from optim import SGD, AdamW, clip_grad_norm, lr_schedule  # noqa: E402

from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402


def _tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, flash=False))


def _groups(m):
    return [{"params": [p for p in m.parameters() if p.dim() >= 2], "weight_decay": 0.1},
            {"params": [p for p in m.parameters() if p.dim() < 2], "weight_decay": 0.0}]


def test_adamw_matches_torch():
    # float64: in float32 the two drift to 2e-4 after 25 steps. Not a bug:
    # the same sums in a different order round differently, and training
    # amplifies it. In float64 the gap is 4e-13 (journal, Stage 08).
    a, b = _tiny().double(), _tiny().double()
    ours = AdamW(_groups(a), lr=1e-2, betas=(0.9, 0.95))
    ref = torch.optim.AdamW(_groups(b), lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
    torch.manual_seed(1)
    for _ in range(25):
        x = torch.randint(0, 64, (4, 16))
        y = torch.randint(0, 64, (4, 16))
        for m, opt in ((a, ours), (b, ref)):
            _, loss = m(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert (pa - pb).abs().max() < 1e-10


def test_l2_in_loss_differs_from_decoupled():
    a, b = _tiny(), _tiny()
    dec = AdamW(_groups(a), lr=1e-2)
    l2 = AdamW(_groups(b), lr=1e-2, l2_in_loss=True)
    torch.manual_seed(1)
    for _ in range(10):
        x = torch.randint(0, 64, (4, 16))
        for m, opt in ((a, dec), (b, l2)):
            _, loss = m(x, x.roll(-1, 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    diff = max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))
    assert diff > 1e-4


def test_first_adam_step_moves_every_weight_by_lr():
    """docs §2c: after bias correction, step 1 is ±lr whatever the gradient size."""
    p = torch.nn.Parameter(torch.zeros(5))
    opt = AdamW([p], lr=0.01)
    p.grad = torch.tensor([1e-6, 1e-3, 1.0, -50.0, 3.0])
    opt.step()
    assert torch.allclose(p.detach().abs(), torch.full((5,), 0.01), rtol=1e-2)


def test_sgd_step():
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([1.0, 2.0, 3.0])
    SGD([p], lr=0.1).step()
    assert torch.allclose(p.detach(), torch.tensor([0.9, 0.8, 0.7]))


def test_schedule_shape():
    peak, total, warm = 1e-3, 1000, 100
    lrs = [lr_schedule(s, total, peak, warm) for s in range(total)]
    assert lrs[0] == pytest.approx(peak / warm)
    assert lrs[warm - 1] == pytest.approx(peak)
    assert max(lrs) == pytest.approx(peak)
    assert lrs[-1] == pytest.approx(0.1 * peak, rel=1e-3)
    assert all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:]))     # monotone decay


def test_clip_caps_norm_keeps_direction():
    p = torch.nn.Parameter(torch.zeros(2))
    q = torch.nn.Parameter(torch.zeros(1))
    p.grad, q.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])        # global norm 5
    before = clip_grad_norm([p, q], 1.0)
    assert before == pytest.approx(5.0)
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.0]), atol=1e-5)
    assert torch.allclose(q.grad, torch.tensor([0.8]), atol=1e-5)


def test_clip_matches_torch():
    a, b = _tiny(), _tiny()
    x = torch.randint(0, 64, (4, 16))
    for m in (a, b):
        _, loss = m(x, x)
        (loss * 100).backward()
    n1 = clip_grad_norm(list(a.parameters()), 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0).item()
    assert n1 == pytest.approx(n2, rel=1e-5)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_accumulation_equals_big_batch():
    a, b = _tiny(), _tiny()
    torch.manual_seed(3)
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))
    _, loss = a(x, y)
    loss.backward()
    for i in range(4):                                           # 4 micro-batches of 2
        _, l = b(x[2 * i:2 * i + 2], y[2 * i:2 * i + 2])
        (l / 4).backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6)


def test_accumulation_without_divide_is_4x():
    a, b = _tiny(), _tiny()
    x = torch.randint(0, 64, (8, 16))
    _, loss = a(x, x)
    loss.backward()
    for i in range(4):
        _, l = b(x[2 * i:2 * i + 2], x[2 * i:2 * i + 2])
        l.backward()                                             # the trap in docs §6
    ratio = b.lm_head.weight.grad.norm() / a.lm_head.weight.grad.norm()
    assert ratio.item() == pytest.approx(4.0, rel=1e-3)


def test_overfit_one_batch():
    """docs §7 check 2: memorise 4 sequences of 16."""
    m = _tiny()
    opt = AdamW(_groups(m), lr=3e-3)
    torch.manual_seed(0)
    x = torch.randint(0, 64, (4, 17))
    first = None
    for step in range(300):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert first == pytest.approx(math.log(64), abs=0.3)
    assert loss.item() < 0.05
