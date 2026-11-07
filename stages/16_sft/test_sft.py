"""Acceptance criteria for Stage 16 (docs/16_sft.md §7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sft import IGNORE, collate, render, split_function  # noqa: E402

from llmfs.tokenizer import Tokenizer  # noqa: E402

TOK = Path(__file__).resolve().parents[2] / "data" / "tokenizers" / "split_8192.json"


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.load(str(TOK))


def test_loss_mask_covers_exactly_the_reply(tok):
    ids, labels = render(tok, "def f(x):\n    \"\"\"Double x.\"\"\"\n", "def f(x):\n    return 2 * x\n")
    kept = [t for t, l in zip(ids, labels) if l != IGNORE]
    E = tok.special_tokens["<|im_end|>"]
    assert kept[-1] == E                                       # the stop token is trained
    assert tok.decode(kept[:-1]) == "def f(x):\n    return 2 * x\n"
    masked = [t for t, l in zip(ids, labels) if l == IGNORE]
    assert "Double x." in tok.decode([t for t in masked if t < 8192])
    assert labels[:len(masked)] == [IGNORE] * len(masked)      # prompt first, then reply


def test_collate_shift_and_padding(tok):
    a = dict(zip(("ids", "labels"), render(tok, "u", "aaa")))
    b = dict(zip(("ids", "labels"), render(tok, "u", "a much longer reply here")))
    x, y = collate([a, b], pad_id=tok.special_tokens["<|pad|>"], device="cpu")
    assert x.shape == y.shape
    n = len(a["ids"])
    assert (x[0, n:] == tok.special_tokens["<|pad|>"]).all()    # right padding
    assert (y[0, n - 1:] == IGNORE).all()                      # pads are never targets
    assert y[1, -1].item() == tok.special_tokens["<|im_end|>"]
    # shift: y[t] is the token at x position t+1
    assert torch.equal(y[1][y[1] != IGNORE], torch.tensor(b["labels"][1:])[torch.tensor(b["labels"][1:]) != IGNORE])


def test_prompt_loss_ablation_trains_on_everything(tok):
    a = dict(zip(("ids", "labels"), render(tok, "user text", "reply")))
    _, y = collate([a], 0, device="cpu", include_prompt_loss=True)
    assert (y != IGNORE).sum().item() == len(a["ids"]) - 1


def test_right_padding_does_not_change_real_logits():
    """docs §4: with right padding, pads cannot affect earlier positions."""
    from llmfs.model.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=16)).eval()
    x = torch.randint(0, 49, (1, 10))
    padded = torch.cat([x, torch.full((1, 6), 49)], 1)
    a, _ = m(x)
    b, _ = m(padded)
    assert torch.allclose(a, b[:, :10], atol=1e-6)


def test_split_function_keeps_free_functions_only():
    good = 'def add(a, b):\n    """Add."""\n    return a + b\n'
    h, f = split_function(good)
    assert h == 'def add(a, b):\n    """Add."""\n' and f == good
    assert split_function('def m(self):\n    """x"""\n    return 1\n') is None
    assert split_function('def nodoc(a):\n    return a\n') is None
    assert split_function('def onlydoc(a):\n    """x"""\n') is None
