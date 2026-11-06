"""Acceptance criteria for Stage 06 (docs/06_gpt_model.md §7)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt import GPT, GPT2_SIZES, GPTConfig, count_params, count_params_by_hand, load_gpt2_state  # noqa: E402

CUDA = torch.cuda.is_available()
SHAPES = [
    dict(vocab_size=8196, block_size=256, n_layer=4, n_head=4, n_embd=256),
    dict(vocab_size=8196, block_size=512, n_layer=8, n_head=8, n_embd=512),
    dict(vocab_size=100, block_size=16, n_layer=1, n_head=1, n_embd=8),
    dict(vocab_size=8196, block_size=1024, n_layer=12, n_head=12, n_embd=768),
    dict(vocab_size=50257, block_size=1024, n_layer=12, n_head=12, n_embd=768),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tie", [True, False])
def test_hand_count_matches(shape, tie):
    cfg = GPTConfig(**shape, tie=tie)
    assert count_params_by_hand(cfg)["total"] == count_params(GPT(cfg))


def test_gpt2_small_is_124_439_808():
    cfg = GPTConfig(vocab_size=50257, block_size=1024, **GPT2_SIZES["gpt2"])
    parts = count_params_by_hand(cfg)
    assert parts["total"] == 124_439_808
    assert parts["blocks"] == 12 * 7_087_872
    assert parts["wte"] == 38_597_376


def test_initial_loss_is_ln_V():
    torch.manual_seed(0)
    cfg = GPTConfig()
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, 64))
    tgt = torch.randint(0, cfg.vocab_size, (8, 64))              # random, not idx (Stage 05 trap)
    _, loss = m(idx, tgt)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1


@pytest.mark.parametrize("scaled", [True, False])
def test_scaled_init_std(scaled):
    torch.manual_seed(0)
    cfg = GPTConfig(n_layer=8, n_embd=256, scaled_init=scaled)
    m = GPT(cfg)
    want = 0.02 / math.sqrt(2 * 8) if scaled else 0.02
    for blk in m.transformer.h:
        for w in (blk.attn.c_proj.weight, blk.mlp.c_proj.weight):
            assert abs(w.std().item() - want) / want < 0.05
        assert abs(blk.attn.c_attn.weight.std().item() - 0.02) < 0.001   # untouched


def test_tied_head_is_one_tensor():
    m = GPT(GPTConfig(tie=True))
    assert m.lm_head.weight is m.transformer.wte.weight
    m2 = GPT(GPTConfig(tie=False))
    assert m2.lm_head.weight is not m2.transformer.wte.weight


def test_flash_matches_manual():
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2))
    idx = torch.randint(0, 8196, (2, 50))
    a, _ = m(idx)
    for blk in m.transformer.h:
        blk.attn.flash = True
    b, _ = m(idx)
    assert (a - b).abs().max().item() < 1e-5


def test_causal():
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=2)).eval()
    idx = torch.randint(0, 8196, (1, 40))
    a, _ = m(idx)
    idx2 = idx.clone()
    idx2[0, 20] = (idx2[0, 20] + 1) % 8196
    b, _ = m(idx2)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-6)      # past unchanged
    assert not torch.allclose(a[0, 20], b[0, 20])               # the edited position moves


def test_loader_fails_on_missing_and_extra():
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=1, n_embd=8)
    src = GPT(cfg).state_dict()
    theirs = {k.removeprefix("transformer."): v.clone() for k, v in src.items()
              if k != "lm_head.weight"}
    for k in list(theirs):                                       # make it look like Conv1D
        if k.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            theirs[k] = theirs[k].T.contiguous()
    load_gpt2_state(GPT(cfg), dict(theirs))                      # complete: loads fine

    missing = dict(theirs)
    del missing["h.0.mlp.c_fc.bias"]
    with pytest.raises(KeyError):
        load_gpt2_state(GPT(cfg), missing)

    extra = dict(theirs)
    extra["h.1.ln_1.weight"] = torch.ones(8)
    with pytest.raises(KeyError):
        load_gpt2_state(GPT(cfg), extra)


def test_loader_round_trip_reproduces_model():
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=8)
    a = GPT(cfg).eval()
    theirs = {k.removeprefix("transformer."): v.clone() for k, v in a.state_dict().items()
              if k != "lm_head.weight"}
    for k in list(theirs):
        if k.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            theirs[k] = theirs[k].T.contiguous()
    b = GPT(cfg).eval()
    load_gpt2_state(b, theirs)
    idx = torch.randint(0, 50, (2, 8))
    assert torch.equal(a(idx)[0], b(idx)[0])


# --- the oracle: real GPT-2 weights vs Hugging Face ------------------------

def _hf_available():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache("openai-community/gpt2", "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _hf_available(), reason="GPT-2 weights not downloaded")
def test_gpt2_matches_huggingface():
    import tiktoken
    from transformers import GPT2LMHeadModel

    enc = tiktoken.get_encoding("gpt2")
    text = ("def fibonacci(n):\n    if n < 2:\n        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
            "The quick brown fox jumps over the lazy dog.")
    ids = torch.tensor([enc.encode(text)])
    x, y = ids[:, :-1], ids[:, 1:]

    ours = GPT.from_pretrained("gpt2").eval()
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager").eval()
    with torch.no_grad():
        a, la = ours(x, y)
        out = hf(x, labels=x)        # HF shifts internally; ours takes shifted targets
        b = out.logits
    assert (a - b).abs().max().item() < 1e-3
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    # HF's loss uses logits[:, :-1] vs x[:, 1:]; recompute ours the same way.
    lb = torch.nn.functional.cross_entropy(b[0, :-1], x[0, 1:])
    la2 = torch.nn.functional.cross_entropy(a[0, :-1], x[0, 1:])
    assert abs(la2.item() - lb.item()) < 1e-4
    assert abs(out.loss.item() - lb.item()) < 1e-5
