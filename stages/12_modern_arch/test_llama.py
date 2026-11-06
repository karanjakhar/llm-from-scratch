"""Acceptance criteria for Stage 12 (docs/12_modern_arch.md §8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.llama import KVCache, Llama, LlamaConfig, kv_bytes_per_token  # noqa: E402

SMALL = dict(vocab_size=101, block_size=64, n_layer=2, n_head=4, n_embd=64)


@pytest.mark.parametrize("n_kv", [4, 2, 1])
def test_matches_hf_llama(n_kv):
    from transformers import LlamaConfig as HFConfig
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(**SMALL, n_kv_head=n_kv, tie=False, mlp_hidden=172, norm_eps=1e-5)
    ours = Llama(cfg).eval()
    hf = LlamaForCausalLM(HFConfig(
        vocab_size=101, hidden_size=64, intermediate_size=172, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=n_kv, max_position_embeddings=64,
        rms_norm_eps=1e-5, rope_theta=10000.0, tie_word_embeddings=False,
        attention_bias=False, mlp_bias=False)).eval()
    sd = {k.removeprefix("model."): v for k, v in hf.state_dict().items()}
    missing = ours.load_state_dict(sd, strict=False)
    assert not missing.missing_keys, missing.missing_keys
    assert not [k for k in missing.unexpected_keys if "rotary" not in k], missing.unexpected_keys
    ids = torch.randint(0, 101, (2, 40))
    with torch.no_grad():
        a, _ = ours(ids)
        b = hf(ids).logits
    assert (a - b).abs().max() < 1e-4


@pytest.mark.parametrize("kw", [dict(), dict(n_kv_head=2), dict(n_kv_head=1), dict(qk_norm=True),
                                dict(window=8), dict(pos="learned", norm="layer", mlp="gelu", bias=True)])
def test_cache_matches_full(kw):
    torch.manual_seed(0)
    m = Llama(LlamaConfig(**SMALL, **kw)).eval()
    ids = torch.randint(0, 101, (2, 30))
    full, _ = m(ids)
    cache = KVCache(m.cfg, 2, 64, "cpu", torch.float32)
    out, _ = m(ids[:, :12], cache=cache)
    assert (out - full[:, :12]).abs().max() < 1e-4
    out, _ = m(ids[:, 12:20], cache=cache)                      # multi-token chunk after a cache
    assert (out - full[:, 12:20]).abs().max() < 1e-4
    for t in range(20, 30):
        out, _ = m(ids[:, t:t + 1], cache=cache)
        assert (out[:, 0] - full[:, t]).abs().max() < 1e-4


def test_sliding_window_cannot_see_far_tokens():
    torch.manual_seed(0)
    m = Llama(LlamaConfig(**{**SMALL, "n_layer": 1}, window=4)).eval()
    ids = torch.randint(0, 101, (1, 20))
    a, _ = m(ids)
    ids2 = ids.clone()
    ids2[0, 5] = (ids2[0, 5] + 1) % 101
    b, _ = m(ids2)
    # one layer, window 4: position 5 is visible to positions 5..8 only
    assert torch.allclose(a[0, 9:], b[0, 9:], atol=1e-6)
    assert not torch.allclose(a[0, 8], b[0, 8])


def test_gqa_shrinks_kv_cache():
    base = LlamaConfig(**SMALL)
    assert kv_bytes_per_token(base) == 2 * 2 * 4 * 16 * 2
    assert kv_bytes_per_token(LlamaConfig(**SMALL, n_kv_head=1)) * 4 == kv_bytes_per_token(base)
    c = KVCache(LlamaConfig(**SMALL, n_kv_head=2), 1, 10, "cpu", torch.bfloat16)
    assert c.nbytes() == 10 * kv_bytes_per_token(LlamaConfig(**SMALL, n_kv_head=2))


def test_gqa_params():
    mha = Llama(LlamaConfig(**SMALL)).num_params()
    mqa = Llama(LlamaConfig(**SMALL, n_kv_head=1)).num_params()
    # each layer's k and v shrink from C×C to C×d: saves 2·L·C·(C−d)
    assert mha - mqa == 2 * 2 * 64 * (64 - 16)


def test_ntk_scaling_keeps_fast_clocks():
    from llmfs.model.llama import rope_cache
    c1, _ = rope_cache(100, 64, 10000.0, 1.0, "cpu")
    c4, _ = rope_cache(100, 64, 10000.0, 4.0, "cpu")
    # first pair (fastest clock) nearly unchanged, last pair slowed by ~4x
    _, s1 = rope_cache(100, 64, 10000.0, 1.0, "cpu")
    _, s4 = rope_cache(100, 64, 10000.0, 4.0, "cpu")
    ang1 = torch.atan2(s1[1], c1[1])
    ang4 = torch.atan2(s4[1], c4[1])
    assert abs(ang1[0] - ang4[0]) < 1e-6
    assert ang1[-1] / ang4[-1] == pytest.approx(4.0, rel=0.02)


def test_trainer_builds_llama():
    from llmfs.model import build_model
    m = build_model(LlamaConfig(**SMALL).to_dict())
    assert isinstance(m, Llama)


def test_mla_cache_matches_full_and_is_small():
    torch.manual_seed(0)
    cfg = LlamaConfig(**SMALL, attn="mla", kv_latent=16, rope_dim=8)
    m = Llama(cfg).eval()
    ids = torch.randint(0, 101, (2, 30))
    full, _ = m(ids)
    cache = m.make_cache(2, 64, "cpu", torch.float32)
    out, _ = m(ids[:, :10], cache=cache)
    assert (out - full[:, :10]).abs().max() < 1e-4
    out, _ = m(ids[:, 10:18], cache=cache)
    assert (out - full[:, 10:18]).abs().max() < 1e-4
    for t in range(18, 30):
        out, _ = m(ids[:, t:t + 1], cache=cache)
        assert (out[:, 0] - full[:, t]).abs().max() < 1e-4
    # 24 numbers per token per layer vs 2·4·16 = 128 for MHA
    assert kv_bytes_per_token(cfg) * 128 == kv_bytes_per_token(LlamaConfig(**SMALL)) * 24


def test_sampler_works_on_llama_and_mla():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_generation"))
    from sample import generate_cached, generate_naive
    for kw in (dict(n_kv_head=2), dict(attn="mla", kv_latent=16, rope_dim=8)):
        torch.manual_seed(0)
        m = Llama(LlamaConfig(**SMALL, **kw)).eval()
        ids = torch.randint(0, 101, (1, 5))
        assert torch.equal(generate_naive(m, ids, 20, temperature=0.0),
                           generate_cached(m, ids, 20, temperature=0.0))
