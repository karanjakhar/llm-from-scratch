"""Acceptance criteria for Stage 17 (docs/17_peft_lora.md §7)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.post.lora import (NF4, LoRALinear, NF4Linear, add_lora, dequantize_nf4, merge_lora,  # noqa: E402
                             quantize_base_nf4, quantize_nf4, trainable)

CFG = dict(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32)


def _model(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(**CFG)).eval()


def test_zero_init_changes_nothing():
    m = _model()
    x = torch.randint(0, 64, (2, 16))
    before, _ = m(x)
    add_lora(m, r=4, alpha=8)
    after, _ = m(x)
    assert torch.equal(before, after)


def test_only_adapters_train():
    m = add_lora(_model(), r=4, alpha=8)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    # per wrapped layer: r·(in + out)
    want = 0
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            want += 4 * (mod.base.in_features + mod.base.out_features)
    assert trainable(m) == want


def test_matches_peft():
    from peft import LoraConfig, inject_adapter_in_model
    ours, ref = _model(), _model()
    add_lora(ours, r=4, alpha=8, targets=("c_attn", "c_fc"))
    ref = inject_adapter_in_model(LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn", "c_fc"],
                                             lora_dropout=0.0), ref)
    torch.manual_seed(5)
    with torch.no_grad():
        for (n, mod) in ours.named_modules():
            if isinstance(mod, LoRALinear):
                pm = dict(ref.named_modules())[n]
                mod.lora_A.normal_()
                mod.lora_B.normal_()
                pm.lora_A["default"].weight.copy_(mod.lora_A)
                pm.lora_B["default"].weight.copy_(mod.lora_B)
    x = torch.randint(0, 64, (2, 16))
    a, _ = ours(x)
    b, _ = ref(x)
    assert (a - b).abs().max() < 1e-5


def test_merge_equals_unmerged():
    m = add_lora(_model(), r=4, alpha=8)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_B.normal_(std=0.1)
    x = torch.randint(0, 64, (2, 16))
    a, _ = m(x)
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    b, _ = m(x)
    assert (a - b).abs().max() < 1e-5


def test_delta_has_rank_r():
    m = add_lora(_model(), r=3, alpha=6)
    mod = next(x for x in m.modules() if isinstance(x, LoRALinear))
    with torch.no_grad():
        mod.lora_B.normal_()
    assert torch.linalg.matrix_rank(mod.delta()).item() == 3


def test_nf4_levels_match_paper():
    paper = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
             0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
    assert torch.allclose(NF4, torch.tensor(paper), atol=2e-4)


def test_nf4_roundtrip_error_and_size():
    torch.manual_seed(0)
    w = torch.randn(256, 128) * 0.02
    codes, absmax, shape = quantize_nf4(w)
    back = dequantize_nf4(codes, absmax, shape)
    rel = ((back - w).norm() / w.norm()).item()
    assert rel < 0.12                                       # ~4 bits of precision
    lin = torch.nn.Linear(128, 256)
    lin.weight.data = w
    q = NF4Linear(lin)
    # 0.5 byte per weight + 2 bytes per 64-weight block = 0.53 B/weight vs 4 for fp32
    assert q.nbytes() == 256 * 128 // 2 + (256 * 128 // 64) * 2


def test_qlora_forward_close_to_lora():
    m = add_lora(_model(), r=4, alpha=8)
    x = torch.randint(0, 64, (2, 16))
    a, _ = m(x)
    quantize_base_nf4(m)
    assert any(isinstance(mod, NF4Linear) for mod in m.modules())
    b, _ = m(x)
    assert (a - b).abs().max() < 0.05 * a.abs().max()
