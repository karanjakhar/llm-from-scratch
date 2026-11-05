"""Acceptance criteria for Stage 05 (docs/05_transformer_block.md §11).

    uv run pytest stages/05_block -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from block import MLP, Block, LayerNorm, RMSNorm, SwiGLU, gelu_exact, gelu_tanh, silu  # noqa: E402
from model import Config, TinyLM  # noqa: E402

ATOL = 1e-5


def _rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64)


# --- norms ------------------------------------------------------------------

def test_layernorm_matches_torch_outputs_and_grads():
    C = 48
    ours, ref = LayerNorm(C).double(), nn.LayerNorm(C).double()
    with torch.no_grad():                     # non-trivial γ, β so they are tested too
        g, b = _rand(C, seed=1), _rand(C, seed=2)
        ours.weight.copy_(g); ours.bias.copy_(b)
        ref.weight.copy_(g); ref.bias.copy_(b)
    x1 = (_rand(3, 7, C) * 4 + 2).requires_grad_()   # off-centre, large: LayerNorm must not care
    x2 = x1.detach().clone().requires_grad_()
    up = _rand(3, 7, C, seed=3)
    (ours(x1) * up).sum().backward()
    (ref(x2) * up).sum().backward()
    torch.testing.assert_close(ours(x1), ref(x2), atol=ATOL, rtol=0)
    torch.testing.assert_close(x1.grad, x2.grad, atol=ATOL, rtol=0)
    torch.testing.assert_close(ours.weight.grad, ref.weight.grad, atol=ATOL, rtol=0)
    torch.testing.assert_close(ours.bias.grad, ref.bias.grad, atol=ATOL, rtol=0)


def test_rmsnorm_matches_torch_outputs_and_grads():
    C = 48
    ours, ref = RMSNorm(C, eps=1e-6).double(), nn.RMSNorm(C, eps=1e-6).double()
    with torch.no_grad():
        g = _rand(C, seed=1)
        ours.weight.copy_(g); ref.weight.copy_(g)
    x1 = (_rand(3, 7, C) * 4 + 2).requires_grad_()
    x2 = x1.detach().clone().requires_grad_()
    up = _rand(3, 7, C, seed=3)
    (ours(x1) * up).sum().backward()
    (ref(x2) * up).sum().backward()
    torch.testing.assert_close(ours(x1), ref(x2), atol=ATOL, rtol=0)
    torch.testing.assert_close(x1.grad, x2.grad, atol=ATOL, rtol=0)
    torch.testing.assert_close(ours.weight.grad, ref.weight.grad, atol=ATOL, rtol=0)


def test_rmsnorm_matches_hf_llama():
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    C = 32
    ours, ref = RMSNorm(C, eps=1e-6), LlamaRMSNorm(C, eps=1e-6)
    x = _rand(2, 5, C).float()
    torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=0)


def test_norm_statistics_per_token():
    x = _rand(4, 9, 64) * 10 + 3
    y = LayerNorm(64).double()(x)
    torch.testing.assert_close(y.mean(-1), torch.zeros(4, 9, dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(y.var(-1, unbiased=False), torch.ones(4, 9, dtype=torch.float64),
                               atol=1e-4, rtol=0)   # not exactly 1: ε in the denominator
    z = RMSNorm(64).double()(x)
    torch.testing.assert_close(z.pow(2).mean(-1), torch.ones(4, 9, dtype=torch.float64), atol=1e-4, rtol=0)


def test_doc_worked_examples():
    """The numbers printed in docs §3."""
    x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    torch.testing.assert_close(LayerNorm(4)(x), torch.tensor([[-1.3416, -0.4472, 0.4472, 1.3416]]),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(RMSNorm(4, eps=0.0)(x), torch.tensor([[0.3651, 0.7303, 1.0954, 1.4606]]),
                               atol=1e-4, rtol=0)
    torch.testing.assert_close(LayerNorm(4)(x + 100), LayerNorm(4)(x), atol=1e-4, rtol=0)


def test_layernorm_needs_biased_variance():
    """The trap in LayerNorm.forward: x.var() is the C-1 version. It is close
    enough to look right and wrong by a factor sqrt(C/(C-1))."""
    x = _rand(2, 3, 8)
    m = x.mean(-1, keepdim=True)
    wrong = (x - m) / torch.sqrt(x.var(-1, keepdim=True) + 1e-5)
    assert (wrong - F.layer_norm(x, (8,))).abs().max() > 1e-2
    torch.testing.assert_close(LayerNorm(8).double()(x), F.layer_norm(x, (8,)), atol=ATOL, rtol=0)


# --- activations and MLPs ---------------------------------------------------

def test_activations_match_torch():
    x = torch.linspace(-6, 6, 1001, dtype=torch.float64)
    torch.testing.assert_close(gelu_tanh(x), F.gelu(x, approximate="tanh"), atol=1e-7, rtol=0)
    torch.testing.assert_close(gelu_exact(x), F.gelu(x), atol=1e-7, rtol=0)
    torch.testing.assert_close(silu(x), F.silu(x), atol=1e-7, rtol=0)
    # The two GELUs are close but not equal, which is why the GPT-2 check needs the tanh one.
    assert 1e-5 < (gelu_tanh(x) - gelu_exact(x)).abs().max() < 1e-3


def test_swiglu_matches_hf_llama_mlp():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP
    C, h = 64, 170
    ours = SwiGLU(C, hidden=h)
    ref = LlamaMLP(LlamaConfig(hidden_size=C, intermediate_size=h, hidden_act="silu"))
    with torch.no_grad():
        ours.gate.weight.copy_(ref.gate_proj.weight)
        ours.up.weight.copy_(ref.up_proj.weight)
        ours.proj.weight.copy_(ref.down_proj.weight)
    x = _rand(2, 5, C).float()
    torch.testing.assert_close(ours(x), ref(x), atol=ATOL, rtol=0)


def test_swiglu_param_count_matches_mlp():
    for C in (256, 768, 1024):
        n_mlp = sum(p.numel() for p in MLP(C, bias=False).parameters())
        n_glu = sum(p.numel() for p in SwiGLU(C).parameters())
        assert abs(n_glu - n_mlp) / n_mlp < 0.01, (C, n_mlp, n_glu)


# --- the block --------------------------------------------------------------

def _copy_gpt2_block(ours: Block, ref) -> None:
    """HF GPT-2 uses Conv1D, which stores weights (in, out). nn.Linear stores
    (out, in). Every weight needs a transpose; for the square c_proj a missing
    transpose keeps the shape and gives the wrong answer (docs §9)."""
    with torch.no_grad():
        ours.ln1.weight.copy_(ref.ln_1.weight); ours.ln1.bias.copy_(ref.ln_1.bias)
        ours.ln2.weight.copy_(ref.ln_2.weight); ours.ln2.bias.copy_(ref.ln_2.bias)
        ours.attn.qkv.weight.copy_(ref.attn.c_attn.weight.T); ours.attn.qkv.bias.copy_(ref.attn.c_attn.bias)
        ours.attn.proj.weight.copy_(ref.attn.c_proj.weight.T); ours.attn.proj.bias.copy_(ref.attn.c_proj.bias)
        ours.mlp.fc.weight.copy_(ref.mlp.c_fc.weight.T); ours.mlp.fc.bias.copy_(ref.mlp.c_fc.bias)
        ours.mlp.proj.weight.copy_(ref.mlp.c_proj.weight.T); ours.mlp.proj.bias.copy_(ref.mlp.c_proj.bias)


def _hf_gpt2_block(C, H):
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    cfg = GPT2Config(n_embd=C, n_head=H, activation_function="gelu_new",
                     resid_pdrop=0.0, attn_pdrop=0.0, embd_pdrop=0.0)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    ref = GPT2Block(cfg, layer_idx=0).eval()
    with torch.no_grad():                     # HF inits LN to (1, 0); randomise so the copy is tested
        for p in ref.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return ref


def _causal_additive_mask(T):
    m = torch.full((T, T), float("-inf")).triu(1)               # -inf above the diagonal
    return m[None, None]                                        # (1, 1, T, T), broadcast over B, H


def test_block_matches_hf_gpt2_block():
    C, H, T = 64, 4, 11
    ref = _hf_gpt2_block(C, H)
    ours = Block(C, H, norm="layer", act="gelu_tanh", prenorm=True).eval()
    _copy_gpt2_block(ours, ref)
    x = torch.randn(2, T, C)
    out_ref = ref(x, attention_mask=_causal_additive_mask(T))
    out_ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref
    torch.testing.assert_close(ours(x), out_ref, atol=ATOL, rtol=0)


def test_block_without_transpose_is_wrong():
    """The Conv1D trap, demonstrated: copy c_proj without .T and the answer moves."""
    C, H, T = 64, 4, 11
    ref = _hf_gpt2_block(C, H)
    ours = Block(C, H, act="gelu_tanh").eval()
    _copy_gpt2_block(ours, ref)
    with torch.no_grad():
        ours.attn.proj.weight.copy_(ref.attn.c_proj.weight)     # same shape, no error raised
    x = torch.randn(2, T, C)
    out_ref = ref(x, attention_mask=_causal_additive_mask(T))
    out_ref = out_ref[0] if isinstance(out_ref, tuple) else out_ref
    assert (ours(x) - out_ref).abs().max() > 1e-2


@pytest.mark.parametrize("prenorm", [True, False])
@pytest.mark.parametrize("act", ["gelu", "swiglu"])
def test_zeroed_outputs_make_block_identity(prenorm, act):
    """With both sublayers writing zero, a pre-norm block passes x straight
    through (the highway, docs §2). Post-norm does not: the norm on the
    highway still rescales x. That asymmetry is the whole of docs §4."""
    C = 32
    blk = Block(C, 4, act=act, prenorm=prenorm)
    with torch.no_grad():
        for lin in (blk.attn.proj, blk.mlp.proj):
            lin.weight.zero_()
            if lin.bias is not None:
                lin.bias.zero_()
    x = torch.randn(2, 6, C) * 3 + 1
    if prenorm:
        torch.testing.assert_close(blk(x), x, atol=0, rtol=0)
    else:
        assert (blk(x) - x).abs().max() > 0.5


@pytest.mark.parametrize("kw", [dict(), dict(prenorm=False), dict(act="swiglu", norm="rms"),
                                dict(residual=False)])
def test_block_is_causal(kw):
    C, T = 32, 10
    torch.manual_seed(0)
    blk = Block(C, 4, **kw).eval()
    x = torch.randn(1, T, C)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, T - 6, C)                        # change the future only
    torch.testing.assert_close(blk(x)[:, :6], blk(x2)[:, :6], atol=0, rtol=0)


def test_block_param_count_matches_doc_table():
    assert sum(p.numel() for p in Block(256, 4).parameters()) == 789_760   # docs §8
    assert 12 * 256 ** 2 == 786_432


def test_dropout_is_off_in_eval_and_scaled_in_train():
    C = 32
    torch.manual_seed(0)
    blk = Block(C, 4, dropout=0.5)
    x = torch.randn(2, 6, C)
    blk.eval()
    a, b = blk(x), blk(x)
    torch.testing.assert_close(a, b, atol=0, rtol=0)            # deterministic at inference
    blk.train()
    assert (blk(x) - a).abs().max() > 1e-3                      # random during training


def test_tinylm_runs_every_variant():
    torch.manual_seed(0)
    ids = torch.randint(0, 8196, (4, 64))
    tgt = torch.randint(0, 8196, (4, 64))                       # independent of ids: see below
    for kw in [dict(), dict(norm="rms"), dict(act="relu"), dict(act="swiglu"), dict(act="none"),
               dict(prenorm=False), dict(residual=False), dict(dropout=0.1)]:
        m = TinyLM(Config(n_layer=2, max_len=64, **kw))
        _, loss = m(ids, tgt)
        assert torch.isfinite(loss)
        # At init the model should be close to uniform over 8,196 tokens: ln(8196) = 9.01.
        assert abs(loss.item() - 9.01) < 0.3, (kw, loss.item())


def test_tied_head_at_init_predicts_its_own_input():
    """A trap met while writing the test above: with targets = inputs the
    initial loss was 8.46 against 9.07 for random targets (this seed). With a tied head, the logit for token i
    is roughly <E[i] + ..., E[i]> = |E[i]|² > 0, so an untrained model already
    favours repeating the token it was given. Real targets are the NEXT
    token, so this bias does not help; it is why a "copy the input" bug can
    look like early learning."""
    torch.manual_seed(0)
    ids = torch.randint(0, 8196, (4, 64))
    m = TinyLM(Config(n_layer=2, max_len=64))
    _, loss_self = m(ids, ids)
    _, loss_rand = m(ids, torch.randint(0, 8196, (4, 64)))
    assert loss_self.item() < loss_rand.item() - 0.3
