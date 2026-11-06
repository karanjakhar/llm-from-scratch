"""Stage 06: the complete GPT. docs/06_gpt_model.md is the lesson.

Everything from Stages 03-05 in one model, with GPT-2's module names so the
real GPT-2 weights load into it (docs §5). This is the explicit, commented
version. llmfs/model/gpt.py is the same model in the form later stages import.

Shape letters: B batch, T tokens, C width (n_embd), H heads, d = C/H,
L layers, V vocabulary.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "04_attention"))
sys.path.append(str(HERE.parent / "05_block"))

from attention import attention  # noqa: E402   Stage 04, hand-written softmax(qkᵀ/√d)v
from block import LayerNorm, gelu_exact, gelu_tanh  # noqa: E402   Stage 05


@dataclass
class GPTConfig:
    vocab_size: int = 8196        # our tokenizer: 8,192 merges + 4 specials
    block_size: int = 256         # T_max: rows in the position table
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    act: str = "gelu_tanh"        # GPT-2 uses the tanh approximation (Stage 05 §6)
    tie: bool = True              # lm_head shares wte's table (Stage 03 §2)
    scaled_init: bool = True      # c_proj std 0.02/√(2L) (docs §4b)
    flash: bool = False           # F.scaled_dot_product_attention instead of Stage 04's
    bias: bool = True             # GPT-2 has biases on every linear and norm


# GPT-2's four sizes, for counting and loading. vocab 50,257, T_max 1,024.
GPT2_SIZES = {
    "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),     # 124M
    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),    # 350M
    "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),    # 774M
    "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),    # 1558M
}


class CausalSelfAttention(nn.Module):
    """Stage 04's MultiHeadAttention under GPT-2's names: c_attn makes q, k, v
    in one (C, 3C) matmul, c_proj mixes the heads and writes to the stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.flash = cfg.n_head, cfg.flash
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)   # [W_q | W_k | W_v]
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)       # writes to the residual stream
        self.c_proj.writes_residual = True                       # marker for the scaled init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.c_attn(x).split(C, dim=-1)                # 3 × (B, T, C)
        q = q.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        k = k.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        v = v.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        if self.flash:
            # Same arithmetic, fused into one kernel that never stores the
            # (T, T) matrix. Stage 04 checked ours equals it to 1e-5; Stage 09
            # measures what it buys.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = attention(q, k, v, causal=True)                # (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, C)               # heads side by side again
        return self.c_proj(out)                                  # (B, T, C)


class MLP(nn.Module):
    """C -> 4C -> C, Stage 05 §5, under GPT-2's names."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.writes_residual = True
        self.act = {"gelu_tanh": gelu_tanh, "gelu": gelu_exact}[cfg.act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))               # (B, T, C) -> (B, T, 4C) -> (B, T, C)


class Block(nn.Module):
    """Pre-norm block (Stage 05 §4): x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, C)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # nn.ModuleDict named "transformer" so parameter names are exactly
        # GPT-2's ("transformer.h.3.attn.c_attn.weight"), docs §5.
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),       # (V, C)
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),       # (T_max, C)
            h=nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer)),
            ln_f=LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)    # (V, C) stored
        if cfg.tie:
            # One Parameter, two uses. Its gradient is the sum of both.
            self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init)
        self._scale_residual_init()

    # -- initialisation, docs §4 -------------------------------------------

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        # LayerNorm keeps its own init: gain 1, shift 0.

    def _scale_residual_init(self):
        if not self.cfg.scaled_init:
            return
        # 2L layers write into the stream; shrink each so their summed
        # variance at init equals one layer's (docs §4b).
        std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for m in self.modules():
            if getattr(m, "writes_residual", False):
                nn.init.normal_(m.weight, mean=0.0, std=std)

    # -- forward -----------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                keep_stream: bool = False):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"T={T} > block_size={self.cfg.block_size}"
        tr = self.transformer
        pos = torch.arange(T, device=idx.device)                 # (T,)
        x = tr.wte(idx) + tr.wpe(pos)                            # (B, T, C) + (T, C) broadcast
        stream = [x] if keep_stream else None
        for block in tr.h:
            x = block(x)                                         # (B, T, C)
            if keep_stream:
                stream.append(x)
        x = tr.ln_f(x)                                           # (B, T, C)
        logits = self.lm_head(x)                                 # (B, T, V)
        loss = None
        if targets is not None:
            # targets are the inputs shifted left by one, made by the caller:
            # idx = window[:, :-1], targets = window[:, 1:] (docs §2).
            # float() so a bf16 forward still gets an fp32 log-softmax.
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        if keep_stream:
            return logits, loss, stream
        return logits, loss

    # -- loading GPT-2, docs §5 ---------------------------------------------

    @classmethod
    def from_pretrained(cls, name: str = "gpt2", **overrides) -> "GPT":
        """Build GPT-2 `name` and copy OpenAI's weights in, reading the
        safetensors file directly (no transformers model is constructed)."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg = GPTConfig(vocab_size=50257, block_size=1024, act="gelu_tanh", tie=True,
                        **GPT2_SIZES[name], **overrides)
        model = cls(cfg)
        path = hf_hub_download(f"openai-community/{name}", "model.safetensors")
        theirs = load_file(path)
        load_gpt2_state(model, theirs)
        return model


# Conv1D layers store (in, out); nn.Linear stores (out, in). docs §5 trap 1.
TRANSPOSED = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def load_gpt2_state(model: GPT, theirs: dict[str, torch.Tensor]) -> None:
    """Copy a GPT-2 state dict into `model`. Fails on anything unmatched."""
    ours = model.state_dict()
    used: set[str] = set()
    filled: set[str] = set()
    for k, v in theirs.items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            used.add(k)                                          # the causal mask saved as a buffer (trap 2)
            continue
        if k == "lm_head.weight":                                # tied: same table as wte (trap 3)
            used.add(k)
            continue
        name = k if k.startswith("transformer.") else "transformer." + k
        if name not in ours:
            raise KeyError(f"their tensor {k!r} has no home in our model")
        if any(name.endswith(t) for t in TRANSPOSED):
            v = v.T
        if v.shape != ours[name].shape:
            raise ValueError(f"{k}: theirs {tuple(v.shape)} vs ours {tuple(ours[name].shape)}")
        with torch.no_grad():
            ours[name].copy_(v)                                  # state_dict tensors share storage
        used.add(k)
        filled.add(name)
    missing = [n for n in ours if n not in filled and n != "lm_head.weight"]
    if missing:
        raise KeyError(f"our tensors never filled: {missing[:5]} ...")
    if not model.cfg.tie:
        with torch.no_grad():
            model.lm_head.weight.copy_(model.transformer.wte.weight)
    assert set(theirs) == used


# ---------------------------------------------------------------------------
# Counting by hand, docs §3
# ---------------------------------------------------------------------------

def count_params_by_hand(cfg: GPTConfig) -> dict[str, int]:
    V, C, L, Tm = cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.block_size
    b = 1 if cfg.bias else 0
    per_block = (
        2 * 2 * C                         # ln_1, ln_2: gain + shift
        + C * 3 * C + b * 3 * C           # c_attn
        + C * C + b * C                   # attn.c_proj
        + C * 4 * C + b * 4 * C           # c_fc
        + 4 * C * C + b * C               # mlp.c_proj
    )                                     # = 12C² + 13C with biases
    parts = {
        "wte": V * C,
        "wpe": Tm * C,
        "blocks": L * per_block,
        "ln_f": 2 * C,
        "lm_head": 0 if cfg.tie else V * C,
    }
    parts["total"] = sum(parts.values())
    parts["non_embedding"] = parts["blocks"] + parts["ln_f"]
    return parts


def count_params(model: nn.Module) -> int:
    # .parameters() yields a shared (tied) tensor once.
    return sum(p.numel() for p in model.parameters())
