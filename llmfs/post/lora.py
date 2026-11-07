"""LoRA and QLoRA from scratch. docs/17_peft_lora.md.

    add_lora(model, r=16, alpha=32, targets=("c_attn", "c_proj", "c_fc"))
    ... train: only A and B have requires_grad ...
    merge_lora(model)        # W <- W + (alpha/r) B A, back to plain nn.Linear

QLoRA: quantize_base_nf4(model) replaces each frozen base weight with 4-bit
NF4 codes + per-block scales (docs §5), dequantized on the fly in forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = x Wᵀ + b + (alpha/r) · x Aᵀ Bᵀ.   W frozen, A (r, in), B (out, r).

    B starts at zero, so at step 0 the adapter adds exactly nothing and the
    model is unchanged (docs §2). A starts random so that B's gradient
    (which is proportional to A x) is not zero too.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        dev = base.weight.device                               # adapters live where the base does
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))      # same init as nn.Linear and peft
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        # x (.., in) -> (.., r) -> (.., out): two thin matmuls, never the full (out, in) update
        return y + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scale            # (out, in), rank ≤ r


def _parent(model, name):
    parts = name.split(".")
    m = model
    for p in parts[:-1]:
        m = getattr(m, p)
    return m, parts[-1]


def add_lora(model: nn.Module, r: int = 16, alpha: float = 32, targets=("c_attn", "c_proj", "c_fc"),
             dropout: float = 0.0) -> nn.Module:
    """Freeze everything, then wrap every nn.Linear whose name ends in a target."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets and n != "lm_head"]
    for n in names:
        parent, attr = _parent(model, n)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), r, alpha, dropout))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold each adapter into its base weight and remove it. Inference cost -> zero."""
    for n, m in list(model.named_modules()):
        if isinstance(m, LoRALinear):
            base = m.base
            if isinstance(base, NF4Linear):
                base = base.dequantized_linear()
            with torch.no_grad():
                base.weight += m.delta().to(base.weight.dtype)
            parent, attr = _parent(model, n)
            setattr(parent, attr, base)
    return model


def lora_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# NF4 (docs §5): 16 levels placed at the quantiles of a standard normal
# ---------------------------------------------------------------------------

def nf4_levels() -> torch.Tensor:
    """The 16 NF4 values from the QLoRA paper (Dettmers et al. 2023), in [-1, 1].

    Built as normal quantiles: 8 positive levels from the upper half, 7 negative
    from the lower half, plus an exact 0. Asymmetric so that zero is
    representable (weights near zero are the most common).
    """
    from statistics import NormalDist
    nd = NormalDist()
    offset = 0.9677083                                           # the paper's choice, keeps ±1 finite
    pos = torch.tensor([nd.inv_cdf(p) for p in torch.linspace(0.5, offset, 9)[1:].tolist()])
    neg = torch.tensor([nd.inv_cdf(p) for p in torch.linspace(1 - offset, 0.5, 8)[:-1].tolist()])
    v = torch.cat([neg, torch.zeros(1), pos])
    return v / v.abs().max()                                     # (16,), sorted, max |v| = 1


NF4 = nf4_levels()


def quantize_nf4(w: torch.Tensor, block: int = 64):
    """w (any shape) -> (codes uint8 (n,), absmax (n/block,), shape). Per-block absmax scaling."""
    flat = w.detach().float().reshape(-1, block)                 # (nb, block)
    absmax = flat.abs().max(-1, keepdim=True).values.clamp(min=1e-12)
    normed = flat / absmax                                       # in [-1, 1]
    levels = NF4.to(w.device)
    codes = (normed[..., None] - levels).abs().argmin(-1).to(torch.uint8)    # nearest level
    return codes.reshape(-1), absmax.reshape(-1), w.shape


def dequantize_nf4(codes, absmax, shape, block: int = 64, dtype=torch.float32):
    levels = NF4.to(codes.device)
    return (levels[codes.long()].reshape(-1, block) * absmax[:, None]).reshape(shape).to(dtype)


class NF4Linear(nn.Module):
    """A frozen linear whose weight is stored as 4-bit NF4 codes (two per byte)."""

    def __init__(self, lin: nn.Linear, block: int = 64):
        super().__init__()
        codes, absmax, shape = quantize_nf4(lin.weight, block)
        packed = (codes[0::2] << 4) | codes[1::2]                # 2 codes per byte: real 4-bit storage
        self.register_buffer("packed", packed)
        self.register_buffer("absmax", absmax.to(torch.bfloat16))
        self.shape, self.block = shape, block
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def weight_fp(self, dtype):
        codes = torch.stack([self.packed >> 4, self.packed & 15], -1).reshape(-1)
        return dequantize_nf4(codes, self.absmax.float(), self.shape, self.block, dtype)

    def forward(self, x):
        return F.linear(x, self.weight_fp(x.dtype if x.dtype != torch.float32 else torch.float32), self.bias)

    def dequantized_linear(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.weight_fp(torch.float32))
            if self.bias is not None:
                lin.bias.copy_(self.bias)
        return lin.to(self.packed.device)

    def nbytes(self):
        return self.packed.numel() + self.absmax.numel() * 2


def quantize_base_nf4(model: nn.Module, targets=("c_attn", "c_proj", "c_fc"), block: int = 64):
    """Replace target nn.Linear layers (or LoRALinear bases) with NF4Linear."""
    for n, m in list(model.named_modules()):
        if isinstance(m, LoRALinear) and isinstance(m.base, nn.Linear):
            m.base = NF4Linear(m.base, block)
        elif isinstance(m, nn.Linear) and n.split(".")[-1] in targets:
            parent, attr = _parent(model, n)
            setattr(parent, attr, NF4Linear(m, block))
    return model
