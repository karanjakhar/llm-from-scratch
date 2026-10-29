"""A tiny causal language model with a switchable position scheme.

PREVIEW CODE. Attention is built properly in Stage 04 and the block in Stage
05. Here they are as short as possible and use PyTorch's fused attention
kernel, because this stage measures *position*, not attention. Everything
except the position scheme is identical across ablation runs.

    pos = "none"        no position at all; only the causal mask hints at order
          "learned"     GPT-2 table, added to the token vectors
          "sinusoidal"  fixed clocks, added to the token vectors
          "rope"        q and k rotated by position inside every attention layer
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from embeddings import (
    LearnedPosition,
    OutputHead,
    SinusoidalPosition,
    TokenEmbedding,
    rope_angles,
    rope_interleaved,
)

POS_SCHEMES = ("none", "learned", "sinusoidal", "rope")


@dataclass
class Config:
    vocab_size: int = 8196
    dim: int = 256           # C
    n_layer: int = 4
    n_head: int = 4          # d = dim // n_head = 64
    max_len: int = 256       # training context; also the learned table's size
    pos: str = "rope"
    tie: bool = True
    # Multiply token vectors by this before adding position. The 2017 paper
    # uses sqrt(C): its sinusoids have length ~sqrt(C/2) (11.3 at C=256) while
    # our 0.02-init token rows have length ~0.32, so without it position
    # drowns the token 35:1. Found the hard way; docs §8, JOURNAL 2026-10-29.
    emb_scale: float = 1.0


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
                                 nn.Linear(4 * cfg.dim, cfg.dim))

    def forward(self, x: torch.Tensor, angles: torch.Tensor | None) -> torch.Tensor:
        B, T, C = x.shape
        H = self.cfg.n_head
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=-1)         # 3 × (B, T, C)
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in (q, k, v))  # (B, H, T, d)
        if angles is not None:
            # RoPE touches q and k only. v carries content, not position.
            q, k = rope_interleaved(q, angles), rope_interleaved(k, angles)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))    # (B, T, C)
        return x + self.mlp(self.ln2(x))                         # (B, T, C)


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.pos in POS_SCHEMES, cfg.pos
        self.cfg = cfg
        self.tok = TokenEmbedding(cfg.vocab_size, cfg.dim)
        if cfg.pos == "learned":
            self.pos = LearnedPosition(cfg.max_len, cfg.dim)
        elif cfg.pos == "sinusoidal":
            self.pos = SinusoidalPosition(cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = OutputHead(cfg.dim, cfg.vocab_size, tie_to=self.tok if cfg.tie else None)
        for m in self.modules():                                 # GPT-2 style init
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
                reduction: str = "mean"):
        B, T = ids.shape
        x = self.tok(ids) * self.cfg.emb_scale                   # (B, T, C)
        angles = None
        if self.cfg.pos == "learned":
            x = x + self.pos(T)                                  # (T, C) broadcast over B
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos(T, device=ids.device)
        elif self.cfg.pos == "rope":
            angles = rope_angles(T, self.cfg.dim // self.cfg.n_head).to(ids.device)  # (T, d/2)
        for blk in self.blocks:
            x = blk(x, angles)
        logits = self.head(self.ln_f(x))                         # (B, T, V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(),
                               reduction=reduction)              # () or (B*T,)
        if reduction == "none":
            loss = loss.view(B, T)                               # per-position loss
        return logits, loss
