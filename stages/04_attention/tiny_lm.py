"""The Stage 03 preview model, now running on OUR attention.

Still a preview: the block (residuals, LayerNorm, MLP) is built properly in
Stage 05. Positions are GPT-2's learned table, the scheme Stage 06 uses.
The switches that matter here are n_head (so d = dim / n_head) and scale.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
# Stage 03 pieces. Appended, not inserted first: both stages have a tiny_lm.py
# and this stage's own files must win.
sys.path.append(str(HERE.parent / "03_embeddings"))

from attention import MultiHeadAttention
from embeddings import LearnedPosition, OutputHead, TokenEmbedding


@dataclass
class Config:
    vocab_size: int = 8196
    dim: int = 256
    n_layer: int = 4
    n_head: int = 4
    max_len: int = 256
    scale: bool = True        # divide scores by √d (docs §3)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadAttention(cfg.dim, cfg.n_head, causal=True, scale=cfg.scale)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.dim, 4 * cfg.dim), nn.GELU(),
                                 nn.Linear(4 * cfg.dim, cfg.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))                           # (B, T, C)
        return x + self.mlp(self.ln2(x))                         # (B, T, C)


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos = LearnedPosition(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.head = OutputHead(cfg.dim, cfg.vocab_size, tie_to=self.tok)
        for m in self.modules():                                 # GPT-2 style init
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def keep_attention(self, on: bool = True):
        """Make every layer keep its (B, H, T, T) weights, for maps and entropy."""
        for b in self.blocks:
            b.attn.keep_weights = on

    def attention_weights(self) -> list[torch.Tensor]:
        return [b.attn.last_weights for b in self.blocks]       # n_layer × (B, H, T, T)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = ids.shape
        x = self.tok(ids) + self.pos(T)                          # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                         # (B, T, V)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss
