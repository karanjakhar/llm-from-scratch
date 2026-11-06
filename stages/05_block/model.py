"""The Stage 04 tiny model, now built from our own Block.

Still not GPT (that is Stage 06): positions are GPT-2's learned table, the
head is tied to the embedding, init is plain std=0.02 everywhere, with no
depth-scaled init yet. The only new thing is that every block switch from
block.py is exposed in Config, so experiments can flip one at a time.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "03_embeddings"))

from block import Block, make_norm  # noqa: E402
from embeddings import LearnedPosition, OutputHead, TokenEmbedding  # noqa: E402


@dataclass
class Config:
    vocab_size: int = 8196
    dim: int = 256
    n_layer: int = 4
    n_head: int = 4
    max_len: int = 256
    norm: str = "layer"       # "layer" | "rms"
    act: str = "gelu"         # "gelu" | "gelu_tanh" | "relu" | "swiglu" | "none"
    prenorm: bool = True
    residual: bool = True
    dropout: float = 0.0
    emb_scale: bool = False   # multiply token embeddings by √C, as the 2017 paper did


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = TokenEmbedding(cfg.vocab_size, cfg.dim)
        self.pos = LearnedPosition(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.n_head, norm=cfg.norm, act=cfg.act, prenorm=cfg.prenorm,
                  residual=cfg.residual, dropout=cfg.dropout)
            for _ in range(cfg.n_layer))
        # Pre-norm leaves the stream un-normalised, so the head needs one last
        # norm (docs §4). Post-norm already ends every block with a norm, so a
        # second one would be a norm of a norm; the 2017 design has none.
        self.ln_f = make_norm(cfg.norm, cfg.dim) if cfg.prenorm else nn.Identity()
        self.head = OutputHead(cfg.dim, cfg.vocab_size, tie_to=self.tok)
        for m in self.modules():                                 # GPT-2 style init, as Stage 04
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None,
                keep_stream: bool = False):
        B, T = ids.shape
        tok = self.tok(ids)                                      # (B, T, C), rms ≈ 0.02
        if self.cfg.emb_scale:
            tok = tok * self.cfg.dim ** 0.5                      # rms ≈ 0.02·√C (0.32 at C=256)
        x = tok + self.pos(T)                                    # (B, T, C)
        stream = [x] if keep_stream else None
        for blk in self.blocks:
            x = blk(x)                                           # (B, T, C)
            if keep_stream:
                stream.append(x)
        logits = self.head(self.ln_f(x))                         # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        if keep_stream:
            return logits, loss, stream                          # stream: (n_layer+1) × (B, T, C)
        return logits, loss
