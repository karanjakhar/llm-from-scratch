"""Stage 04: attention, from one head up.

docs/04_attention.md is the lesson. Everything here is written out by hand
and checked against F.scaled_dot_product_attention and nn.MultiheadAttention
in test_attention.py.

Shape letters: B batch, T tokens, C model width, H heads, d = C // H head size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """True where attention is ALLOWED: key j at or before query i."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()   # (T, T)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: bool = True,
              return_weights: bool = False):
    """softmax(q kᵀ / √d) v, written out. docs §2-§4.

    q, k, v: (..., T, d). Any leading dims (batch, heads) are carried along.
    scale=False drops the √d, for the ablation in docs §9 only.
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1)                             # (..., T, d) @ (..., d, T) -> (..., T, T)
    if scale:
        scores = scores / math.sqrt(d)                           # variance d -> 1 (docs §3)
    if causal:
        T = q.shape[-2]
        # -inf, not 0: exp(-inf) = 0 exactly, so future tokens get zero weight
        # and the remaining weights still sum to 1 (docs §4).
        scores = scores.masked_fill(~causal_mask(T, q.device), float("-inf"))
    weights = scores.softmax(dim=-1)                             # (..., T, T), rows sum to 1
    out = weights @ v                                            # (..., T, T) @ (..., T, d) -> (..., T, d)
    return (out, weights) if return_weights else out


class SingleHeadAttention(nn.Module):
    """One head: three projections C -> d, then attention. No output projection,
    because with one head there is nothing to mix (docs §5)."""

    def __init__(self, dim: int, head_dim: int, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.q = nn.Linear(dim, head_dim, bias=False)            # W_q (C, d)
        self.k = nn.Linear(dim, head_dim, bias=False)            # W_k
        self.v = nn.Linear(dim, head_dim, bias=False)            # W_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)                # (B, T, d) each
        return attention(q, k, v, causal=self.causal)            # (B, T, d)


class MultiHeadAttention(nn.Module):
    """H heads of size d = C / H, side by side, then an output projection.

    One (C, 3C) matmul makes q, k, v for every head at once; the heads are
    then split out with view + transpose. That reshape is where attention bugs
    hide: a wrong one keeps the shape and scrambles the contents.
    """

    def __init__(self, dim: int, n_head: int, causal: bool = True,
                 scale: bool = True, bias: bool = True):
        super().__init__()
        assert dim % n_head == 0, "heads must split the width evenly"
        self.n_head, self.causal, self.scale = n_head, causal, scale
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)            # [W_q | W_k | W_v], each (C, C)
        self.proj = nn.Linear(dim, dim, bias=bias)               # W_o (C, C): mixes the heads
        self.last_weights: torch.Tensor | None = None            # filled when keep_weights is on
        self.keep_weights = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.qkv(x).split(C, dim=-1)                   # 3 × (B, T, C)
        # Split C into H heads of d, then move the head axis next to batch so
        # each head is an independent attention over (T, d).
        q = q.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        k = k.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        v = v.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        out, w = attention(q, k, v, causal=self.causal, scale=self.scale,
                           return_weights=True)                  # (B, H, T, d), (B, H, T, T)
        if self.keep_weights:
            self.last_weights = w.detach()
        # Undo the split: heads back next to each other along C.
        # transpose makes the tensor non-contiguous, so reshape (which copies
        # when it must) rather than view (which would raise).
        out = out.transpose(1, 2).reshape(B, T, C)               # (B, H, T, d) -> (B, T, C)
        return self.proj(out)                                    # (B, T, C)
