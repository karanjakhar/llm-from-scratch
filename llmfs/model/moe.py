"""Mixture-of-Experts feed-forward layer. docs/13_moe.md.

Replaces the dense SwiGLU MLP of a Llama block with E small SwiGLU experts and
a router. Each token is sent to its top-k experts; their outputs are summed,
weighted by the router's probabilities.

    x (N, C) ─router─> logits (N, E) ─softmax─> probs (N, E)
                         top-k ─> experts e₁..e_k, weights w₁..w_k
    y = Σ_j w_j · expert_{e_j}(x)        (+ shared experts, always on)

Losses returned for the trainer to add (docs §4-§5):
    aux    = E · Σ_i f_i · P_i     load balancing (Switch Transformer)
    z      = mean(logsumexp(logits)²)   router z-loss (ST-MoE)

The dispatch below loops over experts and gathers each expert's tokens. That
is the readable version; real systems sort tokens by expert and run one
grouped matmul, and across GPUs an all-to-all moves tokens to the expert's
device (docs §7).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, C: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(C, hidden, bias=False)
        self.up_proj = nn.Linear(C, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, C, bias=False)
        self.down_proj.writes_residual = True

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, C: int, n_experts: int, top_k: int, hidden: int, n_shared: int = 0,
                 capacity_factor: float = 0.0, router_noise: float = 0.0):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.capacity_factor = capacity_factor            # 0 = no capacity limit (no dropping)
        self.router_noise = router_noise
        self.router = nn.Linear(C, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(C, hidden) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(C, hidden) for _ in range(n_shared))
        # filled every forward, read by the trainer / experiments
        self.aux_loss = torch.tensor(0.0)
        self.z_loss = torch.tensor(0.0)
        self.last_counts: torch.Tensor | None = None     # (E,) tokens routed per expert
        self.last_dropped = 0.0

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])                      # (N, C): routing is per token
        N = x.shape[0]
        logits = self.router(x).float()                   # (N, E), fp32 for a stable softmax
        if self.training and self.router_noise:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = logits.softmax(-1)                        # (N, E)
        w, idx = probs.topk(self.k, dim=-1)               # (N, k) each
        w = w / w.sum(-1, keepdim=True)                   # renormalise over the chosen k (Mixtral)

        # --- the two auxiliary losses (docs §4, §5) ---
        onehot = F.one_hot(idx, self.E).float()           # (N, k, E)
        f = onehot.sum((0, 1)) / (N * self.k)             # fraction of assignments per expert
        P = probs.mean(0)                                 # mean router probability per expert
        self.aux_loss = self.E * (f * P).sum()            # = 1 when perfectly balanced
        self.z_loss = torch.logsumexp(logits, -1).pow(2).mean()
        counts = onehot.sum((0, 1))                       # (E,)
        self.last_counts = counts.detach()

        out = torch.zeros_like(x)
        cap = math.ceil(self.capacity_factor * N * self.k / self.E) if self.capacity_factor else N * self.k
        dropped = 0
        flat_idx = idx.reshape(-1)                        # (N·k,) expert of each assignment
        flat_w = w.reshape(-1)
        flat_tok = torch.arange(N, device=x.device).repeat_interleave(self.k)
        for e, expert in enumerate(self.experts):
            sel = (flat_idx == e).nonzero(as_tuple=True)[0]     # assignments to expert e
            if sel.numel() > cap:
                dropped += sel.numel() - cap
                sel = sel[:cap]                           # first come, first served; the rest skip this expert
            if sel.numel() == 0:
                continue
            tok = flat_tok[sel]
            y = expert(x[tok])                            # (n_e, C)
            out.index_add_(0, tok, (y * flat_w[sel, None]).to(out.dtype))
        self.last_dropped = dropped / (N * self.k)
        for s in self.shared:                             # always-on experts (DeepSeek)
            out = out + s(x)
        return out.reshape(shape)
