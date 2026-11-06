"""Llama-class model with every upgrade as a switch. docs/12_modern_arch.md.

Starting from GPT (Stage 06), each Stage 12 change is one config field, so the
experiments can add them one at a time:

    norm       "layer" -> "rms"                       (Stage 05 §3)
    pos        "learned" -> "rope"                    (Stage 03 §4)
    mlp        "gelu" -> "swiglu"                     (Stage 05 §6)
    n_kv_head  n_head (MHA) -> fewer (GQA) -> 1 (MQA) (docs §2)
    qk_norm    RMSNorm on q and k per head            (docs §5)
    bias       linear biases on/off
    window     sliding-window attention span, 0 = full (docs §4)

With norm="rms", pos="rope", mlp="swiglu", bias=False, n_kv_head < n_head and
an untied head this is the Llama 2/3 architecture; test_llama.py checks it
against HF LlamaForCausalLM with copied weights.

Same interface as llmfs.model.gpt.GPT: forward(idx, targets, cache, last_only),
configure_optimizer, num_params, cfg.to_dict(). So the Stage 08 trainer and
the Stage 07 sampler work on it unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 8196
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int | None = None      # None = n_head (MHA)
    n_embd: int = 384
    norm: str = "rms"
    pos: str = "rope"
    mlp: str = "swiglu"
    mlp_hidden: int | None = None     # None: 4C for gelu, ≈8C/3 rounded to 64 for swiglu
    qk_norm: bool = False
    bias: bool = False
    tie: bool = True
    rope_theta: float = 10000.0
    rope_scale: float = 1.0           # >1: NTK-aware context extension (docs §3)
    window: int = 0
    norm_eps: float = 1e-5
    scaled_init: bool = True
    arch: str = "llama"               # lets a checkpoint say which class to rebuild
    # Mixture of Experts (Stage 13). n_experts = 0 keeps the dense MLP.
    n_experts: int = 0
    top_k: int = 2
    expert_hidden: int | None = None  # None: hidden // top_k (active FLOPs match dense)
    n_shared: int = 0
    capacity_factor: float = 0.0      # 0 = never drop tokens
    router_noise: float = 0.0
    aux_coef: float = 0.01            # load-balancing loss weight (0 = off)
    z_coef: float = 0.001             # router z-loss weight
    moe_every: int = 1                # MoE in every k-th block (1 = all)
    # Multi-head latent attention (Stage 14). attn = "mla" caches a rank-`kv_latent`
    # latent plus a `rope_dim` shared RoPE key per token instead of full k, v.
    attn: str = "gqa"                 # "gqa" (covers MHA/MQA) | "mla"
    kv_latent: int = 64
    rope_dim: int = 16

    def to_dict(self):
        return asdict(self)

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def hidden(self) -> int:
        if self.mlp_hidden:
            return self.mlp_hidden
        if self.mlp == "swiglu":
            return 64 * math.ceil(8 * self.n_embd / 3 / 64)
        return 4 * self.n_embd


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def make_norm(cfg, dim):
    return RMSNorm(dim, cfg.norm_eps) if cfg.norm == "rms" else nn.LayerNorm(dim, eps=cfg.norm_eps)


def rope_cache(T: int, d: int, theta: float, scale: float, device):
    """cos, sin of shape (T, d/2) in the "half" layout HF Llama uses (Stage 03 §4).

    NTK-aware scaling (docs §3): raise the base so the slowest clocks stretch
    by `scale` while the fastest barely change. theta' = theta · scale^(d/(d-2)).
    """
    if scale != 1.0:
        theta = theta * scale ** (d / (d - 2))
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device).float() / d))     # (d/2,)
    ang = torch.arange(T, device=device).float()[:, None] * inv[None]           # (T, d/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x (B, H, T, d), rotate pairs (i, i + d/2)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c, s = cos[None, None], sin[None, None]                     # (1, 1, T, d/2)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).type_as(x)


class KVCache:
    """Like gpt.KVCache, but stores only n_kv_head heads: GQA's whole point (docs §2)."""

    def __init__(self, cfg: LlamaConfig, B: int, max_len: int, device, dtype):
        shape = (cfg.n_layer, B, cfg.kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def nbytes(self):
        return 2 * self.k.numel() * self.k.element_size()


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        H, Hk, d, C = cfg.n_head, cfg.kv_heads, cfg.head_dim, cfg.n_embd
        assert H % Hk == 0, "query heads must split evenly into kv groups"
        self.q_proj = nn.Linear(C, H * d, bias=cfg.bias)
        self.k_proj = nn.Linear(C, Hk * d, bias=cfg.bias)       # fewer heads: smaller k, v
        self.v_proj = nn.Linear(C, Hk * d, bias=cfg.bias)
        self.o_proj = nn.Linear(H * d, C, bias=cfg.bias)
        self.o_proj.writes_residual = True
        if cfg.qk_norm:
            self.q_norm = RMSNorm(d, cfg.norm_eps)
            self.k_norm = RMSNorm(d, cfg.norm_eps)

    def forward(self, x, rope, cache=None):
        cfg = self.cfg
        B, T, C = x.shape
        H, Hk, d = cfg.n_head, cfg.kv_heads, cfg.head_dim
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)      # (B, H, T, d)
        k = self.k_proj(x).view(B, T, Hk, d).transpose(1, 2)     # (B, Hk, T, d)
        v = self.v_proj(x).view(B, T, Hk, d).transpose(1, 2)
        if cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        p0 = 0
        if cache is not None:
            p0 = cache.pos
            cache.k[self.layer, :, :, p0:p0 + T] = k
            cache.v[self.layer, :, :, p0:p0 + T] = v
            k = cache.k[self.layer, :, :, :p0 + T]
            v = cache.v[self.layer, :, :, :p0 + T]
        S = k.shape[2]                                           # keys visible: p0 + T
        if Hk != H:
            # Each group of H/Hk query heads shares one k, v head. Repeat the
            # kv heads so the shapes line up; the cache stays small.
            k = k.repeat_interleave(H // Hk, dim=1)              # (B, H, S, d)
            v = v.repeat_interleave(H // Hk, dim=1)
        if cfg.window or (cache is not None and T > 1 and p0 > 0):
            qi = torch.arange(p0, p0 + T, device=x.device)[:, None]
            kj = torch.arange(S, device=x.device)[None]
            mask = kj <= qi                                      # causal
            if cfg.window:
                mask = mask & (kj > qi - cfg.window)             # only the last `window` keys
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif cache is not None and T == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * d))


class MLACache:
    """MLA caches the latent c (r numbers) and one shared RoPE key (d_r numbers) per token."""

    def __init__(self, cfg: LlamaConfig, B: int, max_len: int, device, dtype):
        self.c = torch.zeros(cfg.n_layer, B, max_len, cfg.kv_latent, device=device, dtype=dtype)
        self.kr = torch.zeros(cfg.n_layer, B, max_len, cfg.rope_dim, device=device, dtype=dtype)
        self.pos = 0

    def nbytes(self):
        return self.c.numel() * self.c.element_size() + self.kr.numel() * self.kr.element_size()


class MLAttention(nn.Module):
    """Multi-head latent attention, DeepSeek-V2 (docs/14_attention_variants.md §2).

    k and v are rebuilt per head from a shared low-rank latent c = x·W_down.
    RoPE cannot go through the latent (a rotation depends on position, and
    the up-projection would have to commute with it), so position travels in a
    separate small key k_rope, shared by all heads, concatenated to each
    head's content key. The query gets a matching rope part per head.
    """

    def __init__(self, cfg: LlamaConfig, layer: int):
        super().__init__()
        self.cfg, self.layer = cfg, layer
        C, H, d, r, dr = cfg.n_embd, cfg.n_head, cfg.head_dim, cfg.kv_latent, cfg.rope_dim
        self.q_proj = nn.Linear(C, H * (d + dr), bias=False)
        self.kv_down = nn.Linear(C, r, bias=False)              # the compression: C -> r
        self.kv_norm = RMSNorm(r, cfg.norm_eps)
        self.k_up = nn.Linear(r, H * d, bias=False)             # latent -> per-head content keys
        self.v_up = nn.Linear(r, H * d, bias=False)             # latent -> per-head values
        self.k_rope = nn.Linear(C, dr, bias=False)              # shared positional key
        self.o_proj = nn.Linear(H * d, C, bias=False)
        self.o_proj.writes_residual = True

    def forward(self, x, rope, cache=None):
        cfg = self.cfg
        B, T, C = x.shape
        H, d, dr = cfg.n_head, cfg.head_dim, cfg.rope_dim
        q = self.q_proj(x).view(B, T, H, d + dr).transpose(1, 2)        # (B, H, T, d+dr)
        q_c, q_r = q[..., :d], q[..., d:]
        c = self.kv_norm(self.kv_down(x))                                # (B, T, r)
        kr = self.k_rope(x)                                              # (B, T, dr)
        cos, sin = rope
        q_r = apply_rope(q_r, cos, sin)                                  # rope tables are (T, dr/2) for MLA
        kr = apply_rope(kr[:, None], cos, sin)[:, 0]                     # (B, T, dr)
        p0 = 0
        if cache is not None:
            p0 = cache.pos
            cache.c[self.layer, :, p0:p0 + T] = c
            cache.kr[self.layer, :, p0:p0 + T] = kr
            c = cache.c[self.layer, :, :p0 + T]
            kr = cache.kr[self.layer, :, :p0 + T]
        S = c.shape[1]
        k_c = self.k_up(c).view(B, S, H, d).transpose(1, 2)             # (B, H, S, d)
        v = self.v_up(c).view(B, S, H, d).transpose(1, 2)
        k = torch.cat([k_c, kr[:, None].expand(B, H, S, dr)], dim=-1)    # (B, H, S, d+dr)
        q = torch.cat([q_c, q_r], dim=-1)
        if cache is not None and p0 > 0 and T > 1:
            mask = torch.arange(S, device=x.device)[None] <= torch.arange(p0, p0 + T, device=x.device)[:, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif cache is not None and T == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * d))


class MLP(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        C, Hd = cfg.n_embd, cfg.hidden
        self.swiglu = cfg.mlp == "swiglu"
        if self.swiglu:
            self.gate_proj = nn.Linear(C, Hd, bias=cfg.bias)
            self.up_proj = nn.Linear(C, Hd, bias=cfg.bias)
        else:
            self.up_proj = nn.Linear(C, Hd, bias=cfg.bias)
        self.down_proj = nn.Linear(Hd, C, bias=cfg.bias)
        self.down_proj.writes_residual = True

    def forward(self, x):
        if self.swiglu:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg, layer):
        super().__init__()
        self.input_layernorm = make_norm(cfg, cfg.n_embd)
        self.self_attn = MLAttention(cfg, layer) if cfg.attn == "mla" else Attention(cfg, layer)
        self.post_attention_layernorm = make_norm(cfg, cfg.n_embd)
        if cfg.n_experts and layer % cfg.moe_every == cfg.moe_every - 1:
            from .moe import MoE
            self.mlp = MoE(cfg.n_embd, cfg.n_experts, cfg.top_k,
                           cfg.expert_hidden or cfg.hidden // cfg.top_k, cfg.n_shared,
                           cfg.capacity_factor, cfg.router_noise)
        else:
            self.mlp = MLP(cfg)

    def forward(self, x, rope, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), rope, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.layers = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.norm = make_norm(cfg, cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init)
        if cfg.scaled_init:
            for m in self.modules():
                if getattr(m, "writes_residual", False):
                    nn.init.normal_(m.weight, std=0.02 / math.sqrt(2 * cfg.n_layer))
        self._rope = None

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def rope(self, p0, T, device):
        if self.cfg.pos != "rope":
            return None
        need = max(p0 + T, self.cfg.block_size)
        if self._rope is None or self._rope[0].shape[0] < need or self._rope[0].device != device:
            d = self.cfg.rope_dim if self.cfg.attn == "mla" else self.cfg.head_dim
            self._rope = rope_cache(need, d, self.cfg.rope_theta, self.cfg.rope_scale, device)
        cos, sin = self._rope
        return cos[p0:p0 + T], sin[p0:p0 + T]

    # kept so llmfs.train.trainer's activation-checkpoint wrapper finds the blocks
    @property
    def transformer(self):
        return _Shim(self)

    def forward(self, idx, targets=None, cache=None, last_only=False):
        B, T = idx.shape
        p0 = cache.pos if cache is not None else 0
        x = self.embed_tokens(idx)                               # (B, T, C)
        if self.wpe is not None:
            assert p0 + T <= self.cfg.block_size
            x = x + self.wpe(torch.arange(p0, p0 + T, device=idx.device))
        rope = self.rope(p0, T, idx.device)
        for blk in self.layers:
            x = blk(x, rope, cache)
        if cache is not None:
            cache.pos += T
        x = self.norm(x)
        if last_only:
            x = x[:, -1:]
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
            if self.cfg.n_experts:
                moes = self.moe_layers()
                self.last_aux = sum(m.aux_loss for m in moes) / len(moes)
                self.last_z = sum(m.z_loss for m in moes) / len(moes)
                self.last_lm_loss = loss.detach()
                loss = loss + self.cfg.aux_coef * self.last_aux + self.cfg.z_coef * self.last_z
        return logits, loss

    def hidden(self, idx):
        """Final-norm hidden states (B, T, C), for value / reward heads (Stage 18)."""
        B, T = idx.shape
        x = self.embed_tokens(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        rope = self.rope(0, T, idx.device)
        for blk in self.layers:
            x = blk(x, rope)
        return self.norm(x)

    def make_cache(self, B, max_len, device, dtype):
        cls = MLACache if self.cfg.attn == "mla" else KVCache
        return cls(self.cfg, B, max_len, device, dtype)

    def moe_layers(self):
        from .moe import MoE
        return [m for m in self.modules() if isinstance(m, MoE)]

    def active_params(self) -> int:
        """Parameters used per token: shared stuff + top_k of each MoE layer's experts."""
        n = self.num_params()
        for m in self.moe_layers():
            per = sum(p.numel() for p in m.experts[0].parameters())
            n -= (m.E - m.k) * per
        return n

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel() + (self.wpe.weight.numel() if self.wpe is not None else 0)
            if not self.cfg.tie:
                n -= self.lm_head.weight.numel()
        return n

    def configure_optimizer(self, lr, weight_decay=0.1, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=lr, betas=betas, fused=torch.cuda.is_available())


class _Shim:
    """Exposes .h so code written for GPT (trainer checkpointing) can swap blocks."""

    def __init__(self, m):
        self.m = m

    @property
    def h(self):
        return self.m.layers

    @h.setter
    def h(self, v):
        self.m.layers = v


def kv_bytes_per_token(cfg: LlamaConfig, bytes_per=2) -> int:
    if cfg.attn == "mla":
        return cfg.n_layer * (cfg.kv_latent + cfg.rope_dim) * bytes_per
    return 2 * cfg.n_layer * cfg.kv_heads * cfg.head_dim * bytes_per
