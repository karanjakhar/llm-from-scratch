"""GPT, promoted from stages/06_gpt/gpt.py for later stages to import.

Same model, same parameter names (GPT-2's), same init. Differences from the
stage version: self-contained (no sys.path imports), and later stages add to
it here (Stage 07: KV cache) rather than editing the Stage 06 record.
The explanations live in stages/06_gpt/gpt.py and docs/06_gpt_model.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8196
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    act: str = "gelu_tanh"
    tie: bool = True
    scaled_init: bool = True
    flash: bool = True            # SDPA; equal to the hand-written path to 1e-5 (Stage 06 test)
    bias: bool = True
    dropout: float = 0.0
    qk_norm: bool = False         # RMS-normalise q and k per head (Stage 10 §8f: attention-logit growth)

    def to_dict(self) -> dict:
        return asdict(self)


class LayerNorm(nn.Module):
    """Biased variance, eps 1e-5: identical to nn.LayerNorm (Stage 05 test)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class KVCache:
    """Per-layer keys and values for positions already processed (Stage 07).

    Preallocated to max_len so decoding never reallocates; `pos` is how many
    positions are filled. k, v: (L, B, H, max_len, d).
    """

    def __init__(self, cfg: GPTConfig, B: int, max_len: int, device, dtype):
        d = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, B, cfg.n_head, max_len, d)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.n_head, self.flash, self.layer = cfg.n_head, cfg.flash, layer
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.writes_residual = True
        self.dropout = cfg.dropout
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            d = cfg.n_embd // cfg.n_head
            self.q_gain = nn.Parameter(torch.ones(d))
            self.k_gain = nn.Parameter(torch.ones(d))

    def _qk(self, q, k):
        """Bound q·k: each head's q and k get rms 1 times a learned gain."""
        if not self.qk_norm:
            return q, k
        q = F.rms_norm(q, (q.shape[-1],), self.q_gain, 1e-6)
        k = F.rms_norm(k, (k.shape[-1],), self.k_gain, 1e-6)
        return q, k

    def forward(self, x, cache: KVCache | None = None):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, H, C // H).transpose(1, 2)              # (B, H, T, d)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        q, k = self._qk(q, k)
        if cache is not None:
            # Write this step's k, v after the ones already cached, then attend
            # over everything so far. New queries sit at positions p0..p0+T-1.
            p0 = cache.pos
            cache.k[self.layer, :, :, p0:p0 + T] = k
            cache.v[self.layer, :, :, p0:p0 + T] = v
            k = cache.k[self.layer, :, :, :p0 + T]               # (B, H, p0+T, d)
            v = cache.v[self.layer, :, :, :p0 + T]
            if T == 1:
                mask = None                                      # one query may see every cached key
            else:
                # query i (absolute p0+i) may see keys 0..p0+i
                mask = torch.ones(T, p0 + T, dtype=torch.bool, device=x.device).tril(diagonal=p0)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        else:
            d = q.shape[-1]
            att = (q @ k.transpose(-2, -1)) / math.sqrt(d)       # (B, H, T, T)
            att = att.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(),
                                  float("-inf"))
            out = att.softmax(-1) @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.writes_residual = True
        self.act = gelu_tanh if cfg.act == "gelu_tanh" else F.gelu

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer: int):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, layer)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        x = x + self.drop(self.attn(self.ln_1(x), cache))
        x = x + self.drop(self.mlp(self.ln_2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            h=nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer)),
            ln_f=LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie:
            self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init)
        # Activation checkpointing (Stage 09): recompute each block's activations
        # in backward instead of storing them. A runtime switch, not a config
        # field, and it changes no parameter names, so checkpoints are unaffected.
        # Post-training the 92M capstone needs it on 8 GB (journal, Session 10).
        self.grad_checkpointing = False
        if cfg.scaled_init:
            std = 0.02 / math.sqrt(2 * cfg.n_layer)
            for m in self.modules():
                if getattr(m, "writes_residual", False):
                    nn.init.normal_(m.weight, mean=0.0, std=std)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, cache: KVCache | None = None, last_only: bool = False):
        B, T = idx.shape
        p0 = cache.pos if cache is not None else 0
        assert p0 + T <= self.cfg.block_size, f"position {p0 + T} > block_size {self.cfg.block_size}"
        tr = self.transformer
        pos = torch.arange(p0, p0 + T, device=idx.device)
        x = tr.wte(idx) + tr.wpe(pos)                            # (B, T, C)
        for block in tr.h:
            x = self._block(block, x, cache)
        if cache is not None:
            cache.pos += T
        x = tr.ln_f(x)
        if last_only:
            x = x[:, -1:]                                        # generation needs only the last position
        logits = self.lm_head(x)                                 # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss

    def hidden(self, idx):
        """Final-norm hidden states (B, T, C), for value / reward heads (Stage 18)."""
        B, T = idx.shape
        tr = self.transformer
        x = tr.wte(idx) + tr.wpe(torch.arange(T, device=idx.device))
        for block in tr.h:
            x = self._block(block, x)
        return tr.ln_f(x)

    def _block(self, block, x, cache=None):
        if self.grad_checkpointing and cache is None and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
        return block(x, cache)

    def make_cache(self, B, max_len, device, dtype):
        return KVCache(self.cfg, B, max_len, device, dtype)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        return n

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)):
        """AdamW with decay on matrices only (biases, norms: no decay). Stage 08 §3."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                  {"params": no_decay, "weight_decay": 0.0}],
                                 lr=lr, betas=betas, fused=torch.cuda.is_available())

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "GPT":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sizes = {"gpt2": (12, 12, 768), "gpt2-medium": (24, 16, 1024),
                 "gpt2-large": (36, 20, 1280), "gpt2-xl": (48, 25, 1600)}
        L, H, C = sizes[name]
        model = cls(GPTConfig(vocab_size=50257, block_size=1024, n_layer=L, n_head=H, n_embd=C))
        theirs = load_file(hf_hub_download(f"openai-community/{name}", "model.safetensors"))
        ours = model.state_dict()
        filled = set()
        for k, v in theirs.items():
            if k.endswith((".attn.bias", ".attn.masked_bias")) or k == "lm_head.weight":
                continue
            n = k if k.startswith("transformer.") else "transformer." + k
            if n.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight")):
                v = v.T                                          # Conv1D stores (in, out)
            assert v.shape == ours[n].shape, (k, v.shape, ours[n].shape)
            with torch.no_grad():
                ours[n].copy_(v)
            filled.add(n)
        missing = set(ours) - filled - {"lm_head.weight"}
        assert not missing, missing
        return model


def save_checkpoint(path, model: GPT, **extra):
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu", **cfg_overrides) -> tuple[GPT, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GPTConfig(**{**ck["config"], **cfg_overrides})
    model = GPT(cfg)
    sd = ck["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}  # torch.compile prefix
    model.load_state_dict(sd)
    return model.to(device), ck
