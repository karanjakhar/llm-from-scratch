"""Stage 07: turning logits into text. docs/07_generation.md is the lesson.

Every knob is a separate function that takes logits (B, V) and returns
logits (B, V), so they compose in a fixed order and each can be tested alone:

    repetition penalty -> temperature -> top-k -> top-p -> softmax -> draw

The model is llmfs.model.gpt.GPT (Stage 06), whose forward takes an optional
KVCache (docs §6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.model.gpt import GPT, KVCache  # noqa: E402

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# The knobs (docs §3-§5). All take and return logits of shape (B, V).
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    # Softmax only sees differences between logits; dividing by τ scales
    # every difference at once (docs §3). τ = 0 is handled as greedy upstream.
    return logits / tau


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]         # (B, 1) the k-th largest
    return logits.masked_fill(logits < kth, NEG_INF)             # ties at the k-th all survive


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of top tokens whose probability sum reaches p."""
    if p >= 1.0:
        return logits
    sorted_logits, order = logits.sort(dim=-1, descending=True)  # (B, V)
    probs = sorted_logits.softmax(-1)
    cum = probs.cumsum(-1)                                       # running total
    # Drop a token if the tokens ABOVE it already reach p. Comparing the sum
    # *before* this token (cum - probs) keeps the token that crosses p, so a
    # single 0.95 token survives p = 0.9 (docs §4).
    drop_sorted = (cum - probs) >= p                             # (B, V) in sorted order
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)  # back to vocab order
    return logits.masked_fill(drop, NEG_INF)


def apply_repetition_penalty(logits: torch.Tensor, context: torch.Tensor,
                             penalty: float) -> torch.Tensor:
    """CTRL's penalty (docs §5). context: (B, T) ids already in the sequence."""
    if penalty == 1.0:
        return logits
    seen = torch.gather(logits, 1, context)                      # (B, T) logits of seen tokens
    # Divide positives, multiply negatives: both move the logit DOWN.
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(1, context, seen)


def sample_next(logits: torch.Tensor, context: torch.Tensor | None = None,
                temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
                repetition_penalty: float = 1.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) for the next position -> ids (B, 1)."""
    logits = logits.float()
    if context is not None:
        logits = apply_repetition_penalty(logits, context, repetition_penalty)
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)                   # greedy
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = logits.softmax(-1)                                   # (B, V), -inf -> 0
    return torch.multinomial(probs, 1, generator=generator)      # (B, 1)


# ---------------------------------------------------------------------------
# The loop, two ways (docs §1, §6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_naive(model: GPT, ids: torch.Tensor, max_new_tokens: int, eot: int | None = None,
                   seed: int | None = None, **knobs) -> torch.Tensor:
    """Re-run the whole sequence every step. O(N²) positions processed."""
    g = _gen(ids.device, seed)
    for _ in range(max_new_tokens):
        logits, _ = model(ids, last_only=True)                   # (B, 1, V)
        nxt = sample_next(logits[:, -1], ids, generator=g, **knobs)
        ids = torch.cat([ids, nxt], dim=1)                       # (B, T+1)
        if eot is not None and (nxt == eot).all():
            break
    return ids


@torch.no_grad()
def generate_cached(model: GPT, ids: torch.Tensor, max_new_tokens: int, eot: int | None = None,
                    seed: int | None = None, cache_dtype=None, **knobs) -> torch.Tensor:
    """Prefill the prompt into a KV cache, then feed one token per step."""
    g = _gen(ids.device, seed)
    B, T = ids.shape
    dtype = cache_dtype or next(model.parameters()).dtype
    # model.make_cache picks the right cache for the architecture (GPT, GQA, MLA)
    cache = model.make_cache(B, T + max_new_tokens, ids.device, dtype)
    logits, _ = model(ids, cache=cache, last_only=True)          # prefill: (B, 1, V)
    out = ids
    for i in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], out, generator=g, **knobs)
        out = torch.cat([out, nxt], dim=1)
        if (eot is not None and (nxt == eot).all()) or i == max_new_tokens - 1:
            break
        logits, _ = model(nxt, cache=cache)                      # decode: one token in, (B, 1, V) out
    return out


def _gen(device, seed):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
