"""Stage 15: speculative decoding. docs/15_speculative.md.

A small draft model guesses γ tokens; the big target model checks all γ in
ONE forward pass; a rejection-sampling test accepts a prefix of the guesses
so that the output distribution is exactly the target's (Leviathan et al.
2023; Chen et al. 2023).

Invariant between rounds: both models' KV caches hold every token except the
newest one, `last`. Each round feeds `last` first.
"""

from __future__ import annotations

import torch


def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """(..., V) logits -> probabilities. temperature 0 = one-hot argmax (greedy)."""
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return (logits.float() / temperature).softmax(-1)


def _draw(p: torch.Tensor, g) -> int:
    return int(torch.multinomial(p, 1, generator=g))


@torch.no_grad()
def speculative_generate(target, draft, ids: torch.Tensor, max_new_tokens: int, gamma: int = 4,
                         temperature: float = 1.0, seed: int | None = None, eot: int | None = None):
    """ids (1, T). Returns (tokens (1, T+n), stats)."""
    assert ids.shape[0] == 1, "batch 1: acceptance lengths differ per sequence"
    dev = ids.device
    g = None
    if seed is not None:
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
    T0 = ids.shape[1]
    max_len = T0 + max_new_tokens + gamma + 2
    dt = next(target.parameters()).dtype
    tc = target.make_cache(1, max_len, dev, dt)
    dc = draft.make_cache(1, max_len, dev, next(draft.parameters()).dtype)
    # prefill everything except the last prompt token
    if T0 > 1:
        target(ids[:, :-1], cache=tc)
        draft(ids[:, :-1], cache=dc)
    out = ids[0].tolist()
    stats = dict(rounds=0, drafted=0, accepted=0, target_calls=0)
    while len(out) - T0 < max_new_tokens:
        base = tc.pos                                            # == len(out) - 1
        last = out[-1]
        # 1. draft gamma tokens, one at a time
        q, d_tokens = [], []
        x = torch.tensor([[last]], device=dev)
        for _ in range(gamma):
            logits, _ = draft(x, cache=dc)
            qi = probs_from_logits(logits[0, -1], temperature)
            tkn = _draw(qi, g) if temperature > 0 else int(qi.argmax())
            q.append(qi)
            d_tokens.append(tkn)
            x = torch.tensor([[tkn]], device=dev)
        # 2. the target scores last + all drafts at once: gamma+1 distributions
        x = torch.tensor([[last] + d_tokens], device=dev)
        logits, _ = target(x, cache=tc)
        p = probs_from_logits(logits[0], temperature)           # (gamma+1, V)
        stats["target_calls"] += 1
        # 3. accept a prefix (docs §2)
        n_acc = 0
        new = []
        for i, tkn in enumerate(d_tokens):
            pi, qi = p[i, tkn], q[i][tkn]
            if temperature == 0:
                ok = pi > 0.5                                    # target's argmax is this token
            else:
                r = torch.rand((), device=dev, generator=g)
                ok = r < torch.clamp(pi / qi, max=1.0)
            if ok:
                n_acc += 1
                new.append(tkn)
                continue
            # rejected: sample from the leftover mass max(0, p - q), renormalised
            resid = torch.clamp(p[i] - q[i], min=0)
            resid = resid / resid.sum()
            new.append(_draw(resid, g) if temperature > 0 else int(p[i].argmax()))
            break
        else:
            bonus = p[gamma]                                     # all accepted: one free token
            new.append(_draw(bonus, g) if temperature > 0 else int(bonus.argmax()))
        stats["rounds"] += 1
        stats["drafted"] += gamma
        stats["accepted"] += n_acc
        # 4. roll back the caches to "everything except the newest token"
        tc.pos = base + 1 + n_acc
        if n_acc == gamma:
            # the draft never processed its own last guess; feed it so it is in sync
            draft(torch.tensor([[d_tokens[-1]]], device=dev), cache=dc)
        dc.pos = base + 1 + n_acc
        out.extend(new)
        if eot is not None and eot in new:
            out = out[:len(out) - len(new) + new.index(eot) + 1]
            break
    out = out[:T0 + max_new_tokens]
    stats["acceptance_rate"] = stats["accepted"] / max(1, stats["drafted"])
    stats["tokens_per_target_call"] = (len(out) - T0) / max(1, stats["target_calls"])
    return torch.tensor([out], device=dev), stats
