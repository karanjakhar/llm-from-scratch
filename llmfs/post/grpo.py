"""Group Relative Policy Optimization. docs/20_reasoning_rlvr.md §4.

For each prompt, sample G completions from the current policy, score each
with a programmatic reward (do the unit tests pass?), and use the group's own
statistics as the baseline:

    A_i = (R_i - mean(R)) / (std(R) + ε)            no value network needed

Then a PPO-style clipped update on the completion tokens, plus a KL penalty
to a frozen reference policy (DeepSeekMath, 2024):

    ratio_t = π(o_t) / π_old(o_t)
    L = -mean_t [ min(ratio_t A, clip(ratio_t, 1-ε, 1+ε) A) ] + β · KL(π || π_ref)
    KL estimator per token (k3, Schulman): π_ref/π - log(π_ref/π) - 1   (≥ 0, unbiased)
"""

from __future__ import annotations

import torch

IGNORE = -100


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards (P, G) -> advantages (P, G). A group with identical rewards gives 0."""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def token_logprobs(model, x, y):
    """(B, T) log π(y_t | x_<=t) at completion positions (others 0), and the mask."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    lp = torch.log_softmax(logits.float(), -1)
    mask = (y != IGNORE).float()
    return lp.gather(2, y.clamp(min=0)[..., None])[..., 0] * mask, mask


def grpo_loss(logp, old_logp, ref_logp, mask, adv, clip=0.2, beta=0.04):
    """logp/old/ref/mask (B, T); adv (B,). Per-sequence token-mean, then batch mean."""
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    d = ref_logp - logp
    kl = torch.exp(d) - d - 1                                    # k3 estimator, per token
    per_tok = -(surr - beta * kl) * mask
    per_seq = per_tok.sum(1) / mask.sum(1).clamp(min=1)
    stats = dict(kl=((kl * mask).sum() / mask.sum()).item(),
                 clipped=(((ratio - 1).abs() > clip).float() * mask).sum().item() / mask.sum().item())
    return per_seq.mean(), stats
