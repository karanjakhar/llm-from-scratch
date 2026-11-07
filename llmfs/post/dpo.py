"""Direct Preference Optimization. docs/19_rlhf_dpo.md §4.

    loss = -log σ( β · [ (log π(c|x) - log π_ref(c|x)) - (log π(r|x) - log π_ref(r|x)) ] )

π is the policy being trained, π_ref a frozen copy of the starting (SFT)
model, c/r the chosen/rejected completions, and log π(y|x) the SUM of the
completion's token log-probs (the prompt is masked out, as in SFT).
β(log π - log π_ref) is the "implicit reward" DPO's derivation gives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


def sequence_logprob(model, x, y) -> torch.Tensor:
    """x, y (B, T) from sft.collate; y = IGNORE on prompt/pad. -> (B,) summed log-probs."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(x)
    lp = torch.log_softmax(logits.float(), -1)
    mask = y != IGNORE
    tok_lp = lp.gather(2, y.clamp(min=0)[..., None])[..., 0]    # (B, T)
    return (tok_lp * mask).sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float):
    """All (B,). Returns loss, chosen reward, rejected reward (both implicit, β-scaled)."""
    rc = beta * (pi_c - ref_c)
    rr = beta * (pi_r - ref_r)
    loss = -F.logsigmoid(rc - rr).mean()
    return loss, rc.detach(), rr.detach()
