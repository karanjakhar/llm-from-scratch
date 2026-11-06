"""Stage 14: attention without the T×T matrix. docs/14_attention_variants.md §3-§4.

Two ideas, each written in its two equivalent forms so the duality can be
checked numerically (test_variants.py):

  linear attention   parallel (T×T, masked)  ==  recurrent (a d×d state)
  diagonal SSM       recurrent scan          ==  parallel (cumulative products)

MLA (§2) lives in llmfs/model/llama.py (attn="mla") because it is trained.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def phi(x: torch.Tensor) -> torch.Tensor:
    """Feature map that keeps scores positive: elu(x) + 1 (Katharopoulos et al. 2020)."""
    return F.elu(x) + 1


def linear_attention_parallel(q, k, v):
    """q, k, v: (B, T, d). Replace exp(q·k) with φ(q)·φ(k); still T×T here."""
    fq, fk = phi(q), phi(k)                                      # (B, T, d)
    scores = fq @ fk.transpose(-1, -2)                           # (B, T, T), all ≥ 0
    T = q.shape[1]
    scores = scores * torch.ones(T, T, device=q.device).tril()   # causal: zero, not -inf (no softmax)
    return (scores @ v) / scores.sum(-1, keepdim=True)           # normalise rows to sum 1


def linear_attention_recurrent(q, k, v):
    """Same output, one token at a time, with a fixed-size state.

    S_t = S_{t-1} + φ(k_t) v_tᵀ     (d, d)   everything seen so far, summarised
    z_t = z_{t-1} + φ(k_t)          (d,)     the normaliser
    out_t = φ(q_t)ᵀ S_t / φ(q_t)ᵀ z_t
    Memory per token: O(1) in T. That is the whole appeal (docs §3).
    """
    B, T, d = q.shape
    fq, fk = phi(q), phi(k)
    S = q.new_zeros(B, d, v.shape[-1])
    z = q.new_zeros(B, d)
    outs = []
    for t in range(T):
        S = S + fk[:, t, :, None] * v[:, t, None, :]             # outer product
        z = z + fk[:, t]
        num = (fq[:, t, None, :] @ S)[:, 0]                      # (B, d_v)
        den = (fq[:, t] * z).sum(-1, keepdim=True)
        outs.append(num / den)
    return torch.stack(outs, 1)


def ssm_recurrent(x, a, b, c):
    """Diagonal linear state-space model, the core of S4/Mamba (docs §4).

    h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t      per channel, a_t in (0, 1): how much to keep
    y_t = c_t ⊙ h_t
    x, a, b, c: (B, T, D). In Mamba a, b, c depend on the input ("selective").
    """
    h = torch.zeros_like(x[:, 0])
    ys = []
    for t in range(x.shape[1]):
        h = a[:, t] * h + b[:, t] * x[:, t]
        ys.append(c[:, t] * h)
    return torch.stack(ys, 1)


def ssm_parallel(x, a, b, c):
    """Same SSM, all positions at once.

    Unrolling: h_t = Σ_{s≤t} (Π_{r=s+1..t} a_r) b_s x_s.
    With A_t = Π_{r≤t} a_r, the product is A_t / A_s, so
    h_t = A_t · cumsum_s( b_s x_s / A_s ).
    Done in log space so long products do not underflow. Real kernels use a
    parallel scan (O(log T) depth) instead, which is stable for any length.
    """
    log_a = torch.log(a)
    L = torch.cumsum(log_a, dim=1)                               # log A_t
    # weight of input s at time t: exp(L_t - L_s); compute as a masked T×T matrix
    W = torch.exp(L[:, :, None, :] - L[:, None, :, :])           # (B, T, T, D) [t, s]
    T = x.shape[1]
    W = W * torch.ones(T, T, device=x.device).tril()[None, :, :, None]
    h = (W * (b * x)[:, None]).sum(2)                            # (B, T, D)
    return c * h
