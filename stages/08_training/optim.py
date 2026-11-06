"""Stage 08: the optimizer and schedule, by hand. docs/08_training_loop.md §2-§6.

AdamW here is written out for reading; test_training.py checks it against
torch.optim.AdamW step for step. Real runs use torch's fused version
(llmfs/train/trainer.py), which is the same arithmetic in one kernel.
"""

from __future__ import annotations

import math

import torch


class SGD:
    """θ ← θ − lr·g. docs §2a. Kept for the comparison in run.py adamw."""

    def __init__(self, params, lr: float, momentum: float = 0.0):
        self.params = [p for p in params]
        self.lr, self.momentum = lr, momentum
        self.m = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, m in zip(self.params, self.m):
            if p.grad is None:
                continue
            if self.momentum:
                m.mul_(self.momentum).add_(p.grad)                # running direction (§2b)
                p.add_(m, alpha=-self.lr)
            else:
                p.add_(p.grad, alpha=-self.lr)


class AdamW:
    """Adam with decoupled weight decay (docs §2c-§2d).

    param_groups: list of {"params": [...], "weight_decay": float}.
    l2_in_loss=True gives the *old* coupled version: decay is added to the
    gradient and then divided by √v like everything else. Only for §8a.
    """

    def __init__(self, param_groups, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8,
                 l2_in_loss: bool = False):
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": list(param_groups), "weight_decay": 0.0}]
        self.param_groups = [dict(g, lr=lr) for g in param_groups]
        self.beta1, self.beta2 = betas
        self.eps, self.l2 = eps, l2_in_loss
        self.t = 0
        # The two state tensors per weight, same shape as the weight (§2b-c).
        self.state = {id(p): (torch.zeros_like(p), torch.zeros_like(p))
                      for g in self.param_groups for p in g["params"]}

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2, t = self.beta1, self.beta2, self.t
        bc1 = 1 - b1 ** t                                        # bias corrections (§2c)
        bc2 = 1 - b2 ** t
        for g in self.param_groups:
            lr, wd = g["lr"], g["weight_decay"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if self.l2 and wd:
                    grad = grad + wd * p                         # coupled: decay rides with g
                elif wd:
                    p.mul_(1 - lr * wd)                          # decoupled: shrink directly
                m, v = self.state[id(p)]
                m.mul_(b1).add_(grad, alpha=1 - b1)              # m ← β₁m + (1−β₁)g
                v.mul_(b2).addcmul_(grad, grad, value=1 - b2)    # v ← β₂v + (1−β₂)g²
                denom = (v / bc2).sqrt_().add_(self.eps)         # √v̂ + ε
                p.addcdiv_(m, denom, value=-lr / bc1)            # θ ← θ − lr·m̂/(√v̂+ε)

    def zero_grad(self):
        for g in self.param_groups:
            for p in g["params"]:
                p.grad = None


def lr_schedule(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup to peak, then cosine to min_ratio·peak (docs §4)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def clip_grad_norm(params, max_norm: float) -> float:
    """Global-norm clipping (docs §5). Returns the norm BEFORE clipping."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
    scale = max_norm / (norm + 1e-6)
    if scale < 1:
        for g in grads:
            g.mul_(scale)                                        # same direction, shorter
    return norm.item()
