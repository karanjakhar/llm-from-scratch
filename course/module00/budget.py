"""Module 0: where the memory goes when you train a model.

Course exercise file (docs/COURSE.md §4, module 0). The book's version of
this arithmetic is inline in stages/00_setup/verify_env.py; this file pulls
it out into functions a learner writes and tests can check.

The rule from docs/00_setup.md §2: training in fp32 with AdamW holds, for
every parameter, the weight (4 bytes), its gradient (4) and Adam's two
running averages (4 + 4): 16 bytes. Activations come on top and depend on
the batch, not on the parameter count.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MB = 1024 ** 2

# Extra tensors each optimizer keeps per parameter, each the size of the weight.
STATE_TENSORS = {"sgd": 0, "momentum": 1, "adamw": 2}


def linear_params(d_in: int, d_out: int, bias: bool = True) -> int:
    """Parameters in nn.Linear(d_in, d_out): a (d_out, d_in) weight, plus d_out biases."""
    return d_in * d_out + (d_out if bias else 0)


def training_bytes(n_params: int, optimizer: str = "adamw", bytes_per_value: int = 4) -> int:
    """Fixed training memory: weights + gradients + optimizer state, no activations.

    adamw, fp32 -> 16 bytes per parameter; plain SGD -> 8; SGD with momentum -> 12.
    """
    copies = 2 + STATE_TENSORS[optimizer]          # weight, gradient, then the optimizer's own tensors
    return n_params * copies * bytes_per_value


def max_params(vram_bytes: int, optimizer: str = "adamw", reserve_bytes: int = 600 * MB) -> int:
    """The largest model whose fixed training memory fits in `vram_bytes`.

    `reserve_bytes` is left for the CUDA context and allocator slack (the book
    measured ~600 MB). Activations still need room on top of this answer.
    """
    usable = max(0, vram_bytes - reserve_bytes)
    return usable // training_bytes(1, optimizer)


def measure_training_bytes(model: nn.Module, x: torch.Tensor) -> dict[str, int]:
    """Run one AdamW step on the GPU and read the allocator at each point.

    Returns bytes allocated by: the weights, the backward pass (gradients plus
    any activations still held), and the first optimizer step (Adam creates
    its two state tensors lazily, on step one).
    """
    assert x.is_cuda, "the allocator counters only exist on CUDA"
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    model = model.to(x.device)
    after_weights = torch.cuda.memory_allocated()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model(x).square().mean().backward()
    after_backward = torch.cuda.memory_allocated()
    opt.step()
    after_step = torch.cuda.memory_allocated()
    return {"weights": after_weights - base, "backward": after_backward - after_weights,
            "optimizer": after_step - after_backward, "total": after_step - base}
