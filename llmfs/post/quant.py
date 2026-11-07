"""Weight quantization by hand. docs/21_distill_quantize.md §4-§6.

    q = clamp(round(w / s) + z, qmin, qmax)        quantize
    w ≈ (q - z) · s                                dequantize

scale s and zero-point z are chosen per tensor, per output channel (row),
or per group of `group` consecutive weights in a row. Symmetric: z = 0 and
s = max|w| / qmax. Asymmetric: the range [min, max] is mapped onto
[qmin, qmax] exactly.

QuantLinear stores integer codes + scales and dequantizes in forward
("weight-only" quantization, as GPTQ/AWQ deployments do for decoding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def qrange(bits: int, symmetric: bool):
    if symmetric:
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1         # int8: -127..127 (keep 0 exact, symmetric)
    return 0, 2 ** bits - 1                                       # uint8: 0..255


def quantize(w: torch.Tensor, bits: int = 8, granularity: str = "channel", group: int = 128,
             symmetric: bool = True):
    """w (out, in) -> (codes int8/uint8 (out, in), scale, zero) with scale/zero broadcastable."""
    out, inn = w.shape
    W = w.detach().float()
    if granularity == "tensor":
        view = W.reshape(1, -1)
    elif granularity == "channel":
        view = W
    elif granularity == "group":
        assert inn % group == 0
        view = W.reshape(out * inn // group, group)
    else:
        raise ValueError(granularity)
    qmin, qmax = qrange(bits, symmetric)
    if symmetric:
        s = view.abs().amax(1, keepdim=True).clamp(min=1e-12) / qmax
        z = torch.zeros_like(s)
    else:
        lo, hi = view.amin(1, keepdim=True), view.amax(1, keepdim=True)
        s = ((hi - lo) / (qmax - qmin)).clamp(min=1e-12)
        z = torch.round(qmin - lo / s)
    q = torch.clamp(torch.round(view / s) + z, qmin, qmax)
    dtype = torch.int8 if symmetric else torch.uint8
    return q.to(dtype).reshape(out, inn), s, z


def dequantize(q, s, z, shape, granularity: str, group: int = 128):
    out, inn = shape
    if granularity == "tensor":
        view = q.reshape(1, -1).float()
    elif granularity == "channel":
        view = q.float()
    else:
        view = q.reshape(out * inn // group, group).float()
    return ((view - z) * s).reshape(out, inn)


def fake_quant(w, **kw):
    q, s, z = quantize(w, **kw)
    return dequantize(q, s, z, w.shape, kw.get("granularity", "channel"), kw.get("group", 128))


class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, bits=8, granularity="channel", group=128, symmetric=True):
        super().__init__()
        q, s, z = quantize(lin.weight, bits, granularity, group, symmetric)
        self.bits, self.gran, self.group = bits, granularity, group
        self.shape = tuple(lin.weight.shape)
        if bits == 4:                                             # pack two 4-bit codes per byte
            u = (q.to(torch.int16) - qrange(4, symmetric)[0]).to(torch.uint8).reshape(-1)
            self.offset = qrange(4, symmetric)[0]
            self.register_buffer("codes", (u[0::2] << 4) | u[1::2])
        else:
            self.offset = 0
            self.register_buffer("codes", q)
        self.register_buffer("scale", s.to(torch.float16))
        self.register_buffer("zero", z.to(torch.float16))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def weight(self, dtype=torch.float32):
        if self.bits == 4:
            u = torch.stack([self.codes >> 4, self.codes & 15], -1).reshape(self.shape)
            q = u.to(torch.int16) + self.offset
        else:
            q = self.codes
        return dequantize(q, self.scale.float(), self.zero.float(), self.shape, self.gran, self.group).to(dtype)

    def forward(self, x):
        return F.linear(x, self.weight(x.dtype), self.bias)

    def nbytes(self):
        return sum(b.numel() * b.element_size() for b in (self.codes, self.scale, self.zero))


def quantize_model(model: nn.Module, skip=("lm_head",), **kw) -> nn.Module:
    """Replace every nn.Linear (except `skip`) with QuantLinear. Embeddings stay bf16."""
    for name, m in list(model.named_modules()):
        for cname, child in list(m.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and not any(full.endswith(s) for s in skip):
                setattr(m, cname, QuantLinear(child, **kw))
    return model


def model_bytes(model: nn.Module) -> int:
    """Bytes as stored: params at their dtype (shared tensors once) + quant buffers."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel() * p.element_size()
    for m in model.modules():
        if isinstance(m, QuantLinear):
            total += m.nbytes()
    return total
