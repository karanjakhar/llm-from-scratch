"""A minimal continuous-batching inference engine. docs/23_serving.md.

One GPU, one model (llmfs GPT), S slots. Each slot holds one request's KV
cache rows. Every engine step:

    1. admit: waiting requests take free slots; each is PREFILLED alone
       (its whole prompt in one forward pass, written into its slot);
    2. decode: ONE batched forward over every active slot, one new token each,
       each slot attending only to its own cache rows (a length mask);
    3. retire: slots whose request hit eot / max_tokens are freed at once,
       so a new request can join on the very next step.

Static batching (the baseline) instead takes up to S requests, runs them to
completion together, and only then looks at the queue again.

The cache here is one contiguous (L, S, H, T_max, d) block per slot: simple,
but it reserves T_max for every request. PagedAttention's fix is described in
docs §4 and not built.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Request:
    rid: int
    prompt: list[int]
    max_tokens: int
    arrival: float = 0.0
    out: list[int] = field(default_factory=list)
    t_first: float | None = None
    t_done: float | None = None
    slot: int | None = None


class Engine:
    def __init__(self, model, slots: int = 16, max_len: int = 1024, eot: int | None = None,
                 temperature: float = 0.0, dtype=torch.bfloat16):
        self.m = model.eval()
        cfg = model.cfg
        self.L, self.H, self.d = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head
        self.S, self.Tmax = slots, min(max_len, cfg.block_size)
        dev = next(model.parameters()).device
        self.k = torch.zeros(self.L, slots, self.H, self.Tmax, self.d, device=dev, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.len = torch.zeros(slots, dtype=torch.long, device=dev)
        self.active: dict[int, Request] = {}
        self.eot, self.temp, self.dev, self.dtype = eot, temperature, dev, dtype

    # -- the model, written against per-slot caches -----------------------

    def _forward(self, idx, slots, pos0):
        """idx (n, T) tokens for `slots` (n,), starting at positions pos0 (n,).
        Writes k, v into each slot's rows and returns last-position logits (n, V)."""
        tr = self.m.transformer
        n, T = idx.shape
        pos = pos0[:, None] + torch.arange(T, device=self.dev)            # (n, T)
        x = tr.wte(idx) + tr.wpe(pos)                                     # (n, T, C)
        span = int(pos0.max()) + T
        kpos = torch.arange(span, device=self.dev)
        # query at absolute position p may see keys 0..p of its own slot
        mask = (kpos[None, None, :] <= pos[:, :, None])[:, None]          # (n, 1, T, span)
        for li, blk in enumerate(tr.h):
            h = blk.ln_1(x)
            a = blk.attn
            q, k, v = a.c_attn(h).split(h.shape[-1], dim=-1)
            q = q.view(n, T, self.H, self.d).transpose(1, 2)
            k = k.view(n, T, self.H, self.d).transpose(1, 2)
            q, k = a._qk(q, k)                                            # QK-norm if the model has it
            k = k.to(self.dtype)
            v = v.view(n, T, self.H, self.d).transpose(1, 2).to(self.dtype)
            for j in range(n):                                            # scatter into each slot
                s, p = int(slots[j]), int(pos0[j])
                self.k[li, s, :, p:p + T] = k[j]
                self.v[li, s, :, p:p + T] = v[j]
            K = self.k[li, slots, :, :span]                               # (n, H, span, d)
            V = self.v[li, slots, :, :span]
            o = F.scaled_dot_product_attention(q.to(self.dtype), K, V, attn_mask=mask)
            x = x + a.c_proj(o.transpose(1, 2).reshape(n, T, -1).to(x.dtype))
            x = x + blk.mlp(blk.ln_2(x))
        x = tr.ln_f(x[:, -1])
        return self.m.lm_head(x)                                          # (n, V)

    def _pick(self, logits):
        if self.temp == 0:
            return logits.argmax(-1)
        return torch.multinomial((logits.float() / self.temp).softmax(-1), 1)[:, 0]

    # -- scheduling -------------------------------------------------------

    def free_slots(self):
        used = {r.slot for r in self.active.values()}
        return [s for s in range(self.S) if s not in used]

    @torch.no_grad()
    def admit(self, req: Request, now):
        s = self.free_slots()[0]
        req.slot = s
        prompt = req.prompt[-(self.Tmax - req.max_tokens):]
        with torch.autocast("cuda", dtype=self.dtype):
            logits = self._forward(torch.tensor([prompt], device=self.dev),
                                   torch.tensor([s], device=self.dev), torch.tensor([0], device=self.dev))
        self.len[s] = len(prompt)
        tok = int(self._pick(logits)[0])
        req.out.append(tok)
        req.t_first = now()
        self.active[req.rid] = req

    @torch.no_grad()
    def decode_step(self, now) -> list[Request]:
        if not self.active:
            return []
        reqs = list(self.active.values())
        slots = torch.tensor([r.slot for r in reqs], device=self.dev)
        idx = torch.tensor([[r.out[-1]] for r in reqs], device=self.dev)
        with torch.autocast("cuda", dtype=self.dtype):
            logits = self._forward(idx, slots, self.len[slots])
        self.len[slots] += 1
        toks = self._pick(logits).tolist()
        done = []
        t = now()
        for r, tk in zip(reqs, toks):
            r.out.append(tk)
            if len(r.out) >= r.max_tokens or tk == self.eot or self.len[r.slot] + 1 >= self.Tmax:
                r.t_done = t
                done.append(r)
                del self.active[r.rid]
        return done


def serve(engine: Engine, requests: list[Request], policy: str = "continuous") -> dict:
    """Replay `requests` (arrival times in seconds from start) against the engine."""
    queue = sorted(requests, key=lambda r: r.arrival)
    t0 = time.perf_counter()

    def now():
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    finished, i = [], 0
    while len(finished) < len(requests):
        t = now()
        if policy == "continuous":
            while i < len(queue) and queue[i].arrival <= t and engine.free_slots():
                engine.admit(queue[i], now)
                i += 1
        else:                                                    # static: only when the batch is empty
            if not engine.active:
                while i < len(queue) and queue[i].arrival <= t and engine.free_slots():
                    engine.admit(queue[i], now)
                    i += 1
        if not engine.active:
            if i < len(queue):
                time.sleep(max(0.0, min(0.002, queue[i].arrival - t)))
            continue
        for r in engine.decode_step(now):
            finished.append(r)
    total = now()
    lat = sorted(r.t_done - r.arrival for r in finished)
    ttft = sorted(r.t_first - r.arrival for r in finished)
    toks = sum(len(r.out) for r in finished)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)]
    return dict(policy=policy, requests=len(finished), seconds=round(total, 2),
                tokens_per_s=round(toks / total, 1),
                latency_p50=round(pct(lat, 0.5), 3), latency_p99=round(pct(lat, 0.99), 3),
                ttft_p50=round(pct(ttft, 0.5), 3), ttft_p99=round(pct(ttft, 0.99), 3),
                tpot_mean=round(sum((r.t_done - r.t_first) / max(1, len(r.out) - 1) for r in finished)
                                / len(finished), 4))
