"""Why did the 4-layer, C=256 runs diverge in Stage 10? docs/10_scaling_laws.md §8f.

    uv run python stages/10_scaling/diagnose.py

Same recipe as the IsoFLOP runs (T=512, B=16, lr 1e-3, warmup 200, cosine,
clip 1.0, compiled), 13,000 steps, with probes every 250 steps on one fixed
batch: the largest output logit, the largest attention score (q·k/√d) per
layer, the token-embedding norm, and which parameter carries the largest
gradient. Then the same run with one change at a time.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from llmfs.data import MixedLoader  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "10_scaling"
BINS = REPO / "data" / "bins" / "split_8192"

BASE = dict(model=dict(n_layer=4, n_head=4, n_embd=256, block_size=512), B=16, T=512, steps=13000, lr=1e-3,
            warmup=200, eval_every=3250, eval_windows=128, compile=True)
VARIANTS = {
    "as_isoflop": {},
    "lr_5e-4": dict(lr=5e-4),
    "untied": dict(model=dict(BASE["model"], tie=False)),
    "B32_T256": dict(B=32, T=256),
    "qk_norm": dict(model=dict(BASE["model"], qk_norm=True)),
}


@torch.no_grad()
def probe(model, x):
    raw = getattr(model, "_orig_mod", model)
    tr = raw.transformer
    h = tr.wte(x) + tr.wpe(torch.arange(x.shape[1], device=x.device))
    att_max = []
    for blk in tr.h:
        a = blk.ln_1(h)
        q, k, _ = blk.attn.c_attn(a).split(a.shape[-1], -1)
        H = blk.attn.n_head
        B, T, C = q.shape
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        q, k = blk.attn._qk(q, k)
        att_max.append(round((q @ k.transpose(-1, -2) / math.sqrt(C // H)).abs().max().item(), 1))
        h = blk(h)
    logits = raw.lm_head(tr.ln_f(h))
    return dict(logit_max=round(logits.abs().max().item(), 1), att_score_max=att_max,
                wte_norm=round(tr.wte.weight.norm().item(), 2),
                lnf_gain_max=round(tr.ln_f.weight.abs().max().item(), 2))


def run(name, over):
    cfg = {**BASE, **over}
    torch.backends.cuda.matmul.allow_tf32 = True
    x, _ = MixedLoader({"code": (BINS / "val_code.bin", 1.0)}, B=4, T=cfg["T"], device="cuda", seed=9).next_batch()
    trace = []

    def hook(step, rec, model):
        if step % 250 == 0:
            raw = getattr(model, "_orig_mod", model)
            # gradients are already zeroed after opt.step; use the recorded total norm,
            # and find the largest per-parameter grad norm on a fresh probe batch
            p = probe(model, x)
            p.update(step=step, loss=rec["loss"], gnorm=rec["gnorm"], lr=rec["lr"])
            trace.append(p)

    r = train(TrainConfig(name=name, **cfg), on_step=hook)
    # which parameter carries the gradient, at the end
    raw = getattr(r["model"], "_orig_mod", r["model"])
    raw.train()
    xx, yy = MixedLoader({"code": (BINS / "train_code.bin", 1.0)}, B=8, T=cfg["T"], device="cuda", seed=3).next_batch()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = raw(xx, yy)
    loss.backward()
    gn = sorted(((n, p.grad.norm().item()) for n, p in raw.named_parameters() if p.grad is not None),
                key=lambda t: -t[1])[:6]
    out = dict(name=name, over={k: v for k, v in over.items()}, final=r["final"], trace=trace,
               top_grad_params=[(n, round(g, 3)) for n, g in gn])
    print(name, r["final"].get("val"), "max gnorm", max(t["gnorm"] for t in trace), gn[:3], flush=True)
    return out


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    p = RESULTS / "diagnose.json"
    res = json.loads(p.read_text()) if p.exists() else {}
    for name in only:
        res[name] = run(name, VARIANTS[name])
        p.write_text(json.dumps(res, indent=1, default=str))
