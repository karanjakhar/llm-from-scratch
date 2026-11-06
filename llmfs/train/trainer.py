"""The training loop every stage from 08 on uses. docs/08_training_loop.md.

    from llmfs.train.trainer import TrainConfig, train
    hist = train(TrainConfig(model=dict(n_layer=8, n_embd=512, n_head=8), steps=5000, ...))

Pieces, each explained in the Stage 08 doc:
  AdamW (torch fused; equal to stages/08_training/optim.py), decay on matrices only;
  warmup + cosine to min_ratio; global-norm clipping; gradient accumulation;
  bf16 autocast; periodic eval on fixed val windows; JSONL log;
  checkpoints that hold model + optimizer + loader + RNG so a resumed run
  continues the same curve (Stage 11 tests this).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from llmfs.data import MixedLoader, open_bin
from llmfs.model.gpt import GPT, GPTConfig

REPO = Path(__file__).resolve().parents[2]
BINS = REPO / "data" / "bins" / "split_8192"


@dataclass
class TrainConfig:
    model: dict = field(default_factory=dict)            # GPTConfig overrides
    sources: dict = field(default_factory=lambda: {
        "code": (str(BINS / "train_code.bin"), 0.85),
        "en": (str(BINS / "train_en.bin"), 0.15)})
    val: dict = field(default_factory=lambda: {"code": str(BINS / "val_code.bin"),
                                               "en": str(BINS / "val_en.bin")})
    sampler: str = "shuffled"
    B: int = 32                  # micro-batch sequences
    T: int = 256
    accum: int = 1               # micro-batches per optimizer step
    steps: int = 4000
    lr: float = 1e-3
    warmup: int = 100
    min_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    clip: float = 1.0            # 0 disables
    eval_every: int = 500
    eval_windows: int = 256
    seed: int = 0
    dtype: str = "bf16"          # "bf16" | "fp32"
    compile: bool = False
    checkpointing: bool = False  # activation checkpointing per block (Stage 09)
    out_dir: str | None = None   # where log.jsonl and ckpt.pt go
    ckpt_every: int = 0
    resume: bool = False
    name: str = "run"
    stop_after: int | None = None    # quit early at this step (for resume tests)
    resume_drop_optimizer: bool = False   # control for Stage 11: resume with fresh Adam state

    @property
    def tokens_per_step(self) -> int:
        return self.B * self.T * self.accum


def lr_at(step: int, c: TrainConfig) -> float:
    if step < c.warmup:
        return c.lr * (step + 1) / c.warmup
    p = min(1.0, (step - c.warmup) / max(1, c.steps - c.warmup))
    return c.lr * (c.min_ratio + (1 - c.min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def make_val(paths: dict, T: int, n: int) -> dict[str, torch.Tensor]:
    out = {}
    for name, p in paths.items():
        data = open_bin(p)
        starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
        out[name] = torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts]))
    return out


@torch.no_grad()
def evaluate(model, val: dict, bs: int = 32, dtype=torch.bfloat16) -> dict[str, float]:
    model.eval()
    res = {}
    for name, w in val.items():
        tot, n = 0.0, 0
        for i in range(0, len(w), bs):
            b = w[i:i + bs].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                _, loss = model(b[:, :-1], b[:, 1:])
            tot += loss.item() * len(b)
            n += len(b)
        res[name] = round(tot / n, 4)
    model.train()
    return res


class _Ckpt(torch.nn.Module):
    """Wrap a block so its activations are recomputed in backward (Stage 09)."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args):
        return torch.utils.checkpoint.checkpoint(self.block, *args, use_reentrant=False)


def build(c: TrainConfig):
    torch.manual_seed(c.seed)
    kw = {k: v for k, v in c.model.items() if k not in ("block_size", "arch")}
    bs = max(c.T, c.model.get("block_size", c.T))
    if c.model.get("arch", "gpt") == "llama":
        from llmfs.model.llama import Llama, LlamaConfig
        model = Llama(LlamaConfig(block_size=bs, **kw)).cuda()
    else:
        model = GPT(GPTConfig(block_size=bs, **kw)).cuda()
    if c.checkpointing:
        model.transformer.h = torch.nn.ModuleList(_Ckpt(b) for b in model.transformer.h)
    opt = model.configure_optimizer(c.lr, c.weight_decay, c.betas)
    loader = MixedLoader({k: tuple(v) for k, v in c.sources.items()}, B=c.B, T=c.T,
                         device="cuda", sampler=c.sampler, seed=c.seed)
    return model, opt, loader


def save(path: Path, model, opt, loader, step: int, c: TrainConfig, hist: list):
    raw = getattr(model, "_orig_mod", model)
    sd = {k.replace(".block.", "."): v for k, v in raw.state_dict().items()}   # strip _Ckpt
    torch.save({"config": raw.cfg.to_dict(), "model": sd, "opt": opt.state_dict(),
                "loader": loader.state_dict(), "step": step, "train_config": asdict(c),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "hist": hist}, path.with_suffix(".tmp"))
    # rename is atomic: a crash mid-save leaves the previous checkpoint intact
    path.with_suffix(".tmp").replace(path)


def train(c: TrainConfig, on_step=None) -> dict:
    """Run c. Returns {"hist": [...], "final": {...}, "model": model}."""
    model, opt, loader = build(c)
    out = Path(c.out_dir) if c.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    start, hist = 0, []
    if c.resume and out and (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", map_location="cuda", weights_only=False)
        assert not c.checkpointing, "resume with activation checkpointing is not supported"
        model.load_state_dict(ck["model"])
        if not c.resume_drop_optimizer:
            opt.load_state_dict(ck["opt"])
        loader.load_state_dict(ck["loader"])
        torch.set_rng_state(ck["torch_rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        start, hist = ck["step"], ck["hist"]
        print(f"[{c.name}] resumed at step {start}", flush=True)
    if c.compile:
        # Each run in a multi-run script compiles a new model through the same
        # forward code; dynamo's per-function cache (8 entries) would fill up
        # and silently fall back to eager. Start every run clean.
        torch._dynamo.reset()
        model = torch.compile(model)
    val = make_val(c.val, c.T, c.eval_windows)
    amp = torch.bfloat16 if c.dtype == "bf16" else torch.float32
    log = open(out / "log.jsonl", "a") if out else None
    t0 = time.time()
    tok_seen = start * c.tokens_per_step
    for step in range(start, c.steps):
        lr = lr_at(step, c)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for _ in range(c.accum):
            x, y = loader.next_batch()
            with torch.autocast("cuda", dtype=amp, enabled=amp != torch.float32):
                _, loss = model(x, y)
            (loss / c.accum).backward()                          # grads add up across micro-batches
            loss_acc += loss.item() / c.accum
        gnorm = (torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip if c.clip else 1e9)
                 .item())
        opt.step()
        opt.zero_grad(set_to_none=True)
        tok_seen += c.tokens_per_step
        rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr, "gnorm": round(gnorm, 4)}
        last = step == c.steps - 1
        if (c.eval_every and step % c.eval_every == 0) or last:
            rec["val"] = evaluate(model, val, dtype=amp)
            rec["elapsed"] = round(time.time() - t0, 1)
            rec["tokens"] = tok_seen
            print(f"[{c.name}] step {step} loss {loss_acc:.4f} val {rec['val']} "
                  f"gnorm {gnorm:.2f} {rec['elapsed']}s", flush=True)
        if step % 10 == 0 or "val" in rec:
            hist.append(rec)
            if log:
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if on_step:
            on_step(step, rec, model)
        if out and c.ckpt_every and ((step + 1) % c.ckpt_every == 0 or last):
            save(out / "ckpt.pt", model, opt, loader, step + 1, c, hist)
        if c.stop_after is not None and step + 1 >= c.stop_after:
            break
        if not math.isfinite(loss_acc):
            print(f"[{c.name}] non-finite loss at step {step}, stopping", flush=True)
            break
    if log:
        log.close()
    final = hist[-1] if hist else {}
    return {"hist": hist, "final": final, "model": model,
            "seconds": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in model.parameters())}
