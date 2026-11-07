"""Stage 21 experiments. docs/21_distill_quantize.md §7.

    uv run python stages/21_distill_quant/run.py distill --teacher checkpoints/11_pretrain/workhorse/final.pt
    uv run python stages/21_distill_quant/run.py quant --model checkpoints/11_pretrain/workhorse/final.pt --tag workhorse
    uv run python stages/21_distill_quant/run.py plot --tag workhorse
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "07_generation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sample import generate_cached  # noqa: E402

from llmfs.data import MixedLoader  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.post.lora import NF4Linear  # noqa: E402
from llmfs.post.quant import model_bytes, quantize_model  # noqa: E402
from llmfs.train.trainer import evaluate, make_val  # noqa: E402

RESULTS = REPO / "results" / "21_distill_quant"
VISUALS = REPO / "docs" / "visuals"
BINS = REPO / "data" / "bins" / "split_8192"
VAL = {"code": str(BINS / "val_code.bin"), "en": str(BINS / "val_en.bin")}


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def cmd_distill(a):
    teacher, _ = load_model(a.teacher, device="cuda")
    teacher = teacher.eval().to(torch.bfloat16)
    T = min(512, teacher.cfg.block_size)
    val = make_val(VAL, T, 256)
    out = dict(teacher=a.teacher, teacher_val=evaluate(teacher, val), runs={})
    torch.backends.cuda.matmul.allow_tf32 = True
    for mode in ("scratch", "distill"):
        torch.manual_seed(0)
        st = GPT(GPTConfig(block_size=T, n_layer=4, n_head=4, n_embd=256)).cuda()
        opt = st.configure_optimizer(1e-3)
        loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85), "en": (BINS / "train_en.bin", 0.15)},
                             B=16, T=T, device="cuda", seed=0)
        curve = []
        t0 = time.time()
        for step in range(a.steps):
            lr = 1e-3 * min(1, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = loader.next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, ce = st(x, y)
                if mode == "distill":
                    with torch.no_grad():
                        tl, _ = teacher(x)
                    Tt = a.temp
                    kl = F.kl_div(F.log_softmax(logits.float() / Tt, -1).flatten(0, 1),
                                  F.log_softmax(tl.float() / Tt, -1).flatten(0, 1),
                                  log_target=True, reduction="batchmean") * Tt * Tt
                    loss = a.alpha * kl + (1 - a.alpha) * ce
                else:
                    loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
            opt.step()
            if step % 500 == 0 or step == a.steps - 1:
                v = evaluate(st, val)
                curve.append([step, round(ce.item(), 4), v["code"], v["en"]])
                print(mode, step, round(ce.item(), 4), v, flush=True)
        out["runs"][mode] = dict(val=evaluate(st, val), curve=curve, seconds=round(time.time() - t0, 1),
                                 params=sum(p.numel() for p in st.parameters()))
        write_json("distill.json", out)
        del st
        torch.cuda.empty_cache()
    out.update(alpha=a.alpha, temperature=a.temp, steps=a.steps)
    write_json("distill.json", out)


SCHEMES = [
    ("bf16", None),
    ("int8 per-tensor", dict(bits=8, granularity="tensor")),
    ("int8 per-channel", dict(bits=8, granularity="channel")),
    ("int4 per-tensor", dict(bits=4, granularity="tensor")),
    ("int4 per-channel", dict(bits=4, granularity="channel")),
    ("int4 group-128", dict(bits=4, granularity="group", group=128)),
    ("int4 group-32", dict(bits=4, granularity="group", group=32)),
    ("int4 group-128 asym", dict(bits=4, granularity="group", group=128, symmetric=False)),
    ("nf4 block-64", "nf4"),
    ("int3 group-32", dict(bits=3, granularity="group", group=32)),
]


@torch.no_grad()
def tok_per_s(model, n=128):
    ids = torch.randint(0, 8192, (1, 16), device="cuda")
    generate_cached(model, ids, 8, temperature=0.0)
    torch.cuda.synchronize()
    t = time.perf_counter()
    generate_cached(model, ids, n, temperature=0.0)
    torch.cuda.synchronize()
    return round(n / (time.perf_counter() - t), 1)


def cmd_quant(a):
    base, _ = load_model(a.model, device="cuda")
    base = base.eval().to(torch.bfloat16)
    T = min(512, base.cfg.block_size)
    val = make_val(VAL, T, 256)
    rows = []
    for name, kw in SCHEMES:
        m = copy.deepcopy(base)
        if kw == "nf4":
            for mod_name, mod in list(m.named_modules()):
                for cn, child in list(mod.named_children()):
                    full = f"{mod_name}.{cn}" if mod_name else cn
                    if isinstance(child, torch.nn.Linear) and not full.endswith("lm_head"):
                        setattr(mod, cn, NF4Linear(child.float()).to(torch.bfloat16))
            size = sum(x.nbytes() for x in m.modules() if isinstance(x, NF4Linear))
            size += sum(p.numel() * p.element_size() for p in {id(p): p for p in m.parameters()}.values())
        elif kw is not None:
            if kw["bits"] == 3:                                   # fake-quant only: no 3-bit packing
                from llmfs.post.quant import fake_quant
                for mod in m.modules():
                    if isinstance(mod, torch.nn.Linear) and mod is not m.lm_head:
                        mod.weight.data = fake_quant(mod.weight.float(), **kw).to(mod.weight.dtype)
                size = None
            else:
                quantize_model(m, **kw)
                size = model_bytes(m)
        else:
            size = model_bytes(m)
        v = evaluate(m, val)
        rec = dict(name=name, val=v, mb=round(size / 2**20, 2) if size else None, tok_s=tok_per_s(m))
        rows.append(rec)
        print(rec, flush=True)
        del m
        torch.cuda.empty_cache()
    write_json(f"quant_{a.tag}.json", rows)


def cmd_plot(a):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    d = RESULTS / "distill.json"
    if d.exists():
        r = json.loads(d.read_text())
        for i, (k, v) in enumerate(r["runs"].items()):
            c = v["curve"]
            ax[0].plot([x[0] for x in c], [x[2] for x in c], "o-", color=f"C{i}", label=f"student, {k}: {v['val']['code']}")
        ax[0].axhline(r["teacher_val"]["code"], color="gray", ls=":", label=f"teacher: {r['teacher_val']['code']}")
        ax[0].set(xlabel="step", ylabel="val loss (code)", title="Same student, same tokens: scratch vs distilled",
                  ylim=(None, 5))
        ax[0].legend()
    q = RESULTS / f"quant_{a.tag}.json"
    if q.exists():
        rows = json.loads(q.read_text())
        base = next(r for r in rows if r["name"] == "bf16")["val"]["code"]
        rows = [r for r in rows if r["name"] != "bf16"]
        rows.sort(key=lambda r: r["val"]["code"] - base)
        y = range(len(rows))
        d = [max(1e-4, r["val"]["code"] - base) for r in rows]
        ax[1].barh(list(y), d, color=["C2" if x < 0.03 else "C1" if x < 0.2 else "C3" for x in d])
        for i, r in enumerate(rows):
            mb = f"{r['mb']:.1f} MB" if r["mb"] else "fake-quant"
            ax[1].text(d[i] * 1.15, i, f"+{d[i]:.4f}  ({mb})", va="center", fontsize=8)
        ax[1].set_yticks(list(y), [r["name"] for r in rows], fontsize=8)
        ax[1].invert_yaxis()
        ax[1].set(xscale="log", xlim=(1e-4, 30), xlabel=f"val code loss increase over bf16 ({base:.3f}, 57 MB)",
                  title="Quantization: damage per scheme (log scale)")
    fig.tight_layout()
    fig.savefig(VISUALS / "21_distill_quant.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["distill", "quant", "plot"])
    p.add_argument("--teacher", default=str(REPO / "checkpoints" / "11_pretrain" / "workhorse" / "final.pt"))
    p.add_argument("--model", default=str(REPO / "checkpoints" / "11_pretrain" / "workhorse" / "final.pt"))
    p.add_argument("--tag", default="workhorse")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=2.0)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
