"""Stage 08 experiments. docs/08_training_loop.md §8.

    uv run python stages/08_training/run.py optimizers   # 8a: SGD / momentum / Adam / AdamW / L2
    uv run python stages/08_training/run.py overfit      # 8b: one batch to ~0
    uv run python stages/08_training/run.py tinystories  # 8c: 10M model smoke test, ~20 min
    uv run python stages/08_training/run.py ablate       # 8d: no warmup / no clip at a high lr
    uv run python stages/08_training/run.py sampling     # 8e: sequential / random / shuffled
    uv run python stages/08_training/run.py plot
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "stages" / "07_generation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from optim import SGD, AdamW, lr_schedule  # noqa: E402
from sample import generate_cached  # noqa: E402

from llmfs.data import MixedLoader  # noqa: E402
from llmfs.model.gpt import GPT, GPTConfig, save_checkpoint  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import TrainConfig, make_val, evaluate, train  # noqa: E402

RESULTS = REPO / "results" / "08_training"
VISUALS = REPO / "docs" / "visuals"
BINS = REPO / "data" / "bins" / "split_8192"
TS = REPO / "data" / "bins" / "tinystories"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"
CODE_ONLY = {"code": (str(BINS / "train_code.bin"), 1.0)}


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def slim(hist):
    return [[h["step"], h["loss"], h["gnorm"], h.get("val", {}).get("code")] for h in hist]


# ---------------------------------------------------------------------------

def cmd_optimizers(args):
    """Same model, same data, five optimizers, our hand-written versions."""
    steps, B, T = 1500, 32, 256
    setups = {
        "sgd_lr0.1":        ("sgd", dict(lr=0.1, momentum=0.0)),
        "sgd_momentum0.9":  ("sgd", dict(lr=0.1, momentum=0.9)),
        "adam_nodecay":     ("adam", dict(wd=0.0, l2=False)),
        "adamw_wd0.1":      ("adam", dict(wd=0.1, l2=False)),
        "adam_l2_wd0.1":    ("adam", dict(wd=0.1, l2=True)),
    }
    runs = {}
    val = make_val({"code": str(BINS / "val_code.bin")}, T, 256)
    for name, (kind, kw) in setups.items():
        torch.manual_seed(0)
        m = GPT(GPTConfig(block_size=T)).cuda()
        if kind == "sgd":
            opt = SGD(m.parameters(), lr=kw["lr"], momentum=kw["momentum"])
            peak = kw["lr"]
        else:
            groups = [{"params": [p for p in m.parameters() if p.dim() >= 2], "weight_decay": kw["wd"]},
                      {"params": [p for p in m.parameters() if p.dim() < 2], "weight_decay": 0.0}]
            opt = AdamW(groups, lr=1e-3, l2_in_loss=kw["l2"])
            peak = 1e-3
        loader = MixedLoader(CODE_ONLY, B=B, T=T, device="cuda", seed=0)
        curve = []
        t0 = time.time()
        for step in range(steps):
            lr = lr_schedule(step, steps, peak, 100)
            if kind == "sgd":
                opt.lr = lr
            else:
                for g in opt.param_groups:
                    g["lr"] = lr
            x, y = loader.next_batch()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = m(x, y)
            for p in m.parameters():
                p.grad = None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            if step % 25 == 0:
                curve.append([step, round(loss.item(), 4)])
        wnorm = math.sqrt(sum(p.float().pow(2).sum().item() for p in m.parameters() if p.dim() >= 2))
        runs[name] = dict(curve=curve, val=evaluate(m, val)["code"], weight_norm=round(wnorm, 2),
                          seconds=round(time.time() - t0, 1))
        print(name, runs[name]["val"], runs[name]["weight_norm"], flush=True)
        write_json("optimizers.json", runs)


def cmd_overfit(args):
    torch.manual_seed(0)
    m = GPT(GPTConfig(block_size=256)).cuda()
    opt = m.configure_optimizer(1e-3)
    x, y = MixedLoader(CODE_ONLY, B=8, T=256, device="cuda", seed=0).next_batch()
    curve = []
    for step in range(400):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append([step, round(loss.item(), 5)])
    print("first", curve[0], "last", curve[-1])
    steps_to = {thr: next((s for s, l in curve if l < thr), None) for thr in (1.0, 0.1, 0.01)}
    write_json("overfit.json", dict(curve=curve, steps_to=steps_to, B=8, T=256))
    print(steps_to)


TS_PROMPTS = ["Once upon a time", "Lily wanted to", "The dog was very", "One day, Tom and his mom"]


def cmd_tinystories(args):
    """docs §8c: ~10M params, ~20 minutes."""
    c = TrainConfig(model=dict(n_layer=6, n_head=5, n_embd=320), B=32, T=256, steps=args.steps,
                    lr=1e-3, warmup=200, eval_every=500, eval_windows=256,
                    sources={"ts": (str(TS / "train.bin"), 1.0)}, val={"ts": str(TS / "val.bin")},
                    name="tinystories", out_dir=str(REPO / "checkpoints" / "08_tinystories"))
    tok = Tokenizer.load(str(TOK))
    snapshots = {}

    def on_step(step, rec, model):
        if step in (0, 200, 1000, c.steps - 1):
            model.eval()
            ids = torch.tensor([tok.encode_ordinary(TS_PROMPTS[0])], device="cuda")
            out = generate_cached(model, ids, 80, eot=tok.eot_id, seed=0, temperature=0.8, top_p=0.95)
            snapshots[step] = tok.decode(out[0].tolist())
            model.train()

    r = train(c, on_step=on_step)
    model = r["model"].eval()
    save_checkpoint(REPO / "checkpoints" / "08_tinystories" / "final.pt", model)
    samples = []
    for i, p in enumerate(TS_PROMPTS):
        ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
        out = generate_cached(model, ids, 150, eot=tok.eot_id, seed=i, temperature=0.8, top_p=0.95)
        samples.append(tok.decode(out[0].tolist()))
    res = dict(params=r["params"], seconds=r["seconds"], final=r["final"], curve=slim(r["hist"]),
               samples=samples, snapshots=snapshots, tokens=c.tokens_per_step * c.steps)
    write_json("tinystories.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


def cmd_ablate(args):
    """docs §8d: a 6L/384 model at lr 3e-3, where the safety rails matter."""
    variants = {"baseline": dict(), "no_warmup": dict(warmup=0), "no_clip": dict(clip=0.0),
                "no_warmup_no_clip": dict(warmup=0, clip=0.0)}
    out = {}
    for name, kw in variants.items():
        c = TrainConfig(model=dict(n_layer=6, n_head=6, n_embd=384), steps=2000, lr=3e-3,
                        warmup=kw.get("warmup", 200), clip=kw.get("clip", 1.0), eval_every=1000,
                        sources=CODE_ONLY, val={"code": str(BINS / "val_code.bin")}, name=name)
        r = train(c)
        out[name] = dict(val=r["final"].get("val"), curve=[[h["step"], h["loss"], h["gnorm"]]
                                                           for h in r["hist"]],
                         max_loss_after_100=max(h["loss"] for h in r["hist"] if h["step"] >= 100))
        del r
        torch.cuda.empty_cache()
        write_json("ablate.json", out)


def cmd_sampling(args):
    out = {}
    for s in ("sequential", "random", "shuffled"):
        c = TrainConfig(steps=4000, sampler=s, eval_every=500, name=s)
        r = train(c)
        out[s] = dict(final=r["final"], curve=slim(r["hist"]),
                      val_curve=[[h["step"], h["val"]] for h in r["hist"] if "val" in h])
        write_json("sampling.json", out)


def cmd_plot(args):
    VISUALS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    opt = json.loads((RESULTS / "optimizers.json").read_text())
    ax = axes[0, 0]
    for i, (k, v) in enumerate(opt.items()):
        c = np.array(v["curve"])
        ax.plot(c[:, 0], c[:, 1], color=f"C{i}", label=f"{k}: val {v['val']}")
    ax.set(ylim=(2.5, 8), xlabel="step", ylabel="train loss", title="Optimizers (hand-written, 4L)")
    ax.legend(fontsize=8)

    ov = json.loads((RESULTS / "overfit.json").read_text())
    ts = json.loads((RESULTS / "tinystories.json").read_text())
    ax = axes[0, 1]
    c = np.array(ov["curve"])
    ax.plot(c[:, 0], c[:, 1], color="C3", label="one batch, over and over")
    c = np.array([[r[0], r[1]] for r in ts["curve"]])
    ax.plot(c[:, 0], c[:, 1], color="C0", label="TinyStories, fresh data")
    ax.set(yscale="log", xlabel="step", ylabel="train loss", title="Overfit one batch vs real training")
    ax.legend()

    ab = json.loads((RESULTS / "ablate.json").read_text())
    ax = axes[1, 0]
    for i, (k, v) in enumerate(ab.items()):
        c = np.array(v["curve"])
        ax.plot(c[:, 0], c[:, 1], color=f"C{i}", alpha=0.85,
                label=f"{k}: val {v['val']['code'] if v['val'] else 'n/a'}")
    ax.set(ylim=(2, 9.5), xlabel="step", ylabel="train loss", title="Safety rails at lr 3e-3 (6L/384)")
    ax.legend(fontsize=8)

    sp = json.loads((RESULTS / "sampling.json").read_text())
    ax = axes[1, 1]
    for i, (k, v) in enumerate(sp.items()):
        c = np.array([[r[0], r[1]] for r in v["curve"]])
        ax.plot(c[:, 0], c[:, 1], color=f"C{i}", alpha=0.35)
        vc = np.array([[s, d["code"]] for s, d in v["val_curve"]])
        ax.plot(vc[:, 0], vc[:, 1], "o-", color=f"C{i}", label=f"{k}: final val {vc[-1, 1]}")
    ax.set(ylim=(2.4, 4.5), xlabel="step", ylabel="loss (faint: train, dots: val)",
           title="Which windows to train on")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "08_training.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["optimizers", "overfit", "tinystories", "ablate", "sampling", "plot"])
    p.add_argument("--steps", type=int, default=6000)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
