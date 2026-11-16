"""Stage 11: the real pretraining runs. docs/11_pretraining.md.

    uv run python stages/11_pretrain/run.py resume_test          # 8b: continuity + the no-optimizer control
    uv run python stages/11_pretrain/run.py workhorse            # 8a
    uv run python stages/11_pretrain/run.py capstone             # 8c  (resumable: rerun the same command)
    uv run python stages/11_pretrain/run.py samples --which workhorse
    uv run python stages/11_pretrain/run.py plot

Configs live in CONFIGS below and are written next to each checkpoint, so a
run is reproducible from its directory alone. Both resume automatically from
checkpoints/11_pretrain/<name>/ckpt.pt if it exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "07_generation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sample import generate_cached  # noqa: E402

from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "11_pretrain"
VISUALS = REPO / "docs" / "visuals"
CK = REPO / "checkpoints" / "11_pretrain"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"

# Filled from Stage 10's measurements (docs/10 §8b, docs/11 §6).
CONFIGS = {
    # 30M non-embedding; ~20 tokens per parameter (590M tokens), ~2.4 h at 70k tok/s (Stage 09 §9d)
    "workhorse": dict(model=dict(n_layer=8, n_head=8, n_embd=512, block_size=1024, qk_norm=True), T=1024, B=16,
                      accum=2, steps=18000, lr=1e-3, warmup=500, eval_every=500, eval_windows=128,
                      ckpt_every=1000, compile=True),
    # 92M; 65,536 tokens per step × 27,500 = 1.80B tokens (~20 per parameter), ~20 h at 25.4k tok/s.
    # B=16 does not fit (Stage 09 §9d), so 8 × 8 accumulation.
    "capstone": dict(model=dict(n_layer=12, n_head=12, n_embd=768, block_size=1024, qk_norm=True), T=1024, B=8,
                     accum=8, steps=27500, lr=6e-4, warmup=1000, eval_every=1000, eval_windows=128,
                     ckpt_every=500, compile=True),
}
PROMPTS = ["def fibonacci(n):\n", "import os\nimport json\n\n\ndef load_config(path):\n",
           "class Stack:\n    def __init__(self):\n", "The history of the printing press"]


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Paths relative to the repo: results are committed, and an absolute path would publish
    # where this machine keeps the project (and mean nothing on anyone else's).
    text = json.dumps(obj, indent=1).replace(json.dumps(str(REPO) + "/")[1:-1], "")
    (RESULTS / name).write_text(text)


def sampler_hook(tok, every, store):
    def hook(step, rec, model):
        if step % every == 0 and step > 0:
            model.eval()
            outs = []
            for i, p in enumerate(PROMPTS[:2]):
                ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
                raw = getattr(model, "_orig_mod", model)
                o = generate_cached(raw, ids, 100, eot=tok.eot_id, seed=i, temperature=0.8, top_p=0.95)
                outs.append(tok.decode(o[0].tolist()))
            store[str(step)] = outs
            model.train()
    return hook


def run_named(name, extra=None):
    cfg = dict(CONFIGS[name], **(extra or {}))
    out = CK / name
    c = TrainConfig(name=name, out_dir=str(out), resume=True, **cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(c), indent=1, default=str))
    tok = Tokenizer.load(str(TOK))
    samples = {}
    sp = RESULTS / f"{name}_samples.json"
    if sp.exists():
        samples = json.loads(sp.read_text())
    every = max(cfg["steps"] // 8, 500)

    def hook(step, rec, model):
        sampler_hook(tok, every, samples)(step, rec, model)
        if step % every == 0 and samples:
            write_json(f"{name}_samples.json", samples)

    r = train(c, on_step=hook)
    model = getattr(r["model"], "_orig_mod", r["model"])
    if r["final"].get("step") == cfg["steps"] - 1:
        save_checkpoint(out / "final.pt", model, final=r["final"])
    hist = [json.loads(l) for l in open(out / "log.jsonl")]
    write_json(f"{name}.json", dict(config=asdict(c), final=r["final"], params=r["params"],
                                    tokens=c.tokens_per_step * c.steps,
                                    curve=[[h["step"], h["loss"], h["gnorm"], h.get("val")] for h in hist]))
    write_json(f"{name}_samples.json", samples)
    return r


def cmd_workhorse(a):
    run_named("workhorse")


def cmd_capstone(a):
    run_named("capstone")


def cmd_resume_test(a):
    """Same 4L model three ways: uninterrupted; stopped at 600 and resumed; resumed with fresh Adam."""
    base = dict(model=dict(n_layer=4, n_head=4, n_embd=256), T=256, B=32, steps=1200, lr=1e-3, warmup=100,
                eval_every=200, ckpt_every=600, compile=False)
    res = {}
    for label, kw in (("uninterrupted", {}), ("resumed", {}), ("resumed_fresh_adam", {})):
        d = CK / "resume_test" / label
        shutil.rmtree(d, ignore_errors=True)
        if label == "uninterrupted":
            r = train(TrainConfig(name=label, out_dir=str(d), **base))
        else:
            train(TrainConfig(name=label, out_dir=str(d), stop_after=600, **base))
            r = train(TrainConfig(name=label, out_dir=str(d), resume=True,
                                  resume_drop_optimizer=label.endswith("adam"), **base))
        hist = [json.loads(l) for l in open(d / "log.jsonl")]
        res[label] = dict(curve=[[h["step"], h["loss"], h["gnorm"]] for h in hist], final=r["final"])
        print(label, r["final"], flush=True)
    u = {s: l for s, l, _ in res["uninterrupted"]["curve"]}
    for label in ("resumed", "resumed_fresh_adam"):
        c = {s: l for s, l, _ in res[label]["curve"]}
        common = [s for s in c if s in u and 600 <= s <= 700]
        res[label]["max_abs_diff_600_700"] = round(max(abs(c[s] - u[s]) for s in common), 4)
        g = {s: gn for s, _, gn in res[label]["curve"]}
        res[label]["gnorm_at_600_610"] = [g.get(600), g.get(610)]
    write_json("resume_test.json", res)
    shutil.rmtree(CK / "resume_test", ignore_errors=True)


def cmd_samples(a):
    tok = Tokenizer.load(str(TOK))
    model, _ = load_model(CK / a.which / "final.pt", device="cuda")
    model.eval()
    out = []
    for i, p in enumerate(PROMPTS):
        for temp in (0.0, 0.8):
            ids = torch.tensor([tok.encode_ordinary(p)], device="cuda")
            o = generate_cached(model, ids, 200, eot=tok.eot_id, seed=i, temperature=temp, top_p=0.95)
            out.append(dict(prompt=p, temperature=temp, text=tok.decode(o[0].tolist())))
    write_json(f"{a.which}_final_samples.json", out)
    for s in out:
        print("=" * 60, s["temperature"])
        print(s["text"])


def cmd_plot(a):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, name in enumerate(("workhorse", "capstone")):
        p = RESULTS / f"{name}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c = r["curve"]
        tps = r["config"]["B"] * r["config"]["T"] * r["config"]["accum"]
        xs = [x[0] * tps for x in c]
        ax[0].plot([a for a in xs if a > 0], [x[1] for x, a in zip(c, xs) if a > 0], color=f"C{i}", alpha=0.3)
        v = [(x[0] * tps, x[3]) for x in c if x[3] and x[0] > 0]      # step 0 has no place on a log axis
        ax[0].plot([t for t, _ in v], [d["code"] for _, d in v], "o-", color=f"C{i}", ms=3,
                   label=f"{name} ({r['params'] / 1e6:.0f}M): val code {v[-1][1]['code']}")
        ax[0].plot([t for t, _ in v], [d["en"] for _, d in v], "s--", color=f"C{i}", ms=3,
                   label=f"{name}: val English {v[-1][1]['en']}")
    ax[0].set(xscale="log", xlabel="tokens seen", ylabel="loss", ylim=(1, 5), title="Pretraining")
    ax[0].legend(fontsize=7)
    rt = RESULTS / "resume_test.json"
    if rt.exists():
        r = json.loads(rt.read_text())
        for i, (k, v) in enumerate(r.items()):
            c = [x for x in v["curve"] if 400 <= x[0] <= 900]
            ax[1].plot([x[0] for x in c], [x[1] for x in c], color=f"C{i}", label=k, alpha=0.8)
        ax[1].axvline(600, color="gray", ls=":", label="stop / resume")
        ax[1].set(xlabel="step", ylabel="train loss", title="Resume: continuous, unless Adam's state is lost")
        ax[1].legend()
    fig.tight_layout()
    fig.savefig(VISUALS / "11_pretrain.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["resume_test", "workhorse", "capstone", "samples", "plot"])
    p.add_argument("--which", default="workhorse")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
