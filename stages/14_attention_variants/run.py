"""Stage 14 experiments. docs/14_attention_variants.md §7.

    uv run python stages/14_attention_variants/run.py mla      # 7a + 7b: MLA vs GQA, latent sweep
    uv run python stages/14_attention_variants/run.py needle   # 7c: retrieval vs loss
    uv run python stages/14_attention_variants/run.py plot

Same base as Stages 12-13 (6 layers, C=384, 8 heads of 48, T=512, 4000 steps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from needle import run_needle  # noqa: E402

from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.model.llama import LlamaConfig, kv_bytes_per_token  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "14_attention_variants"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "14_attention_variants"
CK12 = REPO / "checkpoints" / "12_modern_arch"
BINS = REPO / "data" / "bins" / "split_8192"
BASE = dict(arch="llama", n_layer=6, n_embd=384, n_head=8)

VARIANTS = {
    "mla_r128": dict(attn="mla", kv_latent=128, rope_dim=16),
    "mla_r64":  dict(attn="mla", kv_latent=64, rope_dim=16),
    "mla_r32":  dict(attn="mla", kv_latent=32, rope_dim=16),
    "mla_r16":  dict(attn="mla", kv_latent=16, rope_dim=16),
    "mla_r4":   dict(attn="mla", kv_latent=4, rope_dim=16),
}


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def read_json(name, default):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default


def cmd_mla(args):
    out = read_json("mla.json", [])
    for name, kw in VARIANTS.items():
        if any(r["name"] == name for r in out):
            continue
        torch.backends.cuda.matmul.allow_tf32 = True
        c = TrainConfig(model={**BASE, **kw}, B=16, T=512, steps=4000, lr=1e-3, warmup=200,
                        eval_every=1000, name=name, compile=True)
        r = train(c)
        m = r["model"]
        CKPT.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CKPT / f"{name}.pt", m)
        cfg = LlamaConfig(**{k: v for k, v in {**BASE, **kw}.items() if k != "arch"})
        out.append(dict(name=name, kw=kw, val=r["final"]["val"], params=r["params"],
                        seconds=r["seconds"], kv_bytes_per_token_bf16=kv_bytes_per_token(cfg)))
        write_json("mla.json", out)
        print(out[-1], flush=True)
        del r, m
        torch.cuda.empty_cache()


def cmd_needle(args):
    tok = Tokenizer.load(str(REPO / "data" / "tokenizers" / "split_8192.json"))
    models = {
        "GQA full (Stage 12)": CK12 / "window0.pt",
        "window 128 (Stage 12)": CK12 / "window128.pt",
        "window 64 (Stage 12)": CK12 / "window64.pt",
        "MLA r64": CKPT / "mla_r64.pt",
        "MLA r16": CKPT / "mla_r16.pt",
    }
    out = {}
    lengths = [128, 256, 512]
    depths = [0.1, 0.5, 0.9]
    for name, p in models.items():
        if not p.exists():
            continue
        model, ck = load_model(p, device="cuda")
        model.eval()
        grid = run_needle(model, tok, BINS / "val_code.bin", lengths, depths, n=args.n)
        out[name] = dict(grid=grid, mean=round(float(np.mean(list(grid.values()))), 3))
        print(name, out[name], flush=True)
        del model
        torch.cuda.empty_cache()
    # val loss of each, for the "perplexity hides it" comparison
    w = {r["name"]: r["val"]["code"] for r in read_json("../12_modern_arch/window.json", [])}
    m = {r["name"]: r["val"]["code"] for r in read_json("mla.json", [])}
    loss = {"GQA full (Stage 12)": w.get("full"), "window 128 (Stage 12)": w.get("window=128"),
            "window 64 (Stage 12)": w.get("window=64"), "MLA r64": m.get("mla_r64"),
            "MLA r16": m.get("mla_r16")}
    for k in out:
        out[k]["val_code_loss"] = loss.get(k)
    write_json("needle.json", out)


def cmd_plot(args):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    # GQA/MHA points come from Stage 12's identical-recipe runs (no need to retrain them)
    mla = read_json("../12_modern_arch/gqa.json", [])[:4] + read_json("mla.json", [])
    for r in mla:
        color = "C3" if r["name"].startswith("mla") else "C0"
        ax[0].plot(r["kv_bytes_per_token_bf16"] / 1024, r["val"]["code"], "o", color=color)
        ax[0].annotate(r["name"], (r["kv_bytes_per_token_bf16"] / 1024, r["val"]["code"]),
                       textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax[0].set(xscale="log", xlabel="KV cache per token, all layers (KB, bf16)", ylabel="val loss (code)",
              title="Cache size vs quality: GQA/MHA (blue), MLA (red)")
    nd = read_json("needle.json", {})
    names = list(nd)
    if names:
        cells = list(nd[names[0]]["grid"].keys())
        mat = np.array([[nd[n]["grid"][c] for c in cells] for n in names])
        im = ax[1].imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax[1].set_xticks(range(len(cells)), [c.replace("@", "\n@") for c in cells], fontsize=7)
        ax[1].set_yticks(range(len(names)),
                         [f"{n}\nloss {nd[n]['val_code_loss']}" for n in names], fontsize=7)
        for (a, b), v in np.ndenumerate(mat):
            ax[1].text(b, a, f"{v:.0%}", ha="center", va="center", fontsize=7)
        ax[1].set(title="Needle retrieval: context length @ depth", xlabel="length @ depth")
        fig.colorbar(im, ax=ax[1], fraction=0.03)
    fig.tight_layout()
    fig.savefig(VISUALS / "14_attention_variants.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["mla", "needle", "plot"])
    p.add_argument("--n", type=int, default=30)
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
