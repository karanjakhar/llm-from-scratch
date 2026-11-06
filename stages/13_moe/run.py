"""Stage 13 experiments. docs/13_moe.md §9.

    uv run python stages/13_moe/run.py train      # 9a, 9b, 9d: dense / MoE / no-aux / shared / fine-grained
    uv run python stages/13_moe/run.py analyse    # utilisation histograms + what experts specialise in
    uv run python stages/13_moe/run.py plot

Same base as Stage 12 (6 layers, C=384, 8 q heads / 2 kv, Llama block),
T=512, B=16, 4000 steps. Only the MLP changes.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from llmfs.data import open_bin  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402
from llmfs.train.trainer import TrainConfig, train  # noqa: E402

RESULTS = REPO / "results" / "13_moe"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "13_moe"
BINS = REPO / "data" / "bins" / "split_8192"
BASE = dict(arch="llama", n_layer=6, n_embd=384, n_head=8, n_kv_head=2)

VARIANTS = {
    "dense":            dict(),
    "moe_8e_top2":      dict(n_experts=8, top_k=2),
    "moe_8e_top2_noaux": dict(n_experts=8, top_k=2, aux_coef=0.0, z_coef=0.0),
    "moe_7e_top1_shared1": dict(n_experts=7, top_k=1, n_shared=1, expert_hidden=512),
    "moe_16e_top4":     dict(n_experts=16, top_k=4),
    "dense_4x_wide":    dict(mlp_hidden=4 * 1024),     # total-param-matched dense, for reference
}


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def read_json(name, default):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else default


def cmd_train(args):
    out = read_json("train.json", [])
    names = args.only.split(",") if args.only else list(VARIANTS)
    for name in names:
        if any(r["name"] == name for r in out):
            continue
        torch.backends.cuda.matmul.allow_tf32 = True
        balance = []

        def on_step(step, rec, model):
            if step % 50 == 0 and hasattr(model, "moe_layers") and model.cfg.n_experts:
                cnt = torch.stack([m.last_counts for m in model.moe_layers()])   # (layers, E)
                frac = cnt / cnt.sum(1, keepdim=True)
                balance.append([step, round(frac.max(1).values.mean().item(), 4),
                                round(model.last_aux.item(), 4)])

        c = TrainConfig(model={**BASE, **VARIANTS[name]}, B=16, T=512, steps=4000, lr=1e-3,
                        warmup=200, eval_every=1000, name=name)
        r = train(c, on_step=on_step)
        m = r["model"]
        rec = dict(name=name, kw=VARIANTS[name], val=r["final"]["val"], params=r["params"],
                   active=m.active_params() if m.cfg.n_experts else r["params"],
                   seconds=r["seconds"], curve=[[h["step"], h["loss"]] for h in r["hist"]],
                   balance=balance)
        CKPT.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CKPT / f"{name}.pt", m)
        out.append(rec)
        write_json("train.json", out)
        print({k: v for k, v in rec.items() if k not in ("curve", "balance")}, flush=True)
        del r, m
        torch.cuda.empty_cache()


@torch.no_grad()
def cmd_analyse(args):
    tok = Tokenizer.load(str(REPO / "data" / "tokenizers" / "split_8192.json"))
    out = {}
    for name in ("moe_8e_top2", "moe_8e_top2_noaux"):
        p = CKPT / f"{name}.pt"
        if not p.exists():
            continue
        model, _ = load_model(p, device="cuda")
        model.eval()
        res = {}
        for split in ("val_code", "val_en"):
            data = open_bin(BINS / f"{split}.bin")
            starts = np.linspace(0, len(data) - 600, 32).astype(np.int64)
            x = torch.from_numpy(np.stack([data[s:s + 512].astype(np.int64) for s in starts])).cuda()
            choices = []
            hooks = []
            for m in model.moe_layers():
                def hook(mod, inp, outp, store=choices):
                    xx = inp[0].reshape(-1, inp[0].shape[-1])
                    store.append(mod.router(xx).float().topk(mod.k, -1).indices.cpu())
                hooks.append(m.register_forward_hook(hook))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(x)
            for h in hooks:
                h.remove()
            E = model.cfg.n_experts
            res[split] = dict(
                util=[np.bincount(c.reshape(-1).numpy(), minlength=E).tolist() for c in choices],
                top1=[c[:, 0].numpy() for c in choices], ids=x.reshape(-1).cpu().numpy())
        # specialisation: for each expert in the middle MoE layer, its share of code vs English
        # tokens, and the tokens it is most often the top choice for.
        L = len(res["val_code"]["util"]) // 2
        spec = []
        for e in range(model.cfg.n_experts):
            code = int((res["val_code"]["top1"][L] == e).sum())
            en = int((res["val_en"]["top1"][L] == e).sum())
            ids = res["val_code"]["ids"][res["val_code"]["top1"][L] == e]
            common = np.bincount(ids, minlength=8196)
            top = [tok.decode([int(i)]) for i in np.argsort(-common)[:8]]
            spec.append(dict(expert=e, code_tokens=code, en_tokens=en,
                             en_share=round(en / max(1, code + en), 3), top_code_tokens=top))
        out[name] = dict(util_code=res["val_code"]["util"], util_en=res["val_en"]["util"],
                         middle_layer=L, specialisation=spec)
        print(name, json.dumps(spec[:3]), flush=True)
    write_json("analyse.json", out)


def cmd_plot(args):
    tr = read_json("train.json", [])
    an = read_json("analyse.json", {})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for i, r in enumerate(tr):
        c = np.array(r["curve"])
        k = max(1, len(c) // 80)
        ax[0, 0].plot(c[::k, 0], c[::k, 1], color=f"C{i}",
                      label=f"{r['name']} ({r['active'] / 1e6:.1f}M act / {r['params'] / 1e6:.1f}M): "
                            f"{r['val']['code']}")
    ax[0, 0].set(ylim=(2.2, 4), xlabel="step", ylabel="train loss", title="Dense vs MoE")
    ax[0, 0].legend(fontsize=7)
    for i, r in enumerate(tr):
        if r["balance"]:
            b = np.array(r["balance"])
            ax[0, 1].plot(b[:, 0], b[:, 1], color=f"C{i}", label=r["name"])
    E = 8
    ax[0, 1].axhline(1 / E, color="gray", ls=":", label="perfect balance, 8 experts")
    ax[0, 1].set(xlabel="step", ylabel="busiest expert's share (mean over layers)",
                 title="Load balance during training")
    ax[0, 1].legend(fontsize=7)
    for j, name in enumerate(("moe_8e_top2", "moe_8e_top2_noaux")):
        if name not in an:
            continue
        u = np.array(an[name]["util_code"], dtype=float)
        u = u / u.sum(1, keepdims=True)
        im = ax[1, j].imshow(u, cmap="viridis", vmin=0, vmax=max(0.5, u.max()))
        ax[1, j].set(xlabel="expert", ylabel="MoE layer", title=f"{name}: share of tokens (val code)")
        for (a, b), v in np.ndenumerate(u):
            ax[1, j].text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                          color="white" if v < 0.3 else "black")
        fig.colorbar(im, ax=ax[1, j], fraction=0.04)
    fig.tight_layout()
    fig.savefig(VISUALS / "13_moe.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "analyse", "plot"])
    p.add_argument("--only", default="")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
