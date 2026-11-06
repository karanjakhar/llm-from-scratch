"""Stage 06 experiments. docs/06_gpt_model.md §8.

    uv run python stages/06_gpt/run.py oracle    # 8a: GPT-2 vs HF, and GPT-2's bits/byte on our val
    uv run python stages/06_gpt/run.py init      # 8b: stream at init + 12-layer training, scaled vs plain
    uv run python stages/06_gpt/run.py noshift   # 8c: targets = inputs
    uv run python stages/06_gpt/run.py tie       # 8d: tied vs untied, 4k and 12k steps
    uv run python stages/06_gpt/run.py plot

Training recipe = Stage 04/05's (B=32, T=256, AdamW 0.9/0.95, wd 0.1, lr 1e-3,
warmup 100, cosine to 10%, clip 1.0, bf16), with the fused SDPA attention path
(checked equal to ours in test_gpt.py) to save time.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from gpt import GPT, GPTConfig, count_params, count_params_by_hand  # noqa: E402

from llmfs.data import MixedLoader, open_bin  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

VISUALS = REPO / "docs" / "visuals"
RESULTS = REPO / "results" / "06_gpt"
BINS = REPO / "data" / "bins" / "split_8192"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


def write_json(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1))


def lr_at(step, steps, peak, warmup=100):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def val_windows(T, n, split="val_code"):
    data = open_bin(BINS / f"{split}.bin")
    starts = np.linspace(0, len(data) - T - 2, n).astype(np.int64)
    return torch.from_numpy(np.stack([data[s:s + T + 1].astype(np.int64) for s in starts]))


@torch.no_grad()
def evaluate(model, windows, shift=True, bs=32):
    model.eval()
    out = []
    for i in range(0, len(windows), bs):
        w = windows[i:i + bs].cuda()
        x, y = (w[:, :-1], w[:, 1:]) if shift else (w[:, :-1], w[:, :-1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        out.append(loss.item())
    model.train()
    return round(float(np.mean(out)), 4)


def train(name, cfg_kw, steps=4000, seed=0, B=32, T=256, lr=1e-3, shift=True):
    torch.manual_seed(seed)
    cfg = GPTConfig(block_size=T, flash=True, **cfg_kw)
    model = GPT(cfg).cuda()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True)
    loader = MixedLoader({"code": (BINS / "train_code.bin", 0.85),
                          "en": (BINS / "train_en.bin", 0.15)}, B=B, T=T, device="cuda", seed=seed)
    val = val_windows(T, 512)
    curve = []
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y = loader.next_batch()
        if not shift:
            y = x                                                # docs §8c: predict what you see
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            curve.append([step, round(loss.item(), 4)])
            print(f"{name} s{seed} step {step} loss {loss.item():.3f}", flush=True)
    rec = dict(name=name, cfg=cfg_kw, seed=seed, steps=steps, params=count_params(model),
               train_seconds=round(time.time() - t0, 1), val_code_loss=evaluate(model, val),
               curve=curve)
    return rec, model


# ---------------------------------------------------------------------------

def cmd_oracle(args):
    """8a: our GPT-2 vs HF on real text; then GPT-2's bits per byte on our val."""
    import tiktoken
    from transformers import GPT2LMHeadModel

    enc = tiktoken.get_encoding("gpt2")
    tok = Tokenizer.load(str(TOK))
    ours = GPT.from_pretrained("gpt2").cuda().eval()
    hf = GPT2LMHeadModel.from_pretrained("openai-community/gpt2",
                                         attn_implementation="eager").cuda().eval()
    res = {}
    for split in ("val_code", "val_en"):
        # Decode a slice of our val tokens back to text, so both tokenizers
        # see the identical bytes. 200 windows of 600 of our tokens each.
        data = open_bin(BINS / f"{split}.bin")
        starts = np.linspace(0, len(data) - 700, 200).astype(np.int64)
        nats_gpt2, nbytes, our_tokens, diffs = 0.0, 0, 0, []
        for s in starts:
            ids = data[s:s + 600].astype(np.int64).tolist()
            text = tok.decode_bytes(ids).decode("utf-8", errors="replace")
            g = enc.encode(text, allowed_special="all")[:1024]
            t = torch.tensor([g], device="cuda")
            with torch.no_grad():
                a, _ = ours(t)
                b = hf(t).logits
            diffs.append((a - b).abs().max().item())
            lp = torch.log_softmax(a[0, :-1].float(), -1)
            nll = -lp.gather(1, t[0, 1:, None]).sum().item()
            nats_gpt2 += nll
            # bytes predicted = bytes of tokens 1..n-1 (the first is given)
            nbytes += len(enc.decode_bytes(g[1:]))
            our_tokens += len(tok.encode_ordinary(enc.decode(g[1:])))
        res[split] = dict(
            max_abs_logit_diff=max(diffs),
            gpt2_bits_per_byte=round(nats_gpt2 / math.log(2) / nbytes, 4),
            bytes=nbytes, our_bytes_per_token=round(nbytes / our_tokens, 3),
        )
        print(split, res[split], flush=True)
    write_json("oracle.json", res)


def cmd_bpb(args):
    """GPT-2 124M vs our 5M Stage 06 model on identical text, in bits per byte.

    Loss per token is not comparable across tokenizers (docs/22 §1); bits per
    byte is. Windows of 256 of our tokens (our model's context), decoded to
    text; each model predicts every token of its own encoding after the first,
    and is charged for the bytes of the tokens it predicted.
    """
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tok = Tokenizer.load(str(TOK))
    gpt2 = GPT.from_pretrained("gpt2").cuda().eval()
    ck = torch.load(REPO / "checkpoints" / "06_gpt" / "tied_12000.pt", weights_only=False)
    ours = GPT(GPTConfig(**{**ck["config"], "flash": True})).cuda().eval()
    ours.load_state_dict(ck["model"])
    res = {}
    for split in ("val_code", "val_en"):
        data = open_bin(BINS / f"{split}.bin")
        starts = np.linspace(0, len(data) - 300, 400).astype(np.int64)
        tot = {"gpt2": [0.0, 0], "ours": [0.0, 0]}
        for s in starts:
            text = tok.decode_bytes(data[s:s + 256].astype(np.int64).tolist()).decode("utf-8", errors="replace")
            for name, model, ids in (("gpt2", gpt2, enc.encode(text, allowed_special="all")),
                                     ("ours", ours, tok.encode_ordinary(text)[:256])):
                t = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    lg, _ = model(t)
                lp = torch.log_softmax(lg[0, :-1].float(), -1).gather(1, t[0, 1:, None]).sum().item()
                nbytes = len((enc.decode_bytes(ids[1:]) if name == "gpt2" else tok.decode_bytes(ids[1:])))
                tot[name][0] -= lp
                tot[name][1] += nbytes
        res[split] = {k: round(v[0] / math.log(2) / v[1], 4) for k, v in tot.items()}
        print(split, res[split], flush=True)
    res["params"] = {"gpt2": count_params(gpt2), "ours": count_params(ours)}
    write_json("bpb.json", res)


def cmd_init(args):
    """8b: stream size per block at init, then train 12 layers with and without."""
    torch.manual_seed(0)
    loader = MixedLoader({"code": (BINS / "train_code.bin", 1.0)}, B=8, T=256, device="cuda", seed=0)
    x, _ = loader.next_batch()
    stream = {}
    for L in (12, 48):
        for scaled in (True, False):
            torch.manual_seed(0)
            m = GPT(GPTConfig(n_layer=L, scaled_init=scaled, flash=True)).cuda()
            with torch.no_grad():
                _, _, s = m(x, keep_stream=True)
            stream[f"L{L}_{'scaled' if scaled else 'plain'}"] = [
                round(t.float().pow(2).mean().sqrt().item(), 4) for t in s]
            del m
    runs = []
    for seed in (0, 1):
        for scaled in (True, False):
            rec, _ = train(f"L12_{'scaled' if scaled else 'plain'}",
                           dict(n_layer=12, scaled_init=scaled), seed=seed)
            runs.append(rec)
            write_json("init.json", dict(stream=stream, runs=runs))


def cmd_noshift(args):
    rec, model = train("noshift", dict(), steps=1000, shift=False)
    val = val_windows(256, 512)
    rec["val_loss_on_copy_task"] = evaluate(model, val, shift=False)
    rec["val_loss_next_token"] = evaluate(model, val, shift=True)
    # What does it generate? Greedy, 40 tokens, from a code prompt.
    tok = Tokenizer.load(str(TOK))
    prompt = "def add(a, b):\n    "
    ids = torch.tensor([tok.encode_ordinary(prompt)], device="cuda")
    model.eval()
    with torch.no_grad():
        for _ in range(40):
            logits, _ = model(ids[:, -256:])
            ids = torch.cat([ids, logits[:, -1:].argmax(-1)], 1)
    rec["greedy_sample"] = tok.decode(ids[0].tolist())
    ref, rmodel = train("shift_1000", dict(), steps=1000, shift=True)
    rec["shift_baseline_val"] = ref["val_code_loss"]
    rids = torch.tensor([tok.encode_ordinary(prompt)], device="cuda")
    rmodel.eval()
    with torch.no_grad():
        for _ in range(40):
            logits, _ = rmodel(rids[:, -256:])
            rids = torch.cat([rids, logits[:, -1:].argmax(-1)], 1)
    rec["shift_greedy_sample"] = tok.decode(rids[0].tolist())
    rec["shift_curve"] = ref["curve"]
    print(json.dumps({k: v for k, v in rec.items() if k not in ("curve", "shift_curve")}, indent=1))
    write_json("noshift.json", rec)


def cmd_tie(args):
    runs = []
    for steps in (4000, 12000):
        for tie in (True, False):
            rec, model = train(f"{'tied' if tie else 'untied'}_{steps}", dict(tie=tie), steps=steps)
            runs.append(rec)
            if tie and steps == 12000:                           # Stage 07 samples from this one
                ck = REPO / "checkpoints" / "06_gpt"
                ck.mkdir(parents=True, exist_ok=True)
                torch.save({"config": {**GPTConfig(block_size=256, **rec["cfg"]).__dict__},
                            "model": model.state_dict()}, ck / "tied_12000.pt")
            write_json("tie.json", runs)


def cmd_counts(args):
    rows = []
    for name, kw in [("ours L4 C256", dict()),
                     ("workhorse L8 C512", dict(n_layer=8, n_head=8, n_embd=512, block_size=1024)),
                     ("capstone L12 C768", dict(n_layer=12, n_head=12, n_embd=768, block_size=1024)),
                     ("GPT-2 small", dict(vocab_size=50257, n_layer=12, n_head=12, n_embd=768,
                                          block_size=1024))]:
        cfg = GPTConfig(**kw)
        rows.append(dict(name=name, **count_params_by_hand(cfg), actual=count_params(GPT(cfg))))
        print(rows[-1])
    write_json("counts.json", rows)


def cmd_plot(args):
    VISUALS.mkdir(parents=True, exist_ok=True)
    init = json.loads((RESULTS / "init.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, v in init["stream"].items():
        ax[0].plot(range(len(v)), v, label=k, ls="-" if "scaled" in k else "--",
                   color="C0" if "L12" in k else "C3")
    ax[0].set(xlabel="block (0 = embeddings)", ylabel="rms of residual stream",
              title="Residual stream size at init")
    ax[0].legend()
    for r in init["runs"]:
        c = np.array(r["curve"])
        ax[1].plot(c[:, 0], c[:, 1], color="C0" if "scaled" in r["name"] else "C1",
                   alpha=0.8, label=f"{r['name']} s{r['seed']}: val {r['val_code_loss']}")
    ax[1].set(xlabel="step", ylabel="train loss", ylim=(2.3, 5), title="12 layers: scaled vs plain init")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "06_init.png", dpi=130)

    tie = json.loads((RESULTS / "tie.json").read_text())
    ns = json.loads((RESULTS / "noshift.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c = np.array(ns["curve"])
    ax[0].plot(c[:, 0], c[:, 1], color="C3", label="target = input (no shift)")
    c = np.array(ns["shift_curve"])
    ax[0].plot(c[:, 0], c[:, 1], color="C0", label="target = next token")
    ax[0].axhline(math.log(8196), color="gray", ls=":", label="ln V = 9.01")
    ax[0].set(xlabel="step", ylabel="train loss", title="The shift is the task", yscale="log")
    ax[0].legend()
    for r in tie:
        c = np.array(r["curve"])
        ax[1].plot(c[:, 0], c[:, 1], color="C0" if r["name"].startswith("tied") else "C1",
                   ls="-" if "12000" in r["name"] else "--",
                   label=f"{r['name']} ({r['params'] / 1e6:.2f}M): val {r['val_code_loss']}")
    ax[1].set(xlabel="step", ylabel="train loss", ylim=(2.3, 4), title="Tied vs untied head")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VISUALS / "06_shift_tie.png", dpi=130)
    print("saved figures")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["oracle", "bpb", "init", "noshift", "tie", "counts", "plot"])
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
