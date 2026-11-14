"""Stage 19 experiments. docs/19_rlhf_dpo.md §7.

    uv run python stages/19_dpo/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse --beta 0.1
    uv run python stages/19_dpo/run.py --plot --tag workhorse
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "16_sft"))
sys.path.insert(0, str(REPO / "stages" / "18_reward"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reward import RewardModel, csn_pairs, mbpp_pairs  # noqa: E402
from sft import collate, render  # noqa: E402

from llmfs.model import build_model, load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.post.dpo import dpo_loss, sequence_logprob  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

_spec = importlib.util.spec_from_file_location("sft_run", REPO / "stages" / "16_sft" / "run.py")
_sft_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sft_run)

RESULTS = REPO / "results" / "19_dpo"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "19_dpo"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


MAX_LEN = 512


def to_ex(tok, p, side, max_len=None):
    max_len = max_len or MAX_LEN
    ids, labels = render(tok, p["prompt"], p[side])
    return dict(ids=ids[:max_len], labels=labels[:max_len])


@torch.no_grad()
def implicit_acc(policy, ref, tok, pairs, beta, B=16):
    pad = tok.special_tokens["<|pad|>"]
    margins = []
    policy.eval()
    for i in range(0, len(pairs), B):
        chunk = pairs[i:i + B]
        xc, yc = collate([to_ex(tok, p, "chosen") for p in chunk], pad)
        xr, yr = collate([to_ex(tok, p, "rejected") for p in chunk], pad)
        m = beta * ((sequence_logprob(policy, xc, yc) - sequence_logprob(ref, xc, yc))
                    - (sequence_logprob(policy, xr, yr) - sequence_logprob(ref, xr, yr)))
        margins += m.tolist()
    policy.train()
    # ties (exactly 0 at the start, policy == reference) count as half right
    return round(float(np.mean([1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in margins])), 4)


@torch.no_grad()
def sample_stats(policy, ref, rm, tok, prompts, n_new=200):
    """Mean reward-model score and mean KL(policy || ref) on the policy's own greedy samples."""
    sys.path.insert(0, str(REPO / "stages" / "07_generation"))
    from sample import generate_cached
    pad = tok.special_tokens["<|pad|>"]
    E = tok.special_tokens["<|im_end|>"]
    policy.eval()
    rewards, kls = [], []
    for p in prompts:
        ids = torch.tensor([render(tok, p)[0]], device="cuda")
        out = generate_cached(policy, ids, n_new, eot=E, temperature=0.0)
        seq = out[0].tolist()
        if seq[-1] != E:
            seq.append(E)
        x = torch.tensor([seq], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rewards.append(rm(x, torch.tensor([len(seq)], device="cuda")).item())
            lp = torch.log_softmax(policy(x)[0].float(), -1)
            lr = torch.log_softmax(ref(x)[0].float(), -1)
        T0 = ids.shape[1]
        kls.append((lp[0, T0 - 1:-1].exp() * (lp[0, T0 - 1:-1] - lr[0, T0 - 1:-1])).sum(-1).mean().item())
    policy.train()
    return round(float(np.mean(rewards)), 4), round(float(np.mean(kls)), 4)


def main(a):
    tok = Tokenizer.load(str(TOK))
    policy, _ = load_model(a.sft, device="cuda")
    policy.grad_checkpointing = a.act_ckpt
    global MAX_LEN
    MAX_LEN = min(512, policy.cfg.block_size)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rm = None
    rm_path = REPO / "checkpoints" / "18_reward" / f"{a.tag}.pt"
    if rm_path.exists():
        ck = torch.load(rm_path, map_location="cuda", weights_only=False)
        rm = RewardModel(build_model(ck["config"])).cuda()
        rm.load_state_dict(ck["model"])
        rm.eval()
    train = csn_pairs("train", limit=20000, seed=1)
    mb = mbpp_pairs(("test", "validation"))
    probe_prompts = [p["prompt"] for p in csn_pairs("validation", limit=64, seed=3)]
    pad = tok.special_tokens["<|pad|>"]
    opt = policy.configure_optimizer(a.lr, weight_decay=0.0)
    rng = random.Random(0)
    hist = []
    B = 8
    r0, k0 = sample_stats(policy, ref, rm, tok, probe_prompts) if rm else (None, None)
    hist.append(dict(step=0, mbpp_pref_acc=implicit_acc(policy, ref, tok, mb[:300], a.beta), rm=r0, kl=k0))
    for step in range(a.steps):
        f = 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 30) * (0.1 + 0.9 * f)
        batch = rng.sample(train, B)
        xc, yc = collate([to_ex(tok, p, "chosen") for p in batch], pad)
        xr, yr = collate([to_ex(tok, p, "rejected") for p in batch], pad)
        with torch.no_grad():
            ref_c, ref_r = sequence_logprob(ref, xc, yc), sequence_logprob(ref, xr, yr)
        pi_c, pi_r = sequence_logprob(policy, xc, yc), sequence_logprob(policy, xr, yr)
        loss, rc, rr = dpo_loss(pi_c, pi_r, ref_c, ref_r, a.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            rec = dict(step=step + 1, loss=round(loss.item(), 4), reward_chosen=round(rc.mean().item(), 4),
                       reward_rejected=round(rr.mean().item(), 4),
                       logp_chosen_delta=round((pi_c - ref_c).mean().item(), 3),
                       logp_rejected_delta=round((pi_r - ref_r).mean().item(), 3))
            if step % 200 == 0 or step == a.steps - 1:
                rec["mbpp_pref_acc"] = implicit_acc(policy, ref, tok, mb[:300], a.beta)
                if rm:
                    rec["rm"], rec["kl"] = sample_stats(policy, ref, rm, tok, probe_prompts)
                print(rec, flush=True)
            hist.append(rec)
    res = dict(beta=a.beta, steps=a.steps, hist=hist,
               final_mbpp_pref_acc=implicit_acc(policy, ref, tok, mb, a.beta))
    if rm:
        res["final_rm"], res["final_kl"] = sample_stats(policy, ref, rm, tok, probe_prompts)
    sc = _sft_run.code_scores(policy, tok, _sft_run.chat_fn(tok))
    res.update(humaneval_pass1=sc["humaneval_pass1"], mbpp_pass1=sc["mbpp_pass1"])
    sft_res = REPO / "results" / "16_sft" / f"{a.tag}.json"
    if sft_res.exists():
        s = json.loads(sft_res.read_text())["after"]["chat_mode"]
        res["sft_humaneval_pass1"], res["sft_mbpp_pass1"] = s["humaneval_pass1"], s["mbpp_pass1"]
        # head to head on problems: solved by one and not the other
        ds, ss = set(sc["humaneval_solved"] + sc["mbpp_solved"]), set(s["humaneval_solved"] + s["mbpp_solved"])
        res["solved_dpo_only"], res["solved_sft_only"] = len(ds - ss), len(ss - ds)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{a.tag}_beta{a.beta}.json").write_text(json.dumps(res, indent=1))
    if a.save:
        CKPT.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CKPT / f"{a.tag}.pt", policy)
    print({k: v for k, v in res.items() if k != "hist"}, flush=True)


def plot(a):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    files = sorted(RESULTS.glob(f"{a.tag}_beta*.json"), key=lambda p: float(p.stem.split("beta")[1]))
    for i, f in enumerate(files):
        r = json.loads(f.read_text())
        h = [x for x in r["hist"] if "reward_chosen" in x]
        b = r["beta"]
        if abs(b - 0.1) < 1e-9:
            ax[0].plot([x["step"] for x in h], [x["reward_chosen"] for x in h], color="C2", label="chosen")
            ax[0].plot([x["step"] for x in h], [x["reward_rejected"] for x in h], color="C3", label="rejected")
            ax[0].plot([x["step"] for x in h], [x["logp_chosen_delta"] * 0.1 for x in h], ":", color="C2",
                       label="β·Δlog π(chosen)")
        ax[1].plot(r.get("final_kl") or 0, r.get("final_rm") or 0, "o", color=f"C{i}", label=f"β={b}")
        if i == 0:
            h0 = [x for x in r["hist"] if "rm" in x and x.get("rm") is not None][0]
            ax[1].plot(0, h0["rm"], "*", ms=14, color="black", label="SFT (start)")
        ax[2].bar(i, r["final_mbpp_pref_acc"], color=f"C{i}")
        ax[2].text(i, r["final_mbpp_pref_acc"], f"{r['final_mbpp_pref_acc']:.0%}\nHE {r['humaneval_pass1']:.1%}",
                   ha="center", va="bottom", fontsize=8)
    ax[0].axhline(0, color="gray", ls=":")
    ax[0].set(xlabel="step", ylabel="implicit reward β·log(π/π_ref)", title="DPO margins (β = 0.1)")
    ax[0].legend()
    ax[1].set(xlabel="KL(policy ‖ SFT) on own samples", ylabel="reward model score", title="Reward vs drift, by β")
    ax[1].legend(fontsize=8)
    ax[2].set_xticks(range(len(files)), [f"β={json.loads(f.read_text())['beta']}" for f in files])
    ax[2].set(ylim=(0, 1.1), ylabel="ranks working code above its mutant", title="MBPP verified pairs")
    fig.tight_layout()
    fig.savefig(VISUALS / "19_dpo.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--act-ckpt", action="store_true", help="activation checkpointing (needed for the 92M capstone on 8 GB)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--save", action="store_true")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    plot(a) if a.plot else main(a)
