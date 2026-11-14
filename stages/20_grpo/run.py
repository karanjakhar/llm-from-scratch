"""Stage 20: GRPO with reward = unit tests pass. docs/20_reasoning_rlvr.md.

    uv run python stages/20_grpo/run.py --sft checkpoints/16_sft/workhorse.pt --tag workhorse
    uv run python stages/20_grpo/run.py ... --format-reward           # the ablation
    uv run python stages/20_grpo/run.py --plot --tag workhorse
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stages" / "16_sft"))
sys.path.insert(0, str(REPO / "stages" / "07_generation"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sample import sample_next  # noqa: E402
from sft import IGNORE, render  # noqa: E402

from llmfs.eval.codeeval import evaluate, load_humaneval, load_mbpp, truncate  # noqa: E402
from llmfs.eval.execution import run_many  # noqa: E402
from llmfs.model import load_model  # noqa: E402
from llmfs.model.gpt import save_checkpoint  # noqa: E402
from llmfs.post.grpo import group_advantages, grpo_loss, token_logprobs  # noqa: E402
from llmfs.tokenizer import Tokenizer  # noqa: E402

RESULTS = REPO / "results" / "20_grpo"
VISUALS = REPO / "docs" / "visuals"
CKPT = REPO / "checkpoints" / "20_grpo"
TOK = REPO / "data" / "tokenizers" / "split_8192.json"


@torch.no_grad()
def rollouts(model, tok, prompt_ids, G, max_new, temperature, g):
    """G samples for one prompt, batched with a KV cache. -> list of token lists (completion only)."""
    E = tok.special_tokens["<|im_end|>"]
    ids = torch.tensor([prompt_ids] * G, device="cuda")
    cache = model.make_cache(G, len(prompt_ids) + max_new, "cuda", torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, cache=cache, last_only=True)
    out, done = [], torch.zeros(G, dtype=torch.bool, device="cuda")
    for _ in range(max_new):
        nxt = sample_next(logits[:, -1], temperature=temperature, generator=g)
        nxt[done] = E
        out.append(nxt)
        done |= nxt[:, 0] == E
        if done.all():
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(nxt, cache=cache)
    rows = torch.cat(out, 1).tolist()
    res = []
    for r in rows:
        if E in r:
            r = r[:r.index(E) + 1]                    # keep the stop token: it is part of the action
        res.append(r)
    return res


def fn_name(problem):
    return problem["test"].split("assert ")[1].split("(")[0].strip()


def rewards_for(problem, completions, tok, use_format):
    E = tok.special_tokens["<|im_end|>"]
    texts = [tok.decode([t for t in c if t != E]) for c in completions]
    bodies = [truncate(t, is_mbpp=True) for t in texts]
    res = run_many([b + "\n\n" + problem["test"] for b in bodies], timeout=5.0, workers=16)
    out = []
    name = fn_name(problem)
    for c, b, (ok, _) in zip(completions, bodies, res):
        r = 1.0 if ok else 0.0
        if use_format:
            try:
                tree = ast.parse(b)
                parses = True
            except SyntaxError:
                tree, parses = None, False
            has_def = parses and any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))
            r += 0.1 * parses + 0.1 * has_def + 0.1 * (c[-1] == E)
        out.append(r)
    return out, [ok for ok, _ in res]


def batch_from(prompts, completions, pad):
    seqs = [p + c for p, c in zip(prompts, completions)]
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    y = torch.full((len(seqs), T), IGNORE, dtype=torch.long)
    for i, (p, c) in enumerate(zip(prompts, completions)):
        s = p + c
        x[i, :len(s)] = torch.tensor(s)
        y[i, len(p) - 1:len(s) - 1] = torch.tensor(c)            # predict completion tokens only
    return x.cuda(), y.cuda()


def quick_eval(model, tok, chat, n_mbpp=500):
    mb = evaluate(model, tok, load_mbpp("test")[:n_mbpp], n=1, temperature=0.0, mbpp=True, chat=chat, max_new=256)
    he = evaluate(model, tok, load_humaneval(), n=1, temperature=0.0, chat=chat, max_new=256)
    return round(mb["pass@1"], 4), round(he["pass@1"], 4)


def easy_eval(model, tok, chat):
    sys.path.insert(0, str(HERE))
    from easy_tasks import HELDOUT
    r = evaluate(model, tok, HELDOUT(), n=1, temperature=0.0, mbpp=True, chat=chat, max_new=128)
    return round(r["pass@1"], 4)


def main(a):
    tok = Tokenizer.load(str(TOK))
    policy, _ = load_model(a.sft, device="cuda")
    policy.grad_checkpointing = a.act_ckpt
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    chat = lambda u: render(tok, u)[0]
    if a.tasks == "easy":
        sys.path.insert(0, str(HERE))
        from easy_tasks import TRAIN
        problems = TRAIN()                       # docs §8b: a curriculum with pass@8 > 0
    else:
        problems = load_mbpp("train") + load_mbpp("validation")
    rng = random.Random(0)
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    opt = policy.configure_optimizer(a.lr, weight_decay=0.0)
    pad = tok.special_tokens["<|pad|>"]
    hist = []
    mb0, he0 = quick_eval(policy, tok, chat)
    hist.append(dict(step=0, mbpp_test=mb0, humaneval=he0, easy_heldout=easy_eval(policy, tok, chat)))
    print(hist[-1], flush=True)
    solved_ever = set()
    t0 = time.time()
    for step in range(1, a.steps + 1):
        P_ids, C, A, R_all, lens_pass, lens_fail = [], [], [], [], [], []
        tried = with_signal = 0
        while len(set(map(tuple, P_ids))) < a.prompts and tried < a.max_tries:
            prob = rng.choice(problems)
            tried += 1
            pids = chat(prob["prompt"])
            comps = rollouts(policy, tok, pids, a.G, a.max_new, a.rollout_temp, g)
            r, passed = rewards_for(prob, comps, tok, a.format_reward)
            R_all += r
            for c, ok in zip(comps, passed):
                (lens_pass if ok else lens_fail).append(len(c))
            if any(passed):
                solved_ever.add(prob["task_id"])
            rt = torch.tensor([r])
            if rt.std() == 0:
                continue                                         # no signal in this group (docs §3)
            with_signal += 1
            adv = group_advantages(rt)[0]
            P_ids += [pids] * a.G
            C += comps
            A += adv.tolist()
        rec = dict(step=step, reward_mean=round(float(np.mean(R_all)), 4), groups_tried=tried,
                   groups_with_signal=with_signal,
                   pass_rate_rollouts=round(len(lens_pass) / max(1, len(lens_pass) + len(lens_fail)), 4),
                   len_pass=round(float(np.mean(lens_pass)), 1) if lens_pass else None,
                   len_fail=round(float(np.mean(lens_fail)), 1) if lens_fail else None,
                   solved_ever=len(solved_ever))
        if A:
            x, y = batch_from(P_ids, C, pad)
            adv = torch.tensor(A, device="cuda")
            with torch.no_grad():
                old_lp, mask = token_logprobs(policy, x, y)
                ref_lp, _ = token_logprobs(ref, x, y)
            for _ in range(a.epochs):
                lp, mask = token_logprobs(policy, x, y)
                loss, st = grpo_loss(lp, old_lp, ref_lp, mask, adv, clip=0.2, beta=a.beta)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
            rec.update(loss=round(loss.item(), 4), kl=round(st["kl"], 5), clipped=round(st["clipped"], 4))
        if step % a.eval_every == 0 or step == a.steps:
            rec["easy_heldout"] = easy_eval(policy, tok, chat)
            if step == a.steps or a.tasks != "easy":
                rec["mbpp_test"], rec["humaneval"] = quick_eval(policy, tok, chat)
            rec["elapsed_min"] = round((time.time() - t0) / 60, 1)
        hist.append(rec)
        print(rec, flush=True)
        tag = a.tag + ("_easy" if a.tasks == "easy" else "") + ("_format" if a.format_reward else "")
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{tag}.json").write_text(json.dumps(dict(args=vars(a), hist=hist), indent=1))
    if a.save:
        CKPT.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CKPT / f"{a.tag}.pt", policy)


def plot(a):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, suffix in enumerate(("_easy", "_easy_format")):
        p = RESULTS / f"{a.tag}{suffix}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())["hist"]
        lab = "outcome + format" if "format" in suffix else "outcome only"
        ev = [x for x in h if "easy_heldout" in x]
        ax[0].plot([x["step"] for x in ev], [x["easy_heldout"] for x in ev], "o-", color=f"C{i}",
                   label=f"{lab}: held-out easy tasks")
        tr = [x for x in h if "pass_rate_rollouts" in x]
        ax[1].plot([x["step"] for x in tr], np.convolve([x["pass_rate_rollouts"] for x in tr], np.ones(5) / 5, "same"),
                   color=f"C{i}", label=f"{lab}: rollout pass rate")
        ax[2].plot([x["step"] for x in tr if x["len_fail"]], [x["len_fail"] for x in tr if x["len_fail"]],
                   color=f"C{i}", alpha=0.7, label=f"{lab}: failing")
        ax[2].plot([x["step"] for x in tr if x["len_pass"]], [x["len_pass"] for x in tr if x["len_pass"]], "o",
                   color=f"C{i}", ms=3, label=f"{lab}: passing")
    ax[0].set(xlabel="GRPO step", ylabel="pass@1 (greedy)", title="Held-out easy tasks, pass@1")
    ax[1].set(xlabel="GRPO step", ylabel="fraction of rollouts passing", title="Training rollouts passing (easy train)")
    ax[2].set(xlabel="GRPO step", ylabel="completion tokens", title="Response length")
    for x in ax:
        x.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(VISUALS / "20_grpo.png", dpi=130)
    print("saved")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="")
    p.add_argument("--tag", required=True)
    p.add_argument("--act-ckpt", action="store_true", help="activation checkpointing (needed for the 92M capstone on 8 GB)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--max-tries", type=int, default=24)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--format-reward", action="store_true")
    p.add_argument("--tasks", choices=["mbpp", "easy"], default="mbpp")
    p.add_argument("--rollout-temp", type=float, default=1.0)
    p.add_argument("--save", action="store_true")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    plot(a) if a.plot else main(a)
