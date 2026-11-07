"""HumanEval and MBPP, pass@k. docs/22_evaluation.md §4.

    problems = load_humaneval()      # 164
    problems = load_mbpp("test")     # 500
    res = evaluate(model, tok, problems, n=10, temperature=0.8, ks=(1, 10))

pass@k uses the unbiased estimator of Chen et al. 2021: with n samples per
problem of which c pass, pass@k = 1 - C(n-c, k) / C(n, k). Taking the best of
k fresh samples directly would need k samples for each k and is noisier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from .execution import run_many

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"


def load_humaneval() -> list[dict]:
    t = pq.read_table(RAW / "humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet").to_pylist()
    return [dict(task_id=r["task_id"], prompt=r["prompt"],
                 test=r["test"] + f"\ncheck({r['entry_point']})\n", entry_point=r["entry_point"])
            for r in t]


def mbpp_prompt(r: dict) -> str:
    """MBPP gives a sentence and three asserts. We show the first assert so the
    model knows the function name and signature (the standard 'MBPP with one
    test in the prompt' setup), as a docstring a base model can continue."""
    first = r["test_list"][0]
    return '"""' + r["text"].strip() + '\n' + first.strip() + '\n"""\n'


def load_mbpp(split: str = "test") -> list[dict]:
    t = pq.read_table(RAW / "mbpp" / "full" / f"{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for r in t:
        tests = "\n".join(r["test_list"]) + "\n"
        setup = r.get("test_setup_code") or ""
        out.append(dict(task_id=f"mbpp/{r['task_id']}", prompt=mbpp_prompt(r),
                        test=setup + "\n" + tests, text=r["text"], reference=r["code"]))
    return out


def truncate(completion: str, is_mbpp: bool = False) -> str:
    """Cut a completion where the function ends. Base models keep writing.

    HumanEval: the prompt already opened the function, so stop at the first
    line that starts at column 0. MBPP: the model writes the whole function,
    so allow the first `def`, stop at the next top-level statement.
    """
    if is_mbpp:
        lines = completion.split("\n")
        out, started = [], False
        for ln in lines:
            if ln.startswith(("def ", "import ", "from ")) and not started:
                started = ln.startswith("def ")
                out.append(ln)
                continue
            if started and ln and not ln[0].isspace():
                break
            out.append(ln)
        return "\n".join(out)
    lines = completion.split("\n")
    out = []
    for ln in lines:
        if ln and not ln[0].isspace():
            break
        out.append(ln)
    return "\n".join(out)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


@torch.no_grad()
def ended(text: str, is_mbpp: bool) -> bool:
    """True once a completion can no longer change after truncate(): a later
    line already starts at column 0. Lets generation stop early; the
    truncated program is identical either way."""
    return len(truncate(text, is_mbpp)) < len(text.rstrip("\n")) - 1 and "\n" in text.strip("\n")


def generate_batch(model, tok, prompt, n: int, max_new: int, temperature: float,
                   top_p: float, seed: int, stop_fn=None) -> list[str]:
    """prompt: a string (encoded as plain text) or a list of token ids (chat).
    stop_fn(text) -> bool: optional early stop per row, checked every 16 tokens."""
    sys.path.insert(0, str(REPO / "stages" / "07_generation"))
    from sample import sample_next
    pids = prompt if isinstance(prompt, list) else tok.encode_ordinary(prompt)
    ids = torch.tensor([pids] * n, device="cuda")
    stops = {tok.eot_id, tok.special_tokens.get("<|im_end|>", -1)}
    block = model.cfg.block_size
    ids = ids[:, -(block - max_new):]
    T = ids.shape[1]
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    cache = model.make_cache(n, T + max_new, "cuda", torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, cache=cache, last_only=True)
    out = []
    done = torch.zeros(n, dtype=torch.bool, device="cuda")
    for _ in range(max_new):
        nxt = sample_next(logits[:, -1], temperature=temperature, top_p=top_p, generator=g)
        out.append(nxt)
        done |= torch.isin(nxt[:, 0], torch.tensor(sorted(stops), device="cuda"))
        if stop_fn is not None and len(out) % 16 == 0:
            gen_so_far = torch.cat(out, 1).tolist()
            for r, row in enumerate(gen_so_far):
                if not done[r] and stop_fn(tok.decode(row)):
                    done[r] = True
        if done.all():
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(nxt, cache=cache)
    gen = torch.cat(out, 1).tolist()
    texts = []
    for row in gen:
        cut = [row.index(t) for t in stops if t in row]
        if cut:
            row = row[:min(cut)]
        texts.append(tok.decode(row))
    return texts


def evaluate(model, tok, problems: list[dict], n: int = 1, temperature: float = 0.0,
             top_p: float = 0.95, ks=(1,), max_new: int = 256, mbpp: bool = False,
             seed: int = 0, chat=None, keep_samples: int = 3) -> dict:
    """chat: None for a base model (prompt continued as text), or a function
    user_text -> prompt token ids (Stage 16's chat template). A chat model
    writes the whole function, so its completion is cut like MBPP's and the
    HumanEval prompt is kept in front only for its imports and helpers."""
    model.eval()
    progs, owners, comps = [], [], []
    for pi, p in enumerate(problems):
        # Token-boundary trap (Stage 22 §3): HumanEval prompts end in a bare
        # "\n" token. In our training data newline + indentation is one token
        # ("\n    "), so a bare "\n" token means "next line starts at column 0"
        # and a base model starts a new def. Ending the prompt at the docstring
        # lets the model write "\n    ..." itself. The program is rebuilt from
        # the same stripped prompt, so nothing else changes.
        base_text = p["prompt"] if (chat or mbpp) else p["prompt"].rstrip("\n")
        prompt = chat(p["prompt"]) if chat else base_text
        stop = (lambda t: ended(t, True)) if (chat or mbpp) else (lambda t: ended(t, False))
        texts = generate_batch(model, tok, prompt, n, max_new, temperature, top_p, seed + pi, stop_fn=stop)
        for t in texts:
            if chat:
                body = truncate(t, is_mbpp=True)
                full = ("" if mbpp else p["prompt"] + "\n") + body
            else:
                body = truncate(t, is_mbpp=mbpp)
                full = (body if mbpp else base_text + body)
            progs.append(full + "\n\n" + p["test"])
            owners.append(pi)
            comps.append(body)
    results = run_many(progs)
    per = {}
    for (ok, why), pi, c in zip(results, owners, comps):
        per.setdefault(pi, []).append((ok, why, c))
    scores = {f"pass@{k}": float(np.mean([pass_at_k(len(v), sum(o for o, _, _ in v), k)
                                          for v in per.values()])) for k in ks if k <= n}
    fails = {}
    for v in per.values():
        for ok, why, _ in v:
            if not ok:
                key = "timeout" if why == "timeout" else why.split(":")[0][:40]
                fails[key] = fails.get(key, 0) + 1
    examples = [dict(task_id=problems[pi]["task_id"], passed=v[0][0], completion=v[0][2][:600])
                for pi, v in list(per.items())[:keep_samples]]
    solved = [problems[pi]["task_id"] for pi, v in per.items() if any(o for o, _, _ in v)]
    return dict(**scores, n=n, temperature=temperature, problems=len(problems),
                failure_kinds=dict(sorted(fails.items(), key=lambda kv: -kv[1])[:8]),
                solved=solved, examples=examples)
