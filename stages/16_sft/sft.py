"""Stage 16: supervised fine-tuning. docs/16_sft.md.

Data: CodeSearchNet Python, docstring -> function. Each example becomes

    <|im_start|>user\n{signature + docstring}<|im_end|>\n<|im_start|>assistant\n{function}<|im_end|>

and the loss is computed on the assistant part only (docs §3). Batches are
right-padded; with right padding the ordinary causal mask is already
correct, because pad tokens come after every real token (docs §4).
"""

from __future__ import annotations

import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)   # old CSN code has "\\w" strings
import random
import textwrap
from pathlib import Path

import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
CSN = REPO / "data" / "raw" / "codesearchnet" / "python"
IGNORE = -100


# ---------------------------------------------------------------------------
# The chat template (docs §2)
# ---------------------------------------------------------------------------

def render(tok, user: str, assistant: str | None = None) -> tuple[list[int], list[int]]:
    """-> (ids, labels). labels are IGNORE everywhere except the assistant's reply
    and its closing <|im_end|> (the model must learn to stop)."""
    S, E = tok.special_tokens["<|im_start|>"], tok.special_tokens["<|im_end|>"]
    nl = tok.encode_ordinary("\n")
    prompt = ([S] + tok.encode_ordinary("user\n") + tok.encode_ordinary(user) + [E] + nl
              + [S] + tok.encode_ordinary("assistant\n"))
    if assistant is None:
        return prompt, [IGNORE] * len(prompt)
    reply = tok.encode_ordinary(assistant) + [E]
    return prompt + reply, [IGNORE] * len(prompt) + reply


# ---------------------------------------------------------------------------
# Building pairs from CodeSearchNet (docs §5)
# ---------------------------------------------------------------------------

def split_function(src: str) -> tuple[str, str] | None:
    """'def f(x):\\n    \"\"\"doc\"\"\"\\n    body' -> (header incl. docstring, full function).

    Keeps top-level-style functions only: no `self`/`cls`, parses on its own,
    has a docstring, has a body after the docstring.
    """
    src = textwrap.dedent(src)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return None
    fn = tree.body[0]
    args = [a.arg for a in fn.args.args]
    if args and args[0] in ("self", "cls"):
        return None
    if not (fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)) or len(fn.body) < 2:
        return None
    lines = src.split("\n")
    header = "\n".join(lines[:fn.body[0].end_lineno]) + "\n"    # signature + docstring
    return header, src.rstrip() + "\n"


def build_pairs(tok, split: str = "train", max_tokens: int = 512, limit: int | None = None,
                seed: int = 0) -> list[dict]:
    t = pq.read_table(CSN / f"{split}-00000-of-00001.parquet", columns=["whole_func_string"])
    out = []
    for src in t.column("whole_func_string").to_pylist():
        sp = split_function(src)
        if sp is None:
            continue
        header, func = sp
        ids, labels = render(tok, header, func)
        if len(ids) > max_tokens:
            continue
        out.append(dict(user=header, assistant=func, ids=ids, labels=labels))
        if limit and len(out) >= limit:
            break
    random.Random(seed).shuffle(out)
    return out


def collate(examples: list[dict], pad_id: int, device="cuda", include_prompt_loss: bool = False):
    """Right-pad to the longest example. Returns x, y (shifted) for the model."""
    T = max(len(e["ids"]) for e in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), T), IGNORE, dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["ids"])
        ids[i, :n] = torch.tensor(e["ids"])
        lab[i, :n] = torch.tensor(e["ids"] if include_prompt_loss else e["labels"])
    # next-token shift (Stage 06 §2): position t predicts label t+1
    x, y = ids[:, :-1], lab[:, 1:]
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Training (the Stage 08 recipe, small lr, few epochs)
# ---------------------------------------------------------------------------

def train_sft(model, train_ex, val_ex, tok, steps=2000, B=16, lr=1e-4, warmup=50,
              include_prompt_loss=False, eval_every=250, seed=0, log=print, micro=None):
    """micro: split each batch of B into chunks of this size and accumulate
    gradients (added for the 92M capstone, where B=16 at 512 tokens runs out
    of 8 GB). Each chunk's mean loss is weighted by its share of the batch's
    target tokens, so the summed gradient equals the full batch's exactly."""
    import math
    pad = tok.special_tokens["<|pad|>"]
    opt = model.configure_optimizer(lr, weight_decay=0.0)
    rng = random.Random(seed)
    hist = []
    model.train()
    for step in range(steps):
        p = (step - warmup) / max(1, steps - warmup)
        cur = lr * (step + 1) / warmup if step < warmup else lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        for g in opt.param_groups:
            g["lr"] = cur
        batch = rng.sample(train_ex, B)
        x, y = collate(batch, pad, include_prompt_loss=include_prompt_loss)
        opt.zero_grad(set_to_none=True)
        if micro is None or micro >= B:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
        else:
            n_tok = (y != IGNORE).sum()                       # targets in the whole batch
            loss = torch.zeros((), device=x.device)
            for i in range(0, B, micro):
                xm, ym = x[i:i + micro], y[i:i + micro]       # (micro, T)
                w = (ym != IGNORE).sum() / n_tok              # this chunk's share of targets
                if w == 0:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, lm = model(xm, ym)                     # mean over this chunk's targets
                (lm * w).backward()
                loss = loss + lm.detach() * w
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            v = sft_val_loss(model, val_ex, tok)
            hist.append(dict(step=step, loss=round(loss.item(), 4), val_completion_loss=v))
            log(f"sft step {step} loss {loss.item():.4f} val(completion) {v}")
    return hist


@torch.no_grad()
def sft_val_loss(model, val_ex, tok, B=32):
    """Loss on assistant tokens only, the number SFT optimises."""
    model.eval()
    pad = tok.special_tokens["<|pad|>"]
    tot, n = 0.0, 0
    for i in range(0, len(val_ex), B):
        x, y = collate(val_ex[i:i + B], pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        lp = torch.log_softmax(logits.float(), -1)
        mask = y != IGNORE
        tot += -lp.gather(2, y.clamp(min=0)[..., None])[..., 0][mask].sum().item()
        n += mask.sum().item()
    model.train()
    return round(tot / n, 4)


def chat_prompt_humaneval(p: dict) -> str:
    """HumanEval's prompt already is 'signature + docstring': the same shape as
    our SFT user turns. Used with llmfs.eval.codeeval via prompt_fn."""
    return p["prompt"]
