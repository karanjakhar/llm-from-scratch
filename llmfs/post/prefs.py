"""Preference pairs for code: correct function (chosen) vs a buggy mutant (rejected).

docs/18_reward_model.md §5. No human labels: a bug is introduced by one AST
mutation (flip a comparison, change an operator, nudge a constant, ...). On
MBPP the mutant is *verified* to fail the tests by running them; on
CodeSearchNet there are no tests, so a mutant is only very likely broken.

Both sides are passed through ast.unparse, so formatting is identical and
the only difference between chosen and rejected is the bug. Without that, a
reward model can learn "original formatting good, unparsed formatting bad",
which is a shortcut, not a judgement of code (docs §5, the trap).
"""

from __future__ import annotations

import ast
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)   # old CSN code has "\\w" strings
import copy
import random

CMP_FLIP = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt,
            ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
            ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BIN_FLIP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.FloorDiv, ast.FloorDiv: ast.Mult,
            ast.Div: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult}
BOOL_FLIP = {ast.And: ast.Or, ast.Or: ast.And}


def _sites(tree):
    """Every place one mutation could go, as (kind, node)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            out += [("cmp", n)]
        elif isinstance(n, ast.BinOp) and type(n.op) in BIN_FLIP:
            out += [("bin", n)]
        elif isinstance(n, ast.BoolOp):
            out += [("bool", n)]
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            out += [("const", n)]
        elif isinstance(n, ast.Constant) and isinstance(n.value, bool):
            out += [("boolconst", n)]
        elif isinstance(n, ast.Return) and n.value is not None:
            out += [("return", n)]
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            out += [("not", n)]
    return out


def mutate(src: str, rng: random.Random) -> tuple[str, str] | None:
    """-> (canonical original, mutant) both via ast.unparse, or None."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    original = ast.unparse(tree)
    sites = _sites(tree)
    rng.shuffle(sites)
    for kind, _ in sites:
        t2 = copy.deepcopy(tree)
        # find the same site in the copy by walk order
        idx = [i for i, (k, n) in enumerate(_sites(t2)) if k == kind]
        if not idx:
            continue
        node = _sites(t2)[rng.choice(idx)][1]
        if kind == "cmp":
            op = node.ops[0]
            if type(op) not in CMP_FLIP:
                continue
            node.ops[0] = CMP_FLIP[type(op)]()
        elif kind == "bin":
            node.op = BIN_FLIP[type(node.op)]()
        elif kind == "bool":
            node.op = BOOL_FLIP[type(node.op)]()
        elif kind == "const":
            node.value = node.value + rng.choice([-1, 1])
        elif kind == "boolconst":
            node.value = not node.value
        elif kind == "return":
            node.value = ast.Constant(None)
        elif kind == "not":
            # drop the `not`
            parent_src = ast.unparse(t2)
            new = parent_src.replace("not " + ast.unparse(node.operand), ast.unparse(node.operand), 1)
            if new == parent_src:
                continue
            return original, new
        mutant = ast.unparse(t2)
        if mutant != original:
            return original, mutant
    return None


def pair_texts(header: str, chosen: str, rejected: str) -> dict:
    return dict(prompt=header, chosen=chosen, rejected=rejected)
