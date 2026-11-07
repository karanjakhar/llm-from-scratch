"""An easy, verifiable curriculum for GRPO. docs/20_reasoning_rlvr.md §8b.

GRPO learns only from groups where some samples pass and some fail. On MBPP
our models pass ~0% (§8a), so every group is all-fail and there is no signal.
These templated tasks are within reach of a 30-100M model, so pass rates
start above zero and GRPO has something to reinforce.

Each task is MBPP-shaped: a one-line description plus one visible assert in
a docstring prompt, three hidden asserts as the test. Names and constants
are randomised, so train and held-out tasks share templates but not answers.
"""

from __future__ import annotations

import random

# (description template, function-name template, signature args, reference body, input generator)
TEMPLATES = [
    ("Write a function {f} that returns x plus {k}.", "add_{w}", "x", "return x + {k}", lambda r: r.randint(-20, 50)),
    ("Write a function {f} that returns x minus {k}.", "subtract_{w}", "x", "return x - {k}", lambda r: r.randint(-20, 50)),
    ("Write a function {f} that returns x multiplied by {k}.", "times_{w}", "x", "return x * {k}", lambda r: r.randint(-9, 30)),
    ("Write a function {f} that returns the square of x.", "square_{w}", "x", "return x * x", lambda r: r.randint(-12, 12)),
    ("Write a function {f} that returns True if n is even, else False.", "is_even_{w}", "n", "return n % 2 == 0", lambda r: r.randint(-30, 30)),
    ("Write a function {f} that returns the largest number in a list.", "max_of_{w}", "lst", "return max(lst)",
     lambda r: [r.randint(-50, 50) for _ in range(r.randint(1, 6))]),
    ("Write a function {f} that returns the smallest number in a list.", "min_of_{w}", "lst", "return min(lst)",
     lambda r: [r.randint(-50, 50) for _ in range(r.randint(1, 6))]),
    ("Write a function {f} that returns the sum of a list of numbers.", "sum_of_{w}", "lst", "return sum(lst)",
     lambda r: [r.randint(-20, 20) for _ in range(r.randint(0, 6))]),
    ("Write a function {f} that returns the length of a string.", "length_of_{w}", "s", "return len(s)",
     lambda r: "".join(r.choice("abcdxyz ") for _ in range(r.randint(0, 9)))),
    ("Write a function {f} that returns the string reversed.", "reverse_{w}", "s", "return s[::-1]",
     lambda r: "".join(r.choice("abcdxyz") for _ in range(r.randint(1, 8)))),
    ("Write a function {f} that returns the string in upper case.", "upper_{w}", "s", "return s.upper()",
     lambda r: "".join(r.choice("abcdxyz") for _ in range(r.randint(1, 8)))),
    ("Write a function {f} that returns the first element of a list.", "first_of_{w}", "lst", "return lst[0]",
     lambda r: [r.randint(-9, 99) for _ in range(r.randint(1, 5))]),
    ("Write a function {f} that returns the last element of a list.", "last_of_{w}", "lst", "return lst[-1]",
     lambda r: [r.randint(-9, 99) for _ in range(r.randint(1, 5))]),
    ("Write a function {f} that returns the absolute value of x.", "absolute_{w}", "x", "return abs(x)", lambda r: r.randint(-40, 40)),
    ("Write a function {f} that returns a list with every element doubled.", "double_all_{w}", "lst",
     "return [2 * v for v in lst]", lambda r: [r.randint(-9, 9) for _ in range(r.randint(0, 5))]),
    ("Write a function {f} that returns True if x is greater than {k}, else False.", "above_{w}", "x", "return x > {k}",
     lambda r: r.randint(-10, 60)),
]
WORDS = ["num", "val", "item", "data", "value", "thing", "input", "arg", "elem", "entry", "obj", "q", "z", "foo", "bar", "baz"]


def make_task(rng: random.Random, tid: str) -> dict:
    desc, fname, arg, body, gen = rng.choice(TEMPLATES)
    k = rng.randint(2, 19)
    f = fname.format(w=rng.choice(WORDS))
    ref = f"def {f}({arg}):\n    {body.format(k=k)}\n"
    ns: dict = {}
    exec(ref, ns)
    ins = [gen(rng) for _ in range(4)]
    asserts = [f"assert {f}({x!r}) == {ns[f](x)!r}" for x in ins]
    # Signature + docstring + one doctest: the shape of Stage 16's SFT prompts
    # (and HumanEval's). A prose MBPP-style prompt got 0/80 even on these
    # tasks: the SFT model had never seen a request phrased as prose.
    sentence = desc.format(f=f, k=k).split(" that ", 1)[1]
    sentence = "Return" + sentence[len("returns"):] if sentence.startswith("returns") else sentence
    prompt = (f"def {f}({arg}):\n    \"\"\"{sentence}\n    >>> {f}({ins[0]!r})\n"
              f"    {ns[f](ins[0])!r}\n    \"\"\"\n")
    return dict(task_id=tid, prompt=prompt, test="\n".join(asserts[1:]) + "\n", reference=ref)


def easy_tasks(n: int, seed: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    return [make_task(rng, f"{prefix}/{i}") for i in range(n)]


TRAIN = lambda: easy_tasks(400, 0, "easy_train")          # noqa: E731
HELDOUT = lambda: easy_tasks(150, 1, "easy_heldout")      # noqa: E731
