"""Acceptance criteria for Stage 22 (docs/22_evaluation.md §7): the harness itself.

A harness that scores the reference solutions below 100% is measuring itself,
not the model. These run the canonical answers through the same execution path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmfs.eval.codeeval import RAW, load_humaneval, load_mbpp, pass_at_k, truncate  # noqa: E402
from llmfs.eval.execution import run_many, run_program  # noqa: E402

HAVE = (RAW / "humaneval").exists() and (RAW / "mbpp").exists()


def test_pass_at_k_estimator():
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    # 1 - C(7,2)/C(10,2) = 1 - 21/45
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - 21 / 45)


def test_sandbox_timeout_and_errors():
    assert run_program("print(1)") == (True, "passed")
    ok, why = run_program("while True: pass", timeout=1.0)
    assert not ok and why == "timeout"
    ok, why = run_program("raise ValueError('x')")
    assert not ok and "ValueError" in why
    ok, why = run_program("x = bytearray(2 * 1024**3)")       # 2 GB > 512 MB limit
    assert not ok and "MemoryError" in why


def test_sandbox_rejects_early_exit_hack():
    for hack in ("import os; os._exit(0)", "import sys; sys.exit(0)", "raise SystemExit"):
        ok, why = run_program(hack + "\nassert 1 == 2\n")
        assert not ok, hack


def test_truncate_humaneval_stops_at_top_level():
    comp = "    return a + b\n\n\ndef other():\n    pass\n"
    assert truncate(comp) == "    return a + b\n\n"


def test_truncate_mbpp_keeps_one_function():
    comp = "def f(x):\n    return x\n\nprint(f(2))\ndef g(): pass\n"
    assert truncate(comp, is_mbpp=True) == "def f(x):\n    return x\n"


@pytest.mark.skipif(not HAVE, reason="datasets not downloaded")
def test_humaneval_reference_solutions_all_pass():
    rows = pq.read_table(RAW / "humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet").to_pylist()
    probs = load_humaneval()
    progs = [p["prompt"] + r["canonical_solution"] + "\n\n" + p["test"] for p, r in zip(probs, rows)]
    res = run_many(progs, workers=16)
    assert sum(ok for ok, _ in res) == 164


@pytest.mark.skipif(not HAVE, reason="datasets not downloaded")
def test_mbpp_reference_solutions_pass():
    probs = load_mbpp("test")
    res = run_many([p["reference"] + "\n\n" + p["test"] for p in probs], workers=16)
    # a few MBPP references are known-broken or slow; require 99%
    assert sum(ok for ok, _ in res) >= 0.99 * len(probs)


@pytest.mark.skipif(not HAVE, reason="datasets not downloaded")
def test_wrong_answers_fail():
    probs = load_humaneval()[:20]
    res = run_many([p["prompt"] + "    return None\n\n" + p["test"] for p in probs], workers=16)
    assert sum(ok for ok, _ in res) <= 1


def test_early_stop_never_changes_the_truncated_program():
    """ended(t) => truncate(t + anything) == truncate(t). Generation may stop
    as soon as ended() is true without changing any score."""
    import random
    from llmfs.eval.codeeval import ended
    rng = random.Random(0)
    pieces = ["    x = 1\n", "    return x\n", "\n", "def g():\n", "print(1)\n", "  # c\n",
              "import os\n", "class A:\n", "    pass\n", "if x:\n", "\tq\n", "d", "ef h", "():\n"]
    checked = 0
    for _ in range(3000):
        is_mbpp = rng.random() < 0.5
        t = ("def f(x):\n" if is_mbpp else "") + "".join(rng.choice(pieces) for _ in range(rng.randint(1, 8)))
        if ended(t, is_mbpp):
            more = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 6)))
            assert truncate(t + more, is_mbpp) == truncate(t, is_mbpp), (t, more)
            checked += 1
    assert checked > 300


def test_generation_runs_without_autograd():
    """A lost @torch.no_grad() made every decode step keep its graph: ~20 MB per
    token, OOM after a few hundred tokens (journal, Stage 16-19)."""
    from llmfs.eval import codeeval
    assert getattr(codeeval.generate_batch, "__wrapped__", None) is not None
