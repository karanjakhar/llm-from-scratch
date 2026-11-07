"""Acceptance criteria for Stage 23 (docs/23_serving.md §6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "07_generation"))
from sample import generate_cached  # noqa: E402

from llmfs.model.gpt import GPT, GPTConfig  # noqa: E402
from llmfs.serve.engine import Engine, Request, serve  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="engine is CUDA-only")


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=97, block_size=128, n_layer=2, n_head=2, n_embd=32)).cuda().eval()


@pytest.mark.parametrize("policy", ["continuous", "static"])
def test_engine_output_equals_single_request_greedy(policy):
    m = _model()
    torch.manual_seed(1)
    prompts = [torch.randint(0, 97, (int(n),)).tolist() for n in torch.randint(3, 20, (9,))]
    lens = [5, 30, 12, 8, 25, 3, 17, 9, 20]
    reqs = [Request(i, p, n, arrival=0.001 * i) for i, (p, n) in enumerate(zip(prompts, lens))]
    eng = Engine(m, slots=4, max_len=128, dtype=torch.float32)
    serve(eng, reqs, policy)
    for r, p, n in zip(reqs, prompts, lens):
        want = generate_cached(m, torch.tensor([p], device="cuda"), n, temperature=0.0)[0, len(p):]
        assert r.out == want.tolist(), r.rid
