"""Stage 09 §9d: throughput of the two Stage 11 shapes at T=1024 with the full stack.

    uv run python stages/09_efficiency/shapes.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

rows = []
FULL = dict(tf32=True, bf16=True, flash=True, fused_opt=True, compile=True)
for name, model, B, extra in [
    ("workhorse 8L/512", dict(vocab_size=8196, n_layer=8, n_head=8, n_embd=512), 16, {}),
    ("capstone 12L/768", dict(vocab_size=8196, n_layer=12, n_head=12, n_embd=768), 8, {}),
    ("capstone 12L/768", dict(vocab_size=8196, n_layer=12, n_head=12, n_embd=768), 16, {}),
    ("capstone 12L/768 + ckpt", dict(vocab_size=8196, n_layer=12, n_head=12, n_embd=768), 16, dict(ckpt=True)),
]:
    bench.MODEL, bench.T = model, 1024
    try:
        r = bench.run_config(dict(FULL, B=B, **extra), steps=15, warmup=8)
    except Exception as e:                                  # OOM is a result
        r = dict(error=type(e).__name__)
    bench.torch._dynamo.reset()
    bench.torch.cuda.empty_cache()
    peak = json.loads((bench.RESULTS / "peak.json").read_text())["bf16"]
    if "tflops" in r:
        r["mfu_vs_measured_bf16"] = round(r["tflops"] / peak, 3)
    rows.append(dict(name=name, T=1024, **r))
    print(rows[-1], flush=True)
bench.write_json("shapes.json", rows)
