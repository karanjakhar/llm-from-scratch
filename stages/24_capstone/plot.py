"""Stage 24 figure: the capstone's checkpoints along the pipeline.

    uv run python stages/24_capstone/plot.py

Reads results/24_capstone/pipeline.json (run assemble.py first) and writes
docs/visuals/24_capstone.png. Two panels, two measures, never one dual axis:
what each step did to the pretraining loss, and to HumanEval.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
d = json.loads((REPO / "results" / "24_capstone" / "pipeline.json").read_text())
rows = {r["model"].replace("capstone ", ""): r for r in d["eval"]}
order = ["base", "SFT", "DPO", "GRPO"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# the rise over the base model, so the bars start at zero and their lengths are honest
base = rows["base"]["val_code"]
rise = [rows[k]["val_code"] - base for k in order]
ax[0].bar(order, rise, color="C0", width=0.6)
for i, v in enumerate(rise):
    ax[0].text(i, v + 0.004, f"+{v:.3f}", ha="center", fontsize=9)
ax[0].set(ylim=(0, 0.25), ylabel=f"val code loss above base ({base:.3f})",
          title="Every post-training step raised the pretraining loss")
he = [round(rows[k]["humaneval_pass1"] * 164) for k in order]
p10 = [rows[k]["humaneval_pass10"] * 164 for k in order]
x = range(len(order))
ax[1].bar([i - 0.2 for i in x], he, 0.38, color="C0", label="greedy pass@1")
ax[1].bar([i + 0.2 for i in x], p10, 0.38, color="C1", label="pass@10 (× 164)")
for i, v in enumerate(he):
    ax[1].text(i - 0.2, v + 0.15, str(v), ha="center", fontsize=9)
ax[1].set(xticks=list(x), xticklabels=order, ylabel="HumanEval problems solved (of 164)",
          title="HumanEval: SFT lost 3 problems, GRPO won them back")
ax[1].legend(fontsize=8)
fig.suptitle("The 92M capstone, base → SFT → DPO or GRPO (DPO and GRPO both start from SFT)", fontsize=10)
fig.tight_layout()
fig.savefig(REPO / "docs" / "visuals" / "24_capstone.png", dpi=130)
print("saved")
