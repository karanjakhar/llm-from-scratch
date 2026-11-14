"""Stage 24: collect the capstone's numbers from every stage into one place.

    uv run python stages/24_capstone/assemble.py

Reads results/*/ JSON written by the stage scripts (run with --tag capstone)
and writes results/24_capstone/pipeline.json. Nothing is recomputed here:
every number traces back to the file it came from (listed in "sources").
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
R = REPO / "results"


def load(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out, sources = {}, {}

    pre = load("11_pretrain/capstone.json")
    if pre:
        v = [c for c in pre["curve"] if c[3]][-1][3]
        out["pretrain"] = dict(params=pre["params"], tokens=pre["tokens"], val_code=v["code"], val_en=v["en"],
                               config=pre["config"]["model"])
        sources["pretrain"] = "results/11_pretrain/capstone.json"

    sft = load("16_sft/capstone.json")
    if sft:
        out["sft"] = dict(train_pairs=sft["train_pairs"],
                          before=dict(lm_val=sft["before"]["lm_val"],
                                      humaneval_base=sft["before"]["base_mode"]["humaneval_pass1"],
                                      mbpp_base=sft["before"]["base_mode"]["mbpp_pass1"]),
                          after=dict(lm_val=sft["after"]["lm_val"], sft_val=sft["after"]["sft_val"],
                                     humaneval=sft["after"]["chat_mode"]["humaneval_pass1"],
                                     mbpp=sft["after"]["chat_mode"]["mbpp_pass1"]))
        sources["sft"] = "results/16_sft/capstone.json"

    dpo = sorted((R / "19_dpo").glob("capstone_beta*.json")) if (R / "19_dpo").exists() else []
    if dpo:
        best = max((json.loads(p.read_text()) for p in dpo), key=lambda r: r["humaneval_pass1"] + r["mbpp_pass1"])
        out["dpo"] = {k: best.get(k) for k in ("beta", "humaneval_pass1", "mbpp_pass1", "final_mbpp_pref_acc",
                                                 "final_kl", "final_rm")}
        sources["dpo"] = "results/19_dpo/capstone_beta*.json"

    rm = load("18_reward/capstone.json")
    if rm:
        out["reward_model"] = dict(csn_val_acc=rm["csn_val_acc"], mbpp_acc=rm["mbpp_acc"])
        sources["reward_model"] = "results/18_reward/capstone.json"

    # the run script tags the easy-curriculum run "<tag>_easy" (stages/20_grpo/run.py)
    grpo = load("20_grpo/capstone_easy.json")
    if grpo:
        ev = [h for h in grpo["hist"] if "easy_heldout" in h]
        keep = ("step", "easy_heldout", "mbpp_test", "humaneval", "kl")
        out["grpo"] = dict(tasks="easy", steps=ev[-1]["step"],
                           start={k: ev[0].get(k) for k in keep}, end={k: ev[-1].get(k) for k in keep})
        sources["grpo"] = "results/20_grpo/capstone_easy.json"

    q = load("21_distill_quant/quant_capstone.json")
    if q:
        out["quant"] = [{k: r[k] for k in ("name", "mb", "val", "tok_s")} for r in q]
        sources["quant"] = "results/21_distill_quant/quant_capstone.json"

    tab = load("22_evaluation/table.json")
    if tab:
        out["eval"] = [{k: v for k, v in r.items() if "solved" not in k} for r in tab if "capstone" in r["model"]]
        sources["eval"] = "results/22_evaluation/table.json"

    mc = load("22_evaluation/mc.json")
    if mc:
        out["mc"] = {k: v for k, v in mc["scores"].items() if "capstone" in k}
        sources["mc"] = "results/22_evaluation/mc.json"

    srv = load("23_serving/bursty_capstone.json")
    if srv:
        out["serve"] = {p: {k: srv[p][k] for k in ("tokens_per_s", "latency_p50", "latency_p99", "ttft_p50")}
                        for p in ("static", "continuous")}
        sources["serve"] = "results/23_serving/bursty_capstone.json"

    out["sources"] = sources
    (R / "24_capstone").mkdir(parents=True, exist_ok=True)
    (R / "24_capstone" / "pipeline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
