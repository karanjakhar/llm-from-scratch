---
license: mit
tags:
- from-scratch
- education
- code
language:
- en
---

# llm-from-scratch checkpoints

Trained models from the book and course **Building an LLM from scratch**
(code: https://github.com/karanjakhar/llm-from-scratch, course:
https://karanjakhar.dev/course/). They exist so learners can skip hours of
pretraining; they are teaching artifacts, not coding assistants.

| file | what | params | size |
|---|---|---|---|
| `workhorse/final.pt` | 30M base, pretrained 590M tokens (85% Python / 15% English) | 30M | 120 MB |
| `workhorse_sft/final.pt` | the workhorse after supervised fine-tuning (docstring → function) | 30M | 120 MB |
| `capstone/final.pt` | 92M base, pretrained 1.8B tokens | 92M | 369 MB |

Each file is a plain PyTorch dict: `config` (the model's hyperparameters) and
`model` (the state dict), loaded by the repository's own GPT implementation
with its own 8,192-token tokenizer. Not compatible with `transformers`.

Get them with the course's script, which pins a revision and puts each file
where the stage scripts expect it:

```bash
python course/get_checkpoints.py              # workhorse base + SFT
python course/get_checkpoints.py --capstone   # also the 92M base
```

## Honest numbers

HumanEval pass@1 (greedy): workhorse 0, capstone 3.05% (5 of 164). At this
size there is no world knowledge, no multi-step reasoning and no safety
training. Full results and limitations:
[MODEL_CARD.md](https://github.com/karanjakhar/llm-from-scratch/blob/main/MODEL_CARD.md).
