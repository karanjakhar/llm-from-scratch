# Choosing What the Model Does

A project-level decision, not a stage. It has to be made before Stage 02
(data pipeline) because it determines the corpus, and before Stage 01's final
run because it determines what the tokenizer is trained on.

tinyshakespeare is a **test fixture**, not a training target. It is 1.1 MB —
useful for asserting that code works, useless as a model.

---

## 1. The constraint, stated honestly

From the compute arithmetic (`C ≈ 6ND`, Chinchilla `D ≈ 20N`) on an RTX 4060
Laptop at a realistic 15-25% MFU:

| Params | tok/s | Chinchilla tokens | Wall-clock | Tokenized size |
|---|---|---|---|---|
| 10M | 145-240k | 0.2B | **~0.3 h** | 0.4 GB |
| 30M | 48-80k | 0.6B | **2-3.5 h** | 1.2 GB |
| 50M | 29-48k | 1.0B | **6-10 h** | 2.0 GB |
| 125M | 12-19k | 2.5B | **36-60 h** | 5.0 GB |
| 350M | 4-7k | 7.0B | 280-470 h | 14 GB | ← out of reach |

**The realistic ceiling for this project is ~125M parameters.** Stage 09
measures the actual throughput and replaces these estimates.

### What that buys, and what it does not

This is the part most "train your own LLM" tutorials lie about. At 125M
parameters:

- **No world knowledge.** It will confidently invent facts. Knowledge scales
  terribly with parameters; 125M holds almost none.
- **No multi-step reasoning.** Chain-of-thought is an emergent-at-scale
  behaviour. It will not appear here.
- **No general instruction following.** It can learn *one* narrow task format
  well.
- **Grammar, syntax, and local structure: genuinely good.** This is what small
  models do learn, and learn properly.

So the only domains where the output is *useful* rather than *cute* are
**narrow, highly structured, and ideally machine-verifiable.** That last
property is worth a lot here — see §3.

---

## 2. The options

Ranked by how well they fit both the hardware and the curriculum.

### Option A — Python code ★ recommended

Pretrain on Python source, SFT on docstring→function, then RL against unit
tests.

**Data.** `codeparrot/codeparrot-clean` — 477k Python files, 5.0 GB of actual
source, ungated, with a per-file `license` column to filter on. Verified
sizes and the alternatives are in `docs/datasets.md`.

Two traps that cost real time if you trust the obvious choices:
`HuggingFaceTB/python-edu` ships **metadata only** (blob ids, not code), and
every `bigcode/*` corpus is gated. Details in `docs/datasets.md`.

**Honest expected quality at 125M.** Syntactically valid code, plausible
single functions, correct short completions. HumanEval pass@1 in the **low
single digits** — CodeParrot-small (110M) landed around 3-4%. It is a usable
autocomplete for boilerplate and a bad programmer. That is a real, honest
number that then *improves measurably* across Stages 16-21, which is the point.

**Why it wins on curriculum fit** — this is the decisive argument:

| Stage | What code gives it |
|---|---|
| 01 Tokenizer | Indentation, identifiers, and digit grouping genuinely matter. The regex ablation becomes consequential rather than academic. |
| 16 SFT | Natural instruction data for free: docstring → body. No annotation needed. |
| 20 GRPO / RLVR | **A real verifier.** Run the unit tests, reward = pass/fail. This is precisely how R1-style code RL works. Every other domain needs a learned reward model or a judge. |
| 22 Evaluation | HumanEval / MBPP pass@k are objective. No LLM-as-judge hand-waving. |

Code is the only option that makes Stage 20 a real experiment instead of a toy.

**Risks.** Repetitive data means it can look better than it is. Need a held-out
set that is genuinely not in training — code is heavily duplicated across
GitHub, so Stage 02's dedup work actually matters.

---

### Option B — Text → SQL ★ strong, narrower, higher quality-per-param

Pretrain on a code/SQL mix, SFT on (schema + question → query).

**Data.** Spider, WikiSQL, BIRD, `b-mc2/sql-create-context` (~78k synthetic
pairs). Small datasets, so pretraining still needs a broader code corpus
underneath.

**Honest expected quality.** The best quality-per-parameter of any option
here. Fine-tuned encoder-decoder models in the **60-220M range historically
reached roughly 40-60% execution accuracy on Spider**. A 50M decoder-only
model, trained properly, lands in usable territory. You would have a thing you
could actually point at a database.

**Verifiable**: execute the query, compare result sets. So Stage 20 works here
too.

**Downside.** Very narrow. Less interesting as a *pretraining* exercise
because the task data is small and the domain is thin — you would spend
Stages 02-11 on general code anyway, then specialize. Effectively Option A
with a different stage 16+.

---

### Option C — Python code + a small English slice ★ what I would actually pick

Option A, plus ~15-20% high-quality English (`HuggingFaceFW/fineweb-edu`
sample) in the pretraining mix.

**Data.** `codeparrot/codeparrot-clean` for code + a streamed slice of
`HuggingFaceFW/fineweb-edu` `sample-10BT` for English. SFT pairs come free
from `code-search-net/code_search_net` (412k docstring/function pairs).
Eval is HumanEval + MBPP. All ungated, under 10 GB total — see
`docs/datasets.md`.

**Why the mix.** A pure-code model cannot read a docstring or an instruction —
it has never seen prose. Adding a slice of English costs little and makes
Stages 16-20 possible, because the model can condition on a natural-language
request. Every real code model is trained this way for exactly this reason.

**Cost.** Slightly worse pure-code perplexity in exchange for a model that can
be instructed. Worth it, and the ratio is itself a measurable Stage 10/11
ablation: train 100% code vs 85/15 and compare both code perplexity *and*
instruction-following after SFT.

---

### Option D — Narrow structured generation

Same shape as SQL — tiny, verifiable, genuinely works small:

- **NL → shell command** (tldr pages, nl2bash). Verifiable by parsing/dry-run.
- **Regex generation.** Verifiable by running against test strings.
- **Commit message from diff.** You have git history as data. Useful daily.
- **SMILES / molecule generation.** Small vocab, real scientific use, chemical
  validity is checkable with RDKit.

All are good, all are **too small to pretrain on.** They are Stage-16
fine-tuning targets sitting on top of Option A or C. Keep one as a side
deliverable; do not make it the pretraining domain.

---

### Option E — A low-resource language

Genuinely useful: there is no good 100M-parameter Hindi/Punjabi/Marathi model,
and tokenizer fairness (Stage 01 §2) stops being an abstract point and becomes
the main event.

**Why I am not recommending it as the primary.** Evaluation is the problem.
You would need to judge fluency by reading samples, which is slow and
subjective, and there is no verifier — so Stages 18-22 lose their teeth. Also
needs native fluency in the target language to assess at all.

Good as a **second capstone** after the pipeline works, if a language you read
fluently is of interest.

---

### Option F — TinyStories

Not useful, but worth knowing why it exists: it is the proven existence result
that **10-30M parameters can produce genuinely coherent text** when the data
distribution is narrow enough (toddler-level vocabulary, simple plots).

**Use it as the Stage 08 smoke test, not the target.** A 10M model trains in
20 minutes and produces readable stories — which means when your training loop
is broken, you find out in 20 minutes instead of 10 hours. Keep it for that.

---

### Option G — Domain prose: medical / legal / ArXiv

Avoid. At 125M you get fluent, domain-flavoured, confidently wrong text. The
worst failure mode: output that *reads* authoritative in a domain where being
wrong matters. No verifier, no honest eval, real potential to mislead.

---

## 3. The decisive criterion: is it verifiable?

Half this curriculum (Stages 18, 19, 20, 22) is about optimizing against a
reward and measuring whether it worked. With a verifiable domain:

- Stage 20 GRPO reward = **do the unit tests pass**. Objective, free, no
  annotation, no reward model to train, no judge to debias.
- Stage 22 eval = **pass@k**. A number, reproducible, comparable to published
  results.

With an unverifiable domain (prose, stories, a language), both stages become
"ask an LLM whether this is good", which at 125M is measuring noise.

**This single property is why code wins**, more than any argument about
usefulness.

---

## 4. Decision

**Option C: Python code + ~15% high-quality English.** Chosen 2026-10-14.

Run as two models:

| | Workhorse | Capstone |
|---|---|---|
| Params | ~30-50M | ~125M |
| Tokens | 0.6-1.0B | 2.5B |
| Runtime | 2-10 h | 36-60 h |
| Purpose | learn every stage on it; iterate in an evening | one long run once the pipeline is proven |

Do not start with the 125M run. Every stage from 02 to 11 gets debugged on the
workhorse, where a mistake costs an afternoon rather than two days. Stage 11
is where the capstone run happens, with a known-good pipeline.

Then the post-training stages have real jobs:
- **16 SFT** — docstring → function
- **17 LoRA** — same task, 1% of the trainable parameters, compare
- **18 Reward model** — preference pairs over code completions
- **19 DPO** — align toward the style you prefer
- **20 GRPO** — reward = unit tests pass. The stage this whole choice is for.
- **22 Eval** — HumanEval / MBPP pass@k, tracked across every checkpoint

End state: a ~125M Python code model, trained from a tokenizer you wrote,
measurably improved by each post-training technique, with honest numbers at
every step. Not useful as a product. A completely understood pipeline, which
is the actual goal.

## 5. Consequences for earlier stages

Decided here, consumed later:

- **Stage 01** — retrain the tokenizer on the code+English mix, not Shakespeare.
  Vocab 8192 may be too small for code (identifiers and indentation eat slots);
  measure 8192 vs 16384 on bytes/token. Consider splitting digits individually
  (Llama 1/2 and Qwen2 style), since generated code contains numeric literals.
- **Stage 02** — dedup is now load-bearing (GitHub duplication is severe), and
  the train/val split must be by *repository*, not by file, or held-out
  perplexity is meaningless.
- **Stage 10** — the code/English mixture ratio becomes an extra ablation axis.
