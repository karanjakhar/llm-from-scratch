# Context for AI sessions

Read this file, then `PROGRESS.md`. That is enough to resume with zero prior
conversation history.

## What this project is

A from-scratch build of a modern LLM, structured as a 25-stage curriculum in
`CURRICULUM.md`. The owner's goal is to **understand every part of the
process**, not to ship a product. Teaching quality beats code elegance beats
speed.

## How to work here

1. **Docs before code.** For stage NN, `docs/NN_*.md` is written and reviewed
   first. The doc explains theory, intuition, the math, and the failure modes.
   Then `stages/NN_*/` implements it.
2. **Hand-roll, then verify against a library.** Never import the thing being
   taught. Write it, then assert it matches the reference implementation
   (`tiktoken`, `F.scaled_dot_product_attention`, `peft`, HF GPT-2 weights)
   numerically. The comparison *is* the lesson.
3. **No skipping ahead.** If the owner asks about stage 13, answer the question
   but do not implement stage 13 before the stages it depends on.
4. **Every stage produces a number**, logged in `PROGRESS.md` → Results log.
5. **Every stage has an ablation**: break the component, record the damage.
6. **Update `PROGRESS.md` at the end of every session** — stage status, results,
   open questions, and a session log entry. This is the handoff mechanism.
7. **Explain in comments.** Code here is read by a learner. Comment the *why*,
   not the *what*. Shapes on every tensor line: `# (B, T, C)`.
8. **New terms go in `GLOSSARY.md`** the first time they appear.

## Writing style for docs

- Plain language first, then the formal version. Define a term before using it.
- Always answer "what breaks without this?" — that is where understanding lands.
- Show tensor shapes. Show worked numeric examples with small numbers.
- Be honest about what is simplified and what real systems do differently.
- No hype, no "simply", no "just". If something is genuinely hard, say so.

## Hardware constraints — plan around these

- **RTX 4060 Laptop, 8 GB VRAM.** This is the binding constraint on everything.
- ~98 GB free disk on `/mnt/extra_ssd`.
- Python 3.12, PyTorch 2.11 + CUDA 12.8, managed by `uv`. **Always run code
  with `uv run ...`** — never bare `python3`. The system `python3` is 3.10 with
  no torch. A conda env that sessions 1-2 relied on disappeared between
  sessions, and the project only kept working because `pyproject.toml` +
  `uv.lock` define the environment. Any new dependency goes into
  `pyproject.toml` (`uv add <pkg>`), never `pip install`.
- Practical ceiling: ~150M params trainable locally with bf16 + grad
  accumulation + activation checkpointing. Inference up to ~1-3B quantized.
- Prefer datasets under 10 GB. TinyStories for anything that must train in
  under an hour; FineWeb-Edu sample for the real run.
- A RunPod MCP plugin is available in this environment if a stage genuinely
  needs a bigger GPU. Local-first; rent only when the local path is exhausted,
  and always say what it will cost per hour before creating anything.

## Decisions already made

- **Pure PyTorch** for every core component: tokenizer, attention, transformer,
  MoE, training loop, LoRA, DPO, GRPO, quantizer. HuggingFace libraries are
  used only as (a) a numerical oracle to check against, (b) dataset/weight
  download, (c) comparison baselines in a stage's results table.
- **Two-layer code structure.** `stages/NN_*/` holds the slow, explicit,
  heavily-commented educational version. Code that later stages need is
  promoted into `llmfs/` in cleaner form. Writing it twice is intentional.
- **GPT-2 architecture first** (learned positions, LayerNorm, GELU, MHA) because
  real GPT-2 weights act as a correctness oracle. Stage 12 then converts it to
  Llama-class (RoPE, RMSNorm, SwiGLU, GQA) so each change is measured
  individually instead of arriving as a bundle.
- **Dense before MoE.** MoE (stage 13) is a modification of a working dense
  model, with an active-parameter-matched dense baseline for comparison.
- **Target domain: Python code + ~15% high-quality English** (decided
  2026-10-14). Chosen because it is the only candidate with a *machine
  verifier*: Stage 20's RL reward is "do the unit tests pass" and Stage 22's
  eval is HumanEval/MBPP pass@k. Unverifiable domains (stories, prose, a
  low-resource language) turn half the curriculum into LLM-as-judge
  guesswork, which at 125M params measures noise. Reasoning in
  `docs/domain_choice.md`; verified corpora in `docs/datasets.md`.
- **Two models.** A ~30-50M workhorse (2-10 h/run) to debug every stage on,
  and one ~125M capstone run (36-60 h) at Stage 11 once the pipeline is proven.
  Never debug on the capstone.
- **Honest expectations.** 125M params = no world knowledge, no multi-step
  reasoning, no general instruction following. HumanEval pass@1 will be low
  single digits. Do not let the docs or the commit messages imply otherwise;
  the value is that the number is objective and improves measurably per stage.
- **TinyStories and tinyshakespeare are test fixtures, never targets.**
  tinyshakespeare for unit tests, TinyStories for the Stage 08 training-loop
  smoke test (10M params, ~20 min, obviously-broken output when the loop is
  broken).
- **Production tokenizer: `data/tokenizers/split_8192.json`** (decided
  2026-10-25 from measurement, `docs/01_tokenizer.md` §9). GPT-2 pattern with
  every digit split individually, 8,192 learned tokens + 4 specials = **8,196
  embedding rows** (eot = 8192). 16,384 bought only +6.1% code compression for
  +4.2M params. Both models use this one tokenizer. Regenerate with
  `stages/01_tokenizer/train_production.py` (~1 h). Do not switch without new
  evidence; switching invalidates every trained checkpoint.
- **Complete working code, not exercises.** The owner reads, runs, and breaks
  working implementations rather than filling in TODOs. Each stage ships
  `<topic>.py` (commented implementation), `test_<topic>.py` (acceptance
  criteria from the doc, as tests), and `run.py` (the experiment that produces
  the stage's numbers).
- `data/` and `checkpoints/` are gitignored. Only code, docs, configs, and
  small result artifacts (plots, JSON metrics) are committed.

## Git / publishing state

Public on GitHub: **https://github.com/karanjakhar/llm-from-scratch** (remote
`origin`, branch `main`), published 2026-10-25 at the owner's request.

- **Commit messages must never contain a `Co-Authored-By` trailer or any
  mention of Claude / AI authorship.** Owner's explicit instruction. (Mentioning
  the `CLAUDE.md` *file* by name in a message is fine — it is a filename.)
- Commit every session. Push when the owner asks; do not push on your own
  initiative.
- Never force-push `main` now that it is public.
- Before any push, check that nothing under `data/` or `checkpoints/` is
  staged and that no credentials or account names have crept into files.

## Commit convention

Conventional Commits, one stage concern per commit.
`feat(stage-01): BPE trainer with regex pre-tokenization`
`docs(stage-04): attention notes incl. sqrt(d) derivation`
`exp(stage-10): isoflop sweep results`

## Things to avoid

- Do not install heavy frameworks (transformers/trl/peft) as the *solution* to
  a stage. They are reference points only.
- Do not write placeholder files that claim a stage is done. `PROGRESS.md` must
  reflect reality.
- Do not refactor earlier stages to be "better". They are a historical record
  of the learning. Fix bugs, do not modernize.
