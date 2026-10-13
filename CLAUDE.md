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
- Python 3.12, PyTorch 2.10 + CUDA 12.8, `uv` for environment management.
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
- **Tokenizer vocab 8192** for own-trained models — small enough that the
  embedding table does not dominate a 10-50M parameter model.
- **Complete working code, not exercises.** The owner reads, runs, and breaks
  working implementations rather than filling in TODOs. Each stage ships
  `<topic>.py` (commented implementation), `test_<topic>.py` (acceptance
  criteria from the doc, as tests), and `run.py` (the experiment that produces
  the stage's numbers).
- `data/` and `checkpoints/` are gitignored. Only code, docs, configs, and
  small result artifacts (plots, JSON metrics) are committed.

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
