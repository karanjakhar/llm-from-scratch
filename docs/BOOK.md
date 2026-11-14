# The Book

This project ends in a book. Everything in `docs/` is written as a draft of it,
and `docs/JOURNAL.md` holds the raw story. This file is the plan: what the book
is, how it is written, which chapter comes from which source, and how far each
one has got.

**Working title:** *From Bytes to Code: Building a Language Model From Scratch
on One Laptop GPU* (placeholder; the owner decides the final title)

---

## 1. What the book is

**One-sentence pitch.** A reader who knows Python but not machine learning
follows one real project, from raw GitHub files to a small working code model,
and understands every part because every part is built, measured and broken
on purpose.

**What makes it different from other "build an LLM" books:**

1. **Everything is measured.** Every claim comes with a number from a real run
   on modest hardware (RTX 4060 Laptop, 8 GB). No "this should help". It
   either helped by a measured amount or it did not.
2. **The mistakes stay in.** The wrong tests, the datasets that turned out to
   contain no data, the leak that put 40% of the exam into the textbook. Each
   chapter shows what went wrong and how it was caught, because that is what
   readers will meet in their own work.
3. **Every component gets broken on purpose.** Each chapter removes the thing
   it just built and shows the damage. That ablation is where the
   understanding lands.
4. **Honest about scale.** A 125M-parameter model will not reason or know
   facts. The book says so up front and is about understanding the pipeline,
   not beating benchmarks.

**Reader.** Comfortable with Python and basic high-school math. Has heard of
ChatGPT, has never trained a model. Often a visual learner, so every mechanism
gets a picture.

---

## 2. How every chapter is written

Each chapter follows the same arc, so readers always know where they are:

1. **The question.** What problem does this component solve? What breaks
   without it? One paragraph, plain words.
2. **The picture.** A diagram of the mechanism before any formula.
3. **The idea, plainly.** Intuition first, with small worked numbers.
4. **The idea, formally.** The math, with tensor shapes.
5. **The build.** The code, walked through. Shapes on every tensor line.
6. **The measurement.** The numbers from our run, in a table or chart.
7. **What surprised us.** From `JOURNAL.md`: what we expected, what happened.
8. **What we got wrong.** Mistakes and how they were caught.
9. **Breaking it on purpose.** The ablation and the damage it caused.
10. **Takeaways.** Three to five rules the reader keeps.

### Voice and style

- **Plain language first.** Explain as you would to a smart friend across a
  table. Define every term before using it, then add it to `GLOSSARY.md`.
- Short sentences. Active voice. No hype. No "simply", "just", "obviously".
- **Real numbers only.** Every number in the book traces to a results file, a
  journal entry, or a command the reader can run. No invented examples where a
  real one exists. When a small made-up example is clearer (like the
  "low lower lowest" BPE walkthrough), label it as an illustration.
- Analogies are welcome when they are accurate (the exam and the textbook for
  train/val). Drop any analogy that breaks under a second look.
- Say what is simplified, and what real systems do differently.
- When something is genuinely hard, say so.

---

## 3. Chapter map

Status: `outline` · `draft` (stage doc written, findings in) · `revised` (rewritten
in book voice) · `final`

| # | Chapter (working) | Stages | Sources | Status |
|---|---|---|---|---|
| — | Preface: why one laptop, why code | — | `docs/preface.md` | draft |
| 1 | What a language model actually computes | 00 | `docs/00_setup.md` | draft |
| 2 | Choosing what to teach it | — | `docs/domain_choice.md`, `docs/datasets.md` | draft |
| 3 | Cutting text into tokens | 01 | `docs/01_tokenizer.md` | draft |
| 4 | The data is most of the model | 02 | `docs/02_data.md` | draft |
| 5 | Turning ids into vectors and position | 03 | `docs/03_embeddings_position.md` | draft |
| 6 | Attention, from one head up | 04 | `docs/04_attention.md` | draft |
| 7 | The transformer block | 05 | `docs/05_transformer_block.md` | draft |
| 8 | Assembling GPT | 06 | `docs/06_gpt_model.md` | draft |
| 9 | Making it talk: generation and decoding | 07 | `docs/07_generation.md` | draft |
| 10 | The training loop | 08 | `docs/08_training_loop.md` | draft |
| 11 | Making it fast on 8 GB | 09 | `docs/09_efficiency.md` | draft |
| 12 | How big, how long: scaling laws | 10 | `docs/10_scaling_laws.md` | draft |
| 13 | The real pretraining run | 11 | `docs/11_pretraining.md` | draft |
| 14 | Modern architecture: from GPT-2 to Llama | 12 | `docs/12_modern_arch.md` | draft |
| 15 | Mixture of Experts | 13 | `docs/13_moe.md` | draft |
| 16 | Other kinds of attention, longer context | 14 | `docs/14_attention_variants.md` | draft |
| 17 | Faster decoding | 15 | `docs/15_speculative.md` | draft |
| 18 | Teaching it to follow instructions | 16 | `docs/16_sft.md` | draft |
| 19 | Fine-tuning with 1% of the parameters: LoRA | 17 | `docs/17_peft_lora.md` | draft |
| 20 | Learning what people prefer: reward models | 18 | `docs/18_reward_model.md` | draft |
| 21 | RLHF, DPO and GRPO | 19 | `docs/19_rlhf_dpo.md` | draft |
| 22 | Rewards you can verify: unit tests | 20 | `docs/20_reasoning_rlvr.md` | draft |
| 23 | Smaller and faster: distillation and quantization | 21 | `docs/21_distill_quantize.md` | draft |
| 24 | Measuring what matters | 22 | `docs/22_evaluation.md` | draft |
| 25 | Serving it | 23 | `docs/23_serving.md` | draft |
| 26 | The whole pipeline on one page | 24 | `docs/24_capstone.md`, `MODEL_CARD.md` | draft |
| A | Appendix: glossary | — | `GLOSSARY.md` | draft |
| B | Appendix: every result in one table | — | `PROGRESS.md` results log | draft |
| C | Appendix: reproducing everything | — | `docs/appendix_reproduce.md` | draft |

The chapter order may change; the stage numbers will not.

---

## 4. Figures

Every mechanism gets a picture. Figures live in `docs/visuals/` (interactive
HTML) and, for print, will need static versions.

| Figure | Chapter | Shows | Source | Print-ready? |
|---|---|---|---|---|
| Training memory bar | 1 | weights 48 / grads+acts 64 / Adam 96 MB vs 192 predicted | `visuals/stages_00-02_build_map.html` | no |
| 25-stage track | Preface | where the project is | same | no |
| Data pipeline | 4 | GitHub → license → split → dedup → tokenizer → bins → loader | same | no |
| Leakage bars | 4 | 43.9 / 34.4 / 21.7 / 8.1% rare-line leakage | same | no |
| BPE training loop | 3 | bytes → cut → count → merge → record → repeat | same | no |
| BPE lab (interactive) | 3 | merges happening step by step | same | needs static frames |
| Encoding replays merges | 3 | a new word encoded, merge by merge | same | needs static frames |
| Three digit rules | 3 | grouped / cap3 / split on real input | same | no |
| Compression vs GPT-2/GPT-4 | 3 | 2.22 / 3.34 / 3.38 / 3.59 / 4.24 bytes per token | same | no |
| Sinusoidal table + similarity | 5 | the clocks; similarity depends only on p - q | `visuals/03_sinusoidal.png` | yes (PNG) |
| RoPE score vs offset | 5 | stripes along diagonals; three curves on one line | `visuals/03_rope_offset.png` | yes (PNG) |
| Position ablation | 5 | training curves; loss by position past the training length | `visuals/03_ablation.png` | yes (PNG) |
| √d saturation | 6 | score size, top weight, score gradient vs head size | `visuals/04_sqrt_d.png` | yes (PNG) |
| Cost of T × T | 6 | memory and time, naive vs fused, up to OOM | `visuals/04_cost.png` | yes (PNG) |
| All 16 heads | 6 | attention maps of every head on one snippet | `visuals/04_attention_maps.png` | yes (PNG) |
| Two heads, labelled | 6 | previous-token head vs a diffuse head, with tokens | `visuals/04_attention_heads.png` | yes (PNG) |
| Untrained 24-layer models | 7 | gradient per block, stream size, position similarity: pre / post / no residual | `visuals/05_init_grads.png` | yes (PNG) |
| Depth training curves | 7 | pre / post / no residual at 4 and 24 layers | `visuals/05_depth.png` | yes (PNG) |
| One change at a time | 7 | loss for RMSNorm, ReLU, SwiGLU, no MLP, dropout vs baseline | `visuals/05_sweep.png` | yes (PNG) |
| Post-norm diagnosis | 7 | edit size vs highway, block-1 gradient: pre / post / post + √C | `visuals/05_postnorm_diagnosis.png` | yes (PNG) |
| Residual-scaled init | 8 | stream size per block at init, 12/48 layers; 12-layer training | `visuals/06_init.png` | yes (PNG) |
| Shift and tying | 8 | no-shift loss → 0; tied vs untied at 4k / 12k steps | `visuals/06_shift_tie.png` | yes (PNG) |
| KV cache and temperature | 9 | speedup vs length for 3 sizes; distinct-2 and log-prob vs τ | `visuals/07_generation.png` | yes (PNG) |
| The training loop | 10 | optimizers; overfit spike; warmup/clip at lr 3e-3; sampler order | `visuals/08_training.png` | yes (PNG) |
| The speed stack | 11 | tokens/s, MFU and memory for each optimisation in turn | `visuals/09_stack.png` | yes (PNG) |
| IsoFLOP, overtraining, mix and repeats | 12 | two U-curves with fitted optima; diminishing returns; mix/epoch bars | `visuals/10_scaling.png` | yes (PNG) |
| GPT-2 → Llama | 14 | ladder; KV bytes vs loss; window loss by position; NTK past training length | `visuals/12_modern_arch.png` | yes (PNG) |
| MoE | 15 | dense vs MoE curves; balance over training; utilisation heatmaps with/without aux loss | `visuals/13_moe.png` | yes (PNG) |
| MLA and needles | 16 | cache bytes vs loss for GQA and MLA; needle retrieval grid | `visuals/14_attention_variants.png` | yes (PNG) |
| Speculative decoding | 17 | speedup and acceptance vs γ at three temperatures | `visuals/15_speculative_workhorse.png` | yes (PNG) |
| Speculative decoding, 92M target | 17 | the same sweep with the capstone as target: high acceptance, no speedup | `visuals/15_speculative_capstone.png` | yes (PNG) |
| Reward model | 20 | reward histograms; accuracy under label noise; calibration | `visuals/18_reward.png` | yes (PNG) |
| DPO | 21 | implicit reward margins; RM score vs KL by β; pair accuracy | `visuals/19_dpo.png` | yes (PNG) |
| GRPO | 22 | held-out pass@1, training pass rate, response length: outcome vs format | `visuals/20_grpo.png` | yes (PNG) |
| Distillation and quantization | 23 | scratch vs distilled student; loss increase per quantization scheme | `visuals/21_distill_quant.png` | yes (PNG) |
| Pretraining | 13 | workhorse and capstone loss vs tokens; resume continuity and the lost-Adam control | `visuals/11_pretrain.png` | yes (PNG) |
| Evaluation | 24 | HumanEval problems solved per checkpoint; multiple-choice accuracy summed vs per-token | `visuals/22_evaluation.png` | yes (PNG) |
| Serving | 25 | throughput vs latency by slots; latency CDF static vs continuous | `visuals/23_serving.png` | yes (PNG) |
| The capstone pipeline | 26 | loss rise and HumanEval problems for base, SFT, DPO, GRPO | `visuals/24_capstone.png` | yes (PNG) |
| One-hot × table = row | 5 | a lookup as a matrix product with a one-hot vector | `visuals/onehot_lookup.png` | yes (PNG) |
| RoPE rotation | 5 | a fast and a slow pair rotating with position | `visuals/rope_rotation.png` | yes (PNG) |
| Packing with eot | 4 | documents end to end, separated by <\|endoftext\|>, cut into windows | `visuals/packing_eot.png` | yes (PNG) |
| Sampler coverage | 4 | random offsets leave 37% unseen after one epoch; shuffled windows see all | `visuals/sampler_coverage.png` | yes (PNG) |

**Wanted, not yet made:** a MinHash/LSH diagram (chapter 4). The one-hot,
RoPE-rotation, packing and sampler-coverage figures are made
(`visuals/make_diagrams.py`) but not yet embedded in their chapters.

Every figure is embedded in its chapter with `![caption](visuals/name.png)`,
not only named by path, so it shows on GitHub and in the web edition.

**For print:** the interactive pages will need exporting to static SVG/PNG
before typesetting. Keep interactive versions as the book's companion website.

---

## 5. How the book is kept up to date (every session)

1. **Write findings the same session they happen** in `docs/JOURNAL.md`, using
   its entry format, with real numbers. Never reconstruct later.
2. **Every stage doc ends with a "What we found" section**, beside the
   theory: the measurements, the surprises, the mistakes. Link to journal
   entries rather than duplicating them at length.
3. **New terms go in `GLOSSARY.md`** the first time they appear.
4. **Every number gets a source**: a `results/` JSON, a manifest, or a
   `PROGRESS.md` results-log row.
5. **New figures go in `docs/visuals/`** and get a row in §4 above.
6. **Update the chapter status** in §3 when a stage doc reaches draft.

---

## 6. Building the web edition

```
uv run book/build_web.py                                   # preview: book/_site/index.html
uv run book/build_web.py --out ../personal_website/public/book
```

One page with every drafted chapter, a contents rail listing all planned
chapters (unwritten ones greyed), and the figures copied beside it. It is read
straight from the docs listed in §3, so updating a doc and its status row is
all a new chapter needs. `docs/book_intro.md` is the opening "About this
draft" note. The page is served at https://karanjakhar.dev/book/ from the
owner's personal-site repo (`public/book/`). Pushing that repo does **not**
publish it: after committing, run `npm run deploy` there (Node 22+; it builds
the site and runs `wrangler deploy`).

No PDF or EPUB yet. A book toolchain (Quarto) was considered for those and not
installed (journal, 2026-10-31).

## 7. Open questions for the book

- Final title, and whether the book follows stage order strictly or groups
  some stages (e.g. 14 and 15) into one chapter.
- Format and publisher: self-published (Leanpub, print-on-demand) or
  traditional. This affects how the figures need to be produced.
- Whether the companion repo is tagged per chapter, so readers can check out
  "the code as of chapter 5".
