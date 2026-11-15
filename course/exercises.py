"""Which functions each course module asks the learner to write.

docs/COURSE.md §3-5. make_starter.py blanks exactly these functions in the
starter repo; check.py runs the module's tests and reports which of them are
still blank. Nothing here edits stages/: the solutions stay where they are.

Each exercise is "path::qualname" plus a hint. The hint replaces the body,
under the original docstring, so it should point at the idea without giving
the code: a book section, a shape, a trap.

Modules build on each other: a later module's experiment runs on the code
the learner wrote in earlier ones (module 7 samples from the GPT of module 6,
module 9 trains on the loader of module 2). That is deliberate.
"""

MODULES = {
    0: dict(
        title="Setup and the memory budget",
        stage="00_setup",
        lesson="00_setup",
        chapter="docs/00_setup.md",
        tests=["course/module00/test_budget.py"],
        exercises={
            "course/module00/budget.py::linear_params":
                "A Linear layer stores a (d_out, d_in) weight matrix, plus one bias per output if bias=True.",
            "course/module00/budget.py::training_bytes":
                "Count how many tensors the size of the weights exist during training: the weights, "
                "their gradients, and STATE_TENSORS[optimizer] more. Each value takes bytes_per_value.",
            "course/module00/budget.py::max_params":
                "Take the reserve off first, then divide what is left by the cost of one parameter "
                "(training_bytes(1, ...)). Integer division; never below zero.",
            "course/module00/budget.py::measure_training_bytes":
                "Read torch.cuda.memory_allocated() before anything, after moving the model, after "
                "backward, and after opt.step(). Each entry is a difference between two readings.",
        },
        experiment="uv run python stages/00_setup/verify_env.py",
    ),
    1: dict(
        title="Tokenizer",
        stage="01_tokenizer",
        lesson="01_tokenizer",
        chapter="docs/01_tokenizer.md",
        tests=["stages/01_tokenizer/test_bpe.py"],
        exercises={
            "stages/01_tokenizer/bpe.py::get_stats":
                "Walk neighbouring pairs (zip the list with itself shifted by one) and count them "
                "into `counts`, creating a Counter if none was given.",
            "stages/01_tokenizer/bpe.py::merge":
                "Scan left to right. On a match, emit new_id and skip two; otherwise copy one id "
                "and skip one. Occurrences must not overlap: [1,1,1] with (1,1) -> [new,1].",
            "stages/01_tokenizer/bpe.py::BasicBPE._build_vocab":
                "Start from the 256 single bytes; each merge's bytes are its two parents' bytes "
                "joined. Merges are stored in the order they were learned, so parents come first.",
            "stages/01_tokenizer/bpe.py::BasicBPE.train":
                "docs/01_tokenizer.md §3: bytes -> count pairs -> take the most frequent -> merge "
                "it everywhere -> record it with id 256+i. Stop early if no pairs are left. "
                "Set self.merges and rebuild self.vocab at the end.",
            "stages/01_tokenizer/bpe.py::BasicBPE._encode_chunk":
                "Apply learned merges in the order they were learned (lowest id first), not "
                "longest-match: repeatedly find the present pair with the smallest merge id, "
                "merge it, stop when no present pair is a known merge. §3 'Encoding new text'.",
            "stages/01_tokenizer/bpe.py::BasicBPE.decode":
                "Look each id up in self.vocab (or the special tokens), join the bytes, decode "
                "UTF-8 with errors='replace'. An unknown id is a ValueError, not silence.",
            "stages/01_tokenizer/bpe.py::RegexBPE.train":
                "Same loop as BasicBPE.train, but on chunks from self.compiled.findall(text): "
                "count pairs inside each chunk only, so no merge crosses a chunk boundary. "
                "Faster: count each UNIQUE chunk once, weighted by how often it occurs "
                "(a test checks you get the same merges as the slow way).",
            "stages/01_tokenizer/bpe.py::RegexBPE.encode_ordinary":
                "Split with self.compiled.findall, encode each chunk's UTF-8 bytes on its own, "
                "concatenate.",
        },
        experiment="uv run python stages/01_tokenizer/run.py --vocab 1024",
    ),
    2: dict(
        title="Data",
        stage="02_data",
        lesson="02_data",
        chapter="docs/02_data.md",
        tests=["stages/02_data/test_data.py", "llmfs/data/test_dedup.py"],
        exercises={
            "llmfs/data/corpus.py::split_of":
                "Hash the key with hashlib.blake2b (8-byte digest, read as an int), take it mod "
                "10,000 and compare with VAL_PER_10K. Not Python's hash(): it changes every run.",
            "stages/02_data/prepare.py::Deduper.__call__":
                "A generator. For each doc, hash its text (blake2b, 16-byte digest); if the hash "
                "was seen, count it in stats['dup_docs'] and skip it, else remember it and yield.",
            "llmfs/data/dedup.py::signature":
                "For each of the P hash functions h(x) = a*x + b (uint64, wrapping), the minimum "
                "over all shingles. Process shingles in chunks of `chunk` to bound memory; return "
                "the top 32 bits of each minimum as uint32. docs/02_data.md §4.",
            "llmfs/data/dedup.py::estimated_jaccard":
                "The fraction of positions where the two signatures agree.",
            "llmfs/data/loader.py::ShuffledSampler.next":
                "Return the start offset of the next window in this epoch's permutation. When the "
                "epoch is used up, move to the next epoch and make its permutation first.",
            "llmfs/data/loader.py::MixedLoader.next_batch":
                "Pick a source for each of the B rows with self.mix_rng and self.weights; read T+1 "
                "tokens from that source at its sampler's next offset; count it. Move to "
                "self.device (pinned if self.pin) and return inputs and targets shifted by one.",
        },
        experiment="uv run python stages/02_data/run.py leakage --shards 2",
    ),
    3: dict(
        title="Embeddings and position",
        stage="03_embeddings",
        lesson="03_embeddings",
        chapter="docs/03_embeddings_position.md",
        tests=["stages/03_embeddings/test_embeddings.py"],
        exercises={
            "stages/03_embeddings/embeddings.py::TokenEmbedding.forward":
                "One indexing operation: each id selects its row of self.weight. (B, T) -> (B, T, C).",
            "stages/03_embeddings/embeddings.py::LearnedPosition.forward":
                "The first T rows of the table. Asking for more than max_len rows is a ValueError: "
                "there is no row to return (docs §4, the hard wall).",
            "stages/03_embeddings/embeddings.py::frequencies":
                "omega_i = base^(-2i/dim) for i = 0 .. dim/2 - 1, computed in float64. dim must be even.",
            "stages/03_embeddings/embeddings.py::sinusoidal_table":
                "angles = position * omega, shape (T, dim/2). Even columns get sin, odd columns cos. "
                "Compute in float64, return float32.",
            "stages/03_embeddings/embeddings.py::rope_angles":
                "Like the sinusoidal angles, for positions offset .. offset+T-1, using the head size d. "
                "float64 for the multiply, then cast to `dtype`.",
            "stages/03_embeddings/embeddings.py::rope_interleaved":
                "Pairs are dims (0,1), (2,3), ... For each pair (a, b): (a cos - b sin, a sin + b cos). "
                "Cast cos and sin to x's dtype.",
            "stages/03_embeddings/embeddings.py::rope_half":
                "The same rotation with pairs (i, i + d/2): first half is a, second half is b.",
        },
        experiment="uv run python stages/03_embeddings/run.py ablation",
    ),
    4: dict(
        title="Attention",
        stage="04_attention",
        lesson="04_attention",
        chapter="docs/04_attention.md",
        tests=["stages/04_attention/test_attention.py"],
        exercises={
            "stages/04_attention/attention.py::causal_mask":
                "A (T, T) boolean matrix, True where key j <= query i: the lower triangle.",
            "stages/04_attention/attention.py::attention":
                "scores = q @ k^T, divided by sqrt(d) if scale. If causal, set disallowed scores to "
                "-inf (not 0). Softmax over the last dim, then weights @ v. docs §2-4.",
            "stages/04_attention/attention.py::MultiHeadAttention.forward":
                "qkv -> split into q, k, v (B, T, C) -> view as (B, T, H, d) and transpose to (B, H, T, d). "
                "Attend (keep the weights in self.last_weights if self.keep_weights), transpose back, "
                "reshape to (B, T, C), then self.proj. The reshape is where bugs hide.",
        },
        experiment="uv run python stages/04_attention/run.py variance",
    ),
    5: dict(
        title="The transformer block",
        stage="05_block",
        lesson="05_block",
        chapter="docs/05_transformer_block.md",
        tests=["stages/05_block/test_block.py"],
        exercises={
            "stages/05_block/block.py::LayerNorm.forward":
                "Per token, over the last dim: subtract the mean, divide by sqrt(variance + eps), "
                "scale by weight, add bias if there is one. The BIASED variance (divide by C).",
            "stages/05_block/block.py::RMSNorm.forward":
                "x divided by sqrt(mean(x^2) + eps), times weight. No mean subtraction, no bias.",
            "stages/05_block/block.py::gelu_tanh":
                "0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). GPT-2 uses exactly this.",
            "stages/05_block/block.py::gelu_exact":
                "x times the standard normal CDF, written with erf: 0.5 x (1 + erf(x / sqrt 2)).",
            "stages/05_block/block.py::silu":
                "x times sigmoid(x).",
            "stages/05_block/block.py::MLP.forward":
                "Up to the hidden width, the activation, back down. (B, T, C) -> (B, T, 4C) -> (B, T, C).",
            "stages/05_block/block.py::SwiGLU.forward":
                "proj( silu(gate(x)) * up(x) ): the gate decides how much of `up` passes through.",
        },
        experiment="uv run python stages/05_block/run.py grads --B 8",
    ),
    6: dict(
        title="GPT",
        stage="06_gpt",
        lesson="06_gpt",
        chapter="docs/06_gpt_model.md",
        tests=["stages/06_gpt/test_gpt.py"],
        exercises={
            "stages/06_gpt/gpt.py::CausalSelfAttention.forward":
                "Module 4's multi-head attention under GPT-2's names (c_attn, c_proj). If self.flash, "
                "use F.scaled_dot_product_attention(q, k, v, is_causal=True), else your attention().",
            "stages/06_gpt/gpt.py::Block.forward":
                "Pre-norm: x + attn(ln_1(x)), then x + mlp(ln_2(x)).",
            "stages/06_gpt/gpt.py::GPT._scale_residual_init":
                "Only if cfg.scaled_init: every module marked writes_residual gets normal init with "
                "std 0.02 / sqrt(2 * n_layer). docs §4b.",
            "stages/06_gpt/gpt.py::GPT.forward":
                "Token embedding + position embedding, every block, ln_f, lm_head. With targets, the "
                "cross-entropy over all positions in float32. keep_stream also returns the list of "
                "x after the embedding and after each block.",
            "stages/06_gpt/gpt.py::load_gpt2_state":
                "docs §5's three traps: skip the saved attention-mask buffers, skip lm_head (it is "
                "tied), transpose the TRANSPOSED weights (Conv1D stores (in, out)). Add 'transformer.' "
                "to names that lack it. Raise on any tensor with no home, any shape mismatch, and any "
                "of our tensors left unfilled.",
            "stages/06_gpt/gpt.py::count_params_by_hand":
                "Per block: two LayerNorms (2C each), c_attn C*3C, attn c_proj C*C, c_fc C*4C, mlp "
                "c_proj 4C*C, plus biases if cfg.bias. Then wte V*C, wpe T*C, ln_f 2C, and lm_head "
                "only if untied. Return the parts, 'total' and 'non_embedding'.",
        },
        experiment="uv run python stages/06_gpt/run.py oracle",
    ),
    7: dict(
        title="Generation",
        stage="07_generation",
        lesson="07_generation",
        chapter="docs/07_generation.md",
        tests=["stages/07_generation/test_sample.py"],
        exercises={
            "stages/07_generation/sample.py::apply_temperature":
                "Divide the logits by tau.",
            "stages/07_generation/sample.py::top_k_filter":
                "Nothing to do if k <= 0 or k >= V. Otherwise every logit below the k-th largest "
                "becomes NEG_INF (ties with the k-th survive).",
            "stages/07_generation/sample.py::top_p_filter":
                "Sort descending, softmax, cumulative sum. Drop a token when the tokens ABOVE it "
                "already reach p, so the token that crosses p survives. Scatter the mask back to "
                "vocabulary order before applying it. docs §4.",
            "stages/07_generation/sample.py::apply_repetition_penalty":
                "Gather the logits of tokens already in context; divide the positive ones by the "
                "penalty and multiply the negative ones, so both go down; scatter back.",
            "stages/07_generation/sample.py::sample_next":
                "float() the logits; repetition penalty if context is given; temperature 0 means "
                "argmax. Otherwise temperature, top-k, top-p, softmax, torch.multinomial with the "
                "generator. Returns (B, 1).",
            "stages/07_generation/sample.py::generate_cached":
                "Make a cache with model.make_cache(B, T + max_new_tokens, device, dtype); run the "
                "whole prompt once with it (last_only=True); then feed ONE new token per step. Stop "
                "on eot or after max_new_tokens tokens. Compare generate_naive above.",
        },
        experiment="uv run python stages/07_generation/run.py speed",
    ),
    8: dict(
        title="Training loop",
        stage="08_training",
        lesson="08_training",
        chapter="docs/08_training_loop.md",
        tests=["stages/08_training/test_training.py"],
        exercises={
            "stages/08_training/optim.py::SGD.step":
                "For each parameter with a gradient: with momentum, m = momentum*m + g and step "
                "along m; without, step along g. Step size lr, downhill. In place.",
            "stages/08_training/optim.py::AdamW.step":
                "t += 1. Per parameter: decoupled decay shrinks p by (1 - lr*wd) (or, if self.l2, "
                "wd*p is added to the gradient instead); m and v are running averages of g and g^2; "
                "bias-correct both with 1 - beta^t; p -= lr * m_hat / (sqrt(v_hat) + eps). docs §2.",
            "stages/08_training/optim.py::lr_schedule":
                "Linear warmup from peak/warmup to peak over `warmup` steps, then cosine from peak "
                "down to min_ratio*peak at step `total`.",
            "stages/08_training/optim.py::clip_grad_norm":
                "The norm of ALL gradients together (one vector). If it exceeds max_norm, scale "
                "every gradient by max_norm / (norm + 1e-6). Return the norm before clipping.",
        },
        experiment="uv run python stages/08_training/run.py overfit",
    ),
    9: dict(
        title="Pretraining",
        stage="11_pretrain",
        lesson="09_pretraining",
        chapter="docs/11_pretraining.md",
        tests=["llmfs/train/test_resume.py"],
        exercises={
            "llmfs/train/trainer.py::save":
                "Everything the next step depends on: the model weights (from the uncompiled model, "
                "with '.block.' removed from names), the optimizer state, the loader state, the "
                "step, the configs, both random generators' states and hist. Write to a .tmp file "
                "and rename it over `path`, so a crash mid-save keeps the old checkpoint.",
        },
        experiment="uv run python stages/11_pretrain/run.py resume_test",
    ),
    10: dict(
        title="Evaluation",
        stage="22_evaluation",
        lesson="10_evaluation",
        chapter="docs/22_evaluation.md",
        tests=["stages/22_evaluation/test_eval.py"],
        exercises={
            "llmfs/eval/codeeval.py::truncate":
                "HumanEval: the prompt already opened the function, so keep lines until the first "
                "non-empty line that starts at column 0. MBPP: allow imports and the first def, "
                "then stop at the next top-level line after the def has started.",
            "llmfs/eval/codeeval.py::pass_at_k":
                "The unbiased estimator: 1 - C(n-c, k) / C(n, k), computed as a product so it never "
                "overflows. If n - c < k every draw of k contains a pass: return 1.",
            "llmfs/eval/execution.py::run_program":
                "Write src plus a final print of SENTINEL into a temporary directory and run it with "
                "sys.executable -I, the timeout, preexec_fn=_limits(mem_mb) and a minimal env. Pass "
                "only if the exit code is 0 AND the sentinel is on stdout. Return (passed, reason).",
        },
        experiment="uv run python stages/22_evaluation/run.py table --redo",
    ),
    11: dict(
        title="Instruction tuning",
        stage="16_sft",
        lesson="11_sft",
        chapter="docs/16_sft.md",
        tests=["stages/16_sft/test_sft.py"],
        exercises={
            "stages/16_sft/sft.py::render":
                "<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n, then the reply and "
                "<|im_end|>. Labels are IGNORE for the whole prompt and equal to the ids for the "
                "reply and its closing <|im_end|>. Without a reply, return the prompt only.",
            "stages/16_sft/sft.py::split_function":
                "Parse with ast after textwrap.dedent. Keep only a single top-level function with "
                "no self/cls, a docstring, and a body after it. Return (signature + docstring, "
                "whole function); None for anything else.",
            "stages/16_sft/sft.py::collate":
                "Right-pad ids with pad_id and labels with IGNORE to the longest example (labels = "
                "ids instead if include_prompt_loss). Then shift: x = ids[:, :-1], y = labels[:, 1:].",
        },
        experiment="uv run python stages/16_sft/run.py --base checkpoints/11_pretrain/workhorse/final.pt --tag mine",
    ),
}
