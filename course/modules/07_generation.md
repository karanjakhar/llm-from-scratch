# Module 7: Making it talk

**You will build** the sampling knobs every chat interface exposes
(temperature, top-k, top-p, a repetition penalty), the function that picks
the next token, and generation with a KV cache: the trick that stops a model
from recomputing its whole past for every new token.

**Time:** 2-3 hours. **Hardware:** GPU. **Book chapter:** [9, Making it talk](../../docs/07_generation.md).

---

## 1. Why

A trained model gives a score to every token in the vocabulary for the next
position. Turning that into text means choosing one, appending it and asking
again, hundreds of times. How you choose decides whether the output is
repetitive, sensible or random. How you avoid recomputing decides whether it
takes a second or half a minute.

What breaks without the cache: generating 1,008 tokens from a 92M model took
the book's GPU 23.7 seconds the simple way and 3.2 seconds with the cache,
with exactly the same tokens out.

## 2. Read

Chapter 9 of the book:

- §1 *The loop* and §2 *Greedy*.
- §3 *Sampling, and temperature* and §4 *Top-k and top-p*.
- §6 *The KV cache*, all of it, including §6e: the cached and uncached
  outputs must be identical, and how to test that.

## 3. Implement

Open `stages/07_generation/sample.py`. Six blanks:

| # | Function | What it does |
|---|---|---|
| 1 | `apply_temperature` | sharpen or flatten the distribution |
| 2 | `top_k_filter` | keep only the k best tokens |
| 3 | `top_p_filter` | keep the smallest set whose probability reaches p |
| 4 | `apply_repetition_penalty` | push down tokens already used |
| 5 | `sample_next` | all of the above, then draw one token |
| 6 | `generate_cached` | prefill the prompt once, then one token per step |

`generate_naive`, the version without a cache, is left complete above
`generate_cached`. Read it first: yours must produce identical tokens.

## 4. Check

```bash
uv run python course/check.py 7
```

16 tests, including: a single token holding 95% of the probability survives
top-p at 0.9; the repetition penalty lowers both positive and negative
logits; cached and uncached logits agree; greedy and seeded-sampled outputs
are identical with and without the cache; generation stops at
`<|endoftext|>`.

## 5. Run

```bash
uv run python stages/07_generation/run.py speed
```

About 70 seconds; random untrained models, because speed does not depend on
what a model knows. Three sizes, generating up to 1,024 tokens, with and
without the cache. The book's speedups:

| model | speedup at 512 tokens | at 1,024 tokens |
|---|---|---|
| 5M (4 layers, 256 wide) | 1.27× | 2.28× |
| 30M (8 layers, 512 wide) | 3.32× | 5.51× |
| 92M (12 layers, 768 wide) | 4.47× | 7.23× |

A rerun on the book's GPU while writing this course gave 1.26 / 2.15,
3.37 / 5.53 and 4.50 / 7.38, so expect a few percent of run-to-run timing
noise, and different absolute speeds on a different GPU. Every row must also
say `identical_greedy: True` and `identical_sampled: True`; if one says
False, your cache has a bug the tests missed.

Look at the smallest model's shortest runs: the cache made it slightly
*slower* (0.94× at 128 tokens). Chapter §8a explains why a tiny model gains
little: the GPU is waiting on launch overhead, not on arithmetic.

## 6. Break it

In `generate_cached`, after the prefill, feed the model the whole sequence
`out` each step instead of only the new token `nxt` (keep passing the cache).
Run the tests. When the course was written, two failed
(`test_greedy_naive_equals_cached` and the sampled version) with
`RuntimeError: The expanded size of the tensor (0) must match the existing
size (11)`: every step writes the whole sequence into the cache again, so it
fills up after a few steps and the next write has no room. Before it fills,
the model sees each earlier token several times at the wrong positions. Then
put it back.

## 7. Check yourself

<details><summary>1. Temperature 0 is handled as a special case. What would dividing the logits by 0 do, and what does temperature → 0 approach?</summary>

Division by zero gives infinities and NaNs after softmax. As temperature
falls toward 0 the distribution concentrates on the highest logit, so the
limit is greedy decoding (argmax), which is what the special case does.
</details>

<details><summary>2. Top-p at p = 0.9, and one token has probability 0.95. Why must it survive, and which comparison makes sure it does?</summary>

Removing it would leave nothing. The filter drops a token only when the
tokens ranked above it already reach p; for the top token that sum is 0, so
it always stays.
</details>

<details><summary>3. What exactly does the KV cache store, and why do queries not need caching?</summary>

The keys and values of every earlier position, for every layer. A new token
only needs its own query, compared against all keys; earlier queries are
never used again.
</details>

<details><summary>4. The cache memory grows with what? Work it out for the 92M model at 1,024 tokens in fp32.</summary>

2 (K and V) × layers × positions × width × 4 bytes = 2 × 12 × 1,024 × 768 ×
4 = 75.5 MB, which is the `cache_bytes_theory` the script prints.
</details>

---

**Stuck?** Solution:
[`stages/07_generation/sample.py`](https://github.com/karanjakhar/llm-from-scratch/blob/main/stages/07_generation/sample.py).

**Next:** Module 8, the training loop: AdamW, the learning-rate schedule and
gradient clipping, by hand.
