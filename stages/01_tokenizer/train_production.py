"""Stage 01, domain work — train the production tokenizer on the real corpus.

Shakespeare was a fixture. This trains on what the model will actually see:
permissive-licensed Python from codeparrot-clean plus fineweb-edu English,
TRAIN SPLIT ONLY (the tokenizer must not see validation repositories either),
and decides two open questions from docs/01_tokenizer.md §6 and §4:

    1. Vocab size: is 8192 enough for code, or does 16384 buy enough to pay
       for its larger embedding table?
    2. Digits: grouped (GPT-2, \\p{N}+), capped at 3 (GPT-4 / Llama 3), or
       split one per token (Llama 1/2, Qwen2)?

Only the digit rule differs between variants; everything else in the pattern
is GPT-2's, so the comparison isolates that one choice.

BPE merges are a prefix sequence: training is greedy and deterministic, so
the first k merges of a long run ARE the k-merge tokenizer. We therefore train
each variant once to the largest vocab and read every smaller vocab off as a
prefix. test_bpe.py asserts the prefix property.

Run (needs data/raw from stages/02_data/download.py):
    uv run python stages/01_tokenizer/train_production.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import multiprocessing as mp
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO))

from bpe import GPT2_SPLIT_PATTERN, RegexBPE  # noqa: E402

from llmfs.data.corpus import code_shards, en_shards, iter_code_docs, iter_en_docs  # noqa: E402

RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "tokenizers"

# The digit clause in GPT-2's pattern, and its two replacements.
_GPT2_DIGITS = r" ?\p{N}+"
assert _GPT2_DIGITS in GPT2_SPLIT_PATTERN
PATTERNS = {
    "grouped": GPT2_SPLIT_PATTERN,                                       # "2024" one chunk
    "cap3":    GPT2_SPLIT_PATTERN.replace(_GPT2_DIGITS, r" ?\p{N}{1,3}"),  # "2024" -> "202","4"
    "split":   GPT2_SPLIT_PATTERN.replace(_GPT2_DIGITS, r" ?\p{N}"),       # "2024" -> "2","0","2","4"
}

PREFIX_VOCABS = (2048, 4096, 8192, 16384)


def take_bytes(docs, budget: int) -> list[str]:
    out, n = [], 0
    for d in docs:
        out.append(d.text)
        n += len(d.text.encode("utf-8"))
        if n >= budget:
            break
    return out


def build_texts(code_mb: float, en_mb: float, val_code_mb: float, val_en_mb: float):
    cs, es = code_shards(RAW), en_shards(RAW)
    assert cs and es, "no shards found; run stages/02_data/download.py first"
    MB = 1_000_000
    train = (take_bytes(iter_code_docs(cs, "train"), int(code_mb * MB))
             + take_bytes(iter_en_docs(es, "train"), int(en_mb * MB)))
    val_code = take_bytes(iter_code_docs(cs, "val"), int(val_code_mb * MB))
    val_en = take_bytes(iter_en_docs(es, "val"), int(val_en_mb * MB))
    # Joined with "\n": a document boundary becomes ordinary whitespace for the
    # purposes of merge statistics. <|endoftext|> is a special token and must
    # never take part in merges, so it is deliberately absent here.
    return "\n".join(train), "\n".join(val_code), "\n".join(val_en)


def train_variant(args):
    name, train_text, vocab = args
    t0 = time.perf_counter()
    tok = RegexBPE(PATTERNS[name])
    tok.train(train_text, vocab)
    return name, tok.merges, time.perf_counter() - t0


def truncate(merges: dict, vocab: int, pattern: str) -> RegexBPE:
    """The first (vocab - 256) merges form the smaller tokenizer exactly."""
    tok = RegexBPE(pattern)
    tok.merges = dict(list(merges.items())[: vocab - 256])
    tok.vocab = tok._build_vocab()
    return tok


def ratio(tok, text: str) -> float:
    return len(text.encode("utf-8")) / len(tok.encode_ordinary(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-mb", type=float, default=30)
    ap.add_argument("--en-mb", type=float, default=6,
                    help="~15%% of training bytes, matching the pretraining mix")
    ap.add_argument("--max-vocab", type=int, default=16384)
    args = ap.parse_args()

    t0 = time.perf_counter()
    train_text, val_code, val_en = build_texts(args.code_mb, args.en_mb, 3, 1.5)
    print(f"train sample {len(train_text)/1e6:.1f} MB | held-out code "
          f"{len(val_code)/1e6:.1f} MB, English {len(val_en)/1e6:.1f} MB "
          f"(val repositories only) | built in {time.perf_counter()-t0:.0f}s")

    # One process per variant. Each holds the sample (~36 MB) plus its unique
    # chunk table; RAM is not the constraint here, single-core speed is.
    print(f"training {len(PATTERNS)} variants to vocab {args.max_vocab} in parallel...")
    # spawn, not fork: torch is already imported (threads) -- see prepare.py.
    with mp.get_context("spawn").Pool(len(PATTERNS)) as pool:
        results = pool.map(train_variant,
                           [(n, train_text, args.max_vocab) for n in PATTERNS])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, merges, secs in results:
        print(f"  {name:<8} {len(merges)} merges in {secs/60:.1f} min")
        for v in PREFIX_VOCABS:
            if v > args.max_vocab:
                continue
            tok = truncate(merges, v, PATTERNS[name])
            rows.append({
                "digits": name, "vocab": v,
                "code_bpt": ratio(tok, val_code),
                "en_bpt": ratio(tok, val_en),
            })
            tok.save(str(OUT / f"{name}_{v}.json"))

    # Reference points: production tokenizers on the same held-out text.
    import tiktoken
    refs = []
    for enc_name in ("gpt2", "cl100k_base"):
        enc = tiktoken.get_encoding(enc_name)
        refs.append({
            "name": f"{enc_name} ({enc.n_vocab})",
            "code_bpt": len(val_code.encode()) / len(enc.encode_ordinary(val_code)),
            "en_bpt": len(val_en.encode()) / len(enc.encode_ordinary(val_en)),
        })

    print()
    print("HELD-OUT BYTES PER TOKEN (higher = better compression)")
    print(f"  {'digits':<8} {'vocab':>6}  {'code':>6}  {'English':>8}   "
          f"embedding params @ d=512 / d=768")
    for r in rows:
        print(f"  {r['digits']:<8} {r['vocab']:>6}  {r['code_bpt']:>6.3f}  "
              f"{r['en_bpt']:>8.3f}   {r['vocab']*512/1e6:>5.1f}M / {r['vocab']*768/1e6:>5.1f}M")
    for r in refs:
        print(f"  {r['name']:<15}  {r['code_bpt']:>6.3f}  {r['en_bpt']:>8.3f}")

    (OUT / "results.json").write_text(json.dumps(
        {"args": vars(args), "variants": rows, "reference": refs,
         "train_minutes": {n: s / 60 for n, _, s in results}}, indent=1))
    print(f"\nsaved tokenizers + results.json to {OUT}")


if __name__ == "__main__":
    main()
