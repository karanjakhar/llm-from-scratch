"""Stage 01 experiment — train a real tokenizer, measure it, run the ablation.

Produces the numbers that go into PROGRESS.md -> Results log:

    1. compression ratio (bytes/token) for vocab 256 .. 8192
    2. the regex pre-tokenization ablation: BasicBPE vs RegexBPE merge lists
    3. encode speed vs tiktoken, so the cost of pure Python is explicit
    4. the tokenizer artifact itself, saved to data/tokenizer_8192.json

Run:  uv run python stages/01_tokenizer/run.py
      uv run python stages/01_tokenizer/run.py --vocab 4096 --corpus path.txt
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bpe import BasicBPE, RegexBPE, compression_ratio, render_token  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def get_corpus(path: str | None) -> tuple[str, str]:
    """Return (train_text, heldout_text). Held-out is the last 10%.

    The split matters: compression measured on training text is optimistic,
    because the merges were fitted to exactly those strings.
    """
    if path:
        text = Path(path).read_text(encoding="utf-8")
        name = Path(path).name
    else:
        DATA.mkdir(exist_ok=True)
        local = DATA / "tinyshakespeare.txt"
        if not local.exists():
            print(f"downloading {CORPUS_URL}")
            urllib.request.urlretrieve(CORPUS_URL, local)
        text = local.read_text(encoding="utf-8")
        name = "tinyshakespeare"
    cut = int(len(text) * 0.9)
    print(f"corpus: {name}  {len(text):,} chars  "
          f"{len(text.encode('utf-8')):,} bytes  "
          f"(train {cut:,} / held-out {len(text) - cut:,})")
    return text[:cut], text[cut:]


def experiment_compression(train_text: str, heldout: str, max_vocab: int):
    """How much does each doubling of the vocabulary actually buy?"""
    print()
    print("=" * 72)
    print("1. COMPRESSION vs VOCAB SIZE")
    print("=" * 72)
    print(f"  {'vocab':>7}  {'train b/tok':>12}  {'held-out b/tok':>15}  {'gain':>7}")

    sizes = [v for v in (256, 512, 1024, 2048, 4096, 8192, 16384) if v <= max_vocab]
    prev = None
    best = None
    for v in sizes:
        tok = RegexBPE()
        tok.train(train_text, v)
        r_train = compression_ratio(tok, train_text)
        r_held = compression_ratio(tok, heldout)
        gain = "—" if prev is None else f"{(r_held / prev - 1) * 100:+.1f}%"
        print(f"  {v:>7}  {r_train:>12.3f}  {r_held:>15.3f}  {gain:>7}")
        prev = r_held
        best = tok

    print()
    print("  Read this table for diminishing returns: each doubling of the vocab")
    print("  adds vocab x d_model embedding parameters but buys less and less")
    print("  compression. docs/01_tokenizer.md §6.")
    return best


def experiment_regex_ablation(train_text: str, vocab: int = 2048):
    """THE ablation for this stage: what does pre-tokenization prevent?"""
    print()
    print("=" * 72)
    print(f"2. ABLATION — regex pre-tokenization (vocab {vocab})")
    print("=" * 72)

    # BasicBPE is O(n) per merge over the WHOLE stream, so cap the text.
    sample = train_text[:200_000]

    t0 = time.perf_counter()
    basic = BasicBPE()
    basic.train(sample, vocab)
    t_basic = time.perf_counter() - t0

    t0 = time.perf_counter()
    rbpe = RegexBPE()
    rbpe.train(sample, vocab)
    t_regex = time.perf_counter() - t0

    def spanning(tok):
        """Tokens with an INTERIOR space -- the defect regex splitting prevents.

        Leading spaces (" the") are correct and expected. All-whitespace tokens
        are also fine: the '\\s+' branch makes whitespace runs their own chunk.
        """
        return [t for i, t in tok.vocab.items()
                if i >= 256 and t.strip() != b"" and b" " in t[1:]]

    sb, sr = spanning(basic), spanning(rbpe)
    print(f"  BasicBPE  trained in {t_basic:6.1f}s  "
          f"boundary-spanning tokens: {len(sb):4d} / {vocab - 256}")
    print(f"  RegexBPE  trained in {t_regex:6.1f}s  "
          f"boundary-spanning tokens: {len(sr):4d} / {vocab - 256}")
    print()
    print("  Examples of what BasicBPE learns that RegexBPE cannot:")
    for t in sb[:15]:
        print(f"    {render_token(t)!r}")
    print()
    print("  Each of those spends a vocabulary slot on a phrase whose pieces")
    print("  already have slots, and makes a word's tokenization depend on what")
    print("  follows it. Compression on held-out text:")
    print(f"    BasicBPE {compression_ratio(basic, train_text[-50_000:]):.3f}  "
          f"RegexBPE {compression_ratio(rbpe, train_text[-50_000:]):.3f} bytes/token")
    print()
    print("  Note BasicBPE can score HIGHER on text similar to its training")
    print("  corpus -- memorized phrases compress well. It generalizes worse,")
    print("  which is the point: compression alone is not the whole story.")


def experiment_vs_tiktoken(tok, heldout: str):
    """Compare against the production tokenizer: quality and speed."""
    print()
    print("=" * 72)
    print("3. VS TIKTOKEN (production reference)")
    print("=" * 72)
    try:
        import tiktoken
    except ImportError:
        print("  tiktoken not installed — skipping. `uv add tiktoken`")
        return

    sample = heldout[:100_000]
    n_bytes = len(sample.encode("utf-8"))

    rows = []
    t0 = time.perf_counter()
    ours = tok.encode(sample)
    rows.append((f"ours (vocab {len(tok.vocab)})", n_bytes / len(ours),
                 time.perf_counter() - t0))

    for name in ("gpt2", "cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        t0 = time.perf_counter()
        ids = enc.encode(sample)
        rows.append((f"{name} ({enc.n_vocab})", n_bytes / len(ids),
                     time.perf_counter() - t0))

    print(f"  {'tokenizer':>26}  {'bytes/token':>12}  {'encode s':>9}  {'MB/s':>7}")
    for name, ratio, secs in rows:
        print(f"  {name:>26}  {ratio:>12.3f}  {secs:>9.3f}  "
              f"{n_bytes / 1e6 / secs:>7.2f}")

    slow = rows[0][2] / min(r[2] for r in rows[1:])
    print()
    print(f"  Our pure-Python encoder is ~{slow:.0f}x slower than tiktoken's Rust.")
    print("  Two causes: (a) Python interpreter overhead, (b) our _encode_chunk")
    print("  rescans for the best pair every iteration — O(n*m) — while tiktoken")
    print("  uses a linked list with a priority queue. Algorithmic, not just")
    print("  language. This is why tokenizers are never written in Python.")
    print()
    print("  On quality: a 50k-100k vocab beats our 8k on bytes/token, as it")
    print("  must. The reason we choose 8192 is parameter budget, not quality —")
    print("  at 10-50M params a 50k embedding table would dominate the model.")


def inspect_tokenizer(tok):
    print()
    print("=" * 72)
    print("4. WHAT IT LEARNED")
    print("=" * 72)
    merged = [(i, tok.vocab[i]) for i in sorted(tok.vocab) if i >= 256]
    print("  first 30 merges (the most frequent pairs in the corpus):")
    print("   ", "  ".join(f"{render_token(t)!r}" for _, t in merged[:30]))
    print()
    print("  longest 15 tokens (what the corpus repeats most):")
    for i, t in sorted(merged, key=lambda x: -len(x[1]))[:15]:
        print(f"    {i:>6}  {render_token(t)!r}")

    print()
    print("  the failure cases from docs §7, on this tokenizer:")
    for s in ("strawberry", " strawberry", "1234", "2024", "héllo", "你好"):
        ids = tok.encode(s)
        pieces = [render_token(tok.vocab[i]) for i in ids]
        print(f"    {s!r:>14} -> {len(ids)} tokens  {pieces}")
    print()
    print("  Note 'strawberry' vs ' strawberry' tokenize differently (leading")
    print("  space is part of the token), and no token is a single letter —")
    print("  which is why the model cannot count the r's. docs §7.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192,
                    help="target vocab size for the saved tokenizer")
    ap.add_argument("--corpus", type=str, default=None,
                    help="path to a .txt corpus (default: download tinyshakespeare)")
    ap.add_argument("--skip-ablation", action="store_true",
                    help="skip the slow BasicBPE comparison")
    args = ap.parse_args()

    train_text, heldout = get_corpus(args.corpus)

    tok = experiment_compression(train_text, heldout, args.vocab)
    if not args.skip_ablation:
        experiment_regex_ablation(train_text)
    experiment_vs_tiktoken(tok, heldout)
    inspect_tokenizer(tok)

    DATA.mkdir(exist_ok=True)
    out = DATA / f"tokenizer_{len(tok.vocab)}.json"
    # Reserve ids above the learned vocab. The gap is deliberate: adding a
    # special token later would mean resizing the embedding table and
    # invalidating every checkpoint. docs §5.
    tok.register_special_tokens({
        "<|endoftext|>": len(tok.vocab),
        "<|pad|>": len(tok.vocab) + 1,
        "<|im_start|>": len(tok.vocab) + 2,
        "<|im_end|>": len(tok.vocab) + 3,
    })
    tok.save(str(out))
    print()
    print(f"saved {out}  (vocab {tok.vocab_size} incl. 4 special tokens)")
    print("Record the compression numbers in PROGRESS.md -> Results log.")


if __name__ == "__main__":
    main()
