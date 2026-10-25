"""Fast BPE encoder for production use. Training stays in stages/01_tokenizer.

Same algorithm and same file format as stages/01_tokenizer/bpe.py, so any
tokenizer trained there loads here and produces IDENTICAL ids
(test_tokenizer.py asserts this on real files). What changes is speed, from
two observations in docs/02_data.md §8:

1. Pre-tokenized chunks are Zipf-distributed. 20 MB of Python yields 4.9M
   chunks but only ~110k distinct ones. Caching chunk -> ids means the merge
   loop runs only on chunks never seen before.
2. Whole documents are independent, so corpus-scale encoding parallelizes
   across processes (see stages/02_data/prepare.py).
"""

from __future__ import annotations

import json

import regex as re


class Tokenizer:
    def __init__(self, merges: dict[tuple[int, int], int], pattern: str,
                 special_tokens: dict[str, int] | None = None,
                 cache_size: int = 1_000_000):
        self.merges = merges                       # pair -> new id; insertion order = rank
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.special_tokens = dict(special_tokens or {})
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab

        self._cache: dict[str, list[int]] = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, path: str, **kw) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if not payload.get("pattern"):
            raise ValueError("llmfs.Tokenizer needs a regex tokenizer (RegexBPE output)")
        merges = {(a, b): idx for a, b, idx in payload["merges"]}
        return cls(merges, payload["pattern"], payload.get("special_tokens"), **kw)

    @property
    def n_learned(self) -> int:
        """256 bytes + merges. The model's embedding table needs vocab_size, not this."""
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        """Rows the embedding table needs: learned tokens + special tokens.

        Computed from the ids actually present, not from the requested size --
        BPE can stop early (docs/01_tokenizer.md §3).
        """
        ids = list(self.vocab) + list(self.inverse_special)
        return max(ids) + 1

    @property
    def eot_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    # -- encode -----------------------------------------------------------

    def _encode_chunk(self, chunk: bytes) -> list[int]:
        """Identical to stages/01_tokenizer/bpe.py::_encode_chunk.

        Repeatedly merge the adjacent pair with the lowest rank. Written with
        local-variable lookups because this is the hot loop on a cache miss.
        """
        ids = list(chunk)
        merges = self.merges
        while len(ids) >= 2:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = merges.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            # Merge every occurrence of the winning pair, left to right,
            # non-overlapping -- the same semantics as stage 01's merge().
            a, b = ids[best_i], ids[best_i + 1]
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_rank)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode, treating any special-token string as plain text. docs/01 §5."""
        out: list[int] = []
        cache = self._cache
        for chunk in self.compiled.findall(text):
            ids = cache.get(chunk)
            if ids is None:
                self.cache_misses += 1
                ids = self._encode_chunk(chunk.encode("utf-8"))
                if len(cache) >= self._cache_size:
                    # Crude but adequate: the Zipf head refills almost
                    # immediately, so a full flush costs very little.
                    cache.clear()
                cache[chunk] = ids
            else:
                self.cache_hits += 1
            out.extend(ids)
        return out

    def encode(self, text: str, allowed_special: str = "none") -> list[int]:
        if allowed_special == "none" or not self.special_tokens:
            return self.encode_ordinary(text)
        if allowed_special != "all":
            raise ValueError("allowed_special must be 'none' or 'all'")
        pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def encode_document(self, text: str) -> list[int]:
        """A pretraining document: ordinary encoding + exactly one EOT at the end.

        User/corpus text is ALWAYS encoded with specials disabled. A GitHub file
        that literally contains "<|endoftext|>" (tokenizer code does) must not
        be able to inject a document boundary. docs/02_data.md §7.
        """
        ids = self.encode_ordinary(text)
        ids.append(self.eot_id)
        return ids

    # -- decode -----------------------------------------------------------

    def decode_bytes(self, ids) -> bytes:
        parts = []
        for idx in ids:
            idx = int(idx)
            if idx in self.vocab:
                parts.append(self.vocab[idx])
            elif idx in self.inverse_special:
                parts.append(self.inverse_special[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        return b"".join(parts)

    def decode(self, ids) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")
