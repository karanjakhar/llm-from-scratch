"""Which functions each course module asks the learner to write.

docs/COURSE.md §3-5. make_starter.py blanks exactly these functions in the
starter repo; check.py runs the module's tests and reports which of them are
still blank. Nothing here edits stages/: the solutions stay where they are.

Each exercise is "path::qualname" plus a hint. The hint replaces the body,
under the original docstring, so it should point at the idea without giving
the code: a book section, a shape, a trap.
"""

MODULES = {
    1: dict(
        title="Tokenizer",
        stage="01_tokenizer",
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
}
