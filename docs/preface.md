# Preface: why one laptop, why code

This book follows one project from an empty folder to a language model that
writes Python. It ran from 14 October to 14 November 2026 on a single laptop
GPU, an NVIDIA RTX 4060 with 8 GB of memory. Every component of the model was
written by hand, measured, and then broken on purpose to see what it had been
doing.

The model at the end is small. It has 92 million parameters and read 1.8
billion tokens of training data in about 20 hours. Given 164 standard
programming problems (HumanEval), it solves 5 on its first attempt: 3.05%.
That number was predicted before any training started, and the book is built
around why it is the right result to aim for, not a disappointing one.

## Why build it at all

There are good, free language models you can download. Building one from
scratch is not the fastest way to get a working model. It is the fastest way
to understand one.

Reading about attention, you learn that it "lets tokens look at each other".
Building it, you find that a 4-layer model trained for about 3 minutes grows a
head that puts 68% of its attention on the previous token, and two heads that
park weight on the first position for no use at all (chapter 6). Reading about
training, you learn that warmup "stabilises early training". Taking warmup
away at a high learning rate, you find that the loss does not explode. It
stalls at the level of a model that only knows how often each token appears,
and after 2,000 steps it is at 4.78 where the warmed-up run reached 2.95
(chapter 10). The second kind of knowing is the one that
lasts, and you only get it by doing the work.

So every chapter follows one rule: **build it, check it against a reference,
then break it.** The tokenizer is checked against OpenAI's `tiktoken`. The GPT
model loads OpenAI's real GPT-2 weights and must produce the same outputs as
Hugging Face's implementation. LoRA is checked against the `peft` library. Then
each piece is removed or damaged, and the damage is measured. That
measurement is where the understanding lands.

## Why one laptop

The constraint is the teacher. With 7,797 MB of usable GPU memory (an "8 GB"
card has less than 8 GB, chapter 1), nothing fits by default. The 92M model
does not fit at the batch size it wants, so the book has to explain exactly
where every byte goes, and then use gradient accumulation, mixed precision and
activation checkpointing to make it fit. Each of those is a chapter's worth of
understanding that a large cloud GPU lets you skip.

The constraint also keeps the book honest about scale. The arithmetic in
chapter 2 says that on this hardware the largest sensible model is about 125
million parameters, and that a model this size will not know facts, will not
reason over several steps, and will not follow general instructions. It will
learn grammar, syntax and local structure, and learn them properly. Saying so
up front means no chapter has to pretend.

Everything here was run on the one machine. The book never says "this should
help". A change either helped by a measured amount or it did not, and the
number is printed next to the claim.

## Why Python code

The choice of what the model learns came before any data was downloaded, and
it was decided by one property: **can a machine check the output?**

Stories, essays and translation have no automatic checker. Judging them needs
a person or a much larger model acting as judge, and at 100 million parameters
the differences between versions are small enough that such a judge mostly
measures noise. Python has a perfect checker: run the unit tests. That one
fact turns the later chapters into real experiments. Reinforcement learning
(chapter 22) gets its reward from tests passing. Evaluation (chapter 24)
counts problems solved. When the model gets better or worse, the number says
so without anyone's opinion. About 15% of the training data is high-quality
English, so comments and docstrings read like language.

## The mistakes stay in

A lab journal was kept the whole way through, written the same day things
happened: 116 entries, of which 18 are mistakes we made and caught and 12 are
traps that would have caught anyone. The obvious Python dataset turned out to
contain no Python, only file names and hashes (chapter 2). A training/test
split that looked careful still had 43% of test files mostly copied into the
training set, through forked repositories (chapter 4). One of our own tests
was wrong: it failed on correct code, and the fix belonged in the test
(chapter 3). A plot overwrote another chapter's figure.

These are in the book on purpose. They are what a reader will meet in their
own work, and how each one was caught is as useful as the thing it broke.

## How to read this book

Each chapter asks one question (what problem does this part solve, and what
breaks without it?), shows a picture of the mechanism, explains it in plain
words with small numbers, then gives the math with tensor shapes, the code,
the measurement, what surprised us, what we got wrong, and what happened when
we broke it.

You need to be comfortable with Python and with high-school algebra. You do
not need to have trained a model before. New terms are defined where they
first appear and collected in Appendix A. Every result in the book is in one
table in Appendix B, and Appendix C lists the commands that rebuild all of it
from a clean machine.

All the code is in the companion repository,
[github.com/karanjakhar/llm-from-scratch](https://github.com/karanjakhar/llm-from-scratch).
Paths like `stages/04_attention/run.py` in the text refer to files there.
