"""Static diagrams the book listed as wanted (docs/BOOK.md §4).

    uv run python docs/visuals/make_diagrams.py

Writes onehot_lookup.png, rope_rotation.png, packing_eot.png and
sampler_coverage.png next to this file. Every number drawn is either a toy
value shown as such or computed here (the coverage curve is a simulation).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent


def onehot_lookup():
    rng = np.random.default_rng(0)
    V, C, tok = 6, 4, 3
    E = np.round(rng.normal(0, 1, (V, C)), 1)
    onehot = np.zeros((1, V))
    onehot[0, tok] = 1
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[6, 4, 4]))
    for a, M, title in ((ax[0], onehot, f"one-hot for token id {tok}  (1 × V)"), (ax[1], E, "embedding table  (V × C)"),
                        (ax[2], E[tok:tok + 1], f"result = row {tok}  (1 × C)")):
        a.imshow(M, cmap="Blues" if M is onehot else "coolwarm", vmin=-2 if M is not onehot else 0, vmax=2 if M is not onehot else 1)
        for (i, j), v in np.ndenumerate(M):
            a.text(j, i, f"{v:g}", ha="center", va="center", fontsize=9)
        a.set_xticks([])
        a.set_yticks(range(M.shape[0]))
        a.set_title(title, fontsize=10)
    ax[1].axhline(tok - 0.5, color="black", lw=2)
    ax[1].axhline(tok + 0.5, color="black", lw=2)
    fig.text(0.415, 0.5, "×", fontsize=22, ha="center")
    fig.text(0.70, 0.5, "=", fontsize=22, ha="center")
    fig.suptitle("A lookup is a matrix product with a one-hot vector: only row `id` survives", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "onehot_lookup.png", dpi=130)


def rope_rotation():
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    base = np.array([1.0, 0.3])
    for a, (speed, label) in zip(ax, ((0.6, "fast pair (θ = 0.6 rad per position)"), (0.08, "slow pair (θ = 0.08 rad per position)"))):
        for p in range(6):
            ang = p * speed
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            v = R @ base
            a.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=plt.cm.viridis(p / 6), lw=2))
            a.text(v[0] * 1.08, v[1] * 1.08, f"pos {p}", fontsize=8)
        a.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", title=label)
        a.axhline(0, color="gray", lw=0.5)
        a.axvline(0, color="gray", lw=0.5)
    fig.suptitle("RoPE rotates each (x, y) pair of q and k by position × its own speed;\n"
                 "q·k then depends only on the difference in positions", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "rope_rotation.png", dpi=130)


def packing_eot():
    docs = [("def f(x):\\n  return x", 7), ("import os", 3), ("class A: ...", 5), ("# readme text", 6), ("print(1)", 4)]
    T = 12
    fig, ax = plt.subplots(figsize=(11, 2.8))
    pos, row = 0, 0
    colors = plt.cm.tab10.colors
    for i, (name, n) in enumerate(docs):
        for k in range(n + 1):
            r, c = divmod(pos, T)
            is_eot = k == n
            ax.add_patch(plt.Rectangle((c, -r), 0.95, 0.9, color="black" if is_eot else colors[i], alpha=1 if is_eot else 0.6))
            if is_eot:
                ax.text(c + 0.47, -r + 0.45, "eot", color="white", ha="center", va="center", fontsize=7)
            pos += 1
        r, c = divmod(pos - n - 1, T)
        ax.text(c + 0.05, -r + 0.98, name, fontsize=7, va="bottom")
    ax.set(xlim=(0, T), ylim=(-2.2, 1.4))
    ax.axis("off")
    for r in range(3):
        ax.text(-0.2, -r + 0.45, f"window {r}", ha="right", va="center", fontsize=8)
    ax.set_title("Packing: documents laid end to end, separated by <|endoftext|>, cut into fixed windows (T = 12 here)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "packing_eot.png", dpi=130)


def sampler_coverage():
    rng = np.random.default_rng(0)
    N = 100_000                         # windows in the file (toy size; the fraction is size-independent)
    epochs = np.linspace(0, 3, 61)
    seen_random = []
    hits = np.zeros(N, dtype=bool)
    drawn = 0
    for e in epochs:
        target = int(e * N)
        if target > drawn:
            hits[rng.integers(0, N, target - drawn)] = True
            drawn = target
        seen_random.append(hits.mean())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, seen_random, label="random offsets (simulated)", color="C3")
    ax.plot(epochs, np.minimum(epochs, 1), label="shuffled windows", color="C0")
    ax.plot(epochs, 1 - np.exp(-epochs), ":", color="gray", label="1 − e^(−epochs)")
    ax.axvline(1, color="gray", lw=0.5)
    ax.annotate(f"{1 - seen_random[20]:.0%} never seen\nafter one epoch's draws", (1, seen_random[20]),
                textcoords="offset points", xytext=(15, -40), arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="windows drawn, in epochs", ylabel="fraction of windows seen at least once",
           title="Random offsets leave 37% unseen after one epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sampler_coverage.png", dpi=130)


if __name__ == "__main__":
    onehot_lookup()
    rope_rotation()
    packing_eot()
    sampler_coverage()
    print("saved 4 diagrams")
