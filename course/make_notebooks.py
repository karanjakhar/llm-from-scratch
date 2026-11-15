"""Write one Colab notebook per course module into course/colab/. docs/COURSE.md §5.

    uv run python course/make_notebooks.py

Generated, so every module's notebook has the same shape: set up, write the
code, check, run, break. The lesson text lives in course/modules/; the
notebook links to it rather than copying it, so there is one place to fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "course"))
from exercises import MODULES  # noqa: E402

STARTER = "https://github.com/karanjakhar/llm-from-scratch-course"
# Colab ships PyTorch and NumPy. These are the rest of what the modules import; transformers is
# only a reference to compare against (modules 3, 5, 6), never part of a solution.
PIP = "regex tiktoken pytest pyarrow huggingface_hub matplotlib tqdm transformers safetensors"
REPO_DIR = "/content/drive/MyDrive/llm-course"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def notebook(num: int, mod: dict) -> dict:
    lesson = f"{STARTER}/blob/main/course/modules/{mod['lesson']}.md"
    files = sorted({k.split("::")[0] for k in mod["exercises"]})
    names = ", ".join(f"`{k.split('::')[1]}`" for k in mod["exercises"])
    run = mod["experiment"].replace("uv run python", "!python")
    cells = [
        md(f"# Module {num}: {mod['title']}\n\nThe lesson, with the reading list and the numbers to compare "
           f"against, is [here]({lesson}). Keep it open beside this notebook."),
        md("## 1. Set up\n\nMounts your Google Drive and keeps the course there, so your code and the data "
           "you build in module 2 survive when Colab resets the machine. The first run clones the "
           "repository; later runs reuse it. Run this cell at the start of every session. "
           "For modules 3 onward, choose a GPU: Runtime > Change runtime type > T4 GPU."),
        # Clone only once, into Drive: rerunning the cell must not nest a second copy or
        # overwrite the learner's edits, and a Colab reset must not lose them.
        code("from google.colab import drive\n"
             "drive.mount('/content/drive')\n"
             f"![ -d {REPO_DIR} ] || git clone --depth 1 {STARTER} {REPO_DIR}\n"
             f"%cd {REPO_DIR}\n"
             f"!pip -q install {PIP}"),
        md("## 2. Write the code\n\nOpen " + ", ".join(f"`{f}`" for f in files) +
           " from the **Files** panel on the left (under `drive/MyDrive/llm-course/`; double-click opens "
           f"an editor). Write these functions, in order: {names}.\n\n"
           "The files are on your Drive, so your work is saved as you go."),
        md("## 3. Check\n\nRun this after each function. It lists what is still blank and which tests fail."),
        code(f"!python course/check.py {num}"),
        md("## 4. Run the experiment\n\nCompare the numbers with the table in the lesson's step 5."),
        code(run),
        md("## 5. Break it\n\nFollow step 6 of the lesson: change what it says, rerun the cell above, and "
           "compare with the book's damage. Then answer step 7's questions before opening the answers."),
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out = ROOT / "course" / "colab"
    out.mkdir(exist_ok=True)
    for num, mod in MODULES.items():
        p = out / f"{mod['lesson']}.ipynb"
        p.write_text(json.dumps(notebook(num, mod), indent=1) + "\n")
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
