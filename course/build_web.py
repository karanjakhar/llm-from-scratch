"""Build the course's web page: one static page, served at karanjakhar.dev/course/.

    uv run python course/build_web.py                                    # -> course/_build/site/
    uv run python course/build_web.py --out ../personal_website/public/course

Same machinery and look as the book (book/build_web.py): the module list comes
from the table in docs/COURSE.md §4, a module appears once its lesson file
course/modules/NN_*.md exists, and the rest show greyed out. Links to a stage
doc go to that chapter of the web book, not to GitHub.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "book"))
import build_web as book  # noqa: E402

# Lessons fold their answers in <details>, so raw HTML must pass through here.
book.md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
MARKER = "<!-- built by course/build_web.py -->"
BOOK_URL = "/book/"
STARTER = "https://github.com/karanjakhar/llm-from-scratch-course"


def syllabus() -> list[dict]:
    """Rows of the core-path table in docs/COURSE.md §4."""
    text = (ROOT / "docs/COURSE.md").read_text()
    table = text.split("## 4. The core path", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0].isdigit():
            num = int(cells[0])
            lesson = sorted((ROOT / "course/modules").glob(f"{num:02d}_*.md"))
            rows.append(dict(num=num, title=cells[1], stage=cells[2], lesson=lesson[0] if lesson else None))
    return rows


def book_links(out: str) -> str:
    """A link to a stage doc on GitHub becomes a link to its chapter in the web book."""
    chapters = {s: r["num"] for r in book.chapter_map() for s in r["sources"]}
    def repl(m: re.Match) -> str:
        num = chapters.get(m.group(1))
        return f'href="{BOOK_URL}#ch-{book.slug(num)}"' if num else m.group(0)
    return re.sub(r'href="' + re.escape(book.GITHUB) + r'(docs/[^"#]+\.md)"', repl, out)


def template() -> str:
    """The book's template with the book's words swapped for the course's."""
    page = book.TEMPLATE.read_text()
    swaps = {
        book.MARKER: MARKER,
        "<title>From Bytes to Code · Karan Jakhar</title>": "<title>Build an LLM: the course · Karan Jakhar</title>",
        'content="A book in progress: building a language model from scratch on one laptop GPU. Every part built '
        'by hand, measured, and broken on purpose."':
            'content="A hands-on course: write every part of a small GPT yourself, checked by tests, with a '
            'number to compare at the end of each module."',
        '<meta property="og:type" content="book">': '<meta property="og:type" content="website">',
        '<meta property="og:title" content="From Bytes to Code">':
            '<meta property="og:title" content="Build an LLM: the course">',
        'content="Building a language model from scratch on one laptop GPU. A book in progress."':
            'content="Write every part of a small GPT yourself, checked by tests."',
        "<h2>From Bytes to Code</h2>": "<h2>Build an LLM</h2>",
        "<p>Building a language model from scratch on one laptop GPU. A book in progress; working title.</p>":
            f'<p>The hands-on course for <a href="{BOOK_URL}">From Bytes to Code</a>. You write the code; '
            f"the tests check it.</p>",
        "{{DRAFTED}} of {{TOTAL}} parts drafted · {{WORDS}} words":
            "{{DRAFTED}} of {{TOTAL}} core modules ready",
        # Scoped to main: the contents rail is a <details> too (the book's phone layout).
        "</style>": "main details{margin:.6em 0;padding:.5em .8em;border:1px solid var(--line, #ddd);border-radius:6px}"
                    "main summary{cursor:pointer;font-weight:600}\n</style>",
    }
    for a, b in swaps.items():
        assert a in page, f"book template changed; update course/build_web.py: {a[:50]}"
        page = page.replace(a, b)
    return page


def build(out_dir: Path, site_url: str) -> None:
    rows = syllabus()
    toc = ['<li class="draft"><a href="#start"><span class="n">·</span>Start here</a></li>']
    start = (ROOT / "course/STARTER_README.md").read_text()
    intro_md = start.split("## Modules", 1)[0]                       # the module table is the rail itself
    tmp = ROOT / "course/_build/start.md"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(intro_md)
    body = [book.section("start", "The course", "Start here", None, [tmp.relative_to(ROOT).as_posix()])[0]]

    for r in rows:
        cid = f"m-{r['num']}"
        if r["lesson"]:
            h, _ = book.section(cid, f"Module {r['num']} · stage {r['stage']}", r["title"], None,
                                [r["lesson"].relative_to(ROOT).as_posix()])
            body.append(h)
            toc.append(f'<li class="draft"><a href="#{cid}"><span class="n">{r["num"]}</span>'
                       f'{html.escape(r["title"])}</a></li>')
        else:
            toc.append(f'<li class="pending" title="coming"><span class="n">{r["num"]}</span>'
                       f'{html.escape(r["title"])}</li>')

    ready = sum(1 for r in rows if r["lesson"])
    page = template()
    for key, val in {"TOC": "\n".join(toc), "BODY": book_links("\n".join(body)), "DRAFTED": str(ready),
                     "TOTAL": str(len(rows)), "URL": site_url}.items():
        page = page.replace("{{" + key + "}}", val)
    assert "{{" not in page, "unfilled template slot (or a {{PLACEHOLDER}} left in a lesson)"

    if out_dir.exists():
        old = out_dir / "index.html"
        if any(out_dir.iterdir()) and not (old.exists() and MARKER in old.read_text()):
            raise SystemExit(f"{out_dir} is not a previous course build; refusing to overwrite it")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    print(f"{out_dir}: {ready}/{len(rows)} core modules ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "course/_build/site"))
    ap.add_argument("--url", default="https://karanjakhar.dev/course/")
    a = ap.parse_args()
    build(Path(a.out).resolve(), a.url)
