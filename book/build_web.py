"""Build the web edition of the book: one static page plus its figures.

    uv run book/build_web.py                                  # -> book/_site/
    uv run book/build_web.py --out ../personal_website/public/book

The chapter list, titles and status come from docs/BOOK.md §3, so that table
stays the single source of truth. A chapter appears once its status is
`draft` or later; the rest show greyed out in the contents.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "template.html"
INTRO = "docs/book_intro.md"
GITHUB = "https://github.com/karanjakhar/llm-from-scratch/blob/main/"
WRITTEN = {"draft", "revised", "final"}
MARKER = "<!-- built by book/build_web.py -->"
# Sources that contribute one section rather than the whole file.
SECTIONS = {"PROGRESS.md": "Results log"}

md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def chapter_map() -> list[dict]:
    """Rows of the BOOK.md §3 table: number, title, status, source files."""
    text = (ROOT / "docs/BOOK.md").read_text()
    table = text.split("## 3. Chapter map", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("#", "---"):
            continue
        num, title, _stages, sources, status = cells
        title = re.sub(r"^(Appendix|Preface): ", "", title)          # the eyebrow already says it
        title = title[0].upper() + title[1:]
        rows.append({"num": "Preface" if num == "—" else num, "title": title,
                     "status": status, "sources": re.findall(r"`([^`]+)`", sources)})
    return rows


def source_text(path: str) -> str:
    text = (ROOT / path).read_text()
    if path in SECTIONS:
        head = f"## {SECTIONS[path]}"
        text = text.split(head, 1)[1].split("\n## ", 1)[0]
    return re.sub(r"\A# .*\n", "", text)        # the chapter header replaces the doc's own H1


def render(path: str, prefix: str) -> tuple[str, int]:
    text = source_text(path)
    tokens = md.parse(text)
    for i, t in enumerate(tokens):
        if t.type == "heading_open":
            t.attrSet("id", f"{prefix}-{slug(tokens[i + 1].content)}")
        if t.type != "inline":
            continue
        for c in t.children or []:
            if c.type == "link_open":
                href = c.attrGet("href") or ""
                if re.match(r"[a-z]+:|#", href):
                    continue
                target = (ROOT / Path(path).parent / href).resolve().relative_to(ROOT)
                # Figures ship with the page; everything else lives in the repo.
                if target.parts[:2] == ("docs", "visuals"):
                    c.attrSet("href", "visuals/" + target.name)
                else:
                    c.attrSet("href", GITHUB + target.as_posix())
            if c.type == "image":
                src = c.attrGet("src") or ""
                c.attrSet("src", "visuals/" + Path(src).name)
    out = md.renderer.render(tokens, md.options, {})

    # `stages/04_attention/run.py` in the text becomes a link to that file.
    def link_path(m: re.Match) -> str:
        p = html.unescape(m.group(1)).split(" ")[0]
        if "/" in p and (ROOT / p).exists():
            return f'<a href="{GITHUB}{p}">{m.group(0)}</a>'
        return m.group(0)
    out = re.sub(r"<code>([^<]+)</code>", link_path, out)

    out = out.replace("<table>", '<div class="scroll"><table>').replace("</table>", "</table></div>")
    out = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)" /></p>',
                 r'<figure><img src="\1" alt="\2" loading="lazy"><figcaption>\2</figcaption></figure>', out)
    return out, len(text.split())


def section(cid: str, eyebrow: str, title: str, status: str | None, sources: list[str]) -> tuple[str, int]:
    parts, words = [], 0
    for s in sources:
        h, w = render(s, cid)
        parts.append(h)
        words += w
    meta = ""
    if status:
        links = " · ".join(f'<a href="{GITHUB}{s}">{s}</a>' for s in sources)
        meta = (f'<p class="meta"><span class="pill">{status}</span>'
                f'<span>{words:,} words · about {max(1, round(words / 230))} min</span>'
                f"<span>{links}</span></p>")
    head = (f'<header class="ch-head"><p class="eyebrow">{eyebrow}</p>'
            f"<h1>{html.escape(title)}</h1>{meta}</header>")
    return f'<section class="chapter" id="{cid}">{head}{"<hr class=part>".join(parts)}</section>', words


def build(out_dir: Path, site_url: str) -> None:
    rows = chapter_map()
    toc, body = [], []
    intro, total = section("about", "Before chapter 1", "About this draft", None, [INTRO])
    body.append(intro)
    toc.append('<li class="draft"><a href="#about"><span class="n">·</span>About this draft</a></li>')

    for r in rows:
        cid = "ch-" + slug(r["num"])
        state = "draft" if r["status"] in WRITTEN else "pending"
        label = (r["num"] if r["num"] == "Preface"
                 else f"Appendix {r['num']}" if r["num"].isalpha() else f"Chapter {r['num']}")
        if state == "draft":
            html_, words = section(cid, label, r["title"], r["status"], r["sources"])
            body.append(html_)
            total += words
            toc.append(f'<li class="draft"><a href="#{cid}"><span class="n">{r["num"]}</span>'
                       f'{html.escape(r["title"])}</a></li>')
        else:
            toc.append(f'<li class="pending" title="{r["status"]}"><span class="n">{r["num"]}</span>'
                       f'{html.escape(r["title"])}</li>')

    drafted = sum(r["status"] in WRITTEN for r in rows)
    page = TEMPLATE.read_text()
    for key, val in {"TOC": "\n".join(toc), "BODY": "\n".join(body), "WORDS": f"{total:,}",
                     "DRAFTED": str(drafted), "TOTAL": str(len(rows)), "URL": site_url}.items():
        page = page.replace("{{" + key + "}}", val)
    assert "{{" not in page, "unfilled template slot"

    if out_dir.exists():
        # Only ever wipe a folder this script built, never a mistyped --out.
        old = out_dir / "index.html"
        if any(out_dir.iterdir()) and not (old.exists() and MARKER in old.read_text()):
            raise SystemExit(f"{out_dir} is not a previous book build; refusing to delete it")
        shutil.rmtree(out_dir)
    (out_dir / "visuals").mkdir(parents=True)
    (out_dir / "index.html").write_text(page)
    for f in (ROOT / "docs/visuals").iterdir():
        shutil.copy2(f, out_dir / "visuals" / f.name)
    print(f"{out_dir}: {drafted}/{len(rows)} parts drafted, {total:,} words")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "book/_site")
    ap.add_argument("--url", default="https://karanjakhar.dev/book/", help="canonical URL of the page")
    a = ap.parse_args()
    build(a.out.resolve(), a.url)
