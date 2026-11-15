"""Grade one course module: run its tests, say which functions are still blank.

    uv run python course/check.py 1          # local
    python course/check.py 1                 # Colab

The tests are this project's own acceptance tests (docs/COURSE.md §3, step 4),
unchanged. A module is done when every test passes. Some tests use several
functions at once, so the "still blank" list is the place to start: one
unwritten function can fail many tests.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "course"))
from exercises import MODULES  # noqa: E402
from make_starter import find_function  # noqa: E402


def still_blank(path: str, qualname: str) -> bool:
    """True if the function's body is still the starter's `raise NotImplementedError`."""
    fn = find_function(ast.parse((ROOT / path).read_text()), qualname)
    return any(isinstance(n, ast.Raise) and "NotImplementedError" in ast.unparse(n) for n in fn.body)


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or int(sys.argv[1]) not in MODULES:
        print(f"usage: check.py <module>   modules: {sorted(MODULES)}")
        return 2
    num = int(sys.argv[1])
    mod = MODULES[num]
    print(f"Module {num}: {mod['title']}\n")

    blank = [k.split("::")[1] for k in mod["exercises"] if still_blank(*k.split("::"))]
    for key in mod["exercises"]:
        name = key.split("::")[1]
        print(f"  [{' ' if name in blank else 'x'}] {name}")
    print()

    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                        "--tb=line", *mod["tests"]], cwd=ROOT, capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    failed = [ln for ln in lines if ln.startswith("FAILED") or ln.startswith("ERROR")]
    for ln in failed[:15]:
        print("  " + ln)
    if len(failed) > 15:
        print(f"  ... and {len(failed) - 15} more")
    print("\n  " + (lines[-1] if lines else r.stderr.strip()[-300:]))

    if r.returncode == 0 and not blank:
        print(f"\nModule {num} passes. Next: run the experiment and compare with the book:\n  {mod['experiment']}")
        return 0
    if blank:
        print(f"\nStart with: {blank[0]}  (hint in its body, theory in {mod['chapter']})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
