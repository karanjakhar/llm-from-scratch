"""Check that every course module grades correctly. Run after changing exercises.py.

    uv run python course/verify_starter.py            # all modules
    uv run python course/verify_starter.py 2 6        # some

For each module, in a freshly generated starter:
  1. every other module's files are put back to the solution, so only this
     module is blank; check.py must list all its exercises as blank and fail;
  2. this module's files are put back too; check.py must then pass.
That proves the blanks are real (a test fails without them) and that the
solution satisfies the module's tests inside the starter layout.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "course"))
from exercises import MODULES  # noqa: E402
from make_starter import build  # noqa: E402

OUT = ROOT / "course" / "_build" / "verify"


def files_of(num: int) -> set[str]:
    return {k.split("::")[0] for k in MODULES[num]["exercises"]}


def restore(files: set[str]) -> None:
    for f in files:
        shutil.copy2(ROOT / f, OUT / f)


def check(num: int) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "course/check.py", str(num)], cwd=OUT, capture_output=True, text=True)
    return r.returncode, r.stdout


def main(nums: list[int]) -> int:
    bad = 0
    for num in nums:
        build(OUT)                                   # fresh: every module starts fully blank
        others = set().union(*(files_of(n) for n in MODULES if n != num))
        restore(others - files_of(num))
        code, out = check(num)
        blank = out.count("  [ ] ")
        ok_blank = code == 1 and blank == len(MODULES[num]["exercises"])
        restore(files_of(num))
        code2, out2 = check(num)
        summary = [ln.strip() for ln in out2.splitlines() if " passed" in ln or " failed" in ln]
        ok_full = code2 == 0
        print(f"module {num:2d}: blank -> {blank}/{len(MODULES[num]['exercises'])} listed, "
              f"exit {code} {'ok' if ok_blank else 'WRONG'} | solution -> "
              f"{summary[-1] if summary else '?'} {'ok' if ok_full else 'WRONG'}", flush=True)
        if not ok_full:
            print(out2[-1500:])
        bad += (not ok_blank) + (not ok_full)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]] or sorted(MODULES)))
