"""
Check that a test cleanup did not lose coverage.

Compares two coverage data files line by line. The total percentage can stay
flat while specific lines stop being run (a deleted test reached them, a new one
reaches others), so this lists every line covered BEFORE and not AFTER.

Usage
-----
    python coverage_diff.py BEFORE AFTER

Exits 1 if any line was lost, 0 otherwise. Run from the repository root.
"""

import sys
from collections import defaultdict
from pathlib import Path

from coverage import CoverageData


def covered(datafile):
    data = CoverageData(basename=datafile)
    data.read()
    root = Path.cwd()
    out = {}
    for f in data.measured_files():
        try:
            rel = Path(f).resolve().relative_to(root).as_posix()
        except ValueError:
            rel = f
        out[rel] = set(data.lines(f) or ())
    return out


def ranges(lines):
    lines = sorted(lines)
    spans, start = [], lines[0]
    for a, b in zip(lines, lines[1:] + [None]):
        if b != a + 1:
            spans.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(spans)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    before, after = covered(sys.argv[1]), covered(sys.argv[2])
    lost, gained = defaultdict(set), defaultdict(set)
    for f in set(before) | set(after):
        b, a = before.get(f, set()), after.get(f, set())
        if b - a:
            lost[f] = b - a
        if a - b:
            gained[f] = a - b
    nb, na = sum(map(len, before.values())), sum(map(len, after.values()))
    print(f"Covered lines: {nb} -> {na} ({na - nb:+d})")
    print(f"Lost: {sum(map(len, lost.values()))} lines in {len(lost)} files; "
          f"gained: {sum(map(len, gained.values()))} lines in {len(gained)} files\n")
    for f in sorted(lost):
        print(f"LOST   {f}: {ranges(lost[f])}")
    for f in sorted(gained):
        print(f"gained {f}: {ranges(gained[f])}")
    sys.exit(1 if lost else 0)


if __name__ == "__main__":
    main()
