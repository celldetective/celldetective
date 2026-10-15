"""
Find tests that execute the same code as other tests.

Reads a coverage data file recorded with per-test contexts (run_coverage.py
does this) and compares the set of lines each test runs. It reports:

- identical: groups of tests that run exactly the same lines
- subsumed: a test whose lines are all run by one other, larger test
- near-duplicates: pairs whose line sets overlap by at least --jaccard
- zero-unique: tests that run no line that some other test does not also run
  (taken together the rest of the suite already reaches all of their code)

These are candidates, not verdicts. Running the same lines is not checking the
same thing: two tests can walk one code path and assert different outcomes, and
a regression test usually runs a few lines that a broad test also runs while
pinning down the one value that once was wrong. Read both tests before merging
or deleting anything.

Only the test body ("|run" contexts) counts; fixture setup and teardown are
shared by design and would make every test that uses a fixture look alike.

Usage
-----
    python test_overlap.py DATAFILE [--min-lines 5] [--jaccard 0.9]
                           [--limit 30] [--json OUT]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from coverage import CoverageData


def load(datafile):
    data = CoverageData(basename=datafile)
    data.read()
    root = Path.cwd()
    lines = defaultdict(set)
    for f in data.measured_files():
        try:
            rel = Path(f).resolve().relative_to(root).as_posix()
        except ValueError:
            rel = f
        for lineno, contexts in data.contexts_by_lineno(f).items():
            for ctx in contexts:
                if ctx.endswith("|run"):
                    lines[ctx[: -len("|run")]].add((rel, lineno))
    return {t: frozenset(s) for t, s in lines.items()}


def fmt_files(lines, n=3):
    files = defaultdict(int)
    for f, _ in lines:
        files[f] += 1
    top = sorted(files.items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{Path(f).name}({c})" for f, c in top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datafile")
    ap.add_argument("--min-lines", type=int, default=5,
                    help="ignore tests running fewer source lines than this")
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--json")
    args = ap.parse_args()

    tests = load(args.datafile)
    if not tests:
        raise SystemExit("No per-test contexts in this data file: record it with "
                         "--cov-context=test (run_coverage.py does).")
    small = sorted(t for t, s in tests.items() if len(s) < args.min_lines)
    tests = {t: s for t, s in tests.items() if len(s) >= args.min_lines}
    names = sorted(tests, key=lambda t: len(tests[t]))

    # Lines run by exactly one test.
    hits = defaultdict(int)
    for s in tests.values():
        for line in s:
            hits[line] += 1
    unique = {t: sum(1 for l in s if hits[l] == 1) for t, s in tests.items()}

    groups = defaultdict(list)
    for t, s in tests.items():
        groups[s].append(t)
    identical = [sorted(g) for g in groups.values() if len(g) > 1]
    in_identical = {t for g in identical for t in g}

    subsumed, near = [], []
    for i, a in enumerate(names):
        sa = tests[a]
        best = None
        for b in names[i + 1:]:
            sb = tests[b]
            if sa == sb:
                continue
            if sa < sb and (best is None or len(sb) < len(tests[best])):
                best = b
            inter = len(sa & sb)
            if inter and inter / len(sa | sb) >= args.jaccard:
                near.append((round(inter / len(sa | sb), 3), a, b))
        if best and a not in in_identical:
            subsumed.append((a, best))

    zero = sorted(t for t, u in unique.items() if u == 0 and t not in in_identical)

    print(f"# Test overlap\n\n{len(tests)} tests compared "
          f"({len(small)} skipped for running < {args.min_lines} source lines).\n")

    print(f"## Identical line sets ({len(identical)} groups)\n")
    for g in sorted(identical, key=len, reverse=True)[: args.limit]:
        print(f"- {len(tests[g[0]])} lines in {fmt_files(tests[g[0]])}")
        for t in g:
            print(f"  - {t}")

    print(f"\n## Subsumed by a single other test ({len(subsumed)})\n")
    for a, b in subsumed[: args.limit]:
        print(f"- {a} ({len(tests[a])} lines)\n  within {b} ({len(tests[b])} lines)")

    print(f"\n## Near-duplicates, Jaccard >= {args.jaccard} ({len(near)})\n")
    for j, a, b in sorted(near, reverse=True)[: args.limit]:
        print(f"- {j}: {a} <-> {b}")

    print(f"\n## No unique lines ({len(zero)} tests)\n")
    print("Each of these runs only lines other tests also run. Expected for focused "
          "regression tests; suspicious for broad ones that also appear above.\n")
    for t in zero[: args.limit * 2]:
        print(f"- {t} ({len(tests[t])} lines)")

    # Per test file: how much of what the file runs only it runs.
    per_file = defaultdict(set)
    for t, s in tests.items():
        per_file[t.split("::")[0]] |= s
    file_hits = defaultdict(int)
    for s in per_file.values():
        for line in s:
            file_hits[line] += 1
    print("\n## Test files by unique contribution (lowest first)\n")
    print("| test file | lines run | only this file |\n|---|---|---|")
    rows = sorted(per_file.items(), key=lambda kv: sum(file_hits[l] == 1 for l in kv[1]))
    for f, s in rows[: args.limit]:
        print(f"| {f} | {len(s)} | {sum(file_hits[l] == 1 for l in s)} |")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "identical": identical,
            "subsumed": [{"test": a, "by": b} for a, b in subsumed],
            "near": [{"jaccard": j, "a": a, "b": b} for j, a, b in near],
            "zero_unique": zero,
            "unique_lines": unique,
            "too_small": small,
        }, indent=1))
        print(f"\nFull data: {args.json}")


if __name__ == "__main__":
    main()
