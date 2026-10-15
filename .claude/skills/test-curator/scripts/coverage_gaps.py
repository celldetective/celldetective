"""
Rank the untested code of celldetective by how much it matters.

Reads a coverage data file (from run_coverage.py) and reports, per function and
method, how many of its statements no test executed. Every .py file of the
package is included, imported by the tests or not, so modules no test touches
show up at 0% instead of disappearing from the report.

Each gap gets a level hint from its location (`gui` for celldetective/gui and
celldetective/napari, `api` for the processes/scripts entry points and the
public pipeline modules, `unit` otherwise) and a churn count: the commits that
touched its file in the last --since window. Recently changed, untested code is
where regressions come from, so the ranking is missed statements weighted by
churn.

Usage
-----
    python coverage_gaps.py DATAFILE [--top N] [--since "12 months"]
                            [--path celldetective/measure.py ...] [--level unit|api|gui]
                            [--json OUT]

Run from the repository root.
"""

import argparse
import ast
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

from coverage import Coverage

API_MODULES = {
    "segmentation", "tracking", "measure", "signals", "events", "neighborhood",
    "relative_measurements", "preprocessing", "filters", "extra_properties",
}


def level_of(rel):
    parts = Path(rel).parts
    if len(parts) > 1 and parts[1] in ("gui", "napari"):
        return "gui"
    if len(parts) > 1 and parts[1] in ("processes", "scripts"):
        return "api"
    if Path(rel).stem in API_MODULES and len(parts) == 2:
        return "api"
    return "unit"


def churn(since):
    out = subprocess.run(
        ["git", "log", f"--since={since}", "--format=", "--name-only", "--", "celldetective"],
        capture_output=True, text=True,
    ).stdout
    return Counter(l.strip() for l in out.splitlines() if l.strip().endswith(".py"))


def functions(tree):
    """Yield (qualname, first_line, last_line) for every def, innermost last."""
    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef):
                    start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                    yield name, start, child.end_lineno
                yield from walk(child, name + ".")
            else:
                yield from walk(child, prefix)
    yield from walk(tree, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datafile")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--since", default="12 months")
    ap.add_argument("--path", nargs="*", help="limit to these files/directories")
    ap.add_argument("--level", choices=["unit", "api", "gui"],
                    help="rank only gaps at this level (the per-level table still shows all)")
    ap.add_argument("--json")
    args = ap.parse_args()

    root = Path.cwd()
    cov = Coverage(data_file=args.datafile, config_file=False)
    cov.load()
    commits = churn(args.since)

    files = sorted(p for p in (root / "celldetective").rglob("*.py")
                   if "tests" not in p.relative_to(root).parts[1:2])
    if args.path:
        wanted = [Path(p).resolve() for p in args.path]
        files = [f for f in files if any(f == w or w in f.parents for w in wanted)]

    rows, modules = [], []
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            _, statements, _, missing, _ = cov.analysis2(str(path))
        except Exception as exc:  # unparsable or not Python source
            print(f"skip {rel}: {exc}", file=sys.stderr)
            continue
        if not statements:
            continue
        statements, missing = set(statements), set(missing)
        modules.append({
            "file": rel, "statements": len(statements), "missed": len(missing),
            "pct": round(100 * (1 - len(missing) / len(statements)), 1),
            "churn": commits.get(rel, 0), "level": level_of(rel),
        })
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        # Attribute each statement to its innermost enclosing function.
        owner = {}
        for name, start, end in sorted(functions(tree), key=lambda f: f[2] - f[1], reverse=True):
            for line in range(start, end + 1):
                owner[line] = (name, start)
        per_fn = {}
        for line in statements:
            if line in owner:
                d = per_fn.setdefault(owner[line], [0, []])
                d[0] += 1
                if line in missing:
                    d[1].append(line)
        for (name, start), (n, missed) in per_fn.items():
            if not missed:
                continue
            c = commits.get(rel, 0)
            rows.append({
                "file": rel, "function": name, "line": start,
                "statements": n, "missed": len(missed),
                "pct": round(100 * (1 - len(missed) / n), 1),
                "never_run": len(missed) == n,
                "churn": c, "level": level_of(rel),
                "score": round(len(missed) * (1 + math.log1p(c)), 1),
                "missed_lines": sorted(missed),
            })

    if args.level:
        rows = [r for r in rows if r["level"] == args.level]
    rows.sort(key=lambda r: r["score"], reverse=True)
    modules.sort(key=lambda m: m["missed"] * (1 + math.log1p(m["churn"])), reverse=True)

    tot_s = sum(m["statements"] for m in modules)
    tot_m = sum(m["missed"] for m in modules)
    print(f"# Coverage gaps\n\nTotal: {100 * (1 - tot_m / max(tot_s, 1)):.1f}% "
          f"({tot_s - tot_m}/{tot_s} statements), churn window: {args.since}\n")
    by_level = {}
    for m in modules:
        s = by_level.setdefault(m["level"], [0, 0])
        s[0] += m["statements"]
        s[1] += m["missed"]
    print("| level | statements | covered |\n|---|---|---|")
    for lvl, (s, mi) in sorted(by_level.items()):
        print(f"| {lvl} | {s} | {100 * (1 - mi / s):.1f}% |")

    print("\n## Modules (missed statements, weighted by churn)\n")
    print("| file | level | stmts | covered | commits |\n|---|---|---|---|---|")
    for m in [m for m in modules if not args.level or m["level"] == args.level][: args.top]:
        print(f"| {m['file']} | {m['level']} | {m['statements']} | {m['pct']}% | {m['churn']} |")

    print(f"\n## Top {args.top} functions\n")
    print("| score | function | level | missed/stmts | commits | |\n|---|---|---|---|---|---|")
    for r in rows[: args.top]:
        flag = "never run" if r["never_run"] else ""
        print(f"| {r['score']} | {r['file']}:{r['line']} `{r['function']}` | {r['level']} "
              f"| {r['missed']}/{r['statements']} | {r['churn']} | {flag} |")

    if args.json:
        Path(args.json).write_text(json.dumps({"modules": modules, "functions": rows}, indent=1))
        print(f"\nFull data: {args.json}")


if __name__ == "__main__":
    main()
