"""
Static checks for accumulated cruft in the test suite. No test is run.

Reports:
- shadowed: a test defined twice under the same name in one module or class;
  pytest only collects the last one, the earlier one never runs
- uncollected: files that define test functions under a name pytest does not
  collect (not test_*.py / *_test.py), and Test* classes with an __init__
- no-assert: tests that check nothing (no assert, pytest.raises/warns,
  np/pd testing helpers, self.assert*, qtbot.wait* or mocker/mock assert calls)
- skipped: tests skipped unconditionally (@pytest.mark.skip, or pytest.skip()
  as the first statement)
- main-blocks: `if __name__ == "__main__":` scaffolding left in test files
- copies: test bodies identical up to literal values, across the suite; usually
  the same test pasted with new inputs, i.e. one parametrized test
- fixtures: fixtures with identical bodies defined in several files, which
  belong in a conftest.py

Usage
-----
    python test_lint.py [TESTS_DIR] [--min-stmts 3] [--json OUT]
"""

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ASSERT_CALLS = ("raises", "warns", "deprecated_call", "approx")
# self.assertX, np.testing.assert_x, qtbot.waitSignal, and project helpers that
# assert on the caller's behalf (check_table(...), _verify_labels(...)).
ASSERT_PREFIXES = ("assert", "wait", "check", "verify", "expect")


def is_check(node):
    if isinstance(node, ast.Assert):
        return True
    if isinstance(node, ast.Call):
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        name = name.lstrip("_")
        return name.startswith(ASSERT_PREFIXES) or name in ASSERT_CALLS
    return False


def has_check(fn):
    return any(is_check(n) for n in ast.walk(fn))


def normalized(fn, keep_literals):
    """Hash of a function body, with literals blanked unless keep_literals."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # docstring
    mod = ast.Module(body=body, type_ignores=[])
    if not keep_literals:
        class Blank(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant(value="_"), node)
        mod = Blank().visit(ast.parse(ast.unparse(mod)))
    return hashlib.sha1(ast.dump(mod, annotate_fields=False).encode()).hexdigest(), len(body)


def decorator_names(fn):
    return {ast.unparse(d).split("(")[0] for d in fn.decorator_list}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tests_dir", nargs="?", default="tests")
    ap.add_argument("--min-stmts", type=int, default=3)
    ap.add_argument("--json")
    args = ap.parse_args()

    found = defaultdict(list)
    copies, exact, fixtures = defaultdict(list), defaultdict(list), defaultdict(list)
    n_tests = 0

    for path in sorted(Path(args.tests_dir).rglob("*.py")):
        rel = path.as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            found["unparsable"].append(f"{rel}: {exc}")
            continue
        collected = path.name.startswith("test_") or path.stem.endswith("_test")

        for node in tree.body:
            if collected and isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
                found["main-blocks"].append(f"{rel}:{node.lineno}")

        scopes = [(tree, "")] + [(c, c.name + ".") for c in tree.body if isinstance(c, ast.ClassDef)]
        for scope, prefix in scopes:
            if isinstance(scope, ast.ClassDef) and scope.name.startswith("Test"):
                if any(isinstance(n, ast.FunctionDef) and n.name == "__init__" for n in scope.body):
                    found["uncollected"].append(f"{rel}:{scope.lineno} class {scope.name} has __init__; pytest skips it")
            seen = {}
            for fn in scope.body:
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decos = decorator_names(fn)
                where = f"{rel}:{fn.lineno} {prefix}{fn.name}"
                if any(d.endswith("fixture") for d in decos):
                    h, n = normalized(fn, keep_literals=True)
                    if n >= args.min_stmts:
                        fixtures[h].append(where)
                    continue
                if not fn.name.startswith("test"):
                    continue
                n_tests += 1
                if not collected:
                    found["uncollected"].append(f"{where} (file name is not collected by pytest)")
                if fn.name in seen:
                    found["shadowed"].append(f"{where} redefines line {seen[fn.name]}; the earlier one never runs")
                seen[fn.name] = fn.lineno
                if "pytest.mark.skip" in decos:
                    found["skipped"].append(where)
                elif fn.body and "pytest.skip(" in ast.unparse(fn.body[0]) and not isinstance(fn.body[0], ast.If):
                    found["skipped"].append(where)
                if not has_check(fn):
                    found["no-assert"].append(where)
                h, n = normalized(fn, keep_literals=False)
                if n >= args.min_stmts:
                    copies[h].append(where)
                    exact[normalized(fn, keep_literals=True)[0]].append(where)

    exact_groups = [g for g in exact.values() if len(g) > 1]
    exact_members = {w for g in exact_groups for w in g}
    param_groups = [g for g in copies.values() if len(g) > 1 and not set(g) <= exact_members]
    fixture_groups = [g for g in fixtures.values() if len({w.split(":")[0] for w in g}) > 1]

    print(f"# Test lint\n\n{n_tests} test functions under {args.tests_dir}/\n")
    sections = [
        ("shadowed", "Shadowed tests (never run)"),
        ("uncollected", "Not collected by pytest"),
        ("skipped", "Skipped unconditionally"),
        ("no-assert", "Tests with no check"),
        ("main-blocks", "`__main__` scaffolding in test files"),
        ("unparsable", "Unparsable files"),
    ]
    for key, title in sections:
        items = found.get(key, [])
        print(f"## {title} ({len(items)})\n")
        for i in items:
            print(f"- {i}")
        print()
    print(f"## Exact copies ({len(exact_groups)} groups)\n")
    for g in exact_groups:
        print("- " + "\n  ".join(g))
    print(f"\n## Same body, different literals: parametrize candidates ({len(param_groups)} groups)\n")
    for g in param_groups:
        print("- " + "\n  ".join(g))
    print(f"\n## Fixtures duplicated across files: conftest candidates ({len(fixture_groups)} groups)\n")
    for g in fixture_groups:
        print("- " + "\n  ".join(g))

    if args.json:
        Path(args.json).write_text(json.dumps({
            **found, "exact_copies": exact_groups, "parametrize": param_groups,
            "fixtures": fixture_groups,
        }, indent=1))
        print(f"\nFull data: {args.json}")


if __name__ == "__main__":
    main()
