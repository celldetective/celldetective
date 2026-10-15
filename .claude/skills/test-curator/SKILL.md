---
name: test-curator
description: Measure and curate the celldetective test suite. Runs coverage with per-test contexts over the headless and GUI suites, ranks untested code by risk, finds tests that have piled up re-testing the same thing (identical or subsumed coverage, copy-pasted bodies, shadowed or uncollected tests, tests that assert nothing), consolidates them (parametrize, merge, move fixtures to conftest, delete) while proving coverage did not drop, then writes new unit, public-API and GUI (pytest-qt) tests for the riskiest gaps. Use whenever the user asks about test coverage, untested code, what to test next, redundant/duplicate/overlapping/flaky-looking or bloated tests, cleaning up or reorganizing tests, test debt, or wants new tests written for a module, feature or GUI panel, even if they never say "coverage".
---

# Test curator

The suite grows by accretion: each bug fix adds a test file named after the bug,
each feature adds a class, and nobody removes anything. The result is many tests
walking the same code paths while whole modules go unexecuted. This skill
measures that, cuts the duplication without losing what the tests check, and
spends the effort saved on code nobody tests.

Work through the phases autonomously and report at the end. Ask only when a
decision is genuinely the user's (e.g. a test whose intent you cannot work out,
or deleting a whole test file someone may still rely on).

**The governing rule: coverage measures which lines ran, not what was checked.**
Every number in this skill is a pointer to code you then have to read. A test
"subsumed" by another may be the only one asserting the value that once broke;
an untested function may be dead code that should be deleted rather than
tested. Decide from the code, not the report.

## Setup

- Use the `celldetective` conda env: `C:\Users\remy1\.conda\envs\celldetective\python.exe`
  (base Anaconda's PyQt5 DLLs are broken). Call it `$PY` below.
- `pytest-cov` must be installed in it (`$PY -m pip install pytest-cov`); it is
  not in requirements.txt. Coverage ≥ 7.10 is needed to measure subprocesses.
- Scripts live in this skill's `scripts/` folder; run them from the repo root.
  Write data and reports to the session scratchpad (`$SCRATCH` below), never
  into the repo.
- If on `main` or `dev`, create a branch before changing tests (PRs target `dev`).

## Phase 1 — Measure

```bash
$PY .claude/skills/test-curator/scripts/run_coverage.py --out $SCRATCH/cov-before
```

This runs the headless suite like CI (`pytest tests --ignore=tests/gui`, minus
the notebook test) and the GUI suite through `tests/run_gui_tests.py` (one
process per file, otherwise Qt state accumulates and crashes the interpreter),
records which test ran which line (`--cov-context=test`), measures code run in
child processes, and combines everything into `cov-before/.coverage`. A full
run takes a while: start it in the background and do Phase 2's static lint
meanwhile. For a narrow question ("is the tracking module covered?") pass
`--tests tests/test_tracking.py tests/gui/test_settings_tracking.py` instead.

If tests fail, keep going (coverage of a red suite is still informative) but
record which ones and say so in the report. Never "fix" a failing test by
deleting it as redundant.

## Phase 2 — Find the redundancy

Two scripts, two kinds of evidence:

```bash
$PY .claude/skills/test-curator/scripts/test_lint.py tests          # static, no run needed
$PY .claude/skills/test-curator/scripts/test_overlap.py $SCRATCH/cov-before/.coverage
```

`test_lint.py` finds cruft that needs no judgment call about intent:
- **shadowed** tests (same name defined twice in a module or class; the first
  never runs). Rename or merge; check the hidden one still passes.
- **uncollected**: Test classes with `__init__`, test functions in files pytest
  doesn't collect. Either make them run or delete them.
- **no-assert** tests. Some are legitimate smoke tests ("does not raise"),
  which should say so in a docstring or use an explicit check. Otherwise the
  test needs a real assertion or should go.
- **unconditional skips** and `if __name__ == "__main__"` scaffolding.
- **parametrize candidates**: bodies identical up to literal values.
- **fixtures duplicated across files**: move them to the nearest `conftest.py`.

`test_overlap.py` compares the lines each test ran: identical line sets, tests
whose lines are a strict subset of one other test's, near-duplicates by Jaccard
index, tests with no unique lines, and test files ranked by how much only they
reach. Use it to find *where* to look; then for each candidate, read both tests
and classify:

| Situation | Action |
|---|---|
| Same inputs, same assertions (pasted test, re-written later) | Delete the weaker one |
| Same code path, different inputs, same kind of check | Merge into one `@pytest.mark.parametrize` test, one id per old test |
| Same setup, different assertions on the result | Merge into one test or keep a shared fixture, keep every assertion |
| Subset coverage but asserts a specific regression value | **Keep.** Make sure its docstring names the bug |
| GUI test re-checking pure logic an existing unit test checks | Keep the unit test; the GUI test should check the wiring only |
| Two files testing one module (`test_x.py`, `test_x_new.py`, `test_x_bugfix.py`) | Fold into one file per module, organized by behaviour |

When in doubt, keep it. A redundant test costs seconds; deleting the only check
of a past bug costs a regression.

## Phase 3 — Consolidate, then prove nothing was lost

Make the changes, then re-measure the same scope and diff:

```bash
$PY .claude/skills/test-curator/scripts/run_coverage.py --out $SCRATCH/cov-after [same --tests]
$PY .claude/skills/test-curator/scripts/coverage_diff.py $SCRATCH/cov-before/.coverage $SCRATCH/cov-after/.coverage
```

`coverage_diff.py` lists every line covered before and not after; the total
percentage alone hides this. Lost lines must be zero, or each one explained
(e.g. a deleted test was the only thing reaching dead code you also removed).
Lines are the floor, not the proof: also confirm every assertion from a removed
test survives somewhere, by listing them in the report (old test → where its
check lives now). The suite must pass at least as well as before, and the
test count drop should match what you report.

Keep cleanups in their own commits, separate from new tests, so a reviewer can
check "no behaviour lost" without reading new code.

## Phase 4 — Rank the gaps

```bash
$PY .claude/skills/test-curator/scripts/coverage_gaps.py $SCRATCH/cov-after/.coverage --top 40 --json $SCRATCH/gaps.json
```

Per function: missed/total statements, whether it ever ran, the commits that
touched its file in the last year, and a level hint (`unit`, `api`, `gui`). The
score favours recently changed, untested code. Before writing a test for a gap,
read the function and decide:

- **Dead code?** (`grep` for callers, including string-dispatched names in
  `celldetective/processes`, JSON configs and signal/slot connections.) Report
  it as a removal candidate rather than testing it.
- **Error handling for a real failure mode?** Worth a test: the audit history of
  this repo is full of failures that went silent.
- **Trivial getters, `__repr__`, debug branches**: skip; say so.
- **Which level actually exercises the risk?** See below.

## Phase 5 — Write the tests

Pick the lowest level that can catch the bug you are guarding against:

- **Unit**: a function with array/table inputs and a checkable output
  (`celldetective/utils`, `measure`, `filters`, `extra_properties`, `events`,
  signal maths). Synthetic numpy arrays built in the test, exact expected values
  worked out by hand.
- **Public API**: a pipeline step on a whole position folder (segment, track,
  measure, neighborhood, signals; the `processes/` scripts). Copy the needed
  bits of `tests/ExperimentTest` into `tmp_path`, run the step, assert on the
  written files (table columns, label stack shape/dtype, log manifest).
- **GUI**: wiring and user-visible behaviour that only exists in Qt: a widget
  populates from config, a button runs the right thing with the right
  arguments, a dialog error appears instead of a crash, state survives a reopen.

Read `references/writing_tests.md` before writing tests at any level: it has the
conventions of this suite (fixtures, Qt isolation, mocking dialogs, data) and
the mistakes that made earlier tests flaky.

Covering a line is not testing it at the values that break it. While you read
each function, hunt for inputs it mishandles, and test those first: parameters
that end up inside names or strings (a 2 px distance once collided with the
`_2_` population marker that `str.replace` rewrote everywhere), boundary values
(0, 1, exactly the threshold, empty, a single object or frame), NaN and missing
columns, duplicate labels or IDs, populations that share a name. When such a
test fails, you have found a bug: pin it with a strict
`@pytest.mark.xfail(strict=True, reason=...)` test, report it with the fix you
would make, and leave fixing the source to the user unless they asked.

Every new test must be able to fail. For each one, break the code under test
on purpose (flip a condition, return early), run the test, see it fail, revert
the code, and confirm with `git diff celldetective/` that nothing remained.
A test that passes against broken code is noise in the coverage number.

Write tests for the top gaps until diminishing returns (typically 5–15 per
run), put them in the existing file for that module where one exists, run them
(GUI ones through `tests/run_gui_tests.py` or one file at a time), and
re-measure to report the gain. List the remaining ranked gaps as suggestions,
each with the level and the behaviour to check, not just the function name.

## Report

End with a report in this shape (in chat; keep tables short):

1. **Coverage**: before → after, overall and per level (unit/api/gui); suite
   status (pass/fail counts, runtime before/after).
2. **Removed or merged**: one line per change: old test(s) → what happened →
   where its assertions live now. Lost-lines result from `coverage_diff.py`.
3. **Bugs found**: shadowed/uncollected tests that never ran, and any test that
   failed once it did run or once you wrote it (a test catching a real bug is
   the best outcome of this skill; do not weaken it to pass, report it).
4. **New tests**: file, what each checks, the gap it closes, mutation check done.
5. **Next gaps**: the ranked list left over, with suggested level and behaviour.
6. **Dead-code candidates**, if any.

Commit only if asked; if asked, cleanup and new tests are separate commits.
