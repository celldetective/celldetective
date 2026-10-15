"""
Measure celldetective's test coverage with per-test contexts.

Runs the headless suite and the GUI suite the way CI does (the GUI tests one
file per process, through tests/run_gui_tests.py), records which test executed
which line, and combines everything into a single data file that the other
scripts read.

Usage
-----
    python run_coverage.py --out DIR [--python EXE] [--skip-gui | --skip-headless]
                           [--tests PATH ...] [--no-subprocess]

--out        directory for the data files and logs (created if missing);
             the combined result is DIR/.coverage
--python     interpreter of the environment that can import celldetective and
             PyQt5 (default: the one running this script)
--tests      run only these test paths instead of the full suites (each path
             goes to the headless run, or to the GUI runner if under tests/gui)
--no-subprocess  do not measure code run in child processes. By default it is
             measured (coverage's `patch = subprocess`): the segment/track/
             measure steps run as subprocesses, and without this they would
             look untested even when a test drives them.

Run from the repository root. Prints the overall percentage and the paths of the
logs; a failing test does not stop the measurement (coverage of a red suite is
still worth reading, but say it was red).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

COVERAGERC = """\
[run]
source = celldetective
omit =
    */celldetective/tests/*
{patch}
[report]
skip_empty = true
"""

HEADLESS_DESELECT = ["--deselect", "tests/test_notebooks.py::test_notebook_runs"]


def run(cmd, log, env):
    print("$", " ".join(str(c) for c in cmd), f"> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-gui", action="store_true")
    ap.add_argument("--skip-headless", action="store_true")
    ap.add_argument("--tests", nargs="*")
    ap.add_argument("--no-subprocess", action="store_true")
    args = ap.parse_args()

    if not Path("tests/run_gui_tests.py").exists():
        sys.exit("Run this from the celldetective repository root.")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(".coverage*"):
        old.unlink()
    rc = out / "coveragerc"
    rc.write_text(
        COVERAGERC.format(patch="" if args.no_subprocess else "patch = subprocess\n")
    )

    cov_args = [
        "--cov",
        f"--cov-config={rc}",
        "--cov-context=test",
        "--cov-report=",
        "-p",
        "no:cacheprovider",
    ]

    headless_paths, gui_paths = ["tests", "--ignore=tests/gui", *HEADLESS_DESELECT], None
    if args.tests:
        gui = [t for t in args.tests if Path(t).as_posix().startswith("tests/gui")]
        rest = [t for t in args.tests if t not in gui]
        headless_paths = rest or None
        gui_paths = gui or None
        args.skip_gui = args.skip_gui or not gui
        args.skip_headless = args.skip_headless or not rest

    status = {}
    parts = []
    if not args.skip_headless:
        data = out / ".coverage.headless"
        env = {**os.environ, "COVERAGE_FILE": str(data)}
        status["headless"] = run(
            [args.python, "-m", "pytest", *headless_paths, *cov_args, "-q"],
            out / "headless.log",
            env,
        )
        parts.append(data)

    if not args.skip_gui:
        data = out / ".coverage.gui"
        env = {**os.environ, "COVERAGE_FILE": str(data)}
        if gui_paths:
            # Same isolation as the runner: one process per file.
            codes = [
                run(
                    [args.python, "-m", "pytest", p, *cov_args, "--cov-append", "-q"],
                    out / f"gui_{Path(p).stem}.log",
                    env,
                )
                for p in gui_paths
            ]
            status["gui"] = max(codes)
        else:
            status["gui"] = run(
                [args.python, "tests/run_gui_tests.py", *cov_args, "--cov-append", "-q"],
                out / "gui.log",
                env,
            )
        parts.append(data)

    parts = [p for p in parts if p.exists()]
    if not parts:
        sys.exit("No coverage data was produced; read the logs in " + str(out))

    combined = out / ".coverage"
    env = {**os.environ, "COVERAGE_FILE": str(combined)}
    subprocess.run(
        [args.python, "-m", "coverage", "combine", "--keep", f"--rcfile={rc}",
         *map(str, parts)],
        env=env,
        check=True,
    )
    total = subprocess.run(
        [args.python, "-m", "coverage", "report", f"--rcfile={rc}", "--format=total"],
        env=env,
        capture_output=True,
        text=True,
    ).stdout.strip()

    print(f"\nCombined data: {combined}")
    print(f"Total line coverage: {total}%")
    for name, code in status.items():
        verdict = "passed" if code == 0 else f"FAILED (exit {code}) - see logs"
        print(f"{name} suite: {verdict}")
    print(f"Logs: {out}")


if __name__ == "__main__":
    main()
