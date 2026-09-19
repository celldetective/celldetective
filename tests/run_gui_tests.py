"""
Run the GUI tests one file per process.

Qt state accumulates across tests in a way no fixture fully undoes: widgets are
closed but deleted later, their children's queued events outlive them, and
`QTest.qWait` / `processEvents` in a later test is what finally dispatches the
backlog. When that dispatch reaches an object Qt has already freed, the result
is a `Windows fatal exception: access violation` -- the interpreter dies with no
traceback and no failing assertion, taking the whole run's results with it,
attributed to whichever test happened to be on the stack.

The individual hazards are worth fixing and are being fixed. The accumulation is
what turns any one of them into a dead process, and a process boundary per file
removes it: each file starts with an empty event queue and no leftover widgets,
and a file that does crash reports as that file failing instead of truncating
everything after it.

Usage
-----
    python tests/run_gui_tests.py [pytest args...]

Exits non-zero if any file fails, after running all of them, so one bad file
does not hide the state of the rest.
"""

import os
import subprocess
import sys
from pathlib import Path

GUI_TESTS = Path(__file__).parent / "gui"

VERBOSITY_FLAGS = ("-v", "-vv", "-q", "-qq", "--verbose", "--quiet")


def main(argv):
    # `rglob`, not `glob`: this replaced `pytest tests/gui/`, which collected
    # subdirectories too. A plain `glob` would drop e.g. `table_ops/` silently,
    # leaving those tests unrun on every job with nothing in the log to say so.
    files = sorted(p for p in GUI_TESTS.rglob("test_*.py"))
    if not files:
        print(f"No GUI test files found under {GUI_TESTS}", file=sys.stderr)
        return 1

    # A native crash kills the child before pytest can write its summary, so the
    # only record of where it happened is whatever already reached the log.
    # Default to one line per test and turn off block buffering on the child's
    # stdout, so the last line printed names the test that died instead of
    # leaving a file-sized haystack.
    if not any(a in VERBOSITY_FLAGS for a in argv):
        argv = ["-v", *argv]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    failed = []
    for path in files:
        rel = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
        print(f"\n=== {rel} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), *argv],
            check=False,
            env=env,
        )
        # A native crash shows up as a negative code (signal) or as Windows'
        # 0xC0000005; either way the file did not pass and the message needs to
        # say so rather than let the run look clean.
        if result.returncode != 0:
            failed.append((rel, result.returncode))

    print("\n" + "=" * 70)
    if failed:
        print(f"{len(failed)} of {len(files)} GUI test files failed:")
        for rel, code in failed:
            print(f"  {rel} (exit {code})")
        return 1

    print(f"All {len(files)} GUI test files passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
