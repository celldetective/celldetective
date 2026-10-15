# Writing tests for celldetective

Conventions of this suite, and the mistakes that made earlier tests flaky or
worthless. Before writing, open the existing test file for the module (or the
nearest neighbour) and match its style: plain pytest functions or `Test*`
classes grouped by behaviour, a docstring saying what behaviour is checked and,
for regression tests, which bug it prevents.

## Contents
- What a good test here looks like
- Unit tests
- Public-API tests
- GUI tests (pytest-qt)
- Running them

## What a good test here looks like

- Named after the behaviour, not the function: `test_a_bare_prefix_falls_back_to_the_folder_name`,
  not `test_get_population_2`.
- Expected values worked out independently (by hand, from a tiny input) and
  written as literals. Recomputing the expected value with the code under test
  checks nothing.
- One behaviour per test; several asserts on that behaviour are fine.
- Several inputs for one behaviour → `@pytest.mark.parametrize` with readable `ids`.
- No sleeps, no network, no reliance on test order, no writing inside the repo
  (`tmp_path` for everything written).
- Smoke tests ("does not raise") are acceptable only where not raising *is* the
  behaviour; say so in the docstring.

## Unit tests

Location: `tests/test_<module>.py`, headless (no Qt import at module level, so
they run in the headless CI job).

Build inputs in the test: a 2D label image with two squares, a short
DataFrame with `TRACK_ID`, `FRAME`, `POSITION_X/Y` columns, a 1D signal with a
known step. Small enough that the expected output is obvious. Test edge cases
the pipeline actually meets: empty label images, a single object, NaNs in a
track, objects touching the border, a time series shorter than a window,
anisotropic or 3D/multichannel inputs where the function claims to support them.

Use `np.testing.assert_allclose` / `pd.testing.assert_frame_equal` for numbers
and frames; `pytest.raises(..., match=...)` for errors.

## Public-API tests

These run a pipeline step the way a user or the `processes/` scripts do, on a
position folder, and check what it writes. Location: `tests/test_<step>.py`.

- Data: `tests/ExperimentTest` (wells `W1`–`W10`, `config.ini`, `configs/`), or
  the small `tests/assets/sample*.tif`. Never run a step on the checked-in
  folder: copy the well/position you need into `tmp_path`
  (`shutil.copytree`) and point the step at the copy. `conftest.py` provides
  `write_stacks` for creating empty stack files in a fake experiment.
- Assert on outputs a downstream step relies on: table columns and dtypes,
  row counts, label stack shape and dtype, that a failing step raises or logs
  instead of writing a partial file, the provenance log entry.
- Steps that launch subprocesses: coverage measures them (run_coverage.py sets
  `patch = subprocess`), but a crash in the child must show up as a failure in
  the parent. Assert on the return code / exception, not only on files existing.
- Deep-learning models (StarDist, Cellpose, signal models) are slow and may need
  downloads. Prefer testing the code around inference with a tiny or mocked
  model; mark genuinely slow tests and keep them out of the default path only if
  the suite already has a convention for that.

## GUI tests (pytest-qt)

Location: `tests/gui/test_<widget>.py`. Each file runs in its own process
(`tests/run_gui_tests.py`) because Qt objects deleted late crash later tests;
a file must therefore pass on its own, with no fixture or state from another file.

Test the wiring, not the maths: that a widget fills from the experiment config,
that pressing Apply calls the processing function with the arguments the user
set (patch the function and assert on the call), that an error surfaces as a
message instead of an exception, that settings survive save and reload.

Conventions:
- `qtbot.addWidget(widget)` for every top-level widget you create, so it is
  closed at teardown.
- Most GUI files carry an autouse `disable_logging` fixture
  (`logging.disable(logging.CRITICAL)`) because pytest's capture plus logging
  raises `OSError` on Windows; copy it (or move it to `tests/gui/conftest.py`
  if you are consolidating anyway).
- Never let a modal dialog open: patch `QMessageBox.exec_`/`exec`,
  `QMessageBox.warning/information/question`, `QFileDialog.getSaveFileName`,
  `getOpenFileName`, `getExistingDirectory` with `monkeypatch.setattr`, returning
  what the user would choose. A dialog that opens blocks CI until timeout.
- Heavy windows (annotators, the control panel) take a full experiment to build.
  The suite's pattern for testing one method is
  `patch.object(Cls, "__init__", lambda self, parent: None)` and then setting
  only the attributes the method uses; prefer a real instance when the
  behaviour depends on how `__init__` connects signals.
- Drive the UI as a user would: `qtbot.mouseClick(button, Qt.LeftButton)`,
  `qtbot.keyClicks(line_edit, "...")`, `combo.setCurrentText(...)`. Calling the
  slot directly skips the signal connection, which is the thing most likely to
  be broken.
- Waiting: `qtbot.waitUntil(lambda: cond, timeout=...)` or
  `qtbot.waitSignal(thread.finished)`; not `QTest.qWait(n)` / `time.sleep`.
  Threads (`QThread` workers) must be finished or stopped before the test ends.
- Napari viewers: create with `show=False` when the code allows it and close the
  viewer explicitly; leftover viewers are the most common cause of late crashes.
- Don't add `if __name__ == "__main__":` blocks.

## Running them

```bash
$PY -m pytest tests/test_x.py -q                      # headless file
$PY -m pytest tests/gui/test_x.py -q                   # one GUI file
$PY tests/run_gui_tests.py                             # all GUI files, isolated
```

Run a new GUI file alone *and* through the runner before reporting it as done.
