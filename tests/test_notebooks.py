# test_notebooks.py
import os

import nbformat
import pytest
from nbclient import NotebookClient
from pathlib import Path

DEMOS = Path(__file__).parent.parent / "demos"
SYNTHETIC = DEMOS / "synthetic_parameter_recovery.ipynb"


def test_notebook_runs():
    notebook_path = DEMOS / "ADCC_analysis_demo.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # raises exception if any cell fails


def run_synthetic_demo(monkeypatch, part_one_only):
    # The kernel starts in demos/: make it import the code under test.
    root = str(Path(__file__).parent.parent)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    nb = nbformat.read(SYNTHETIC, as_version=4)
    if part_one_only:
        start = next(i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith("# Part II"))
        nb.cells = nb.cells[:start]
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(DEMOS)}})
    client.execute()  # raises exception if any cell fails


def test_synthetic_parameter_recovery_runs(monkeypatch):
    """Part I of the synthetic demo needs no download: it runs wherever the tests do."""
    run_synthetic_demo(monkeypatch, part_one_only=True)


@pytest.mark.skipif(
    not os.environ.get("CELLDETECTIVE_FULL_NOTEBOOKS"),
    reason="part II simulates seven experiments (about 12 minutes); set CELLDETECTIVE_FULL_NOTEBOOKS=1",
)
def test_synthetic_killing_mechanisms_run(monkeypatch):
    run_synthetic_demo(monkeypatch, part_one_only=False)
