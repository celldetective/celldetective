# test_notebooks.py
import nbformat
from nbclient import NotebookClient
from pathlib import Path


def test_notebook_runs():
    notebook_path = Path(__file__).parent.parent / "demos" / "ADCC_analysis_demo.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # raises exception if any cell fails
