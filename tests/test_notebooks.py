# test_notebooks.py
import nbformat
from nbclient import NotebookClient


def test_notebook_runs():
    nb = nbformat.read("../demos/ADCC_analysis_demo.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()  # raises exception if any cell fails
