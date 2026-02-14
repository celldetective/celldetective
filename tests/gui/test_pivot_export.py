import pytest
import pandas as pd
import os
from PyQt5.QtWidgets import QApplication, QFileDialog
from celldetective.gui.tableUI import PivotTableUI
from unittest.mock import patch
from pathlib import Path

# Create a dummy DataFrame
data = pd.DataFrame(
    {"col1": [1, 2, 3], "col2": [4, 5, 6]}, index=["row1", "row2", "row3"]
)


def test_pivot_table_export(qtbot, tmp_path):
    # Initialize the widget
    widget = PivotTableUI(data, title="Test Pivot Table")
    if qtbot:
        qtbot.addWidget(widget)

    # Define the output file path
    output_file = tmp_path / "test_export.csv"

    # Mock QFileDialog.getSaveFileName return value
    with patch.object(
        QFileDialog,
        "getSaveFileName",
        return_value=(str(output_file), "CSV Files (*.csv)"),
    ):
        # Trigger the export
        widget.export_data()

    # Verify the file exists
    assert output_file.exists()

    # Verify the content
    exported_df = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(data, exported_df)

    print(f"Exported file content:\n{exported_df}")


if __name__ == "__main__":
    # Manually run the test if executed as a script
    app = QApplication([])
    tmp_path = Path("test_output").absolute()
    tmp_path.mkdir(parents=True, exist_ok=True)

    test_pivot_table_export(None, tmp_path)
    print("Test passed!")
