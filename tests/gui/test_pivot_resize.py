import pytest
import pandas as pd
import os
from PyQt5.QtWidgets import QApplication
from celldetective.gui.tableUI import PivotTableUI
from pathlib import Path

# Create dummy DataFrames
small_data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}, index=["row1", "row2"])

large_data = pd.DataFrame(
    {f"col{i}": range(100) for i in range(50)}, index=[f"row{i}" for i in range(100)]
)


def test_pivot_table_resize(qtbot):
    # Test small data
    widget_small = PivotTableUI(small_data, title="Small Pivot Table")
    if qtbot:
        qtbot.addWidget(widget_small)

    # Calculate expected size (roughly)
    # This is hard to predict exactly without rendering, but we can check it's reasonable
    # For small data, it should be smaller than screen size
    screen = QApplication.primaryScreen().availableGeometry()
    assert widget_small.width() < screen.width()
    assert widget_small.height() < screen.height()

    print(f"Small widget size: {widget_small.width()}x{widget_small.height()}")

    # Test large data
    widget_large = PivotTableUI(large_data, title="Large Pivot Table")
    if qtbot:
        qtbot.addWidget(widget_large)

    # For large data, it should be capped at 80% screen size
    max_width = int(screen.width() * 0.8)
    max_height = int(screen.height() * 0.8)

    # Allow small tolerance
    assert widget_large.width() <= max_width
    assert widget_large.height() <= max_height

    # It should be close to max size (content is huge)
    # Checking if it's at least constrained
    print(f"Large widget size: {widget_large.width()}x{widget_large.height()}")
    print(f"Screen cap: {max_width}x{max_height}")


if __name__ == "__main__":
    app = QApplication([])
    test_pivot_table_resize(None)
    print("Test passed!")
