import sys
import pytest
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication

from celldetective.gui.table_ops._maths import BinColWidget


class MockTableUI:
    def __init__(self, data):
        self.data = pd.DataFrame(data)


def test_bin_gui_state():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    data = {"test_col": [0, 0.9, 1.1, 2, 3], "other_col": [1, 2, 3, 4, 5]}
    parent = MockTableUI(data)

    # Initialize widget with selected column "test_col"
    widget = BinColWidget(parent_window=parent, column="test_col")

    print("Initial valid params check:")
    print(f"Width Valid: {widget.width_le.text()}")
    print(f"Min Valid: '{widget.min_le.text()}'")
    print(f"Max Valid: '{widget.max_le.text()}'")
    print(f"Selected Col in Combo: {widget.measurements_cb.currentText()}")
    print(f"Button Enabled: {widget.submit_btn.isEnabled()}")

    # Let's interact with it
    widget.measurements_cb.setCurrentText("other_col")
    print("After switching to 'other_col':")
    print(f"Min Valid: '{widget.min_le.text()}'")
    print(f"Max Valid: '{widget.max_le.text()}'")
    print(f"Button Enabled: {widget.submit_btn.isEnabled()}")


if __name__ == "__main__":
    test_bin_gui_state()
