import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QApplication
import sys

# Import the widget directly
from celldetective.gui.table_ops._maths import BinColWidget


class MockTableUI:
    def __init__(self, data):
        self.data = pd.DataFrame(data)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_linear_binning(qapp):
    data = {"test_col": [0, 0.9, 1.1, 2, 3]}
    parent = MockTableUI(data)

    # Initialize widget
    widget = BinColWidget(parent_window=parent, column="test_col")

    # Set linear scale and bin width of 1
    widget.scale_linear_btn.setChecked(True)
    widget.width_le.setText("1.0")

    # Execution
    widget.compute()

    # Assert resulting column matches expected output [0, 1, 1, 2, 3]
    result_col = parent.data["test_col_binned_1.0_linear"]
    expected = [0.0, 1.0, 1.0, 2.0, 3.0]

    np.testing.assert_array_almost_equal(result_col.values, expected)


def test_logarithmic_binning(qapp):
    data = {"test_col": [0, 1, 2, 11, 99, 100]}
    parent = MockTableUI(data)

    # Initialize widget
    widget = BinColWidget(parent_window=parent, column="test_col")

    # Set log scale and bin width of 1
    widget.scale_log_btn.setChecked(True)
    widget.width_le.setText("1.0")

    # Execution
    widget.compute()

    # Assert resulting column matches expected output [0, 1, 1, 10, 100, 100]
    result_col = parent.data["test_col_binned_1.0_log"]
    expected = [0.0, 1.0, 1.0, 10.0, 100.0, 100.0]

    np.testing.assert_array_almost_equal(result_col.values, expected)


def test_auto_saturation_bounds(qapp):
    data = {"test_col": [-5, 0, 5, 10, 15]}
    parent = MockTableUI(data)

    # Initialize widget
    widget = BinColWidget(parent_window=parent, column="test_col")

    # Set artificial bounds
    widget.min_le.setText("0.0")
    widget.max_le.setText("10.0")

    # Bin width of 5
    widget.width_le.setText("5.0")

    # Execution
    widget.compute()

    # original data [-5, 0, 5, 10, 15] -> clipped to [0, 0, 5, 10, 10]
    # linear bin 5 -> [0, 0, 5, 10, 10]
    result_col = parent.data["test_col_binned_5.0_linear"]
    expected = [0.0, 0.0, 5.0, 10.0, 10.0]

    np.testing.assert_array_almost_equal(result_col.values, expected)
