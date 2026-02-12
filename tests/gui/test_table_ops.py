"""
Unit tests for table operation widgets in celldetective.gui.table_ops.

Covers:
- DifferentiateColWidget (_maths.py)
- OperationOnColsWidget (_maths.py)
- CalibrateColWidget, AbsColWidget, LogColWidget (_maths.py)
- MergeGroupWidget (_merge_groups.py)
- RenameColWidget (_rename_col.py)
"""

import logging
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QMainWindow, QTableView

from celldetective.gui.table_ops._maths import (
    DifferentiateColWidget,
    OperationOnColsWidget,
    CalibrateColWidget,
    AbsColWidget,
    LogColWidget,
)
from celldetective.gui.table_ops._merge_groups import MergeGroupWidget
from celldetective.gui.table_ops._rename_col import RenameColWidget


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def mock_table_parent():
    """Mock parent window with a DataFrame and table view, as expected by table ops."""
    parent = MagicMock(spec=QMainWindow)
    parent.data = pd.DataFrame(
        {
            "TRACK_ID": [1, 1, 1, 2, 2, 2],
            "FRAME": [0, 1, 2, 0, 1, 2],
            "area": [10.0, 12.0, 15.0, 20.0, 22.0, 25.0],
            "intensity": [100.0, 110.0, 105.0, 200.0, 210.0, 205.0],
            "group_A": [0, 0, 0, 1, 1, 1],
            "group_B": [1, 1, 1, 0, 0, 0],
        }
    )
    parent.table_view = MagicMock(spec=QTableView)
    parent.population = "cells"
    return parent


# =============================================================================
# DifferentiateColWidget Tests
# =============================================================================


class TestDifferentiateColWidget:
    """Tests for DifferentiateColWidget."""

    def test_initialization(self, qtbot, mock_table_parent):
        """Test widget initializes with columns from parent data."""
        widget = DifferentiateColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        assert widget.windowTitle() == "d/dt"
        # Measurements combo should contain data columns
        items = [
            widget.measurements_cb.itemText(i)
            for i in range(widget.measurements_cb.count())
        ]
        assert "area" in items
        assert "intensity" in items

    def test_compute_derivative(self, qtbot, mock_table_parent):
        """Test computing derivative adds new column."""
        widget = DifferentiateColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        # Set forward mode
        widget.forward_btn.setChecked(True)

        widget.compute_derivative_and_add_new_column()

        # The parent data should now have a derivative column
        assert (
            any(
                "area" in col and "derivative" in col.lower() or "d(" in col.lower()
                for col in mock_table_parent.data.columns
            )
            or len(mock_table_parent.data.columns) > 6
        )


# =============================================================================
# OperationOnColsWidget Tests
# =============================================================================


class TestOperationOnColsWidget:
    """Tests for OperationOnColsWidget."""

    def test_initialization_divide(self, qtbot, mock_table_parent):
        """Test initialization with divide operation."""
        widget = OperationOnColsWidget(
            parent_window=mock_table_parent,
            column1="area",
            column2="intensity",
            operation="divide",
        )
        qtbot.addWidget(widget)

        assert widget.operation == "divide"

    def test_compute_divide(self, qtbot, mock_table_parent):
        """Test divide operation creates new column."""
        widget = OperationOnColsWidget(
            parent_window=mock_table_parent,
            column1="area",
            column2="intensity",
            operation="divide",
        )
        qtbot.addWidget(widget)

        widget.compute()

        assert "area/intensity" in mock_table_parent.data.columns
        # Check values
        expected = mock_table_parent.data["area"] / mock_table_parent.data["intensity"]
        np.testing.assert_array_almost_equal(
            mock_table_parent.data["area/intensity"].values, expected.values
        )

    def test_compute_multiply(self, qtbot, mock_table_parent):
        """Test multiply operation."""
        widget = OperationOnColsWidget(
            parent_window=mock_table_parent,
            column1="area",
            column2="intensity",
            operation="multiply",
        )
        qtbot.addWidget(widget)

        widget.compute()

        assert "area*intensity" in mock_table_parent.data.columns

    def test_compute_add(self, qtbot, mock_table_parent):
        """Test add operation."""
        widget = OperationOnColsWidget(
            parent_window=mock_table_parent,
            column1="area",
            column2="intensity",
            operation="add",
        )
        qtbot.addWidget(widget)

        widget.compute()

        assert "area+intensity" in mock_table_parent.data.columns

    def test_compute_subtract(self, qtbot, mock_table_parent):
        """Test subtract operation."""
        widget = OperationOnColsWidget(
            parent_window=mock_table_parent,
            column1="area",
            column2="intensity",
            operation="subtract",
        )
        qtbot.addWidget(widget)

        widget.compute()

        assert "area-intensity" in mock_table_parent.data.columns


# =============================================================================
# CalibrateColWidget Tests
# =============================================================================


class TestCalibrateColWidget:
    """Tests for CalibrateColWidget."""

    def test_initialization(self, qtbot, mock_table_parent):
        """Test initialization."""
        widget = CalibrateColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        assert widget.windowTitle() == "Calibrate data"

    def test_compute_calibration(self, qtbot, mock_table_parent):
        """Test calibration multiplies column by factor."""
        widget = CalibrateColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        widget.calibration_factor_le.setText("0.5")
        widget.units_le.setText("um")
        widget.compute()

        assert "area[um]" in mock_table_parent.data.columns
        expected = mock_table_parent.data["area"] * 0.5
        np.testing.assert_array_almost_equal(
            mock_table_parent.data["area[um]"].values, expected.values
        )


# =============================================================================
# AbsColWidget Tests
# =============================================================================


class TestAbsColWidget:
    """Tests for AbsColWidget."""

    def test_compute_abs(self, qtbot, mock_table_parent):
        """Test absolute value computation."""
        # Add negative values
        mock_table_parent.data["signed_col"] = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]

        widget = AbsColWidget(parent_window=mock_table_parent, column="signed_col")
        qtbot.addWidget(widget)

        widget.compute()

        assert "|signed_col|" in mock_table_parent.data.columns
        np.testing.assert_array_equal(
            mock_table_parent.data["|signed_col|"].values,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )


# =============================================================================
# LogColWidget Tests
# =============================================================================


class TestLogColWidget:
    """Tests for LogColWidget."""

    def test_compute_log(self, qtbot, mock_table_parent):
        """Test log10 computation."""
        mock_table_parent.data["positive_col"] = [1.0, 10.0, 100.0, 1000.0, 10.0, 1.0]

        widget = LogColWidget(parent_window=mock_table_parent, column="positive_col")
        qtbot.addWidget(widget)

        widget.compute()

        assert "log10(positive_col)" in mock_table_parent.data.columns
        expected = np.log10([1.0, 10.0, 100.0, 1000.0, 10.0, 1.0])
        np.testing.assert_array_almost_equal(
            mock_table_parent.data["log10(positive_col)"].values, expected
        )


# =============================================================================
# MergeGroupWidget Tests
# =============================================================================


class TestMergeGroupWidget:
    """Tests for MergeGroupWidget."""

    def test_initialization(self, qtbot, mock_table_parent):
        """Test widget initializes with group columns."""
        widget = MergeGroupWidget(
            parent_window=mock_table_parent,
            columns=["group_A", "group_B"],
        )
        qtbot.addWidget(widget)

        assert widget.windowTitle() == "Merge classifications"
        assert len(widget.cbs) >= 2

    def test_compute_merge(self, qtbot, mock_table_parent):
        """Test merging two group columns."""
        widget = MergeGroupWidget(
            parent_window=mock_table_parent,
            columns=["group_A", "group_B"],
            n_cols_init=2,
        )
        qtbot.addWidget(widget)

        widget.name_le.setText("group_merged")
        widget.compute()

        assert "group_merged" in mock_table_parent.data.columns

    def test_add_col(self, qtbot, mock_table_parent):
        """Test adding a column selector."""
        widget = MergeGroupWidget(
            parent_window=mock_table_parent,
            columns=["group_A"],
            n_cols_init=1,
        )
        qtbot.addWidget(widget)

        initial_count = len(widget.cbs)
        widget.add_col()
        assert len(widget.cbs) == initial_count + 1


# =============================================================================
# RenameColWidget Tests
# =============================================================================


class TestRenameColWidget:
    """Tests for RenameColWidget."""

    def test_initialization(self, qtbot, mock_table_parent):
        """Test widget initializes with column name."""
        widget = RenameColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        assert widget.windowTitle() == "Rename column"
        assert widget.new_col_name.text() == "area"

    def test_rename_column(self, qtbot, mock_table_parent):
        """Test renaming a column updates parent data."""
        widget = RenameColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        widget.new_col_name.setText("cell_area")
        widget.rename_col()

        assert "cell_area" in mock_table_parent.data.columns
        assert "area" not in mock_table_parent.data.columns

    def test_rename_preserves_data(self, qtbot, mock_table_parent):
        """Test renaming preserves column data."""
        original_values = mock_table_parent.data["area"].values.copy()

        widget = RenameColWidget(parent_window=mock_table_parent, column="area")
        qtbot.addWidget(widget)

        widget.new_col_name.setText("cell_area")
        widget.rename_col()

        np.testing.assert_array_equal(
            mock_table_parent.data["cell_area"].values, original_values
        )
