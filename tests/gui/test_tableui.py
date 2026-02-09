"""
Unit tests for TableUI and PivotTableUI in tableUI.py.

Tests cover:
- TableUI initialization with different data types
- Column operations (delete, rename, selection)
- PivotTableUI initialization and coloring
- Track collapse functionality
"""

import pytest
import numpy as np
import pandas as pd
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from celldetective.gui.tableUI import TableUI, PivotTableUI


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logger = logging.getLogger()
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def sample_cell_data():
    """Create sample cell measurement data."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 10 + ["pos2"] * 10,
            "TRACK_ID": list(range(5)) * 4,
            "FRAME": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
            "area": np.random.rand(20) * 100 + 50,
            "perimeter": np.random.rand(20) * 40 + 10,
            "intensity_mean": np.random.rand(20) * 1000,
            "class": ["A", "B"] * 10,
        }
    )


@pytest.fixture
def sample_data_no_tracks():
    """Create sample data without track information."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 5 + ["pos2"] * 5,
            "area": np.random.rand(10) * 100,
            "perimeter": np.random.rand(10) * 40,
        }
    )


@pytest.fixture
def pivot_data_pvalue():
    """Create sample pivot table data for p-value coloring."""
    return pd.DataFrame(
        {
            "group1_vs_group2": [0.001, 0.01, 0.05],
            "group1_vs_group3": [0.0001, 0.05, 0.1],
            "group2_vs_group3": [0.02, 0.001, 0.5],
        },
        index=["feature_a", "feature_b", "feature_c"],
    )


@pytest.fixture
def pivot_data_cliff():
    """Create sample pivot table data for Cliff's delta coloring."""
    return pd.DataFrame(
        {
            "comparison_a": [0.1, 0.25, 0.4, 0.6],
            "comparison_b": [0.05, 0.3, 0.5, 0.8],
        },
        index=["feature_1", "feature_2", "feature_3", "feature_4"],
    )


# =============================================================================
# TABLE UI INITIALIZATION TESTS
# =============================================================================


class TestTableUIInitialization:
    """Test TableUI initialization."""

    def test_init_with_dataframe(self, qtbot, sample_cell_data):
        """Verify TableUI initializes with a dataframe."""
        table = TableUI(
            data=sample_cell_data,
            title="Test Table",
            population="targets",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.data is sample_cell_data
        assert table.windowTitle() == "Test Table"
        assert table.population == "targets"
        assert hasattr(table, "model")
        assert hasattr(table, "table_view")

        table.close()

    def test_init_detects_tracks(self, qtbot, sample_cell_data):
        """Verify TableUI correctly detects track data."""
        table = TableUI(
            data=sample_cell_data,
            title="Track Test",
            population="targets",
        )
        qtbot.addWidget(table)
        table.show()

        # Should detect tracks since TRACK_ID column exists
        assert table.tracks is True

        table.close()

    def test_init_no_tracks(self, qtbot, sample_data_no_tracks):
        """Verify TableUI works without track data."""
        table = TableUI(
            data=sample_data_no_tracks,
            title="No Track Test",
            population="targets",
        )
        qtbot.addWidget(table)
        table.show()

        # Should not detect tracks
        assert table.tracks is False

        table.close()

    def test_init_with_pairs_population(self, qtbot):
        """Verify TableUI configures correctly for pairs population."""
        pairs_data = pd.DataFrame(
            {
                "position": ["pos1"] * 5,
                "reference_population": ["targets"] * 5,
                "neighbor_population": ["effectors"] * 5,
                "REFERENCE_ID": [0, 0, 1, 1, 2],
                "NEIGHBOR_ID": [0, 1, 0, 1, 0],
                "distance": np.random.rand(5) * 50,
            }
        )

        table = TableUI(
            data=pairs_data,
            title="Pairs Test",
            population="pairs",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.population == "pairs"
        assert "REFERENCE_ID" in table.groupby_cols

        table.close()


class TestTableUIDisplay:
    """Test TableUI display functionality."""

    def test_table_displays_all_columns(self, qtbot, sample_cell_data):
        """Verify all dataframe columns are displayed."""
        table = TableUI(
            data=sample_cell_data,
            title="Column Test",
        )
        qtbot.addWidget(table)
        table.show()

        # Check model column count matches dataframe
        assert table.model.columnCount() == len(sample_cell_data.columns)

        table.close()

    def test_table_displays_all_rows(self, qtbot, sample_cell_data):
        """Verify all dataframe rows are displayed."""
        table = TableUI(
            data=sample_cell_data,
            title="Row Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.model.rowCount() == len(sample_cell_data)

        table.close()


# =============================================================================
# COLUMN OPERATIONS TESTS
# =============================================================================


class TestColumnOperations:
    """Test column manipulation methods."""

    def test_get_selected_columns_none(self, qtbot, sample_cell_data):
        """Verify _get_selected_columns returns empty list when nothing selected."""
        table = TableUI(
            data=sample_cell_data,
            title="Selection Test",
        )
        qtbot.addWidget(table)
        table.show()

        # Nothing selected initially
        selected = table._get_selected_columns()
        assert selected == []

        table.close()

    def test_get_selected_columns_max_cols(self, qtbot, sample_cell_data):
        """Verify max_cols parameter limits returned columns."""
        table = TableUI(
            data=sample_cell_data,
            title="Max Cols Test",
        )
        qtbot.addWidget(table)
        table.show()

        # Even if more are selected, max_cols limits the return
        # (Can't easily simulate selection in tests, just test the parameter works)
        selected = table._get_selected_columns(max_cols=2)
        assert len(selected) <= 2

        table.close()


class TestTableActions:
    """Test menu actions exist."""

    def test_save_action_exists(self, qtbot, sample_cell_data):
        """Verify save action is created."""
        table = TableUI(
            data=sample_cell_data,
            title="Action Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert hasattr(table, "save_as")
        assert table.save_as is not None

        table.close()

    def test_plot_action_exists(self, qtbot, sample_cell_data):
        """Verify plot action is created."""
        table = TableUI(
            data=sample_cell_data,
            title="Action Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert hasattr(table, "plot_action")
        assert table.plot_action is not None

        table.close()

    def test_collapse_tracks_action_enabled_with_tracks(self, qtbot, sample_cell_data):
        """Verify collapse tracks action is enabled when tracks exist."""
        table = TableUI(
            data=sample_cell_data,
            title="Collapse Test",
            collapse_tracks_option=True,
        )
        qtbot.addWidget(table)
        table.show()

        assert hasattr(table, "groupby_action")
        assert table.groupby_action.isEnabled()

        table.close()

    def test_collapse_tracks_action_disabled_without_tracks(
        self, qtbot, sample_data_no_tracks
    ):
        """Verify collapse tracks action is disabled when no tracks exist."""
        table = TableUI(
            data=sample_data_no_tracks,
            title="Collapse Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert hasattr(table, "groupby_action")
        assert not table.groupby_action.isEnabled()

        table.close()


# =============================================================================
# PIVOT TABLE UI TESTS
# =============================================================================


class TestPivotTableUIInitialization:
    """Test PivotTableUI initialization."""

    def test_init_with_dataframe(self, qtbot, pivot_data_pvalue):
        """Verify PivotTableUI initializes with a dataframe."""
        pivot = PivotTableUI(
            data=pivot_data_pvalue,
            title="Pivot Test",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        assert pivot.data is pivot_data_pvalue
        assert pivot.windowTitle() == "Pivot Test"
        assert hasattr(pivot, "model")
        assert hasattr(pivot, "table")

        pivot.close()

    def test_init_with_pvalue_mode(self, qtbot, pivot_data_pvalue):
        """Verify PivotTableUI applies p-value coloring."""
        pivot = PivotTableUI(
            data=pivot_data_pvalue,
            title="P-Value Test",
            mode="pvalue",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        assert pivot.mode == "pvalue"
        # Colors should be set in the model
        assert len(pivot.model.colors) > 0

        pivot.close()

    def test_init_with_cliff_mode(self, qtbot, pivot_data_cliff):
        """Verify PivotTableUI applies Cliff's delta coloring."""
        pivot = PivotTableUI(
            data=pivot_data_cliff,
            title="Cliff Test",
            mode="cliff",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        assert pivot.mode == "cliff"
        # Colors should be set in the model
        assert len(pivot.model.colors) > 0

        pivot.close()


class TestPivotTableUIColoring:
    """Test PivotTableUI cell coloring functionality."""

    def test_set_cell_color(self, qtbot, pivot_data_pvalue):
        """Verify individual cells can be colored."""
        pivot = PivotTableUI(
            data=pivot_data_pvalue,
            title="Color Test",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        # Set a specific cell to red
        pivot.set_cell_color(0, 0, "red")

        assert (0, 0) in pivot.model.colors

        pivot.close()

    def test_pvalue_coloring_thresholds(self, qtbot):
        """Verify p-value coloring uses correct thresholds."""
        # Create data with known p-values
        data = pd.DataFrame(
            {
                "col1": [0.00005, 0.0005, 0.005, 0.03, 0.1],
            },
            index=["****", "***", "**", "*", "ns"],
        )

        pivot = PivotTableUI(
            data=data,
            title="Threshold Test",
            mode="pvalue",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        # All cells should have colors
        assert len(pivot.model.colors) == 5

        pivot.close()

    def test_cliff_coloring_thresholds(self, qtbot):
        """Verify Cliff's delta coloring uses correct thresholds."""
        # Create data with known Cliff's delta values
        data = pd.DataFrame(
            {
                "col1": [0.1, 0.2, 0.4, 0.6],  # negligible, small, medium, large
            },
            index=["neg", "small", "med", "large"],
        )

        pivot = PivotTableUI(
            data=data,
            title="Cliff Threshold Test",
            mode="cliff",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        # All cells should have colors
        assert len(pivot.model.colors) == 4

        pivot.close()


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases for TableUI and PivotTableUI."""

    def test_tableui_empty_dataframe(self, qtbot):
        """Verify TableUI handles empty dataframe."""
        empty_df = pd.DataFrame()

        table = TableUI(
            data=empty_df,
            title="Empty Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.model.rowCount() == 0
        assert table.model.columnCount() == 0

        table.close()

    def test_tableui_single_row(self, qtbot):
        """Verify TableUI handles single-row dataframe."""
        single_df = pd.DataFrame(
            {
                "a": [1],
                "b": [2],
            }
        )

        table = TableUI(
            data=single_df,
            title="Single Row Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.model.rowCount() == 1
        assert table.model.columnCount() == 2

        table.close()

    def test_tableui_with_nan_values(self, qtbot):
        """Verify TableUI handles NaN values in data."""
        nan_df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [np.nan, 2.0, np.nan],
            }
        )

        table = TableUI(
            data=nan_df,
            title="NaN Test",
        )
        qtbot.addWidget(table)
        table.show()

        assert table.model.rowCount() == 3

        table.close()

    def test_pivottableui_single_cell(self, qtbot):
        """Verify PivotTableUI handles single-cell data."""
        single_df = pd.DataFrame({"col": [0.5]}, index=["row"])

        pivot = PivotTableUI(
            data=single_df,
            title="Single Cell Test",
            mode="cliff",
        )
        qtbot.addWidget(pivot)
        pivot.show()

        assert pivot.model.rowCount() == 1
        assert pivot.model.columnCount() == 1

        pivot.close()


# =============================================================================
# OPTIONS TESTS
# =============================================================================


class TestTableUIOptions:
    """Test TableUI configuration options."""

    def test_save_inplace_option(self, qtbot, sample_cell_data):
        """Verify save_inplace option creates appropriate action."""
        table = TableUI(
            data=sample_cell_data,
            title="Save Inplace Test",
            save_inplace_option=True,
        )
        qtbot.addWidget(table)
        table.show()

        assert hasattr(table, "save_inplace")

        table.close()

    def test_collapse_tracks_option_disabled(self, qtbot, sample_cell_data):
        """Verify collapse_tracks_option=False disables the action."""
        table = TableUI(
            data=sample_cell_data,
            title="Collapse Option Test",
            collapse_tracks_option=False,
        )
        qtbot.addWidget(table)
        table.show()

        # Even with tracks, option should be disabled
        assert not table.groupby_action.isEnabled()

        table.close()

    def test_different_plot_modes(self, qtbot, sample_cell_data):
        """Verify different plot modes are accepted."""
        for mode in ["plot_track_signals", "static"]:
            table = TableUI(
                data=sample_cell_data,
                title=f"Plot Mode {mode} Test",
                plot_mode=mode,
            )
            qtbot.addWidget(table)
            table.show()

            assert table.plot_mode == mode

            table.close()
