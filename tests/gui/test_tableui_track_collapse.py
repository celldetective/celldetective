"""
Regression test for TableUI track collapse functionality.
Tests the fix for ValueError: numeric_only accepts only Boolean values
when using groupby aggregation methods (mean, sum, etc.) in set_proj_mode.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from PyQt5 import QtCore

from celldetective.gui.tableUI import TableUI


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
def sample_track_data():
    """Create sample DataFrame with track data for testing."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 6 + ["pos2"] * 6,
            "TRACK_ID": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "FRAME": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            "POSITION_X": [
                10.0,
                12.0,
                14.0,
                20.0,
                22.0,
                24.0,
                30.0,
                32.0,
                34.0,
                40.0,
                42.0,
                44.0,
            ],
            "POSITION_Y": [
                10.0,
                12.0,
                14.0,
                20.0,
                22.0,
                24.0,
                30.0,
                32.0,
                34.0,
                40.0,
                42.0,
                44.0,
            ],
            "area": [
                100.0,
                110.0,
                120.0,
                200.0,
                210.0,
                220.0,
                300.0,
                310.0,
                320.0,
                400.0,
                410.0,
                420.0,
            ],
            "intensity": [
                50.0,
                55.0,
                60.0,
                70.0,
                75.0,
                80.0,
                90.0,
                95.0,
                100.0,
                110.0,
                115.0,
                120.0,
            ],
            # Non-numeric columns to ensure numeric_only=True works
            "well_name": ["W1"] * 12,
            "pos_name": ["100"] * 6 + ["200"] * 6,
            "status": [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )


def test_tableui_track_collapse_mean(qtbot, sample_track_data):
    """
    Test that collapsing tracks with 'mean' operation works without ValueError.
    This is a regression test for the fix: numeric_only=True in groupby aggregation.
    """
    table_ui = TableUI(
        data=sample_track_data,
        title="Test Table",
        population="targets",
        plot_mode="plot_track_signals",
        collapse_tracks_option=True,
    )
    qtbot.addWidget(table_ui)
    table_ui.show()
    qtbot.wait(100)

    # Open the projection mode dialog
    table_ui.set_projection_mode_tracks()
    qtbot.wait(100)

    # Set up projection parameters - mean is the default
    assert table_ui.projection_option.isChecked()
    table_ui.projection_op_cb.setCurrentText("mean")
    table_ui.current_data = sample_track_data

    # Execute the collapse - this should NOT raise ValueError
    try:
        table_ui.set_proj_mode()
        qtbot.wait(100)
    except ValueError as e:
        if "numeric_only" in str(e):
            pytest.fail(f"Regression: numeric_only ValueError occurred: {e}")
        raise

    # Verify subtable was created
    assert hasattr(table_ui, "subtable")
    assert table_ui.subtable is not None

    # Cleanup
    table_ui.subtable.close()
    table_ui.close()


def test_tableui_track_collapse_all_operations(qtbot, sample_track_data):
    """
    Test that all aggregation operations work in track collapse without ValueError.
    Operations tested: mean, median, min, max, first, last, prod, sum
    """
    operations = ["mean", "median", "min", "max", "first", "last", "prod", "sum"]

    for op in operations:
        table_ui = TableUI(
            data=sample_track_data.copy(),
            title=f"Test Table - {op}",
            population="targets",
            plot_mode="plot_track_signals",
            collapse_tracks_option=True,
        )
        qtbot.addWidget(table_ui)
        table_ui.show()
        qtbot.wait(50)

        # Open the projection mode dialog
        table_ui.set_projection_mode_tracks()
        qtbot.wait(50)

        # Configure for the current operation
        table_ui.projection_option.setChecked(True)
        table_ui.projection_op_cb.setCurrentText(op)
        table_ui.current_data = sample_track_data.copy()

        # Execute the collapse
        try:
            table_ui.set_proj_mode()
            qtbot.wait(50)
        except ValueError as e:
            if "numeric_only" in str(e):
                pytest.fail(
                    f"Regression: numeric_only ValueError occurred for '{op}': {e}"
                )
            raise

        # Verify subtable was created
        assert hasattr(table_ui, "subtable"), f"subtable not created for '{op}'"
        assert table_ui.subtable is not None, f"subtable is None for '{op}'"

        # Cleanup
        table_ui.subtable.close()
        table_ui.close()


def test_tableui_track_collapse_with_mixed_dtypes(qtbot):
    """
    Test track collapse with a DataFrame containing mixed data types.
    Ensures numeric_only=True properly handles non-numeric columns.
    """
    mixed_data = pd.DataFrame(
        {
            "position": ["pos1"] * 4,
            "TRACK_ID": [1, 1, 2, 2],
            "FRAME": [0, 1, 0, 1],
            "numeric_col": [1.5, 2.5, 3.5, 4.5],
            "string_col": ["a", "b", "c", "d"],
            "bool_col": [True, False, True, False],
            "category_col": pd.Categorical(["cat1", "cat1", "cat2", "cat2"]),
        }
    )

    table_ui = TableUI(
        data=mixed_data,
        title="Mixed Types Table",
        population="targets",
        plot_mode="plot_track_signals",
        collapse_tracks_option=True,
    )
    qtbot.addWidget(table_ui)
    table_ui.show()
    qtbot.wait(100)

    # Open the projection mode dialog
    table_ui.set_projection_mode_tracks()
    qtbot.wait(100)

    table_ui.projection_option.setChecked(True)
    table_ui.projection_op_cb.setCurrentText("mean")
    table_ui.current_data = mixed_data

    # This should not raise ValueError about numeric_only
    try:
        table_ui.set_proj_mode()
        qtbot.wait(100)
    except ValueError as e:
        if "numeric_only" in str(e):
            pytest.fail(f"Regression: numeric_only ValueError with mixed types: {e}")
        raise

    assert hasattr(table_ui, "subtable")

    # Cleanup
    table_ui.subtable.close()
    table_ui.close()
