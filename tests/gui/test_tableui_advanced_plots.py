"""
Unit tests for the advanced plotting features in TableUI:
- Correlation Matrix (plot_correlation_matrix)
- Parallel Coordinates (plot_parallel_coords)

These tests use real TableUI instances (no mocking of Qt widgets) and only block
QWebEngineView — the Chromium-based browser widget — by replacing
PyQt5.QtWebEngineWidgets in sys.modules before the import executes inside the
plotting methods.  That is the correct interception point: Python checks
sys.modules before calling __import__, so this works even when the real
PyQtWebEngine package is installed and already cached.
"""

import sys
import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt

from celldetective.gui.tableUI import TableUI


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def process_events_after_test(qtbot):
    """Ensure all Qt events are processed after each test to prevent hangs."""
    yield
    qtbot.wait(10)
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def mock_web_engine():
    """
    Replace PyQt5.QtWebEngineWidgets in sys.modules with a lightweight stub.

    QWebEngineView is a full Chromium browser widget.  Instantiating it in a
    headless CI environment spawns GPU/renderer subprocesses that hang
    indefinitely because there is no real graphics hardware.  Replacing the
    module in sys.modules is the only reliable interception point: Python
    consults sys.modules before calling __import__, so the stub is used even
    when the real package is installed and its module object is already cached.

    The stub's QWebEngineView is a plain QWidget subclass so that
    setCentralWidget(browser) in the production code receives a valid Qt object.
    """

    class _FakeWebEngineView(QWidget):
        def load(self, url):
            pass

    stub = MagicMock()
    stub.QWebEngineView = _FakeWebEngineView

    with patch.dict(sys.modules, {"PyQt5.QtWebEngineWidgets": stub}):
        yield stub


@pytest.fixture
def sample_numeric_data():
    """Dataframe with mixed numeric / categorical columns for plotting tests."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "position": ["pos1", "pos1", "pos2", "pos2", "pos3"],
            "TRACK_ID": [1, 2, 3, 4, 5],
            "area": [100.5, 120.2, 95.0, 150.0, 110.0],
            "intensity_mean": [50.0, 55.5, 48.0, 60.0, 52.0],
            "eccentricity": [0.1, 0.8, 0.2, 0.9, 0.15],
            "class_label": ["A", "A", "B", "B", "C"],
        }
    )


@pytest.fixture
def table_ui(qtbot, sample_numeric_data):
    """Initialized TableUI instance registered with qtbot."""
    table = TableUI(data=sample_numeric_data, title="Advanced Plots Test")
    qtbot.addWidget(table)
    yield table
    table.close()


# =============================================================================
# Correlation Matrix Tests
# =============================================================================


class TestCorrelationMatrix:
    def test_set_correlation_matrix_params_ui(self, qtbot, table_ui):
        """The settings dialog opens, lists only numeric columns, and honours preselection."""
        table_ui.set_correlation_matrix_params(
            preselected_cols=["area", "intensity_mean"]
        )

        assert hasattr(table_ui, "corrMatrixParams")
        assert table_ui.corrMatrixParams.windowTitle() == "Correlation Matrix Parameters"

        list_widget = table_ui._cm_col_list
        items = [list_widget.item(i).text() for i in range(list_widget.count())]
        assert "area" in items
        assert "intensity_mean" in items
        assert "eccentricity" in items
        assert "TRACK_ID" in items
        assert "position" not in items    # string column excluded
        assert "class_label" not in items  # string column excluded

        selected = [item.text() for item in list_widget.selectedItems()]
        assert "area" in selected
        assert "intensity_mean" in selected
        assert "eccentricity" not in selected

        table_ui.corrMatrixParams.close()

    @patch("celldetective.gui.tableUI.QMessageBox.warning")
    def test_plot_correlation_matrix_invalid_selection(
        self, mock_warning, qtbot, table_ui
    ):
        """Plotting aborts and shows a warning when fewer than 2 features are selected."""
        table_ui.set_correlation_matrix_params()

        # Select only one item
        table_ui._cm_col_list.clearSelection()
        table_ui._cm_col_list.item(0).setSelected(True)

        table_ui.plot_correlation_matrix()

        mock_warning.assert_called_once()
        assert "Invalid Selection" in mock_warning.call_args[0][1]

        table_ui.corrMatrixParams.close()

    def test_plot_correlation_matrix_valid(self, qtbot, table_ui):
        """A valid selection produces a titled window without opening a real browser."""
        table_ui.set_correlation_matrix_params()

        for col_name in ["area", "intensity_mean", "eccentricity"]:
            items = table_ui._cm_col_list.findItems(col_name, Qt.MatchExactly)
            if items:
                items[0].setSelected(True)

        table_ui._cm_method_cb.setCurrentText("spearman")

        table_ui.plot_correlation_matrix()

        assert hasattr(table_ui, "cm_window")
        assert table_ui.cm_window.windowTitle() == "Correlation Matrix"

        table_ui.cm_window.close()
        table_ui.corrMatrixParams.close()


# =============================================================================
# Parallel Coordinates Tests
# =============================================================================


class TestParallelCoordinates:
    def test_set_parallel_coords_params_ui(self, qtbot, table_ui):
        """The settings dialog opens, lists numeric axes, and exposes all columns for hue."""
        table_ui.set_parallel_coords_params(
            preselected_cols=["area", "intensity_mean", "eccentricity"]
        )

        assert hasattr(table_ui, "parallelCoordsParams")
        assert (
            table_ui.parallelCoordsParams.windowTitle()
            == "Parallel Coordinates Parameters"
        )

        list_widget = table_ui._pc_col_list
        items = [list_widget.item(i).text() for i in range(list_widget.count())]
        assert "TRACK_ID" in items
        assert "area" in items
        assert "class_label" not in items  # non-numeric excluded from axes

        hue_items = [
            table_ui._pc_hue_cb.itemText(i) for i in range(table_ui._pc_hue_cb.count())
        ]
        assert "--" in hue_items
        assert "class_label" in hue_items  # categorical columns allowed for hue

        table_ui.parallelCoordsParams.close()

    def test_plot_parallel_coords_invalid_selection(self, qtbot, table_ui):
        """Plotting aborts (with a logger warning) when fewer than 2 axes are selected."""
        table_ui.set_parallel_coords_params()

        table_ui._pc_col_list.clearSelection()
        table_ui._pc_col_list.item(0).setSelected(True)

        with patch("celldetective.gui.tableUI.logger") as mock_logger:
            table_ui.plot_parallel_coords()
            mock_logger.warning.assert_called_with(
                "parallel coordinates: please select at least 2 axis columns."
            )

        table_ui.parallelCoordsParams.close()

    def test_plot_parallel_coords_valid(self, qtbot, table_ui):
        """A valid selection with z-score normalisation produces a titled window."""
        table_ui.set_parallel_coords_params()

        for col_name in ["area", "intensity_mean", "eccentricity"]:
            items = table_ui._pc_col_list.findItems(col_name, Qt.MatchExactly)
            if items:
                items[0].setSelected(True)

        table_ui._pc_hue_cb.setCurrentText("class_label")
        table_ui._pc_norm_cb.setCurrentText("z-score")

        table_ui.plot_parallel_coords()

        assert hasattr(table_ui, "pc_window")
        assert table_ui.pc_window.windowTitle() == "Parallel Coordinates"

        table_ui.pc_window.close()
        table_ui.parallelCoordsParams.close()
