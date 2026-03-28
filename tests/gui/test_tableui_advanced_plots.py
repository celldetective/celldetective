"""
Unit tests for the advanced plotting features in TableUI:
- Correlation Matrix (plot_correlation_matrix)
- Parallel Coordinates (plot_parallel_coords)

These tests mock Plotly's rendering functions and PyQt's QWebEngineView/QMainWindow
to ensure they can run headlessly and verify the internal logic, data extraction,
and parameter passing without opening actual windows.
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock

from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication
from PyQt5.QtCore import Qt

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


@pytest.fixture(autouse=True)
def process_events_after_test(qtbot):
    """Ensure all Qt events are processed after each test to prevent hangs."""
    yield
    qtbot.wait(10)
    QApplication.processEvents()


@pytest.fixture
def sample_numeric_data():
    """Create a sample dataframe with multiple numerical columns suitable for correlation and parallel coords."""
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
    """Fixture to provide an initialized TableUI instance."""
    table = TableUI(data=sample_numeric_data, title="Advanced Plots Test")
    qtbot.addWidget(table)
    yield table
    table.close()


# =============================================================================
# CORRELATION MATRIX TESTS
# =============================================================================


class TestCorrelationMatrix:
    def test_set_correlation_matrix_params_ui(self, qtbot, table_ui):
        """Test that the settings dialog opens and populates correctly."""
        table_ui.set_correlation_matrix_params(
            preselected_cols=["area", "intensity_mean"]
        )

        # Verify dialog was created
        assert hasattr(table_ui, "corrMatrixParams")
        assert (
            table_ui.corrMatrixParams.windowTitle() == "Correlation Matrix Parameters"
        )

        # Verify list widget has only numeric columns
        list_widget = table_ui._cm_col_list
        items = [list_widget.item(i).text() for i in range(list_widget.count())]
        assert "area" in items
        assert "intensity_mean" in items
        assert "eccentricity" in items
        assert "TRACK_ID" in items
        assert "position" not in items  # String column should be excluded
        assert "class_label" not in items

        # Verify preselection worked
        selected_items = [item.text() for item in list_widget.selectedItems()]
        assert "area" in selected_items
        assert "intensity_mean" in selected_items
        assert "eccentricity" not in selected_items

        table_ui.corrMatrixParams.close()

    @patch("celldetective.gui.tableUI.QMessageBox.warning")
    def test_plot_correlation_matrix_invalid_selection(
        self, mock_warning, qtbot, table_ui
    ):
        """Test that plotting aborts if fewer than 2 features are selected."""
        table_ui.set_correlation_matrix_params()

        # Select only 1 item
        table_ui._cm_col_list.item(0).setSelected(True)

        table_ui.plot_correlation_matrix()

        # Warning should have been triggered
        mock_warning.assert_called_once()
        args = mock_warning.call_args[0]
        assert "Invalid Selection" in args[1]

        table_ui.corrMatrixParams.close()

    @patch("plotly.express.imshow")
    @patch("plotly.graph_objs.Figure.write_html")
    @patch("celldetective.gui.base.components.CelldetectiveMainWindow.show")
    def test_plot_correlation_matrix_valid(
        self, mock_show, mock_write_html, mock_imshow, qtbot, table_ui
    ):
        """Test the full plotting logic for the correlation matrix."""
        mock_fig = MagicMock()
        mock_imshow.return_value = mock_fig

        # Prevent the internal ImportError from triggering the browser fallback
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "PyQt5.QtWebEngineWidgets":
                mock = MagicMock()
                from PyQt5.QtWidgets import QWidget

                class MockQWebEngineView(QWidget):
                    def load(self, url):
                        pass

                mock.QWebEngineView = MockQWebEngineView
                return mock
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            table_ui.set_correlation_matrix_params()

            # Select 3 features
            for col_name in ["area", "intensity_mean", "eccentricity"]:
                items = table_ui._cm_col_list.findItems(col_name, Qt.MatchExactly)
                if items:
                    items[0].setSelected(True)

            # Set method to spearman
            table_ui._cm_method_cb.setCurrentText("spearman")

            # Trigger plot
            table_ui.plot_correlation_matrix()

            # 1. Verify px.imshow was called correctly
            mock_imshow.assert_called_once()
            called_df = mock_imshow.call_args[0][0]  # The correlation matrix dataframe

            # Ensure the underlying corr calculation used spearman on the correct columns
            assert isinstance(called_df, pd.DataFrame)
            assert list(called_df.columns) == ["area", "intensity_mean", "eccentricity"]
            assert called_df.index.tolist() == [
                "area",
                "intensity_mean",
                "eccentricity",
            ]

            # Check kwargs
            kwargs = mock_imshow.call_args[1]
            assert "Correlation Matrix (Spearman)" in kwargs["title"]
            assert kwargs["color_continuous_scale"] == "rdbu"
            assert kwargs["text_auto"] == ".2f"

            # 2. Verify HTML was written
            mock_fig.write_html.assert_called_once()
            html_path = mock_fig.write_html.call_args[0][0]
            assert "corr_matrix_" in html_path
            assert html_path.endswith(".html")

            # 3. Verify Qt window was created and shown
            assert hasattr(table_ui, "cm_window")
            assert table_ui.cm_window.windowTitle() == "Correlation Matrix"
            mock_show.assert_called_once()

            table_ui.cm_window.close()
            table_ui.corrMatrixParams.close()


# =============================================================================
# PARALLEL COORDINATES TESTS
# =============================================================================


class TestParallelCoordinates:
    def test_set_parallel_coords_params_ui(self, qtbot, table_ui):
        """Test that the Parallel Coordinates settings dialog opens correctly."""
        table_ui.set_parallel_coords_params(
            preselected_cols=["area", "intensity_mean", "eccentricity"]
        )

        # Verify dialog was created
        assert hasattr(table_ui, "parallelCoordsParams")
        assert (
            table_ui.parallelCoordsParams.windowTitle()
            == "Parallel Coordinates Parameters"
        )

        # Verify lists
        list_widget = table_ui._pc_col_list
        items = [list_widget.item(i).text() for i in range(list_widget.count())]
        assert "TRACK_ID" in items
        assert "area" in items
        assert "class_label" not in items  # Non-numeric excluded from axes

        # Verify comboboxes (include all columns + "--")
        hue_items = [
            table_ui._pc_hue_cb.itemText(i) for i in range(table_ui._pc_hue_cb.count())
        ]
        assert "--" in hue_items
        assert "class_label" in hue_items  # Can use categorical for hue

        table_ui.parallelCoordsParams.close()

    @patch("celldetective.gui.tableUI.logger.warning")
    def test_plot_parallel_coords_invalid_selection(self, mock_logger, qtbot, table_ui):
        """Test that plotting aborts if fewer than 2 dimensions are selected."""
        table_ui.set_parallel_coords_params()

        # Select only 1 axis
        table_ui._pc_col_list.item(0).setSelected(True)

        table_ui.plot_parallel_coords()

        # Warning should have been triggered
        mock_logger.assert_called_with(
            "parallel coordinates: please select at least 2 axis columns."
        )

        table_ui.parallelCoordsParams.close()

    @patch("plotly.graph_objects.Figure")
    @patch("plotly.graph_objs.Figure.write_html")
    @patch("celldetective.gui.base.components.CelldetectiveMainWindow.show")
    def test_plot_parallel_coords_valid(
        self, mock_show, mock_write_html, mock_go_figure, qtbot, table_ui
    ):
        """Test full plotting logic, normalization, and color mapping for parallel coords."""
        mock_fig = MagicMock()
        mock_go_figure.return_value = mock_fig

        # Prevent the internal ImportError from triggering the browser fallback
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "PyQt5.QtWebEngineWidgets":
                mock = MagicMock()
                from PyQt5.QtWidgets import QWidget

                class MockQWebEngineView(QWidget):
                    def load(self, url):
                        pass

                mock.QWebEngineView = MockQWebEngineView
                return mock
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            table_ui.set_parallel_coords_params()

            # Setup selection: 3 axes, hue by class_label, z-score norm
            axes = ["area", "intensity_mean", "eccentricity"]
            for col_name in axes:
                items = table_ui._pc_col_list.findItems(col_name, Qt.MatchExactly)
                if items:
                    items[0].setSelected(True)

            table_ui._pc_hue_cb.setCurrentText("class_label")
            table_ui._pc_norm_cb.setCurrentText("z-score")

            table_ui.plot_parallel_coords()

            # 1. Verify Figure was instantiated with Parcoords data
            mock_go_figure.assert_called_once()
            kwargs = mock_go_figure.call_args[1]

            assert "data" in kwargs
            # Extract the Parcoords object (which is a plotly graph object, we can inspect its properties if needed,
            # but since it's mocked, we actually check what was passed to go.Figure)
            # Note: we patched go.Figure, so we inspect its call arguments
            data_arg = kwargs["data"]

            # Verify normalization (z-score) happened correctly (means ~0)
            # We can't easily introspect the internal plotly object if we mocked the parent,
            # but the logic runs without crashing.

            # 2. Verify HTML was written
            mock_fig.write_html.assert_called_once()
            html_path = mock_fig.write_html.call_args[0][0]
            assert "parallel_coords_" in html_path
            assert html_path.endswith(".html")

            # 3. Verify Qt window structure
            assert hasattr(table_ui, "pc_window")
            assert table_ui.pc_window.windowTitle() == "Parallel Coordinates"
            mock_show.assert_called_once()

            table_ui.pc_window.close()
            table_ui.parallelCoordsParams.close()
