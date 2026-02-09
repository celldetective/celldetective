"""
Unit tests for plot widgets in celldetective.gui.

Covers:
- ConfigMeasurementsPlot (in celldetective.gui.plot_measurements)
- GenericSignalPlotWidget (in celldetective.gui.generic_signal_plot)
"""

import pytest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QMainWindow, QWidget

from celldetective.gui.plot_measurements import ConfigMeasurementsPlot
from celldetective.gui.generic_signal_plot import GenericSignalPlotWidget


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
def mock_parent_window():
    """Mock the complex parent window hierarchy required by ConfigMeasurementsPlot."""
    # Hierarchy: self.parent_window.parent_window.parent_window...
    # ConfigMeasurementsPlot uses:
    # self.parent_window.exp_dir
    # self.parent_window.parent_window.wells
    # self.parent_window.parent_window.FrameToMin
    # self.parent_window.parent_window.well_list.getSelectedIndices()
    # self.parent_window.parent_window.position_list.getSelectedIndices()
    # self.parent_window.parent_window.populations
    # self.parent_window.cbs (for GenericSignalPlotWidget cmap)

    # Main Window Mock
    main_window = MagicMock()
    main_window.screen_height = 1000
    main_window.wells = ["well1", "well2"]
    main_window.FrameToMin = 5.0
    main_window.populations = ["cells", "nuclei"]

    # Lists
    main_window.well_list = MagicMock()
    main_window.well_list.getSelectedIndices.return_value = 0  # First well

    main_window.position_list = MagicMock()
    main_window.position_list.getSelectedIndices.return_value = (
        0  # All positions? Or specific?
    )
    # In code: if position_option == 0: position_indices = None (all)

    # Middle Window (e.g., PlottingWindow or similar)
    middle_window = MagicMock()
    middle_window.parent_window = main_window
    middle_window.FrameToMin = 5.0
    middle_window.exp_dir = "C:/fake/exp/dir/"

    # For GenericSignalPlotWidget, parent_window represents the ConfigMeasurementsPlot or similar
    # It accesses parent_window.cbs[-1].currentText() for cmap
    cmap_cb = MagicMock()
    cmap_cb.currentText.return_value = "viridis"
    middle_window.cbs = [MagicMock(), MagicMock(), MagicMock(), cmap_cb]

    # GenericSignalWindow needs well_indices, position_indices
    middle_window.well_indices = [0]
    middle_window.position_indices = [1]
    middle_window.well_labels = ["well1: condA", "well2: condB"]

    return middle_window


@pytest.fixture
def dummy_dataframes():
    """Create dummy DataFrames for GenericSignalPlotWidget."""
    # df: Main data
    # Columns usually: FRAME, TRACK_ID, position, well_name, <feature>
    df = pd.DataFrame(
        {
            "FRAME": np.tile(np.arange(10), 5),
            "TRACK_ID": np.repeat(np.arange(5), 10),
            "position": ["pos1"] * 50,
            "well_name": ["well1"] * 50,
            "area": np.random.rand(50) * 100,
            "intensity": np.random.rand(50) * 255,
        }
    )

    # Signal data must be a dict-like object with specific keys
    signal_data = {
        "timeline": np.arange(10),
        "mean_all": np.random.rand(10),
        "std_all": np.random.rand(10),
        "mean_event": np.random.rand(10),
        "std_event": np.random.rand(10),
        "mean_no_event": np.random.rand(10),
        "std_no_event": np.random.rand(10),
    }

    # df_pos_info: Position metadata
    # Columns: pos_name, pos_path, stack_path, select, x, y, metadata_tag, well_index, well_name
    df_pos_info = pd.DataFrame(
        {
            "pos_name": ["pos1"],
            "pos_path": ["path/to/pos1"],
            "stack_path": ["path/to/pos1/stack.tif"],
            "select": [True],
            "x": [100.0],
            "y": [100.0],
            "metadata_tag": ["Pos1"],
            "well_index": [0],
            "well_name": ["well1"],
            "pos_index": [0],
            "signal": [signal_data],  # Dummy signal array for plotting
        }
    )

    # df_well_info
    df_well_info = pd.DataFrame(
        {"well_name": ["well1"], "well_index": [0], "select": [True]}
    )

    return df, df_pos_info, df_well_info


# =============================================================================
# GenericSignalPlotWidget Tests
# =============================================================================


class TestGenericSignalPlotWidget:
    """Tests for GenericSignalPlotWidget."""

    def test_initialization(self, qtbot, mock_parent_window, dummy_dataframes):
        """Test initialization with dummy data."""
        df, df_pos, df_well = dummy_dataframes

        widget = GenericSignalPlotWidget(
            df=df,
            df_pos_info=df_pos,
            df_well_info=df_well,
            feature_selected="area",
            parent_window=mock_parent_window,
            title="Test Plot",
        )
        qtbot.addWidget(widget)
        widget.show()

        assert widget.windowTitle() == "Test Plot"
        assert widget.feature_selected == "area"
        # Check if plot canvas exists
        assert hasattr(widget, "plot_widget")

    def test_plot_signals_basic(self, qtbot, mock_parent_window, dummy_dataframes):
        """Test that plot_signals runs without error."""
        df, df_pos, df_well = dummy_dataframes

        widget = GenericSignalPlotWidget(
            df=df,
            df_pos_info=df_pos,
            df_well_info=df_well,
            feature_selected="area",
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(widget)

        # Trigger plot
        widget.plot_signals(0)

        # Verify axis labels
        assert widget.ax.get_ylabel() == "area"
        assert widget.ax.get_xlabel() == "time [min]"

    def test_ui_interactions(self, qtbot, mock_parent_window, dummy_dataframes):
        """Test UI buttons like legend, log scale."""
        df, df_pos, df_well = dummy_dataframes

        widget = GenericSignalPlotWidget(
            df=df,
            df_pos_info=df_pos,
            df_well_info=df_well,
            feature_selected="intensity",
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(widget)

        # Legend toggle
        assert widget.legend_visible is True
        widget.legend_btn.click()
        assert widget.legend_visible is False

        # Log toggle
        initial_scale = widget.ax.get_yscale()
        widget.log_btn.click()
        assert widget.ax.get_yscale() != initial_scale

    def test_rescale(self, qtbot, mock_parent_window, dummy_dataframes):
        """Test rescaling functionality."""
        df, df_pos, df_well = dummy_dataframes

        widget = GenericSignalPlotWidget(
            df=df,
            df_pos_info=df_pos,
            df_well_info=df_well,
            feature_selected="area",
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(widget)

        widget.scaling_factor_le.setText("2.0")
        widget.rescale_btn.click()

        assert widget.scaling_factor == 2.0


# =============================================================================
# ConfigMeasurementsPlot Tests
# =============================================================================


class TestConfigMeasurementsPlot:
    """Tests for ConfigMeasurementsPlot."""

    @patch("celldetective.gui.plot_measurements._extract_labels_from_config")
    def test_initialization(self, mock_extract, qtbot, mock_parent_window):
        """Test initialization."""
        mock_extract.return_value = ["well1", "well2"]

        # ConfigMeasurementsPlot expects parent_window to have parent_window...
        # Our mock_parent_window has parent_window (Main).
        # And accessing parent_window.parent_window.well_list works.

        widget = ConfigMeasurementsPlot(parent_window=mock_parent_window)
        qtbot.addWidget(widget)
        widget.show()

        assert widget.windowTitle() == "Configure signal plot"
        assert widget.cbs[0].count() > 0  # Populations

    @patch("celldetective.gui.plot_measurements._extract_labels_from_config")
    @patch("celldetective.gui.plot_measurements.glob")
    @patch("celldetective.gui.plot_measurements.pd.read_csv")
    def test_population_change_updates_classes(
        self, mock_read_csv, mock_glob, mock_extract, qtbot, mock_parent_window
    ):
        """Test that changing population finds available tables and updates classes."""
        mock_extract.return_value = ["well1"]
        mock_glob.return_value = ["fake_table.csv"]

        # Mock reading csv header
        mock_df = pd.DataFrame(
            columns=["TRACK_ID", "class_positive", "group_treatment"]
        )
        mock_read_csv.return_value = mock_df

        widget = ConfigMeasurementsPlot(parent_window=mock_parent_window)
        qtbot.addWidget(widget)

        # Trigger set_classes_and_times (called on init, but lets call explicitly or via signal)
        widget.set_classes_and_times()

        # Check if classes combobox (index 1) has items
        # It should find "class_positive"
        # The code filters columns starting with "class_"
        # mock_df columns: class_positive -> yes

        # widget.cbs[1] is classes conf
        items = [widget.cbs[1].itemText(i) for i in range(widget.cbs[1].count())]
        assert "class_positive" in items

    @patch("celldetective.gui.plot_measurements._extract_labels_from_config")
    @patch("celldetective.gui.plot_measurements.load_experiment_tables")
    def test_process_signal_opens_dialog(
        self, mock_load, mock_extract, qtbot, mock_parent_window
    ):
        """Test process_signal loads tables and asks for feature."""
        mock_extract.return_value = ["well1"]

        # Mock load_experiment_tables returning DF
        df = pd.DataFrame(
            {
                "area": [1, 2],
                "intensity": [3, 4],
                "well_name": ["w1", "w1"],
                "FRAME": [0, 1],
            }
        )

        # Signal data must be a dict-like object with specific keys
        signal_data = {
            "timeline": np.arange(10),
            "mean_all": np.random.rand(10),
            "std_all": np.random.rand(10),
            "mean_event": np.random.rand(10),
            "std_event": np.random.rand(10),
            "mean_no_event": np.random.rand(10),
            "std_no_event": np.random.rand(10),
        }

        df_pos = pd.DataFrame(
            {
                "well_path": ["p1"],
                "well_index": [0],
                "well_name": ["w1"],
                "well_number": [1],
                "well_alias": ["a1"],
                "stack_path": ["s1"],
                "pos_name": ["pos1"],
                "pos_path": ["path/to/pos1"],
                "select": [True],
                "x": [100.0],
                "y": [100.0],
                "metadata_tag": ["Pos1"],
                "pos_index": [0],
                "signal": [signal_data],  # Signal is a dict
            }
        )
        mock_load.return_value = (df, df_pos)

        widget = ConfigMeasurementsPlot(parent_window=mock_parent_window)
        qtbot.addWidget(widget)

        # We need to mock ask_for_features to verify it's called
        with patch.object(widget, "ask_for_features") as mock_ask:
            widget.process_signal()
            assert mock_load.called
            assert mock_ask.called
