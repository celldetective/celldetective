"""
Unit tests for GUI utility components in gui_utils.py.

Tests cover:
- PandasModel for displaying DataFrames in Qt views
- PreprocessingLayout for filter configuration
- FilterChoice for filter selection dialogs
"""

import pytest
import numpy as np
import pandas as pd
import logging
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from celldetective.gui.gui_utils import PandasModel, FilterChoice


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
def sample_dataframe():
    """Create a sample dataframe for testing."""
    data = {
        "cell_id": [1, 2, 3, 4, 5],
        "area": [100.5, 150.2, 120.8, 180.3, 90.1],
        "perimeter": [40.2, 55.1, 48.3, 62.0, 35.5],
        "class": ["A", "B", "A", "B", "A"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_dataframe():
    """Create a larger dataframe for performance testing."""
    n_rows = 1000
    data = {
        "id": range(n_rows),
        "x": np.random.rand(n_rows),
        "y": np.random.rand(n_rows),
        "z": np.random.rand(n_rows),
    }
    return pd.DataFrame(data)


# =============================================================================
# PANDAS MODEL TESTS
# =============================================================================


class TestPandasModelInitialization:
    """Test PandasModel initialization."""

    def test_init_with_dataframe(self, sample_dataframe):
        """Verify PandasModel initializes with a dataframe."""
        model = PandasModel(sample_dataframe)

        assert model._data is sample_dataframe
        assert model.colors == {}

    def test_init_with_empty_dataframe(self):
        """Verify PandasModel handles empty dataframe."""
        empty_df = pd.DataFrame()
        model = PandasModel(empty_df)

        assert model.rowCount() == 0
        assert model.columnCount() == 0


class TestPandasModelRowColumn:
    """Test row and column count methods."""

    def test_rowcount_matches_dataframe(self, sample_dataframe):
        """Verify rowCount matches dataframe rows."""
        model = PandasModel(sample_dataframe)

        assert model.rowCount() == len(sample_dataframe)
        assert model.rowCount() == 5

    def test_columncount_matches_dataframe(self, sample_dataframe):
        """Verify columnCount matches dataframe columns."""
        model = PandasModel(sample_dataframe)

        assert model.columnCount() == len(sample_dataframe.columns)
        assert model.columnCount() == 4


class TestPandasModelData:
    """Test data retrieval methods."""

    def test_data_retrieval_display_role(self, qtbot, sample_dataframe):
        """Verify data returns correct string value for DisplayRole."""
        model = PandasModel(sample_dataframe)

        # Get data at row 0, column 0 (cell_id = 1)
        index = model.index(0, 0)
        data = model.data(index, Qt.DisplayRole)

        assert data == "1"

    def test_data_retrieval_string_column(self, qtbot, sample_dataframe):
        """Verify data correctly handles string columns."""
        model = PandasModel(sample_dataframe)

        # Get data at row 1, column 3 (class = "B")
        index = model.index(1, 3)
        data = model.data(index, Qt.DisplayRole)

        assert data == "B"

    def test_data_retrieval_float_column(self, qtbot, sample_dataframe):
        """Verify data correctly handles float columns."""
        model = PandasModel(sample_dataframe)

        # Get data at row 0, column 1 (area = 100.5)
        index = model.index(0, 1)
        data = model.data(index, Qt.DisplayRole)

        assert data == "100.5"

    def test_data_invalid_index(self, qtbot, sample_dataframe):
        """Verify data returns None for invalid index."""
        model = PandasModel(sample_dataframe)

        # Create an invalid index
        index = model.index(-1, -1)
        data = model.data(index, Qt.DisplayRole)

        assert data is None


class TestPandasModelHeaders:
    """Test header data retrieval."""

    def test_horizontal_header_data(self, sample_dataframe):
        """Verify horizontal headers match column names."""
        model = PandasModel(sample_dataframe)

        # Check each column header
        assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "cell_id"
        assert model.headerData(1, Qt.Horizontal, Qt.DisplayRole) == "area"
        assert model.headerData(2, Qt.Horizontal, Qt.DisplayRole) == "perimeter"
        assert model.headerData(3, Qt.Horizontal, Qt.DisplayRole) == "class"

    def test_vertical_header_data(self, sample_dataframe):
        """Verify vertical headers match row indices."""
        model = PandasModel(sample_dataframe)

        # Check row headers (default integer index)
        assert model.headerData(0, Qt.Vertical, Qt.DisplayRole) == 0
        assert model.headerData(1, Qt.Vertical, Qt.DisplayRole) == 1
        assert model.headerData(4, Qt.Vertical, Qt.DisplayRole) == 4

    def test_header_data_wrong_role(self, sample_dataframe):
        """Verify headerData returns None for non-DisplayRole."""
        model = PandasModel(sample_dataframe)

        result = model.headerData(0, Qt.Horizontal, Qt.EditRole)
        assert result is None


class TestPandasModelColors:
    """Test cell coloring functionality."""

    def test_change_color(self, qtbot, sample_dataframe):
        """Verify change_color sets cell background color."""
        model = PandasModel(sample_dataframe)
        test_color = QColor(255, 0, 0)  # Red

        # Change color of cell (1, 1)
        model.change_color(1, 1, test_color)

        assert (1, 1) in model.colors
        assert model.colors[(1, 1)] == test_color

    def test_background_role_returns_color(self, qtbot, sample_dataframe):
        """Verify BackgroundRole returns the set color."""
        model = PandasModel(sample_dataframe)
        test_color = QColor(0, 255, 0)  # Green

        model.change_color(2, 2, test_color)

        index = model.index(2, 2)
        result = model.data(index, Qt.BackgroundRole)

        assert result == test_color

    def test_background_role_returns_none_when_no_color(self, qtbot, sample_dataframe):
        """Verify BackgroundRole returns None when no color is set."""
        model = PandasModel(sample_dataframe)

        index = model.index(0, 0)
        result = model.data(index, Qt.BackgroundRole)

        assert result is None

    def test_multiple_color_changes(self, qtbot, sample_dataframe):
        """Verify multiple cells can have different colors."""
        model = PandasModel(sample_dataframe)

        color1 = QColor(255, 0, 0)
        color2 = QColor(0, 255, 0)
        color3 = QColor(0, 0, 255)

        model.change_color(0, 0, color1)
        model.change_color(1, 1, color2)
        model.change_color(2, 2, color3)

        assert model.colors[(0, 0)] == color1
        assert model.colors[(1, 1)] == color2
        assert model.colors[(2, 2)] == color3


class TestPandasModelLargeData:
    """Test PandasModel with larger datasets."""

    def test_large_dataframe_performance(self, large_dataframe):
        """Verify PandasModel handles large dataframes efficiently."""
        model = PandasModel(large_dataframe)

        assert model.rowCount() == 1000
        assert model.columnCount() == 4

        # Access data at various points
        index = model.index(500, 1)
        data = model.data(index, Qt.DisplayRole)
        assert data is not None


# =============================================================================
# FILTER CHOICE TESTS
# =============================================================================


class MockListWidget:
    """Mock list widget for testing FilterChoice."""

    def __init__(self):
        self.items = []

    def addItems(self, items):
        self.items.extend(items)


class MockParentWindow:
    """Mock parent window for FilterChoice testing."""

    def __init__(self):
        self.items = []
        self.list_widget = MockListWidget()


class TestFilterChoice:
    """Test FilterChoice dialog functionality."""

    def test_init_creates_combobox(self, qtbot):
        """Verify FilterChoice initializes with filter combobox."""
        parent = MockParentWindow()
        widget = FilterChoice(parent)
        qtbot.addWidget(widget)

        assert hasattr(widget, "combo_box")
        assert widget.combo_box.count() > 0

    def test_available_filters(self, qtbot):
        """Verify all expected filters are available."""
        parent = MockParentWindow()
        widget = FilterChoice(parent)
        qtbot.addWidget(widget)

        expected_filters = [
            "gauss_filter",
            "median_filter",
            "maximum_filter",
            "minimum_filter",
            "invert_filter",
            "dog_filter",
            "log_filter",
        ]

        available = [
            widget.combo_box.itemText(i) for i in range(widget.combo_box.count())
        ]

        for f in expected_filters:
            assert f in available, f"Filter {f} should be available"

    def test_gauss_filter_default_params(self, qtbot):
        """Verify gauss_filter has correct default parameters."""
        parent = MockParentWindow()
        widget = FilterChoice(parent)
        qtbot.addWidget(widget)

        # Select gauss_filter
        widget.combo_box.setCurrentText("gauss_filter")
        qtbot.wait(50)

        # Check default params
        assert widget.default_params["gauss_filter"]["sigma"] == 2

    def test_update_arguments_changes_fields(self, qtbot):
        """Verify update_arguments updates the argument line edits."""
        parent = MockParentWindow()
        widget = FilterChoice(parent)
        qtbot.addWidget(widget)

        # Select invert_filter
        widget.combo_box.setCurrentText("invert_filter")
        widget.update_arguments()
        qtbot.wait(50)

        # Check that argument label is updated
        assert widget.arguments_labels[0].text() == "value"
        assert widget.arguments_le[0].text() == "65535"

    def test_filter_without_params(self, qtbot):
        """Verify filters without params disable argument fields."""
        parent = MockParentWindow()
        widget = FilterChoice(parent)
        qtbot.addWidget(widget)

        # Select laplace_filter (no params)
        widget.combo_box.setCurrentText("laplace_filter")
        widget.update_arguments()
        qtbot.wait(50)

        # Argument fields should be disabled
        assert not widget.arguments_le[0].isEnabled()
        assert not widget.arguments_le[1].isEnabled()


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases for GUI utilities."""

    def test_pandas_model_with_nan_values(self, qtbot):
        """Verify PandasModel handles NaN values."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [np.nan, 2.0, np.nan],
            }
        )
        model = PandasModel(df)

        # Should display "nan" for NaN values
        index = model.index(0, 1)
        data = model.data(index, Qt.DisplayRole)
        assert data == "nan"

    def test_pandas_model_with_mixed_types(self, qtbot):
        """Verify PandasModel handles columns with mixed types."""
        df = pd.DataFrame(
            {
                "mixed": [1, "text", 3.14, None],
            }
        )
        model = PandasModel(df)

        assert model.rowCount() == 4

        # Access different rows
        assert model.data(model.index(0, 0), Qt.DisplayRole) == "1"
        assert model.data(model.index(1, 0), Qt.DisplayRole) == "text"
        assert model.data(model.index(2, 0), Qt.DisplayRole) == "3.14"

    def test_pandas_model_custom_index(self, qtbot):
        """Verify PandasModel works with custom DataFrame index."""
        df = pd.DataFrame({"value": [10, 20, 30]}, index=["row_a", "row_b", "row_c"])
        model = PandasModel(df)

        # Vertical headers should show custom index
        assert model.headerData(0, Qt.Vertical, Qt.DisplayRole) == "row_a"
        assert model.headerData(1, Qt.Vertical, Qt.DisplayRole) == "row_b"
        assert model.headerData(2, Qt.Vertical, Qt.DisplayRole) == "row_c"
