"""
Unit tests for ClassifierWidget and query classification logic.
Tests the classify_cells_from_query function with various data types and query patterns.
"""

import pytest
import pandas as pd
import numpy as np
import logging

from celldetective.measure import classify_cells_from_query
from celldetective.exceptions import EmptyQueryError, MissingColumnsError, QueryError
from celldetective.gui.classifier_widget import ClassifierWidget


class MockParentChain:
    """
    Mock parent object chain for ClassifierWidget.
    ClassifierWidget needs: parent_window.parent_window.parent_window.screen_height/width/button_select_all
    and parent_window.mode, parent_window.df
    """

    def __init__(self, df, mode="targets"):
        self.df = df
        self.mode = mode
        # Create nested parent chain
        self.parent_window = self._create_parent()

    def _create_parent(self):
        class GrandParent:
            screen_height = 1080
            screen_width = 1920
            button_select_all = ""

        class Parent:
            parent_window = GrandParent()

        return Parent()


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
def sample_data():
    """Create sample DataFrame with various data types for testing queries."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame(
        {
            "FRAME": list(range(10)) * 2,
            "TRACK_ID": [1.0] * 10 + [2.0] * 10,
            "area": np.random.uniform(100, 500, n),
            "intensity": np.random.uniform(10, 100, n),
            "d/dt.area": np.random.uniform(-1.0, 1.0, n),
            "well": ["W1"] * 10 + ["W2"] * 10,
            "label": ["A", "B"] * 10,
            "category": pd.Categorical(["cat1", "cat2"] * 10),
            "col with space": np.random.uniform(1, 10, n),
        }
    )


@pytest.fixture
def data_with_nans():
    """Create sample DataFrame with NaN values for edge case testing."""
    return pd.DataFrame(
        {
            "FRAME": [0, 1, 2, 3, 4],
            "TRACK_ID": [1.0, 1.0, 1.0, 2.0, 2.0],
            "area": [100.0, np.nan, 300.0, 400.0, np.nan],
            "intensity": [50.0, 60.0, np.nan, 80.0, 90.0],
        }
    )


class TestNumericColumnDetection:
    """Test numeric column detection with various dtypes."""

    def test_select_numeric_columns_basic(self, sample_data):
        """Test that select_dtypes correctly identifies numeric columns."""
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()

        # These should be numeric
        assert "FRAME" in numeric_cols
        assert "TRACK_ID" in numeric_cols
        assert "area" in numeric_cols
        assert "intensity" in numeric_cols
        assert "d/dt.area" in numeric_cols
        assert "col with space" in numeric_cols

        # These should NOT be numeric
        assert "well" not in numeric_cols
        assert "label" not in numeric_cols
        assert "category" not in numeric_cols

    def test_select_numeric_columns_with_extension_dtypes(self):
        """Test numeric detection with pandas extension dtypes like StringDtype."""
        df = pd.DataFrame(
            {
                "numeric_int": [1, 2, 3],
                "numeric_float": [1.0, 2.0, 3.0],
                "string_dtype": pd.array(["a", "b", "c"], dtype="string"),
                "object_str": ["x", "y", "z"],
            }
        )
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        assert "numeric_int" in numeric_cols
        assert "numeric_float" in numeric_cols
        assert "string_dtype" not in numeric_cols
        assert "object_str" not in numeric_cols


class TestSimpleNumericQueries:
    """Test simple numeric comparison queries."""

    def test_greater_than(self, sample_data):
        """Test area > 300 query."""
        result = classify_cells_from_query(sample_data, "test_class", "area > 300")

        assert "status_test_class" in result.columns
        # Verify classification is correct
        expected_matches = sample_data["area"] > 300
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_less_than(self, sample_data):
        """Test intensity < 50 query."""
        result = classify_cells_from_query(sample_data, "test_class", "intensity < 50")

        expected_matches = sample_data["intensity"] < 50
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_numeric_equality_track_id(self, sample_data):
        """Test TRACK_ID == 1 query."""
        result = classify_cells_from_query(sample_data, "test_class", "TRACK_ID == 1")

        expected_matches = sample_data["TRACK_ID"] == 1
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_numeric_equality_frame(self, sample_data):
        """Test FRAME == 0 query."""
        result = classify_cells_from_query(sample_data, "test_class", "FRAME == 0")

        expected_matches = sample_data["FRAME"] == 0
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()


class TestLogicalOperators:
    """Test queries with logical operators."""

    def test_and_operator(self, sample_data):
        """Test area > 200 and intensity < 80 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "area > 200 and intensity < 80"
        )

        expected_matches = (sample_data["area"] > 200) & (sample_data["intensity"] < 80)
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_or_operator(self, sample_data):
        """Test TRACK_ID == 1 or FRAME == 0 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "TRACK_ID == 1 or FRAME == 0"
        )

        expected_matches = (sample_data["TRACK_ID"] == 1) | (sample_data["FRAME"] == 0)
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_multiple_and_conditions(self, sample_data):
        """Test area > 200 and intensity < 80 and FRAME > 2 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "area > 200 and intensity < 80 and FRAME > 2"
        )

        expected_matches = (
            (sample_data["area"] > 200)
            & (sample_data["intensity"] < 80)
            & (sample_data["FRAME"] > 2)
        )
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_multiple_or_conditions(self, sample_data):
        """Test TRACK_ID == 1 or FRAME == 0 or area > 400 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "TRACK_ID == 1 or FRAME == 0 or area > 400"
        )

        expected_matches = (
            (sample_data["TRACK_ID"] == 1)
            | (sample_data["FRAME"] == 0)
            | (sample_data["area"] > 400)
        )
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_mixed_and_or_conditions(self, sample_data):
        """Test (area > 200 and intensity < 80) or FRAME == 0 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "(area > 200 and intensity < 80) or FRAME == 0"
        )

        expected_matches = (
            (sample_data["area"] > 200) & (sample_data["intensity"] < 80)
        ) | (sample_data["FRAME"] == 0)
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()


class TestSpecialColumnNames:
    """Test queries with special column names requiring backticks."""

    def test_derivative_column(self, sample_data):
        """Test `d/dt.area` > 0 query with special characters."""
        result = classify_cells_from_query(sample_data, "test_class", "`d/dt.area` > 0")

        expected_matches = sample_data["d/dt.area"] > 0
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_column_with_space(self, sample_data):
        """Test `col with space` > 5 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", "`col with space` > 5"
        )

        expected_matches = sample_data["col with space"] > 5
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()


class TestStringQueries:
    """Test string equality queries."""

    def test_string_equality(self, sample_data):
        """Test well == 'W1' query."""
        result = classify_cells_from_query(sample_data, "test_class", 'well == "W1"')

        expected_matches = sample_data["well"] == "W1"
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_string_inequality(self, sample_data):
        """Test label != 'A' query."""
        result = classify_cells_from_query(sample_data, "test_class", 'label != "A"')

        expected_matches = sample_data["label"] != "A"
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()

    def test_string_and_numeric_combined(self, sample_data):
        """Test well == 'W1' and area > 300 query."""
        result = classify_cells_from_query(
            sample_data, "test_class", 'well == "W1" and area > 300'
        )

        expected_matches = (sample_data["well"] == "W1") & (sample_data["area"] > 300)
        assert (result.loc[expected_matches, "status_test_class"] == 1).all()
        assert (result.loc[~expected_matches, "status_test_class"] == 0).all()


class TestNaNHandling:
    """Test NaN handling in queries."""

    def test_nan_values_set_to_nan(self, data_with_nans):
        """Test that rows with NaN in query columns get NaN classification."""
        result = classify_cells_from_query(data_with_nans, "test_class", "area > 200")

        # Rows with NaN in 'area' should have NaN classification
        nan_rows = data_with_nans["area"].isna()
        assert result.loc[nan_rows, "status_test_class"].isna().all()

        # Non-NaN rows should be classified normally
        non_nan_rows = ~nan_rows
        expected_matches = data_with_nans.loc[non_nan_rows, "area"] > 200
        assert (
            result.loc[non_nan_rows & expected_matches, "status_test_class"] == 1
        ).all()

    def test_multiple_columns_with_nans(self, data_with_nans):
        """Test query involving multiple columns where either has NaN."""
        result = classify_cells_from_query(
            data_with_nans, "test_class", "area > 200 and intensity > 50"
        )

        # Rows with NaN in either column should have NaN classification
        nan_rows = data_with_nans["area"].isna() | data_with_nans["intensity"].isna()
        assert result.loc[nan_rows, "status_test_class"].isna().all()


class TestErrorHandling:
    """Test error handling for invalid queries."""

    def test_empty_query_raises_error(self, sample_data):
        """Test that empty query raises EmptyQueryError."""
        with pytest.raises(EmptyQueryError):
            classify_cells_from_query(sample_data, "test_class", "")

    def test_whitespace_query_raises_error(self, sample_data):
        """Test that whitespace-only query raises EmptyQueryError."""
        with pytest.raises(EmptyQueryError):
            classify_cells_from_query(sample_data, "test_class", "   ")

    def test_missing_column_raises_error(self, sample_data):
        """Test that query with non-existent column raises MissingColumnsError."""
        with pytest.raises(MissingColumnsError):
            classify_cells_from_query(
                sample_data, "test_class", "nonexistent_column > 5"
            )

    def test_invalid_syntax_raises_error(self, sample_data):
        """Test that invalid query syntax raises QueryError."""
        with pytest.raises(QueryError):
            classify_cells_from_query(sample_data, "test_class", "area >> 5")


class TestStatusColumnNaming:
    """Test that status column naming works correctly."""

    def test_status_prefix_added(self, sample_data):
        """Test that 'status_' prefix is added to class name."""
        result = classify_cells_from_query(sample_data, "my_class", "area > 300")
        assert "status_my_class" in result.columns

    def test_status_prefix_not_duplicated(self, sample_data):
        """Test that 'status_' prefix is not duplicated if already present."""
        result = classify_cells_from_query(sample_data, "status_my_class", "area > 300")
        assert "status_my_class" in result.columns
        assert "status_status_my_class" not in result.columns


class TestClassifierWidgetUI:
    """
    UI-level tests for ClassifierWidget.
    These tests instantiate the real widget and interact with it via qtbot.
    """

    @pytest.fixture
    def widget_data(self):
        """Create sample DataFrame for widget testing."""
        np.random.seed(42)
        n = 20
        return pd.DataFrame(
            {
                "FRAME": list(range(10)) * 2,
                "TRACK_ID": [1.0] * 10 + [2.0] * 10,
                "POSITION_X": np.random.uniform(0, 100, n),
                "POSITION_Y": np.random.uniform(0, 100, n),
                "area": np.random.uniform(100, 500, n),
                "intensity": np.random.uniform(10, 100, n),
                "d/dt.area": np.random.uniform(-1.0, 1.0, n),
                "well": ["W1"] * 10 + ["W2"] * 10,
            }
        )

    def test_widget_instantiation(self, qtbot, widget_data):
        """Test that ClassifierWidget can be instantiated with mock parent."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)

        assert widget is not None
        assert widget.df is widget_data
        assert widget.mode == "targets"

        widget.close()

    def test_query_line_edit_exists(self, qtbot, widget_data):
        """Test that the query line edit exists and is accessible."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "property_query_le")
        assert widget.property_query_le is not None

        widget.close()

    def test_submit_button_disabled_initially(self, qtbot, widget_data):
        """Test that submit button is disabled when query is empty."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "submit_query_btn")
        assert not widget.submit_query_btn.isEnabled()

        widget.close()

    def test_submit_button_enabled_after_typing_query(self, qtbot, widget_data):
        """Test that submit button is enabled after typing a query."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Type a query
        widget.property_query_le.setText("area > 200")
        qtbot.wait(50)

        assert widget.submit_query_btn.isEnabled()

        widget.close()

    def test_numeric_columns_detected(self, qtbot, widget_data):
        """Test that numeric columns are correctly detected."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)

        # Check numeric columns
        assert "area" in widget.cols
        assert "intensity" in widget.cols
        assert "FRAME" in widget.cols
        assert "TRACK_ID" in widget.cols
        assert "d/dt.area" in widget.cols

        # Non-numeric should not be in cols
        assert "well" not in widget.cols

        widget.close()

    def test_apply_query_simple_numeric(self, qtbot, widget_data):
        """Test applying a simple numeric query via UI."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Set query and apply
        widget.property_query_le.setText("area > 300")
        widget.name_le.setText("high_area")
        qtbot.wait(50)

        # Click the submit button
        widget.submit_query_btn.click()
        qtbot.wait(100)

        # Check that classification was applied
        assert "status_high_area" in widget.df.columns

        widget.close()

    def test_apply_query_with_and_condition(self, qtbot, widget_data):
        """Test applying a query with AND condition via UI."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        widget.property_query_le.setText("area > 200 and intensity < 80")
        widget.name_le.setText("filtered")
        qtbot.wait(50)

        widget.submit_query_btn.click()
        qtbot.wait(100)

        assert "status_filtered" in widget.df.columns

        widget.close()

    def test_apply_query_with_or_condition(self, qtbot, widget_data):
        """Test applying a query with OR condition via UI."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        widget.property_query_le.setText("TRACK_ID == 1 or FRAME == 0")
        widget.name_le.setText("selected")
        qtbot.wait(50)

        widget.submit_query_btn.click()
        qtbot.wait(100)

        assert "status_selected" in widget.df.columns

        widget.close()

    def test_apply_query_frame_equality(self, qtbot, widget_data):
        """Test applying FRAME == 0 query via UI."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        widget.property_query_le.setText("FRAME == 0")
        widget.name_le.setText("first_frame")
        qtbot.wait(50)

        widget.submit_query_btn.click()
        qtbot.wait(100)

        assert "status_first_frame" in widget.df.columns

        widget.close()

    def test_apply_query_derivative_column(self, qtbot, widget_data):
        """Test applying query with derivative-style column name via UI."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        widget.property_query_le.setText("`d/dt.area` > 0")
        widget.name_le.setText("increasing")
        qtbot.wait(50)

        widget.submit_query_btn.click()
        qtbot.wait(100)

        assert "status_increasing" in widget.df.columns

        widget.close()

    def test_project_times_btn_click(self, qtbot, widget_data):
        """Test clicking the project times button toggles projection mode."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Initially projection should be off
        assert widget.project_times is False

        # Click the button
        widget.project_times_btn.click()
        qtbot.wait(50)

        # Projection mode should be toggled
        assert widget.project_times is True

        # Click again to toggle back
        widget.project_times_btn.click()
        qtbot.wait(50)

        assert widget.project_times is False

        widget.close()

    def test_log_button_click(self, qtbot, widget_data):
        """Test clicking the log buttons for features."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Check log buttons exist
        assert hasattr(widget, "log_btns")
        assert len(widget.log_btns) == 2

        # Click each log button
        for i, btn in enumerate(widget.log_btns):
            btn.click()
            qtbot.wait(50)

        widget.close()

    def test_feature_combo_boxes(self, qtbot, widget_data):
        """Test feature combo boxes contain correct columns and can be changed."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "features_cb")
        assert len(widget.features_cb) == 2

        # Each combo box should have numeric columns as items
        for cb in widget.features_cb:
            assert cb.count() > 0
            # Try changing the selection
            if cb.count() > 1:
                cb.setCurrentIndex(1)
                qtbot.wait(50)

        widget.close()

    def test_frame_slider(self, qtbot, widget_data):
        """Test frame slider can be moved."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "frame_slider")

        # Set slider value
        widget.frame_slider.setValue(5)
        qtbot.wait(50)

        assert widget.currentFrame == 5

        widget.close()

    def test_alpha_slider(self, qtbot, widget_data):
        """Test transparency (alpha) slider can be moved."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "alpha_slider")
        assert hasattr(widget, "currentAlpha")

        # Set slider value
        widget.alpha_slider.setValue(0.5)
        qtbot.wait(50)

        assert widget.currentAlpha == pytest.approx(0.5, abs=0.01)

        widget.close()

    def test_time_correlated_checkbox_exists(self, qtbot, widget_data):
        """Test that time correlated checkbox exists and is enabled when TRACK_ID present."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "time_corr")
        # Should be enabled because TRACK_ID is in the data
        assert widget.time_corr.isEnabled()

        widget.close()

    def test_time_correlated_checkbox_disabled_without_track_id(self, qtbot):
        """Test that time correlated checkbox is disabled when TRACK_ID is missing."""
        np.random.seed(42)
        n = 10
        data_no_track = pd.DataFrame(
            {
                "FRAME": list(range(10)),
                "area": np.random.uniform(100, 500, n),
                "intensity": np.random.uniform(10, 100, n),
            }
        )

        parent = MockParentChain(data_no_track, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Should be disabled because TRACK_ID is not in the data
        assert not widget.time_corr.isEnabled()

        widget.close()

    def test_time_corr_enables_radio_buttons(self, qtbot, widget_data):
        """Test that checking time_corr checkbox enables the radio buttons."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Initially radio buttons should be disabled
        assert not widget.irreversible_event_btn.isEnabled()
        assert not widget.unique_state_btn.isEnabled()
        assert not widget.transient_event_btn.isEnabled()

        # Check the time_corr checkbox
        widget.time_corr.setChecked(True)
        qtbot.wait(50)

        # Now radio buttons should be enabled
        assert widget.irreversible_event_btn.isEnabled()
        assert widget.unique_state_btn.isEnabled()
        assert widget.transient_event_btn.isEnabled()

        widget.close()

    def test_radio_buttons_are_exclusive(self, qtbot, widget_data):
        """Test that radio buttons are mutually exclusive."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        # Enable time corr first
        widget.time_corr.setChecked(True)
        qtbot.wait(50)

        # unique_state_btn is initially checked
        assert widget.unique_state_btn.isChecked()

        # Click irreversible_event_btn
        widget.irreversible_event_btn.click()
        qtbot.wait(50)

        assert widget.irreversible_event_btn.isChecked()
        assert not widget.unique_state_btn.isChecked()
        assert not widget.transient_event_btn.isChecked()

        # Click transient_event_btn
        widget.transient_event_btn.click()
        qtbot.wait(50)

        assert widget.transient_event_btn.isChecked()
        assert not widget.irreversible_event_btn.isChecked()
        assert not widget.unique_state_btn.isChecked()

        widget.close()

    def test_name_line_edit(self, qtbot, widget_data):
        """Test that class name line edit can be changed."""
        parent = MockParentChain(widget_data, mode="targets")
        widget = ClassifierWidget(parent)
        qtbot.addWidget(widget)
        widget.show()
        qtbot.wait(100)

        assert hasattr(widget, "name_le")
        assert widget.name_le.text() == "custom"

        widget.name_le.setText("my_new_class")
        qtbot.wait(50)

        assert widget.name_le.text() == "my_new_class"

        widget.close()
