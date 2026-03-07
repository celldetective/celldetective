"""
Unit tests for the merge classification feature (MergeGroupWidget).

Tests cover:
- Merging two binary classification columns into a multi-label group
- Merging three binary columns
- Merging columns with multi-label (non-binary) values
- NaN propagation: if any source column is NaN, the merged column is NaN
- Single column selected → no merge performed
- No columns selected → no merge performed
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QMainWindow, QTableView

from celldetective.gui.table_ops._merge_groups import MergeGroupWidget


@pytest.fixture
def mock_parent_window():
    """Create a mock parent window with a dataframe-backed data attribute."""
    parent = MagicMock(spec=QMainWindow)
    parent.data = pd.DataFrame()
    parent.model = MagicMock()
    parent.table_view = MagicMock(spec=QTableView)
    return parent


@pytest.fixture
def binary_classification_data():
    """Sample data with two binary classification columns."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 6,
            "area": [100, 200, 150, 180, 120, 160],
            "group_spread": [0, 1, 0, 1, 0, 1],
            "group_dead": [0, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture
def three_binary_classification_data():
    """Sample data with three binary classification columns."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 8,
            "group_a": [0, 1, 0, 1, 0, 1, 0, 1],
            "group_b": [0, 0, 1, 1, 0, 0, 1, 1],
            "group_c": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def multilabel_classification_data():
    """Sample data with multi-label (non-binary) classification columns."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 6,
            "group_type": [0, 1, 2, 0, 1, 2],  # 3 classes
            "group_size": [0, 0, 1, 1, 0, 1],  # 2 classes
        }
    )


@pytest.fixture
def nan_classification_data():
    """Sample data with NaN values in classification columns."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 5,
            "group_x": [0, 1, np.nan, 1, 0],
            "group_y": [0, 0, 1, np.nan, 1],
        }
    )


def _setup_widget_and_compute(
    qtbot, parent_window, data, cols_to_merge, group_name="group_merged"
):
    """
    Create a real MergeGroupWidget, override its UI state, and trigger compute.
    """
    parent_window.data = data.copy()

    widget = MergeGroupWidget(
        parent_window, columns=cols_to_merge, n_cols_init=len(cols_to_merge)
    )
    qtbot.addWidget(widget)

    # Override name field
    widget.name_le.setText(group_name)

    # Override combo boxes to reflect the desired columns
    for i, col in enumerate(cols_to_merge):
        idx = widget.cbs[i].findText(col)
        if idx >= 0:
            widget.cbs[i].setCurrentIndex(idx)

    widget.compute()

    return parent_window


# =============================================================================
# MERGE CLASSIFICATION TESTS
# =============================================================================


class TestMergeClassification:
    """Test the merge classification logic using real Qt widgets."""

    def test_merge_two_binary_columns(
        self, qtbot, mock_parent_window, binary_classification_data
    ):
        """Merge two binary columns."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            binary_classification_data,
            ["group_spread", "group_dead"],
            "group_merged",
        )

        assert "group_merged" in parent.data.columns
        merged = parent.data["group_merged"].tolist()
        # 0*1 + 0*2 = 0; 1*1 + 0*2 = 1; 0*1 + 1*2 = 2; 1*1 + 1*2 = 3
        expected = [0, 1, 2, 3, 0, 1]
        assert merged == expected

    def test_merge_three_binary_columns(
        self, qtbot, mock_parent_window, three_binary_classification_data
    ):
        """Merge three binary columns."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            three_binary_classification_data,
            ["group_a", "group_b", "group_c"],
            "group_abc",
        )

        assert "group_abc" in parent.data.columns
        merged = parent.data["group_abc"].tolist()
        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        assert merged == expected

    def test_merge_multilabel_columns(
        self, qtbot, mock_parent_window, multilabel_classification_data
    ):
        """Merge multi-label columns."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            multilabel_classification_data,
            ["group_type", "group_size"],
            "group_multi",
        )

        assert "group_multi" in parent.data.columns
        merged = parent.data["group_multi"].tolist()
        # type(base3)*1 + size(base2)*3
        # 0,0->0; 1,0->1; 2,0->2; 0,1->3; 1,1->4; 2,1->5
        expected = [0, 1, 5, 3, 1, 5]
        assert merged == expected

    def test_nan_propagation(self, qtbot, mock_parent_window, nan_classification_data):
        """Verify NaN propagation."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            nan_classification_data,
            ["group_x", "group_y"],
            "group_nan_test",
        )

        merged = parent.data["group_nan_test"]
        assert merged.iloc[0] == 0
        assert merged.iloc[1] == 1
        assert np.isnan(merged.iloc[2])
        assert np.isnan(merged.iloc[3])
        assert merged.iloc[4] == 2

    def test_single_column_no_merge(
        self, qtbot, mock_parent_window, binary_classification_data
    ):
        """Single column selection should not trigger merge."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            binary_classification_data,
            ["group_spread"],
            "group_should_not_exist",
        )

        assert "group_should_not_exist" not in parent.data.columns

    def test_group_prefix_auto_added(
        self, qtbot, mock_parent_window, binary_classification_data
    ):
        """Verify 'group_' prefix addition."""
        parent = _setup_widget_and_compute(
            qtbot,
            mock_parent_window,
            binary_classification_data,
            ["group_spread", "group_dead"],
            "my_phenotype",  # No prefix
        )

        assert "group_my_phenotype" in parent.data.columns
