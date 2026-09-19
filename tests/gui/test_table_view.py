"""
Unit tests for DataTableView and the display roles of PandasModel.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication

from celldetective.gui.base.table_view import DataTableView
from celldetective.gui.gui_utils import PandasModel


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def frame():
    """A small measurement table."""
    return pd.DataFrame(
        {
            "TRACK_ID": [0, 1, 2],
            "area": [318.48084366072715, np.nan, 3.0],
            "class": ["A", "B", None],
        }
    )


class TestPandasModelDisplay:
    """The way cells are written and aligned."""

    def test_floats_are_shortened_for_display(self, frame):
        model = PandasModel(frame)
        assert model.data(model.index(0, 1), Qt.DisplayRole) == "318.481"

    def test_full_value_is_kept_for_tooltip_and_copy(self, frame):
        model = PandasModel(frame)
        full = str(frame.iloc[0, 1])
        assert model.data(model.index(0, 1), Qt.ToolTipRole) == full
        assert model.data(model.index(0, 1), Qt.EditRole) == full

    def test_numbers_are_right_aligned_and_text_left_aligned(self, frame):
        model = PandasModel(frame)
        numeric = model.data(model.index(0, 1), Qt.TextAlignmentRole)
        text = model.data(model.index(0, 2), Qt.TextAlignmentRole)
        assert numeric & Qt.AlignRight
        assert text & Qt.AlignLeft

    def test_missing_values_are_muted(self, frame):
        model = PandasModel(frame)
        assert isinstance(model.data(model.index(1, 1), Qt.ForegroundRole), QBrush)
        assert model.data(model.index(0, 1), Qt.ForegroundRole) is None

    def test_text_turns_white_on_dark_cells(self, frame):
        model = PandasModel(frame)
        model.change_color(0, 0, QBrush(QColor("#a50f15")))
        model.change_color(0, 1, QBrush(QColor("#fee5d9")))
        dark = model.data(model.index(0, 0), Qt.ForegroundRole)
        assert dark.color() == QColor(Qt.white)
        assert model.data(model.index(0, 1), Qt.ForegroundRole) is None

    def test_column_name_is_the_header_tooltip(self, frame):
        model = PandasModel(frame)
        assert model.headerData(1, Qt.Horizontal, Qt.ToolTipRole) == "area"


class TestDataTableView:
    """The table view of the software."""

    def test_new_model_emits_and_fits_columns(self, qtbot, frame):
        view = DataTableView()
        qtbot.addWidget(view)
        with qtbot.waitSignal(view.model_changed, timeout=1000):
            view.setModel(PandasModel(frame))
        for column in range(frame.shape[1]):
            assert view.columnWidth(column) <= view.max_column_width

    def test_selection_follows_a_replaced_model(self, qtbot, frame):
        view = DataTableView()
        qtbot.addWidget(view)
        view.setModel(PandasModel(frame))
        view.setModel(PandasModel(frame.assign(extra=1)))
        with qtbot.waitSignal(view.selection_changed, timeout=1000):
            view.selectColumn(3)
        assert view.selected_columns() == [3]
        assert view.selected_cell_count() == len(frame)

    def test_copy_selection_writes_tab_separated_values(self, qtbot, frame):
        view = DataTableView()
        qtbot.addWidget(view)
        view.setModel(PandasModel(frame))
        view.selectColumn(1)

        text = view.copy_selection()

        lines = text.split("\n")
        assert lines[0] == "area"
        assert lines[1] == str(frame.iloc[0, 1])
        assert QApplication.clipboard().text() == text

    def test_copy_with_nothing_selected_is_empty(self, qtbot, frame):
        view = DataTableView()
        qtbot.addWidget(view)
        view.setModel(PandasModel(frame))
        assert view.copy_selection() == ""
