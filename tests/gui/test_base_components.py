"""
Unit tests for base GUI components in celldetective.gui.base.

Covers:
- QCheckableComboBox (components.py)
- ListWidget (list_widget.py)
- FeatureChoice (feature_choice.py)
"""

import logging
import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import Qt, QRect, QEvent, QPoint
from PyQt5.QtGui import QPainter, QPixmap, QHelpEvent
from PyQt5.QtWidgets import QMainWindow, QStyle, QStyleOptionViewItem

from celldetective.gui.base.components import (
    CheckIndicatorDelegate,
    QCheckableComboBox,
    QHSeperationLine,
    HoverButton,
)
from celldetective.gui.base.list_widget import ListWidget
from celldetective.gui.base.feature_choice import FeatureChoice


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# =============================================================================
# QCheckableComboBox Tests
# =============================================================================


class TestQCheckableComboBox:
    """Tests for QCheckableComboBox."""

    def test_initialization(self, qtbot):
        """Test basic initialization."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        assert cb.obj == "well"
        assert cb.anySelected is False
        assert cb.title() == ""

    def test_add_items(self, qtbot):
        """Test adding items populates the combo box and menu."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["well1", "well2", "well3"])

        assert cb.count() == 3
        assert len(cb.toolMenu.actions()) == 3

    def test_select_all(self, qtbot):
        """Test selectAll checks all items."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["well1", "well2", "well3"])
        cb.selectAll()

        assert cb.anySelected is True
        indices = cb.getSelectedIndices()
        assert len(indices) == 3
        assert indices == [0, 1, 2]

    def test_unselect_all(self, qtbot):
        """Test unselectAll unchecks all items."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["well1", "well2"])
        cb.selectAll()
        cb.unselectAll()

        assert cb.anySelected is False
        assert cb.getSelectedIndices() == []

    def test_get_selected_indices(self, qtbot):
        """Test getSelectedIndices returns correct indices."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="position", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["pos0", "pos1", "pos2", "pos3"])
        # Manually toggle index 1 and 3
        cb.setCurrentIndex(1)
        cb.setCurrentIndex(3)

        indices = cb.getSelectedIndices()
        assert 1 in indices
        assert 3 in indices

    def test_clear(self, qtbot):
        """Test clearing the combo box."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["a", "b", "c"])
        cb.clear()

        assert cb.count() == 0

    def test_is_single_and_multiple_selection(self, qtbot):
        """Test isSingleSelection and isMultipleSelection."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItems(["w1", "w2", "w3"])

        # Select one
        cb.setCurrentIndex(0)
        assert cb.isSingleSelection() is True
        assert cb.isMultipleSelection() is False

        # Select another (now 2 selected)
        cb.setCurrentIndex(1)
        assert cb.isMultipleSelection() is True
        assert cb.isSingleSelection() is False

    def test_add_item_with_tooltip(self, qtbot):
        """Test adding a single item with a tooltip."""
        parent = QMainWindow()
        qtbot.addWidget(parent)
        cb = QCheckableComboBox(obj="well", parent_window=parent)
        qtbot.addWidget(cb)

        cb.addItem("well_A", tooltip="This is well A")
        assert cb.count() == 1
        assert cb.itemData(0, Qt.ToolTipRole) == "This is well A"


# =============================================================================
# CheckIndicatorDelegate Tests
# =============================================================================


class TestCheckIndicatorDelegate:
    """Tests for the custom check indicator of checkable combo boxes."""

    def test_delegate_installed_on_popup(self, qtbot):
        """The popup of a QCheckableComboBox uses the custom indicator."""
        cb = QCheckableComboBox(obj="well")
        qtbot.addWidget(cb)
        cb.addItems(["well1", "well2"])

        assert isinstance(cb.view().itemDelegate(), CheckIndicatorDelegate)

    def test_rows_leave_room_for_the_indicator(self, qtbot):
        """Rows are tall and wide enough for the rounded indicator."""
        cb = QCheckableComboBox(obj="well")
        qtbot.addWidget(cb)
        cb.addItems(["well1", "well2"])

        delegate = cb.view().itemDelegate()
        option = QStyleOptionViewItem()
        option.initFrom(cb.view())
        size = delegate.sizeHint(option, cb.model().index(0, 0))

        assert size.height() >= delegate.box_size + 2 * delegate.row_padding

    def test_tooltip_only_when_text_is_cut(self, qtbot):
        """The tooltip of a row is shown only for a shortened or elided text."""
        full = "a well label far too long for the box"
        cb = QCheckableComboBox(obj="well")
        qtbot.addWidget(cb)
        # A readable row, a row shortened by the caller (the usual case in the
        # software), and a row left to the elision of the delegate.
        cb.addItem("w1", tooltip="w1")
        cb.addItem(full[:20] + "...", tooltip=full)
        cb.addItem(full, tooltip=full)

        delegate = cb.view().itemDelegate()
        option = QStyleOptionViewItem()
        option.initFrom(cb.view())
        option.rect = QRect(0, 0, 120, 27)
        event = QHelpEvent(QEvent.ToolTip, QPoint(5, 5), QPoint(5, 5))

        shown = [
            delegate.helpEvent(event, cb.view(), option, cb.model().index(row, 0))
            for row in range(3)
        ]

        assert shown == [False, True, True]

    def test_paints_every_check_state(self, qtbot):
        """Painting checked, unchecked and partially checked items works."""
        cb = QCheckableComboBox(obj="well")
        qtbot.addWidget(cb)
        cb.addItems(["well1", "well2", "well3"])
        cb.setCurrentIndex(0)
        cb.model().item(2, 0).setCheckState(Qt.PartiallyChecked)

        pixmap = QPixmap(200, 3 * 27)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        delegate = cb.view().itemDelegate()
        try:
            for row in range(3):
                option = QStyleOptionViewItem()
                option.initFrom(cb.view())
                option.rect = QRect(0, row * 27, 200, 27)
                if row == 0:
                    option.state |= QStyle.State_Selected
                delegate.paint(painter, option, cb.model().index(row, 0))
        finally:
            painter.end()


# =============================================================================
# ListWidget Tests
# =============================================================================


class TestListWidget:
    """Tests for ListWidget."""

    def test_initialization(self, qtbot):
        """Test ListWidget initializes with features."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["area", "perimeter"],
        )
        qtbot.addWidget(widget)

        assert widget.list_widget.count() == 2
        assert widget.list_widget.item(0).text() == "area"
        assert widget.list_widget.item(1).text() == "perimeter"

    def test_get_items_simple(self, qtbot):
        """Test getItems returns correct items."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["area", "perimeter"],
            dtype=str,
        )
        qtbot.addWidget(widget)

        items = widget.getItems()
        assert items == ["area", "perimeter"]

    def test_get_items_numeric(self, qtbot):
        """Test getItems casts to numeric type."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["10", "20", "30"],
            dtype=int,
        )
        qtbot.addWidget(widget)

        items = widget.getItems()
        assert items == [10, 20, 30]

    def test_get_items_range(self, qtbot):
        """Test getItems parses tuple format (min,max)."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["(10,20)"],
            dtype=float,
        )
        qtbot.addWidget(widget)

        items = widget.getItems()
        assert len(items) == 1
        assert items[0] == [10.0, 20.0]

    def test_clear(self, qtbot):
        """Test clearing all items."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["a", "b", "c"],
        )
        qtbot.addWidget(widget)

        widget.clear()
        assert widget.list_widget.count() == 0
        assert widget.items == []

    def test_remove_selected(self, qtbot):
        """Test removing selected items."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=["a", "b", "c"],
        )
        qtbot.addWidget(widget)

        # Select the second item
        widget.list_widget.setCurrentRow(1)
        widget.removeSel()

        assert widget.list_widget.count() == 2
        texts = [
            widget.list_widget.item(i).text() for i in range(widget.list_widget.count())
        ]
        assert "b" not in texts

    def test_add_item_to_list(self, qtbot):
        """Test addItemToList adds an item."""
        widget = ListWidget(
            choiceWidget=MagicMock,
            initial_features=[],
        )
        qtbot.addWidget(widget)

        widget.addItemToList("new_feature")
        assert widget.list_widget.count() == 1
        assert widget.list_widget.item(0).text() == "new_feature"


# =============================================================================
# FeatureChoice Tests
# =============================================================================


class TestFeatureChoice:
    """Tests for FeatureChoice."""

    @pytest.fixture(autouse=True)
    def restore_cache(self):
        """Restore the module-level cache after each test so other tests are not polluted."""
        import celldetective.gui.base.feature_choice as fc
        original = fc.CACHED_EXTRA_PROPERTIES
        yield
        fc.CACHED_EXTRA_PROPERTIES = original

    @patch("celldetective.gui.base.feature_choice.get_extra_properties_functions")
    def test_initialization(self, mock_extras, qtbot):
        """Test FeatureChoice populates standard measurements."""
        mock_extras.return_value = []
        # Reset cached properties
        import celldetective.gui.base.feature_choice as fc

        fc.CACHED_EXTRA_PROPERTIES = None

        parent = MagicMock()
        parent.list_widget = MagicMock()

        widget = FeatureChoice(parent_window=parent)
        qtbot.addWidget(widget)

        assert widget.windowTitle() == "Add feature"
        # Standard measurements should be present
        items = [widget.combo_box.itemText(i) for i in range(widget.combo_box.count())]
        assert "area" in items
        assert "perimeter" in items
        assert "intensity_mean" in items

    @patch("celldetective.gui.base.feature_choice.get_extra_properties_functions")
    def test_add_current_feature(self, mock_extras, qtbot):
        """Test clicking Add adds the feature to parent's list."""
        mock_extras.return_value = []
        import celldetective.gui.base.feature_choice as fc

        fc.CACHED_EXTRA_PROPERTIES = None

        parent = MagicMock()
        parent.list_widget = MagicMock()

        widget = FeatureChoice(parent_window=parent)
        qtbot.addWidget(widget)

        widget.combo_box.setCurrentIndex(0)  # "area"
        widget.add_current_feature()

        parent.list_widget.addItems.assert_called_once_with(["area"])

    @patch("celldetective.gui.base.feature_choice.get_extra_properties_functions")
    def test_extra_properties_included(self, mock_extras, qtbot):
        """Test extra properties are appended to the list."""
        mock_extras.return_value = ["custom_prop_1", "custom_prop_2"]
        import celldetective.gui.base.feature_choice as fc

        fc.CACHED_EXTRA_PROPERTIES = None

        parent = MagicMock()
        parent.list_widget = MagicMock()

        widget = FeatureChoice(parent_window=parent)
        qtbot.addWidget(widget)

        items = [widget.combo_box.itemText(i) for i in range(widget.combo_box.count())]
        assert "custom_prop_1" in items
        assert "custom_prop_2" in items


# =============================================================================
# QHSeperationLine Tests
# =============================================================================


class TestQHSeperationLine:
    """Tests for QHSeperationLine."""

    def test_initialization(self, qtbot):
        """Test separator line is created."""
        line = QHSeperationLine()
        qtbot.addWidget(line)
        assert line.maximumHeight() == 20


# =============================================================================
# HoverButton Tests
# =============================================================================


class TestHoverButton:
    """Tests for HoverButton."""

    def test_initialization(self, qtbot):
        """Test HoverButton creates with text."""
        from fonticon_mdi6 import MDI6

        btn = HoverButton(text="Test", icon_enum=MDI6.plus)
        qtbot.addWidget(btn)
        assert btn.text() == "Test"
