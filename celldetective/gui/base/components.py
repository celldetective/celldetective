import logging
import numpy as np
from PyQt5.QtGui import QStandardItemModel, QPalette, QFontMetrics
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QDialog,
    QMessageBox,
    QComboBox,
    QToolButton,
    QMenu,
    QStylePainter,
    QStyleOptionComboBox,
    QStyle,
    QSizePolicy,
    QProgressDialog,
    QPushButton,
    QFrame,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QEvent,
    QModelIndex,
    QObject,
    QRectF,
    QPointF,
    QSize,
)
from PyQt5.QtGui import QPaintEvent, QPainter, QColor, QPen, QPainterPath
from superqt.fonticon import icon
from celldetective.gui.base.styles import Styles, CELLDETECTIVE_BLUE
from typing import Optional

logger = logging.getLogger("celldetective")


class CelldetectiveWidget(QWidget, Styles):
    def __init__(self, *args, **kwargs):
        """Initialize the CelldetectiveWidget."""
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveMainWindow(QMainWindow, Styles):
    def __init__(self, *args, **kwargs):
        """Initialize the CelldetectiveMainWindow."""
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveDialog(QDialog, Styles):
    def __init__(self, *args, **kwargs):
        """Initialize the CelldetectiveDialog."""
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)


class CelldetectiveProgressDialog(QProgressDialog, Styles):
    def __init__(
        self,
        title: Optional[str] = "Progress",
        label_text: Optional[str] = "Processing...",
        minimum: Optional[int] = 0,
        maximum: Optional[int] = 100,
        parent: Optional[QWidget] = None,
        window_title: Optional[str] = "Progress Dialog",
    ) -> None:
        """
        Initialize the CelldetectiveProgressDialog.

        Parameters
        ----------
        title : str, optional
            The title of the dialog.
        label_text : str
            The label text.
        minimum : int, optional
            The minimum value.
        maximum : int, optional
            The maximum value.
        parent : QWidget, optional
            The parent widget.
        window_title : str, optional
            The window title.
        """
        super().__init__(
            label_text, "Cancel", minimum, maximum, parent
        )  # The super call needs to match the original parameters, not the new ones.
        self.setWindowIcon(self.celldetective_icon)
        self.setWindowTitle(window_title)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowContextHelpButtonHint
            & ~Qt.WindowCloseButtonHint
        )
        self.setMinimumDuration(0)
        self.setValue(0)

        fm = QFontMetrics(self.font())
        width = max(350, fm.horizontalAdvance(window_title) + 120)
        self.setMinimumWidth(width)


def generic_message(message: str, msg_type: Optional[str] = "info") -> None:
    """
    Show a generic message box.

    Parameters
    ----------
    message : str
        The message text.
    msg_type : str, optional
        The message type ('warning', 'info', 'critical').
    """

    logger.info(message)
    message_box = QMessageBox()
    if msg_type == "warning":
        message_box.setIcon(QMessageBox.Warning)
    elif msg_type == "info":
        message_box.setIcon(QMessageBox.Information)
    elif msg_type == "critical":
        message_box.setIcon(QMessageBox.Critical)
    message_box.setText(message)
    message_box.setWindowTitle(msg_type)
    message_box.setStandardButtons(QMessageBox.Ok)
    _ = message_box.exec()


class CheckIndicatorDelegate(QStyledItemDelegate):
    """
    Item delegate drawing a rounded, celldetective-blue check indicator instead
    of the blocky native one, for item views holding checkable items (typically
    the popup of a :class:`QCheckableComboBox`).
    """

    box_size = 15
    left_margin = 9
    text_gap = 9
    right_margin = 9
    row_padding = 6

    def __init__(
        self, accent: Optional[str] = CELLDETECTIVE_BLUE, parent: Optional[QObject] = None
    ) -> None:
        """
        Initialize the delegate.

        Parameters
        ----------
        accent : str, optional
            Color of the checked indicator and of the selected row.
        parent : QObject, optional
            The parent object.
        """

        super().__init__(parent)
        self.accent = QColor(accent)
        self.hover_color = QColor("#ECEFF1")
        self.border_color = QColor("#B8B8B8")

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Reserve room for the indicator and give the rows some air."""

        size = super().sizeHint(option, index)
        size.setHeight(
            max(size.height() + self.row_padding, self.box_size + 2 * self.row_padding)
        )
        size.setWidth(size.width() + self.left_margin + self.text_gap)

        return size

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        """
        Paint the row: background, custom check indicator, then the text.

        Parameters
        ----------
        painter : QPainter
            The painter to draw with.
        option : QStyleOptionViewItem
            The style options for the item.
        index : QModelIndex
            The index of the item to paint.
        """

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        selected = bool(opt.state & QStyle.State_Selected)
        hovered = bool(opt.state & QStyle.State_MouseOver)
        enabled = bool(opt.state & QStyle.State_Enabled)
        check_state = index.data(Qt.CheckStateRole)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Row background, rounded and slightly inset.
        if selected or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.accent if selected else self.hover_color)
            painter.drawRoundedRect(QRectF(opt.rect).adjusted(2.5, 1.5, -2.5, -1.5), 5, 5)

        box = QRectF(
            opt.rect.left() + self.left_margin,
            opt.rect.center().y() - self.box_size / 2.0 + 1,
            self.box_size,
            self.box_size,
        ).adjusted(0.5, 0.5, -0.5, -0.5)

        if check_state == Qt.Unchecked or check_state is None:
            painter.setBrush(Qt.NoBrush if selected else QColor(Qt.white))
            painter.setPen(
                QPen(QColor(255, 255, 255, 170) if selected else self.border_color, 1.4)
            )
            painter.drawRoundedRect(box, 4, 4)
        else:
            # On a selected (blue) row, invert the indicator to keep it visible.
            fill = QColor(Qt.white) if selected else self.accent
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(box, 4, 4)

            mark = QColor(self.accent if selected else Qt.white)
            if check_state == Qt.PartiallyChecked:
                painter.setPen(QPen(mark, 2.0, Qt.SolidLine, Qt.RoundCap))
                painter.drawLine(
                    QPointF(box.left() + 0.26 * box.width(), box.center().y()),
                    QPointF(box.left() + 0.74 * box.width(), box.center().y()),
                )
            else:
                path = QPainterPath()
                path.moveTo(
                    box.left() + 0.24 * box.width(), box.top() + 0.53 * box.height()
                )
                path.lineTo(
                    box.left() + 0.43 * box.width(), box.top() + 0.72 * box.height()
                )
                path.lineTo(
                    box.left() + 0.77 * box.width(), box.top() + 0.30 * box.height()
                )
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(mark, 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawPath(path)

        # Item text, elided to the room left by the indicator.
        text_rect = opt.rect.adjusted(
            self.left_margin + self.box_size + self.text_gap, 0, -self.right_margin, 0
        )
        if selected:
            text_color = opt.palette.color(QPalette.HighlightedText)
        elif enabled:
            text_color = opt.palette.color(QPalette.Text)
        else:
            text_color = opt.palette.color(QPalette.Disabled, QPalette.Text)

        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            int(Qt.AlignLeft | Qt.AlignVCenter),
            opt.fontMetrics.elidedText(opt.text, Qt.ElideRight, text_rect.width()),
        )

        painter.restore()


class QCheckableComboBox(QComboBox):
    """
    adapted from https://stackoverflow.com/questions/22775095/pyqt-how-to-set-combobox-items-be-checkable
    """

    activated = pyqtSignal(str)

    def __init__(
        self,
        obj: Optional[str] = None,
        parent_window: Optional[QMainWindow] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the QCheckableComboBox.

        Parameters
        ----------
        obj : str, optional
            Object name for display.
        parent_window : QMainWindow, optional
            The parent window.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(parent_window, *args, **kwargs)

        self.setTitle("")
        self.setModel(QStandardItemModel(self))
        self.obj = obj
        self.toolButton = QToolButton(parent_window)
        self.toolButton.setText("")
        self.toolMenu = QMenu(parent_window)
        self.toolButton.setMenu(self.toolMenu)
        self.toolButton.setPopupMode(QToolButton.InstantPopup)
        self.anySelected = False

        self.setItemDelegate(CheckIndicatorDelegate(parent=self))
        self.view().setMouseTracking(True)
        self.view().viewport().installEventFilter(self)
        self.view().pressed.connect(self.handleItemPressed)

    def clear(self) -> None:
        """Clear the combo box and uncheck all items."""

        self.unselectAll()
        self.toolMenu.clear()
        super().clear()

    def handleItemPressed(self, index: QModelIndex) -> None:
        """
        Handle item press events to toggle check state.

        Parameters
        ----------
        index : QModelIndex
            The index of the item pressed.
        """

        idx = index.row()
        actions = self.toolMenu.actions()

        item = self.model().itemFromIndex(index)
        if item is None:
            return
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
            actions[idx].setChecked(False)
        else:
            item.setCheckState(Qt.Checked)
            actions[idx].setChecked(True)
            self.anySelected = True

        options_checked = np.array([a.isChecked() for a in actions])
        if len(options_checked[options_checked]) > 1:
            self.setTitle(f'Multiple {self.obj+"s"} selected...')
        elif len(options_checked[options_checked]) == 1:
            idx_selected = np.where(options_checked)[0][0]
            if idx_selected != idx:
                item = self.model().item(idx_selected)
            self.setTitle(item.text())
        elif len(options_checked[options_checked]) == 0:
            self.setTitle(f"No {self.obj} selected...")
            self.anySelected = False

        self.activated.emit(self.title())

    def setCurrentIndex(self, index: int) -> None:
        """
        Set the current index and toggle its state.

        Parameters
        ----------
        index : int
            The index to set.
        """

        super().setCurrentIndex(index)

        item = self.model().item(index)
        modelIndex = self.model().indexFromItem(item)

        self.handleItemPressed(modelIndex)

    def selectAll(self) -> None:
        """Select all items."""

        actions = self.toolMenu.actions()
        for i, a in enumerate(actions):
            if not a.isChecked():
                self.setCurrentIndex(i)
        self.anySelected = True

    def unselectAll(self) -> None:
        """Unselect all items."""

        actions = self.toolMenu.actions()
        for i, a in enumerate(actions):
            if a.isChecked():
                self.setCurrentIndex(i)
        self.anySelected = False

    def title(self) -> str:
        """Return the current title."""
        return self._title

    def setTitle(self, title: str) -> None:
        """
        Set the title of the combo box.

        Parameters
        ----------
        title : str
            The new title.
        """
        self._title = title
        self.update()
        self.repaint()

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paint the combo box.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """

        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.Text))
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = self._title
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)

    def addItem(self, item: str, tooltip: Optional[str] = None) -> None:
        """
        Add an item to the combo box.

        Parameters
        ----------
        item : str
            The item text.
        tooltip : str, optional
            The tooltip for the item.
        """

        super().addItem(item)
        idx = self.findText(item)
        if tooltip is not None:
            self.setItemData(idx, tooltip, Qt.ToolTipRole)
        item2 = self.model().item(idx, 0)
        item2.setCheckState(Qt.Unchecked)
        action = self.toolMenu.addAction(item)
        action.setCheckable(True)

    def addItems(self, items: list[str]) -> None:
        """
        Add multiple items to the combo box.

        Parameters
        ----------
        items : list of str
            The items to add.
        """

        super().addItems(items)

        for item in items:

            idx = self.findText(item)
            item2 = self.model().item(idx, 0)
            item2.setCheckState(Qt.Unchecked)
            action = self.toolMenu.addAction(item)
            action.setCheckable(True)

    def getSelectedIndices(self) -> list[int]:
        """Return the indices of selected items."""

        actions = self.toolMenu.actions()
        options_checked = np.array([a.isChecked() for a in actions])
        idx_selected = np.where(options_checked)[0]

        return list(idx_selected)

    def currentText(self) -> str:
        """Return the current text."""
        return self.title()

    def isMultipleSelection(self) -> bool:
        """Check if multiple items are selected."""
        return self.currentText().startswith("Multiple")

    def isSingleSelection(self) -> bool:
        """Check if a single item is selected."""
        return not self.currentText().startswith(
            "Multiple"
        ) and not self.title().startswith("No")

    def isAnySelected(self) -> bool:
        """Check if any item is selected."""
        return not self.title().startswith("No")

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """
        Filter events to keep the popup open on click.

        Parameters
        ----------
        source : QObject
            The event source.
        event : QEvent
            The event.
        """
        if source is self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                return True  # Prevent the popup from closing
        return super().eventFilter(source, event)


class QHSeperationLine(QFrame):
    """
    a horizontal seperation line\n
    """

    def __init__(self) -> None:
        """Initialize the QHSeperationLine."""
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)


class HoverButton(QPushButton):
    def __init__(
        self,
        text: str,
        icon_enum: str,
        default_color: Optional[str] = "gray",
        hover_color: Optional[str] = "white",
    ) -> None:
        """
        Initialize the HoverButton.

        Parameters
        ----------
        text : str
            Button text.
        icon_enum : str
            Icon name from MDI6.
        default_color : str, optional
            Default icon color.
        hover_color : str, optional
            Hover icon color.
        """
        super().__init__(text)
        self.icon_enum = icon_enum
        self.default_color = default_color
        self.hover_color = hover_color
        self.setIcon(icon(self.icon_enum, color=self.default_color))

    def enterEvent(self, event: QEvent) -> None:
        """
        Change icon color on hover enter.

        Parameters
        ----------
        event : QEvent
            The enter event.
        """
        self.setIcon(icon(self.icon_enum, color=self.hover_color))
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """
        Revert icon color on hover leave.

        Parameters
        ----------
        event : QEvent
            The leave event.
        """
        self.setIcon(icon(self.icon_enum, color=self.default_color))
        super().leaveEvent(event)
