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
    QTabWidget,
    QToolTip,
    QAbstractButton,
    QAbstractItemView,
    QHBoxLayout,
)
from PyQt5.QtCore import (
    Qt,
    pyqtSignal,
    QEvent,
    QModelIndex,
    QObject,
    QRect,
    QRectF,
    QPointF,
    QSize,
)
from PyQt5.QtGui import QPaintEvent, QPainter, QColor, QPen, QShowEvent, QHelpEvent
from superqt.fonticon import icon
from celldetective.gui.base.styles import (
    Styles,
    button_style,
    CELLDETECTIVE_BLUE,
    DANGER_COLOR,
    DISABLED_FG,
    INK_COLOR,
    TOOL_BUTTON_SIZE,
    TOOL_ICON_SIZE,
    TOOL_IDLE_COLOR,
)
from celldetective.gui.base.app_style import draw_check_indicator
from typing import Optional

logger = logging.getLogger("celldetective")


class CelldetectiveStyledMixin(object):
    """
    Mixin applying the celldetective look to the children of a window.

    The combo box popups are the only part the style cannot reach on its own
    (Qt installs its own delegate on them), and the focus policy of the buttons
    is a property rather than a style, so both are set here, once the window is
    shown and its children exist.
    """

    def showEvent(self, event: QShowEvent) -> None:
        """Style the combo boxes of the window, then show it."""

        style_comboboxes(self)
        soften_button_focus(self)
        super().showEvent(event)


class CelldetectiveWidget(CelldetectiveStyledMixin, QWidget, Styles):
    def __init__(self, *args, **kwargs):
        """Initialize the CelldetectiveWidget."""
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveMainWindow(CelldetectiveStyledMixin, QMainWindow, Styles):
    def __init__(self, *args, **kwargs):
        """Initialize the CelldetectiveMainWindow."""
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveDialog(CelldetectiveStyledMixin, QDialog, Styles):
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


# Most of the per position actions of the control panel need one, and only one,
# position to work on: they all say so the same way.
POSITION_NEEDED = "Select a single position first."


class _DisabledReason(QObject):
    """
    Tells in the tooltip of a widget why it is disabled, while it is.

    The reason replaces the tooltip of the widget when it is disabled and gives
    it back when it is enabled again, so that the call sites keep enabling and
    disabling their buttons the way they always did.
    """

    def __init__(self, widget: QWidget, reason: str) -> None:
        """
        Initialize the filter.

        Parameters
        ----------
        widget : QWidget
            The widget to explain.
        reason : str
            Why the widget is disabled.
        """

        super().__init__(widget)

        self.widget = widget
        self.reason = reason
        self.enabled_tooltip = widget.toolTip()
        self._setting = False

        widget.installEventFilter(self)
        self._show_the_right_tooltip()

    def _show_the_right_tooltip(self) -> None:
        """Show the reason while disabled, the usual tooltip otherwise."""

        self._setting = True
        self.widget.setToolTip(
            self.enabled_tooltip if self.widget.isEnabled() else self.reason
        )
        self._setting = False

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """
        Follow the enabled state and the tooltip of the widget.

        Parameters
        ----------
        source : QObject
            The event source.
        event : QEvent
            The event.
        """

        if source is self.widget and not self._setting:
            if event.type() == QEvent.EnabledChange:
                self._show_the_right_tooltip()
            elif event.type() == QEvent.ToolTipChange and self.widget.isEnabled():
                # A call site setting its own tooltip: it is the one to give
                # back when the widget is enabled again.
                self.enabled_tooltip = self.widget.toolTip()

        return super().eventFilter(source, event)


def set_disabled_reason(widget: QWidget, reason: str) -> _DisabledReason:
    """
    Say in the tooltip of a widget why it is disabled, whenever it is.

    A disabled button says nothing about what would make it available again;
    this puts the answer where the user looks for it, and follows the widget
    from then on without the call sites having to do anything.

    Parameters
    ----------
    widget : QWidget
        The widget to explain, typically a button.
    reason : str
        Why the widget is disabled, as a sentence: "Select a position first."

    Returns
    -------
    _DisabledReason
        The filter, parented to the widget.
    """

    return _DisabledReason(widget, reason)


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


class CelldetectiveItemDelegate(QStyledItemDelegate):
    """
    Item delegate for the popups of the combo boxes and for the list widgets of
    the software.

    Compared to the delegate Qt installs on a combo box popup, it gives the
    rows some air, a rounded accent colored highlight and a thin separator,
    instead of the blocky native rendering.

    Item tooltips are only shown for the rows whose text does not fit, so that
    hovering a readable row stays quiet.
    """

    show_indicator = False

    box_size = 15
    left_margin = 9
    text_gap = 9
    right_margin = 9
    row_padding = 5
    separator_height = 7

    def __init__(
        self,
        accent: Optional[str] = CELLDETECTIVE_BLUE,
        parent: Optional[QObject] = None,
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
        self.separator_color = QColor("#E0E0E0")

    @staticmethod
    def is_separator(index: QModelIndex) -> bool:
        """
        Tell whether an index is one of the separators of a combo box.

        Parameters
        ----------
        index : QModelIndex
            The index to test.

        Returns
        -------
        bool
            True for a separator row.
        """

        return index.data(Qt.AccessibleDescriptionRole) == "separator"

    def text_offset(self) -> int:
        """Return the horizontal room taken by the indicator, if any."""

        if not self.show_indicator:
            return self.left_margin

        return self.left_margin + self.box_size + self.text_gap

    def text_rect(self, option: QStyleOptionViewItem) -> QRect:
        """Return the room left to the text of a row, once the indicator is out."""

        return option.rect.adjusted(self.text_offset(), 0, -self.right_margin, 0)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Reserve room for the indicator and give the rows some air."""

        if self.is_separator(index):
            return QSize(super().sizeHint(option, index).width(), self.separator_height)

        size = super().sizeHint(option, index)

        # The height is set from the font rather than grown from the hint of the
        # base class, which already carries the padding the style adds to every
        # list view row: adding ours on top of it would count it twice.
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        size.setHeight(
            max(opt.fontMetrics.height(), self.box_size) + 2 * self.row_padding
        )
        size.setWidth(size.width() + self.text_offset() + self.right_margin)

        return size

    def helpEvent(
        self,
        event: QHelpEvent,
        view: QAbstractItemView,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> bool:
        """
        Show the tooltip of a row only when its text is cut.

        A row is cut either because the caller shortened the text itself, the
        usual case in the software (a long model or column name inserted as
        ``name[:thresh] + "..."``, with the full name set as tooltip), or
        because the text does not fit the popup and is elided when painted.
        Everywhere else the tooltip would only repeat a label that is already
        fully readable, which is noise, so it is dropped here rather than at
        every call site setting a tooltip on an item.

        Parameters
        ----------
        event : QHelpEvent
            The help event to handle.
        view : QAbstractItemView
            The view the row belongs to.
        option : QStyleOptionViewItem
            The style options of the row.
        index : QModelIndex
            The index of the row.

        Returns
        -------
        bool
            True if the event was handled.
        """

        if event is not None and event.type() == QEvent.ToolTip and index.isValid():
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)

            tooltip = index.data(Qt.ToolTipRole)
            shortened = tooltip is not None and str(tooltip).strip() != opt.text.strip()
            elided = opt.fontMetrics.width(opt.text) > self.text_rect(opt).width()

            if not opt.text or not (shortened or elided):
                QToolTip.hideText()
                return False

        return super().helpEvent(event, view, option, index)

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        """
        Paint the row: background, optional check indicator, then the text.

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

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.is_separator(index):
            painter.setPen(QPen(self.separator_color, 1.0))
            painter.drawLine(
                QPointF(opt.rect.left() + 8, opt.rect.center().y() + 0.5),
                QPointF(opt.rect.right() - 8, opt.rect.center().y() + 0.5),
            )
            painter.restore()
            return

        selected = bool(opt.state & QStyle.State_Selected)
        hovered = bool(opt.state & QStyle.State_MouseOver)
        enabled = bool(opt.state & QStyle.State_Enabled)

        # Row background, rounded and slightly inset.
        if selected or hovered:
            painter.setPen(Qt.NoPen)
            painter.setBrush(self.accent if selected else self.hover_color)
            painter.drawRoundedRect(QRectF(opt.rect).adjusted(2.5, 1.5, -2.5, -1.5), 5, 5)

        if self.show_indicator:
            check_state = index.data(Qt.CheckStateRole)
            box = QRectF(
                opt.rect.left() + self.left_margin,
                opt.rect.center().y() - self.box_size / 2.0 + 1,
                self.box_size,
                self.box_size,
            )
            draw_check_indicator(
                painter,
                box,
                check_state=Qt.Unchecked if check_state is None else check_state,
                enabled=enabled,
                hovered=hovered,
                on_accent=selected,
                accent=self.accent.name(),
            )

        # Item text, elided to the room left by the indicator.
        text_rect = self.text_rect(opt)
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


class CheckIndicatorDelegate(CelldetectiveItemDelegate):
    """
    Item delegate drawing a rounded, celldetective-blue check indicator instead
    of the blocky native one, for item views holding checkable items (typically
    the popup of a :class:`QCheckableComboBox`).
    """

    show_indicator = True


def style_comboboxes(root: QWidget) -> None:
    """
    Install :class:`CelldetectiveItemDelegate` on the popup of every combo box
    below a widget.

    Called when a celldetective window is shown, so that the popups of the
    software are styled without touching the combo boxes of the libraries it
    embeds (napari, matplotlib).

    Parameters
    ----------
    root : QWidget
        The window whose combo boxes must be styled.
    """

    for combo in root.findChildren(QComboBox):
        if not isinstance(combo.itemDelegate(), CelldetectiveItemDelegate):
            combo.setItemDelegate(CelldetectiveItemDelegate(parent=combo))


def soften_button_focus(root: QWidget) -> None:
    """
    Let the buttons below a widget take the focus from the keyboard only.

    Qt gives a clicked button the keyboard focus, so the focus ring of the
    button styles stayed on after a click and read as a selection that is not
    one. Tab still walks the buttons, and the ring then means what it says.

    Parameters
    ----------
    root : QWidget
        The window whose buttons must be softened.
    """

    for button in root.findChildren(QAbstractButton):
        if button.focusPolicy() == Qt.StrongFocus:
            button.setFocusPolicy(Qt.TabFocus)


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


class CurrentPageTabWidget(QTabWidget):
    """
    A tab widget as tall as the page on show, not as its tallest page.

    Qt sizes a tab widget from the tallest of its pages, which leaves a band of
    nothing under every shorter one. The height of the page currently shown is
    used instead, the chrome around it (the tab bar, the frame) being left to
    Qt to measure.
    """

    def _page_heights(self) -> list:
        """Return the height asked for by each page."""

        return [self.widget(i).sizeHint().height() for i in range(self.count())]

    def sizeHint(self) -> QSize:
        """Return the size of the current page, plus the chrome around it."""

        hint = super().sizeHint()
        current = self.currentWidget()
        heights = self._page_heights()

        if current is not None and heights:
            hint.setHeight(
                hint.height() - max(heights) + current.sizeHint().height()
            )

        return hint

    def minimumSizeHint(self) -> QSize:
        """Return the smallest size the current page can take."""

        hint = super().minimumSizeHint()
        current = self.currentWidget()

        if current is None:
            return hint

        heights = [
            self.widget(i).minimumSizeHint().height() for i in range(self.count())
        ]
        if heights:
            hint.setHeight(
                hint.height() - max(heights) + current.minimumSizeHint().height()
            )

        return hint


class QHSeperationLine(QFrame):
    """
    a horizontal seperation line\n
    """

    def __init__(self) -> None:
        """Initialize the QHSeperationLine."""
        super().__init__()
        self.setMinimumWidth(1)
        # Enough air around the rule to separate two groups, not so much that
        # it opens a hole in a block.
        self.setFixedHeight(13)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)


class HoverButton(QPushButton):
    """
    A button whose icon takes the accent color while the mouse is over it.

    The defaults follow the `chip` button role these buttons wear (the model
    zoo actions): a dark icon that turns celldetective blue under the mouse,
    on the light tint the role paints behind it.
    """

    def __init__(
        self,
        text: str,
        icon_enum: str,
        default_color: Optional[str] = INK_COLOR,
        hover_color: Optional[str] = CELLDETECTIVE_BLUE,
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


class ToolButton(QPushButton):
    """
    One of the round icon buttons closing a row.

    The cogs, eyes, helpers and the like: secondary controls, so the icon rests
    in a muted blue grey and only takes the accent color under the mouse, where
    a light disc appears behind it. A disabled button fades instead, rather
    than keeping the weight of a control that cannot be used.

    The button keeps the room it takes when hidden, so that the strips of
    several rows stay lined up with one another even when a row hides one of
    its tools (the delete button of the tracking row does).
    """

    size = TOOL_BUTTON_SIZE
    icon_size = TOOL_ICON_SIZE

    def __init__(
        self,
        icon_enum: str,
        tooltip: Optional[str] = "",
        hover_color: Optional[str] = CELLDETECTIVE_BLUE,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the button.

        Parameters
        ----------
        icon_enum : str
            Icon name from MDI6.
        tooltip : str, optional
            What the button does, as a sentence.
        hover_color : str, optional
            The color the icon takes under the mouse. Destructive actions pass
            :data:`DANGER_COLOR` here.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.icon_enum = icon_enum
        self.hover_color = hover_color
        self._hovered = False

        self.setToolTip(tooltip)
        self.setFixedSize(self.size, self.size)
        self.setIconSize(QSize(self.icon_size, self.icon_size))
        self.setStyleSheet(button_style("tool"))

        policy = self.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)

        self.toggled.connect(lambda _: self._paint_icon())
        self._paint_icon()

    def _paint_icon(self) -> None:
        """Draw the icon in the color the current state calls for."""

        if not self.isEnabled():
            color = DISABLED_FG
        elif self._hovered or self.isChecked():
            # A checked button stays lit: it is the only thing marking a tool
            # that is currently on, the disc behind it being a faint tint.
            color = self.hover_color
        else:
            color = TOOL_IDLE_COLOR

        self.setIcon(icon(self.icon_enum, color=color))

    def set_icon_enum(self, icon_enum: str) -> None:
        """
        Swap the icon the button carries, keeping its colors.

        Parameters
        ----------
        icon_enum : str
            The new icon name from MDI6.
        """

        self.icon_enum = icon_enum
        self._paint_icon()

    def enterEvent(self, event: QEvent) -> None:
        """Take the accent color under the mouse."""

        self._hovered = True
        self._paint_icon()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Go back to the resting color."""

        self._hovered = False
        self._paint_icon()
        super().leaveEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        """Follow the enabled state in the color of the icon."""

        if event.type() == QEvent.EnabledChange:
            if not self.isEnabled():
                # The mouse cannot leave a disabled button, so a button
                # disabled under the cursor would stay painted as hovered.
                self._hovered = False
            self._paint_icon()

        super().changeEvent(event)


def tool_strip(*tools, spacing: Optional[int] = 2) -> "QHBoxLayout":
    """
    Lay the tools closing a row out in fixed slots.

    Every row of a panel offers the same kinds of tool in the same order, but
    not every row offers all of them. Passing ``None`` for a slot a row has
    nothing to put in leaves it empty rather than closing it up, so that the
    cogs of the rows sit in one column, the eyes in another, and the strip of
    a row reads as a group rather than as items scattered along it.

    Parameters
    ----------
    *tools : ToolButton or None
        The tools, in slot order; ``None`` for an empty slot.
    spacing : int, optional
        The room between two slots.

    Returns
    -------
    QHBoxLayout
        The strip, to add at the end of the row.
    """

    strip = QHBoxLayout()
    strip.setContentsMargins(0, 0, 0, 0)
    strip.setSpacing(spacing)

    for tool in tools:
        if tool is None:
            placeholder = QWidget()
            placeholder.setFixedSize(TOOL_BUTTON_SIZE, TOOL_BUTTON_SIZE)
            strip.addWidget(placeholder)
        else:
            strip.addWidget(tool)

    return strip
