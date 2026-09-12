"""
Collapsible blocks.

The panels of the control panel (preprocessing, populations, interactions) and
the sections of the settings windows are cards the user opens one at a time.
This module holds the pieces they are built from: :class:`ChevronButton`, the
only click target of a block, :class:`CollapsibleHeader`, the band carrying the
centered title and whatever buttons belong to the section, and
:class:`CollapsibleFrame`, the card wiring that header to a content widget and
animating it open and closed.

An open card is marked by a tint and a celldetective blue edge, so that an open
block reads as open without having to look at the direction of the chevron.
"""

import logging
from typing import Optional

from PyQt5.QtCore import (
    QEasingCurve,
    QEvent,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from celldetective.gui.base.app_style import draw_chevron
from celldetective.gui.base.styles import CELLDETECTIVE_BLUE

logger = logging.getLogger("celldetective")

CARD_BACKGROUND = "#FFFFFF"
CARD_BORDER = "#E3E7EB"
CARD_BORDER_OPEN = "#C9DCF3"
HEADER_HOVER = "#F2F5F8"
HEADER_OPEN = "#F4F8FD"
TITLE_COLOR = "#000000"

# Qt's own "no maximum", used to release the height of a content once it is
# fully open.
UNCONSTRAINED = 16777215


class ChevronButton(QAbstractButton):
    """
    The small chevron opening and closing a block.

    Checked means the block is open. The chevron turns between the two states
    instead of being swapped for another icon, and it is the only click target
    of the header.
    """

    button_size = 26
    chevron_size = 16
    duration = 140

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the button.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(self.button_size, self.button_size))
        self.setAttribute(Qt.WA_Hover, True)
        self.color = QColor(TITLE_COLOR)

        self._angle = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(self.duration)
        self._animation.valueChanged.connect(self._set_angle)
        self.toggled.connect(self._turn)

    def sizeHint(self) -> QSize:
        """Return the fixed size of the button."""

        return QSize(self.button_size, self.button_size)

    def set_color(self, color: QColor) -> None:
        """
        Set the color the chevron is drawn with.

        Parameters
        ----------
        color : QColor
            The new color.
        """

        self.color = QColor(color)
        self.update()

    def _set_angle(self, angle: float) -> None:
        """Store the angle of the chevron and repaint."""

        self._angle = float(angle)
        self.update()

    def _turn(self, checked: bool) -> None:
        """Turn the chevron towards its open or closed position."""

        self._animation.stop()
        target = 180.0 if checked else 0.0

        if not self.isVisible():
            # A block opened while its window is still being built has nothing
            # to animate: the chevron is simply drawn in its final position.
            self._set_angle(target)
            return

        self._animation.setStartValue(self._angle)
        self._animation.setEndValue(target)
        self._animation.start()

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paint the chevron, on a round highlight when hovered.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect())

        if self.underMouse() and self.isEnabled():
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(HEADER_HOVER))
            painter.drawEllipse(rect.adjusted(1, 1, -1, -1))

        chevron = QRectF(
            rect.center().x() - self.chevron_size / 2.0,
            rect.center().y() - self.chevron_size / 2.0,
            self.chevron_size,
            self.chevron_size,
        )
        painter.translate(chevron.center())
        painter.rotate(self._angle)
        painter.translate(-chevron.center())
        draw_chevron(
            painter, chevron, color=self.color.name(), enabled=self.isEnabled()
        )

    def hideEvent(self, event: QEvent) -> None:
        """Stop turning when the button is hidden, and face the right way."""

        if self._animation.state() == QVariantAnimation.Running:
            self._animation.stop()
            self._set_angle(180.0 if self.isChecked() else 0.0)

        super().hideEvent(event)

    def event(self, event: QEvent) -> bool:
        """Repaint on hover, so that the highlight follows the mouse."""

        if event.type() in (QEvent.HoverEnter, QEvent.HoverLeave):
            self.update()

        return super().event(event)


class CollapsibleHeader(QWidget):
    """
    Title band of a :class:`CollapsibleFrame`.

    The band paints the title, centered, and the accent edge marking an open
    block. It is not a click target: the block is opened and closed with the
    chevron on its right, and the buttons of the section sit in the band on
    either side of the title.
    """

    header_height = 38
    side_margin = 8
    accent_width = 3
    radius = 7

    def __init__(
        self, title: Optional[str] = "", parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the header.

        Parameters
        ----------
        title : str, optional
            The title of the block.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.setFixedHeight(self.header_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._title = title
        self.toggle_btn = ChevronButton(parent=self)

        self.box = QHBoxLayout(self)
        self.box.setContentsMargins(self.side_margin, 0, self.side_margin, 0)
        self.box.setSpacing(4)
        self.box.addStretch(1)
        self.box.addWidget(self.toggle_btn)

        self.toggle_btn.toggled.connect(self._follow_state)
        self._follow_state(False)

    def text(self) -> str:
        """Return the title of the block."""

        return self._title

    def setText(self, title: str) -> None:
        """
        Set the title of the block.

        Parameters
        ----------
        title : str
            The new title.
        """

        self._title = title
        self.update()

    def add_widget(self, widget: QWidget, leading: Optional[bool] = False) -> None:
        """
        Add a button of the section to the band.

        Parameters
        ----------
        widget : QWidget
            The widget to add, typically a help or a select all button.
        leading : bool, optional
            True to place it on the left of the band, False (default) to place
            it on the right, before the chevron.
        """

        if leading:
            self.box.insertWidget(0, widget)
        else:
            self.box.insertWidget(self.box.count() - 1, widget)

    def is_expanded(self) -> bool:
        """Tell whether the block is open."""

        return self.toggle_btn.isChecked()

    def title_color(self) -> QColor:
        """Return the color of the title and of the chevron."""

        if not self.isEnabled():
            return QColor(TITLE_COLOR).lighter(150)

        if self.is_expanded():
            return QColor(CELLDETECTIVE_BLUE)

        return QColor(TITLE_COLOR)

    def _follow_state(self, expanded: bool) -> None:
        """Repaint the band and recolor the chevron on a state change."""

        self.toggle_btn.set_color(self.title_color())
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paint the band, its accent edge and the title.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(self.rect())
        opened = self.is_expanded()

        if opened:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(HEADER_OPEN))
            painter.drawRoundedRect(
                rect.adjusted(0.5, 0.5, -0.5, -0.5), self.radius, self.radius
            )
            # The celldetective blue edge of an open card, in the spirit of the
            # tooltips and of the outlined buttons.
            painter.setBrush(QColor(CELLDETECTIVE_BLUE))
            painter.drawRoundedRect(
                QRectF(
                    rect.left() + 1,
                    rect.top() + 7,
                    self.accent_width,
                    rect.height() - 14,
                ),
                self.accent_width / 2.0,
                self.accent_width / 2.0,
            )

        font = QFont(self.font())
        font.setBold(True)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
        painter.setFont(font)
        painter.setPen(self.title_color())
        painter.drawText(rect, int(Qt.AlignCenter), self._title)

    def changeEvent(self, event: QEvent) -> None:
        """Follow the enabled state in the color of the title."""

        if event.type() == QEvent.EnabledChange:
            self._follow_state(self.is_expanded())

        super().changeEvent(event)


class CollapsibleFrame(QFrame):
    """
    Card holding a title band and the content it shows or hides.

    The content is given with :meth:`set_content` and animated open and closed.
    ``collapse_btn`` is an alias of the header, kept because the panels and the
    tests of the software drive the blocks through it.

    Attributes
    ----------
    header : CollapsibleHeader
        The band opening and closing the card.
    animation_finished : pyqtSignal
        Emitted with the new state once the card has finished opening or
        closing, for the callers that resize a window around it.
    """

    animation_finished = pyqtSignal(bool)

    duration = 150
    radius = 8

    def __init__(
        self, title: Optional[str] = "", parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize the card.

        Parameters
        ----------
        title : str, optional
            The title shown in the band.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.setFrameStyle(QFrame.NoFrame)

        self.header = CollapsibleHeader(title, parent=self)
        self.collapse_btn = self.header.toggle_btn
        self.content = None
        self.animation = None
        self._animate_next = True

        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(1, 1, 1, 1)
        self.box.setSpacing(0)
        self.box.addWidget(self.header)

        self.collapse_btn.toggled.connect(self._toggle_content)

    @property
    def toggled(self) -> pyqtSignal:
        """The signal emitted when the block is opened or closed."""

        return self.collapse_btn.toggled

    def set_title(self, title: str) -> None:
        """
        Set the title of the block.

        Parameters
        ----------
        title : str
            The new title.
        """

        self.header.setText(title)

    def add_header_widget(
        self, widget: QWidget, leading: Optional[bool] = False
    ) -> None:
        """
        Add a button of the section to the band.

        Parameters
        ----------
        widget : QWidget
            The widget to add.
        leading : bool, optional
            True to place it before the title.
        """

        self.header.add_widget(widget, leading=leading)

    def set_content(self, content: QWidget) -> None:
        """
        Set the widget shown when the block is open.

        Parameters
        ----------
        content : QWidget
            The content of the block.
        """

        self.content = content
        self.box.addWidget(content, alignment=Qt.AlignTop)
        content.hide()

        self.animation = QPropertyAnimation(content, b"maximumHeight", self)
        self.animation.setDuration(self.duration)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.animation.finished.connect(self._animation_done)

    def is_expanded(self) -> bool:
        """Tell whether the block is open."""

        return self.collapse_btn.isChecked()

    def set_expanded(self, expanded: bool, animate: Optional[bool] = True) -> None:
        """
        Open or close the block.

        Parameters
        ----------
        expanded : bool
            True to open the block.
        animate : bool, optional
            False to jump to the new state, as when a window is built with some
            of its blocks already open.
        """

        if self.is_expanded() == expanded:
            return

        self._animate_next = animate
        self.collapse_btn.setChecked(expanded)

    def _content_height(self) -> int:
        """Return the height the content asks for."""

        return max(
            self.content.sizeHint().height(), self.content.minimumSizeHint().height()
        )

    def _toggle_content(self, expanded: bool) -> None:
        """Show or hide the content, animating its height."""

        self.update()

        if self.content is None:
            self.animation_finished.emit(expanded)
            return

        self.animation.stop()

        if not self._animate_next:
            self._animate_next = True
            self.content.setVisible(expanded)
            self.content.setMaximumHeight(UNCONSTRAINED)
            self.animation_finished.emit(expanded)
            return

        if expanded:
            # Shown right away, so that the content is live from the first frame
            # of the animation and only revealed by the growing height.
            start = 0 if self.content.isHidden() else self.content.height()
            self.content.setMaximumHeight(start)
            self.content.show()
            self.animation.setStartValue(start)
            self.animation.setEndValue(self._content_height())
        else:
            self.animation.setStartValue(self.content.height())
            self.animation.setEndValue(0)

        self.animation.start()

    def _animation_done(self) -> None:
        """Hide a folded content, and release the height of an open one."""

        expanded = self.is_expanded()

        if not expanded:
            self.content.hide()

        # The constraint only serves the animation: leaving it behind would
        # freeze the content at the height it had when it was opened.
        self.content.setMaximumHeight(UNCONSTRAINED)
        self.animation_finished.emit(expanded)

    def settle(self) -> None:
        """
        Stop the animation and put the content in its final state.

        A running animation on a widget that is going away is both pointless
        and a way to touch it after its window is gone, so a hidden or closed
        block jumps to the state it was heading for.
        """

        if self.animation is None or self.animation.state() != QPropertyAnimation.Running:
            return

        self.animation.stop()
        expanded = self.is_expanded()
        self.content.setVisible(expanded)
        self.content.setMaximumHeight(UNCONSTRAINED)

    def hideEvent(self, event: QEvent) -> None:
        """Settle the animation when the block is hidden."""

        self.settle()
        super().hideEvent(event)

    def closeEvent(self, event: QEvent) -> None:
        """Settle the animation when the block is closed."""

        self.settle()
        super().closeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Paint the card: a rounded frame, marked when the block is open.

        Parameters
        ----------
        event : QPaintEvent
            The paint event.
        """

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(QColor(CARD_BACKGROUND))
        painter.setPen(
            QPen(QColor(CARD_BORDER_OPEN if self.is_expanded() else CARD_BORDER), 1.0)
        )
        painter.drawRoundedRect(
            QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), self.radius, self.radius
        )
