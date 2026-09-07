"""
Application-wide look and feel for celldetective.

This module holds the low level painting routines for the celldetective
indicators (check boxes, radio buttons, chevrons, sliders) and the
:class:`CelldetectiveStyle` proxy style that installs them on every widget of
the software. The same routines are reused by the item delegates of the
checkable combo boxes, so a tick looks identical wherever it is drawn.
"""

import logging
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QListView,
    QProxyStyle,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionSlider,
    QWidget,
    QStyleOption,
)
from PyQt5.QtCore import QSize
from typing import Optional

from celldetective.gui.base.styles import CELLDETECTIVE_BLUE

logger = logging.getLogger("celldetective")

BORDER_COLOR = "#B8B8B8"
BORDER_DISABLED = "#DCDCDC"
FIELD_DISABLED = "#F5F5F5"
DISABLED_COLOR = "#BDBDBD"
GROOVE_COLOR = "#D8DCE0"
HOVER_COLOR = "#ECEFF1"


def _indicator_colors(
    accent: QColor, enabled: bool, hovered: bool, on_accent: bool
) -> tuple:
    """
    Resolve the fill, border and mark colors of an indicator.

    Parameters
    ----------
    accent : QColor
        The accent color of the checked state.
    enabled : bool
        Whether the widget is enabled.
    hovered : bool
        Whether the mouse is over the indicator.
    on_accent : bool
        Whether the indicator is drawn on an accent colored background, in
        which case it is inverted to stay visible.

    Returns
    -------
    tuple
        The (fill, border, mark) colors.
    """

    if not enabled:
        return QColor(DISABLED_COLOR), QColor(DISABLED_COLOR), QColor(Qt.white)

    if on_accent:
        return QColor(Qt.white), QColor(255, 255, 255, 170), QColor(accent)

    border = QColor(accent) if hovered else QColor(BORDER_COLOR)

    return QColor(accent), border, QColor(Qt.white)


def draw_check_indicator(
    painter: QPainter,
    rect: QRectF,
    check_state: int = Qt.Checked,
    enabled: bool = True,
    hovered: bool = False,
    on_accent: bool = False,
    accent: Optional[str] = CELLDETECTIVE_BLUE,
) -> None:
    """
    Draw a rounded check indicator in the given rectangle.

    Parameters
    ----------
    painter : QPainter
        The painter to draw with.
    rect : QRectF
        The rectangle of the indicator. The largest centered square that fits
        in it is used.
    check_state : int, optional
        One of Qt.Unchecked, Qt.PartiallyChecked or Qt.Checked.
    enabled : bool, optional
        Whether the widget is enabled.
    hovered : bool, optional
        Whether the mouse is over the indicator.
    on_accent : bool, optional
        Whether the indicator is drawn on an accent colored background (e.g. a
        selected row), in which case its colors are inverted.
    accent : str, optional
        The accent color of the checked state.
    """

    fill, border, mark = _indicator_colors(
        QColor(accent), enabled, hovered, on_accent
    )

    side = min(rect.width(), rect.height())
    box = QRectF(
        rect.center().x() - side / 2.0, rect.center().y() - side / 2.0, side, side
    ).adjusted(0.5, 0.5, -0.5, -0.5)
    radius = max(2.0, side / 4.0)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    if check_state == Qt.Unchecked or check_state is None:
        painter.setBrush(Qt.NoBrush if on_accent else QColor(Qt.white))
        painter.setPen(QPen(border, 1.4))
        painter.drawRoundedRect(box, radius, radius)
        painter.restore()
        return

    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawRoundedRect(box, radius, radius)

    weight = max(1.6, side / 7.5)
    if check_state == Qt.PartiallyChecked:
        painter.setPen(QPen(mark, weight, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(
            QPointF(box.left() + 0.26 * box.width(), box.center().y()),
            QPointF(box.left() + 0.74 * box.width(), box.center().y()),
        )
    else:
        path = QPainterPath()
        path.moveTo(box.left() + 0.24 * box.width(), box.top() + 0.53 * box.height())
        path.lineTo(box.left() + 0.43 * box.width(), box.top() + 0.72 * box.height())
        path.lineTo(box.left() + 0.77 * box.width(), box.top() + 0.30 * box.height())
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(mark, weight, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(path)

    painter.restore()


def draw_radio_indicator(
    painter: QPainter,
    rect: QRectF,
    checked: bool = True,
    enabled: bool = True,
    hovered: bool = False,
    on_accent: bool = False,
    accent: Optional[str] = CELLDETECTIVE_BLUE,
) -> None:
    """
    Draw a round radio indicator in the given rectangle.

    Parameters
    ----------
    painter : QPainter
        The painter to draw with.
    rect : QRectF
        The rectangle of the indicator.
    checked : bool, optional
        Whether the radio button is checked.
    enabled : bool, optional
        Whether the widget is enabled.
    hovered : bool, optional
        Whether the mouse is over the indicator.
    on_accent : bool, optional
        Whether the indicator is drawn on an accent colored background.
    accent : str, optional
        The accent color of the checked state.
    """

    fill, border, mark = _indicator_colors(
        QColor(accent), enabled, hovered, on_accent
    )

    side = min(rect.width(), rect.height())
    circle = QRectF(
        rect.center().x() - side / 2.0, rect.center().y() - side / 2.0, side, side
    ).adjusted(0.5, 0.5, -0.5, -0.5)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)

    if not checked:
        painter.setBrush(Qt.NoBrush if on_accent else QColor(Qt.white))
        painter.setPen(QPen(border, 1.4))
        painter.drawEllipse(circle)
        painter.restore()
        return

    painter.setPen(Qt.NoPen)
    painter.setBrush(fill)
    painter.drawEllipse(circle)

    dot = side * 0.24
    painter.setBrush(mark)
    painter.drawEllipse(circle.center(), dot, dot)

    painter.restore()


def draw_chevron(
    painter: QPainter,
    rect: QRectF,
    color: Optional[str] = "#616161",
    enabled: bool = True,
) -> None:
    """
    Draw a downward chevron, used as drop-down arrow.

    Parameters
    ----------
    painter : QPainter
        The painter to draw with.
    rect : QRectF
        The rectangle the chevron is centered in.
    color : str, optional
        The color of the chevron.
    enabled : bool, optional
        Whether the widget is enabled.
    """

    side = min(rect.width(), rect.height())
    if side < 4:
        return

    half = side * 0.28
    center = rect.center()

    path = QPainterPath()
    path.moveTo(center.x() - half, center.y() - half / 2.0)
    path.lineTo(center.x(), center.y() + half / 2.0)
    path.lineTo(center.x() + half, center.y() - half / 2.0)

    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(
        QPen(
            QColor(color) if enabled else QColor(DISABLED_COLOR),
            max(1.3, side / 9.0),
            Qt.SolidLine,
            Qt.RoundCap,
            Qt.RoundJoin,
        )
    )
    painter.drawPath(path)
    painter.restore()


class CelldetectiveStyle(QProxyStyle):
    """
    Proxy style giving the software its own check boxes, radio buttons,
    drop-down arrows and sliders.

    It wraps a base style (Fusion by default) and only takes over the few
    elements that look dated when drawn natively; everything else is delegated
    to the base style. Widgets that carry their own style sheet (napari's, for
    instance) are unaffected, since a style sheet takes precedence over the
    style.
    """

    check_size = 16
    radio_size = 16
    slider_handle = 16
    slider_groove = 5
    row_padding = 6

    def __init__(
        self, base_style: Optional[str] = "Fusion", accent: Optional[str] = CELLDETECTIVE_BLUE
    ) -> None:
        """
        Initialize the style.

        Parameters
        ----------
        base_style : str, optional
            The name of the style to delegate to.
        accent : str, optional
            The accent color of the checked/selected states.
        """

        super().__init__(base_style)
        self.accent = accent

    def pixelMetric(
        self,
        metric: int,
        option: Optional[QStyleOption] = None,
        widget: Optional[QWidget] = None,
    ) -> int:
        """Give the indicators and the slider handle a comfortable size."""

        if metric in (QStyle.PM_IndicatorWidth, QStyle.PM_IndicatorHeight):
            return self.check_size
        if metric in (
            QStyle.PM_ExclusiveIndicatorWidth,
            QStyle.PM_ExclusiveIndicatorHeight,
        ):
            return self.radio_size
        if metric == QStyle.PM_SliderLength:
            return self.slider_handle

        return super().pixelMetric(metric, option, widget)

    def sizeFromContents(
        self,
        content_type: int,
        option: QStyleOption,
        size: QSize,
        widget: Optional[QWidget] = None,
    ) -> QSize:
        """Give list view rows (combo box popups included) some air."""

        size = super().sizeFromContents(content_type, option, size, widget)

        if content_type == QStyle.CT_ItemViewItem and isinstance(widget, QListView):
            size.setHeight(size.height() + self.row_padding)

        return size

    def drawPrimitive(
        self,
        element: int,
        option: QStyleOption,
        painter: QPainter,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Draw the celldetective indicators, delegate everything else."""

        if element == QStyle.PE_PanelItemViewItem and isinstance(widget, QListView):
            selected = bool(option.state & QStyle.State_Selected)
            hovered = bool(option.state & QStyle.State_MouseOver)

            if selected or hovered:
                painter.save()
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setPen(Qt.NoPen)
                painter.setBrush(
                    QColor(self.accent) if selected else QColor(HOVER_COLOR)
                )
                painter.drawRoundedRect(
                    QRectF(option.rect).adjusted(2.5, 1.5, -2.5, -1.5), 5, 5
                )
                painter.restore()
                return

        if element in (
            QStyle.PE_IndicatorCheckBox,
            QStyle.PE_IndicatorRadioButton,
            QStyle.PE_IndicatorArrowDown,
        ):
            enabled = bool(option.state & QStyle.State_Enabled)
            hovered = bool(option.state & QStyle.State_MouseOver)
            rect = QRectF(option.rect)

            if element == QStyle.PE_IndicatorCheckBox:
                if option.state & QStyle.State_NoChange:
                    check_state = Qt.PartiallyChecked
                elif option.state & QStyle.State_On:
                    check_state = Qt.Checked
                else:
                    check_state = Qt.Unchecked
                draw_check_indicator(
                    painter,
                    rect,
                    check_state=check_state,
                    enabled=enabled,
                    hovered=hovered,
                    accent=self.accent,
                )
            elif element == QStyle.PE_IndicatorRadioButton:
                draw_radio_indicator(
                    painter,
                    rect,
                    checked=bool(option.state & QStyle.State_On),
                    enabled=enabled,
                    hovered=hovered,
                    accent=self.accent,
                )
            else:
                draw_chevron(painter, rect, enabled=enabled)

            return

        super().drawPrimitive(element, option, painter, widget)

    def drawComplexControl(
        self,
        control: int,
        option: QStyleOption,
        painter: QPainter,
        widget: Optional[QWidget] = None,
    ) -> None:
        """Draw flat, accent colored sliders, delegate everything else."""

        if (
            control == QStyle.CC_Slider
            and isinstance(option, QStyleOptionSlider)
            and not (option.subControls & QStyle.SC_SliderTickmarks)
        ):
            self._draw_slider(option, painter, widget)
            return

        if control == QStyle.CC_ComboBox and isinstance(option, QStyleOptionComboBox):
            self._draw_combobox(option, painter, widget)
            return

        super().drawComplexControl(control, option, painter, widget)

    def _draw_combobox(
        self,
        option: QStyleOptionComboBox,
        painter: QPainter,
        widget: Optional[QWidget] = None,
    ) -> None:
        """
        Paint a combo box as a rounded frame closed by a chevron.

        Parameters
        ----------
        option : QStyleOptionComboBox
            The style options of the combo box.
        painter : QPainter
            The painter to draw with.
        widget : QWidget, optional
            The combo box being painted.
        """

        enabled = bool(option.state & QStyle.State_Enabled)
        active = bool(
            option.state
            & (QStyle.State_MouseOver | QStyle.State_On | QStyle.State_HasFocus)
        )

        if option.frame:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(QColor(Qt.white) if enabled else QColor(FIELD_DISABLED))
            painter.setPen(
                QPen(
                    QColor(self.accent)
                    if (enabled and active)
                    else QColor(BORDER_COLOR if enabled else BORDER_DISABLED),
                    1.0,
                )
            )
            painter.drawRoundedRect(
                QRectF(option.rect).adjusted(0.5, 0.5, -0.5, -0.5), 5, 5
            )
            painter.restore()

        if option.subControls & QStyle.SC_ComboBoxArrow:
            self.draw_combo_arrow(option, painter, widget)

    def draw_combo_arrow(
        self,
        option: QStyleOptionComboBox,
        painter: QPainter,
        widget: Optional[QWidget] = None,
    ) -> None:
        """
        Draw the drop-down chevron of a combo box.

        Exposed separately because combo boxes carrying a style sheet are
        painted by the style sheet style, which draws no arrow of its own.

        Parameters
        ----------
        option : QStyleOptionComboBox
            The style options of the combo box.
        painter : QPainter
            The painter to draw with.
        widget : QWidget, optional
            The combo box being painted.
        """

        rect = self.subControlRect(
            QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxArrow, widget
        )
        draw_chevron(
            painter, QRectF(rect), enabled=bool(option.state & QStyle.State_Enabled)
        )

    def _draw_slider(
        self,
        option: QStyleOptionSlider,
        painter: QPainter,
        widget: Optional[QWidget] = None,
    ) -> None:
        """
        Paint a slider as a thin rounded groove with a circular handle.

        Parameters
        ----------
        option : QStyleOptionSlider
            The style options of the slider.
        painter : QPainter
            The painter to draw with.
        widget : QWidget, optional
            The slider being painted.
        """

        groove = self.subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderGroove, widget
        )
        handle = self.subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderHandle, widget
        )
        enabled = bool(option.state & QStyle.State_Enabled)
        accent = QColor(self.accent) if enabled else QColor(DISABLED_COLOR)
        horizontal = option.orientation == Qt.Horizontal

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)

        thickness = self.slider_groove
        if horizontal:
            line = QRectF(
                groove.left(),
                groove.center().y() - thickness / 2.0 + 1,
                groove.width(),
                thickness,
            )
            filled = QRectF(line)
            # `upsideDown` tells on which end of the groove the minimum sits.
            if option.upsideDown:
                filled.setLeft(min(handle.center().x(), line.right()))
            else:
                filled.setRight(max(handle.center().x(), line.left()))
        else:
            line = QRectF(
                groove.center().x() - thickness / 2.0 + 1,
                groove.top(),
                thickness,
                groove.height(),
            )
            filled = QRectF(line)
            if option.upsideDown:
                filled.setTop(min(handle.center().y(), line.bottom()))
            else:
                filled.setBottom(max(handle.center().y(), line.top()))

        radius = thickness / 2.0
        painter.setBrush(QColor(GROOVE_COLOR))
        painter.drawRoundedRect(line, radius, radius)

        if filled.width() > 0 and filled.height() > 0:
            painter.setBrush(accent)
            painter.drawRoundedRect(filled, radius, radius)

        diameter = min(handle.width(), handle.height(), self.slider_handle)
        center = QRectF(handle).center()
        pressed = bool(option.state & QStyle.State_Sunken)
        hovered = bool(option.state & QStyle.State_MouseOver)

        painter.setBrush(QColor(Qt.white))
        painter.setPen(QPen(accent, 3.0 if (pressed or hovered) else 2.2))
        painter.drawEllipse(center, diameter / 2.0 - 1.5, diameter / 2.0 - 1.5)

        painter.restore()
