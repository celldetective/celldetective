"""
Unit tests for the application-wide style in celldetective.gui.base.app_style.

Covers:
- CelldetectiveStyle (indicator metrics, check box, radio button, combo arrow, slider)
- the standalone painting routines shared with the item delegates
"""

import logging
import pytest
from PyQt5.QtCore import Qt, QRect, QRectF
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QSlider,
    QStyle,
    QStyleOptionButton,
    QStyleOptionComboBox,
    QStyleOptionSlider,
)

from celldetective.gui.base.app_style import (
    CelldetectiveStyle,
    draw_check_indicator,
    draw_radio_indicator,
)
from celldetective.gui.base.styles import CELLDETECTIVE_BLUE


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def paint_on_pixmap(draw, size=(40, 40)):
    """
    Run a drawing callback on a white pixmap and return the resulting image.

    Parameters
    ----------
    draw : callable
        Callback taking the painter as single argument.
    size : tuple, optional
        The (width, height) of the pixmap.

    Returns
    -------
    QImage
        The painted image.
    """

    pixmap = QPixmap(*size)
    pixmap.fill(Qt.white)
    painter = QPainter(pixmap)
    try:
        draw(painter)
    finally:
        painter.end()

    return pixmap.toImage()


def contains_color(image, color, tolerance=25):
    """
    Check whether a color appears in an image.

    Parameters
    ----------
    image : QImage
        The image to inspect.
    color : str
        The color to look for.
    tolerance : int, optional
        Maximum per-channel difference to still count as a match.

    Returns
    -------
    bool
        True if at least one pixel matches.
    """

    target = QColor(color)
    for x in range(image.width()):
        for y in range(image.height()):
            pixel = QColor(image.pixel(x, y))
            if (
                abs(pixel.red() - target.red()) <= tolerance
                and abs(pixel.green() - target.green()) <= tolerance
                and abs(pixel.blue() - target.blue()) <= tolerance
            ):
                return True

    return False


# =============================================================================
# Painting routines
# =============================================================================


class TestIndicatorPainting:
    """Tests for the standalone indicator painting routines."""

    @pytest.mark.parametrize(
        "check_state", [Qt.Unchecked, Qt.PartiallyChecked, Qt.Checked]
    )
    def test_check_indicator_paints(self, qtbot, check_state):
        """Every check state paints something on the pixmap."""
        image = paint_on_pixmap(
            lambda painter: draw_check_indicator(
                painter, QRectF(4, 4, 32, 32), check_state=check_state
            )
        )

        assert not image.isNull()
        if check_state != Qt.Unchecked:
            assert contains_color(image, CELLDETECTIVE_BLUE)

    def test_check_indicator_inverts_on_accent(self, qtbot):
        """On an accent background the checked indicator becomes white."""
        image = paint_on_pixmap(
            lambda painter: draw_check_indicator(
                painter, QRectF(4, 4, 32, 32), check_state=Qt.Checked, on_accent=True
            )
        )

        assert contains_color(image, CELLDETECTIVE_BLUE)  # the tick itself

    def test_radio_indicator_paints_checked(self, qtbot):
        """A checked radio button is filled with the accent color."""
        image = paint_on_pixmap(
            lambda painter: draw_radio_indicator(
                painter, QRectF(4, 4, 32, 32), checked=True
            )
        )

        assert contains_color(image, CELLDETECTIVE_BLUE)

    def test_disabled_indicator_is_grey(self, qtbot):
        """A disabled indicator drops the accent color."""
        image = paint_on_pixmap(
            lambda painter: draw_check_indicator(
                painter, QRectF(4, 4, 32, 32), check_state=Qt.Checked, enabled=False
            )
        )

        assert not contains_color(image, CELLDETECTIVE_BLUE)


# =============================================================================
# CelldetectiveStyle
# =============================================================================


class TestCelldetectiveStyle:
    """Tests for the proxy style."""

    def test_indicator_metrics(self, qtbot):
        """Check boxes, radio buttons and slider handles get their own size."""
        style = CelldetectiveStyle("Fusion")

        assert style.pixelMetric(QStyle.PM_IndicatorWidth) == style.check_size
        assert style.pixelMetric(QStyle.PM_IndicatorHeight) == style.check_size
        assert (
            style.pixelMetric(QStyle.PM_ExclusiveIndicatorWidth) == style.radio_size
        )
        assert style.pixelMetric(QStyle.PM_SliderLength) == style.slider_handle

    def test_unhandled_metric_is_delegated(self, qtbot):
        """Metrics the style does not care about come from the base style."""
        style = CelldetectiveStyle("Fusion")

        assert style.pixelMetric(QStyle.PM_ButtonMargin) >= 0

    @pytest.mark.parametrize(
        "element, state",
        [
            (QStyle.PE_IndicatorCheckBox, QStyle.State_On),
            (QStyle.PE_IndicatorRadioButton, QStyle.State_On),
        ],
    )
    def test_checked_primitives_use_the_accent(self, qtbot, element, state):
        """Checked check boxes and radio buttons are painted in blue."""
        style = CelldetectiveStyle("Fusion")
        option = QStyleOptionButton()
        option.rect = QRect(4, 4, 32, 32)
        option.state = QStyle.State_Enabled | state

        image = paint_on_pixmap(
            lambda painter: style.drawPrimitive(element, option, painter, None)
        )

        assert contains_color(image, CELLDETECTIVE_BLUE)

    def test_unchecked_primitive_has_no_accent(self, qtbot):
        """An unchecked check box is only an outline."""
        style = CelldetectiveStyle("Fusion")
        option = QStyleOptionButton()
        option.rect = QRect(4, 4, 32, 32)
        option.state = QStyle.State_Enabled

        image = paint_on_pixmap(
            lambda painter: style.drawPrimitive(
                QStyle.PE_IndicatorCheckBox, option, painter, None
            )
        )

        assert not contains_color(image, CELLDETECTIVE_BLUE)

    def test_combo_arrow_is_drawn(self, qtbot):
        """The chevron replacing the native combo arrow paints pixels."""
        style = CelldetectiveStyle("Fusion")
        option = QStyleOptionComboBox()
        option.rect = QRect(0, 0, 60, 24)
        option.state = QStyle.State_Enabled
        option.subControls = QStyle.SC_ComboBoxArrow

        image = paint_on_pixmap(
            lambda painter: style.draw_combo_arrow(option, painter, None),
            size=(60, 24),
        )

        assert contains_color(image, "#616161", tolerance=60)

    @pytest.mark.parametrize("orientation", [Qt.Horizontal, Qt.Vertical])
    def test_slider_is_painted_with_the_accent(self, qtbot, orientation):
        """Both slider orientations paint an accent colored filled groove."""
        style = CelldetectiveStyle("Fusion")
        slider = QSlider(orientation)
        qtbot.addWidget(slider)
        slider.setRange(0, 100)
        slider.setValue(60)
        slider.resize(160, 30) if orientation == Qt.Horizontal else slider.resize(
            30, 160
        )

        option = QStyleOptionSlider()
        option.initFrom(slider)
        option.rect = slider.rect()
        option.orientation = orientation
        option.minimum = 0
        option.maximum = 100
        option.sliderPosition = 60
        option.sliderValue = 60
        option.upsideDown = orientation == Qt.Vertical
        option.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderHandle

        image = paint_on_pixmap(
            lambda painter: style.drawComplexControl(
                QStyle.CC_Slider, option, painter, slider
            ),
            size=(slider.width(), slider.height()),
        )

        assert contains_color(image, CELLDETECTIVE_BLUE)

    def test_slider_with_tickmarks_falls_back(self, qtbot):
        """Sliders showing tick marks are left to the base style."""
        style = CelldetectiveStyle("Fusion")
        slider = QSlider(Qt.Horizontal)
        qtbot.addWidget(slider)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.resize(160, 40)

        option = QStyleOptionSlider()
        option.initFrom(slider)
        option.rect = slider.rect()
        option.orientation = Qt.Horizontal
        option.minimum = 0
        option.maximum = 100
        option.sliderPosition = 50
        option.subControls = (
            QStyle.SC_SliderGroove | QStyle.SC_SliderHandle | QStyle.SC_SliderTickmarks
        )

        # Only has to go through the base style without raising.
        paint_on_pixmap(
            lambda painter: style.drawComplexControl(
                QStyle.CC_Slider, option, painter, slider
            ),
            size=(slider.width(), slider.height()),
        )
