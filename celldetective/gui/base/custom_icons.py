"""Custom, hand-composed icons for the GUI.

Most icons come straight from ``superqt.fonticon`` / Material Design Icons, but a
few need more than a single glyph can express (e.g. stacking, masking, or drawing
on top of a glyph). Those composited icons live here so they are built in one
place and reused across the GUI, next to :mod:`celldetective.gui.base.styles`.
"""

from PyQt5.QtCore import Qt, QSize, QPointF
from PyQt5.QtGui import QIcon, QPainter, QPen, QColor
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6


def scatter_with_divider_icon(color: str = "black", size: int = 128) -> QIcon:
    """Build the CLASSIFY step icon: the classifier's scatter-plot glyph split by
    a diagonal boundary, with the bottom point hollowed to an outline to denote
    the other class — conveying "split the points into classes".

    ``superqt.fonticon.icon`` renders a single glyph and cannot stack, so the
    scatter glyph is rendered to a pixmap; the bottom-right dot's core is then
    cleared (leaving its rim as an outline ring) and a divider is painted across
    the max-margin gap between the two filled dots and the outlined one. The
    geometry is expressed as fractions of ``size`` (the glyph's three dots sit at
    fixed relative positions), so it holds at any render size. Rendered large and
    downscaled by Qt for crispness at button size.
    """
    pixmap = icon(MDI6.scatter_plot, color=color).pixmap(QSize(size, size))
    s = float(size)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Hollow the bottom-right dot (centre ~(0.66, 0.70) of the glyph box): clear
    # its core so only the rim survives as an outline = the other class.
    painter.setCompositionMode(QPainter.CompositionMode_Clear)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("black"))  # Clear uses alpha only; colour is ignored
    painter.drawEllipse(QPointF(0.664 * s, 0.703 * s), 0.075 * s, 0.075 * s)

    # Divider across the max-margin gap (steeper than 45°), keeping both filled
    # dots on one side and the outlined dot on the other.
    painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
    pen = QPen(QColor(color))
    pen.setWidthF(max(2.0, s / 20))
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(0.336 * s, 0.922 * s), QPointF(0.781 * s, 0.078 * s))
    painter.end()
    return QIcon(pixmap)
