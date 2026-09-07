from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QResizeEvent
from PyQt5.QtWidgets import QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective import get_logger

logger = get_logger(__name__)


_SAFE_CANVAS_CLS = None


def _safe_canvas_class():
    """
    Return a ``FigureCanvasQTAgg`` subclass whose paint cannot fault the process.

    matplotlib's ``paintEvent`` opens ``QPainter(self)`` and then, without
    checking that it actually started, calls ``painter.eraseRect(rect)``.
    ``QPainter::eraseRect()`` is one of the few QPainter methods that reads
    ``d->state`` with no active-painter guard, so when ``begin()`` fails the
    call dereferences a null pointer and the interpreter dies with
    "Windows fatal exception: access violation" -- no traceback, no failing
    assertion, the whole process gone.

    ``begin()`` fails whenever the widget has no usable paint engine: a zero
    width or height, a backing store already torn down with the window, or a
    paint that re-enters while another painter is still open on the widget.
    Those all happen while a viewer is being closed or resized, and a paint
    that could not begin would have drawn nothing anyway -- so probe the paint
    device first and skip the paint instead of faulting on it.

    The class is built lazily so importing this module does not pull in the Qt
    backend before a figure is actually needed.
    """

    global _SAFE_CANVAS_CLS
    if _SAFE_CANVAS_CLS is not None:
        return _SAFE_CANVAS_CLS

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from PyQt5.QtGui import QPainter

    class SafeFigureCanvasQTAgg(FigureCanvasQTAgg):
        """FigureCanvasQTAgg that no-ops a paint it cannot safely perform."""

        def paintEvent(self, event):
            if self.width() <= 0 or self.height() <= 0:
                return

            # Ask Qt the same question matplotlib's paintEvent fails to ask.
            probe = QPainter()
            if not probe.begin(self):
                logger.debug(
                    "Skipping canvas paint: no paint engine available "
                    "(size=%sx%s, visible=%s).",
                    self.width(),
                    self.height(),
                    self.isVisible(),
                )
                return
            probe.end()

            super().paintEvent(event)

    _SAFE_CANVAS_CLS = SafeFigureCanvasQTAgg
    return _SAFE_CANVAS_CLS


class FigureCanvas(CelldetectiveWidget):
    """
    Generic figure canvas.
    """

    def __init__(
        self,
        fig: Figure,
        title: Optional[str] = None,
        interactive: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the FigureCanvas.

        Parameters
        ----------
        fig : Figure
            The matplotlib figure.
        title : str, optional
            The window title.
        interactive : bool, optional
            Whether to include a navigation toolbar.
        """
        super().__init__()
        self.fig = fig
        self.setWindowTitle(title)
        self.canvas = _safe_canvas_class()(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;")
        if interactive:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

            self.toolbar = NavigationToolbar2QT(self.canvas)
            self.toolbar.setStyleSheet(
                "QToolButton:checked {background-color: darkgray;} QToolButton:hover {background-color: lightgray;} QToolButton {background-color: transparent; border: none;}"
            )
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas, 90)
        if interactive:
            self.layout.addWidget(self.toolbar)

        self.manual_layout = False
        # center_window(self)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handle resize events.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.
        """
        super().resizeEvent(event)
        try:
            manual_layout = getattr(self, "manual_layout", False)

            # Double check for profile axes manually (robust fallback)
            if not manual_layout and hasattr(self.fig, "axes"):
                for ax in self.fig.axes:
                    if ax.get_label() == "profile_axes":
                        manual_layout = True
                        break

            if not manual_layout:
                self.fig.tight_layout()
        except Exception as e:
            logger.debug(f"tight_layout failed on resize: {e}")

    def draw(self):
        """Draw the canvas."""
        self.canvas.draw()

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Delete figure on closing window.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        # Silence the canvas before the figure goes. WA_DeleteOnClose only
        # *schedules* deletion, so Qt can still deliver a paint between here
        # and the deleteLater(), and that paint would run against a figure
        # that no longer has any axes on a window that is already going away.
        try:
            self.canvas.setUpdatesEnabled(False)
            self.canvas.hide()
        except RuntimeError as e:
            logger.debug(f"Canvas already destroyed during cleanup: {e}")

        self.fig.clf()
        plt.close(self.fig)
        super(FigureCanvas, self).closeEvent(event)
