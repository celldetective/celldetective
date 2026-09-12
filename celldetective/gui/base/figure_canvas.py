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
    Return a ``FigureCanvasQTAgg`` subclass that cannot kill the interpreter.

    Qt keeps delivering paints and queued idle-draws to a canvas that is in no
    state to serve them -- a window mid-close, a widget whose C++ half sip has
    already freed, one shown on a CI session that never composites it.
    matplotlib's handlers assume none of that:

    * ``paintEvent`` opens ``QPainter(self)`` and then calls
      ``painter.eraseRect(rect)`` without checking that ``begin()`` succeeded.
      ``QPainter::eraseRect()`` is one of the few QPainter methods that reads
      ``d->state`` with no active-painter guard, so a painter that failed to
      begin is a null dereference: "Windows fatal exception: access violation",
      no traceback, no failing assertion, the whole run's results gone.
    * ``paintEvent`` and ``_draw_idle`` -- the latter reached from a
      ``QTimer.singleShot`` that outlives the widget it was armed for -- touch
      the C++ object outside any try block. Once sip has freed it that is a
      ``RuntimeError`` raised inside a Qt virtual, which PyQt5 answers with
      ``abort()``.

    Whether ``begin()`` will succeed cannot be predicted from the widget's
    visibility or size -- a live, visible, non-zero-sized canvas can still have
    no paint engine. The only reliable answer is the painter itself, so open
    one, ask it, and drive the draw from it. That is matplotlib's own paint
    body with the missing guard added, rather than a second painter opened
    behind its back.

    The class is built lazily so importing this module does not pull in the Qt
    backend before a figure is actually needed.
    """

    global _SAFE_CANVAS_CLS
    if _SAFE_CANVAS_CLS is not None:
        return _SAFE_CANVAS_CLS

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.transforms import Bbox
    from PyQt5 import sip
    from PyQt5.QtCore import QPoint
    from PyQt5.QtGui import QImage, QPainter

    class SafeFigureCanvasQTAgg(FigureCanvasQTAgg):
        """FigureCanvasQTAgg that no-ops a draw it cannot safely perform."""

        def paintEvent(self, event):
            # Ask sip before Qt: every other call here goes through the
            # wrapper, and on a freed object that is the RuntimeError we are
            # trying to avoid rather than an answer.
            if sip.isdeleted(self):
                return

            try:
                self._draw_idle()  # Only does something if a draw is pending.

                # No renderer yet: give up and wait for the first real draw,
                # exactly as matplotlib does.
                if not hasattr(self, "renderer"):
                    return

                painter = QPainter(self)
                if not painter.isActive():
                    # begin() failed -- no paint engine on this device. This is
                    # the case matplotlib walks straight into.
                    logger.debug(
                        "Skipped canvas paint: no paint engine "
                        f"(size={self.width()}x{self.height()}, "
                        f"visible={self.isVisible()})."
                    )
                    return

                try:
                    # See documentation of QRect: bottom() and right() are off
                    # by 1, so use left() + width() and top() + height().
                    rect = event.rect()
                    # Scale the rect by the screen dpi ratio to get Figure
                    # coordinates rather than Qt ones.
                    width = rect.width() * self.device_pixel_ratio
                    height = rect.height() * self.device_pixel_ratio
                    left, top = self.mouseEventCoords(rect.topLeft())
                    # Shift "top" by the image height, and "left" by its width,
                    # to reach the corners in our coordinate system.
                    bottom = top - height
                    right = left + width

                    buf = memoryview(
                        self.copy_from_bbox(Bbox([[left, bottom], [right, top]]))
                    )
                    painter.eraseRect(rect)
                    qimage = QImage(
                        buf, buf.shape[1], buf.shape[0], QImage.Format_RGBA8888
                    )
                    qimage.setDevicePixelRatio(self.device_pixel_ratio)
                    painter.drawImage(QPoint(rect.left(), rect.top()), qimage)

                    # The rubber band the zoom tool draws over the figure.
                    draw_rect = getattr(self, "_draw_rect_callback", None)
                    if draw_rect is not None:
                        draw_rect(painter)
                finally:
                    painter.end()
            except RuntimeError as e:
                # Freed between the check above and the paint itself.
                logger.debug(f"Skipped paint on a canvas being destroyed: {e}")

        def _draw_idle(self):
            # Armed by QTimer.singleShot(0, self._draw_idle); the widget can be
            # gone by the time it fires.
            if sip.isdeleted(self):
                return
            try:
                super()._draw_idle()
            except RuntimeError as e:
                logger.debug(f"Skipped idle draw on a canvas being destroyed: {e}")

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
