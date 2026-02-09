from typing import Optional

from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QResizeEvent
from PyQt5.QtWidgets import QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget


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
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        self.canvas = FigureCanvasQTAgg(self.fig)
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
        except:
            pass

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
        # self.canvas.ax.cla() # ****
        # self.canvas.ax.cla() # ****
        self.fig.clf()  # ****
        import matplotlib.pyplot as plt

        plt.close(self.fig)
        super(FigureCanvas, self).closeEvent(event)
