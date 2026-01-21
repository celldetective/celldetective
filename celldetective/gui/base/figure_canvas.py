from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window


class FigureCanvas(CelldetectiveWidget):
    """
    Generic figure canvas.
    """

    def __init__(self, fig, title="", interactive=True):
        super().__init__()
        self.fig = fig
        self.setWindowTitle(title)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color: transparent;")
        if interactive:
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

            self.toolbar = NavigationToolbar2QT(self.canvas)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.canvas, 90)
        if interactive:
            self.layout.addWidget(self.toolbar)

        #center_window(self)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def resizeEvent(self, event):

        super().resizeEvent(event)
        try:
            self.fig.tight_layout()
        except:
            pass

    def draw(self):
        self.canvas.draw()

    def closeEvent(self, event):
        """Delete figure on closing window."""
        # self.canvas.ax.cla() # ****
        # self.canvas.ax.cla() # ****
        self.fig.clf()  # ****
        import matplotlib.pyplot as plt

        plt.close(self.fig)
        super(FigureCanvas, self).closeEvent(event)
