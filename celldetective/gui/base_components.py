from PyQt5.QtWidgets import QMainWindow, QWidget, QDialog, QApplication
from PyQt5.QtCore import Qt
from celldetective.gui import Styles


class CelldetectiveWidget(QWidget, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def get_screen_dimensions(self):
        app = QApplication.instance()
        screen = app.primaryScreen()
        geometry = screen.availableGeometry()
        self._screen_width, self._screen_height = geometry.getRect()[-2:]

class CelldetectiveMainWindow(QMainWindow, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveDialog(QDialog, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
