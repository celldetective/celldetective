from PyQt5.QtWidgets import QMainWindow, QWidget, QDialog, QMessageBox
from PyQt5.QtCore import Qt
from celldetective.gui.base.styles import Styles


class CelldetectiveWidget(QWidget, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveMainWindow(QMainWindow, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)
        self.setAttribute(Qt.WA_DeleteOnClose)


class CelldetectiveDialog(QDialog, Styles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowIcon(self.celldetective_icon)


def generic_message(message, msg_type="warning"):

    print(message)
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
