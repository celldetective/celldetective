import numpy as np
from PyQt5.QtGui import QStandardItemModel, QPalette
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QDialog,
    QMessageBox,
    QComboBox,
    QToolButton,
    QMenu,
    QStylePainter,
    QStyleOptionComboBox,
    QStyle,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal, QEvent
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


class QCheckableComboBox(QComboBox):
    """
    adapted from https://stackoverflow.com/questions/22775095/pyqt-how-to-set-combobox-items-be-checkable
    """

    activated = pyqtSignal(str)

    def __init__(self, obj="", parent_window=None, *args, **kwargs):

        super().__init__(parent_window, *args, **kwargs)

        self.setTitle("")
        self.setModel(QStandardItemModel(self))
        self.obj = obj
        self.toolButton = QToolButton(parent_window)
        self.toolButton.setText("")
        self.toolMenu = QMenu(parent_window)
        self.toolButton.setMenu(self.toolMenu)
        self.toolButton.setPopupMode(QToolButton.InstantPopup)
        self.anySelected = False

        self.view().viewport().installEventFilter(self)
        self.view().pressed.connect(self.handleItemPressed)

    def clear(self):

        self.unselectAll()
        self.toolMenu.clear()
        super().clear()

    def handleItemPressed(self, index):

        idx = index.row()
        actions = self.toolMenu.actions()

        item = self.model().itemFromIndex(index)
        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
            actions[idx].setChecked(False)
        else:
            item.setCheckState(Qt.Checked)
            actions[idx].setChecked(True)
            self.anySelected = True

        options_checked = np.array([a.isChecked() for a in actions])
        if len(options_checked[options_checked]) > 1:
            self.setTitle(f'Multiple {self.obj+"s"} selected...')
        elif len(options_checked[options_checked]) == 1:
            idx_selected = np.where(options_checked)[0][0]
            if idx_selected != idx:
                item = self.model().item(idx_selected)
            self.setTitle(item.text())
        elif len(options_checked[options_checked]) == 0:
            self.setTitle(f"No {self.obj} selected...")
            self.anySelected = False

        self.activated.emit(self.title())

    def setCurrentIndex(self, index):

        super().setCurrentIndex(index)

        item = self.model().item(index)
        modelIndex = self.model().indexFromItem(item)

        self.handleItemPressed(modelIndex)

    def selectAll(self):

        actions = self.toolMenu.actions()
        for i, a in enumerate(actions):
            if not a.isChecked():
                self.setCurrentIndex(i)
        self.anySelected = True

    def unselectAll(self):

        actions = self.toolMenu.actions()
        for i, a in enumerate(actions):
            if a.isChecked():
                self.setCurrentIndex(i)
        self.anySelected = False

    def title(self):
        return self._title

    def setTitle(self, title):
        self._title = title
        self.update()
        self.repaint()

    def paintEvent(self, event):

        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.Text))
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        opt.currentText = self._title
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        painter.drawControl(QStyle.CE_ComboBoxLabel, opt)

    def addItem(self, item, tooltip=None):

        super().addItem(item)
        idx = self.findText(item)
        if tooltip is not None:
            self.setItemData(idx, tooltip, Qt.ToolTipRole)
        item2 = self.model().item(idx, 0)
        item2.setCheckState(Qt.Unchecked)
        action = self.toolMenu.addAction(item)
        action.setCheckable(True)

    def addItems(self, items):

        super().addItems(items)

        for item in items:

            idx = self.findText(item)
            item2 = self.model().item(idx, 0)
            item2.setCheckState(Qt.Unchecked)
            action = self.toolMenu.addAction(item)
            action.setCheckable(True)

    def getSelectedIndices(self):

        actions = self.toolMenu.actions()
        options_checked = np.array([a.isChecked() for a in actions])
        idx_selected = np.where(options_checked)[0]

        return list(idx_selected)

    def currentText(self):
        return self.title()

    def isMultipleSelection(self):
        return self.currentText().startswith("Multiple")

    def isSingleSelection(self):
        return not self.currentText().startswith(
            "Multiple"
        ) and not self.title().startswith("No")

    def isAnySelected(self):
        return not self.title().startswith("No")

    def eventFilter(self, source, event):
        if source is self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                return True  # Prevent the popup from closing
        return super().eventFilter(source, event)


class QHSeperationLine(QFrame):
    """
    a horizontal seperation line\n
    """

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(20)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
