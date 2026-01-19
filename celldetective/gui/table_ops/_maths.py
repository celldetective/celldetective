import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QMessageBox,
    QLineEdit,
)
from superqt import QLabeledSlider

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import PandasModel, GenericOpColWidget
from celldetective import get_logger
from celldetective.utils.maths import differentiate_per_track, safe_log

logger = get_logger(__name__)


class DifferentiateColWidget(CelldetectiveWidget):

    def __init__(self, parent_window, column=None):

        super().__init__()
        self.parent_window = parent_window
        self.column = column

        self.setWindowTitle("d/dt")
        # Create the QComboBox and add some items
        center_window(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        self.measurements_cb = QComboBox()
        self.measurements_cb.addItems(list(self.parent_window.data.columns))
        if self.column is not None:
            idx = self.measurements_cb.findText(self.column)
            self.measurements_cb.setCurrentIndex(idx)

        measurement_layout = QHBoxLayout()
        measurement_layout.addWidget(QLabel("measurements: "), 25)
        measurement_layout.addWidget(self.measurements_cb, 75)
        layout.addLayout(measurement_layout)

        self.window_size_slider = QLabeledSlider()
        self.window_size_slider.setRange(
            1, int(np.nanmax(self.parent_window.data.FRAME.to_numpy()))
        )
        self.window_size_slider.setValue(3)
        window_layout = QHBoxLayout()
        window_layout.addWidget(QLabel("window size: "), 25)
        window_layout.addWidget(self.window_size_slider, 75)
        layout.addLayout(window_layout)

        self.backward_btn = QRadioButton("backward")
        self.bi_btn = QRadioButton("bi")
        self.bi_btn.click()
        self.forward_btn = QRadioButton("forward")
        self.mode_btn_group = QButtonGroup()
        self.mode_btn_group.addButton(self.backward_btn)
        self.mode_btn_group.addButton(self.bi_btn)
        self.mode_btn_group.addButton(self.forward_btn)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("mode: "), 25)
        mode_sublayout = QHBoxLayout()
        mode_sublayout.addWidget(self.backward_btn, 33, alignment=Qt.AlignCenter)
        mode_sublayout.addWidget(self.bi_btn, 33, alignment=Qt.AlignCenter)
        mode_sublayout.addWidget(self.forward_btn, 33, alignment=Qt.AlignCenter)
        mode_layout.addLayout(mode_sublayout, 75)
        layout.addLayout(mode_layout)

        self.submit_btn = QPushButton("Compute")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.compute_derivative_and_add_new_column)
        layout.addWidget(self.submit_btn, 30)

        self.setAttribute(Qt.WA_DeleteOnClose)

    def compute_derivative_and_add_new_column(self):

        if self.bi_btn.isChecked():
            mode = "bi"
        elif self.forward_btn.isChecked():
            mode = "forward"
        elif self.backward_btn.isChecked():
            mode = "backward"
        self.parent_window.data = differentiate_per_track(
            self.parent_window.data,
            self.measurements_cb.currentText(),
            window_size=self.window_size_slider.value(),
            mode=mode,
        )
        self.parent_window.model = PandasModel(self.parent_window.data)
        self.parent_window.table_view.setModel(self.parent_window.model)
        self.close()


class OperationOnColsWidget(CelldetectiveWidget):

    def __init__(self, parent_window, column1=None, column2=None, operation="divide"):

        super().__init__()
        self.parent_window = parent_window
        self.column1 = column1
        self.column2 = column2
        self.operation = operation

        self.setWindowTitle(self.operation)
        # Create the QComboBox and add some items
        center_window(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        self.col1_cb = QComboBox()
        self.col1_cb.addItems(list(self.parent_window.data.columns))
        if self.column1 is not None:
            idx = self.col1_cb.findText(self.column1)
            self.col1_cb.setCurrentIndex(idx)

        numerator_layout = QHBoxLayout()
        numerator_layout.addWidget(QLabel("column 1: "), 25)
        numerator_layout.addWidget(self.col1_cb, 75)
        layout.addLayout(numerator_layout)

        self.col2_cb = QComboBox()
        self.col2_cb.addItems(list(self.parent_window.data.columns))
        if self.column2 is not None:
            idx = self.col2_cb.findText(self.column2)
            self.col2_cb.setCurrentIndex(idx)

        denominator_layout = QHBoxLayout()
        denominator_layout.addWidget(QLabel("column 2: "), 25)
        denominator_layout.addWidget(self.col2_cb, 75)
        layout.addLayout(denominator_layout)

        self.submit_btn = QPushButton("Compute")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.compute)
        layout.addWidget(self.submit_btn, 30)

        self.setAttribute(Qt.WA_DeleteOnClose)

    def compute(self):

        test = self._check_cols_before_operation()
        if not test:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(
                f"Operation could not be performed, one of the column types is object..."
            )
            msg_box.setWindowTitle("Warning")
            msg_box.setStandardButtons(QMessageBox.Ok)
            return_value = msg_box.exec()
            if return_value == QMessageBox.Ok:
                return None
            else:
                return None
        else:
            if self.operation == "divide":
                name = f"{self.col1_txt}/{self.col2_txt}"
                with np.errstate(divide="ignore", invalid="ignore"):
                    res = np.true_divide(self.col1, self.col2)
                    res[res == np.inf] = np.nan
                    res[self.col1 != self.col1] = np.nan
                    res[self.col2 != self.col2] = np.nan
                    self.parent_window.data[name] = res

            elif self.operation == "multiply":
                name = f"{self.col1_txt}*{self.col2_txt}"
                res = np.multiply(self.col1, self.col2)

            elif self.operation == "add":
                name = f"{self.col1_txt}+{self.col2_txt}"
                res = np.add(self.col1, self.col2)

            elif self.operation == "subtract":
                name = f"{self.col1_txt}-{self.col2_txt}"
                res = np.subtract(self.col1, self.col2)
            else:
                logger.info(f"Operation {self.operation} not implemented...")

            self.parent_window.data[name] = res
            self.parent_window.model = PandasModel(self.parent_window.data)
            self.parent_window.table_view.setModel(self.parent_window.model)
            self.close()

    def _check_cols_before_operation(self):

        self.col1_txt = self.col1_cb.currentText()
        self.col2_txt = self.col2_cb.currentText()

        self.col1 = self.parent_window.data[self.col1_txt].to_numpy()
        self.col2 = self.parent_window.data[self.col2_txt].to_numpy()

        test = np.all([self.col1.dtype != "O", self.col2.dtype != "O"])

        return test


class CalibrateColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):

        super().__init__(title="Calibrate data", *args, **kwargs)

        self.floatValidator = QDoubleValidator()
        self.calibration_factor_le = QLineEdit("1")
        self.calibration_factor_le.setPlaceholderText(
            "multiplicative calibration factor..."
        )
        self.calibration_factor_le.setValidator(self.floatValidator)

        self.units_le = QLineEdit("um")
        self.units_le.setPlaceholderText("units...")

        self.calibration_factor_le.textChanged.connect(self.check_valid_params)
        self.units_le.textChanged.connect(self.check_valid_params)

        calib_layout = QHBoxLayout()
        calib_layout.addWidget(QLabel("calibration factor: "), 33)
        calib_layout.addWidget(self.calibration_factor_le, 66)
        self.sublayout.addLayout(calib_layout)

        units_layout = QHBoxLayout()
        units_layout.addWidget(QLabel("units: "), 33)
        units_layout.addWidget(self.units_le, 66)
        self.sublayout.addLayout(units_layout)

        # info_layout = QHBoxLayout()
        # info_layout.addWidget(QLabel('For reference: '))
        # self.sublayout.addLayout(info_layout)

        # info_layout2 = QHBoxLayout()
        # info_layout2.addWidget(QLabel(f'PxToUm = {self.parent_window.parent_window.parent_window.PxToUm}'), 50)
        # info_layout2.addWidget(QLabel(f'FrameToMin = {self.parent_window.parent_window.parent_window.FrameToMin}'), 50)
        # self.sublayout.addLayout(info_layout2)

    def check_valid_params(self):

        try:
            factor = float(self.calibration_factor_le.text().replace(",", "."))
            factor_valid = True
        except Exception as _:
            factor_valid = False

        if self.units_le.text() == "":
            units_valid = False
        else:
            units_valid = True

        if factor_valid and units_valid:
            self.submit_btn.setEnabled(True)
        else:
            self.submit_btn.setEnabled(False)

    def compute(self):
        self.parent_window.data[
            self.measurements_cb.currentText() + f"[{self.units_le.text()}]"
        ] = self.parent_window.data[self.measurements_cb.currentText()] * float(
            self.calibration_factor_le.text().replace(",", ".")
        )


class AbsColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):

        super().__init__(title="abs(.)", *args, **kwargs)

    def compute(self):
        self.parent_window.data["|" + self.measurements_cb.currentText() + "|"] = (
            self.parent_window.data[self.measurements_cb.currentText()].abs()
        )


class LogColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):

        super().__init__(title="log10(.)", *args, **kwargs)

    def compute(self):
        self.parent_window.data["log10(" + self.measurements_cb.currentText() + ")"] = (
            safe_log(self.parent_window.data[self.measurements_cb.currentText()].values)
        )
