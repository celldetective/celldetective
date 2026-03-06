from typing import Optional

import pandas as pd
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
    QMainWindow,
)
from superqt import QLabeledSlider

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import PandasModel, GenericOpColWidget
from celldetective import get_logger
from celldetective.utils.maths import differentiate_per_track, safe_log

logger = get_logger(__name__)


class DifferentiateColWidget(CelldetectiveWidget):

    def __init__(
        self, parent_window: QMainWindow, column: Optional[str] = None
    ) -> None:
        """
        Initialize the DifferentiateColWidget.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        column : str, optional
            The column to differentiate.
        """

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
        """Compute the derivative and add as a new column."""

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

    def __init__(
        self,
        parent_window: QMainWindow,
        column1: Optional[str] = None,
        column2: Optional[str] = None,
        operation: str = "divide",
    ) -> None:
        """
        Initialize the OperationOnColsWidget.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        column1 : str, optional
            First column name.
        column2 : str, optional
            Second column name.
        operation : str, optional
            Operation to perform ("divide", "multiply", "add", "subtract").
        """

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
        """Perform the operation and add result to table."""

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
        """
        Check if columns are valid for operation.

        Returns
        -------
        bool
            True if columns are numeric, False otherwise.
        """

        self.col1_txt = self.col1_cb.currentText()
        self.col2_txt = self.col2_cb.currentText()

        self.col1 = self.parent_window.data[self.col1_txt].to_numpy()
        self.col2 = self.parent_window.data[self.col2_txt].to_numpy()

        test = np.all([self.col1.dtype != "O", self.col2.dtype != "O"])

        return test


class CalibrateColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):
        """
        Initialize the CalibrateColWidget.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

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
        """Check if calibration parameters are valid."""

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
        """Apply calibration to the selected column."""
        self.parent_window.data[
            self.measurements_cb.currentText() + f"[{self.units_le.text()}]"
        ] = self.parent_window.data[self.measurements_cb.currentText()] * float(
            self.calibration_factor_le.text().replace(",", ".")
        )


class AbsColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):
        """
        Initialize the AbsColWidget.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(title="abs(.)", *args, **kwargs)

    def compute(self):
        """Compute absolute value of the column."""
        self.parent_window.data["|" + self.measurements_cb.currentText() + "|"] = (
            self.parent_window.data[self.measurements_cb.currentText()].abs()
        )


class LogColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):
        """
        Initialize the LogColWidget.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(title="log10(.)", *args, **kwargs)

    def compute(self):
        """Compute log10 of the column."""
        self.parent_window.data["log10(" + self.measurements_cb.currentText() + ")"] = (
            safe_log(self.parent_window.data[self.measurements_cb.currentText()].values)
        )


class BinColWidget(GenericOpColWidget):

    def __init__(self, *args, **kwargs):
        """
        Initialize the BinColWidget.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(title="Bin data", *args, **kwargs)

        self.floatValidator = QDoubleValidator()
        self.floatValidator.setBottom(0)  # Width should be positive

        self.width_le = QLineEdit("1.0")
        self.width_le.setPlaceholderText("bin width...")
        self.width_le.setValidator(self.floatValidator)
        self.width_le.textChanged.connect(self.check_valid_params)

        self.scale_linear_btn = QRadioButton("linear")
        self.scale_linear_btn.setChecked(True)
        self.scale_log_btn = QRadioButton("log")

        self.scale_group = QButtonGroup()
        self.scale_group.addButton(self.scale_linear_btn)
        self.scale_group.addButton(self.scale_log_btn)

        self.min_le = QLineEdit("")
        self.min_le.setPlaceholderText("auto min...")
        self.min_le.setValidator(QDoubleValidator())
        self.min_le.textChanged.connect(self.check_valid_params)

        self.max_le = QLineEdit("")
        self.max_le.setPlaceholderText("auto max...")
        self.max_le.setValidator(QDoubleValidator())
        self.max_le.textChanged.connect(self.check_valid_params)

        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("bin width: "), 33)
        width_layout.addWidget(self.width_le, 66)
        self.sublayout.addLayout(width_layout)

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("scale: "), 33)
        scale_sublayout = QHBoxLayout()
        scale_sublayout.addWidget(self.scale_linear_btn, 50, alignment=Qt.AlignCenter)
        scale_sublayout.addWidget(self.scale_log_btn, 50, alignment=Qt.AlignCenter)
        scale_layout.addLayout(scale_sublayout, 66)
        self.sublayout.addLayout(scale_layout)

        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("min loop edge: "), 33)
        min_layout.addWidget(self.min_le, 66)
        self.sublayout.addLayout(min_layout)

        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("max loop edge: "), 33)
        max_layout.addWidget(self.max_le, 66)
        self.sublayout.addLayout(max_layout)

        # Additional validation on measurements cb change
        self.measurements_cb.currentIndexChanged.connect(self.update_limits)
        self.measurements_cb.currentIndexChanged.connect(self.check_valid_params)

        # Fire initial sync if an item is already selected
        if self.measurements_cb.count() > 0:
            self.update_limits()
            self.check_valid_params()

    def update_limits(self):
        """Pre-fill min and max limits based on the selected column."""
        if not hasattr(self, "min_le") or not hasattr(self, "max_le"):
            return

        try:
            col = self.measurements_cb.currentText()
            if pd.api.types.is_numeric_dtype(self.parent_window.data[col]):
                data = self.parent_window.data[col].values
                min_val = np.nanmin(data)
                max_val = np.nanmax(data)
                self.min_le.setText(str(min_val))
                self.max_le.setText(str(max_val))
            else:
                self.min_le.setText("")
                self.max_le.setText("")
        except Exception as _:
            pass

    def check_valid_params(self):
        """Check if binning parameters are valid."""

        try:
            width = float(self.width_le.text().replace(",", "."))
            width_valid = width > 0
        except Exception as _:
            width_valid = False

        min_valid = True
        try:
            if self.min_le.text() != "":
                float(self.min_le.text().replace(",", "."))
        except Exception as _:
            min_valid = False

        max_valid = True
        try:
            if self.max_le.text() != "":
                float(self.max_le.text().replace(",", "."))
        except Exception as _:
            max_valid = False

        data_valid = False
        try:
            col = self.measurements_cb.currentText()
            if col != "":
                # check the col is numeric to allow binning
                if pd.api.types.is_numeric_dtype(self.parent_window.data[col]):
                    data_valid = True
        except Exception as _:
            pass

        if not hasattr(self, "submit_btn"):
            return

        if width_valid and min_valid and max_valid and data_valid:
            self.submit_btn.setEnabled(True)
        else:
            self.submit_btn.setEnabled(False)

    def compute(self):
        """Apply binning to the selected column."""

        col = self.measurements_cb.currentText()
        data = self.parent_window.data[col].values

        width = float(self.width_le.text().replace(",", "."))
        scale = "linear" if self.scale_linear_btn.isChecked() else "log"

        min_val = None
        if self.min_le.text() != "":
            min_val = float(self.min_le.text().replace(",", "."))
        else:
            min_val = np.nanmin(data)

        max_val = None
        if self.max_le.text() != "":
            max_val = float(self.max_le.text().replace(",", "."))
        else:
            max_val = np.nanmax(data)

        # Clip data
        clipped = np.clip(data, min_val, max_val)

        if scale == "linear":
            binned = np.round(clipped / width) * width
        elif scale == "log":
            # Create a mask for positive values
            mask = clipped > 0
            binned = np.zeros_like(clipped, dtype=float)

            binned[mask] = 10 ** (np.round(np.log10(clipped[mask]) / width) * width)

            # Handle non-positive values (e.g., 0 stays 0)
            binned[~mask] = clipped[~mask]

        name = f"{col}_binned_{width}_{scale}"
        self.parent_window.data[name] = binned
