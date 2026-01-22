import os

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtWidgets import QGridLayout, QLabel, QComboBox, QButtonGroup, QRadioButton, QPushButton, QCheckBox, \
    QLineEdit, QHBoxLayout, QMessageBox, QDialog
from fonticon_mdi6 import MDI6
from superqt import QLabeledRangeSlider, QLabeledSlider, QLabeledDoubleRangeSlider
from superqt.fonticon import icon
from tifffile import imread

from celldetective.gui.base.components import CelldetectiveProgressDialog
from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import ThresholdLineEdit, QuickSliderLayout
from celldetective.gui.layouts.operation_layout import OperationLayout
from celldetective.processes.background_correction import BackgroundCorrectionProcess
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective import get_logger

logger = get_logger(__name__)

class BackgroundModelFreeCorrectionLayout(QGridLayout, Styles):
    """docstring for ClassName"""

    def __init__(self, parent_window=None, *args):
        super().__init__(*args)

        self.parent_window = parent_window

        if hasattr(self.parent_window.parent_window, "exp_config"):
            self.attr_parent = self.parent_window.parent_window
        else:
            self.attr_parent = self.parent_window.parent_window.parent_window

        self.channel_names = self.attr_parent.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.acquistion_lbl = QLabel("Stack mode: ")
        self.acq_mode_group = QButtonGroup()
        self.timeseries_rb = QRadioButton("timeseries")
        self.timeseries_rb.setChecked(True)
        self.tiles_rb = QRadioButton("tiles")
        self.acq_mode_group.addButton(self.timeseries_rb, 0)
        self.acq_mode_group.addButton(self.tiles_rb, 1)

        self.frame_range_slider = QLabeledRangeSlider(parent=None)

        self.timeseries_rb.toggled.connect(self.activate_time_range)
        self.tiles_rb.toggled.connect(self.activate_time_range)

        self.thresh_lbl = QLabel("Threshold: ")
        self.thresh_lbl.setToolTip(
            "Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation."
        )
        self.threshold_viewer_btn = QPushButton()
        self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.threshold_viewer_btn.setStyleSheet(self.button_select_all)
        self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)

        self.background_viewer_btn = QPushButton()
        self.background_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.background_viewer_btn.setStyleSheet(self.button_select_all)
        self.background_viewer_btn.setToolTip("View reconstructed background.")

        self.corrected_stack_viewer_btn = QPushButton("")
        self.corrected_stack_viewer_btn.setStyleSheet(self.button_select_all)
        self.corrected_stack_viewer_btn.setIcon(icon(MDI6.eye_outline, color="black"))
        self.corrected_stack_viewer_btn.setToolTip("View corrected image")
        self.corrected_stack_viewer_btn.clicked.connect(self.preview_correction)
        self.corrected_stack_viewer_btn.setIconSize(QSize(20, 20))

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.threshold_le = ThresholdLineEdit(
            init_value=2,
            connected_buttons=[
                self.threshold_viewer_btn,
                self.background_viewer_btn,
                self.corrected_stack_viewer_btn,
                self.add_correction_btn,
            ],
        )

        self.well_slider = QLabeledSlider(parent=None)

        self.background_viewer_btn.clicked.connect(self.estimate_bg)

        self.regress_cb = QCheckBox("Optimize for each frame?")
        self.regress_cb.toggled.connect(self.activate_coef_options)
        self.regress_cb.setChecked(False)

        self.coef_range_slider = QLabeledDoubleRangeSlider(parent=None)
        self.coef_range_layout = QuickSliderLayout(
            label="Coef. range: ",
            slider=self.coef_range_slider,
            slider_initial_value=(0.95, 1.05),
            slider_range=(0.75, 1.25),
            slider_tooltip="Coefficient range to increase or decrease the background intensity level...",
        )

        self.nbr_coefs_lbl = QLabel("Nbr of coefs: ")
        self.nbr_coefs_lbl.setToolTip(
            "Number of coefficients to be tested within range.\nThe more, the slower."
        )

        self.nbr_coef_le = QLineEdit()
        self.nbr_coef_le.setText("100")
        self.nbr_coef_le.setValidator(QIntValidator())
        self.nbr_coef_le.setPlaceholderText("nbr of coefs")

        self.coef_widgets = [
            self.coef_range_layout.qlabel,
            self.coef_range_slider,
            self.nbr_coefs_lbl,
            self.nbr_coef_le,
        ]
        for c in self.coef_widgets:
            c.setEnabled(False)

        self.interpolate_check = QCheckBox("interpolate NaNs")

    def add_to_layout(self):

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_lbl, 25)
        channel_layout.addWidget(self.channels_cb, 75)
        self.addLayout(channel_layout, 0, 0, 1, 3)

        acquisition_layout = QHBoxLayout()
        acquisition_layout.addWidget(self.acquistion_lbl, 25)
        acquisition_layout.addWidget(
            self.timeseries_rb, 75 // 2, alignment=Qt.AlignCenter
        )
        acquisition_layout.addWidget(self.tiles_rb, 75 // 2, alignment=Qt.AlignCenter)
        self.addLayout(acquisition_layout, 1, 0, 1, 3)

        frame_selection_layout = QuickSliderLayout(
            label="Time range: ",
            slider=self.frame_range_slider,
            slider_initial_value=(0, 5),
            slider_range=(0, self.attr_parent.len_movie),
            slider_tooltip="frame [#]",
            decimal_option=False,
        )
        frame_selection_layout.qlabel.setToolTip(
            "Frame range for which the background\nis most likely to be observed."
        )
        self.time_range_options = [
            self.frame_range_slider,
            frame_selection_layout.qlabel,
        ]
        self.addLayout(frame_selection_layout, 2, 0, 1, 3)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.thresh_lbl, 25)
        subthreshold_layout = QHBoxLayout()
        subthreshold_layout.addWidget(self.threshold_le, 95)
        subthreshold_layout.addWidget(self.threshold_viewer_btn, 5)
        threshold_layout.addLayout(subthreshold_layout, 75)
        self.addLayout(threshold_layout, 3, 0, 1, 3)

        background_layout = QuickSliderLayout(
            label="QC for well: ",
            slider=self.well_slider,
            slider_initial_value=1,
            slider_range=(1, len(self.attr_parent.wells)),
            slider_tooltip="well [#]",
            decimal_option=False,
            layout_ratio=(0.25, 0.70),
        )
        background_layout.addWidget(self.background_viewer_btn, 5)
        self.addLayout(background_layout, 4, 0, 1, 3)

        self.addWidget(self.regress_cb, 5, 0, 1, 3)

        self.addLayout(self.coef_range_layout, 6, 0, 1, 3)

        coef_nbr_layout = QHBoxLayout()
        coef_nbr_layout.addWidget(self.nbr_coefs_lbl, 25)
        coef_nbr_layout.addWidget(self.nbr_coef_le, 75)
        self.addLayout(coef_nbr_layout, 7, 0, 1, 3)

        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset: "), 25)
        self.camera_offset_le = QLineEdit("0")
        self.camera_offset_le.setPlaceholderText("camera black level")
        self.camera_offset_le.setValidator(QDoubleValidator())
        offset_layout.addWidget(self.camera_offset_le, 75)
        self.addLayout(offset_layout, 8, 0, 1, 3)

        self.operation_layout = OperationLayout()
        self.addLayout(self.operation_layout, 9, 0, 1, 3)

        self.addWidget(self.interpolate_check, 10, 0, 1, 1)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.add_correction_btn, 95)
        correction_layout.addWidget(self.corrected_stack_viewer_btn, 5)
        self.addLayout(correction_layout, 11, 0, 1, 3)

        # verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.addItem(verticalSpacer, 5, 0, 1, 3)

    def add_instructions_to_parent_list(self):

        self.generate_instructions()
        self.parent_window.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_list.addItem(correction_description)

    def generate_instructions(self):

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"

        if self.regress_cb.isChecked():
            optimize_option = True
            opt_coef_range = self.coef_range_slider.value()
            opt_coef_nbr = int(self.nbr_coef_le.text())
        else:
            optimize_option = False
            opt_coef_range = None
            opt_coef_nbr = None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        if self.camera_offset_le.text() == "":
            offset = None
        else:
            offset = float(self.camera_offset_le.text().replace(",", "."))

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "model-free",
            "threshold_on_std": self.threshold_le.get_threshold(),
            "frame_range": self.frame_range_slider.value(),
            "mode": mode,
            "optimize_option": optimize_option,
            "opt_coef_range": opt_coef_range,
            "opt_coef_nbr": opt_coef_nbr,
            "operation": operation,
            "clip": clip,
            "offset": offset,
            "fix_nan": self.interpolate_check.isChecked(),
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def set_threshold_graphically(self):
        from celldetective.gui.viewers.threshold_viewer import (
            ThresholdedStackVisualizer,
        )

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:
            self.viewer = ThresholdedStackVisualizer(
                initial_threshold=thresh,
                parent_le=self.threshold_le,
                preprocessing=[["gauss", 2], ["std", 4]],
                stack_path=self.attr_parent.current_stack,
                n_channels=len(self.channel_names),
                target_channel=self.target_channel,
                window_title="Set the exclusion threshold",
            )
            self.viewer.show()

    def preview_correction(self):
        from celldetective.gui.viewers.base_viewer import StackVisualizer

        if (
            self.attr_parent.well_list.isMultipleSelection()
            or not self.attr_parent.well_list.isAnySelected()
            or self.attr_parent.position_list.isMultipleSelection()
            or not self.attr_parent.position_list.isAnySelected()
        ):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a single position...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"
        else:
            mode = "tiles"

        if self.regress_cb.isChecked():
            optimize_option = True
            opt_coef_range = self.coef_range_slider.value()
            opt_coef_nbr = int(self.nbr_coef_le.text())
        else:
            optimize_option = False
            opt_coef_range = None
            opt_coef_nbr = None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        process_args = {
            "exp_dir": self.attr_parent.exp_dir,
            "well_option": self.attr_parent.well_list.getSelectedIndices(),
            "position_option": self.attr_parent.position_list.getSelectedIndices(),
            "target_channel": self.channels_cb.currentText(),
            "mode": mode,
            "threshold_on_std": self.threshold_le.get_threshold(),
            "frame_range": self.frame_range_slider.value(),
            "optimize_option": optimize_option,
            "opt_coef_range": opt_coef_range,
            "opt_coef_nbr": opt_coef_nbr,
            "operation": operation,
            "clip": clip,
            "fix_nan": self.interpolate_check.isChecked(),
            "activation_protocol": [["gauss", 2], ["std", 4]],
            "correction_type": "model-free",
        }
        from celldetective.gui.workers import ProgressWindow

        self.job = ProgressWindow(
            BackgroundCorrectionProcess,
            parent_window=self,
            title="Background Correction",
            position_info=False,
            process_args=process_args,
        )
        result = self.job.exec_()

        if result == QDialog.Accepted:
            temp_path = os.path.join(
                self.attr_parent.exp_dir, "temp_corrected_stack.tif"
            )
            if os.path.exists(temp_path):
                corrected_stack = imread(temp_path)
                os.remove(temp_path)

                self.viewer = StackVisualizer(
                    stack=corrected_stack,
                    window_title="Corrected channel",
                    frame_slider=True,
                    contrast_slider=True,
                    target_channel=self.channels_cb.currentIndex(),
                )
                self.viewer.show()
            else:
                print("Corrected stack could not be generated... No stack available...")
        else:
            print("Background correction cancelled.")

    def activate_time_range(self):

        if self.timeseries_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(True)
        elif self.tiles_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(False)

    def activate_coef_options(self):

        if self.regress_cb.isChecked():
            for c in self.coef_widgets:
                c.setEnabled(True)
        else:
            for c in self.coef_widgets:
                c.setEnabled(False)

    def estimate_bg(self):

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"
        else:
            mode = "tiles"

        # Create progress dialog
        window_title = "Background reconstruction"
        self.bg_progress = CelldetectiveProgressDialog(
            "Loading libraries...", "Cancel", 0, 100, None, window_title=window_title
        )

        self.bg_worker = BackgroundEstimatorThread(
            self.attr_parent.exp_dir,
            self.well_slider.value() - 1,
            self.frame_range_slider.value(),
            self.channels_cb.currentText(),
            self.threshold_le.get_threshold(),
            mode,
        )
        from celldetective.gui.viewers.base_viewer import StackVisualizer

        self.bg_worker.progress.connect(self.bg_progress.setValue)
        self.bg_worker.status_update.connect(self.bg_progress.setLabelText)

        def on_finished(bg):
            self.bg_progress.blockSignals(True)
            self.bg_progress.close()
            if self.bg_worker._is_cancelled:
                logger.info("Background estimation cancelled.")
                return

            if bg and len(bg) > 0:
                bg_img = bg[0]["bg"]
                if len(bg_img) > 0:
                    self.viewer = StackVisualizer(
                        stack=[bg_img],
                        window_title="Reconstructed background",
                        frame_slider=False,
                    )
                    self.viewer.show()
                else:
                    QMessageBox.warning(
                        None, "Warning", "Background estimation returned empty image."
                    )
            elif bg is None:
                QMessageBox.critical(None, "Error", "Background estimation failed.")

        self.bg_worker.finished_with_result.connect(on_finished)
        self.bg_progress.canceled.connect(self.bg_worker.stop)

        self.bg_worker.start()


class BackgroundEstimatorThread(QThread):
    progress = pyqtSignal(int)
    finished_with_result = pyqtSignal(object)
    status_update = pyqtSignal(str)

    def __init__(self, exp_dir, well_idx, frame_range, channel, threshold, mode):
        super().__init__()
        self.exp_dir = exp_dir
        self.well_idx = well_idx
        self.frame_range = frame_range
        self.channel = channel
        self.threshold = threshold
        self.mode = mode
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        from celldetective.preprocessing import estimate_background_per_condition

        self.first_update = True

        def callback(**kwargs):
            if self._is_cancelled:
                return False

            if self.first_update:
                self.status_update.emit("Estimating background...")
                self.first_update = False

            if kwargs.get("level") == "position":

                iter_ = kwargs.get("iter", 0)
                total = kwargs.get("total", 1)
                # Avoid division by zero
                if total > 0:
                    p = int((iter_ / total) * 100)
                    self.progress.emit(p)
            return True

        try:
            bg = estimate_background_per_condition(
                self.exp_dir,
                well_option=self.well_idx,
                frame_range=self.frame_range,
                target_channel=self.channel,
                show_progress_per_pos=False,
                threshold_on_std=self.threshold,
                mode=self.mode,
                progress_callback=callback,
            )
            if not self._is_cancelled:
                self.finished_with_result.emit(bg)
            else:
                self.finished_with_result.emit(None)
        except Exception as e:
            print(f"Error in background estimation thread: {e}")
            self.finished_with_result.emit(None)
