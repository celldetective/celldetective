import numpy as np
from PyQt5.QtCore import QSize, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveProgressDialog
from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.gui.layouts.operation_layout import OperationLayout
from celldetective.preprocessing import correct_background_model
from celldetective.processes.background_correction import BackgroundCorrectionProcess
from celldetective.utils.image_loaders import auto_load_number_of_frames
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective import get_logger
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective.gui.viewers.threshold_viewer import ThresholdedStackVisualizer

logger = get_logger(__name__)


class BackgroundFitCorrectionLayout(QGridLayout, Styles):
    """docstring for ClassName"""

    def __init__(self, parent_window=None, *args):
        super().__init__(*args)

        self.parent_window = parent_window

        if hasattr(self.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window
        elif hasattr(self.parent_window.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window.parent_window
        else:
            self.attr_parent = (
                self.parent_window.parent_window.parent_window.parent_window
            )

        self.channel_names = self.attr_parent.exp_channels
        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.thresh_lbl = QLabel("Threshold: ")
        self.thresh_lbl.setToolTip(
            "Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation."
        )
        self.threshold_viewer_btn = QPushButton()
        self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.threshold_viewer_btn.setStyleSheet(self.button_select_all)
        self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)
        self.threshold_viewer_btn.setToolTip("Set the threshold graphically.")

        self.model_lbl = QLabel("Model: ")
        self.model_lbl.setToolTip("2D model to fit the background with.")
        self.models_cb = QComboBox()
        self.models_cb.addItems(["paraboloid", "plane"])
        self.models_cb.setToolTip("2D model to fit the background with.")

        self.corrected_stack_viewer = QPushButton("")
        self.corrected_stack_viewer.setStyleSheet(self.button_select_all)
        self.corrected_stack_viewer.setIcon(icon(MDI6.eye_outline, color="black"))
        self.corrected_stack_viewer.setToolTip("View corrected image")
        self.corrected_stack_viewer.clicked.connect(self.preview_correction)
        self.corrected_stack_viewer.setIconSize(QSize(20, 20))

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
                self.corrected_stack_viewer,
                self.add_correction_btn,
            ],
        )

        self.downsample_lbl = QLabel("Downsample: ")
        self.downsample_lbl.setToolTip(
            "Factor by which to downsample the image for fitting (for speed)."
        )
        self.downsample_le = QLineEdit("10")
        self.downsample_le.setValidator(QIntValidator())

    def add_to_layout(self):

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_lbl, 25)
        channel_layout.addWidget(self.channels_cb, 75)
        self.addLayout(channel_layout, 0, 0, 1, 3)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.thresh_lbl, 25)
        subthreshold_layout = QHBoxLayout()
        subthreshold_layout.addWidget(self.threshold_le, 95)
        subthreshold_layout.addWidget(self.threshold_viewer_btn, 5)

        threshold_layout.addLayout(subthreshold_layout, 75)
        self.addLayout(threshold_layout, 1, 0, 1, 3)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_lbl, 25)
        model_layout.addWidget(self.models_cb, 75)
        self.addLayout(model_layout, 2, 0, 1, 3)

        downsample_layout = QHBoxLayout()
        downsample_layout.addWidget(self.downsample_lbl, 25)
        downsample_layout.addWidget(self.downsample_le, 75)
        self.addLayout(downsample_layout, 3, 0, 1, 3)

        self.operation_layout = OperationLayout()
        self.addLayout(self.operation_layout, 4, 0, 1, 3)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.add_correction_btn, 95)
        correction_layout.addWidget(self.corrected_stack_viewer, 5)
        self.addLayout(correction_layout, 5, 0, 1, 3)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(verticalSpacer, 6, 0, 1, 3)

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

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "fit",
            "model": self.models_cb.currentText(),
            "threshold_on_std": self.threshold_le.get_threshold(),
            "operation": operation,
            "clip": clip,
            "downsample": int(self.downsample_le.text()),
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def set_threshold_graphically(self):

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

        if (
            self.attr_parent.well_list.isMultipleSelection()
            or not self.attr_parent.well_list.isAnySelected()
            or self.attr_parent.position_list.isMultipleSelection()
            or not self.attr_parent.position_list.isAnySelected()
        ):

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Please select a single position...")
            msgBox.setWindowTitle("Critical")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        self.attr_parent.locate_image()
        self.set_target_channel()

        subset_indices = None
        if (
            hasattr(self.attr_parent, "current_stack")
            and self.attr_parent.current_stack is not None
        ):

            n_frames = auto_load_number_of_frames(self.attr_parent.current_stack)
            if n_frames is not None:
                midpoint = int(n_frames // 2)
                n_channels = len(self.attr_parent.exp_channels)
                subset_indices = [midpoint * n_channels]

        process_args = {
            "exp_dir": self.attr_parent.exp_dir,
            "well_option": self.attr_parent.well_list.getSelectedIndices(),
            "position_option": self.attr_parent.position_list.getSelectedIndices(),
            "target_channel": self.channels_cb.currentText(),
            "model": self.models_cb.currentText(),
            "threshold_on_std": self.threshold_le.get_threshold(),
            "operation": operation,
            "clip": clip,
            "activation_protocol": [["gauss", 2], ["std", 4]],
            "downsample": int(self.downsample_le.text()),
            "subset_indices": subset_indices,
        }

        self.bg_progress = CelldetectiveProgressDialog(
            "Correcting background (Preview)...",
            "Cancel",
            0,
            0,
            None,
            window_title="Processing",
        )
        self.bg_progress.setRange(0, 0)

        self.preview_worker = PreviewWorker(
            BackgroundCorrectionProcess, process_args=process_args
        )

        def on_result(corrected_stack):

            if corrected_stack is not None:
                if subset_indices is not None and len(self.channel_names) > 0:
                    # Logic to extract the specific channel
                    if corrected_stack.ndim == 3 and corrected_stack.shape[0] == len(
                        self.channel_names
                    ):
                        # Shape likely (C, Y, X)
                        corrected_stack = corrected_stack[
                            self.channels_cb.currentIndex()
                        ]
                    elif corrected_stack.ndim == 4 and corrected_stack.shape[-1] == len(
                        self.channel_names
                    ):
                        # Shape likely (T, Y, X, C)
                        corrected_stack = corrected_stack[
                            ..., self.channels_cb.currentIndex()
                        ]

                    # Ensure (T=1, Y, X, C=1) for display or similar that StackVisualizer likes for single channel
                    if corrected_stack.ndim == 2:
                        corrected_stack = corrected_stack[np.newaxis, :, :, np.newaxis]
                    elif corrected_stack.ndim == 3:
                        # (1, Y, X)
                        corrected_stack = corrected_stack[:, :, :, np.newaxis]

                self.viewer = StackVisualizer(
                    stack=corrected_stack,
                    window_title="Corrected channel",
                    target_channel=0,
                    frame_slider=True,
                    contrast_slider=True,
                )
                self.viewer.show()
            else:
                print("Corrected stack could not be generated...")

        def on_finished():
            self.bg_progress.close()

        def on_error(msg):
            self.bg_progress.close()
            QMessageBox.critical(None, "Error", f"Correction failed: {msg}")

        self.preview_worker.result_ready.connect(on_result)
        self.preview_worker.finished.connect(on_finished)
        self.preview_worker.error.connect(on_error)
        self.bg_progress.canceled.connect(self.preview_worker.stop)

        self.preview_worker.start()
        self.bg_progress.exec_()


class PreviewWorker(QThread):
    finished = pyqtSignal()
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, process_class, process_args):
        super().__init__()
        # process_class is unused now as we call function directly
        self.process_args = process_args

    def run(self):
        try:
            result = correct_background_model(
                experiment=self.process_args["exp_dir"],
                well_option=self.process_args["well_option"],
                position_option=self.process_args["position_option"],
                target_channel=self.process_args["target_channel"],
                model=self.process_args["model"],
                threshold_on_std=self.process_args["threshold_on_std"],
                operation=self.process_args["operation"],
                clip=self.process_args["clip"],
                export=False,
                return_stacks=True,
                activation_protocol=self.process_args["activation_protocol"],
                downsample=self.process_args["downsample"],
                subset_indices=self.process_args["subset_indices"],
                show_progress_per_well=False,
                show_progress_per_pos=False,
            )

            if result is not None and len(result) > 0:
                self.result_ready.emit(result[0])

        except Exception as e:
            self.error.emit(str(e))

        self.finished.emit()

    def stop(self):
        self.quit()
