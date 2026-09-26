import os
from typing import Optional

import numpy as np

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QComboBox,
    QButtonGroup,
    QRadioButton,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QHBoxLayout,
    QMessageBox,
    QDialog,
    QMainWindow,
)
from fonticon_mdi6 import MDI6
from superqt import QLabeledRangeSlider
from celldetective.gui.base.sliders import QLabeledSlider, QLabeledDoubleRangeSlider
from superqt.fonticon import icon
from tifffile import imread

from celldetective.gui.base.components import (
    CelldetectiveProgressDialog,
    generic_message,
)
from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import (
    QuickSliderLayout,
    RadiusLineEdit,
    ThresholdLineEdit,
)
from celldetective.gui.layouts.operation_layout import OperationLayout
from celldetective.processes.background_correction import BackgroundCorrectionProcess
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective import get_logger
from celldetective.gui.base.threads import start_tracked

logger = get_logger(__name__)

# Frames corrected by the preview, spread over the movie.
PREVIEW_FRAMES = 5


class BackgroundModelFreeCorrectionLayout(QGridLayout, Styles):
    """docstring for ClassName"""

    def __init__(self, parent_window: Optional[QMainWindow] = None) -> None:
        """
        Initialize the BackgroundModelFreeCorrectionLayout.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The parent window.
        *args
            Variable length argument list.
        """
        super().__init__()

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
        """Generate the widgets."""

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
            slider_tooltip="Range the coefficient scaling the background intensity\n"
            "is kept within. The optimal coefficient is computed exactly\n"
            "and set to the closest bound if it falls outside.",
        )

        self.radius_lbl = QLabel("Fit radius: ")
        self.radius_lbl.setToolTip(
            "Radius [px] of the disk centred on the image over which\n"
            "the coefficient is optimized. Pixels outside (e.g. a diaphragm\n"
            "close to the camera black level) are ignored by the fit\n"
            "but still corrected. Leave empty to use the full frame."
        )
        self.radius_le = RadiusLineEdit()

        self.radius_viewer_btn = QPushButton()
        self.radius_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.radius_viewer_btn.setStyleSheet(self.button_select_all)
        self.radius_viewer_btn.setToolTip(
            "Tune the fit radius on a frame of the current position."
        )
        self.radius_viewer_btn.clicked.connect(self.open_radius_viewer)

        self.coef_widgets = [
            self.coef_range_layout.qlabel,
            self.coef_range_slider,
            self.radius_lbl,
            self.radius_le,
            self.radius_viewer_btn,
        ]
        for c in self.coef_widgets:
            c.setEnabled(False)

        self.interpolate_check = QCheckBox("interpolate NaNs")

    def add_to_layout(self):
        """Add widgets to the layout."""

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

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(self.radius_lbl, 25)
        radius_field_layout = QHBoxLayout()
        radius_field_layout.addWidget(self.radius_le, 95)
        radius_field_layout.addWidget(self.radius_viewer_btn, 5)
        radius_layout.addLayout(radius_field_layout, 75)
        self.addLayout(radius_layout, 7, 0, 1, 3)

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
        """Add instructions to the parent protocol list."""

        if not self.generate_instructions():
            return
        self.parent_window.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_list.addItem(correction_description)

    def generate_instructions(self) -> bool:
        """
        Generate the instructions dictionary.

        Returns
        -------
        bool
            False if a parameter is invalid, in which case a warning is shown.
        """

        parameters = self.correction_parameters()
        if parameters is None:
            return False
        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "model-free",
            **parameters,
        }
        return True

    def correction_parameters(self) -> Optional[dict]:
        """
        Read the correction parameters shared by the protocol and the preview.

        Returns
        -------
        dict or None
            The parameters, or None if one is invalid, in which case a warning is shown.
        """

        mode = "tiles" if self.tiles_rb.isChecked() else "timeseries"

        if self.regress_cb.isChecked():
            optimize_option = True
            opt_coef_range = self.coef_range_slider.value()
            valid_radius, opt_radius = self.radius_le.radius_or_warn()
            if not valid_radius:
                return None
        else:
            optimize_option = False
            opt_coef_range = None
            opt_radius = None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
            clip = self.operation_layout.clip_btn.isChecked()
        else:
            operation = "divide"
            clip = False

        valid_offset, offset = self.offset_or_warn()
        if not valid_offset:
            return None

        return {
            "threshold_on_std": self.threshold_le.get_threshold(),
            "frame_range": self.frame_range_slider.value(),
            "mode": mode,
            "optimize_option": optimize_option,
            "opt_coef_range": opt_coef_range,
            "opt_radius": opt_radius,
            "operation": operation,
            "clip": clip,
            "offset": offset,
            "fix_nan": self.interpolate_check.isChecked(),
        }

    def offset_or_warn(self) -> tuple:
        """
        Read the camera offset.

        Returns
        -------
        tuple
            ``(valid, offset)``: offset is None if the field is empty. If the field is not a
            number, a warning is shown and valid is False.
        """

        offset_text = self.camera_offset_le.text().strip().replace(",", ".")
        try:
            return True, float(offset_text) if offset_text else None
        except ValueError:
            # Intermediate input such as "-" or "1e" that the validator lets through.
            generic_message("The offset must be a number, or empty.", "warning")
            return False, None

    def open_radius_viewer(self):
        """Open a frame of the current position to tune the fit radius."""
        from celldetective.gui.viewers.registration_roi_viewer import DiskROIViewer

        self.attr_parent.locate_image()
        if self.attr_parent.current_stack is None:
            return
        self.set_target_channel()
        self.viewer = DiskROIViewer(
            self,
            stack_path=self.attr_parent.current_stack,
            channel_names=self.channel_names,
            n_channels=len(self.channel_names),
            channel_cb=True,
            target_channel=self.target_channel,
            window_title="Coefficient fit radius",
            initial_radius=self.radius_le.radius_or_none(),
        )
        self.viewer.show()

    def set_target_channel(self):
        """Set the target channel index."""

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def set_threshold_graphically(self):
        """Open the threshold viewer to set the threshold graphically."""
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
                channel_names=self.channel_names,
                target_channel=self.target_channel,
                window_title="Set the exclusion threshold",
            )
            self.viewer.show()

    def preview_correction(self):
        """Preview the background correction on the current image."""
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

        parameters = self.correction_parameters()
        if parameters is None:
            return None

        self.attr_parent.locate_image()
        if self.attr_parent.current_stack is None:
            return None
        process_args = {
            "exp_dir": self.attr_parent.exp_dir,
            "well_option": self.attr_parent.well_list.getSelectedIndices(),
            "position_option": self.attr_parent.position_list.getSelectedIndices(),
            "target_channel": self.channels_cb.currentText(),
            **parameters,
            "activation_protocol": [["gauss", 2], ["std", 4]],
            "correction_type": "model-free",
            "subset_indices": self.preview_frame_indices(),
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
                logger.warning("Corrected stack could not be generated... No stack available...")
        else:
            logger.info("Background correction cancelled.")

    def preview_frame_indices(self) -> Optional[list]:
        """
        Frames of the current position to correct for the preview.

        Returns
        -------
        list of int or None
            The absolute frame indices (IFDs) of up to ``PREVIEW_FRAMES`` frames spread over
            the movie, or None (whole movie) if its length cannot be read.
        """
        from celldetective.utils.image_loaders import auto_load_number_of_frames

        stack = getattr(self.attr_parent, "current_stack", None)
        n_frames = auto_load_number_of_frames(stack) if stack is not None else None
        if not n_frames:
            return None
        frames = np.unique(
            np.linspace(0, n_frames - 1, min(PREVIEW_FRAMES, n_frames)).round()
        )
        return [int(t) * len(self.channel_names) for t in frames]

    def activate_time_range(self):
        """Enable or disable time range options based on acquisition mode."""

        if self.timeseries_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(True)
        elif self.tiles_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(False)

    def activate_coef_options(self):
        """Enable or disable coefficient options based on regression checkbox."""

        if self.regress_cb.isChecked():
            for c in self.coef_widgets:
                c.setEnabled(True)
        else:
            for c in self.coef_widgets:
                c.setEnabled(False)

    def estimate_bg(self):
        """Estimate the background and display the result."""

        mode = "tiles" if self.tiles_rb.isChecked() else "timeseries"
        # The background as applied: offset subtracted, NaNs interpolated if asked.
        valid_offset, offset = self.offset_or_warn()
        if not valid_offset:
            return

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
            offset=offset,
            fix_nan=self.interpolate_check.isChecked(),
        )
        from celldetective.gui.viewers.base_viewer import StackVisualizer

        self.bg_worker.progress.connect(self.bg_progress.setValue)
        self.bg_worker.status_update.connect(self.bg_progress.setLabelText)

        def on_finished(bg: list) -> None:
            """
            Handle background estimation completion.

            Parameters
            ----------
            bg : list
                The background estimation result.
            """
            self.bg_progress.blockSignals(True)
            self.bg_progress.close()
            if self.bg_worker._is_cancelled:
                logger.info("Background estimation cancelled.")
                return

            if bg is None or (bg and bg[0] is None):
                # None for the well whose background could not be computed.
                QMessageBox.critical(None, "Error", "Background estimation failed.")
            elif bg:
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

        self.bg_worker.finished_with_result.connect(on_finished)
        self.bg_progress.canceled.connect(self.bg_worker.stop)

        start_tracked(self.bg_worker)

class BackgroundEstimatorThread(QThread):
    progress = pyqtSignal(int)
    finished_with_result = pyqtSignal(object)
    status_update = pyqtSignal(str)

    def __init__(
        self,
        exp_dir: str,
        well_idx: int,
        frame_range: tuple,
        channel: str,
        threshold: float,
        mode: str,
        offset: Optional[float] = None,
        fix_nan: bool = False,
    ) -> None:
        """
        Initialize the BackgroundEstimatorThread.

        Parameters
        ----------
        exp_dir : str
            The experiment directory.
        well_idx : int
            The well index.
        frame_range : tuple
            The frame range.
        channel : str
            The target channel.
        threshold : float
            The threshold on STD.
        mode : str
            The acquisition mode ('timeseries' or 'tiles').
        offset : float, optional
            The camera offset subtracted from the background.
        fix_nan : bool, optional
            Whether to interpolate the NaNs of the background.
        """
        super().__init__()
        self.exp_dir = exp_dir
        self.well_idx = well_idx
        self.frame_range = frame_range
        self.channel = channel
        self.threshold = threshold
        self.mode = mode
        self.offset = offset
        self.fix_nan = fix_nan
        self._is_cancelled = False

    def stop(self):
        """Stop the thread."""
        self._is_cancelled = True

    def run(self):
        """Run the background estimation."""
        from celldetective.preprocessing import estimate_background_per_condition

        self.first_update = True

        def callback(**kwargs):
            """Progress callback."""
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
                frame_range=list(self.frame_range),
                target_channel=self.channel,
                show_progress_per_pos=False,
                threshold_on_std=self.threshold,
                mode=self.mode,
                offset=self.offset,
                fix_nan=self.fix_nan,
                progress_callback=callback,
            )
            if not self._is_cancelled:
                self.finished_with_result.emit(bg)
            else:
                self.finished_with_result.emit(None)
        except Exception as e:
            logger.error(f"Error in background estimation thread: {e}")
            self.finished_with_result.emit(None)
