import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import QSize, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QGridLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QCheckBox,
    QLineEdit,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QMessageBox,
    QMainWindow,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveProgressDialog
from celldetective.gui.base.styles import Styles
from celldetective.gui.layouts.operation_layout import OperationLayout
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective.utils.image_loaders import auto_load_number_of_frames, load_frames
from celldetective import get_logger

logger = get_logger(__name__)


class BackgroundRollingBallCorrectionLayout(QGridLayout, Styles):
    """Rolling ball background correction layout (ImageJ-style)."""

    def __init__(self, parent_window: Optional[QMainWindow] = None) -> None:
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

        self.radius_lbl = QLabel("Radius (px): ")
        self.radius_lbl.setToolTip(
            "Rolling ball radius in pixels.\n"
            "Should be at least as large as the largest cell diameter.\n"
            "The shrink factor is chosen automatically:\n"
            "  radius ≤ 10  →  no shrink\n"
            "  radius ≤ 50  →  2×\n"
            "  radius ≤ 100 →  4×\n"
            "  radius > 100 →  8×"
        )
        self.radius_le = QLineEdit("100")
        self.radius_le.setValidator(QDoubleValidator(1.0, 10000.0, 1))
        self.radius_le.setPlaceholderText("e.g. 100")

        # Quick background preview button — runs on one frame, very fast
        self.bg_viewer_btn = QPushButton()
        self.bg_viewer_btn.setStyleSheet(self.button_select_all)
        self.bg_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.bg_viewer_btn.setToolTip(
            "Preview the estimated background for the current radius.\n"
            "Runs on the middle frame only — useful for tuning radius."
        )
        self.bg_viewer_btn.setIconSize(QSize(20, 20))
        self.bg_viewer_btn.clicked.connect(self.preview_background)

        self.light_background_cb = QCheckBox("Light background")
        self.light_background_cb.setToolTip(
            "Enable for transmitted-light images (phase contrast, brightfield)\n"
            "where background is brighter than foreground.\n"
            "Equivalent to the 'Light background' checkbox in ImageJ."
        )

        self.smooth_cb = QCheckBox("Pre-smooth")
        self.smooth_cb.setChecked(True)
        self.smooth_cb.setToolTip(
            "Apply a 3×3 smoothing pass to the downsampled image before rolling.\n"
            "Matches ImageJ's smoothing step. Reduces noise sensitivity."
        )

        self.corrected_stack_viewer = QPushButton("")
        self.corrected_stack_viewer.setStyleSheet(self.button_select_all)
        self.corrected_stack_viewer.setIcon(icon(MDI6.eye_outline, color="black"))
        self.corrected_stack_viewer.setToolTip("Preview corrected image (middle frame)")
        self.corrected_stack_viewer.clicked.connect(self.preview_correction)
        self.corrected_stack_viewer.setIconSize(QSize(20, 20))

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

    def add_to_layout(self):
        """Add widgets to the layout."""

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_lbl, 25)
        channel_layout.addWidget(self.channels_cb, 75)
        self.addLayout(channel_layout, 0, 0, 1, 3)

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(self.radius_lbl, 25)
        radius_layout.addWidget(self.radius_le, 70)
        radius_layout.addWidget(self.bg_viewer_btn, 5)
        self.addLayout(radius_layout, 1, 0, 1, 3)

        options_layout = QHBoxLayout()
        options_layout.addWidget(self.light_background_cb)
        options_layout.addWidget(self.smooth_cb)
        options_layout.addStretch()
        self.addLayout(options_layout, 2, 0, 1, 3)

        self.operation_layout = OperationLayout()
        self.addLayout(self.operation_layout, 3, 0, 1, 3)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.add_correction_btn, 95)
        correction_layout.addWidget(self.corrected_stack_viewer, 5)
        self.addLayout(correction_layout, 4, 0, 1, 3)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(spacer, 5, 0, 1, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_radius(self) -> float:
        text = self.radius_le.text().replace(",", ".")
        try:
            return max(1.0, float(text))
        except ValueError:
            return 100.0

    def _require_single_position(self) -> bool:
        if (
            self.attr_parent.well_list.isMultipleSelection()
            or not self.attr_parent.well_list.isAnySelected()
            or self.attr_parent.position_list.isMultipleSelection()
            or not self.attr_parent.position_list.isAnySelected()
        ):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a single position.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return False
        return True

    def set_target_channel(self):
        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

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
        operation = "subtract" if self.operation_layout.subtract_btn.isChecked() else "divide"
        clip = (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        )
        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "fit",
            "model": "rolling_ball",
            "operation": operation,
            "clip": clip,
            "radius": self._get_radius(),
            "light_background": self.light_background_cb.isChecked(),
            "smooth": self.smooth_cb.isChecked(),
        }

    # ------------------------------------------------------------------
    # Background preview — fast single-frame path, bypasses full pipeline
    # ------------------------------------------------------------------

    def preview_background(self):
        """Estimate and display the rolling ball background for the current radius.

        Loads the middle frame directly and calls fit_rolling_ball — no config
        parsing or well iteration — so the result appears in ~1 second.
        Useful for tuning radius, light_background, and smooth interactively.
        """
        if not self._require_single_position():
            return

        self.attr_parent.locate_image()
        self.set_target_channel()

        stack_path = getattr(self.attr_parent, "current_stack", None)
        if stack_path is None:
            QMessageBox.warning(None, "Warning", "No stack found for the selected position.")
            return

        n_frames = auto_load_number_of_frames(stack_path)
        if n_frames is None or n_frames == 0:
            QMessageBox.warning(None, "Warning", "Could not determine stack length.")
            return

        n_channels = len(self.attr_parent.exp_channels)
        frame_idx = int(n_frames // 2) * n_channels
        radius = self._get_radius()
        light_background = self.light_background_cb.isChecked()
        smooth = self.smooth_cb.isChecked()

        self.bg_progress = CelldetectiveProgressDialog(
            f"Estimating background  (radius={radius:.0f} px)...",
            "Cancel",
            0, 0, None,
            window_title="Background preview",
        )
        self.bg_progress.setRange(0, 0)

        self.bg_worker = RollingBallBackgroundWorker(
            stack_path=stack_path,
            frame_idx=frame_idx,
            channel_idx=self.target_channel,
            n_channels=n_channels,
            radius=radius,
            light_background=light_background,
            smooth=smooth,
        )

        def on_result(bg: np.ndarray) -> None:
            from celldetective.gui.viewers.base_viewer import StackVisualizer

            if bg is not None:
                display = bg[np.newaxis, :, :, np.newaxis]
                title = (
                    f"Rolling ball background  |  radius={radius:.0f} px"
                    + ("  |  light bg" if light_background else "")
                    + ("  |  smoothed" if smooth else "")
                )
                self.bg_viewer = StackVisualizer(
                    stack=display,
                    window_title=title,
                    target_channel=0,
                    frame_slider=False,
                    contrast_slider=True,
                )
                self.bg_viewer.show()
            else:
                logger.warning("Background estimation returned None.")

        def on_finished() -> None:
            self.bg_progress.close()

        def on_error(msg: str) -> None:
            self.bg_progress.close()
            QMessageBox.critical(None, "Error", f"Background estimation failed: {msg}")

        self.bg_worker.result_ready.connect(on_result)
        self.bg_worker.finished.connect(on_finished)
        self.bg_worker.error.connect(on_error)
        self.bg_progress.canceled.connect(self.bg_worker.stop)

        self.bg_worker.start()
        self.bg_progress.exec_()

    # ------------------------------------------------------------------
    # Corrected-image preview — full pipeline on one frame
    # ------------------------------------------------------------------

    def preview_correction(self):
        """Preview the background-corrected image (middle frame)."""
        if not self._require_single_position():
            return

        operation = "subtract" if self.operation_layout.subtract_btn.isChecked() else "divide"
        clip = (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        )
        radius = self._get_radius()
        light_background = self.light_background_cb.isChecked()
        smooth = self.smooth_cb.isChecked()

        self.attr_parent.locate_image()
        self.set_target_channel()

        subset_indices = None
        stack_path = getattr(self.attr_parent, "current_stack", None)
        if stack_path is not None:
            n_frames = auto_load_number_of_frames(stack_path)
            if n_frames is not None:
                midpoint = int(n_frames // 2)
                n_channels = len(self.attr_parent.exp_channels)
                subset_indices = [midpoint * n_channels]

        process_args = {
            "exp_dir": self.attr_parent.exp_dir,
            "well_option": self.attr_parent.well_list.getSelectedIndices(),
            "position_option": self.attr_parent.position_list.getSelectedIndices(),
            "target_channel": self.channels_cb.currentText(),
            "model": "rolling_ball",
            "threshold_on_std": 1e9,
            "operation": operation,
            "clip": clip,
            "activation_protocol": [["gauss", 2], ["std", 4]],
            "radius": radius,
            "light_background": light_background,
            "smooth": smooth,
            "correction_type": "fit",
            "subset_indices": subset_indices,
        }

        self.bg_progress = CelldetectiveProgressDialog(
            "Correcting background (Rolling ball preview)...",
            "Cancel", 0, 0, None,
            window_title="Processing",
        )
        self.bg_progress.setRange(0, 0)

        self.preview_worker = RollingBallCorrectionPreviewWorker(process_args=process_args)

        def on_result(corrected_stack: np.ndarray) -> None:
            from celldetective.gui.viewers.base_viewer import StackVisualizer

            if corrected_stack is not None:
                if subset_indices is not None and len(self.channel_names) > 0:
                    if corrected_stack.ndim == 3 and corrected_stack.shape[0] == len(self.channel_names):
                        corrected_stack = corrected_stack[self.channels_cb.currentIndex()]
                    elif corrected_stack.ndim == 4 and corrected_stack.shape[-1] == len(self.channel_names):
                        corrected_stack = corrected_stack[..., self.channels_cb.currentIndex()]
                    if corrected_stack.ndim == 2:
                        corrected_stack = corrected_stack[np.newaxis, :, :, np.newaxis]
                    elif corrected_stack.ndim == 3:
                        corrected_stack = corrected_stack[:, :, :, np.newaxis]

                self.viewer = StackVisualizer(
                    stack=corrected_stack,
                    window_title=f"Corrected  |  rolling ball radius={radius:.0f} px",
                    target_channel=0,
                    frame_slider=True,
                    contrast_slider=True,
                )
                self.viewer.show()
            else:
                logger.warning("Rolling ball correction could not be generated.")

        def on_finished() -> None:
            self.bg_progress.close()

        def on_error(msg: str) -> None:
            self.bg_progress.close()
            QMessageBox.critical(None, "Error", f"Correction failed: {msg}")

        self.preview_worker.result_ready.connect(on_result)
        self.preview_worker.finished.connect(on_finished)
        self.preview_worker.error.connect(on_error)
        self.bg_progress.canceled.connect(self.preview_worker.stop)

        self.preview_worker.start()
        self.bg_progress.exec_()


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

class RollingBallBackgroundWorker(QThread):
    """Loads one frame and returns the raw rolling-ball background estimate.

    Bypasses the full correction pipeline — fast feedback for radius tuning.
    """

    finished = pyqtSignal()
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        stack_path: str,
        frame_idx: int,
        channel_idx: int,
        n_channels: int,
        radius: float,
        light_background: bool = False,
        smooth: bool = True,
    ) -> None:
        super().__init__()
        self.stack_path = stack_path
        self.frame_idx = frame_idx
        self.channel_idx = channel_idx
        self.n_channels = n_channels
        self.radius = radius
        self.light_background = light_background
        self.smooth = smooth
        self._cancelled = False

    def stop(self):
        self._cancelled = True
        self.quit()

    def run(self):
        from celldetective.preprocessing import fit_rolling_ball

        try:
            frames = load_frames(
                list(range(self.frame_idx, self.frame_idx + self.n_channels)),
                self.stack_path,
                normalize_input=False,
            ).astype(float)

            if self._cancelled:
                self.result_ready.emit(None)
                self.finished.emit()
                return

            img = frames[:, :, self.channel_idx]
            bg = fit_rolling_ball(
                img,
                radius=self.radius,
                light_background=self.light_background,
                smooth=self.smooth,
            )
            self.result_ready.emit(bg)

        except Exception as e:
            self.error.emit(str(e))

        self.finished.emit()


class RollingBallCorrectionPreviewWorker(QThread):
    """Runs the full correction pipeline on a subset of frames for preview."""

    finished = pyqtSignal()
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, process_args: dict) -> None:
        super().__init__()
        self.process_args = process_args

    def run(self):
        from celldetective.preprocessing import correct_background_model

        try:
            result = correct_background_model(
                experiment=self.process_args["exp_dir"],
                well_option=self.process_args["well_option"],
                position_option=self.process_args["position_option"],
                target_channel=self.process_args["target_channel"],
                model="rolling_ball",
                threshold_on_std=self.process_args["threshold_on_std"],
                operation=self.process_args["operation"],
                clip=self.process_args["clip"],
                export=False,
                return_stacks=True,
                activation_protocol=self.process_args["activation_protocol"],
                radius=self.process_args["radius"],
                light_background=self.process_args.get("light_background", False),
                smooth=self.process_args.get("smooth", True),
                subset_indices=self.process_args.get("subset_indices"),
                show_progress_per_well=False,
                show_progress_per_pos=False,
            )
            self.result_ready.emit(result[0] if result and len(result) > 0 else None)

        except Exception as e:
            self.error.emit(str(e))

        self.finished.emit()

    def stop(self):
        self.quit()
