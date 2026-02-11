from glob import glob
from pathlib import Path
from natsort import natsorted
import os
from PyQt5.QtWidgets import QMessageBox

from PyQt5.QtGui import QIntValidator

from celldetective.gui.layouts.model_fit_layout import BackgroundFitCorrectionLayout
from celldetective import get_logger

logger = get_logger(__name__)


class LocalCorrectionLayout(BackgroundFitCorrectionLayout):
    """docstring for ClassName"""

    def __init__(self, *args):
        """
        Initialize the LocalCorrectionLayout.

        Parameters
        ----------
        *args
            Variable length argument list.
        """

        super().__init__(*args)

        if hasattr(self.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window
        elif hasattr(self.parent_window.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window.parent_window
        else:
            self.attr_parent = (
                self.parent_window.parent_window.parent_window.parent_window
            )

        self.thresh_lbl.setText("Distance: ")
        self.thresh_lbl.setToolTip(
            "Distance from the cell mask over which to estimate local intensity."
        )

        self.models_cb.clear()
        self.models_cb.addItems(["mean", "median"])

        self.threshold_le.set_threshold(5)
        self.threshold_le.connected_buttons = [
            self.threshold_viewer_btn,
            self.add_correction_btn,
        ]
        self.threshold_le.setValidator(QIntValidator())

        self.threshold_viewer_btn.disconnect()
        self.threshold_viewer_btn.clicked.connect(self.set_distance_graphically)

        self.corrected_stack_viewer.hide()

    def check_mask_existence(self, population: str) -> bool:
        """Check if masks exist for the given population."""
        if self.attr_parent.current_stack is None:
            return False

        labels_path = (
            str(Path(self.attr_parent.current_stack).parent.parent)
            + os.sep
            + f"labels_{population}"
            + os.sep
        )
        masks = natsorted(glob(labels_path + "*.tif"))
        if len(masks) == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText(
                f"No mask found in {labels_path}\nPlease segment your data first."
            )
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return False
        return True

    def set_distance_graphically(self):
        """Set the distance graphically using the contour viewer."""
        from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:
            population = self.parent_window.parent_window.mode
            if not self.check_mask_existence(population):
                return

            self.viewer = CellEdgeVisualizer(
                cell_type=population,
                stack_path=self.attr_parent.current_stack,
                parent_le=self.threshold_le,
                n_channels=len(self.channel_names),
                target_channel=self.channels_cb.currentIndex(),
                edge_range=(0, 30),
                initial_edge=-int(thresh),
                invert=True,
                window_title="Set an edge distance to estimate local intensity",
                channel_cb=False,
                PxToUm=1,
                single_value_mode=True,
            )
            self.viewer.show()

    def generate_instructions(self):
        """Generate the instructions dictionary."""

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

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "local",
            "model": self.models_cb.currentText(),
            "distance": int(self.threshold_le.get_threshold()),
            "operation": operation,
            "clip": clip,
        }
