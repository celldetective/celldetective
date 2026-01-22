from PyQt5.QtGui import QIntValidator

from celldetective.gui.layouts.model_fit_layout import BackgroundFitCorrectionLayout
from celldetective import get_logger

logger = get_logger(__name__)

class LocalCorrectionLayout(BackgroundFitCorrectionLayout):
    """docstring for ClassName"""

    def __init__(self, *args):

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

    def set_distance_graphically(self):
        from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:

            self.viewer = CellEdgeVisualizer(
                cell_type=self.parent_window.parent_window.mode,
                stack_path=self.attr_parent.current_stack,
                parent_le=self.threshold_le,
                n_channels=len(self.channel_names),
                target_channel=self.channels_cb.currentIndex(),
                edge_range=(0, 30),
                initial_edge=int(thresh),
                invert=True,
                window_title="Set an edge distance to estimate local intensity",
                channel_cb=False,
                PxToUm=1,
            )
            self.viewer.show()

    def generate_instructions(self):

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
