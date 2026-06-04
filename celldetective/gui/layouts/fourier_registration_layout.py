from typing import Optional

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QMainWindow,
    QCheckBox,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective import get_logger

logger = get_logger(__name__)


class FourierRegistrationOptionsLayout(QVBoxLayout, Styles):

    def __init__(
        self, parent_window: Optional[QMainWindow] = None, *args, **kwargs
    ) -> None:
        """
        Initialize the FourierRegistrationOptionsLayout.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The parent window.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self.parent_window = parent_window
        if hasattr(self.parent_window.parent_window, "exp_config"):
            self.attr_parent = self.parent_window.parent_window
        else:
            self.attr_parent = self.parent_window.parent_window.parent_window

        self.channel_names = self.attr_parent.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()
        self.on_method_changed()

    def generate_widgets(self):
        """Generate the widgets."""

        self.method_lbl = QLabel("Method: ")
        self.method_cb = QComboBox()
        self.method_cb.addItem("Fourier (Phase Cross-Correlation)", "fourier")
        self.method_cb.addItem("SIFT Feature Matching", "sift")
        self.method_cb.addItem("Hybrid (SIFT + Fourier Fallback)", "hybrid")
        self.method_cb.addItem("Single-Particle Tracking (SPT)", "spt")
        self.method_cb.currentIndexChanged.connect(self.on_method_changed)

        self.reference_channel_lbl = QLabel("Reference Channel: ")
        self.reference_channels_cb = QComboBox()
        self.reference_channels_cb.addItems(self.channel_names)

        self.reference_frame_lbl = QLabel("Reference Frame: ")
        
        self.add_correction_btn = QPushButton("Add registration")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add Fourier registration correction step.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.reference_frame_le = ThresholdLineEdit(
            init_value=0,
            connected_buttons=[self.add_correction_btn],
            placeholder="frame index (e.g. 0)",
            value_type="int",
        )

        self.upsample_lbl = QLabel("Resolution: ")
        self.upsample_factor_cb = QComboBox()
        self.upsample_factor_cb.addItem("1 (Pixel-level)", 1)
        self.upsample_factor_cb.addItem("10 (0.1 pixel)", 10)
        self.upsample_factor_cb.addItem("100 (0.01 pixel)", 100)
        self.upsample_factor_cb.setCurrentIndex(1)  # Default to 0.1 pixel level for high quality

        self.shift_method_lbl = QLabel("Subpixel Shift: ")
        self.shift_method_cb = QComboBox()
        self.shift_method_cb.addItem("Spatial Spline Interpolation", "spatial")
        self.shift_method_cb.addItem("Fourier Domain Phase Multiplier", "fourier")
        self.shift_method_cb.setCurrentIndex(0)  # Default to spatial for backward compatibility
        self.shift_method_cb.setToolTip(
            "Select sub-pixel translation method:\n"
            "- Spatial Spline Interpolation: Standard spline-based shifting (bilinear/bicubic)\n"
            "- Fourier Domain Phase Multiplier: Frequency-domain phase shifting (no interpolation blur, artifact-resistant)"
        )

        self.sliding_cb = QCheckBox("Sliding registration (wrt previous frame)")
        self.sliding_cb.setToolTip(
            "If enabled, registers each frame relative to the previous frame sequentially\n"
            "and accumulates the translation shifts back to the reference frame. Otherwise,\n"
            "registers all frames directly to the fixed reference frame."
        )

        self.sigma_lbl = QLabel("Gaussian Blur (sigma): ")
        self.sigma_le = ThresholdLineEdit(
            init_value=1.0,
            connected_buttons=[self.add_correction_btn],
            placeholder="sigma (e.g. 1.0)",
            value_type="float",
        )
        self.sigma_le.setToolTip(
            "Sigma value for Gaussian smoothing before cross-correlation (0.0 to disable).\n"
            "Pre-smoothing helps suppress high-frequency noise and stabilizes correlation peaks."
        )

        self.max_shift_lbl = QLabel("Max Shift (pixels): ")
        self.max_shift_le = ThresholdLineEdit(
            init_value=0.0,
            connected_buttons=[self.add_correction_btn],
            placeholder="pixels (e.g. 15.0)",
            value_type="float",
        )
        self.max_shift_le.setToolTip(
            "Maximum shift magnitude allowed per frame in pixels (0.0 to disable).\n"
            "Spurious noise jumps exceeding this threshold are discarded and replaced with safe fallbacks."
        )

        self.filter_outliers_cb = QCheckBox("Filter Outliers (Median Trajectory)")
        self.filter_outliers_cb.setToolTip(
            "Applies a 1D median filter of window size 3 to the computed shift trajectory\n"
            "to mathematically remove isolated single-frame spike artifacts."
        )

        # Multi-Channel Joint Consensus
        self.joint_consensus_cb = QCheckBox("Multi-Channel Joint Consensus")
        self.joint_consensus_cb.setToolTip(
            "If enabled, performs joint drift estimation across multiple channels\n"
            "by taking a robust weighted consensus of estimated translation shifts."
        )
        self.joint_consensus_cb.stateChanged.connect(self.on_consensus_changed)

        self.channel_checkboxes = {}
        for chan in self.channel_names:
            cb = QCheckBox(chan)
            cb.setChecked(True)
            cb.setEnabled(False)
            self.channel_checkboxes[chan] = cb

        # Plot trajectory option
        self.plot_trajectory_cb = QCheckBox("Plot drift trajectory after completion")
        self.plot_trajectory_cb.setChecked(True)
        self.plot_trajectory_cb.setToolTip(
            "If checked, pops up an interactive, premium Matplotlib trajectory plot\n"
            "after drift registration has finished, letting you inspect the drift curves."
        )

        # SPT parameters
        self.min_distance_lbl = QLabel("Min Distance (px): ")
        self.min_distance_le = ThresholdLineEdit(
            init_value=15.0,
            connected_buttons=[self.add_correction_btn],
            placeholder="px (e.g. 15.0)",
            value_type="float",
        )
        self.min_distance_le.setToolTip("Minimum distance between detected spots (beads) in pixels.")

        self.detection_threshold_lbl = QLabel("Detection Threshold (0-1): ")
        self.detection_threshold_le = ThresholdLineEdit(
            init_value=0.1,
            connected_buttons=[self.add_correction_btn],
            placeholder="threshold (e.g. 0.1)",
            value_type="float",
        )
        self.detection_threshold_le.setToolTip("Relative peak detection threshold on min-max normalized frames.")

        self.search_range_lbl = QLabel("Search Range (px): ")
        self.search_range_le = ThresholdLineEdit(
            init_value=5.0,
            connected_buttons=[self.add_correction_btn],
            placeholder="px (e.g. 5.0)",
            value_type="float",
        )
        self.search_range_le.setToolTip("Maximum displacement for trackpy linking between consecutive frames.")

        self.memory_lbl = QLabel("Memory (frames): ")
        self.memory_le = ThresholdLineEdit(
            init_value=1,
            connected_buttons=[self.add_correction_btn],
            placeholder="frames (e.g. 1)",
            value_type="int",
        )
        self.memory_le.setToolTip("Maximum number of skipped frames allowed for a trajectory in trackpy.")

        self.preview_spots_btn = QPushButton("Preview Spot Detection")
        self.preview_spots_btn.setStyleSheet(self.button_style_sheet_2)
        self.preview_spots_btn.setIcon(icon(MDI6.eye, color="#1565c0"))
        self.preview_spots_btn.setToolTip("Launch live interactive visualizer to tune spot detection parameters.")
        self.preview_spots_btn.setIconSize(QSize(25, 25))
        self.preview_spots_btn.clicked.connect(self.preview_spot_detection)

        # SPT Preprocessing Layout
        from celldetective.gui.gui_utils import PreprocessingLayout2
        self.spt_preprocessing = PreprocessingLayout2(fraction=40, parent_window=self)

    def on_method_changed(self):
        """Slot to show/hide and enable/disable widgets based on registration method."""
        method = self.method_cb.currentData()
        
        is_spt = (method == "spt")
        
        # Show/hide SPT parameters
        self.min_distance_lbl.setVisible(is_spt)
        self.min_distance_le.setVisible(is_spt)
        self.detection_threshold_lbl.setVisible(is_spt)
        self.detection_threshold_le.setVisible(is_spt)
        self.search_range_lbl.setVisible(is_spt)
        self.search_range_le.setVisible(is_spt)
        self.memory_lbl.setVisible(is_spt)
        self.memory_le.setVisible(is_spt)
        self.preview_spots_btn.setVisible(is_spt)

        self.spt_preprocessing.list.setVisible(is_spt)
        self.spt_preprocessing.add_filter_btn.setVisible(is_spt)
        self.spt_preprocessing.delete_filter_btn.setVisible(is_spt)
        self.spt_preprocessing.preprocess_lbl.setVisible(is_spt)
        
        # Joint Consensus Handling for SPT
        if is_spt:
            self.joint_consensus_cb.setChecked(False)
            self.joint_consensus_cb.setVisible(False)
            for cb in self.channel_checkboxes.values():
                cb.setVisible(False)
            self.reference_channels_cb.setEnabled(True)
            self.reference_channel_lbl.setEnabled(True)
        else:
            self.joint_consensus_cb.setVisible(True)
            for cb in self.channel_checkboxes.values():
                cb.setVisible(True)
            self.on_consensus_changed()
        
        if method == "sift":
            self.sigma_lbl.setText("Gaussian Blur (sigma): ")
            self.sigma_lbl.setToolTip("Sigma value for Gaussian smoothing before SIFT (0.0 to disable).")
            self.upsample_lbl.setVisible(True)
            self.upsample_factor_cb.setVisible(True)
            self.upsample_lbl.setEnabled(False)
            self.upsample_factor_cb.setEnabled(False)
            self.sigma_lbl.setVisible(True)
            self.sigma_le.setVisible(True)
            self.sigma_lbl.setEnabled(False)
            self.sigma_le.setEnabled(False)
            self.shift_method_lbl.setVisible(True)
            self.shift_method_cb.setVisible(True)
        elif method == "spt":
            self.sigma_lbl.setText("Spot Size [px]: ")
            self.sigma_lbl.setToolTip(
                "Expected diameter of fluorescent spots/landmarks (odd integer, minimum is 5).\n"
                "Diameters below 5px are automatically treated as 5px in trackpy to prevent numerical instabilities."
            )
            self.min_distance_lbl.setText("Min Distance [px]: ")
            self.min_distance_lbl.setToolTip(
                "Minimum distance (separation) between detected spots in pixels.\n"
                "Features closer than this are filtered out. If not specified, trackpy defaults to Spot Size + 1."
            )
            try:
                val = float(self.sigma_le.get_threshold())
                if val < 5.0:
                    self.sigma_le.setText("15.0")
            except Exception:
                pass
            self.upsample_lbl.setVisible(False)
            self.upsample_factor_cb.setVisible(False)
            self.shift_method_lbl.setVisible(False)
            self.shift_method_cb.setVisible(False)
            # Keep Spot Size visible and enabled for SPT spot detection
            self.sigma_lbl.setVisible(True)
            self.sigma_le.setVisible(True)
            self.sigma_lbl.setEnabled(True)
            self.sigma_le.setEnabled(True)
        else:  # fourier or hybrid
            self.sigma_lbl.setText("Gaussian Blur (sigma): ")
            self.sigma_lbl.setToolTip("Sigma value for Gaussian smoothing before cross-correlation (0.0 to disable).")
            self.upsample_lbl.setVisible(True)
            self.upsample_factor_cb.setVisible(True)
            self.upsample_lbl.setEnabled(True)
            self.upsample_factor_cb.setEnabled(True)
            self.shift_method_lbl.setVisible(True)
            self.shift_method_cb.setVisible(True)
            self.sigma_lbl.setVisible(True)
            self.sigma_le.setVisible(True)
            self.sigma_lbl.setEnabled(True)
            self.sigma_le.setEnabled(True)

    def on_consensus_changed(self):
        """Slot to enable/disable single channel selection and individual consensus channel checkboxes."""
        is_consensus = self.joint_consensus_cb.isChecked()
        self.reference_channels_cb.setEnabled(not is_consensus)
        self.reference_channel_lbl.setEnabled(not is_consensus)
        for cb in self.channel_checkboxes.values():
            cb.setEnabled(is_consensus)

    def add_to_layout(self):
        """Add widgets to the layout."""

        method_hbox = QHBoxLayout()
        method_hbox.addWidget(self.method_lbl, 40)
        method_hbox.addWidget(self.method_cb, 60)
        self.addLayout(method_hbox)

        channel_ch_hbox = QHBoxLayout()
        channel_ch_hbox.addWidget(self.reference_channel_lbl, 40)
        channel_ch_hbox.addWidget(self.reference_channels_cb, 60)
        self.addLayout(channel_ch_hbox)

        consensus_hbox = QHBoxLayout()
        consensus_hbox.addWidget(self.joint_consensus_cb)
        self.addLayout(consensus_hbox)

        self.consensus_channels_layout = QHBoxLayout()
        self.consensus_channels_layout.setContentsMargins(20, 0, 0, 0)
        for cb in self.channel_checkboxes.values():
            self.consensus_channels_layout.addWidget(cb)
        self.addLayout(self.consensus_channels_layout)

        frame_hbox = QHBoxLayout()
        frame_hbox.addWidget(self.reference_frame_lbl, 40)
        frame_hbox.addWidget(self.reference_frame_le, 60)
        self.addLayout(frame_hbox)

        upsample_hbox = QHBoxLayout()
        upsample_hbox.addWidget(self.upsample_lbl, 40)
        upsample_hbox.addWidget(self.upsample_factor_cb, 60)
        self.addLayout(upsample_hbox)

        shift_method_hbox = QHBoxLayout()
        shift_method_hbox.addWidget(self.shift_method_lbl, 40)
        shift_method_hbox.addWidget(self.shift_method_cb, 60)
        self.addLayout(shift_method_hbox)

        sigma_hbox = QHBoxLayout()
        sigma_hbox.addWidget(self.sigma_lbl, 40)
        sigma_hbox.addWidget(self.sigma_le, 60)
        self.addLayout(sigma_hbox)

        # SPT parameters added to layout
        min_distance_hbox = QHBoxLayout()
        min_distance_hbox.addWidget(self.min_distance_lbl, 40)
        min_distance_hbox.addWidget(self.min_distance_le, 60)
        self.addLayout(min_distance_hbox)

        detection_threshold_hbox = QHBoxLayout()
        detection_threshold_hbox.addWidget(self.detection_threshold_lbl, 40)
        detection_threshold_hbox.addWidget(self.detection_threshold_le, 60)
        self.addLayout(detection_threshold_hbox)

        search_range_hbox = QHBoxLayout()
        search_range_hbox.addWidget(self.search_range_lbl, 40)
        search_range_hbox.addWidget(self.search_range_le, 60)
        self.addLayout(search_range_hbox)

        memory_hbox = QHBoxLayout()
        memory_hbox.addWidget(self.memory_lbl, 40)
        memory_hbox.addWidget(self.memory_le, 60)
        self.addLayout(memory_hbox)

        self.addLayout(self.spt_preprocessing)

        preview_spots_hbox = QHBoxLayout()
        preview_spots_hbox.addWidget(self.preview_spots_btn, 95)
        self.addLayout(preview_spots_hbox)

        max_shift_hbox = QHBoxLayout()
        max_shift_hbox.addWidget(self.max_shift_lbl, 40)
        max_shift_hbox.addWidget(self.max_shift_le, 60)
        self.addLayout(max_shift_hbox)

        sliding_hbox = QHBoxLayout()
        sliding_hbox.addWidget(self.sliding_cb)
        self.addLayout(sliding_hbox)

        outliers_hbox = QHBoxLayout()
        outliers_hbox.addWidget(self.filter_outliers_cb)
        self.addLayout(outliers_hbox)

        plot_hbox = QHBoxLayout()
        plot_hbox.addWidget(self.plot_trajectory_cb)
        self.addLayout(plot_hbox)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.add_correction_btn, 95)
        self.addLayout(btn_hbox)

    def add_instructions_to_parent_list(self):
        """Add instructions to the parent protocol list."""

        self.generate_instructions()
        if hasattr(self.parent_window, "protocol_layout"):
            parent = self.parent_window.protocol_layout
        else:
            parent = self.parent_window

        parent.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        parent.protocol_list.addItem(correction_description)

    def generate_instructions(self):
        """Generate the instructions dictionary."""

        method = self.method_cb.currentData()
        is_spt = (method == "spt")

        self.instructions = {
            "correction_type": "registration",
            "method": method,
            "shift_method": self.shift_method_cb.currentData(),
            "reference_channel": ",".join([ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()]) if (self.joint_consensus_cb.isChecked() and not is_spt) else self.reference_channels_cb.currentText(),
            "reference_frame_idx": int(self.reference_frame_le.get_threshold()),
            "upsample_factor": int(self.upsample_factor_cb.currentData()) if not is_spt else 1,
            "sliding": self.sliding_cb.isChecked(),
            "sigma": float(self.sigma_le.get_threshold()),
            "max_shift": float(self.max_shift_le.get_threshold()),
            "filter_outliers": self.filter_outliers_cb.isChecked(),
            "joint_consensus": self.joint_consensus_cb.isChecked() if not is_spt else False,
            "consensus_channels": [ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()] if not is_spt else [self.reference_channels_cb.currentText()],
            "plot_trajectory": self.plot_trajectory_cb.isChecked(),
            "min_distance": float(self.min_distance_le.get_threshold()),
            "detection_threshold": float(self.detection_threshold_le.get_threshold()),
            "search_range": float(self.search_range_le.get_threshold()),
            "memory": int(self.memory_le.get_threshold()),
            "image_preprocessing": self.spt_preprocessing.list.items if is_spt else None,
        }

    def preview_spot_detection(self):
        """Launch the spot detection preview dialog."""
        from celldetective.gui.viewers.spt_preview_viewer import SPTPreviewVisualizer
        
        self.attr_parent.locate_image()
        stack_path = getattr(self.attr_parent, "current_stack", None)
        if stack_path is None:
            logger.warning("No stack selected. Cannot preview spot detection.")
            return
            
        try:
            initial_sigma = float(self.sigma_le.get_threshold())
        except Exception:
            initial_sigma = 1.0

        try:
            initial_min_dist = float(self.min_distance_le.get_threshold())
        except Exception:
            initial_min_dist = 15.0

        try:
            initial_thresh = float(self.detection_threshold_le.get_threshold())
        except Exception:
            initial_thresh = 0.1

        # Use the first channel as the tracking channel.
        # If consensus is checked, use the first selected channel; otherwise use reference_channels_cb.
        if self.joint_consensus_cb.isChecked():
            consensus = [ch for ch, cb in self.channel_checkboxes.items() if cb.isChecked()]
            tracking_chan = consensus[0] if len(consensus) > 0 else self.channel_names[0]
        else:
            tracking_chan = self.reference_channels_cb.currentText()
            
        try:
            target_chan_idx = list(self.channel_names).index(tracking_chan)
        except (ValueError, AttributeError):
            target_chan_idx = 0

        self.spt_viewer = SPTPreviewVisualizer(
            parent_window=self.parent_window,
            parent_sigma_le=self.sigma_le,
            parent_min_distance_le=self.min_distance_le,
            parent_detection_threshold_le=self.detection_threshold_le,
            parent_preprocessing_list=self.spt_preprocessing.list,
            initial_sigma=initial_sigma,
            initial_min_distance=initial_min_dist,
            initial_detection_threshold=initial_thresh,
            initial_preprocessing=list(self.spt_preprocessing.list.items),
            stack_path=stack_path,
            channel_names=self.channel_names,
            n_channels=len(self.channel_names),
            channel_cb=True,
            target_channel=target_chan_idx,
            window_title="SPT Spot Detection Preview",
        )
        self.spt_viewer.show()
