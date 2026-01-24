import os
from glob import glob
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QListWidget,
    QLineEdit,
    QMessageBox,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from fonticon_mdi6 import MDI6
from natsort import natsorted
from superqt import QLabeledSlider, QLabeledDoubleSlider, QLabeledRangeSlider
from superqt.fonticon import icon
from collections import OrderedDict

from celldetective.gui.gui_utils import QuickSliderLayout
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective import get_logger
from tifffile import imread
import re

logger = get_logger(__name__)


class CellEdgeVisualizer(StackVisualizer):
    """
    A widget for visualizing cell edges with interactive sliders and channel selection.

    Parameters:
    - cell_type (str): Type of cells ('effectors' by default).
    - edge_range (tuple): Range of edge sizes (-30, 30) by default.
    - invert (bool): Flag to invert the edge size (False by default).
    - parent_list_widget: The parent QListWidget instance to add edge measurements.
    - parent_le: The parent QLineEdit instance to set the edge size.
    - labels (array or None): Array of labels for cell segmentation.
    - initial_edge (int): Initial edge size (5 by default).
    - initial_mask_alpha (float): Initial mask opacity value (0.5 by default).
    - args, kwargs: Additional arguments to pass to the parent class constructor.

    Methods:
    - load_labels(): Load the cell labels.
    - locate_labels_virtual(): Locate virtual labels.
    - generate_add_to_list_btn(): Generate the add to list button.
    - generate_add_to_le_btn(): Generate the set measurement button for QLineEdit.
    - set_measurement_in_parent_le(): Set the edge size in the parent QLineEdit.
    - set_measurement_in_parent_list(): Add the edge size to the parent QListWidget.
    - generate_label_imshow(): Generate the label imshow.
    - generate_edge_slider(): Generate the edge size slider.
    - generate_opacity_slider(): Generate the opacity slider for the mask.
    - change_mask_opacity(value): Change the opacity of the mask.
    - change_edge_size(value): Change the edge size.
    - change_frame(value): Change the displayed frame and update the edge labels.
    - compute_edge_labels(): Compute the edge labels.

    Notes:
    - This class extends the functionality of StackVisualizer to visualize cell edges
      with interactive sliders for edge size adjustment and mask opacity control.
    """

    def __init__(
        self,
        cell_type="effectors",
        edge_range=(-30, 30),
        invert=False,
        parent_list_widget=None,
        parent_le=None,
        labels=None,
        initial_edge=5,
        initial_mask_alpha=0.5,
        *args,
        **kwargs,
    ):

        # Initialize the widget and its attributes
        super().__init__(*args, **kwargs)
        self.edge_size = initial_edge
        self.mask_alpha = initial_mask_alpha
        self.cell_type = cell_type
        self.labels = labels
        self.edge_range = edge_range
        self.invert = invert
        self.parent_list_widget = parent_list_widget
        self.parent_le = parent_le

        # SDF cache (stores label + dist_in + dist_out + voronoi)
        self.sdf_cache = OrderedDict()
        self.max_sdf_cache_size = 128

        self.load_labels()
        self.generate_label_imshow()
        self.generate_edge_slider()
        self.generate_opacity_slider()
        if isinstance(self.parent_list_widget, QListWidget):
            self.generate_add_to_list_btn()
        if isinstance(self.parent_le, QLineEdit):
            self.generate_add_to_le_btn()

    def closeEvent(self, event):
        if hasattr(self, "sdf_cache") and isinstance(self.sdf_cache, OrderedDict):
            self.sdf_cache.clear()
        super().closeEvent(event)

    def load_labels(self):
        # Load the cell labels

        if self.labels is not None:

            if isinstance(self.labels, list):
                self.labels = np.array(self.labels)

            assert (
                self.labels.ndim == 3
            ), "Wrong dimensions for the provided labels, expect TXY"
            assert len(self.labels) == self.stack_length

            self.label_mode = "direct"
            self.init_label = self.labels[self.mid_time, :, :]
        else:
            self.label_mode = "virtual"
            assert isinstance(self.stack_path, str)
            assert self.stack_path.endswith(".tif")
            self.locate_labels_virtual()

        self.compute_edge_labels()

    def locate_labels_virtual(self):
        # Locate virtual labels

        labels_path = (
            str(Path(self.stack_path).parent.parent)
            + os.sep
            + f"labels_{self.cell_type}"
            + os.sep
        )
        self.mask_paths = natsorted(glob(labels_path + "*.tif"))

        if len(self.mask_paths) == 0:

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("No labels were found for the selected cells. Abort.")
            msgBox.setWindowTitle("Critical")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            self.close()

        self.label_map = {}
        for path in self.mask_paths:
            # Find the last number in the filename
            match = re.findall(r"(\d+)", Path(path).stem)
            if match:
                # Assume the last number is the frame index (Use direct mapping: 0000 -> 0)
                frame_num = int(match[-1])
                self.label_map[frame_num] = path

        current_frame = self.frame_slider.value()
        if current_frame in self.label_map:
            self.init_label = imread(self.label_map[current_frame])
        else:
            if hasattr(self, "init_frame"):
                self.init_label = np.zeros_like(self.init_frame)
            else:
                self.init_label = imread(self.mask_paths[0])

    def generate_add_to_list_btn(self):
        # Generate the add to list button

        add_hbox = QHBoxLayout()
        self.add_measurement_btn = QPushButton("Add measurement")
        self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
        self.add_measurement_btn.setIcon(icon(MDI6.plus, color="white"))
        self.add_measurement_btn.setIconSize(QSize(20, 20))
        self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
        add_hbox.addWidget(QLabel(""), 33)
        add_hbox.addWidget(self.add_measurement_btn, 33)
        add_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(add_hbox)

    def generate_add_to_le_btn(self):
        # Generate the set measurement button for QLineEdit

        add_hbox = QHBoxLayout()
        self.set_measurement_btn = QPushButton("Set")
        self.set_measurement_btn.clicked.connect(self.set_measurement_in_parent_le)
        self.set_measurement_btn.setStyleSheet(self.button_style_sheet)
        add_hbox.addWidget(QLabel(""), 33)
        add_hbox.addWidget(self.set_measurement_btn, 33)
        add_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(add_hbox)

    def set_measurement_in_parent_le(self):
        # Set the edge size in the parent QLineEdit

        self.parent_le.setText(str(int(self.edge_slider.value())))
        self.close()

    def set_measurement_in_parent_list(self):
        # Add the edge size to the parent QListWidget

        self.parent_list_widget.addItems([str(self.edge_slider.value())])
        self.close()

    def generate_label_imshow(self):
        # Generate the label imshow

        self.im_mask = self.ax.imshow(
            np.ma.masked_where(self.edge_labels == 0, self.edge_labels),
            alpha=self.mask_alpha,
            interpolation="none",
            cmap="viridis",
        )
        self.canvas.canvas.draw()

    def generate_edge_slider(self):
        # Generate the edge size slider using a Range Slider

        edge_layout = QHBoxLayout()
        edge_layout.setContentsMargins(15, 0, 15, 0)

        # Determine range for universal slider (symmetric)
        max_range = (
            self.edge_range[1]
            if hasattr(self, "edge_range") and self.edge_range
            else 30
        )
        slider_min, slider_max = -max_range, max_range

        self.edge_slider = QLabeledRangeSlider(Qt.Horizontal)
        self.edge_slider.setRange(slider_min, slider_max)

        # Initial Value: convert scalar edge_size to range
        # precise initial state might need adjustment, defaulting to [0, edge_size] if positive
        init_val = self.edge_size if hasattr(self, "edge_size") else 1
        if init_val >= 0:
            self.edge_slider.setValue((0, init_val))
        else:
            self.edge_slider.setValue((init_val, 0))

        self.edge_slider.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self.edge_slider.valueChanged.connect(self.change_edge_size)

        edge_layout.addWidget(QLabel("Edge: "), 15)
        edge_layout.addWidget(self.edge_slider, 85)

        self.canvas.layout.addLayout(edge_layout)

    def generate_opacity_slider(self):
        # Generate the opacity slider for the mask

        self.opacity_slider = QLabeledDoubleSlider()
        opacity_layout = QuickSliderLayout(
            label="Opacity: ",
            slider=self.opacity_slider,
            slider_initial_value=0.5,
            slider_range=(0, 1),
            decimal_option=True,
            precision=3,
        )
        opacity_layout.setContentsMargins(15, 0, 15, 0)
        self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
        self.canvas.layout.addLayout(opacity_layout)

    def change_mask_opacity(self, value):
        # Change the opacity of the mask

        self.mask_alpha = value
        self.im_mask.set_alpha(self.mask_alpha)
        self.canvas.canvas.draw_idle()

    def change_edge_size(self, value):
        self.edge_size = value  # Tuple (min, max)
        self.compute_edge_labels()
        mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
        self.im_mask.set_data(mask)
        self.canvas.canvas.draw_idle()

    def change_frame(self, value):
        # Change the displayed frame and update the edge labels

        super().change_frame(value)

        # Check unified cache first
        if hasattr(self, "sdf_cache") and value in self.sdf_cache:
            self.init_label, self.dist_in, self.dist_out, self.voronoi_map = (
                self.sdf_cache[value]
            )
            self.sdf_cache.move_to_end(value)
        else:
            # Cache Miss: Load Label
            if self.label_mode == "virtual":
                if hasattr(self, "label_map") and value in self.label_map:
                    try:
                        self.init_label = imread(self.label_map[value])
                    except Exception:
                        self.init_label = np.zeros_like(self.init_frame)
                else:
                    self.init_label = np.zeros_like(self.init_frame)
            elif self.label_mode == "direct":
                if value < len(self.labels):
                    self.init_label = self.labels[value, :, :]
                else:
                    self.init_label = np.zeros_like(self.init_frame)

            # Compute SDFs
            self.update_sdf()

            # Store in unified cache

            if hasattr(self, "sdf_cache"):
                self.sdf_cache[value] = (
                    self.init_label,
                    self.dist_in,
                    self.dist_out,
                    self.voronoi_map,
                )
                if len(self.sdf_cache) > self.max_sdf_cache_size:
                    self.sdf_cache.popitem(last=False)

        self.compute_edge_labels()
        mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
        self.im_mask.set_data(mask)

    def update_sdf(self):
        # Compute Signed Distance Functions (SDFs) and Voronoi map for the current labels
        from scipy.ndimage import distance_transform_edt

        # Ensure init_label is not empty
        if self.init_label is None or self.init_label.size == 0:
            self.dist_in = np.zeros_like(self.init_label, dtype=float)
            self.dist_out = np.zeros_like(self.init_label, dtype=float)
            self.voronoi_map = np.zeros_like(self.init_label, dtype=int)
            return

        # Inner Distance
        # Distance to background. Touching cells might merge, but this matches legacy behavior
        # and is much faster than looping.
        self.dist_in = distance_transform_edt(self.init_label > 0)

        # Outer Distance & Voronoi Map
        # Distance to boundaries from outside.
        # return_indices=True gives us the coordinates of the nearest foreground pixel for every background pixel.
        # We use this to look up the label of the nearest object (Voronoi tessellation).
        self.dist_out, indices = distance_transform_edt(
            self.init_label == 0, return_indices=True
        )
        self.voronoi_map = self.init_label[indices[0], indices[1]]

    def compute_edge_labels(self):
        # Compute the edge labels using Composite SDF and Range Masking
        # Delegates to the unified utility in celldetective.utils

        # Ensure SDFs exist
        if not hasattr(self, "dist_in"):
            self.update_sdf()

        # Ensure array shapes
        if self.dist_in.shape != self.init_label.shape:
            self.edge_labels = np.zeros_like(self.init_label)
            return

        # Composite SDF
        sdf = self.dist_in - self.dist_out

        # Use unified utility
        from celldetective.utils.masks import contour_of_instance_segmentation

        self.edge_labels = contour_of_instance_segmentation(
            self.init_label,
            distance=self.edge_size,
            sdf=sdf,
            voronoi_map=self.voronoi_map,
        )

        # Handle scalar edge_size (fallback or initial init)
        if isinstance(self.edge_size, (int, float)):
            if self.edge_size >= 0:
                r_min, r_max = 0, self.edge_size
            else:
                r_min, r_max = self.edge_size, 0
        else:
            r_min, r_max = self.edge_size

        # Create Mask [r_min, r_max]
        # Example: [2, 5] -> Stripe inside. [-5, -2] -> Stripe outside. [-2, 2] -> Border crossing.
        mask = (sdf >= r_min) & (sdf <= r_max)

        # Apply Instance Identities using Voronoi Map
        # Note: Voronoi map covers entire space.
        self.edge_labels = self.voronoi_map * mask
