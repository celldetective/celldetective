from pathlib import Path
from typing import Optional, Union, Tuple, Any

import numpy as np
from PyQt5.QtCore import QSize, Qt, QEvent
from PyQt5.QtWidgets import (
    QListWidget,
    QLineEdit,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from fonticon_mdi6 import MDI6
from celldetective.gui.base.sliders import QLabeledDoubleSlider, QLabeledRangeSlider
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
        cell_type: str = "effectors",
        edge_range: Tuple[int, int] = (-30, 30),
        invert: bool = False,
        parent_list_widget: Optional[QListWidget] = None,
        parent_le: Optional[QLineEdit] = None,
        labels: Optional[np.ndarray] = None,
        initial_edge: int = 5,
        initial_mask_alpha: float = 0.5,
        single_value_mode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CellEdgeVisualizer.

        Parameters
        ----------
        cell_type : str, optional
            Type of cells ('effectors' by default).
        edge_range : tuple, optional
            Range of edge sizes (-30, 30) by default.
        invert : bool, optional
            Flag to invert the edge size (False by default).
        parent_list_widget : QListWidget, optional
            The parent QListWidget instance to add edge measurements.
        parent_le : QLineEdit, optional
            The parent QLineEdit instance to set the edge size.
        labels : numpy.ndarray, optional
            Array of labels for cell segmentation.
        initial_edge : int, optional
            Initial edge size (5 by default).
        initial_mask_alpha : float, optional
             Initial mask opacity value (0.5 by default).
        single_value_mode : bool, optional
            Whether to output a single value or tuple.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

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
        self.single_value_mode = single_value_mode

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

    def closeEvent(self, event: QEvent) -> None:
        """
        Handle the close event.

        Parameters
        ----------
        event : QEvent
            The close event.
        """
        if hasattr(self, "sdf_cache") and isinstance(self.sdf_cache, OrderedDict):
            self.sdf_cache.clear()
        super().closeEvent(event)

    def load_labels(self):
        """Load the cell labels."""
        # Load the cell labels

        if self.labels is not None:
            self.labels = self.check_labels(self.labels)
            self.label_mode = "direct"
            self.label_map = None
            self.init_label = self.labels[self.mid_time, :, :]
        else:
            self.label_mode = "virtual"
            self.check_virtual_stack()
            self.locate_labels_virtual()

        self.compute_edge_labels()

    def locate_labels_virtual(self):
        """Locate virtual labels."""
        # Locate virtual labels

        self.mask_paths = self.find_label_paths(self.cell_type)
        self.label_map = self.map_label_paths(self.mask_paths)

        current_frame = self.frame_slider.value()
        if current_frame in self.label_map:
            self.init_label = imread(self.label_map[current_frame])
        else:
            if hasattr(self, "init_frame"):
                self.init_label = np.zeros_like(self.init_frame)
            else:
                self.init_label = imread(self.mask_paths[0])

    @staticmethod
    def map_label_paths(paths) -> dict:
        """Frame index -> label image path, from the last number of each file name (0000 -> 0)."""

        label_map = {}
        for path in paths:
            match = re.findall(r"(\d+)", Path(path).stem)
            if match:
                label_map[int(match[-1])] = path
        return label_map

    def read_labels(self, frame: int, labels, label_map) -> np.ndarray:
        """The label image of a population at a frame, from ``labels`` (TYX) or
        a ``label_map`` of paths; empty when there is none."""

        empty = np.zeros(self.init_frame.shape[:2], dtype=int)
        if labels is not None:
            return labels[frame] if frame < len(labels) else empty
        if label_map and frame in label_map:
            try:
                return imread(label_map[frame])
            except Exception as e:
                logger.warning(f"Could not read {label_map[frame]}: {e}")
        return empty

    def generate_add_to_list_btn(self):
        """Generate the add to list button."""
        # Generate the add to list button

        self.add_measurement_btn = QPushButton("Add measurement")
        self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
        self.add_measurement_btn.setIcon(icon(MDI6.plus, color="white"))
        self.add_measurement_btn.setIconSize(QSize(20, 20))
        self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
        self.add_centered_button(self.add_measurement_btn)

    def generate_add_to_le_btn(self):
        """Generate the set measurement button for QLineEdit."""
        # Generate the set measurement button for QLineEdit

        self.set_measurement_btn = QPushButton("Set")
        self.set_measurement_btn.clicked.connect(self.set_measurement_in_parent_le)
        self.set_measurement_btn.setStyleSheet(self.button_style_sheet)
        self.add_centered_button(self.set_measurement_btn)

    def set_measurement_in_parent_le(self):
        """Set the edge size in the parent QLineEdit."""
        # Set the edge size in the parent QLineEdit

        slider_val = self.edge_slider.value()
        if isinstance(slider_val, tuple):
            val = max(abs(slider_val[0]), abs(slider_val[1]))
            self.parent_le.setText(str(int(val)))
        else:
            self.parent_le.setText(str(int(slider_val)))
        self.close()

    def set_measurement_in_parent_list(self):
        """Add the edge size to the parent QListWidget."""
        # Add the edge size to the parent QListWidget
        # edge_slider is a QLabeledRangeSlider returning (min, max) tuple
        slider_val = self.edge_slider.value()
        if isinstance(slider_val, tuple):
            if self.single_value_mode:
                # For neighborhood: just output the max absolute distance
                edge_value = str(max(abs(slider_val[0]), abs(slider_val[1])))
            else:
                # For contour measurements: output as "(min,max)" tuple format
                edge_value = f"({slider_val[0]},{slider_val[1]})"
        else:
            edge_value = str(abs(slider_val))
        self.parent_list_widget.addItems([edge_value])
        self.close()

    def generate_label_imshow(self):
        """Generate the label imshow."""
        # Generate the label imshow

        self.im_mask = self.ax.imshow(
            np.ma.masked_where(self.edge_labels == 0, self.edge_labels),
            alpha=self.mask_alpha,
            interpolation="none",
            cmap="viridis",
        )
        self.canvas.canvas.draw()

    def generate_edge_slider(self):
        """Generate the edge size slider."""
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
        """Generate the opacity slider for the mask."""
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

    def change_mask_opacity(self, value: float) -> None:
        """
        Change the opacity of the mask.

        Parameters
        ----------
        value : float
            The alpha value.
        """
        # Change the opacity of the mask

        self.mask_alpha = value
        self.im_mask.set_alpha(self.mask_alpha)
        self.canvas.canvas.draw_idle()

    def change_edge_size(self, value: Union[Tuple[int, int], int, float]) -> None:
        """
        Change the edge size.

        Parameters
        ----------
        value : tuple or int
             The new edge size.
        """
        self.edge_size = value  # Tuple (min, max)
        self.compute_edge_labels()
        mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
        self.im_mask.set_data(mask)
        self.canvas.canvas.draw_idle()

    def change_frame(self, value: int) -> None:
        """
        Change the displayed frame and update the edge labels.

        Parameters
        ----------
        value : int
            The frame index.
        """
        # Change the displayed frame and update the edge labels

        super().change_frame(value)
        if not hasattr(self, "im_mask"):
            return

        # Check unified cache first
        if hasattr(self, "sdf_cache") and value in self.sdf_cache:
            self.init_label, self.dist_in, self.dist_out, self.voronoi_map = (
                self.sdf_cache[value]
            )
            self.sdf_cache.move_to_end(value)
        else:
            # Cache Miss: Load Label
            self.init_label = self.read_labels(value, self.labels, self.label_map)

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
        """Compute Signed Distance Functions (SDFs) and Voronoi map for the current labels."""
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
        """Compute the edge labels using Composite SDF and Range Masking."""
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

        # [2, 5] is a stripe inside, [-5, -2] outside, [-2, 2] across the border;
        # the Voronoi map carries the identity of the cells outside them.
        self.edge_labels = contour_of_instance_segmentation(
            self.init_label,
            distance=self.edge_size,
            sdf=sdf,
            voronoi_map=self.voronoi_map,
        )


class ContactEdgeVisualizer(CellEdgeVisualizer):
    """
    Set the contact distance of a mask-contact neighborhood.

    Both populations are grown outward by the distance, as the computation
    does (:func:`celldetective.neighborhood.contact_neighborhood`): a reference
    and a neighbor cell are in contact when their grown masks touch or overlap,
    so a neighbor inside a reference mask is always in contact. Each population
    is drawn in its own color, the mask in a dark shade and the grown edge in a
    light one.

    Parameters
    ----------
    neighbor_cell_type : str, optional
        The neighbor population; ``None`` or the reference population for a
        self neighborhood, then only one population is drawn.
    neighbor_labels : numpy.ndarray, optional
        Labels of the neighbor population (TYX), instead of the ones on disk.
    *args, **kwargs
        Passed to :class:`CellEdgeVisualizer`.
    """

    # (mask, edge) colors of the reference and neighbor populations.
    reference_colors = ("#1f77b4", "#9ecae9")
    neighbor_colors = ("#d95f02", "#fdbf6f")

    def __init__(
        self,
        neighbor_cell_type: Optional[str] = None,
        neighbor_labels: Optional[np.ndarray] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # Read while the parent class builds the viewer.
        self.neighbor_cell_type = neighbor_cell_type
        self.neighbor_labels = neighbor_labels
        super().__init__(*args, **kwargs)

    @property
    def two_populations(self) -> bool:
        return (
            self.neighbor_cell_type is not None
            and self.neighbor_cell_type != self.cell_type
        ) or self.neighbor_labels is not None

    def load_labels(self):
        """Load the labels of both populations at the first frame shown."""

        self.label_map = self.neighbor_label_map = None
        if self.labels is not None:
            self.labels = self.check_labels(self.labels)
            self.label_mode = "direct"
        else:
            self.label_mode = "virtual"
            self.check_virtual_stack()
            self.label_map = self.map_label_paths(self.find_label_paths(self.cell_type))

        if self.two_populations:
            if self.neighbor_labels is not None:
                self.neighbor_labels = self.check_labels(self.neighbor_labels)
            else:
                self.check_virtual_stack()
                self.neighbor_label_map = self.map_label_paths(
                    self.find_label_paths(self.neighbor_cell_type)
                )

        self.edge_size = abs(int(self.edge_size))
        self.load_frame_labels(self.frame_slider.value())

    def load_frame_labels(self, frame: int) -> None:
        """
        Per population, the distance in pixels of each pixel to the masks at
        a frame: 0 on a mask, capped at 255 (the slider stops well before).
        """
        from scipy.ndimage import distance_transform_edt

        if frame in self.sdf_cache:
            self.distances = self.sdf_cache[frame]
            self.sdf_cache.move_to_end(frame)
            return
        sources = [(self.labels, self.label_map)]
        if self.two_populations:
            sources.append((self.neighbor_labels, self.neighbor_label_map))
        self.distances = []
        for labels, label_map in sources:
            outside = self.read_labels(frame, labels, label_map) == 0
            if outside.all():
                distance = np.full(outside.shape, 255, dtype=np.uint8)
            else:
                distance = np.minimum(np.ceil(distance_transform_edt(outside)), 255).astype(np.uint8)
            self.distances.append(distance)
        self.sdf_cache[frame] = self.distances
        if len(self.sdf_cache) > self.max_sdf_cache_size:
            self.sdf_cache.popitem(last=False)

    def edge_norm(self):
        """0 (the mask) to the dark shade, up to the distance to the light one,
        further out to the transparent "over" color."""
        from matplotlib.colors import BoundaryNorm

        return BoundaryNorm([-0.5, 0.5, max(self.edge_size, 0.25) + 0.5], ncolors=2)

    def grown_masks(self):
        """Per population: 2 on the mask, 1 on its edge grown by the distance, masked elsewhere."""

        layers = []
        for distance in self.distances:
            layer = np.where(distance == 0, 2, np.where(distance <= self.edge_size, 1, 0))
            layers.append(np.ma.masked_where(layer == 0, layer))
        return layers

    def generate_label_imshow(self):
        """One layer per population, its colors set by :meth:`edge_norm`."""
        from matplotlib.colors import ListedColormap

        self.im_masks = []
        for distance, (dark, light) in zip(self.distances, [self.reference_colors, self.neighbor_colors]):
            cmap = ListedColormap([dark, light])
            cmap.set_over((0, 0, 0, 0))
            self.im_masks.append(
                self.ax.imshow(distance, alpha=self.mask_alpha, interpolation="none",
                               cmap=cmap, norm=self.edge_norm())
            )
        self.canvas.canvas.draw()

    def generate_edge_slider(self):
        """A single slider: how far outside the masks the edge reaches."""
        from celldetective.gui.base.sliders import QLabeledSlider

        # Negative like the outer contours of the measurements: outside the masks.
        self.edge_slider = QLabeledSlider(Qt.Horizontal)
        self.edge_slider.setRange(-(self.edge_range[1] if self.edge_range else 30), 0)
        self.edge_slider.setValue(-self.edge_size)
        self.edge_slider.setToolTip(
            "Distance in pixels the masks of both populations are grown by, outward.\n"
            "Cells are in contact when their grown masks touch or overlap."
        )
        self.edge_slider.valueChanged.connect(self.change_edge_size)

        edge_layout = QHBoxLayout()
        edge_layout.setContentsMargins(15, 0, 15, 0)
        edge_layout.addWidget(QLabel("Outer edge: "), 15)
        edge_layout.addWidget(self.edge_slider, 85)
        self.canvas.layout.addLayout(edge_layout)

    def change_mask_opacity(self, value: float) -> None:
        self.mask_alpha = value
        for im in self.im_masks:
            im.set_alpha(value)
        self.canvas.canvas.draw_idle()

    def change_edge_size(self, value: int) -> None:
        # The distance maps stay: only where the colors stop changes.
        self.edge_size = abs(int(value))
        for im in self.im_masks:
            im.set_norm(self.edge_norm())
        self.canvas.canvas.draw_idle()

    def change_frame(self, value: int) -> None:
        StackVisualizer.change_frame(self, value)
        if not hasattr(self, "im_masks"):
            return
        self.load_frame_labels(value)
        for im, distance in zip(self.im_masks, self.distances):
            im.set_data(distance)
        self.canvas.canvas.draw_idle()

    def set_measurement_in_parent_le(self):
        # The parent writes the larger end of a range; here the slider is negative.
        self.parent_le.setText(str(self.edge_size))
        self.close()
