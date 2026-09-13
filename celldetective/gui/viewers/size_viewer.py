from typing import Any, Literal, Optional, Tuple

import matplotlib.axes
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QLineEdit, QListWidget, QHBoxLayout, QPushButton, QLabel
from fonticon_mdi6 import MDI6
from superqt import QLabeledDoubleSlider
from superqt.fonticon import icon

from celldetective.gui.gui_utils import QuickSliderLayout
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective import get_logger

logger = get_logger(__name__)


class CellSizeViewer(StackVisualizer):
    """
    A widget for visualizing cell size with interactive sliders and circle display.

    Parameters:
    - initial_diameter (int): Initial diameter of the circle (40 by default).
    - set_radius_in_list (bool): Flag to set radius instead of diameter in the list (False by default).
    - diameter_slider_range (tuple): Range of the diameter slider (0, 200) by default.
    - parent_le: The parent QLineEdit instance to set the diameter.
    - parent_list_widget: The parent QListWidget instance to add diameter measurements.
    - follow_view_center (bool): Keep the circle at the centre of the zoomed view (True by default)
      rather than fixed at the centre of the image.
    - measure (str): Whether the slider and the value sent to `parent_le` are a "diameter"
      (default) or a "radius". `initial_diameter` and `diameter_slider_range` are always diameters.
    - args, kwargs: Additional arguments to pass to the parent class constructor.

    Methods:
    - generate_circle(): Generate the circle for visualization.
    - circle_center(): Centre of the circle in the image frame.
    - circle_radius(): Radius of the circle in pixels.
    - update_circle(): Redraw the circle after a change; extend it to redraw extra artists.
    - generate_add_to_list_btn(): Generate the add to list button.
    - set_measurement_in_parent_list(): Add the diameter to the parent QListWidget.
    - on_xlims_or_ylims_change(event_ax): Update the circle position on axis limits change.
    - generate_set_btn(): Generate the set button for QLineEdit.
    - set_threshold_in_parent_le(): Set the diameter in the parent QLineEdit.
    - generate_diameter_slider(): Generate the diameter slider.
    - change_diameter(value): Change the diameter of the circle.

    Notes:
    - This class extends the functionality of StackVisualizer to visualize cell size
      with interactive sliders for diameter adjustment and circle display.
    """

    def __init__(
        self,
        initial_diameter: float = 40,
        set_radius_in_list: bool = False,
        diameter_slider_range: Tuple[float, float] = (5, 200),
        parent_le: Optional[QLineEdit] = None,
        parent_list_widget: Optional[QListWidget] = None,
        *args: Any,
        follow_view_center: bool = True,
        measure: Literal["diameter", "radius"] = "diameter",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the CellSizeViewer.

        Parameters
        ----------
        initial_diameter : float, optional
            Initial diameter of the circle.
        set_radius_in_list : bool, optional
            Flag to set radius instead of diameter in the list.
        diameter_slider_range : tuple, optional
            Range of the diameter slider.
        parent_le : QLineEdit, optional
             Parent line edit for diameter.
        parent_list_widget : QListWidget, optional
            Parent list widget for measurements.
        *args
            Variable length argument list.
        follow_view_center : bool, optional
            Keep the circle at the centre of the zoomed view (default). If False, the circle
            stays at the centre of the image.
        measure : {"diameter", "radius"}, optional
            Unit of the slider and of the value set in `parent_le` (default "diameter").
        **kwargs
            Arbitrary keyword arguments.
        """
        # Initialize the widget and its attributes

        if measure not in ("diameter", "radius"):
            raise ValueError(f"measure must be 'diameter' or 'radius', got {measure!r}.")

        super().__init__(*args, **kwargs)
        self.diameter = initial_diameter
        self.parent_le = parent_le
        self.diameter_slider_range = diameter_slider_range
        self.parent_list_widget = parent_list_widget
        self.set_radius_in_list = set_radius_in_list
        self.follow_view_center = follow_view_center
        self.measure = measure
        self.generate_circle()
        self.generate_diameter_slider()

        if isinstance(self.parent_le, QLineEdit):
            self.generate_set_btn()
        if isinstance(self.parent_list_widget, QListWidget):
            self.generate_add_to_list_btn()

    def circle_center(self) -> Tuple[float, float]:
        """Centre (x, y) of the circle when it is not following the view."""
        return (self.init_frame.shape[1] // 2, self.init_frame.shape[0] // 2)

    def circle_radius(self) -> float:
        """Radius of the circle in pixels."""
        return float(self.diameter / 2.0 / self.PxToUm)

    def generate_circle(self):
        """Generate the circle for visualization."""
        # Generate the circle for visualization

        import matplotlib.pyplot as plt

        self.circ = plt.Circle(
            self.circle_center(),
            self.circle_radius(),
            ec="tab:red",
            fill=False,
        )
        self.ax.add_patch(self.circ)

        if self.follow_view_center:
            self.ax.callbacks.connect("xlim_changed", self.on_xlims_or_ylims_change)
            self.ax.callbacks.connect("ylim_changed", self.on_xlims_or_ylims_change)

    def update_circle(self):
        """Redraw the circle after a change of diameter."""
        self.circ.set_radius(self.circle_radius())
        self.canvas.canvas.draw_idle()

    def generate_add_to_list_btn(self):
        """Generate the add to list button."""
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

    def set_measurement_in_parent_list(self):
        """Add the diameter to the parent QListWidget."""
        # Add the diameter to the parent QListWidget

        if self.set_radius_in_list:
            val = int(self.diameter // 2)
        else:
            val = int(self.diameter)

        self.parent_list_widget.addItems([str(val)])
        self.close()

    def on_xlims_or_ylims_change(self, event_ax: matplotlib.axes.Axes) -> None:
        """
        Update the circle position on axis limits change.

        Parameters
        ----------
        event_ax : matplotlib.axes.Axes
            The axes object.
        """
        # Update the circle position on axis limits change

        xmin, xmax = event_ax.get_xlim()
        ymin, ymax = event_ax.get_ylim()
        self.circ.center = np.mean([xmin, xmax]), np.mean([ymin, ymax])

    def generate_set_btn(self):
        """Generate the set button for QLineEdit."""
        # Generate the set button for QLineEdit

        apply_hbox = QHBoxLayout()
        self.apply_threshold_btn = QPushButton("Set")
        self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
        self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
        apply_hbox.addWidget(QLabel(""), 33)
        apply_hbox.addWidget(self.apply_threshold_btn, 33)
        apply_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(apply_hbox)

    def set_threshold_in_parent_le(self):
        """Set the diameter (or radius, depending on `measure`) in the parent QLineEdit."""
        # Set the diameter in the parent QLineEdit

        self.parent_le.set_threshold(self.diameter_slider.value())
        self.close()

    def generate_diameter_slider(self):
        """Generate the diameter slider, expressed as a radius if `measure` is "radius"."""
        # Generate the diameter slider

        scale = 0.5 if self.measure == "radius" else 1.0
        self.diameter_slider = QLabeledDoubleSlider()
        diameter_layout = QuickSliderLayout(
            label=f"{self.measure.capitalize()}: ",
            slider=self.diameter_slider,
            slider_initial_value=self.diameter * scale,
            slider_range=tuple(v * scale for v in self.diameter_slider_range),
            decimal_option=True,
            precision=5,
        )
        diameter_layout.setContentsMargins(15, 0, 15, 0)
        self.diameter_slider.valueChanged.connect(self.change_diameter)
        self.canvas.layout.addLayout(diameter_layout)

    def change_diameter(self, value: float) -> None:
        """
        Change the diameter of the circle.

        Parameters
        ----------
        value : float
            The new slider value, a diameter (or a radius if `measure` is "radius").
        """
        # Change the diameter of the circle
        self.diameter = value * 2.0 if self.measure == "radius" else value
        self.update_circle()
