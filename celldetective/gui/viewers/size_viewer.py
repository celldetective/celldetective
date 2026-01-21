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
    - args, kwargs: Additional arguments to pass to the parent class constructor.

    Methods:
    - generate_circle(): Generate the circle for visualization.
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
        initial_diameter=40,
        set_radius_in_list=False,
        diameter_slider_range=(0, 500),
        parent_le=None,
        parent_list_widget=None,
        *args,
        **kwargs,
    ):
        # Initialize the widget and its attributes

        super().__init__(*args, **kwargs)
        self.diameter = initial_diameter
        self.parent_le = parent_le
        self.diameter_slider_range = diameter_slider_range
        self.parent_list_widget = parent_list_widget
        self.set_radius_in_list = set_radius_in_list
        self.generate_circle()
        self.generate_diameter_slider()

        if isinstance(self.parent_le, QLineEdit):
            self.generate_set_btn()
        if isinstance(self.parent_list_widget, QListWidget):
            self.generate_add_to_list_btn()

    def generate_circle(self):
        # Generate the circle for visualization

        import matplotlib.pyplot as plt

        self.circ = plt.Circle(
            (self.init_frame.shape[0] // 2, self.init_frame.shape[1] // 2),
            self.diameter // 2 / self.PxToUm,
            ec="tab:red",
            fill=False,
        )
        self.ax.add_patch(self.circ)

        self.ax.callbacks.connect("xlim_changed", self.on_xlims_or_ylims_change)
        self.ax.callbacks.connect("ylim_changed", self.on_xlims_or_ylims_change)

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

    def set_measurement_in_parent_list(self):
        # Add the diameter to the parent QListWidget

        if self.set_radius_in_list:
            val = int(self.diameter_slider.value() // 2)
        else:
            val = int(self.diameter_slider.value())

        self.parent_list_widget.addItems([str(val)])
        self.close()

    def on_xlims_or_ylims_change(self, event_ax):
        # Update the circle position on axis limits change

        xmin, xmax = event_ax.get_xlim()
        ymin, ymax = event_ax.get_ylim()
        self.circ.center = np.mean([xmin, xmax]), np.mean([ymin, ymax])

    def generate_set_btn(self):
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
        # Set the diameter in the parent QLineEdit

        self.parent_le.set_threshold(self.diameter_slider.value())
        self.close()

    def generate_diameter_slider(self):
        # Generate the diameter slider

        self.diameter_slider = QLabeledDoubleSlider()
        diameter_layout = QuickSliderLayout(
            label="Diameter: ",
            slider=self.diameter_slider,
            slider_initial_value=self.diameter,
            slider_range=self.diameter_slider_range,
            decimal_option=True,
            precision=5,
        )
        diameter_layout.setContentsMargins(15, 0, 15, 0)
        self.diameter_slider.valueChanged.connect(self.change_diameter)
        self.canvas.layout.addLayout(diameter_layout)

    def change_diameter(self, value):
        # Change the diameter of the circle
        self.diameter = value
        self.circ.set_radius(self.diameter // 2 / self.PxToUm)
        self.canvas.canvas.draw_idle()
