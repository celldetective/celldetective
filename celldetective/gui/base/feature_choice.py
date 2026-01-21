import ast
import os
from celldetective import get_software_location
from PyQt5.QtWidgets import QComboBox, QPushButton, QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window

CACHED_EXTRA_PROPERTIES = None


def get_extra_properties_functions():
    """
    Parses celldetective/extra_properties.py using AST to find function definitions
    without importing the module (which would trigger heavy dependencies).
    """
    try:
        software_path = get_software_location()
        extra_props_path = os.path.join(
            software_path, "celldetective", "extra_properties.py"
        )

        if not os.path.exists(extra_props_path):
            return []

        with open(extra_props_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        functions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        return functions
    except Exception as e:
        print(f"Failed to parse extra_properties.py: {e}")
        return []


class FeatureChoice(CelldetectiveWidget):

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Add feature")
        # Create the QComboBox and add some items
        self.combo_box = QComboBox(self)

        standard_measurements = [
            "area",
            "area_bbox",
            "area_convex",
            "area_filled",
            "major_axis_length",
            "minor_axis_length",
            "eccentricity",
            "equivalent_diameter_area",
            "euler_number",
            "extent",
            "feret_diameter_max",
            "orientation",
            "perimeter",
            "perimeter_crofton",
            "solidity",
            "intensity_mean",
            "intensity_max",
            "intensity_min",
        ]

        global CACHED_EXTRA_PROPERTIES
        if CACHED_EXTRA_PROPERTIES is None:
            CACHED_EXTRA_PROPERTIES = get_extra_properties_functions()

        if CACHED_EXTRA_PROPERTIES:
            standard_measurements.extend(CACHED_EXTRA_PROPERTIES)

        self.combo_box.addItems(standard_measurements)

        self.add_btn = QPushButton("Add")
        self.add_btn.setStyleSheet(self.button_style_sheet)
        self.add_btn.clicked.connect(self.add_current_feature)

        # Create the layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.add_btn)
        center_window(self)

    def add_current_feature(self):
        filtername = self.combo_box.currentText()
        self.parent_window.list_widget.addItems([filtername])
        self.close()
