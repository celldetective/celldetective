from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QCheckBox,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import DistanceChoice
from celldetective.gui.base.list_widget import ListWidget
from celldetective.gui.base.utils import center_window
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
import json
import os
from glob import glob
import pandas as pd
from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer
from celldetective.gui.viewers.size_viewer import CellSizeViewer

from celldetective.gui.base.components import CelldetectiveWidget


class SettingsNeighborhood(CelldetectiveWidget):
    """
    Widget to configure neighborhood measurements.

    """

    def __init__(
        self,
        neighborhood_type="distance_threshold",
        neighborhood_parameter_name="threshold distance",
        parent_window=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.parent_window = parent_window
        self.attr_parent = self.parent_window.parent_window

        self.neighborhood_type = neighborhood_type
        self.neighborhood_parameter_name = neighborhood_parameter_name

        self.setWindowTitle("Configure neighborhoods")
        self.neigh_instructions = self.attr_parent.exp_dir + os.sep.join(
            ["configs", "neighborhood_instructions.json"]
        )
        self.clear_previous = False
        self.not_status_reference = False
        self.not_status_neighbor = False

        self.screen_height = self.attr_parent.screen_height
        self.setMinimumWidth(750)
        self.setMinimumHeight(int(0.5 * self.screen_height))
        self.setMaximumHeight(int(0.95 * self.screen_height))

        self.generate_main_layout()
        self.load_previous_neighborhood_instructions()
        center_window(self)

    def generate_main_layout(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)

        populations_layout = QHBoxLayout()

        # Reference population
        self.reference_population_frame = QFrame()
        self.reference_population_frame.setFrameStyle(
            QFrame.StyledPanel | QFrame.Raised
        )
        self.populate_reference_frame()
        populations_layout.addWidget(self.reference_population_frame, 50)

        # Neighbor population
        self.neigh_population_frame = QFrame()
        self.neigh_population_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_neighbor_frame()
        populations_layout.addWidget(self.neigh_population_frame, 50)
        main_layout.addLayout(populations_layout)

        # Measurements

        self.measurement_frame = QFrame()
        self.measurement_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.populate_measurement_frame()
        # if self.neighborhood_type=='distance_threshold':
        # 	self.populate_radii_frame()
        # elif self.neighborhood_type=='mask_contact':
        # 	self.populate_contact_frame()
        main_layout.addWidget(self.measurement_frame)

        self.clear_previous_btn = QCheckBox("clear previous neighborhoods")
        self.clear_previous_btn.setToolTip(
            "Clear all previous neighborhood measurements."
        )
        self.clear_previous_btn.setIcon(icon(MDI6.broom, color="black"))

        main_layout.addWidget(self.clear_previous_btn, alignment=Qt.AlignRight)

        main_layout.addWidget(QLabel(""))
        self.submit_btn = QPushButton("Set")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.write_instructions)
        main_layout.addWidget(self.submit_btn)

        self.adjustSize()
        QApplication.processEvents()

    def populate_measurement_frame(self):
        """
        Add widgets and layout in the radii frame.
        """

        grid = QVBoxLayout(self.measurement_frame)

        self.dist_lbl = QLabel(
            f"NEIGHBORHOOD {self.neighborhood_parameter_name.upper()}"
        )
        self.dist_lbl.setStyleSheet(self.block_title)
        grid.addWidget(self.dist_lbl, alignment=Qt.AlignCenter)

        self.ContentsMeasurements = QFrame()
        layout = QVBoxLayout(self.ContentsMeasurements)
        layout.setContentsMargins(0, 0, 0, 0)

        list_header_layout = QHBoxLayout()
        meas = self.neighborhood_parameter_name.split(" ")[-1]
        lbl = QLabel(f"{meas} [px]:")
        lbl.setToolTip("")
        list_header_layout.addWidget(lbl, 85)

        self.delete_measurement_btn = QPushButton("")
        self.delete_measurement_btn.setStyleSheet(self.button_select_all)
        self.delete_measurement_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_measurement_btn.setToolTip("Remove measurement.")
        self.delete_measurement_btn.setIconSize(QSize(20, 20))
        list_header_layout.addWidget(self.delete_measurement_btn, 5)

        self.add_measurement_btn = QPushButton("")
        self.add_measurement_btn.setStyleSheet(self.button_select_all)
        self.add_measurement_btn.setIcon(icon(MDI6.plus, color="black"))
        self.add_measurement_btn.setToolTip("Add a neighborhood measurement.")
        self.add_measurement_btn.setIconSize(QSize(20, 20))
        list_header_layout.addWidget(self.add_measurement_btn, 5)

        self.viewer_btn = QPushButton()
        self.viewer_btn.setStyleSheet(self.button_select_all)
        self.viewer_btn.setIcon(icon(MDI6.image_check, color="black"))
        self.viewer_btn.setToolTip("View stack and set measurement.")
        self.viewer_btn.setIconSize(QSize(20, 20))
        if self.neighborhood_type == "distance_threshold":
            self.viewer_btn.clicked.connect(self.view_current_stack_with_circle)
        elif self.neighborhood_type == "mask_contact":
            self.viewer_btn.clicked.connect(self.view_current_stack_with_edge)
        list_header_layout.addWidget(self.viewer_btn, 5)

        layout.addLayout(list_header_layout)

        self.measurements_list = ListWidget(
            DistanceChoice, initial_features=[], dtype=int
        )
        self.measurements_list.setToolTip("Neighborhoods to compute.")
        layout.addWidget(self.measurements_list)

        self.delete_measurement_btn.clicked.connect(self.measurements_list.removeSel)
        self.add_measurement_btn.clicked.connect(self.measurements_list.addItem)

        grid.addWidget(self.ContentsMeasurements)

    def view_current_stack_with_circle(self):

        self.parent_window.parent_window.locate_image()
        if self.parent_window.parent_window.current_stack is not None:
            self.viewer = CellSizeViewer(
                initial_diameter=100,
                parent_list_widget=self.measurements_list.list_widget,
                set_radius_in_list=True,
                stack_path=self.parent_window.parent_window.current_stack,
                window_title=f"Position {self.parent_window.parent_window.position_list.currentText()}",
                frame_slider=True,
                contrast_slider=True,
                channel_cb=True,
                diameter_slider_range=(0, 300),
                channel_names=self.parent_window.parent_window.exp_channels,
                n_channels=self.parent_window.parent_window.nbr_channels,
                PxToUm=1,
            )
            self.viewer.show()

    def view_current_stack_with_edge(self):

        self.attr_parent.locate_image()
        if self.attr_parent.current_stack is not None:
            self.viewer = CellEdgeVisualizer(
                cell_type=self.reference_population_cb.currentText(),
                edge_range=(1, 30),
                invert=True,
                initial_edge=3,
                parent_list_widget=self.measurements_list.list_widget,
                stack_path=self.attr_parent.current_stack,
                window_title=f"Position {self.attr_parent.position_list.currentText()}",
                frame_slider=True,
                contrast_slider=True,
                channel_cb=True,
                channel_names=self.attr_parent.exp_channels,
                n_channels=self.attr_parent.nbr_channels,
                PxToUm=1,
            )
            self.viewer.show()

    def populate_reference_frame(self):
        """
        Add widgets and layout in the reference population frame.
        """

        grid = QVBoxLayout(self.reference_population_frame)
        grid.setSpacing(15)
        self.ref_lbl = QLabel("REFERENCE")
        self.ref_lbl.setStyleSheet(self.block_title)
        self.ref_lbl.setToolTip("Reference population settings.")
        grid.addWidget(self.ref_lbl, 30, alignment=Qt.AlignCenter)

        self.ContentsReference = QFrame()
        layout = QVBoxLayout(self.ContentsReference)
        layout.setContentsMargins(15, 15, 15, 15)

        population_layout = QHBoxLayout()
        population_layout.addWidget(QLabel("population: "), 30)
        self.reference_population_cb = QComboBox()
        self.reference_population_cb.addItems(
            self.parent_window.parent_window.populations
        )
        self.reference_population_cb.setToolTip("Select a reference population.")
        population_layout.addWidget(self.reference_population_cb, 70)
        layout.addLayout(population_layout)

        status_layout = QHBoxLayout()

        # status_layout.addWidget(QLabel('status: '), 30)

        status_sublayout = QHBoxLayout()
        self.reference_population_status_cb = QComboBox()
        self.reference_population_status_cb.setToolTip(
            "Status of the reference population."
        )
        self.reference_population_status_cb.hide()
        status_sublayout.addWidget(self.reference_population_status_cb, 95)

        self.reference_switch_status_btn = QPushButton("")
        self.reference_switch_status_btn.setStyleSheet(self.button_select_all)
        self.reference_switch_status_btn.setIcon(
            icon(MDI6.invert_colors, color="black")
        )
        self.reference_switch_status_btn.setIconSize(QSize(20, 20))
        self.reference_switch_status_btn.clicked.connect(self.switch_not_reference)
        self.reference_switch_status_btn.setToolTip("Invert status values.")
        self.reference_switch_status_btn.hide()

        status_sublayout.addWidget(self.reference_switch_status_btn, 5)

        status_layout.addLayout(status_sublayout, 70)
        layout.addLayout(status_layout)

        event_layout = QHBoxLayout()
        event_layout.addWidget(QLabel("event time: "), 30)
        self.event_time_cb = QComboBox()
        self.event_time_cb.setToolTip(
            "Compute average neighborhood metrics before and after this event time."
        )
        event_layout.addWidget(self.event_time_cb, 70)
        layout.addLayout(event_layout)

        self.fill_cbs_of_reference_population()
        self.reference_population_cb.currentIndexChanged.connect(
            self.fill_cbs_of_reference_population
        )

        grid.addWidget(self.ContentsReference, 70)

    def populate_neighbor_frame(self):
        """
        Add widgets and layout in the neighbor population frame.
        """

        grid = QVBoxLayout(self.neigh_population_frame)
        grid.setSpacing(15)
        self.ref_lbl = QLabel("NEIGHBOR")
        self.ref_lbl.setStyleSheet(self.block_title)
        self.ref_lbl.setToolTip("Neighbor population settings.")
        grid.addWidget(self.ref_lbl, 30, alignment=Qt.AlignCenter)

        self.ContentsNeigh = QFrame()
        layout = QVBoxLayout(self.ContentsNeigh)
        layout.setContentsMargins(15, 15, 15, 15)

        population_layout = QHBoxLayout()
        population_layout.addWidget(QLabel("population: "), 30)
        self.neighbor_population_cb = QComboBox()
        self.neighbor_population_cb.addItems(
            self.parent_window.parent_window.populations
        )
        self.neighbor_population_cb.setToolTip("Select a neighbor population.")
        population_layout.addWidget(self.neighbor_population_cb, 70)
        layout.addLayout(population_layout)

        status_layout = QHBoxLayout()

        status_layout.addWidget(QLabel("status: "), 30)
        status_sublayout = QHBoxLayout()

        self.neighbor_population_status_cb = QComboBox()
        self.neighbor_population_status_cb.setToolTip(
            "Status of the neighbor population."
        )
        status_sublayout.addWidget(self.neighbor_population_status_cb, 95)

        self.neighbor_switch_status_btn = QPushButton("")
        self.neighbor_switch_status_btn.setStyleSheet(self.button_select_all)
        self.neighbor_switch_status_btn.setIcon(icon(MDI6.invert_colors, color="black"))
        self.neighbor_switch_status_btn.setToolTip("Invert status values.")
        self.neighbor_switch_status_btn.setIconSize(QSize(20, 20))
        self.neighbor_switch_status_btn.clicked.connect(self.switch_not_neigh)
        status_sublayout.addWidget(self.neighbor_switch_status_btn, 5)
        status_layout.addLayout(status_sublayout, 70)
        layout.addLayout(status_layout)

        self.cumulated_presence_btn = QCheckBox("cumulated presence")
        self.cumulated_presence_btn.setToolTip(
            "Compute the cumulated presence time of each neighbor around a reference cell."
        )
        self.cumulated_presence_btn.setIcon(icon(MDI6.timer_outline, color="black"))

        layout.addWidget(self.cumulated_presence_btn)

        # self.symmetrize_btn = QCheckBox('symmetrize')
        # self.symmetrize_btn.setToolTip("Write the neighborhood of the neighbor cells with respect to the reference cells.")
        # layout.addWidget(self.symmetrize_btn)

        self.fill_cbs_of_neighbor_population()
        self.neighbor_population_cb.currentIndexChanged.connect(
            self.fill_cbs_of_neighbor_population
        )

        grid.addWidget(self.ContentsNeigh, 70)

    def fill_cbs_of_neighbor_population(self):

        population = self.neighbor_population_cb.currentText()
        class_cols, status_cols, group_cols, time_cols = (
            self.locate_population_specific_columns(population)
        )
        self.neighbor_population_status_cb.clear()
        self.neighbor_population_status_cb.addItems(
            list(
                np.unique(
                    ["--", "class", "status"] + class_cols + status_cols + group_cols
                )
            )
        )

    def fill_cbs_of_reference_population(self):

        population = self.reference_population_cb.currentText()
        class_cols, status_cols, group_cols, time_cols = (
            self.locate_population_specific_columns(population)
        )
        self.event_time_cb.clear()
        self.event_time_cb.addItems(list(np.unique(["--", "t0"] + time_cols)))

    def switch_not_reference(self):

        self.not_status_reference = not self.not_status_reference
        if self.not_status_reference:
            self.reference_switch_status_btn.setIcon(
                icon(MDI6.invert_colors, color=self.celldetective_blue)
            )
            self.reference_switch_status_btn.setIconSize(QSize(20, 20))
        else:
            self.reference_switch_status_btn.setIcon(
                icon(MDI6.invert_colors, color="black")
            )
            self.reference_switch_status_btn.setIconSize(QSize(20, 20))

    def switch_not_neigh(self):

        self.not_status_neighbor = not self.not_status_neighbor
        if self.not_status_neighbor:
            self.neighbor_switch_status_btn.setIcon(
                icon(MDI6.invert_colors, color=self.celldetective_blue)
            )
            self.neighbor_switch_status_btn.setIconSize(QSize(20, 20))
        else:
            self.neighbor_switch_status_btn.setIcon(
                icon(MDI6.invert_colors, color="black")
            )
            self.neighbor_switch_status_btn.setIconSize(QSize(20, 20))

    def locate_population_specific_columns(self, population):

        # Look for all classes and times
        tables = glob(
            self.attr_parent.exp_dir
            + os.sep.join(
                ["W*", "*", "output", "tables", f"trajectories_{population}.csv"]
            )
        )
        self.all_columns = []
        for tab in tables:
            cols = pd.read_csv(tab, nrows=1).columns.tolist()
            self.all_columns.extend(cols)
        self.all_columns = np.unique(self.all_columns)

        class_idx = np.array([s.startswith("class_") for s in self.all_columns])
        status_idx = np.array([s.startswith("status_") for s in self.all_columns])
        group_idx = np.array([s.startswith("group_") for s in self.all_columns])
        time_idx = np.array([s.startswith("t_") for s in self.all_columns])

        if len(class_idx) > 0:
            class_columns = list(self.all_columns[class_idx])
            for c in ["class_id", "class_color"]:
                if c in class_columns:
                    class_columns.remove(c)
        else:
            class_columns = []

        if len(status_idx) > 0:
            status_columns = list(self.all_columns[status_idx])
        else:
            status_columns = []

        if len(group_idx) > 0:
            group_columns = list(self.all_columns[group_idx])
        else:
            group_columns = []

        if len(time_idx) > 0:
            time_columns = list(self.all_columns[time_idx])
        else:
            time_columns = []

        return class_columns, status_columns, group_columns, time_columns

    def write_instructions(self):
        """
        Write the selected options in a json file for later reading by the software.
        """

        print("Writing instructions...")

        neighborhood_options = {}
        pop = [
            self.reference_population_cb.currentText(),
            self.neighbor_population_cb.currentText(),
        ]
        neighborhood_options.update({"population": pop})

        status_options = [
            self.reference_population_status_cb.currentText(),
            self.neighbor_population_status_cb.currentText(),
        ]
        for k in range(2):
            if status_options[k] == "--" or status_options[k] == "":
                status_options[k] = None
        if pop[0] != pop[1]:
            mode = "two-pop"
        else:
            mode = "self"

        # TO ADAPT
        distances = self.measurements_list.getItems()
        neighborhood_options.update({"neighborhood_type": self.neighborhood_type})
        neighborhood_options.update({"distance": distances})
        neighborhood_options.update(
            {"clear_neigh": self.clear_previous_btn.isChecked()}
        )
        event_time_col = self.event_time_cb.currentText()
        if event_time_col == "--":
            event_time_col = None
        neighborhood_options.update({"event_time_col": event_time_col})

        neighborhood_kwargs = {
            "mode": mode,
            "status": status_options,
            "not_status_option": [self.not_status_reference, self.not_status_neighbor],
            "compute_cum_sum": self.cumulated_presence_btn.isChecked(),
            "attention_weight": True,
            "symmetrize": False,
            "include_dead_weight": True,
        }

        neighborhood_options.update({"neighborhood_kwargs": neighborhood_kwargs})

        print("Neighborhood instructions: ", neighborhood_options)
        file_name = self.neigh_instructions
        with open(file_name, "w") as f:
            json.dump(neighborhood_options, f, indent=4)

        self.parent_window.protocols.append(neighborhood_options)
        correction_description = ""
        for index, (key, value) in enumerate(neighborhood_options.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_list.addItem(correction_description)

        print("Done.")
        self.close()

    def load_previous_neighborhood_instructions(self):
        """
        Read the measurmeent options from a previously written json file and format properly for the UI.
        """

        print("Reading instructions..")
        if os.path.exists(self.neigh_instructions):
            with open(self.neigh_instructions, "r") as f:

                neigh_instructions = json.load(f)
                print(f"Loading the instructions: {neigh_instructions}...")

                if "neighborhood_type" not in neigh_instructions:
                    neigh_instructions.update(
                        {"neighborhood_type": self.neighborhood_type}
                    )

                if self.neighborhood_type == neigh_instructions["neighborhood_type"]:

                    if "distance" in neigh_instructions:
                        distances = neigh_instructions["distance"]
                        distances = [str(d) for d in distances]
                        self.measurements_list.list_widget.clear()
                        self.measurements_list.list_widget.addItems(distances)

                    if "population" in neigh_instructions:

                        pop = neigh_instructions["population"]
                        idx0 = self.reference_population_cb.findText(pop[0])
                        self.reference_population_cb.setCurrentIndex(idx0)
                        idx1 = self.neighbor_population_cb.findText(pop[1])
                        self.neighbor_population_cb.setCurrentIndex(idx1)

                    if "clear_neigh" in neigh_instructions:
                        clear_neigh = neigh_instructions["clear_neigh"]
                        self.clear_previous_btn.setChecked(clear_neigh)

                    if "event_time_col" in neigh_instructions:
                        event_time_col = neigh_instructions["event_time_col"]
                        if event_time_col is None:
                            event_time_col = "--"
                        idx = self.event_time_cb.findText(event_time_col)
                        self.event_time_cb.setCurrentIndex(idx)

                    if "neighborhood_kwargs" in neigh_instructions:
                        neighborhood_kwargs = neigh_instructions["neighborhood_kwargs"]
                        if "compute_cum_sum" in neighborhood_kwargs:
                            self.cumulated_presence_btn.setChecked(
                                neighborhood_kwargs["compute_cum_sum"]
                            )
                        if "status" in neighborhood_kwargs:
                            status_options = neighborhood_kwargs["status"]
                            status_options = [
                                "--" if s is None else s for s in status_options
                            ]
                            idx0 = self.reference_population_status_cb.findText(
                                status_options[0]
                            )
                            self.reference_population_status_cb.setCurrentIndex(idx0)
                            idx1 = self.neighbor_population_status_cb.findText(
                                status_options[1]
                            )
                            self.neighbor_population_status_cb.setCurrentIndex(idx1)
                        if "not_status_option" in neighborhood_kwargs:
                            not_status_option = neighborhood_kwargs["not_status_option"]
                            if not_status_option[0]:
                                self.reference_switch_status_btn.click()
                            if not_status_option[1]:
                                self.neighbor_switch_status_btn.click()
