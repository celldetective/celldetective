from PyQt5.QtWidgets import (
    QComboBox,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from celldetective.gui.base.components import generic_message
from celldetective.gui.base.utils import center_window
from superqt import QColormapComboBox
from celldetective.gui.generic_signal_plot import SurvivalPlotWidget
from celldetective import (
    get_software_location,
)
from celldetective.utils.data_cleaning import extract_cols_from_table_list
from celldetective.utils.parsing import _extract_labels_from_config
from celldetective.utils.data_loaders import load_experiment_tables
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
from glob import glob
from celldetective.gui.base.styles import Styles
from celldetective.gui.base.components import CelldetectiveWidget
from matplotlib import colormaps
from celldetective.events import compute_survival
from celldetective.relative_measurements import expand_pair_table
import matplotlib.cm
from celldetective.neighborhood import extract_neighborhood_in_pair_table


class ConfigSurvival(CelldetectiveWidget):
    """
    UI to set survival instructions.

    """

    def __init__(self, parent_window=None):

        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Configure survival")

        self.exp_dir = self.parent_window.exp_dir
        self.soft_path = get_software_location()
        self.exp_config = self.exp_dir + "config.ini"
        self.wells = np.array(self.parent_window.parent_window.wells, dtype=str)
        self.well_labels = _extract_labels_from_config(self.exp_config, len(self.wells))
        self.FrameToMin = self.parent_window.parent_window.FrameToMin
        self.float_validator = QDoubleValidator()
        self.auto_close = False

        self.well_option = (
            self.parent_window.parent_window.well_list.getSelectedIndices()
        )
        self.position_option = (
            self.parent_window.parent_window.position_list.getSelectedIndices()
        )
        self.interpret_pos_location()
        # self.config_path = self.exp_dir + self.config_name

        self.screen_height = (
            self.parent_window.parent_window.parent_window.screen_height
        )

        self.setMinimumWidth(350)
        self.populate_widget()
        if self.auto_close:
            self.close()

    def interpret_pos_location(self):
        """
        Read the well/position selection from the control panel to decide which data to load
        Set position_indices to None if all positions must be taken

        """

        self.well_indices = (
            self.parent_window.parent_window.well_list.getSelectedIndices()
        )
        self.position_indices = (
            self.parent_window.parent_window.position_list.getSelectedIndices()
        )
        if self.position_indices == []:
            self.position_indices = None

    def populate_widget(self):
        """
        Create the multibox design.

        """

        # Create button widget and layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        main_layout.setContentsMargins(30, 30, 30, 30)

        panel_title = QLabel("Options")
        panel_title.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )
        main_layout.addWidget(panel_title, alignment=Qt.AlignCenter)

        pops = []
        self.cols_per_pop = {}
        for population in self.parent_window.parent_window.populations:
            tables = glob(
                self.exp_dir
                + os.sep.join(
                    ["W*", "*", "output", "tables", f"trajectories_{population}.csv"]
                )
            )
            if len(tables) > 0:
                pops.append(population)
                cols = extract_cols_from_table_list(tables)

                # check for neighbor pairs
                neigh_cols = [
                    c for c in cols if c.startswith("inclusive_count_neighborhood")
                ]
                neigh_pairs = [
                    c.split("_(")[-1].split(")_")[0].split("-") for c in neigh_cols
                ]
                neigh_pairs = ["-".join(c) for c in neigh_pairs]
                for k in range(len(neigh_pairs)):
                    if "_self_" in neigh_pairs[k]:
                        neigh_pairs[k] = "-".join([population, population])
                pops.extend(neigh_pairs)

                self.cols_per_pop.update({population: cols})

        labels = [
            QLabel("population: "),
            QLabel("time of\nreference: "),
            QLabel("time of\ninterest: "),
            QLabel("cmap: "),
        ]  # QLabel('class: '),
        self.cb_options = [pops, ["0"], [], []]  # ['class'],
        self.cbs = [QComboBox() for _ in range(len(labels))]

        self.cbs[-1] = QColormapComboBox()
        self.cbs[0].currentIndexChanged.connect(self.set_classes_and_times)

        choice_layout = QVBoxLayout()
        choice_layout.setContentsMargins(20, 20, 20, 20)
        for i in range(len(labels)):
            hbox = QHBoxLayout()
            hbox.addWidget(labels[i], 33)
            hbox.addWidget(self.cbs[i], 66)
            if i < len(labels) - 1:
                self.cbs[i].addItems(self.cb_options[i])
            choice_layout.addLayout(hbox)

        all_cms = list(colormaps)
        for cm in all_cms:
            if hasattr(matplotlib.cm, str(cm).lower()):
                try:
                    self.cbs[-1].addColormap(cm.lower())
                except Exception as _:
                    pass

        main_layout.addLayout(choice_layout)

        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("select cells\nwith query: "), 33)
        self.query_le = QLineEdit()
        select_layout.addWidget(self.query_le, 66)
        main_layout.addLayout(select_layout)

        time_cut_layout = QHBoxLayout()
        cut_time_lbl = QLabel("cut obs.\ntime [min]: ")
        cut_time_lbl.setToolTip(
            "Filter out later events from\nthe analysis (in absolute time)."
        )
        time_cut_layout.addWidget(cut_time_lbl, 33)
        self.query_time_cut = QLineEdit()
        self.query_time_cut.setValidator(self.float_validator)
        time_cut_layout.addWidget(self.query_time_cut, 66)
        main_layout.addLayout(time_cut_layout)

        self.set_classes_and_times()
        self.cbs[1].setCurrentText("t_firstdetection")

        time_calib_layout = QHBoxLayout()
        time_calib_layout.setContentsMargins(20, 20, 20, 20)
        time_calib_layout.addWidget(QLabel("time calibration\n(frame to min)"), 33)
        self.time_calibration_le = QLineEdit(str(self.FrameToMin).replace(".", ","))
        self.time_calibration_le.setValidator(self.float_validator)
        time_calib_layout.addWidget(self.time_calibration_le, 66)
        # time_calib_layout.addWidget(QLabel(' min'))
        main_layout.addLayout(time_calib_layout)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.process_survival)
        main_layout.addWidget(self.submit_btn)

        # self.populate_left_panel()
        # grid.addLayout(self.left_side, 0, 0, 1, 1)

        # self.setCentralWidget(self.scroll_area)
        # self.show()

    def set_classes_and_times(self):

        # Look for all classes and times
        self.neighborhood_keys = None
        self.population = self.cbs[0].currentText()
        pop_split = self.population.split("-")

        if len(pop_split) == 2:

            self.population = "pairs"
            tables_pairs = glob(
                self.exp_dir
                + os.sep.join(
                    ["W*", "*", "output", "tables", f"trajectories_pairs.csv"]
                )
            )
            if not tables_pairs:
                print("No pair table found... please compute the pair measurements...")
                return None
            self.cols_pairs = extract_cols_from_table_list(tables_pairs)

            self.population_reference = pop_split[0]
            self.population_neigh = pop_split[1]

            cols_ref = self.cols_per_pop[self.population_reference]
            cols_neigh = self.cols_per_pop[self.population_neigh]

            time_cols_ref = np.array(
                [s.startswith("t_") or s == "t0" for s in cols_ref]
            )
            if len(time_cols_ref) > 0:
                time_cols_ref = list(cols_ref[time_cols_ref])
                time_cols_ref = ["reference_" + t for t in time_cols_ref]

            time_cols_neigh = np.array(
                [s.startswith("t_") or s == "t0" for s in cols_neigh]
            )
            if len(time_cols_neigh) > 0:
                time_cols_neigh = list(cols_neigh[time_cols_neigh])
                time_cols_neigh = ["neighbor_" + t for t in time_cols_neigh]

            if self.population_reference != self.population_neigh:
                self.neighborhood_keys = [
                    c[16:]
                    for c in cols_ref
                    if c.startswith("inclusive_count_neighborhood")
                    and str(self.population_neigh) in c
                ]
            else:
                self.neighborhood_keys = [
                    c[16:]
                    for c in cols_ref
                    if c.startswith("inclusive_count_neighborhood")
                    and str(self.population_neigh) not in c
                ]

            print(f"{self.neighborhood_keys=}")

            time_idx = np.array(
                [s.startswith("t_") or s.startswith("t0") for s in self.cols_pairs]
            )
            time_cols_pairs = list(self.cols_pairs[time_idx])

            time_columns = time_cols_ref + time_cols_neigh + time_cols_pairs

        else:
            self.all_columns = self.cols_per_pop[self.population]
            time_idx = np.array(
                [s.startswith("t_") or s == "t0" for s in self.all_columns]
            )

            try:
                time_columns = list(self.all_columns[time_idx])
            except:
                print("no column starts with t")
                self.auto_close = True
                return None

        self.cbs[1].clear()
        self.cbs[1].addItems(np.unique(self.cb_options[1] + time_columns))
        self.cbs[1].setCurrentText("t_firstdetection")

        self.cbs[2].clear()
        self.cbs[2].addItems(np.unique(self.cb_options[2] + time_columns))

    def process_survival(self):

        self.FrameToMin = float(self.time_calibration_le.text().replace(",", "."))
        self.time_of_interest = self.cbs[2].currentText()
        if self.time_of_interest == "t0":
            self.class_of_interest = "class"
        elif self.time_of_interest.startswith("t0"):
            self.class_of_interest = self.time_of_interest.replace("t0_", "class_")
        else:
            self.class_of_interest = self.time_of_interest.replace("t_", "class_")

        # read instructions from combobox options
        self.load_available_tables_local()
        if self.df is not None:

            try:
                query_text = self.query_le.text()
                if query_text != "":
                    self.df = self.df.query(query_text)
            except Exception as e:
                print(e, " The query is misunderstood and will not be applied...")

            self.interpret_pos_location()

            if self.class_of_interest in list(self.df.columns) and self.cbs[
                2
            ].currentText() in list(self.df.columns):
                self.compute_survival_functions()
            else:
                generic_message(
                    "The class and/or event time of interest is not found in the dataframe..."
                )
                return None

            if "survival_fit" in list(self.df_pos_info.columns):
                self.plot_window = SurvivalPlotWidget(
                    parent_window=self,
                    df=self.df,
                    df_pos_info=self.df_pos_info,
                    df_well_info=self.df_well_info,
                    title="plot survivals",
                )
                self.plot_window.show()
            else:
                generic_message(
                    "No survival function was successfully computed...\nCheck your parameter choice."
                )
                return None

    def load_available_tables_local(self):
        """
        Load the tables of the selected wells/positions from the control Panel for the population of interest

        """

        self.well_option = (
            self.parent_window.parent_window.well_list.getSelectedIndices()
        )
        self.position_option = (
            self.parent_window.parent_window.position_list.getSelectedIndices()
        )

        self.df, self.df_pos_info = load_experiment_tables(
            self.exp_dir,
            well_option=self.well_option,
            position_option=self.position_option,
            population=self.population,
            return_pos_info=True,
        )

        if self.df is None:
            generic_message("No table could be found.. Abort.")
            return None
        else:
            self.df_well_info = self.df_pos_info.loc[
                :, ["well_path", "well_index", "well_name", "well_number", "well_alias"]
            ].drop_duplicates()

        if self.population == "pairs":
            self.df = expand_pair_table(self.df)
            self.df = extract_neighborhood_in_pair_table(
                self.df,
                reference_population=self.population_reference,
                neighbor_population=self.population_neigh,
                neighborhood_key=self.neighborhood_keys[0],
                contact_only=True,
            )

    def compute_survival_functions(self):

        cut_observation_time = None
        try:
            if self.query_time_cut.text() != "":
                cut_observation_time = (
                    float(self.query_time_cut.text().replace(",", "."))
                    / self.FrameToMin
                )
                if not 0 < cut_observation_time <= (self.df["FRAME"].max()):
                    print("Invalid cut time (larger than movie length)... Not applied.")
                    cut_observation_time = None
        except Exception as e:
            print(f"{e=}")
            pass

        pairs = False
        if self.neighborhood_keys is not None:
            pairs = True

        # Per position survival
        for block, movie_group in self.df.groupby(["well", "position"]):
            print(f"{block=}")
            ks = compute_survival(
                movie_group,
                self.class_of_interest,
                self.cbs[2].currentText(),
                t_reference=self.cbs[1].currentText(),
                FrameToMin=self.FrameToMin,
                cut_observation_time=cut_observation_time,
                pairs=pairs,
            )
            print(f"{ks=}")
            if ks is not None:
                self.df_pos_info.loc[
                    self.df_pos_info["pos_path"] == block[1], "survival_fit"
                ] = ks

        # Per well survival
        for well, well_group in self.df.groupby("well"):
            ks = compute_survival(
                well_group,
                self.class_of_interest,
                self.cbs[2].currentText(),
                t_reference=self.cbs[1].currentText(),
                FrameToMin=self.FrameToMin,
                cut_observation_time=cut_observation_time,
                pairs=pairs,
            )
            if ks is not None:
                self.df_well_info.loc[
                    self.df_well_info["well_path"] == well, "survival_fit"
                ] = ks

        self.df_pos_info.loc[:, "select"] = True
        self.df_well_info.loc[:, "select"] = True
