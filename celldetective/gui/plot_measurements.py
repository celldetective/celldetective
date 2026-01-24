from PyQt5.QtWidgets import (
    QMessageBox,
    QScrollArea,
    QButtonGroup,
    QComboBox,
    QCheckBox,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QDoubleValidator

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.utils import center_window

from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective import get_software_location
from celldetective.utils.parsing import _extract_labels_from_config
from celldetective.utils.data_loaders import load_experiment_tables
from celldetective.utils.experiment import (
    get_experiment_wells,
    get_experiment_concentrations,
    get_experiment_cell_types,
    get_experiment_antibodies,
    get_positions_in_well,
)
import numpy as np
import json
import os
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
from glob import glob
import pandas as pd
from matplotlib.cm import tab10
import math
import seaborn as sns


class ConfigMeasurementsPlot(CelldetectiveWidget):
    """
    UI to set survival instructions.

    """

    def __init__(self, parent_window=None):

        super().__init__()
        self.parent_window = parent_window
        self.setWindowTitle("Configure signal plot")
        self.exp_dir = self.parent_window.exp_dir
        self.soft_path = get_software_location()
        self.exp_config = self.exp_dir + "config.ini"
        self.wells = np.array(self.parent_window.parent_window.wells, dtype=str)
        self.well_labels = _extract_labels_from_config(self.exp_config, len(self.wells))
        self.FrameToMin = self.parent_window.parent_window.FrameToMin
        self.float_validator = QDoubleValidator()
        self.target_class = [0, 1]
        self.show_ci = True
        self.show_cell_lines = False
        self.ax2 = None
        self.auto_close = False
        self.palette = sns.cubehelix_palette()
        # sns.color_palette("cubehelix", as_cmap=True)
        # sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

        print("Parent wells: ", self.wells)

        self.well_option = (
            self.parent_window.parent_window.well_list.getSelectedIndices()
        )
        self.position_option = (
            self.parent_window.parent_window.position_list.getSelectedIndices()
        )
        self.interpret_pos_location()
        # self.load_available_tables()
        # self.config_path = self.exp_dir + self.config_name

        self.screen_height = (
            self.parent_window.parent_window.parent_window.screen_height
        )
        center_window(self)
        # self.setMinimumHeight(int(0.8*self.screen_height))
        # self.setMaximumHeight(int(0.8*self.screen_height))
        self.populate_widget()
        # self.load_previous_measurement_instructions()
        if self.auto_close:
            self.close()
        self.hue_colors = {0: self.palette[0], 1: self.palette[3]}

    def interpret_pos_location(self):
        """
        Read the well/position selection from the control panel to decide which data to load
        Set position_indices to None if all positions must be taken

        """

        if self.well_option == len(self.wells):
            self.well_indices = np.arange(len(self.wells))
        else:
            self.well_indices = np.array([self.well_option], dtype=int)

        if self.position_option == 0:
            self.position_indices = None
        else:
            self.position_indices = np.array([self.position_option], dtype=int)

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

        labels = [
            QLabel("population: "),
            QLabel("class: "),
            QLabel("group: "),
        ]  # , QLabel('time of\ninterest: ')]
        self.cb_options = [
            self.parent_window.parent_window.populations,
            ["class"],
            ["group"],
        ]  # , ['t0','first detection']]
        self.cbs = [QComboBox() for i in range(len(labels))]
        self.cbs[0].currentIndexChanged.connect(self.set_classes_and_times)

        choice_layout = QVBoxLayout()
        choice_layout.setContentsMargins(20, 20, 20, 20)
        for i in range(len(labels)):
            hbox = QHBoxLayout()
            hbox.addWidget(labels[i], 33)
            hbox.addWidget(self.cbs[i], 66)
            self.cbs[i].addItems(self.cb_options[i])
            if i == 1:
                vbox = QVBoxLayout()
                self.check_class = QCheckBox("plot by class")
                vbox.addWidget(self.check_class)
                vbox.addLayout(hbox)
                choice_layout.addLayout(vbox)
            if i == 2:
                vbox = QVBoxLayout()
                self.check_group = QCheckBox("plot by group")
                vbox.addWidget(self.check_group)
                vbox.addLayout(hbox)
                choice_layout.addLayout(vbox)
            else:
                choice_layout.addLayout(hbox)

        self.cbs[0].setCurrentIndex(1)
        self.cbs[0].setCurrentIndex(0)
        self.cbs[1].setEnabled(False)
        self.check_class.toggled.connect(self.class_enable)
        self.cbs[2].setEnabled(False)
        self.check_group.toggled.connect(self.group_enable)

        # self.abs_time_checkbox = QCheckBox('absolute time')
        # self.frame_slider = QLabeledSlider()
        # self.frame_slider.setSingleStep(1)
        # self.frame_slider.setOrientation(Qt.Horizontal)
        # self.frame_slider.setRange(0,self.parent.parent.len_movie)
        # self.frame_slider.setValue(0)
        # self.frame_slider.setEnabled(False)
        # slider_hbox = QHBoxLayout()
        # slider_hbox.addWidget(self.abs_time_checkbox, 33)
        # slider_hbox.addWidget(self.frame_slider, 66)
        # choice_layout.addLayout(slider_hbox)
        main_layout.addLayout(choice_layout)

        # self.abs_time_checkbox.stateChanged.connect(self.switch_ref_time_mode)

        # time_calib_layout = QHBoxLayout()
        # time_calib_layout.setContentsMargins(20,20,20,20)
        # time_calib_layout.addWidget(QLabel('time calibration\n(frame to min)'), 33)
        # self.time_calibration_le = QLineEdit(str(self.FrameToMin).replace('.',','))
        # self.time_calibration_le.setValidator(self.float_validator)
        # time_calib_layout.addWidget(self.time_calibration_le, 66)
        # #time_calib_layout.addWidget(QLabel(' min'))
        # main_layout.addLayout(time_calib_layout)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.process_signal)
        main_layout.addWidget(self.submit_btn)

    # self.populate_left_panel()
    # grid.addLayout(self.left_side, 0, 0, 1, 1)

    # self.setCentralWidget(self.scroll_area)
    # self.show()
    def class_enable(self):
        if self.check_class.isChecked():
            self.cbs[1].setEnabled(True)
            self.check_group.setChecked(False)
        else:
            self.cbs[1].setEnabled(False)

    def group_enable(self):
        if self.check_group.isChecked():
            self.cbs[2].setEnabled(True)
            self.check_class.setChecked(False)
        else:
            self.cbs[2].setEnabled(False)

    def set_classes_and_times(self):
        ext = "csv"
        # Look for all classes and times
        tables = glob(
            self.exp_dir
            + os.sep.join(["W*", "*", "output", "tables", f"trajectories_*{ext}"])
        )
        self.all_columns = []
        for tab in tables:
            cols = pd.read_csv(tab, nrows=1).columns.tolist()
            self.all_columns.extend(cols)
        self.all_columns = np.unique(self.all_columns)
        class_idx = np.array([s.startswith("class_") for s in self.all_columns])
        group_idx = np.array([s.startswith("group_") for s in self.all_columns])

        print(f"{class_idx=} {group_idx=} {self.all_columns=}")
        # time_idx = np.array([s.startswith('t_') for s in self.all_columns])
        try:
            if len(class_idx) > 0:
                class_columns = list(self.all_columns[class_idx])
            else:
                class_columns = []
            if len(group_idx) > 0:
                group_columns = list(self.all_columns[group_idx])
            else:
                group_columns = []
        # time_columns = list(self.all_columns[time_idx])
        except Exception as e:
            print(f"L210 columns not found {e}")
            self.auto_close = True
            return None

        # self.cbs[2].clear()
        # self.cbs[2].addItems(np.unique(self.cb_options[2]+time_columns))

        self.cbs[1].clear()
        self.cbs[1].addItems(np.unique(self.cb_options[1] + class_columns))
        self.cbs[2].clear()
        self.cbs[2].addItems(np.unique(group_columns))

    def ask_for_feature(self):

        cols = np.array(list(self.df.columns))
        feats = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]

        self.feature_choice_widget = CelldetectiveWidget()
        self.feature_choice_widget.setWindowTitle("Select numeric feature")
        layout = QVBoxLayout()
        self.feature_choice_widget.setLayout(layout)
        self.feature_cb = QComboBox()
        self.feature_cb.addItems(feats)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("feature: "), 33)
        hbox.addWidget(self.feature_cb, 66)
        layout.addLayout(hbox)

        self.set_feature_btn = QPushButton("set")
        self.set_feature_btn.clicked.connect(self.compute_signals)
        layout.addWidget(self.set_feature_btn)
        self.feature_choice_widget.show()
        center_window(self.feature_choice_widget)

    def ask_for_features(self):

        cols = np.array(list(self.df.columns))
        feats = [c for c in cols if pd.api.types.is_numeric_dtype(self.df[c])]

        self.feature_choice_widget = CelldetectiveWidget()
        self.feature_choice_widget.setWindowTitle("Select numeric feature")
        layout = QVBoxLayout()
        self.feature_choice_widget.setLayout(layout)
        self.feature_cb = QComboBox()
        self.feature_cb.addItems(feats)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("feature: "), 33)
        hbox.addWidget(self.feature_cb, 66)
        # hbox.addWidget((QLabel('Plot two features')))
        layout.addLayout(hbox)
        self.checkBox_feature = QCheckBox(self.feature_choice_widget)
        self.checkBox_feature.setGeometry(QRect(170, 120, 81, 20))
        # self.checkBox_feature.setText('Plot two features')
        # #hbox.addWidget(self.checkBox_feature)
        # layout.addWidget(self.checkBox_feature, alignment=Qt.AlignCenter)
        # self.checkBox_feature.stateChanged.connect(self.enable_second_feature)

        # layout = QVBoxLayout()
        # self.feature_two_choice_widget.setLayout(layout)
        # self.feature_two_cb = QComboBox()
        # self.feature_two_cb.addItems(feats)
        # self.feature_two_cb.setEnabled(False)
        # hbox_two = QHBoxLayout()
        # hbox_two.addWidget(QLabel('feature two: '), 33)
        # hbox_two.addWidget(self.feature_two_cb, 66)
        # layout.addLayout(hbox_two)

        self.set_feature_btn = QPushButton("set")
        self.set_feature_btn.clicked.connect(self.compute_signals)
        layout.addWidget(self.set_feature_btn)
        self.feature_choice_widget.show()
        center_window(self.feature_choice_widget)

    def enable_second_feature(self):
        if self.checkBox_feature.isChecked():
            self.feature_two_cb.setEnabled(True)
        else:
            self.feature_two_cb.setEnabled(False)

    def compute_signals(self):

        if self.df is not None:
            self.feature_selected = self.feature_cb.currentText()
            self.feature_choice_widget.close()
            self.compute_signal_functions()
            # prepare survival

            # plot survival
            self.survivalWidget = CelldetectiveWidget()
            self.scroll = QScrollArea()
            self.survivalWidget.setMinimumHeight(int(0.8 * self.screen_height))
            self.survivalWidget.setWindowTitle("signals")
            self.plotvbox = QVBoxLayout(self.survivalWidget)
            self.plotvbox.setContentsMargins(30, 30, 30, 30)
            self.survival_title = QLabel("Signal function")
            self.survival_title.setStyleSheet(
                """
				font-weight: bold;
				padding: 0px;
				"""
            )
            self.plotvbox.addWidget(self.survival_title, alignment=Qt.AlignCenter)

            plot_buttons_hbox = QHBoxLayout()
            plot_buttons_hbox.addWidget(QLabel(""), 80, alignment=Qt.AlignLeft)

            self.legend_btn = QPushButton("")
            self.legend_btn.setIcon(icon(MDI6.text_box, color="black"))
            self.legend_btn.setStyleSheet(self.button_select_all)
            self.legend_btn.setToolTip("Show or hide the legend")
            self.legend_visible = True
            self.legend_btn.clicked.connect(self.show_hide_legend)
            plot_buttons_hbox.addWidget(self.legend_btn, 5, alignment=Qt.AlignRight)

            self.log_btn = QPushButton("")
            self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
            self.log_btn.setStyleSheet(self.button_select_all)
            self.log_btn.clicked.connect(self.switch_to_log)
            self.log_btn.setToolTip("Enable or disable log scale")
            plot_buttons_hbox.addWidget(self.log_btn, 5, alignment=Qt.AlignRight)

            # self.ci_btn = QPushButton('')
            # self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal,color="blue"))
            # self.ci_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
            # self.ci_btn.clicked.connect(self.switch_ci)
            # self.ci_btn.setToolTip('Show or hide confidence intervals.')
            # plot_buttons_hbox.addWidget(self.ci_btn, 5, alignment=Qt.AlignRight)

            self.cell_lines_btn = QPushButton("")
            self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color="black"))
            self.cell_lines_btn.setStyleSheet(self.button_select_all)
            self.cell_lines_btn.clicked.connect(self.switch_cell_lines)
            self.cell_lines_btn.setToolTip("Show or hide individual cell signals.")
            plot_buttons_hbox.addWidget(self.cell_lines_btn, 5, alignment=Qt.AlignRight)

            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))  # ,figsize=(10,10)
            # self.setMinimumHeight(100)
            self.survival_window = FigureCanvas(self.fig, title="Survival")
            self.survival_window.setContentsMargins(0, 0, 0, 0)
            if self.df is not None:
                self.initialize_axis()
            plt.tight_layout()

            self.fig.set_facecolor("none")  # or 'None'
            self.fig.canvas.setStyleSheet("background-color: transparent;")
            self.survival_window.canvas.draw()

            # self.survival_window.layout.addWidget(QLabel('WHAAAAATTT???'))

            self.plot_options = [QRadioButton() for i in range(3)]
            self.radio_labels = ["well", "pos", "both"]
            radio_hbox = QHBoxLayout()
            radio_hbox.setContentsMargins(30, 30, 30, 30)
            self.plot_btn_group = QButtonGroup()
            for i in range(3):
                self.plot_options[i].setText(self.radio_labels[i])
                # self.plot_options[i].toggled.connect(self.plot_survivals)
                self.plot_btn_group.addButton(self.plot_options[i])
                radio_hbox.addWidget(self.plot_options[i], 33, alignment=Qt.AlignCenter)
            self.plot_btn_group.buttonClicked[int].connect(self.plot_survivals)

            if self.position_indices is not None:
                if len(self.well_indices) > 1 and len(self.position_indices) == 1:
                    self.plot_btn_group.buttons()[0].click()
                    for i in [1, 2]:
                        self.plot_options[i].setEnabled(False)
                elif len(self.well_indices) > 1:
                    self.plot_btn_group.buttons()[0].click()
                elif len(self.well_indices) == 1 and len(self.position_indices) == 1:
                    self.plot_btn_group.buttons()[1].click()
                    for i in [0, 2]:
                        self.plot_options[i].setEnabled(False)
            # else:
            # 	if len(self.well_indices)>1:
            # 		self.plot_btn_group.buttons()[0].click()
            # 	elif len(self.well_indices)==1:
            # 		self.plot_btn_group.buttons()[2].click()

            # elif len(self.well_indices)>1:
            # 	self.plot_btn_group.buttons()[0].click()
            # else:
            # 	self.plot_btn_group.buttons()[1].click()

            if self.position_indices is not None:
                for i in [0, 2]:
                    self.plot_options[i].setEnabled(False)

            # self.plot_options[0].setChecked(True)
            # self.plotvbox.addLayout(radio_hbox)
            self.plotvbox.addLayout(plot_buttons_hbox)
            self.plotvbox.addWidget(self.survival_window)
            # self.class_selection_lbl = QLabel('Select class')
            # self.class_selection_lbl.setStyleSheet("""
            # 	font-weight: bold;
            # 	padding: 0px;
            # 	""")
            # self.plotvbox.addWidget(self.class_selection_lbl, alignment=Qt.AlignCenter)
            # class_selection_hbox = QHBoxLayout()
            # class_selection_hbox.setContentsMargins(30,30,30,30)
            # self.all_btn = QRadioButton('*')
            # self.all_btn.setChecked(True)
            # self.event_btn = QRadioButton('event')
            # self.no_event_btn = QRadioButton('no event')
            # self.class_btn_group = QButtonGroup()
            # for btn in [self.all_btn, self.event_btn, self.no_event_btn]:
            # 	self.class_btn_group.addButton(btn)
            #
            # self.class_btn_group.buttonClicked[int].connect(self.set_class_to_plot)
            #
            # class_selection_hbox.addWidget(self.all_btn, 33, alignment=Qt.AlignLeft)
            # class_selection_hbox.addWidget(self.event_btn, 33, alignment=Qt.AlignCenter)
            # class_selection_hbox.addWidget(self.no_event_btn, 33, alignment=Qt.AlignRight)
            # self.plotvbox.addLayout(class_selection_hbox)

            self.select_pos_label = QLabel("Select positions")
            self.select_pos_label.setStyleSheet(
                """
				font-weight: bold;
				padding: 0px;
				"""
            )
            self.plotvbox.addWidget(self.select_pos_label, alignment=Qt.AlignCenter)
            #
            # self.select_option = [QRadioButton() for i in range(2)]
            # self.select_label = ['name', 'spatial']
            # select_hbox = QHBoxLayout()
            # select_hbox.setContentsMargins(30,30,30,30)
            # self.select_btn_group = QButtonGroup()
            # for i in range(2):
            # 	self.select_option[i].setText(self.select_label[i])
            # 	#self.select_option[i].toggled.connect(self.switch_selection_mode)
            # 	self.select_btn_group.addButton(self.select_option[i])
            # 	select_hbox.addWidget(self.select_option[i],33, alignment=Qt.AlignCenter)
            # self.select_btn_group.buttonClicked[int].connect(self.switch_selection_mode)
            # self.plotvbox.addLayout(select_hbox)

            self.look_for_metadata()
            if self.metadata_found:
                self.fig_scatter, self.ax_scatter = plt.subplots(1, 1, figsize=(4, 3))
                self.position_scatter = FigureCanvas(self.fig_scatter)
                self.load_coordinates()
                self.plot_spatial_location()
                # self.plot_positions()
                self.ax_scatter.spines["top"].set_visible(False)
                self.ax_scatter.spines["right"].set_visible(False)
                self.ax_scatter.set_aspect("equal")
                self.ax_scatter.set_xticks([])
                self.ax_scatter.set_yticks([])
                plt.tight_layout()

                self.fig_scatter.set_facecolor("none")  # or 'None'
                self.fig_scatter.canvas.setStyleSheet("background-color: transparent;")
                self.plotvbox.addWidget(self.position_scatter)

            self.generate_pos_selection_widget()

            # if self.df is not None and len(self.ks_estimators_per_position)>0:
            # 	self.plot_survivals()
            # self.select_btn_group.buttons()[0].click()
            self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # self.scroll.setWidgetResizable(True)
            self.scroll.setWidget(self.survivalWidget)

            self.scroll.setMinimumHeight(int(0.8 * self.screen_height))
            self.survivalWidget.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            self.scroll.show()

    def process_signal(self):

        print("you clicked!!")
        # self.FrameToMin = float(self.time_calibration_le.text().replace(',','.'))
        print(self.FrameToMin, "set")

        # read instructions from combobox options
        self.load_available_tables()
        if self.df is not None:
            self.ask_for_features()
        else:
            return None

    def generate_pos_selection_widget(self):

        self.well_names = self.df["well_name"].unique()
        self.pos_names = self.df_pos_info[
            "pos_name"
        ].unique()  # pd.DataFrame(self.ks_estimators_per_position)['position_name'].unique()
        print(f"POSITION NAMES: ", self.pos_names)
        self.usable_well_labels = []
        for name in self.well_names:
            for lbl in self.well_labels:
                if name + ":" in lbl:
                    self.usable_well_labels.append(lbl)

        self.line_choice_widget = CelldetectiveWidget()
        self.line_check_vbox = QVBoxLayout()
        self.line_choice_widget.setLayout(self.line_check_vbox)
        if len(self.well_indices) > 1:
            self.well_display_options = [
                QCheckBox(self.usable_well_labels[i])
                for i in range(len(self.usable_well_labels))
            ]
            for i in range(len(self.well_names)):
                self.line_check_vbox.addWidget(
                    self.well_display_options[i], alignment=Qt.AlignLeft
                )
                self.well_display_options[i].setChecked(True)
                self.well_display_options[i].toggled.connect(self.select_survival_lines)
        else:
            self.pos_display_options = [
                QCheckBox(self.pos_names[i]) for i in range(len(self.pos_names))
            ]
            for i in range(len(self.pos_names)):
                self.line_check_vbox.addWidget(
                    self.pos_display_options[i], alignment=Qt.AlignLeft
                )
                self.pos_display_options[i].setChecked(True)
                self.pos_display_options[i].toggled.connect(self.select_survival_lines)

        self.plotvbox.addWidget(self.line_choice_widget, alignment=Qt.AlignCenter)

    def load_available_tables(self):
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
            population=self.cbs[0].currentText(),
            return_pos_info=True,
        )

        if self.df is None:

            print("No table could be found...")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No table could be found to compute survival...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                self.close()
                return None
            else:
                self.close()
                return None
        else:
            self.df_well_info = self.df_pos_info.loc[
                :, ["well_path", "well_index", "well_name", "well_number", "well_alias"]
            ].drop_duplicates()

    def compute_signal_functions(self):
        wells = get_experiment_wells(self.exp_dir)
        antibodies = get_experiment_antibodies(self.exp_dir)
        cell_types = get_experiment_cell_types(self.exp_dir)
        concentrations = get_experiment_concentrations(self.exp_dir)
        well_name = [
            " ".join([s1, "+", s2, s3, "pM"])
            for s1, s2, s3 in zip(cell_types, antibodies, concentrations)
        ]

        data = []
        full_data = []
        print(self.well_option)
        if self.well_option > 1:
            self.plot_mode = "wells"
            for z, well in enumerate(wells):  # loop over wells
                # print(well)
                if z not in self.well_indices:
                    pass
                else:
                    positions = get_positions_in_well(well)
                    for ind, pos in enumerate(positions):  # loop over positions
                        if self.position_indices is not None:
                            if ind + 1 in self.position_indices:
                                print(f"Processing position {pos}...")
                                tab_tc = pos + os.sep.join(
                                    [
                                        "output",
                                        "tables",
                                        f"trajectories_{self.cbs[0].currentText()}.csv",
                                    ]
                                )
                                if not os.path.exists(tab_tc):
                                    return None
                                else:
                                    data = pd.read_csv(tab_tc)
                                    data["well_name"] = well_name[z]
                                full_data.append(data)

                        else:
                            print(f"Processing position {pos}...")
                            tab_tc = pos + os.sep.join(
                                [
                                    "output",
                                    "tables",
                                    f"trajectories_{self.cbs[0].currentText()}.csv",
                                ]
                            )
                            if not os.path.exists(tab_tc):
                                return None
                            else:
                                data = pd.read_csv(tab_tc)
                                data["well_name"] = well_name[z]
                            full_data.append(data)
        else:
            self.plot_mode = "positions"
            positions = get_positions_in_well(wells[self.well_indices[0]])
            for ind, pos in enumerate(positions):  # loop over positions
                pos_name = pos.split(os.sep)[-2]
                if self.position_indices is not None:
                    if ind + 1 in self.position_indices:
                        print(f"Processing position {pos}...")
                        tab_tc = pos + os.sep.join(
                            [
                                "output",
                                "tables",
                                f"trajectories_{self.cbs[0].currentText()}.csv",
                            ]
                        )
                        if not os.path.exists(tab_tc):
                            return None
                        else:
                            data = pd.read_csv(tab_tc)
                            data["position"] = pos_name
                        full_data.append(data)
                else:
                    print(f"Processing position {pos}...")
                    tab_tc = pos + os.sep.join(
                        [
                            "output",
                            "tables",
                            f"trajectories_{self.cbs[0].currentText()}.csv",
                        ]
                    )
                    if not os.path.exists(tab_tc):
                        return None
                    else:
                        data = pd.read_csv(tab_tc)
                        data["position"] = pos_name
                    full_data.append(data)
        self.plot_data = pd.concat(full_data, ignore_index=True)

    def generate_synchronized_matrix(
        self, well_group, feature_selected, cclass, max_time
    ):

        if isinstance(cclass, int):
            cclass = [cclass]

        n_cells = len(well_group.groupby(["position", "TRACK_ID"]))
        depth = int(2 * max_time + 3)
        matrix = np.zeros((n_cells, depth))
        matrix[:, :] = np.nan
        mapping = np.arange(-max_time - 1, max_time + 2)
        cid = 0
        for block, movie_group in well_group.groupby("position"):
            for tid, track_group in movie_group.loc[
                movie_group[self.cbs[1].currentText()].isin(cclass)
            ].groupby("TRACK_ID"):
                try:
                    timeline = track_group["FRAME"].to_numpy().astype(int)
                    feature = track_group[feature_selected].to_numpy()
                    if self.checkBox_feature.isChecked():
                        second_feature = track_group[
                            self.second_feature_selected
                        ].to_numpy()
                    if (
                        self.cbs[2].currentText().startswith("t")
                        and not self.abs_time_checkbox.isChecked()
                    ):
                        t0 = math.floor(
                            track_group[self.cbs[2].currentText()].to_numpy()[0]
                        )
                        timeline -= t0
                    elif (
                        self.cbs[2].currentText() == "first detection"
                        and not self.abs_time_checkbox.isChecked()
                    ):

                        if "area" in list(track_group.columns):
                            print("area in list")
                            feat = track_group["area"].values
                        else:
                            feat = feature

                        first_detection = timeline[feat == feat][0]
                        timeline -= first_detection
                        print(first_detection, timeline)

                    elif self.abs_time_checkbox.isChecked():
                        timeline -= int(self.frame_slider.value())

                    loc_t = [np.where(mapping == t)[0][0] for t in timeline]
                    matrix[cid, loc_t] = feature
                    if second_feature:
                        matrix[cid, loc_t + 1] = second_feature
                    print(timeline, loc_t)

                    cid += 1
                except:
                    pass
        return matrix

    def col_mean(self, matrix):

        mean_line = np.zeros(matrix.shape[1])
        mean_line[:] = np.nan
        std_line = np.copy(mean_line)

        for k in range(matrix.shape[1]):
            values = matrix[:, k]
            # values = values[values!=0]
            if len(values[values == values]) > 2:
                mean_line[k] = np.nanmean(values)
                std_line[k] = np.nanstd(values)

        return mean_line, std_line

    def initialize_axis(self):
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["font.family"] = "sans-serif"
        SMALL_SIZE = 5
        MEDIUM_SIZE = 6
        BIGGER_SIZE = 7

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if self.well_option > 1:
            if self.check_class.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    hue=self.cbs[1].currentText(),
                    dodge=True,
                    palette=self.palette,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        hue=self.cbs[1].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.palette,
                    )
            elif self.check_group.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    hue=self.cbs[2].currentText(),
                    dodge=True,
                    palette=self.hue_colors,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        hue=self.cbs[2].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.hue_colors,
                    )
            else:
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    color=self.palette[0],
                    dodge=True,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        color=self.palette[0],
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                    )
        else:
            if self.check_class.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    hue=self.cbs[1].currentText(),
                    dodge=True,
                    palette=self.palette,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        hue=self.cbs[1].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.palette,
                    )
            elif self.check_group.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    hue=self.cbs[2].currentText(),
                    dodge=True,
                    palette=self.hue_colors,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        hue=self.cbs[2].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.hue_colors,
                    )
            else:
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    color=self.palette[0],
                    dodge=True,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        color=self.palette[0],
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                    )
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.set_ylabel(self.feature_selected)
        # ax.set_ylim(0.95,1.3)
        plt.tight_layout()

    def plot_survivals(self, id):
        self.ax.clear()
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["font.family"] = "sans-serif"
        SMALL_SIZE = 5
        MEDIUM_SIZE = 6
        BIGGER_SIZE = 7

        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if self.well_option > 1:
            if self.check_class.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    hue=self.cbs[1].currentText(),
                    dodge=True,
                    palette=self.palette,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        hue=self.cbs[1].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.palette,
                    )
            elif self.check_group.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    hue=self.cbs[2].currentText(),
                    dodge=True,
                    palette=self.hue_colors,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        hue=self.cbs[2].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.hue_colors,
                    )
            else:
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="well_name",
                    color=self.palette[0],
                    dodge=True,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="well_name",
                        color=self.palette[0],
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                    )
        else:
            if self.check_class.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    hue=self.cbs[1].currentText(),
                    dodge=True,
                    palette=self.palette,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        hue=self.cbs[1].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.palette,
                    )
            elif self.check_group.isChecked():
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    hue=self.cbs[2].currentText(),
                    dodge=True,
                    palette=self.hue_colors,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        hue=self.cbs[2].currentText(),
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                        palette=self.hue_colors,
                    )
            else:
                sns.boxenplot(
                    ax=self.ax,
                    data=self.plot_data,
                    y=self.feature_selected,
                    x="position",
                    color=self.palette[0],
                    dodge=True,
                )
                if self.show_cell_lines:
                    sns.stripplot(
                        ax=self.ax,
                        data=self.plot_data,
                        y=self.feature_selected,
                        x="position",
                        color=self.palette[0],
                        dodge=True,
                        alpha=0.3,
                        linewidth=0.5,
                        size=3,
                    )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel(self.feature_selected)
        plt.tight_layout()
        self.survival_window.setMinimumHeight(int(0.5 * self.screen_height))
        self.survival_window.setMinimumWidth(int(0.8 * self.survivalWidget.width()))
        self.survival_window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.survival_window.canvas.draw()

    def plot_line(
        self,
        line,
        color,
        label,
        mean_signal,
        ci_option=True,
        cell_lines_option=False,
        alpha_ci=0.5,
        alpha_cell_lines=0.5,
        std_signal=None,
        matrix=None,
    ):
        try:
            if "second" in str(mean_signal):
                self.ax2.plot(
                    line["timeline"] * self.FrameToMin,
                    line[mean_signal],
                    color=color,
                    label=label,
                )
            else:
                self.ax.plot(
                    line["timeline"] * self.FrameToMin,
                    line[mean_signal],
                    color=color,
                    label=label,
                )
            if ci_option and std_signal is not None:
                if "second" in str(mean_signal):
                    self.ax2.fill_between(
                        line["timeline"] * self.FrameToMin,
                        [a - b for a, b in zip(line[mean_signal], line[std_signal])],
                        [a + b for a, b in zip(line[mean_signal], line[std_signal])],
                        color=color,
                        alpha=alpha_ci,
                    )
                else:
                    self.ax.fill_between(
                        line["timeline"] * self.FrameToMin,
                        [a - b for a, b in zip(line[mean_signal], line[std_signal])],
                        [a + b for a, b in zip(line[mean_signal], line[std_signal])],
                        color=color,
                        alpha=alpha_ci,
                    )
            if cell_lines_option and matrix is not None:
                print(mean_signal)
                mat = line[matrix]
                if "second" in str(mean_signal):
                    for i in range(mat.shape[0]):
                        self.ax2.plot(
                            line["timeline"] * self.FrameToMin,
                            mat[i, :],
                            color=color,
                            alpha=alpha_cell_lines,
                        )
                else:
                    for i in range(mat.shape[0]):
                        self.ax.plot(
                            line["timeline"] * self.FrameToMin,
                            mat[i, :],
                            color=color,
                            alpha=alpha_cell_lines,
                        )

        except Exception as e:
            print(f"Exception {e}")

    def switch_to_log(self):
        """
        Switch threshold histogram to log scale. Auto adjust.
        """

        if self.ax.get_yscale() == "linear":
            self.ax.set_yscale("log")
        # self.ax.set_ylim(0.01,1.05)
        else:
            self.ax.set_yscale("linear")
        # self.ax.set_ylim(0.01,1.05)

        # self.ax.autoscale()
        self.survival_window.canvas.draw_idle()

    def show_hide_legend(self):
        if self.legend_visible:
            self.ax.legend().set_visible(False)
            self.legend_visible = False
            self.legend_btn.setIcon(icon(MDI6.text_box_outline, color="black"))
        else:
            self.ax.legend().set_visible(True)
            self.legend_visible = True
            self.legend_btn.setIcon(icon(MDI6.text_box, color="black"))

        self.survival_window.canvas.draw_idle()

    def look_for_metadata(self):

        self.metadata_found = False
        self.metafiles = (
            glob(self.exp_dir + os.sep.join([f"W*", "*", "movie", "*metadata.txt"]))
            + glob(self.exp_dir + os.sep.join([f"W*", "*", "*metadata.txt"]))
            + glob(self.exp_dir + os.sep.join([f"W*", "*metadata.txt"]))
            + glob(self.exp_dir + "*metadata.txt")
        )
        print(f"Found {len(self.metafiles)} metadata files...")
        if len(self.metafiles) > 0:
            self.metadata_found = True

    def switch_selection_mode(self, id):
        print(f"button {id} was clicked")
        for i in range(2):
            if self.select_option[i].isChecked():
                self.selection_mode = self.select_label[i]
        if self.selection_mode == "name":
            if len(self.metafiles) > 0:
                self.position_scatter.hide()
            self.line_choice_widget.show()
        else:
            if len(self.metafiles) > 0:
                self.position_scatter.show()
            self.line_choice_widget.hide()

    def load_coordinates(self):
        """
        Read metadata and try to extract position coordinates
        """

        self.no_meta = False
        try:
            with open(self.metafiles[0], "r") as f:
                data = json.load(f)
                positions = data["Summary"]["InitialPositionList"]
        except Exception as e:
            print(f"Trouble loading metadata: error {e}...")
            return None

        for k in range(len(positions)):
            pos_label = positions[k]["Label"]
            try:
                coords = positions[k]["DeviceCoordinatesUm"]["XYStage"]
            except:
                try:
                    coords = positions[k]["DeviceCoordinatesUm"]["PIXYStage"]
                except:
                    self.no_meta = True

            if not self.no_meta:
                files = self.df_pos_info["stack_path"].values
                pos_loc = [pos_label in f for f in files]
                self.df_pos_info.loc[pos_loc, "x"] = coords[0]
                self.df_pos_info.loc[pos_loc, "y"] = coords[1]
                self.df_pos_info.loc[pos_loc, "metadata_tag"] = pos_label

    def update_annot(self, ind):

        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = self.scat_labels[ind["ind"][0]]
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor("k")
        self.annot.get_bbox_patch().set_alpha(0.4)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax_scatter:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig_scatter.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig_scatter.canvas.draw_idle()

    def unselect_position(self, event):
        print("unselecting position")
        self.survival_window.canvas.clear()
        ind = event.ind  # index of selected position
        well_idx = self.df_pos_info.iloc[ind]["well_index"].values[0]
        selectedPos = self.df_pos_info.iloc[ind]["pos_path"].values[0]
        currentSelState = self.df_pos_info.iloc[ind]["select"].values[0]
        if self.plot_options[0].isChecked() or self.plot_options[2].isChecked():
            self.df_pos_info.loc[
                self.df_pos_info["well_index"] == well_idx, "select"
            ] = not currentSelState
            self.df_well_info.loc[
                self.df_well_info["well_index"] == well_idx, "select"
            ] = not currentSelState
            if len(self.well_indices) > 1:
                self.well_display_options[well_idx].setChecked(not currentSelState)
            else:
                for p in self.pos_display_options:
                    p.setChecked(not currentSelState)
        else:
            self.df_pos_info.loc[
                self.df_pos_info["pos_path"] == selectedPos, "select"
            ] = not currentSelState
            if len(self.well_indices) <= 1:
                self.pos_display_options[ind[0]].setChecked(not currentSelState)

        self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
        self.position_scatter.canvas.draw_idle()
        self.plot_survivals(0)

    def select_survival_lines(self):
        if self.plot_mode == "wells":
            selected_wells = []
            for i in range(len(self.well_display_options)):
                if self.well_display_options[i].isChecked():
                    selected_wells.append(i)
                self.df_well_info.loc[
                    self.df_well_info["well_index"] == i, "select"
                ] = self.well_display_options[i].isChecked()
                self.df_pos_info.loc[self.df_pos_info["well_index"] == i, "select"] = (
                    self.well_display_options[i].isChecked()
                )
            if len(selected_wells) == 0:
                print("No wells selected")
                self.ax.clear()
            else:
                self.ax.clear()
                self.well_indices = selected_wells
                self.compute_signal_functions()
                self.plot_survivals(0)
        else:
            selected_pos = []
            for i in range(len(self.pos_display_options)):
                if self.pos_display_options[i].isChecked():
                    selected_pos.append(i + 1)
                self.df_pos_info.loc[self.df_pos_info["pos_index"] == i, "select"] = (
                    self.pos_display_options[i].isChecked()
                )
            if len(selected_pos) == 0:
                self.ax.clear()
            else:
                self.position_indices = selected_pos
                self.ax.clear()
                self.compute_signal_functions()
                self.plot_survivals(0)

        if len(self.metafiles) > 0:
            self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
            self.position_scatter.canvas.draw_idle()

    def select_color(self, selection):
        colors = [tab10(0) if s else tab10(0.1) for s in selection]
        return colors

    def plot_spatial_location(self):

        try:
            self.sc = self.ax_scatter.scatter(
                self.df_pos_info["x"].values,
                self.df_pos_info["y"].values,
                picker=True,
                pickradius=1,
                color=self.select_color(self.df_pos_info["select"].values),
            )
            self.scat_labels = self.df_pos_info["metadata_tag"].values
            self.ax_scatter.invert_xaxis()
            self.annot = self.ax_scatter.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            self.annot.set_visible(False)
            self.fig_scatter.canvas.mpl_connect("motion_notify_event", self.hover)
            self.fig_scatter.canvas.mpl_connect("pick_event", self.unselect_position)
        except Exception as e:
            pass

    def switch_ref_time_mode(self):
        if self.abs_time_checkbox.isChecked():
            self.frame_slider.setEnabled(True)
            self.cbs[-1].setEnabled(False)
        else:
            self.frame_slider.setEnabled(False)
            self.cbs[-1].setEnabled(True)

    def switch_ci(self):

        if self.show_ci:
            self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal, color="black"))
        else:
            self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal, color="blue"))
        self.show_ci = not self.show_ci
        self.plot_survivals(0)

    def switch_cell_lines(self):

        if self.show_cell_lines:
            self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color="black"))
        else:
            self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color="blue"))
        self.show_cell_lines = not self.show_cell_lines
        self.plot_survivals(0)

    def set_class_to_plot(self):

        if self.all_btn.isChecked():
            self.target_class = [0, 1]
        elif self.event_btn.isChecked():
            self.target_class = [0]
        else:
            self.target_class = [1]

        self.plot_survivals(0)
