import json
import os
from glob import glob

import numpy as np
from PyQt5.QtCore import QSize, QTimer, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from fonticon_mdi6 import MDI6
from natsort import natsorted
from superqt.fonticon import icon

import celldetective.gui.preprocessing_block
from celldetective import get_software_location
from celldetective.gui.base.components import QHSeperationLine, HoverButton

from celldetective.gui.base.styles import Styles
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import help_generic
from celldetective.utils.data_loaders import load_experiment_tables
from celldetective.utils.experiment import extract_position_name
from celldetective.utils.model_getters import get_pair_signal_models_list
from celldetective import get_logger

logger = get_logger(__name__)


class NeighPanel(QFrame, Styles):
    def __init__(self, parent_window):

        super().__init__()
        self.parent_window = parent_window
        self.exp_channels = self.parent_window.exp_channels
        self.exp_dir = self.parent_window.exp_dir
        self.wells = np.array(self.parent_window.wells, dtype=str)
        self.protocols = []
        self.mode = "neighborhood"

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.grid = QGridLayout(self)
        self.generate_header()

    def generate_header(self):
        """
        Read the mode and prepare a collapsable block to process a specific cell population.

        """

        panel_title = QLabel(f"INTERACTIONS")
        panel_title.setStyleSheet(self.block_title)

        self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

        # self.select_all_btn = QPushButton()
        # self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
        # self.select_all_btn.setIconSize(QSize(20, 20))
        # self.all_ticked = False
        # self.select_all_btn.setStyleSheet(self.button_select_all)
        # self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)

        self.collapse_btn = QPushButton()
        self.collapse_btn.setIcon(icon(MDI6.chevron_down, color="black"))
        self.collapse_btn.setIconSize(QSize(25, 25))
        self.collapse_btn.setStyleSheet(self.button_select_all)
        self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

        self.populate_contents()

        self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
        self.collapse_btn.clicked.connect(
            lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden())
        )
        self.collapse_btn.clicked.connect(self.collapse_advanced)
        self.ContentsFrame.hide()

    def collapse_advanced(self):

        panels_open = [
            not p.ContentsFrame.isHidden()
            for p in self.parent_window.ProcessPopulations
        ]
        interactions_open = not self.parent_window.NeighPanel.ContentsFrame.isHidden()
        preprocessing_open = (
            not self.parent_window.PreprocessingPanel.ContentsFrame.isHidden()
        )
        is_open = np.array(panels_open + [interactions_open, preprocessing_open])

        if self.ContentsFrame.isHidden():
            self.collapse_btn.setIcon(icon(MDI6.chevron_down, color="black"))
            self.collapse_btn.setIconSize(QSize(20, 20))
            if len(is_open[is_open]) == 0:
                self.parent_window.scroll.setMinimumHeight(int(550))
                self.parent_window.adjustSize()
        else:
            self.collapse_btn.setIcon(icon(MDI6.chevron_up, color="black"))
            self.collapse_btn.setIconSize(QSize(20, 20))
            self.parent_window.scroll.setMinimumHeight(
                min(int(1000), int(0.9 * self.parent_window.screen_height))
            )
            try:
                QTimer.singleShot(10, lambda: center_window(self.window()))
            except:
                pass

    def populate_contents(self):

        self.ContentsFrame = QFrame()
        self.grid_contents = QGridLayout(self.ContentsFrame)
        self.grid_contents.setContentsMargins(0, 0, 0, 0)
        self.grid_contents.setSpacing(3)

        # Button to compute the neighborhoods
        neigh_option_hbox = QHBoxLayout()
        self.neigh_action = QCheckBox("NEIGHBORHOODS")
        self.neigh_action.setStyleSheet(self.menu_check_style)

        # self.neigh_action.setIcon(icon(MDI6.eyedropper, color="black"))
        # self.neigh_action.setIconSize(QSize(20, 20))
        self.neigh_action.setToolTip("Compute neighborhoods in list below.")

        neigh_option_hbox.addWidget(self.neigh_action, 90)

        self.help_neigh_btn = QPushButton()
        self.help_neigh_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        self.help_neigh_btn.setIconSize(QSize(20, 20))
        self.help_neigh_btn.clicked.connect(self.help_neighborhood)
        self.help_neigh_btn.setStyleSheet(self.button_select_all)
        self.help_neigh_btn.setToolTip("Help.")
        neigh_option_hbox.addWidget(self.help_neigh_btn, 5, alignment=Qt.AlignRight)

        self.grid_contents.addLayout(neigh_option_hbox, 1, 0, 1, 4)

        neigh_options_layout = QVBoxLayout()

        neigh_options_vbox = QVBoxLayout()

        # DISTANCE NEIGHBORHOOD
        dist_neigh_hbox = QHBoxLayout()
        dist_neigh_hbox.setContentsMargins(0, 0, 0, 0)
        dist_neigh_hbox.setSpacing(0)

        self.dist_neigh_action = QLabel("ISOTROPIC DISTANCE THRESHOLD")
        self.dist_neigh_action.setStyleSheet(self.action_lbl_style_sheet)
        # self.dist_neigh_action.setIcon(icon(MDI6.circle_expand, color='black'))
        self.dist_neigh_action.setToolTip("")
        self.dist_neigh_action.setToolTip(
            "Define an isotropic neighborhood between the center of mass\nof the cells, within a threshold distance."
        )
        # self.segment_action.toggled.connect(self.enable_segmentation_model_list)
        # self.to_disable.append(self.segment_action)

        self.config_distance_neigh_btn = QPushButton()
        self.config_distance_neigh_btn.setIcon(icon(MDI6.plus, color="black"))
        self.config_distance_neigh_btn.setIconSize(QSize(20, 20))
        self.config_distance_neigh_btn.setToolTip("Configure.")
        self.config_distance_neigh_btn.setStyleSheet(self.button_select_all)
        self.config_distance_neigh_btn.clicked.connect(
            self.open_config_distance_threshold_neighborhood
        )
        dist_neigh_hbox.addWidget(self.config_distance_neigh_btn, 5)
        dist_neigh_hbox.addWidget(self.dist_neigh_action, 95)
        neigh_options_vbox.addLayout(dist_neigh_hbox)

        # CONTACT NEIGHBORHOOD
        contact_neighborhood_layout = QHBoxLayout()
        contact_neighborhood_layout.setContentsMargins(0, 0, 0, 0)
        contact_neighborhood_layout.setSpacing(0)

        self.contact_neigh_action = QLabel("MASK CONTACT")
        self.contact_neigh_action.setToolTip(
            "Identify touching cell masks, within a threshold edge distance."
        )
        self.contact_neigh_action.setStyleSheet(self.action_lbl_style_sheet)
        # self.contact_neigh_action.setIcon(icon(MDI6.transition_masked, color='black'))
        self.contact_neigh_action.setToolTip("")

        self.config_contact_neigh_btn = QPushButton()
        self.config_contact_neigh_btn.setIcon(icon(MDI6.plus, color="black"))
        self.config_contact_neigh_btn.setIconSize(QSize(20, 20))
        self.config_contact_neigh_btn.setToolTip("Configure.")
        self.config_contact_neigh_btn.setStyleSheet(self.button_select_all)
        self.config_contact_neigh_btn.clicked.connect(
            self.open_config_contact_neighborhood
        )
        contact_neighborhood_layout.addWidget(self.config_contact_neigh_btn, 5)
        contact_neighborhood_layout.addWidget(self.contact_neigh_action, 95)
        neigh_options_vbox.addLayout(contact_neighborhood_layout)
        # self.grid_contents.addLayout(neigh_options_vbox, 2,0,1,4)

        # self.grid_contents.addWidget(QHSeperationLine(), 3, 0, 1, 4)

        self.delete_protocol_btn = QPushButton("")
        self.delete_protocol_btn.setStyleSheet(self.button_select_all)
        self.delete_protocol_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_protocol_btn.setToolTip("Remove a neighborhood computation.")
        self.delete_protocol_btn.setIconSize(QSize(20, 20))
        self.delete_protocol_btn.clicked.connect(self.remove_protocol_from_list)

        self.protocol_list_lbl = QLabel("Neighborhoods to compute: ")
        self.protocol_list = QListWidget()
        self.protocol_list.setToolTip("Neighborhoods to compute sequentially.")

        list_header_layout = QHBoxLayout()
        list_header_layout.addWidget(self.protocol_list_lbl)
        list_header_layout.addWidget(self.delete_protocol_btn, alignment=Qt.AlignRight)
        # self.grid_contents.addLayout(list_header_layout, 4, 0, 1, 4)
        # self.grid_contents.addWidget(self.protocol_list, 5, 0, 1, 4)

        neigh_options_layout.addLayout(neigh_options_vbox)
        neigh_options_layout.addWidget(QHSeperationLine())
        neigh_options_layout.addLayout(list_header_layout)
        neigh_options_layout.addWidget(self.protocol_list)

        neigh_options_layout.setContentsMargins(30, 5, 30, 5)
        neigh_options_layout.setSpacing(1)
        self.grid_contents.addLayout(neigh_options_layout, 5, 0, 1, 4)

        rel_layout = QHBoxLayout()
        self.measure_pairs_action = QCheckBox("MEASURE PAIRS")
        self.measure_pairs_action.setStyleSheet(self.menu_check_style)

        self.measure_pairs_action.setIcon(icon(MDI6.eyedropper, color="black"))
        self.measure_pairs_action.setIconSize(QSize(20, 20))
        self.measure_pairs_action.setToolTip(
            "Measure the relative quantities defined for the cell pairs, for all neighborhoods."
        )
        rel_layout.addWidget(self.measure_pairs_action, 90)

        self.classify_pairs_btn = QPushButton()
        self.classify_pairs_btn.setIcon(icon(MDI6.scatter_plot, color="black"))
        self.classify_pairs_btn.setIconSize(QSize(20, 20))
        self.classify_pairs_btn.setToolTip("Classify data.")
        self.classify_pairs_btn.setStyleSheet(self.button_select_all)
        self.classify_pairs_btn.clicked.connect(self.open_classifier_ui_pairs)
        rel_layout.addWidget(
            self.classify_pairs_btn, 5
        )  # 4,2,1,1, alignment=Qt.AlignRight

        self.grid_contents.addLayout(rel_layout, 6, 0, 1, 4)

        signal_layout = QVBoxLayout()
        signal_hlayout = QHBoxLayout()
        self.signal_analysis_action = QCheckBox("DETECT PAIR EVENTS")
        self.signal_analysis_action.setStyleSheet(self.menu_check_style)

        self.signal_analysis_action.setIcon(
            icon(MDI6.chart_bell_curve_cumulative, color="black")
        )
        self.signal_analysis_action.setIconSize(QSize(20, 20))
        self.signal_analysis_action.setToolTip(
            "Detect cell pair events using a DL model."
        )
        self.signal_analysis_action.toggled.connect(self.enable_signal_model_list)
        signal_hlayout.addWidget(self.signal_analysis_action, 90)

        self.check_signals_btn = QPushButton()
        self.check_signals_btn.setIcon(icon(MDI6.eye_check_outline, color="black"))
        self.check_signals_btn.setIconSize(QSize(20, 20))
        self.check_signals_btn.clicked.connect(self.check_signals2)
        self.check_signals_btn.setToolTip("Annotate dynamic cell pairs.")
        self.check_signals_btn.setStyleSheet(self.button_select_all)
        signal_hlayout.addWidget(self.check_signals_btn, 6)

        self.config_signal_annotator_btn = QPushButton()
        self.config_signal_annotator_btn.setIcon(icon(MDI6.cog_outline, color="black"))
        self.config_signal_annotator_btn.setIconSize(QSize(20, 20))
        self.config_signal_annotator_btn.setToolTip(
            "Configure the animation of the annotation tool."
        )
        self.config_signal_annotator_btn.setStyleSheet(self.button_select_all)
        self.config_signal_annotator_btn.clicked.connect(
            self.open_signal_annotator_configuration_ui
        )
        signal_hlayout.addWidget(self.config_signal_annotator_btn, 6)
        signal_layout.addLayout(signal_hlayout)
        # self.to_disable.append(self.measure_action_tc)
        pair_signal_model_vbox = QVBoxLayout()
        pair_signal_model_vbox.setContentsMargins(25, 0, 25, 0)

        pair_model_zoo_layout = QHBoxLayout()
        pair_model_zoo_layout.addWidget(QLabel("Model zoo:"), 90)

        self.pair_signal_models_list = QComboBox()
        self.pair_signal_models_list.setEnabled(False)
        self.refresh_signal_models()
        # self.to_disable.append(self.cell_models_list)

        self.pair_train_signal_model_btn = HoverButton(
            "TRAIN", MDI6.redo_variant, "black", "white"
        )
        self.pair_train_signal_model_btn.setToolTip(
            "Train a cell pair event detection model."
        )
        self.pair_train_signal_model_btn.setIconSize(QSize(20, 20))
        self.pair_train_signal_model_btn.setStyleSheet(self.button_style_sheet_3)
        pair_model_zoo_layout.addWidget(self.pair_train_signal_model_btn, 5)
        self.pair_train_signal_model_btn.clicked.connect(
            self.open_signal_model_config_ui
        )

        pair_signal_model_vbox.addLayout(pair_model_zoo_layout)
        pair_signal_model_vbox.addWidget(self.pair_signal_models_list)

        signal_layout.addLayout(pair_signal_model_vbox)
        self.grid_contents.addLayout(signal_layout, 7, 0, 1, 4)
        self.grid_contents.addWidget(QHSeperationLine(), 11, 0, 1, 4)

        self.view_tab_btn = QPushButton("Explore table")
        self.view_tab_btn.setStyleSheet(self.button_style_sheet_2)
        self.view_tab_btn.clicked.connect(self.view_table_ui)
        self.view_tab_btn.setToolTip("Explore table")
        self.view_tab_btn.setIcon(icon(MDI6.table, color="#1565c0"))
        self.view_tab_btn.setIconSize(QSize(20, 20))
        # self.view_tab_btn.setEnabled(False)
        self.grid_contents.addWidget(self.view_tab_btn, 12, 0, 1, 4)

        # self.grid_contents.addWidget(QLabel(''), 12, 0, 1, 4)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.setToolTip(
            "Compute the neighborhoods of the selected positions."
        )
        self.submit_btn.clicked.connect(self.process_neighborhood)
        self.grid_contents.addWidget(self.submit_btn, 14, 0, 1, 4)

        self.neigh_action.toggled.connect(self.activate_neigh_options)
        self.neigh_action.setChecked(True)
        self.neigh_action.setChecked(False)

    def open_classifier_ui_pairs(self):
        from celldetective.gui.classifier_widget import ClassifierWidget

        self.mode = "pairs"
        self.load_available_tables()
        if self.df is None:

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("No table was found...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
            else:
                return None
        else:
            self.ClassifierWidget = ClassifierWidget(self)
            self.ClassifierWidget.show()

    def help_neighborhood(self):
        """
        Helper for neighborhood strategy.
        """

        dict_path = os.sep.join(
            [
                get_software_location(),
                "celldetective",
                "gui",
                "help",
                "neighborhood.json",
            ]
        )

        with open(dict_path) as f:
            d = json.load(f)

        suggestion = help_generic(d)
        if isinstance(suggestion, str):
            logger.info(f"{suggestion=}")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setTextFormat(Qt.RichText)
            msgBox.setText(
                f"{suggestion}\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/interactions.html#neighborhood'>here</a>."
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

    def load_available_tables(self):
        """
        Load the tables of the selected wells/positions from the control Panel for the population of interest

        """

        self.well_option = self.parent_window.well_list.getSelectedIndices()
        self.position_option = self.parent_window.position_list.getSelectedIndices()

        self.df, self.df_pos_info = load_experiment_tables(
            self.exp_dir,
            well_option=self.well_option,
            position_option=self.position_option,
            population="pairs",
            return_pos_info=True,
        )
        if self.df is None:
            logger.info("No table could be found...")

    def view_table_ui(self):
        from celldetective.gui.tableUI import TableUI
        from celldetective.gui.workers import ProgressWindow
        from celldetective.processes.load_table import TableLoaderProcess

        logger.info("Load table...")

        # Prepare args for the process
        self.well_option = self.parent_window.well_list.getSelectedIndices()
        self.position_option = self.parent_window.position_list.getSelectedIndices()

        process_args = {
            "experiment": self.exp_dir,
            "population": "pairs",
            "well_option": self.well_option,
            "position_option": self.position_option,
            "show_frame_progress": False,
        }

        self.df = None

        def on_table_loaded(df):
            self.df = df
            if self.df is not None:
                plot_mode = "static"
                self.tab_ui = TableUI(
                    self.df,
                    f"{self.parent_window.well_list.currentText()}; Position {self.parent_window.position_list.currentText()}",
                    population="pairs",
                    plot_mode=plot_mode,
                    save_inplace_option=True,
                )
                self.tab_ui.show()
                center_window(self.tab_ui)
            else:
                logger.info("Table could not be loaded...")
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText("No table could be loaded...")
                msgBox.setWindowTitle("Info")
                msgBox.setStandardButtons(QMessageBox.Ok)
                returnValue = msgBox.exec()

        self.job = ProgressWindow(
            TableLoaderProcess,
            parent_window=self,
            title="Loading tables...",
            process_args=process_args,
            position_info=False,
            well_label="Wells loaded:",
            pos_label="Positions loaded:",
        )
        self.job._ProgressWindow__runner.signals.result.connect(on_table_loaded)

        result = self.job.exec_()

    def activate_neigh_options(self):

        if self.neigh_action.isChecked():
            self.dist_neigh_action.setEnabled(True)
            self.contact_neigh_action.setEnabled(True)
            self.config_distance_neigh_btn.setEnabled(True)
            self.config_contact_neigh_btn.setEnabled(True)
            self.protocol_list_lbl.setEnabled(True)
            self.protocol_list.setEnabled(True)
            self.delete_protocol_btn.setEnabled(True)
        else:
            self.dist_neigh_action.setEnabled(False)
            self.contact_neigh_action.setEnabled(False)
            self.config_distance_neigh_btn.setEnabled(False)
            self.config_contact_neigh_btn.setEnabled(False)
            self.protocol_list_lbl.setEnabled(False)
            self.protocol_list.setEnabled(False)
            self.delete_protocol_btn.setEnabled(False)

    def refresh_signal_models(self):
        signal_models = get_pair_signal_models_list()
        self.pair_signal_models_list.clear()
        self.pair_signal_models_list.addItems(signal_models)

    def open_signal_annotator_configuration_ui(self):
        from celldetective.gui.settings._settings_signal_annotator import (
            SettingsSignalAnnotator,
        )

        self.mode = "pairs"
        self.config_signal_annotator = SettingsSignalAnnotator(self)
        self.config_signal_annotator.show()

    def open_signal_model_config_ui(self):
        from celldetective.gui.settings._settings_event_model_training import (
            SettingsEventDetectionModelTraining,
        )

        self.settings_pair_event_detection_training = (
            SettingsEventDetectionModelTraining(self, signal_mode="pairs")
        )
        self.settings_pair_event_detection_training.show()

    def remove_protocol_from_list(self):

        current_item = self.protocol_list.currentRow()
        if current_item > -1:
            del self.protocols[current_item]
            self.protocol_list.takeItem(current_item)

    def open_config_distance_threshold_neighborhood(self):
        from celldetective.gui.settings._settings_neighborhood import (
            SettingsNeighborhood,
        )

        self.ConfigNeigh = SettingsNeighborhood(
            parent_window=self,
            neighborhood_type="distance_threshold",
            neighborhood_parameter_name="threshold distance",
        )
        self.ConfigNeigh.show()

    def open_config_contact_neighborhood(self):
        from celldetective.gui.settings._settings_neighborhood import (
            SettingsNeighborhood,
        )

        self.ConfigNeigh = SettingsNeighborhood(
            parent_window=self,
            neighborhood_type="mask_contact",
            neighborhood_parameter_name="tolerance contact distance",
        )
        self.ConfigNeigh.show()

    def enable_signal_model_list(self):
        if self.signal_analysis_action.isChecked():
            self.pair_signal_models_list.setEnabled(True)
        else:
            self.pair_signal_models_list.setEnabled(False)

    def process_neighborhood(self):
        from celldetective.gui.workers import ProgressWindow
        from celldetective.processes.compute_neighborhood import NeighborhoodProcess
        from celldetective.signals import analyze_pair_signals_at_position
        from celldetective.relative_measurements import rel_measure_at_position

        # if self.parent_window.well_list.currentText().startswith('Multiple'):
        # 	self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
        # else:
        self.well_index = self.parent_window.well_list.getSelectedIndices()
        logger.info(f"Processing well {self.parent_window.well_list.currentText()}...")

        # self.freeze()
        # QApplication.setOverrideCursor(Qt.WaitCursor)

        loop_iter = 0

        if self.parent_window.position_list.isMultipleSelection():
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText(
                "If you continue, all positions will be processed.\nDo you want to proceed?"
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.No:
                return None

        total_wells = len(self.well_index)
        for i, w_idx in enumerate(self.well_index):

            well_progress = round(i / total_wells * 100)

            pos = self.parent_window.positions[w_idx]
            pos_indices = self.parent_window.position_list.getSelectedIndices()

            well = self.parent_window.wells[w_idx]

            total_positions = len(pos_indices)

            # Optimization: Glob once per well - REMOVED as per user's instruction
            # all_well_positions = natsorted(
            #     glob(
            #         well
            #         + f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*{os.sep}"
            #     )
            # )

            for j, pos_idx in enumerate(pos_indices):

                pos_progress = round(j / total_positions * 100)

                # if pos_idx < len(all_well_positions): # REMOVED as per user's instruction
                #     self.pos = all_well_positions[pos_idx] # REMOVED as per user's instruction
                # else: # REMOVED as per user's instruction
                #     continue # REMOVED as per user's instruction
                self.pos = natsorted(
                    glob(
                        well
                        + f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*{os.sep}"
                    )
                )[pos_idx]
                self.pos_name = extract_position_name(self.pos)
                logger.info(f"Position {self.pos}...\nLoading stack movie...")

                if not os.path.exists(self.pos + "output" + os.sep):
                    os.mkdir(self.pos + "output" + os.sep)
                if not os.path.exists(
                    self.pos + os.sep.join(["output", "tables"]) + os.sep
                ):
                    os.mkdir(self.pos + os.sep.join(["output", "tables"]) + os.sep)

                if self.neigh_action.isChecked():
                    for protocol in self.protocols:

                        process_args = {
                            "pos": self.pos,
                            "pos_name": self.pos_name,
                            "protocol": protocol,
                            "img_shape": (
                                self.parent_window.shape_x,
                                self.parent_window.shape_y,
                            ),
                            "log_file": getattr(
                                self.parent_window.parent_window, "log_file", None
                            ),
                            "well_progress": well_progress,
                            "pos_progress": pos_progress,
                            "measure_pairs": self.measure_pairs_action.isChecked(),
                        }  # "n_threads": self.n_threads
                        self.job = ProgressWindow(
                            NeighborhoodProcess,
                            parent_window=self,
                            title="Neighborhood",
                            process_args=process_args,
                        )
                        result = self.job.exec_()
                        if result == QDialog.Accepted:
                            pass
                        elif result == QDialog.Rejected:
                            return None

                if (
                    self.measure_pairs_action.isChecked()
                    and not self.neigh_action.isChecked()
                ):
                    rel_measure_at_position(self.pos)

                if self.signal_analysis_action.isChecked():

                    analyze_pair_signals_at_position(
                        self.pos,
                        self.pair_signal_models_list.currentText(),
                        use_gpu=self.parent_window.parent_window.use_gpu,
                        populations=self.parent_window.populations,
                    )

        self.parent_window.update_position_options()
        for action in [
            self.neigh_action,
            self.measure_pairs_action,
            self.signal_analysis_action,
        ]:
            if action.isChecked():
                action.setChecked(False)

        logger.info("Done.")
        # 	self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
        # else:

    def check_signals2(self):
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        test = self.parent_window.locate_selected_position()
        if test:
            self.pair_event_annotator = PairEventAnnotator(self)
            self.pair_event_annotator.show()
