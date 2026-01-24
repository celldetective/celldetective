from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QComboBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QMessageBox,
    QApplication,
)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc

from celldetective.utils.model_getters import (
    get_signal_models_list,
    get_segmentation_models_list,
)
from celldetective.utils.data_loaders import load_experiment_tables
from celldetective.utils.model_loaders import (
    locate_signal_model,
    locate_segmentation_model,
)
from celldetective.utils.image_loaders import fix_missing_labels

from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveProgressDialog,
    QHSeperationLine,
    HoverButton,
)

import numpy as np
from glob import glob
from celldetective import get_logger

logger = get_logger("celldetective")


class NapariLoaderThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_with_result = pyqtSignal(object)

    def __init__(self, pos, prefix, population, threads):
        super().__init__()
        self.pos = pos
        self.prefix = prefix
        self.population = population
        self.threads = threads
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        from celldetective.napari.utils import control_tracks

        def callback(p):
            if self._is_cancelled:
                return False
            self.progress.emit(p)
            return True

        try:
            res = control_tracks(
                self.pos,
                prefix=self.prefix,
                population=self.population,
                threads=self.threads,
                progress_callback=callback,
                prepare_only=True,
            )
            self.finished_with_result.emit(res)
        except Exception as e:
            self.finished_with_result.emit(e)


from natsort import natsorted
import os

from celldetective.gui.base.utils import center_window
from celldetective.utils.io import remove_file_if_exists
from tifffile import imwrite
import json
from celldetective.gui.gui_utils import help_generic
from celldetective.gui.base.styles import Styles
from celldetective import get_software_location
import pandas as pd

import logging

logger = logging.getLogger("celldetective")


class ProcessPanel(QFrame, Styles):

    def __init__(self, parent_window, mode):

        super().__init__()
        self.parent_window = parent_window
        self.mode = mode
        self.exp_channels = self.parent_window.exp_channels
        self.exp_dir = self.parent_window.exp_dir
        self.exp_config = self.parent_window.exp_config
        self.movie_prefix = self.parent_window.movie_prefix
        self.threshold_configs = [
            None for _ in range(len(self.parent_window.populations))
        ]
        self.wells = np.array(self.parent_window.wells, dtype=str)
        self.cellpose_calibrated = False
        self.stardist_calibrated = False
        self.segChannelsSet = False
        self.signalChannelsSet = False
        self.flipSeg = False

        self.use_gpu = self.parent_window.parent_window.use_gpu
        self.n_threads = self.parent_window.parent_window.n_threads

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(5, 5, 5, 5)
        self.generate_header()

    def generate_header(self):
        """
        Read the mode and prepare a collapsable block to process a specific cell population.

        """

        panel_title = QLabel(f"PROCESS {self.mode.upper()}   ")
        panel_title.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        title_hbox = QHBoxLayout()
        self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

        # self.help_pop_btn = QPushButton()
        # self.help_pop_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        # self.help_pop_btn.setIconSize(QSize(20, 20))
        # self.help_pop_btn.clicked.connect(self.help_population)
        # self.help_pop_btn.setStyleSheet(self.button_select_all)
        # self.help_pop_btn.setToolTip("Help.")
        # self.grid.addWidget(self.help_pop_btn, 0, 0, 1, 3, alignment=Qt.AlignRight)

        # self.select_all_btn = QPushButton()
        # self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
        # self.select_all_btn.setIconSize(QSize(20, 20))
        # self.all_ticked = False
        # self.select_all_btn.clicked.connect(self.tick_all_actions)
        # self.select_all_btn.setStyleSheet(self.button_select_all)
        # self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
        # self.to_disable.append(self.all_tc_actions)

        self.collapse_btn = QPushButton()
        self.collapse_btn.setIcon(icon(MDI6.chevron_down, color="black"))
        self.collapse_btn.setIconSize(QSize(25, 25))
        self.collapse_btn.setStyleSheet(self.button_select_all)
        # self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

        title_hbox.addWidget(QLabel(), 5)  # self.select_all_btn
        title_hbox.addWidget(QLabel(), 85, alignment=Qt.AlignCenter)
        # title_hbox.addWidget(self.help_pop_btn, 5)
        title_hbox.addWidget(self.collapse_btn, 5)

        self.grid.addLayout(title_hbox, 0, 0, 1, 4)
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
                min(int(930), int(0.9 * self.parent_window.screen_height))
            )
            try:
                QTimer.singleShot(10, lambda: center_window(self.window()))
            except:
                pass

    def populate_contents(self):
        self.ContentsFrame = QFrame()
        self.ContentsFrame.setContentsMargins(5, 5, 5, 5)
        self.grid_contents = QGridLayout(self.ContentsFrame)
        self.grid_contents.setContentsMargins(0, 0, 0, 0)
        self.generate_segmentation_options()
        self.generate_tracking_options()
        self.generate_measure_options()
        self.generate_signal_analysis_options()

        self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
        self.view_tab_btn = QPushButton("Explore table")
        self.view_tab_btn.setStyleSheet(self.button_style_sheet_2)
        self.view_tab_btn.clicked.connect(self.view_table_ui)
        self.view_tab_btn.setToolTip("Explore table")
        self.view_tab_btn.setIcon(icon(MDI6.table, color="#1565c0"))
        self.view_tab_btn.setIconSize(QSize(20, 20))
        # self.view_tab_btn.setEnabled(False)
        self.grid_contents.addWidget(self.view_tab_btn, 10, 0, 1, 4)

        self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.process_population)
        self.grid_contents.addWidget(self.submit_btn, 11, 0, 1, 4)

        for action in [
            self.segment_action,
            self.track_action,
            self.measure_action,
            self.signal_analysis_action,
        ]:
            action.toggled.connect(self.check_readiness)
        self.check_readiness()

    def check_readiness(self):
        if (
            self.segment_action.isChecked()
            or self.track_action.isChecked()
            or self.measure_action.isChecked()
            or self.signal_analysis_action.isChecked()
        ):
            self.submit_btn.setEnabled(True)
        else:
            self.submit_btn.setEnabled(False)

    def generate_measure_options(self):

        measure_layout = QHBoxLayout()

        self.measure_action = QCheckBox("MEASURE")
        self.measure_action.setStyleSheet(self.menu_check_style)

        self.measure_action.setIcon(icon(MDI6.eyedropper, color="black"))
        self.measure_action.setIconSize(QSize(20, 20))
        self.measure_action.setToolTip("Measure.")
        measure_layout.addWidget(self.measure_action, 90)
        # self.to_disable.append(self.measure_action_tc)

        self.classify_btn = QPushButton()
        self.classify_btn.setIcon(icon(MDI6.scatter_plot, color="black"))
        self.classify_btn.setIconSize(QSize(20, 20))
        self.classify_btn.setToolTip("Classify data.")
        self.classify_btn.setStyleSheet(self.button_select_all)
        self.classify_btn.clicked.connect(self.open_classifier_ui)
        measure_layout.addWidget(
            self.classify_btn, 5
        )  # 4,2,1,1, alignment=Qt.AlignRight

        self.check_measurements_btn = QPushButton()
        self.check_measurements_btn.setIcon(icon(MDI6.eye_check_outline, color="black"))
        self.check_measurements_btn.setIconSize(QSize(20, 20))
        self.check_measurements_btn.setToolTip("Explore measurements in-situ.")
        self.check_measurements_btn.setStyleSheet(self.button_select_all)
        self.check_measurements_btn.clicked.connect(self.check_measurements)
        measure_layout.addWidget(self.check_measurements_btn, 5)

        self.measurements_config_btn = QPushButton()
        self.measurements_config_btn.setIcon(icon(MDI6.cog_outline, color="black"))
        self.measurements_config_btn.setIconSize(QSize(20, 20))
        self.measurements_config_btn.setToolTip("Configure measurements.")
        self.measurements_config_btn.setStyleSheet(self.button_select_all)
        self.measurements_config_btn.clicked.connect(
            self.open_measurement_configuration_ui
        )
        measure_layout.addWidget(
            self.measurements_config_btn, 5
        )  # 4,2,1,1, alignment=Qt.AlignRight

        self.grid_contents.addLayout(measure_layout, 5, 0, 1, 4)

    def generate_signal_analysis_options(self):

        signal_layout = QVBoxLayout()
        signal_hlayout = QHBoxLayout()
        self.signal_analysis_action = QCheckBox("DETECT EVENTS")
        self.signal_analysis_action.setStyleSheet(self.menu_check_style)
        self.signal_analysis_action.setIcon(
            icon(MDI6.chart_bell_curve_cumulative, color="black")
        )
        self.signal_analysis_action.setIconSize(QSize(20, 20))
        self.signal_analysis_action.setToolTip("Detect events in single-cell signals.")
        self.signal_analysis_action.toggled.connect(self.enable_signal_model_list)
        signal_hlayout.addWidget(self.signal_analysis_action, 90)

        self.check_signals_btn = QPushButton()
        self.check_signals_btn.setIcon(icon(MDI6.eye_check_outline, color="black"))
        self.check_signals_btn.setIconSize(QSize(20, 20))
        self.check_signals_btn.clicked.connect(self.check_signals)
        self.check_signals_btn.setToolTip("Explore signals in-situ.")
        self.check_signals_btn.setStyleSheet(self.button_select_all)
        signal_hlayout.addWidget(self.check_signals_btn, 6)

        self.config_signal_annotator_btn = QPushButton()
        self.config_signal_annotator_btn.setIcon(icon(MDI6.cog_outline, color="black"))
        self.config_signal_annotator_btn.setIconSize(QSize(20, 20))
        self.config_signal_annotator_btn.setToolTip("Configure the dynamic visualizer.")
        self.config_signal_annotator_btn.setStyleSheet(self.button_select_all)
        self.config_signal_annotator_btn.clicked.connect(
            self.open_signal_annotator_configuration_ui
        )
        signal_hlayout.addWidget(self.config_signal_annotator_btn, 6)

        # self.to_disable.append(self.measure_action_tc)
        signal_layout.addLayout(signal_hlayout)

        signal_model_vbox = QVBoxLayout()
        signal_model_vbox.setContentsMargins(25, 0, 25, 0)

        model_zoo_layout = QHBoxLayout()
        model_zoo_layout.addWidget(QLabel("Model zoo:"), 90)

        self.signal_models_list = QComboBox()
        self.signal_models_list.setEnabled(False)
        self.refresh_signal_models()
        # self.to_disable.append(self.cell_models_list)

        self.train_signal_model_btn = HoverButton(
            "TRAIN", MDI6.redo_variant, "black", "white"
        )
        self.train_signal_model_btn.setToolTip(
            "Train or retrain an event detection model\non newly annotated data."
        )
        self.train_signal_model_btn.setIconSize(QSize(20, 20))
        self.train_signal_model_btn.setStyleSheet(self.button_style_sheet_3)
        model_zoo_layout.addWidget(self.train_signal_model_btn, 5)
        self.train_signal_model_btn.clicked.connect(self.open_signal_model_config_ui)

        signal_model_vbox.addLayout(model_zoo_layout)
        signal_model_vbox.addWidget(self.signal_models_list)

        signal_layout.addLayout(signal_model_vbox)

        self.grid_contents.addLayout(signal_layout, 6, 0, 1, 4)

    def refresh_signal_models(self):
        self.signal_models = get_signal_models_list()
        self.signal_models_list.clear()

        thresh = 35
        models_truncated = [
            m[: thresh - 3] + "..." if len(m) > thresh else m
            for m in self.signal_models
        ]

        self.signal_models_list.addItems(models_truncated)
        for i in range(len(self.signal_models)):
            self.signal_models_list.setItemData(
                i, self.signal_models[i], Qt.ToolTipRole
            )

    def generate_tracking_options(self):

        grid_track = QHBoxLayout()

        self.track_action = QCheckBox("TRACK")
        self.track_action.setStyleSheet(self.menu_check_style)
        self.track_action.setIcon(icon(MDI6.chart_timeline_variant, color="black"))
        self.track_action.setIconSize(QSize(20, 20))
        self.track_action.setToolTip(f"Track the {self.mode[:-1]} cells.")
        grid_track.addWidget(self.track_action, 75)

        self.delete_tracks_btn = QPushButton()
        self.delete_tracks_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_tracks_btn.setIconSize(QSize(20, 20))
        self.delete_tracks_btn.setToolTip("Delete existing tracks.")
        self.delete_tracks_btn.setStyleSheet(self.button_select_all)
        self.delete_tracks_btn.clicked.connect(self.delete_tracks)
        self.delete_tracks_btn.setEnabled(True)
        self.delete_tracks_btn.hide()
        grid_track.addWidget(
            self.delete_tracks_btn, 6
        )  # 4,3,1,1, alignment=Qt.AlignLeft

        self.check_tracking_result_btn = QPushButton()
        self.check_tracking_result_btn.setIcon(
            icon(MDI6.eye_check_outline, color="black")
        )
        self.check_tracking_result_btn.setIconSize(QSize(20, 20))
        self.check_tracking_result_btn.setToolTip("View tracking output in napari.")
        self.check_tracking_result_btn.setStyleSheet(self.button_select_all)
        self.check_tracking_result_btn.clicked.connect(self.open_napari_tracking)
        self.check_tracking_result_btn.setEnabled(False)
        grid_track.addWidget(
            self.check_tracking_result_btn, 6
        )  # 4,3,1,1, alignment=Qt.AlignLeft

        self.track_config_btn = QPushButton()
        self.track_config_btn.setIcon(icon(MDI6.cog_outline, color="black"))
        self.track_config_btn.setIconSize(QSize(20, 20))
        self.track_config_btn.setToolTip("Configure tracking.")
        self.track_config_btn.setStyleSheet(self.button_select_all)
        self.track_config_btn.clicked.connect(self.open_tracking_configuration_ui)
        grid_track.addWidget(
            self.track_config_btn, 6
        )  # 4,2,1,1, alignment=Qt.AlignRight

        self.help_track_btn = QPushButton()
        self.help_track_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        self.help_track_btn.setIconSize(QSize(20, 20))
        self.help_track_btn.clicked.connect(self.help_tracking)
        self.help_track_btn.setStyleSheet(self.button_select_all)
        self.help_track_btn.setToolTip("Help.")
        grid_track.addWidget(self.help_track_btn, 6)  # 4,2,1,1, alignment=Qt.AlignRight

        self.grid_contents.addLayout(grid_track, 4, 0, 1, 4)

    def delete_tracks(self):

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText(
            "Do you want to erase the tracks? All subsequent annotations will be erased..."
        )
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.No:
            return None
        elif returnValue == QMessageBox.Yes:
            remove_file_if_exists(
                os.sep.join(
                    [
                        self.parent_window.pos,
                        "output",
                        "tables",
                        f"trajectories_{self.mode}.csv",
                    ]
                )
            )
            remove_file_if_exists(
                os.sep.join(
                    [
                        self.parent_window.pos,
                        "output",
                        "tables",
                        f"trajectories_{self.mode}.pkl",
                    ]
                )
            )
            remove_file_if_exists(
                os.sep.join(
                    [
                        self.parent_window.pos,
                        "output",
                        "tables",
                        f"napari_{self.mode[:-1]}_trajectories.npy",
                    ]
                )
            )
            remove_file_if_exists(
                os.sep.join(
                    [
                        self.parent_window.pos,
                        "output",
                        "tables",
                        f"trajectories_pairs.csv",
                    ]
                )
            )
            try:
                QTimer.singleShot(
                    100, lambda: self.parent_window.update_position_options()
                )
            except Exception as _:
                pass
        else:
            return None

    def generate_segmentation_options(self):

        grid_segment = QHBoxLayout()
        grid_segment.setContentsMargins(0, 0, 0, 0)
        grid_segment.setSpacing(0)

        self.segment_action = QCheckBox("SEGMENT")
        self.segment_action.setStyleSheet(self.menu_check_style)
        self.segment_action.setIcon(icon(MDI6.bacteria, color="black"))
        self.segment_action.setToolTip(
            f"Segment the {self.mode[:-1]} cells on the images."
        )
        self.segment_action.toggled.connect(self.enable_segmentation_model_list)
        # self.to_disable.append(self.segment_action)
        grid_segment.addWidget(self.segment_action, 90)

        # self.flip_segment_btn = QPushButton()
        # self.flip_segment_btn.setIcon(icon(MDI6.camera_flip_outline,color="black"))
        # self.flip_segment_btn.setIconSize(QSize(20, 20))
        # self.flip_segment_btn.clicked.connect(self.flip_segmentation)
        # self.flip_segment_btn.setStyleSheet(self.button_select_all)
        # self.flip_segment_btn.setToolTip("Flip the order of the frames for segmentation.")
        # grid_segment.addWidget(self.flip_segment_btn, 5)

        self.segmentation_config_btn = QPushButton()
        self.segmentation_config_btn.setIcon(icon(MDI6.cog_outline, color="black"))
        self.segmentation_config_btn.setIconSize(QSize(20, 20))
        self.segmentation_config_btn.setToolTip("Configure segmentation.")
        self.segmentation_config_btn.setStyleSheet(self.button_select_all)
        self.segmentation_config_btn.clicked.connect(
            self.open_segmentation_configuration_ui
        )
        grid_segment.addWidget(self.segmentation_config_btn, 5)

        self.check_seg_btn = QPushButton()
        self.check_seg_btn.setIcon(icon(MDI6.eye_check_outline, color="black"))
        self.check_seg_btn.setIconSize(QSize(20, 20))
        self.check_seg_btn.clicked.connect(self.check_segmentation)
        self.check_seg_btn.setStyleSheet(self.button_select_all)
        self.check_seg_btn.setToolTip("View segmentation output in napari.")
        grid_segment.addWidget(self.check_seg_btn, 5)

        self.help_seg_btn = QPushButton()
        self.help_seg_btn.setIcon(icon(MDI6.help_circle, color=self.help_color))
        self.help_seg_btn.setIconSize(QSize(20, 20))
        self.help_seg_btn.clicked.connect(self.help_segmentation)
        self.help_seg_btn.setStyleSheet(self.button_select_all)
        self.help_seg_btn.setToolTip("Help.")
        grid_segment.addWidget(self.help_seg_btn, 5)
        self.grid_contents.addLayout(grid_segment, 0, 0, 1, 4)

        seg_option_vbox = QVBoxLayout()
        seg_option_vbox.setContentsMargins(25, 0, 25, 0)
        model_zoo_layout = QHBoxLayout()
        model_zoo_layout.addWidget(QLabel("Model zoo:"), 90)
        self.seg_model_list = QComboBox()
        self.seg_model_list.currentIndexChanged.connect(self.reset_generalist_setup)
        # self.to_disable.append(self.tc_seg_model_list)
        self.seg_model_list.setGeometry(50, 50, 200, 30)
        self.init_seg_model_list()

        self.upload_model_btn = HoverButton("UPLOAD", MDI6.upload, "black", "white")
        self.upload_model_btn.setIconSize(QSize(20, 20))
        self.upload_model_btn.setStyleSheet(self.button_style_sheet_3)
        self.upload_model_btn.setToolTip(
            "Upload a new segmentation model\n(Deep learning or threshold-based)."
        )
        model_zoo_layout.addWidget(self.upload_model_btn, 5)
        self.upload_model_btn.clicked.connect(self.upload_segmentation_model)
        # self.to_disable.append(self.upload_tc_model)

        self.train_btn = HoverButton("TRAIN", MDI6.redo_variant, "black", "white")
        self.train_btn.setToolTip(
            "Train or retrain a segmentation model\non newly annotated data."
        )
        self.train_btn.setIconSize(QSize(20, 20))
        self.train_btn.setStyleSheet(self.button_style_sheet_3)
        self.train_btn.clicked.connect(self.open_segmentation_model_config_ui)
        model_zoo_layout.addWidget(self.train_btn, 5)
        # self.train_button_tc.clicked.connect(self.train_stardist_model_tc)
        # self.to_disable.append(self.train_button_tc)

        seg_option_vbox.addLayout(model_zoo_layout)
        seg_option_vbox.addWidget(self.seg_model_list)
        self.seg_model_list.setEnabled(False)
        self.grid_contents.addLayout(seg_option_vbox, 2, 0, 1, 4)

    def flip_segmentation(self):
        if not self.flipSeg:
            self.flipSeg = True
            self.flip_segment_btn.setIcon(
                icon(MDI6.camera_flip, color=self.celldetective_blue)
            )
            self.flip_segment_btn.setIconSize(QSize(20, 20))
            self.flip_segment_btn.setToolTip(
                "Unflip the order of the frames for segmentation."
            )
        else:
            self.flipSeg = False
            self.flip_segment_btn.setIcon(icon(MDI6.camera_flip_outline, color="black"))
            self.flip_segment_btn.setIconSize(QSize(20, 20))
            self.flip_segment_btn.setToolTip(
                "Flip the order of the frames for segmentation."
            )

    def help_segmentation(self):
        """
        Widget with different decision helper decision trees.
        """

        self.help_w = CelldetectiveWidget()
        self.help_w.setWindowTitle("Helper")
        layout = QVBoxLayout()
        seg_strategy_btn = QPushButton("A guide to choose a segmentation strategy.")
        seg_strategy_btn.setIcon(icon(MDI6.help_circle, color=self.celldetective_blue))
        seg_strategy_btn.setIconSize(QSize(40, 40))
        seg_strategy_btn.setStyleSheet(self.button_style_sheet_5)
        seg_strategy_btn.clicked.connect(self.help_seg_strategy)

        dl_strategy_btn = QPushButton(
            "A guide to choose your Deep learning segmentation strategy."
        )
        dl_strategy_btn.setIcon(icon(MDI6.help_circle, color=self.celldetective_blue))
        dl_strategy_btn.setIconSize(QSize(40, 40))
        dl_strategy_btn.setStyleSheet(self.button_style_sheet_5)
        dl_strategy_btn.clicked.connect(self.help_seg_dl_strategy)

        layout.addWidget(seg_strategy_btn)
        layout.addWidget(dl_strategy_btn)

        self.help_w.setLayout(layout)
        center_window(self.help_w)
        self.help_w.show()

        return None

    def help_seg_strategy(self):
        """
        Helper for segmentation strategy between threshold-based and Deep learning.
        """

        dict_path = os.sep.join(
            [
                get_software_location(),
                "celldetective",
                "gui",
                "help",
                "Threshold-vs-DL.json",
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
                f"The suggested technique is {suggestion}.\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/segment.html'>here</a>."
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

    def help_seg_dl_strategy(self):
        """
        Helper for DL segmentation strategy, between pretrained models and custom models.
        """

        dict_path = os.sep.join(
            [
                get_software_location(),
                "celldetective",
                "gui",
                "help",
                "DL-segmentation-strategy.json",
            ]
        )

        with open(dict_path) as f:
            d = json.load(f)

        suggestion = help_generic(d)
        if isinstance(suggestion, str):
            logger.info(f"{suggestion=}")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText(f"The suggested technique is {suggestion}.")
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

    def help_tracking(self):
        """
        Helper for segmentation strategy between threshold-based and Deep learning.
        """

        dict_path = os.sep.join(
            [get_software_location(), "celldetective", "gui", "help", "tracking.json"]
        )

        with open(dict_path) as f:
            d = json.load(f)

        suggestion = help_generic(d)
        if isinstance(suggestion, str):
            logger.info(f"{suggestion=}")
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setTextFormat(Qt.RichText)
            msgBox.setText(f"{suggestion}")
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

    def check_segmentation(self):
        from celldetective.napari.utils import control_segmentation_napari

        if not os.path.exists(
            os.sep.join([self.parent_window.pos, f"labels_{self.mode}", os.sep])
        ):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText(
                "No labels can be found for this position. Do you want to annotate from scratch?"
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.No:
                return None
            else:
                os.mkdir(os.sep.join([self.parent_window.pos, f"labels_{self.mode}"]))
                lbl = np.zeros(
                    (self.parent_window.shape_x, self.parent_window.shape_y), dtype=int
                )
                for i in range(self.parent_window.len_movie):
                    imwrite(
                        os.sep.join(
                            [
                                self.parent_window.pos,
                                f"labels_{self.mode}",
                                str(i).zfill(4) + ".tif",
                            ]
                        ),
                        lbl,
                    )

        # self.freeze()
        # QApplication.setOverrideCursor(Qt.WaitCursor)
        test = self.parent_window.locate_selected_position()
        if test:
            # print('Memory use: ', dict(psutil.virtual_memory()._asdict()))
            logger.info(f"Loading images and labels into napari...")
            try:
                control_segmentation_napari(
                    self.parent_window.pos,
                    prefix=self.parent_window.movie_prefix,
                    population=self.mode,
                    flush_memory=True,
                )
            except FileNotFoundError as e:
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText(str(e))
                msgBox.setWindowTitle("Warning")
                msgBox.setStandardButtons(QMessageBox.Ok)
                _ = msgBox.exec()
                return
            except Exception as e:
                logger.error(f"Task unsuccessful... Exception {e}...")
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText(str(e))
                msgBox.setWindowTitle("Warning")
                msgBox.setStandardButtons(QMessageBox.Ok)
                _ = msgBox.exec()

                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Question)
                msgBox.setText(
                    "Would you like to pass empty frames to fix the asymmetry?"
                )
                msgBox.setWindowTitle("Question")
                msgBox.setStandardButtons(
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                returnValue = msgBox.exec()
                if returnValue == QMessageBox.Yes:
                    logger.info("Fixing the missing labels...")
                    fix_missing_labels(
                        self.parent_window.pos,
                        prefix=self.parent_window.movie_prefix,
                        population=self.mode,
                    )
                    try:
                        control_segmentation_napari(
                            self.parent_window.pos,
                            prefix=self.parent_window.movie_prefix,
                            population=self.mode,
                            flush_memory=True,
                        )
                    except Exception as e:
                        logger.error(f"Error {e}")
                        return None
                else:
                    return None

            gc.collect()

    def check_signals(self):
        from celldetective.gui.event_annotator import EventAnnotator, StackLoaderThread
        from celldetective.utils.experiment import interpret_wells_and_positions

        self.well_option = self.parent_window.well_list.getSelectedIndices()
        self.position_option = self.parent_window.position_list.getSelectedIndices()

        # Count selected positions
        well_indices, position_indices = interpret_wells_and_positions(
            self.exp_dir, self.well_option, self.position_option
        )
        total_positions = 0
        from celldetective.utils.experiment import (
            get_positions_in_well,
            get_experiment_wells,
        )

        wells = get_experiment_wells(self.exp_dir)
        for widx in well_indices:
            positions = get_positions_in_well(wells[widx])
            if position_indices is not None:
                total_positions += len(position_indices)
            else:
                total_positions += len(positions)

        if total_positions == 1:
            test = self.parent_window.locate_selected_position()
            if test:
                self.event_annotator = EventAnnotator(self, lazy_load=True)

                if not getattr(self.event_annotator, "proceed", True):
                    return

                self.signal_loader = StackLoaderThread(self.event_annotator)

                self.signal_progress = CelldetectiveProgressDialog(
                    "Loading data...",
                    "Cancel",
                    0,
                    100,
                    self,
                    window_title="Please wait",
                )

                self.signal_progress.setValue(0)

                self.signal_loader.progress.connect(self.signal_progress.setValue)
                self.signal_loader.status_update.connect(
                    self.signal_progress.setLabelText
                )
                self.signal_progress.canceled.connect(self.signal_loader.stop)

                def on_finished():
                    self.signal_progress.blockSignals(True)
                    self.signal_progress.close()
                    if not self.signal_loader._is_cancelled:
                        try:
                            self.event_annotator.finalize_init()
                            self.event_annotator.show()
                            try:
                                QTimer.singleShot(
                                    100,
                                    lambda: self.event_annotator.resize(
                                        self.event_annotator.width() + 1,
                                        self.event_annotator.height() + 1,
                                    ),
                                )
                            except:
                                pass
                        except Exception as e:
                            print(f"Error finalizing annotator: {e}")
                    else:
                        self.event_annotator.close()

                self.signal_loader.finished.connect(on_finished)
                self.signal_loader.start()
        else:
            # Multi position explorer: redirect to TableUI with progress bar
            self.view_table_ui()

    def check_measurements(self):
        from celldetective.gui.measure_annotator import MeasureAnnotator
        from celldetective.utils.experiment import interpret_wells_and_positions

        self.well_option = self.parent_window.well_list.getSelectedIndices()
        self.position_option = self.parent_window.position_list.getSelectedIndices()

        # Count selected positions
        well_indices, position_indices = interpret_wells_and_positions(
            self.exp_dir, self.well_option, self.position_option
        )
        total_positions = 0
        from celldetective.utils.experiment import (
            get_positions_in_well,
            get_experiment_wells,
        )

        wells = get_experiment_wells(self.exp_dir)
        for widx in well_indices:
            positions = get_positions_in_well(wells[widx])
            if position_indices is not None:
                total_positions += len(position_indices)
            else:
                total_positions += len(positions)

        if total_positions == 1:
            test = self.parent_window.locate_selected_position()
            if test:
                self.measure_annotator = MeasureAnnotator(self)
                self.measure_annotator.show()
        else:
            # Multi position explorer: redirect to TableUI with progress bar
            self.view_table_ui()

    def enable_segmentation_model_list(self):
        if self.segment_action.isChecked():
            self.seg_model_list.setEnabled(True)
        else:
            self.seg_model_list.setEnabled(False)

    def enable_signal_model_list(self):
        if self.signal_analysis_action.isChecked():
            self.signal_models_list.setEnabled(True)
        else:
            self.signal_models_list.setEnabled(False)

    def init_seg_model_list(self):

        self.seg_model_list.clear()
        self.seg_models_specific = get_segmentation_models_list(
            mode=self.mode, return_path=False
        )
        self.seg_models = self.seg_models_specific.copy()
        self.n_specific_seg_models = len(self.seg_models)

        self.seg_models_generic = get_segmentation_models_list(
            mode="generic", return_path=False
        )
        self.seg_models.append("Threshold")
        self.seg_models.extend(self.seg_models_generic)

        thresh = 35
        self.models_truncated = [
            m[: thresh - 3] + "..." if len(m) > thresh else m for m in self.seg_models
        ]

        self.seg_model_list.addItems(self.models_truncated)

        for i in range(len(self.seg_models)):
            self.seg_model_list.setItemData(i, self.seg_models[i], Qt.ToolTipRole)

        self.seg_model_list.insertSeparator(self.n_specific_seg_models)

    # def tick_all_actions(self):
    # 	self.switch_all_ticks_option()
    # 	if self.all_ticked:
    # 		self.select_all_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
    # 		self.select_all_btn.setIconSize(QSize(20, 20))
    # 		self.segment_action.setChecked(True)
    # 	else:
    # 		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
    # 		self.select_all_btn.setIconSize(QSize(20, 20))
    # 		self.segment_action.setChecked(False)

    # def switch_all_ticks_option(self):
    # 	if self.all_ticked == True:
    # 		self.all_ticked = False
    # 	else:
    # 		self.all_ticked = True

    def upload_segmentation_model(self):
        from celldetective.gui.seg_model_loader import SegmentationModelLoader

        logger.info("Load a segmentation model or pipeline...")
        self.seg_model_loader = SegmentationModelLoader(self)
        self.seg_model_loader.show()
        center_window(self.seg_model_loader)

    def open_tracking_configuration_ui(self):
        from celldetective.gui.settings._settings_tracking import SettingsTracking

        logger.info("Set the tracking parameters...")
        self.settings_tracking = SettingsTracking(self)
        self.settings_tracking.show()
        center_window(self.settings_tracking)

    def open_signal_model_config_ui(self):
        from celldetective.gui.settings._settings_event_model_training import (
            SettingsEventDetectionModelTraining,
        )

        logger.info("Set the training parameters for new signal models...")
        self.settings_event_detection_training = SettingsEventDetectionModelTraining(
            self
        )
        self.settings_event_detection_training.show()
        center_window(self.settings_event_detection_training)

    def open_segmentation_model_config_ui(self):
        from celldetective.gui.settings._settings_segmentation_model_training import (
            SettingsSegmentationModelTraining,
        )

        logger.info("Set the training parameters for a new segmentation model...")
        self.settings_segmentation_training = SettingsSegmentationModelTraining(self)
        self.settings_segmentation_training.show()
        center_window(self.settings_segmentation_training)

    def open_measurement_configuration_ui(self):
        from celldetective.gui.settings._settings_measurements import (
            SettingsMeasurements,
        )

        logger.info("Set the measurements to be performed...")
        self.settings_measurements = SettingsMeasurements(self)
        self.settings_measurements.show()
        center_window(self.settings_measurements)

    def open_segmentation_configuration_ui(self):
        from celldetective.gui.settings._settings_segmentation import (
            SettingsSegmentation,
        )

        logger.info("Set the segmentation settings to be performed...")
        self.settings_segmentation = SettingsSegmentation(self)
        self.settings_segmentation.show()

    def open_classifier_ui(self):
        from celldetective.gui.classifier_widget import ClassifierWidget

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
            self.classifier_widget = ClassifierWidget(self)
            self.classifier_widget.show()
            try:

                def post_widget(wdg):
                    try:
                        wdg.resize(wdg.width() + 1, wdg.height() + 1)
                        center_window(wdg)
                    except Exception as _:
                        pass

                QTimer.singleShot(100, lambda: post_widget(self.classifier_widget))
            except Exception as _:
                pass

    def open_signal_annotator_configuration_ui(self):
        from celldetective.gui.settings._settings_signal_annotator import (
            SettingsSignalAnnotator,
        )

        self.settings_signal_annotator = SettingsSignalAnnotator(self)
        self.settings_signal_annotator.show()
        try:
            QTimer.singleShot(
                100, lambda: center_window(self.settings_signal_annotator)
            )
        except Exception as _:
            pass

    def reset_generalist_setup(self, index):
        self.cellpose_calibrated = False
        self.stardist_calibrated = False
        self.segChannelsSet = False

    def reset_signals(self):
        self.signalChannelsSet = False

    def process_population(self):
        from celldetective.processes.unified_process import UnifiedBatchProcess
        from celldetective.gui.workers import ProgressWindow

        # Check positions/wells
        self.well_index = self.parent_window.well_list.getSelectedIndices()
        if len(self.well_index) == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select at least one well first...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
            else:
                return None

        logger.info(f"Processing {self.parent_window.well_list.currentText()}...")

        idx = self.parent_window.populations.index(self.mode)
        self.threshold_config = self.threshold_configs[idx]

        self.load_available_tables()

        # Checks for segmentation action
        if self.df is not None and self.segment_action.isChecked():
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText(
                "Measurement tables have been found... Re-segmenting may create mismatches between the cell labels and the associated measurements. Do you want to erase the tables post-segmentation?"
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.No:
                pass
            elif returnValue == QMessageBox.Cancel:
                return None
            else:
                logger.info("erase tabs!")
                tabs = [
                    pos
                    + os.sep.join(["output", "tables", f"trajectories_{self.mode}.csv"])
                    for pos in self.df_pos_info["pos_path"].unique()
                ]
                # tabs += [pos+os.sep.join(['output', 'tables', f'trajectories_pairs.csv']) for pos in self.df_pos_info['pos_path'].unique()]
                tabs += [
                    pos
                    + os.sep.join(
                        ["output", "tables", f"napari_{self.mode}_trajectories.npy"]
                    )
                    for pos in self.df_pos_info["pos_path"].unique()
                ]
                for t in tabs:
                    remove_file_if_exists(t.replace(".csv", ".pkl"))
                    try:
                        os.remove(t)
                    except:
                        pass

        if self.seg_model_list.currentIndex() > self.n_specific_seg_models:
            self.model_name = self.seg_models[self.seg_model_list.currentIndex() - 1]
        else:
            self.model_name = self.seg_models[self.seg_model_list.currentIndex()]

        if (
            self.segment_action.isChecked()
            and self.model_name.startswith("CP")
            and self.model_name in self.seg_models_generic
            and not self.cellpose_calibrated
        ):
            from celldetective.gui.settings._cellpose_model_params import (
                CellposeParamsWidget,
            )

            self.diamWidget = CellposeParamsWidget(self, model_name=self.model_name)
            self.diamWidget.show()

            return None
        elif (
            self.segment_action.isChecked()
            and self.model_name.startswith("SD")
            and self.model_name in self.seg_models_generic
            and not self.stardist_calibrated
        ):
            from celldetective.gui.settings._stardist_model_params import (
                StarDistParamsWidget,
            )

            self.diamWidget = StarDistParamsWidget(self, model_name=self.model_name)
            self.diamWidget.show()

            return None

        elif (
            self.segment_action.isChecked()
            and self.model_name in self.seg_models_specific
            and not self.segChannelsSet
        ):
            from celldetective.gui.settings._segmentation_model_params import (
                SegModelParamsWidget,
            )

            self.segChannelWidget = SegModelParamsWidget(
                self, model_name=self.model_name
            )
            self.segChannelWidget.show()

            return None

        if self.signal_analysis_action.isChecked() and not self.signalChannelsSet:
            from celldetective.gui.settings._event_detection_model_params import (
                SignalModelParamsWidget,
            )

            self.signal_model_name = self.signal_models[
                self.signal_models_list.currentIndex()
            ]
            self.signalChannelWidget = SignalModelParamsWidget(
                self, model_name=self.signal_model_name
            )
            self.signalChannelWidget.show()

            return None

        self.movie_prefix = self.parent_window.movie_prefix

        if self.parent_window.position_list.isMultipleSelection():
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText(
                "If you continue, several positions will be processed.\nDo you want to proceed?"
            )
            msgBox.setWindowTitle("Info")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.No:
                return None

        self.movie_prefix = self.parent_window.movie_prefix

        # COLLECT POSITIONS
        batch_structure = {}
        all_positions_flat = (
            []
        )  # Keep flat list for legacy check logic or easy counting

        for w_idx in self.well_index:
            well = self.parent_window.wells[w_idx]

            batch_structure[w_idx] = {"well_name": well, "positions": []}

            pos_indices = self.parent_window.position_list.getSelectedIndices()
            # Optimization: Glob once per well
            all_well_positions = natsorted(
                glob(
                    well
                    + f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*/"
                )
            )
            for pos_idx in pos_indices:
                if pos_idx < len(all_well_positions):
                    pos = all_well_positions[pos_idx]
                else:
                    logger.warning(
                        f"Position index {pos_idx} out of range for well {well}"
                    )
                    continue

                batch_structure[w_idx]["positions"].append(pos)
                all_positions_flat.append(pos)

                # Check output folders creation
                if not os.path.exists(pos + "output/"):
                    os.mkdir(pos + "output/")
                if not os.path.exists(pos + "output/tables/"):
                    os.mkdir(pos + "output/tables/")

        # BATCH SEGMENTATION
        # --- UNIFIED BATCH PROCESS SETUP ---

        run_segmentation = self.segment_action.isChecked()
        run_tracking = self.track_action.isChecked()
        run_measurement = self.measure_action.isChecked()
        run_signals = self.signal_analysis_action.isChecked()

        seg_args = {}
        track_args = {}
        measure_args = {}
        signal_args = {}

        # 1. SEGMENTATION CHECKS & ARGS
        if run_segmentation:
            # Single-position overwrite check
            if (
                len(all_positions_flat) == 1
                and not self.parent_window.position_list.isMultipleSelection()
            ):
                p = all_positions_flat[0]
                if len(glob(os.sep.join([p, f"labels_{self.mode}", "*.tif"]))) > 0:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Question)
                    msgBox.setText(
                        "Labels have already been produced for this position. Do you want to segment again?"
                    )
                    msgBox.setWindowTitle("Info")
                    msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    if msgBox.exec() == QMessageBox.No:
                        run_segmentation = False

            # Threshold config check
            if run_segmentation and self.seg_model_list.currentText() == "Threshold":
                if self.threshold_config is None:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Warning)
                    msgBox.setText(
                        "Please set a threshold configuration from the upload menu first. Abort."
                    )
                    msgBox.setWindowTitle("Warning")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec()
                    return None

                seg_args = {
                    "mode": self.mode,
                    "n_threads": self.n_threads,
                    "threshold_instructions": self.threshold_config,
                    "use_gpu": self.use_gpu,
                }
            elif run_segmentation:  # Deep Learning
                # Prepare representative position for process initialization
                first_pos = None
                if all_positions_flat and isinstance(all_positions_flat[0], str):
                    first_pos = all_positions_flat[0]

                seg_args = {
                    "mode": self.mode,
                    "pos": first_pos,
                    "n_threads": self.n_threads,
                    "model_name": self.model_name,
                    "use_gpu": self.use_gpu,
                }

        # 2. TRACKING CHECKS & ARGS
        if run_tracking:

            # Single-position overwrite check
            if (
                len(all_positions_flat) == 1
                and not self.parent_window.position_list.isMultipleSelection()
            ):
                p = all_positions_flat[0]
                table_path = os.sep.join(
                    [p, "output", "tables", f"trajectories_{self.mode}.csv"]
                )
                if os.path.exists(table_path):
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Question)
                    msgBox.setText(
                        "A measurement table already exists. Previously annotated data for\nthis position will be lost. Do you want to proceed?"
                    )
                    msgBox.setWindowTitle("Info")
                    msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    if msgBox.exec() == QMessageBox.No:
                        run_tracking = False

            if run_tracking:
                track_args = {"mode": self.mode, "n_threads": self.n_threads}

        # 3. MEASUREMENT ARGS
        if run_measurement:
            measure_args = {"mode": self.mode, "n_threads": self.n_threads}

        # 4. SIGNAL ANALYSIS CHECKS & ARGS
        if run_signals:
            if (
                len(all_positions_flat) == 1
                and not self.parent_window.position_list.isMultipleSelection()
            ):
                p = all_positions_flat[0]
                table_path = os.sep.join(
                    [p, "output", "tables", f"trajectories_{self.mode}.csv"]
                )
                if os.path.exists(table_path):
                    # Check for annotations (logic from original code)
                    try:
                        # Optimized reading: Check header first, then read interesting column
                        header = pd.read_csv(table_path, nrows=0).columns
                        if "class_color" in list(header):
                            colors = pd.read_csv(table_path, usecols=["class_color"])[
                                "class_color"
                            ].unique()
                            if "tab:orange" in colors or "tab:cyan" in colors:
                                msgBox = QMessageBox()
                                msgBox.setIcon(QMessageBox.Question)
                                msgBox.setText(
                                    "The signals of the cells in the position appear to have been annotated... Do you want to proceed?"
                                )
                                msgBox.setWindowTitle("Info")
                                msgBox.setStandardButtons(
                                    QMessageBox.Yes | QMessageBox.No
                                )
                                if msgBox.exec() == QMessageBox.No:
                                    run_signals = False
                    except Exception as e:
                        logger.warning(f"Could not check table for annotations: {e}")

            if run_signals:
                self.signal_model_name = self.signal_models[
                    self.signal_models_list.currentIndex()
                ]

                model_complete_path = locate_signal_model(self.signal_model_name)
                input_config_path = os.path.join(
                    model_complete_path, "config_input.json"
                )
                with open(input_config_path) as config_file:
                    input_config = json.load(config_file)

                channels = input_config.get(
                    "selected_channels", input_config.get("channels", [])
                )

                signal_args = {
                    "model_name": self.signal_model_name,
                    "mode": self.mode,
                    "channels": channels,
                }

        # --- EXECUTE UNIFIED PROCESS ---
        if any([run_segmentation, run_tracking, run_measurement, run_signals]):

            process_args = {
                "batch_structure": batch_structure,
                "run_segmentation": run_segmentation,
                "run_tracking": run_tracking,
                "run_measurement": run_measurement,
                "run_signals": run_signals,
                "seg_args": seg_args,
                "track_args": track_args,
                "measure_args": measure_args,
                "signal_args": signal_args,
                "log_file": getattr(self.parent_window.parent_window, "log_file", None),
            }

            self.job = ProgressWindow(
                UnifiedBatchProcess,
                parent_window=self,
                title=f"Processing {self.mode}",
                process_args=process_args,
            )
            result = self.job.exec_()

            if result == QDialog.Rejected:
                self.reset_generalist_setup(0)
                return None

            # Post-Process actions (like updating list)
            if run_tracking:
                self.parent_window.update_position_options()

            if run_signals:
                from celldetective.gui.interactive_timeseries_viewer import (
                    InteractiveEventViewer,
                )

                self.parent_window.update_position_options()

                if len(all_positions_flat) == 1:
                    p = all_positions_flat[0]
                    mode_fixed = self.mode
                    if self.mode.lower() in ["target", "targets"]:
                        mode_fixed = "targets"
                    elif self.mode.lower() in ["effector", "effectors"]:
                        mode_fixed = "effectors"

                    table_path = os.sep.join(
                        [p, "output", "tables", f"trajectories_{mode_fixed}.csv"]
                    )

                    if os.path.exists(table_path):
                        # Determine event label
                        event_label = None
                        signal_name = None
                        try:
                            if hasattr(self, "signal_model_name"):
                                model_complete_path = locate_signal_model(
                                    self.signal_model_name
                                )
                                input_config_path = os.path.join(
                                    model_complete_path, "config_input.json"
                                )
                                if os.path.exists(input_config_path):
                                    with open(input_config_path) as f:
                                        conf = json.load(f)
                                    event_label = conf.get("label", None)
                                    channels = conf.get("channels", [])
                                    if channels:
                                        signal_name = channels[0]
                        except Exception as e:
                            logger.warning(f"Could not determine event label: {e}")

                        logger.info(
                            f"Launching Interactive Event Viewer for {table_path} with label {event_label}"
                        )
                        self.viewer = InteractiveEventViewer(
                            table_path,
                            signal_name=signal_name,
                            event_label=event_label,
                            parent=self,
                        )
                        self.viewer.show()
                        center_window(self.viewer)
                    else:
                        logger.warning(
                            f"Could not find table for interactive viewer: {table_path}"
                        )
        for action in [
            self.segment_action,
            self.track_action,
            self.measure_action,
            self.signal_analysis_action,
        ]:
            if action.isChecked():
                action.setChecked(False)

        self.reset_generalist_setup(0)
        self.reset_signals()

    def open_napari_tracking(self):

        logger.info(
            f"View the tracks before post-processing for position {self.parent_window.pos} in napari..."
        )

        self.napari_loader = NapariLoaderThread(
            self.parent_window.pos,
            self.parent_window.movie_prefix,
            self.mode,
            self.parent_window.parent_window.n_threads,
        )

        self.napari_progress = CelldetectiveProgressDialog(
            "Loading images, tracks and relabeling masks...",
            "Cancel",
            0,
            100,
            self,
            window_title="Preparing the napari viewer...",
        )

        self.napari_progress.setAutoClose(False)
        self.napari_progress.setAutoReset(False)

        self.napari_progress.setValue(0)
        self.napari_loader.progress.connect(self.napari_progress.setValue)
        self.napari_loader.status.connect(self.napari_progress.setLabelText)
        self.napari_progress.canceled.connect(self.napari_loader.stop)

        def on_finished(result):
            from celldetective.napari.utils import launch_napari_viewer

            self.napari_progress.blockSignals(True)
            # self.napari_progress.close()
            if self.napari_loader._is_cancelled:
                logger.info("Task was cancelled...")
                self.napari_progress.close()
                return

            if isinstance(result, Exception):
                logger.error(f"napari loading error: {result}")
                self.napari_progress.close()
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText(str(result))
                msgBox.setWindowTitle("Warning")
                msgBox.setStandardButtons(QMessageBox.Ok)
                _ = msgBox.exec()
                return

            if result:
                logger.info("Launching the napari viewer with tracks...")
                self.napari_progress.setLabelText("Initializing Napari viewer...")
                self.napari_progress.setRange(0, 0)
                QApplication.processEvents()

                def progress_cb(msg):
                    if isinstance(msg, str):
                        self.napari_progress.setLabelText(msg)
                    QApplication.processEvents()

                if "flush_memory" in result:
                    result.pop("flush_memory")

                try:
                    launch_napari_viewer(
                        **result,
                        block=False,
                        flush_memory=False,
                        progress_callback=progress_cb,
                    )
                    logger.info("napari viewer was closed...")
                except Exception as e:
                    logger.error(f"Failed to launch Napari: {e}")
                    QMessageBox.warning(self, "Error", f"Failed to launch Napari: {e}")
                finally:
                    self.napari_progress.close()
            else:
                self.napari_progress.close()
                logger.warning(
                    "napari loading returned None (likely no trajectories found)."
                )
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Could not load tracks. Please ensure trajectories are computed.",
                )

        self.napari_loader.finished_with_result.connect(on_finished)
        self.napari_loader.start()

    def view_table_ui(self):
        from celldetective.gui.tableUI import TableUI
        from celldetective.gui.workers import ProgressWindow
        from celldetective.processes.load_table import TableLoaderProcess
        from celldetective.utils.experiment import interpret_wells_and_positions

        logger.info("Load table...")

        # Prepare args for the process
        self.well_option = self.parent_window.well_list.getSelectedIndices()
        self.position_option = self.parent_window.position_list.getSelectedIndices()

        # Count selected positions
        well_indices, position_indices = interpret_wells_and_positions(
            self.exp_dir, self.well_option, self.position_option
        )
        total_positions = 0
        from celldetective.utils.experiment import (
            get_positions_in_well,
            get_experiment_wells,
        )

        wells = get_experiment_wells(self.exp_dir)
        for widx in well_indices:
            positions = get_positions_in_well(wells[widx])
            if position_indices is not None:
                total_positions += len(position_indices)
            else:
                total_positions += len(positions)

        def show_table(df):
            if df is not None:
                plot_mode = "plot_track_signals"
                if "TRACK_ID" not in list(df.columns):
                    plot_mode = "static"
                self.tab_ui = TableUI(
                    df,
                    f"{self.parent_window.well_list.currentText()}; Position {self.parent_window.position_list.currentText()}",
                    population=self.mode,
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
                msgBox.exec()

        if total_positions == 1:
            # Synchronous load for single position
            from celldetective.utils.data_loaders import load_experiment_tables

            df = load_experiment_tables(
                self.exp_dir,
                population=self.mode,
                well_option=self.well_option,
                position_option=self.position_option,
            )
            show_table(df)
        else:
            # Asynchronous load for multiple positions
            process_args = {
                "experiment": self.exp_dir,
                "population": self.mode,
                "well_option": self.well_option,
                "position_option": self.position_option,
                "show_frame_progress": False,
            }

            self.df = None

            def on_table_loaded(df):
                self.df = df
                show_table(self.df)

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
            self.job.exec_()

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
            population=self.mode,
            return_pos_info=True,
        )
        self.signals = []
        if self.df is not None:
            self.signals = list(self.df.columns)
        else:
            logger.info(
                "No table could be found for the selected position(s)... Anticipating measurements..."
            )

            from celldetective.utils.experiment import extract_experiment_channels

            channel_names, _ = extract_experiment_channels(self.exp_dir)

            # Standard measurements
            self.signals = ["area"]
            for ch in channel_names:
                self.signals.append(f"{ch}_mean")

            # Anticipate from instructions
            instr_path = os.path.join(
                self.exp_dir, "configs", f"measurement_instructions_{self.mode}.json"
            )
            if os.path.exists(instr_path):
                try:
                    with open(instr_path, "r") as f:
                        instr = json.load(f)

                    # 1. Features
                    features = instr.get("features", [])
                    if features:
                        for f_name in features:
                            if f_name == "intensity_mean":
                                continue  # handled by standard
                            if f_name == "area":
                                continue

                            # For other features, skimage/celldetective might suffix them.
                            # If it's a generic feature, skimage usually keeps the name.
                            # If it's multichannel, it might need channel names.
                            # For now, let's keep it simple as requested for intensity_mean and area.
                            pass

                    # 2. Isotropic measurements
                    radii = instr.get("intensity_measurement_radii", [])
                    ops = instr.get("isotropic_operations", [])
                    if radii and ops:
                        for r in radii if isinstance(radii, list) else [radii]:
                            for op in ops:
                                for ch in channel_names:
                                    if isinstance(r, list):
                                        self.signals.append(
                                            f"{ch}_ring_{int(min(r))}_{int(max(r))}_{op}"
                                        )
                                    else:
                                        self.signals.append(
                                            f"{ch}_circle_{int(r)}_{op}"
                                        )

                    # 3. Border distances
                    borders = instr.get("border_distances", [])
                    if borders:
                        for b in borders if isinstance(borders, list) else [borders]:
                            # Logic from measure.py for suffix
                            b_str = (
                                str(b)
                                .replace("(", "")
                                .replace(")", "")
                                .replace(", ", "_")
                                .replace(",", "_")
                            )
                            suffix = (
                                f"_slice_{b_str.replace('-', 'm')}px"
                                if ("-" in str(b) or "," in str(b))
                                else f"_edge_{b_str}px"
                            )
                            for ch in channel_names:
                                # In measure_features, it's {ch}_mean{suffix}
                                self.signals.append(f"{ch}_mean{suffix}")

                except Exception as e:
                    logger.warning(f"Could not parse measurement instructions: {e}")

            # Remove duplicates and keep order
            seen = set()
            self.signals = [x for x in self.signals if not (x in seen or seen.add(x))]

    def set_cellpose_scale(self):

        scale = (
            self.parent_window.PxToUm
            * float(self.diamWidget.diameter_le.get_threshold())
            / 30.0
        )
        if self.model_name == "CP_nuclei":
            scale = (
                self.parent_window.PxToUm
                * float(self.diamWidget.diameter_le.get_threshold())
                / 17.0
            )
        flow_thresh = self.diamWidget.flow_slider.value()
        cellprob_thresh = self.diamWidget.cellprob_slider.value()
        model_complete_path = locate_segmentation_model(self.model_name)
        input_config_path = model_complete_path + "config_input.json"
        new_channels = [
            self.diamWidget.cellpose_channel_cb[i].currentText() for i in range(2)
        ]
        with open(input_config_path) as config_file:
            input_config = json.load(config_file)

        input_config["spatial_calibration"] = scale
        input_config["channels"] = new_channels
        input_config["flow_threshold"] = flow_thresh
        input_config["cellprob_threshold"] = cellprob_thresh
        with open(input_config_path, "w") as f:
            json.dump(input_config, f, indent=4)

        self.cellpose_calibrated = True
        logger.info(f"model scale automatically computed: {scale}")
        self.diamWidget.close()
        self.process_population()

    def set_stardist_scale(self):

        model_complete_path = locate_segmentation_model(self.model_name)
        input_config_path = model_complete_path + "config_input.json"
        new_channels = [
            self.diamWidget.stardist_channel_cb[i].currentText()
            for i in range(len(self.diamWidget.stardist_channel_cb))
        ]
        with open(input_config_path) as config_file:
            input_config = json.load(config_file)

        input_config["channels"] = new_channels
        with open(input_config_path, "w") as f:
            json.dump(input_config, f, indent=4)

        self.stardist_calibrated = True
        self.diamWidget.close()
        self.process_population()

    def set_selected_channels_for_segmentation(self):

        model_complete_path = locate_segmentation_model(self.model_name)
        input_config_path = model_complete_path + "config_input.json"
        new_channels = [
            self.segChannelWidget.channel_cbs[i].currentText()
            for i in range(len(self.segChannelWidget.channel_cbs))
        ]
        target_cell_size = None
        if hasattr(self.segChannelWidget, "diameter_le"):
            target_cell_size = float(self.segChannelWidget.diameter_le.get_threshold())

        with open(input_config_path) as config_file:
            input_config = json.load(config_file)

        input_config.update(
            {"selected_channels": new_channels, "target_cell_size_um": target_cell_size}
        )

        # input_config['channels'] = new_channels
        with open(input_config_path, "w") as f:
            json.dump(input_config, f, indent=4)

        self.segChannelsSet = True
        self.segChannelWidget.close()
        self.process_population()

    def set_selected_signals_for_event_detection(self):
        self.signal_model_name = self.signal_models[
            self.signal_models_list.currentIndex()
        ]
        model_complete_path = locate_signal_model(self.signal_model_name)
        input_config_path = model_complete_path + "config_input.json"
        new_channels = [
            self.signalChannelWidget.channel_cbs[i].currentText()
            for i in range(len(self.signalChannelWidget.channel_cbs))
        ]
        with open(input_config_path) as config_file:
            input_config = json.load(config_file)

        input_config.update({"selected_channels": new_channels})

        # input_config['channels'] = new_channels
        with open(input_config_path, "w") as f:
            json.dump(input_config, f, indent=4)

        self.signalChannelsSet = True
        self.signalChannelWidget.close()
        self.process_population()
