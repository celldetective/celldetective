from PyQt5.QtWidgets import QDialog, QFrame, QGridLayout, QComboBox, QListWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QCheckBox, \
	QMessageBox, QWidget
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from celldetective.gui.signal_annotator import MeasureAnnotator
from celldetective.gui.signal_annotator2 import SignalAnnotator2
from celldetective.io import get_segmentation_models_list, control_segmentation_napari, get_signal_models_list, \
	control_tracks, load_experiment_tables, get_pair_signal_models_list
from celldetective.io import locate_segmentation_model, extract_position_name, fix_missing_labels, auto_load_number_of_frames, load_frames, locate_signal_model
from celldetective.gui import SegmentationModelLoader, ClassifierWidget, ConfigNeighborhoods, ConfigSegmentationModelTraining, ConfigTracking, SignalAnnotator, ConfigSignalModelTraining, ConfigMeasurements, ConfigSignalAnnotator, TableUI
from celldetective.gui.gui_utils import QHSeperationLine
from celldetective.relative_measurements import rel_measure_at_position
from celldetective.segmentation import segment_at_position, segment_from_threshold_at_position
from celldetective.tracking import track_at_position
from celldetective.measure import measure_at_position
from celldetective.signals import analyze_signals_at_position, analyze_pair_signals_at_position
from celldetective.utils import extract_experiment_channels
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pandas as pd
from celldetective.gui.gui_utils import center_window
from tifffile import imwrite
import json
import psutil
from celldetective.neighborhood import compute_neighborhood_at_position, compute_contact_neighborhood_at_position
from celldetective.gui.gui_utils import FigureCanvas
from celldetective.preprocessing import correct_background_model_free, correct_background_model, correct_channel_offset
from celldetective.utils import _estimate_scale_factor, _extract_channel_indices_from_config, _extract_channel_indices, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel, normalize_per_channel
from celldetective.gui.gui_utils import ThresholdLineEdit, QuickSliderLayout, help_generic
from celldetective.gui.layouts import CellposeParamsWidget, StarDistParamsWidget, BackgroundModelFreeCorrectionLayout, ProtocolDesignerLayout, BackgroundFitCorrectionLayout, ChannelOffsetOptionsLayout
from celldetective.gui import Styles
from celldetective.utils import get_software_location

from celldetective.gui.workers import ProgressWindow
from celldetective.gui.processes.segment_cells import SegmentCellThresholdProcess, SegmentCellDLProcess
from celldetective.gui.processes.track_cells import TrackingProcess
from celldetective.gui.processes.measure_cells import MeasurementProcess

import time
import asyncio

class ProcessPanel(QFrame, Styles):
	def __init__(self, parent_window, mode):

		super().__init__()
		self.parent_window = parent_window
		self.mode = mode
		self.exp_channels = self.parent_window.exp_channels
		self.exp_dir = self.parent_window.exp_dir
		self.exp_config = self.parent_window.exp_config
		self.movie_prefix = self.parent_window.movie_prefix
		self.threshold_config_targets = None
		self.threshold_config_effectors = None
		self.wells = np.array(self.parent_window.wells,dtype=str)
		self.cellpose_calibrated = False
		self.stardist_calibrated = False
		self.use_gpu = self.parent_window.parent_window.use_gpu
		self.n_threads = self.parent_window.parent_window.n_threads

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)
		self.grid.setContentsMargins(5,5,5,5)
		self.generate_header()

	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"PROCESS {self.mode.upper()}   ")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		title_hbox = QHBoxLayout()
		self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.help_pop_btn = QPushButton()
		self.help_pop_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_pop_btn.setIconSize(QSize(20, 20))
		self.help_pop_btn.clicked.connect(self.help_population)
		self.help_pop_btn.setStyleSheet(self.button_select_all)
		self.help_pop_btn.setToolTip("Help.")
		#self.grid.addWidget(self.help_pop_btn, 0, 0, 1, 3, alignment=Qt.AlignRight)


		self.select_all_btn = QPushButton()
		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_all_btn.setIconSize(QSize(20, 20))
		self.all_ticked = False
		self.select_all_btn.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.button_select_all)
		#self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)

		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.button_select_all)
		#self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		title_hbox.addWidget(self.select_all_btn, 5)
		title_hbox.addWidget(QLabel(), 85, alignment=Qt.AlignCenter)
		title_hbox.addWidget(self.help_pop_btn, 5)
		title_hbox.addWidget(self.collapse_btn, 5)

		self.grid.addLayout(title_hbox, 0,0,1,4)
		self.populate_contents()

		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):

		effector_open = not self.parent_window.ProcessEffectors.ContentsFrame.isHidden()
		targets_open = not self.parent_window.ProcessTargets.ContentsFrame.isHidden()
		interactions_open = not self.parent_window.NeighPanel.ContentsFrame.isHidden()
		preprocessing_open = not self.parent_window.PreprocessingPanel.ContentsFrame.isHidden()
		is_open = np.array([effector_open, targets_open, interactions_open, preprocessing_open])

		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.parent_window.scroll.setMinimumHeight(int(550))
				self.parent_window.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent_window.scroll.setMinimumHeight(min(int(930), int(0.9*self.parent_window.screen_height)))


	def help_population(self):

		"""
		Helper to choose a proper cell population structure.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','cell-populations.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(suggestion)
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			

	def populate_contents(self):
		self.ContentsFrame = QFrame()
		self.ContentsFrame.setContentsMargins(5,5,5,5)
		self.grid_contents = QGridLayout(self.ContentsFrame)
		self.grid_contents.setContentsMargins(0,0,0,0)
		self.generate_segmentation_options()
		self.generate_tracking_options()
		self.generate_measure_options()
		self.generate_signal_analysis_options()

		self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
		self.view_tab_btn = QPushButton("Explore table")
		self.view_tab_btn.setStyleSheet(self.button_style_sheet_2)
		self.view_tab_btn.clicked.connect(self.view_table_ui)
		self.view_tab_btn.setToolTip('Explore table')
		self.view_tab_btn.setIcon(icon(MDI6.table,color="#1565c0"))
		self.view_tab_btn.setIconSize(QSize(20, 20))
		#self.view_tab_btn.setEnabled(False)
		self.grid_contents.addWidget(self.view_tab_btn, 10, 0, 1, 4)

		self.grid_contents.addWidget(QHSeperationLine(), 9, 0, 1, 4)
		self.submit_btn = QPushButton("Submit")
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.process_population)
		self.grid_contents.addWidget(self.submit_btn, 11, 0, 1, 4)

	def generate_measure_options(self):

		measure_layout = QHBoxLayout()

		self.measure_action = QCheckBox("MEASURE")
		self.measure_action.setStyleSheet(self.menu_check_style)

		self.measure_action.setIcon(icon(MDI6.eyedropper,color="black"))
		self.measure_action.setIconSize(QSize(20, 20))
		self.measure_action.setToolTip("Measure.")
		measure_layout.addWidget(self.measure_action, 90)
		#self.to_disable.append(self.measure_action_tc)

		self.classify_btn = QPushButton()
		self.classify_btn.setIcon(icon(MDI6.scatter_plot, color="black"))
		self.classify_btn.setIconSize(QSize(20, 20))
		self.classify_btn.setToolTip("Classify data.")
		self.classify_btn.setStyleSheet(self.button_select_all)
		self.classify_btn.clicked.connect(self.open_classifier_ui)
		measure_layout.addWidget(self.classify_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.check_measurements_btn=QPushButton()
		self.check_measurements_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_measurements_btn.setIconSize(QSize(20, 20))
		self.check_measurements_btn.setToolTip("Explore measurements in-situ.")
		self.check_measurements_btn.setStyleSheet(self.button_select_all)
		self.check_measurements_btn.clicked.connect(self.check_measurements)
		measure_layout.addWidget(self.check_measurements_btn, 5)


		self.measurements_config_btn = QPushButton()
		self.measurements_config_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.measurements_config_btn.setIconSize(QSize(20, 20))
		self.measurements_config_btn.setToolTip("Configure measurements.")
		self.measurements_config_btn.setStyleSheet(self.button_select_all)
		self.measurements_config_btn.clicked.connect(self.open_measurement_configuration_ui)
		measure_layout.addWidget(self.measurements_config_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.grid_contents.addLayout(measure_layout,5,0,1,4)

	def generate_signal_analysis_options(self):

		signal_layout = QVBoxLayout()
		signal_hlayout = QHBoxLayout()
		self.signal_analysis_action = QCheckBox("DETECT EVENTS")
		self.signal_analysis_action.setStyleSheet(self.menu_check_style)
		self.signal_analysis_action.setIcon(icon(MDI6.chart_bell_curve_cumulative,color="black"))
		self.signal_analysis_action.setIconSize(QSize(20, 20))
		self.signal_analysis_action.setToolTip("Detect events in single-cell signals.")
		self.signal_analysis_action.toggled.connect(self.enable_signal_model_list)
		signal_hlayout.addWidget(self.signal_analysis_action, 90)

		self.check_signals_btn = QPushButton()
		self.check_signals_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_signals_btn.setIconSize(QSize(20, 20))
		self.check_signals_btn.clicked.connect(self.check_signals)
		self.check_signals_btn.setToolTip("Explore signals in-situ.")
		self.check_signals_btn.setStyleSheet(self.button_select_all)
		signal_hlayout.addWidget(self.check_signals_btn, 6)

		self.config_signal_annotator_btn = QPushButton()
		self.config_signal_annotator_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.config_signal_annotator_btn.setIconSize(QSize(20, 20))
		self.config_signal_annotator_btn.setToolTip("Configure the dynamic visualizer.")
		self.config_signal_annotator_btn.setStyleSheet(self.button_select_all)
		self.config_signal_annotator_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
		signal_hlayout.addWidget(self.config_signal_annotator_btn, 6)

		#self.to_disable.append(self.measure_action_tc)
		signal_layout.addLayout(signal_hlayout)

		signal_model_vbox = QVBoxLayout()
		signal_model_vbox.setContentsMargins(25,0,25,0)

		model_zoo_layout = QHBoxLayout()
		model_zoo_layout.addWidget(QLabel("Model zoo:"),90)

		self.signal_models_list = QComboBox()
		self.signal_models_list.setEnabled(False)
		self.refresh_signal_models()
		#self.to_disable.append(self.cell_models_list)

		self.train_signal_model_btn = QPushButton("TRAIN")
		self.train_signal_model_btn.setToolTip("Train or retrain an event detection model\non newly annotated data.")
		self.train_signal_model_btn.setIcon(icon(MDI6.redo_variant,color='black'))
		self.train_signal_model_btn.setIconSize(QSize(20, 20))
		self.train_signal_model_btn.setStyleSheet(self.button_style_sheet_3)
		model_zoo_layout.addWidget(self.train_signal_model_btn, 5)
		self.train_signal_model_btn.clicked.connect(self.open_signal_model_config_ui)

		signal_model_vbox.addLayout(model_zoo_layout)
		signal_model_vbox.addWidget(self.signal_models_list)

		signal_layout.addLayout(signal_model_vbox)

		self.grid_contents.addLayout(signal_layout,6,0,1,4)

	def refresh_signal_models(self):
		signal_models = get_signal_models_list()
		self.signal_models_list.clear()
		self.signal_models_list.addItems(signal_models)

	def generate_tracking_options(self):

		grid_track = QHBoxLayout()

		self.track_action = QCheckBox("TRACK")
		self.track_action.setStyleSheet(self.menu_check_style)
		self.track_action.setIcon(icon(MDI6.chart_timeline_variant,color="black"))
		self.track_action.setIconSize(QSize(20, 20))
		self.track_action.setToolTip(f"Track the {self.mode[:-1]} cells.")
		grid_track.addWidget(self.track_action, 75)

		self.delete_tracks_btn = QPushButton()
		self.delete_tracks_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.delete_tracks_btn.setIconSize(QSize(20, 20))
		self.delete_tracks_btn.setToolTip("Delete existing tracks.")
		self.delete_tracks_btn.setStyleSheet(self.button_select_all)
		self.delete_tracks_btn.clicked.connect(self.delete_tracks)
		self.delete_tracks_btn.setEnabled(True)
		self.delete_tracks_btn.hide()
		grid_track.addWidget(self.delete_tracks_btn, 6)  #4,3,1,1, alignment=Qt.AlignLeft

		self.check_tracking_result_btn = QPushButton()
		self.check_tracking_result_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_tracking_result_btn.setIconSize(QSize(20, 20))
		self.check_tracking_result_btn.setToolTip("View tracking output in napari.")
		self.check_tracking_result_btn.setStyleSheet(self.button_select_all)
		self.check_tracking_result_btn.clicked.connect(self.open_napari_tracking)
		self.check_tracking_result_btn.setEnabled(False)
		grid_track.addWidget(self.check_tracking_result_btn, 6)  #4,3,1,1, alignment=Qt.AlignLeft

		self.track_config_btn = QPushButton()
		self.track_config_btn.setIcon(icon(MDI6.cog_outline,color="black"))
		self.track_config_btn.setIconSize(QSize(20, 20))
		self.track_config_btn.setToolTip("Configure tracking.")
		self.track_config_btn.setStyleSheet(self.button_select_all)
		self.track_config_btn.clicked.connect(self.open_tracking_configuration_ui)
		grid_track.addWidget(self.track_config_btn, 6) #4,2,1,1, alignment=Qt.AlignRight

		self.help_track_btn = QPushButton()
		self.help_track_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_track_btn.setIconSize(QSize(20, 20))
		self.help_track_btn.clicked.connect(self.help_tracking)
		self.help_track_btn.setStyleSheet(self.button_select_all)
		self.help_track_btn.setToolTip("Help.")
		grid_track.addWidget(self.help_track_btn, 6) #4,2,1,1, alignment=Qt.AlignRight

		self.grid_contents.addLayout(grid_track, 4, 0, 1,4)

	def delete_tracks(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Question)
		msgBox.setText("Do you want to erase the tracks? All subsequent annotations will be erased...")
		msgBox.setWindowTitle("Info")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		elif returnValue == QMessageBox.Yes:
			if os.path.exists(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_{self.mode}.csv'])):
				os.remove(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_{self.mode}.csv']))
			if os.path.exists(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_{self.mode}.pkl'])):
				os.remove(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_{self.mode}.pkl']))
			if os.path.exists(os.sep.join([self.parent_window.pos,'output','tables',f'napari_{self.mode[:-1]}_trajectories.npy'])):
				os.remove(os.sep.join([self.parent_window.pos,'output','tables',f'napari_{self.mode[:-1]}_trajectories.npy']))
			if os.path.exists(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_pairs.csv'])):
				os.remove(os.sep.join([self.parent_window.pos,'output','tables',f'trajectories_pairs.csv']))
			self.parent_window.update_position_options()
		else:
			return None

	def generate_segmentation_options(self):

		grid_segment = QHBoxLayout()
		grid_segment.setContentsMargins(0,0,0,0)
		grid_segment.setSpacing(0)

		self.segment_action = QCheckBox("SEGMENT")
		self.segment_action.setStyleSheet(self.menu_check_style)
		self.segment_action.setIcon(icon(MDI6.bacteria, color='black'))
		self.segment_action.setToolTip(f"Segment the {self.mode[:-1]} cells on the images.")
		self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)
		grid_segment.addWidget(self.segment_action, 90)

		self.check_seg_btn = QPushButton()
		self.check_seg_btn.setIcon(icon(MDI6.eye_check_outline,color="black"))
		self.check_seg_btn.setIconSize(QSize(20, 20))
		self.check_seg_btn.clicked.connect(self.check_segmentation)
		self.check_seg_btn.setStyleSheet(self.button_select_all)
		self.check_seg_btn.setToolTip("View segmentation output in napari.")
		grid_segment.addWidget(self.check_seg_btn, 5)

		self.help_seg_btn = QPushButton()
		self.help_seg_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_seg_btn.setIconSize(QSize(20, 20))
		self.help_seg_btn.clicked.connect(self.help_segmentation)
		self.help_seg_btn.setStyleSheet(self.button_select_all)
		self.help_seg_btn.setToolTip("Help.")
		grid_segment.addWidget(self.help_seg_btn, 5)
		self.grid_contents.addLayout(grid_segment, 0,0,1,4)

		seg_option_vbox = QVBoxLayout()
		seg_option_vbox.setContentsMargins(25,0,25,0)
		model_zoo_layout = QHBoxLayout()
		model_zoo_layout.addWidget(QLabel("Model zoo:"),90)
		self.seg_model_list = QComboBox()
		self.seg_model_list.currentIndexChanged.connect(self.reset_generalist_setup)
		#self.to_disable.append(self.tc_seg_model_list)
		self.seg_model_list.setGeometry(50, 50, 200, 30)
		self.init_seg_model_list()


		self.upload_model_btn = QPushButton("UPLOAD")
		self.upload_model_btn.setIcon(icon(MDI6.upload,color="black"))
		self.upload_model_btn.setIconSize(QSize(20, 20))
		self.upload_model_btn.setStyleSheet(self.button_style_sheet_3)
		self.upload_model_btn.setToolTip("Upload a new segmentation model\n(Deep learning or threshold-based).")
		model_zoo_layout.addWidget(self.upload_model_btn, 5)
		self.upload_model_btn.clicked.connect(self.upload_segmentation_model)
		# self.to_disable.append(self.upload_tc_model)

		self.train_btn = QPushButton("TRAIN")
		self.train_btn.setToolTip("Train or retrain a segmentation model\non newly annotated data.")
		self.train_btn.setIcon(icon(MDI6.redo_variant,color='black'))
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

	def help_segmentation(self):

		"""
		Widget with different decision helper decision trees.
		"""

		self.help_w = QWidget()
		self.help_w.setWindowTitle('Helper')
		layout = QVBoxLayout()
		seg_strategy_btn = QPushButton('A guide to choose a segmentation strategy.')
		seg_strategy_btn.setIcon(icon(MDI6.help_circle,color=self.celldetective_blue))
		seg_strategy_btn.setIconSize(QSize(40, 40))
		seg_strategy_btn.setStyleSheet(self.button_style_sheet_5)
		seg_strategy_btn.clicked.connect(self.help_seg_strategy)

		dl_strategy_btn = QPushButton('A guide to choose your Deep learning segmentation strategy.')
		dl_strategy_btn.setIcon(icon(MDI6.help_circle,color=self.celldetective_blue))
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

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','Threshold-vs-DL.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(f"The suggested technique is {suggestion}.\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/segment.html'>here</a>.")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

	def help_seg_dl_strategy(self):
		
		"""
		Helper for DL segmentation strategy, between pretrained models and custom models.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','DL-segmentation-strategy.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
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

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','tracking.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
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

		if not os.path.exists(os.sep.join([self.parent_window.pos,f'labels_{self.mode}', os.sep])):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("No labels can be found for this position. Do you want to annotate from scratch?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				os.mkdir(os.sep.join([self.parent_window.pos,f'labels_{self.mode}']))
				lbl = np.zeros((self.parent_window.shape_x, self.parent_window.shape_y), dtype=int)
				for i in range(self.parent_window.len_movie):
					imwrite(os.sep.join([self.parent_window.pos,f'labels_{self.mode}', str(i).zfill(4)+'.tif']), lbl)

		#self.freeze()
		#QApplication.setOverrideCursor(Qt.WaitCursor)
		test = self.parent_window.locate_selected_position()
		if test:
			print('Memory use: ', dict(psutil.virtual_memory()._asdict()))
			try:
				control_segmentation_napari(self.parent_window.pos, prefix=self.parent_window.movie_prefix, population=self.mode,flush_memory=True)
			except Exception as e:
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText(str(e))
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()

				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Question)
				msgBox.setText("Would you like to pass empty frames to fix the asymmetry?")
				msgBox.setWindowTitle("Question")
				msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					print('Fixing the missing labels...')
					fix_missing_labels(self.parent_window.pos, prefix=self.parent_window.movie_prefix,population=self.mode)
					try:
						control_segmentation_napari(self.parent_window.pos, prefix=self.parent_window.movie_prefix, population=self.mode,flush_memory=True)
					except Exception as e:
						print(f'Error {e}')
						return None
				else:
					return None

			gc.collect()

	def check_signals(self):

		test = self.parent_window.locate_selected_position()
		if test:
			self.SignalAnnotator = SignalAnnotator(self)
			self.SignalAnnotator.show()

	def check_measurements(self):

		test = self.parent_window.locate_selected_position()
		if test:
			self.MeasureAnnotator = MeasureAnnotator(self)
			self.MeasureAnnotator.show()

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
		self.seg_models = get_segmentation_models_list(mode=self.mode, return_path=False)
		thresh = 40
		self.models_truncated = [m[:thresh - 3]+'...' if len(m)>thresh else m for m in self.seg_models]
		#self.seg_model_list.addItems(models_truncated)

		self.seg_models_generic = get_segmentation_models_list(mode="generic", return_path=False)
		self.seg_models.append('Threshold')
		self.seg_models.extend(self.seg_models_generic)

		#self.seg_models_generic.insert(0,'Threshold')
		self.seg_model_list.addItems(self.seg_models)

		for i in range(len(self.seg_models)):
			self.seg_model_list.setItemData(i, self.seg_models[i], Qt.ToolTipRole)

		self.seg_model_list.insertSeparator(len(self.models_truncated))

	def tick_all_actions(self):
		self.switch_all_ticks_option()
		if self.all_ticked:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(20, 20))
			self.segment_action.setChecked(True)
		else:
			self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_all_btn.setIconSize(QSize(20, 20))
			self.segment_action.setChecked(False)

	def switch_all_ticks_option(self):
		if self.all_ticked == True:
			self.all_ticked = False
		else:
			self.all_ticked = True

	def upload_segmentation_model(self):
		print('Load a segmentation model or pipeline...')
		self.SegModelLoader = SegmentationModelLoader(self)
		self.SegModelLoader.show()

	def open_tracking_configuration_ui(self):
		print('Set the tracking parameters...')
		self.ConfigTracking = ConfigTracking(self)
		self.ConfigTracking.show()

	def open_signal_model_config_ui(self):
		print('Set the training parameters for new signal models...')
		self.ConfigSignalTrain = ConfigSignalModelTraining(self)
		self.ConfigSignalTrain.show()

	def open_segmentation_model_config_ui(self):
		print('Set the training parameters for a new segmentation model...')
		self.ConfigSegmentationTrain = ConfigSegmentationModelTraining(self)
		self.ConfigSegmentationTrain.show()

	def open_measurement_configuration_ui(self):
		print('Set the measurements to be performed...')
		self.ConfigMeasurements = ConfigMeasurements(self)
		self.ConfigMeasurements.show()

	def open_classifier_ui(self):

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

	def open_signal_annotator_configuration_ui(self):
		self.ConfigSignalAnnotator = ConfigSignalAnnotator(self)
		self.ConfigSignalAnnotator.show()

	def reset_generalist_setup(self, index):
		self.cellpose_calibrated = False
		self.stardist_calibrated = False

	def process_population(self):

		# if self.parent_window.well_list.currentText().startswith('Multiple'):
		# 	self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
		# else:
		
		self.well_index = self.parent_window.well_list.getSelectedIndices()
		if len(self.well_index)==0:
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

		print(f"Processing {self.parent_window.well_list.currentText()}...")

		# self.freeze()
		# QApplication.setOverrideCursor(Qt.WaitCursor)

		if self.mode=="targets":
			self.threshold_config = self.threshold_config_targets
		elif self.mode=="effectors":
			self.threshold_config = self.threshold_config_effectors

		self.load_available_tables()

		if self.df is not None and self.segment_action.isChecked():
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("Measurement tables have been found... Re-segmenting may create mismatches between the cell labels and the associated measurements. Do you want to erase the tables post-segmentation?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				pass
			elif returnValue == QMessageBox.Cancel:
				return None
			else:
				print('erase tabs!')
				tabs = [pos+os.sep.join(['output', 'tables', f'trajectories_{self.mode}.csv']) for pos in self.df_pos_info['pos_path'].unique()]
				for t in tabs:
					if os.path.exists(t.replace('.csv','.pkl')):
						os.remove(t.replace('.csv','.pkl'))
					os.remove(t)				

		loop_iter=0

		if self.parent_window.position_list.isMultipleSelection():
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("If you continue, several positions will be processed.\nDo you want to proceed?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None

		if self.seg_model_list.currentIndex() > len(self.models_truncated):
			self.model_name = self.seg_models[self.seg_model_list.currentIndex()-1]
		else:
			self.model_name = self.seg_models[self.seg_model_list.currentIndex()]
		print(self.model_name, self.seg_model_list.currentIndex())

		if self.segment_action.isChecked() and self.model_name.startswith('CP') and self.model_name in self.seg_models_generic and not self.cellpose_calibrated:

			self.diamWidget = CellposeParamsWidget(self, model_name=self.model_name)
			self.diamWidget.show()
			return None

		if self.segment_action.isChecked() and self.model_name.startswith('SD') and self.model_name in self.seg_models_generic and not self.stardist_calibrated:

			self.diamWidget = StarDistParamsWidget(self, model_name = self.model_name)
			self.diamWidget.show()
			return None

		self.movie_prefix = self.parent_window.movie_prefix

		for w_idx in self.well_index:

			pos = self.parent_window.positions[w_idx]
			pos_indices = self.parent_window.position_list.getSelectedIndices()
			#print(f"Processing position {self.parent_window.position_list.currentText()}...")

			well = self.parent_window.wells[w_idx]

			for pos_idx in pos_indices:

				self.pos = natsorted(glob(well+f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*/"))[pos_idx]
				print(f"Position {self.pos}...\nLoading stack movie...")
				self.pos_name = extract_position_name(self.pos)

				if not os.path.exists(self.pos + 'output/'):
					os.mkdir(self.pos + 'output/')
				if not os.path.exists(self.pos + 'output/tables/'):
					os.mkdir(self.pos + 'output/tables/')

				if self.segment_action.isChecked():

					if len(glob(os.sep.join([self.pos, f'labels_{self.mode}','*.tif'])))>0 and not self.parent_window.position_list.isMultipleSelection():
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Question)
						msgBox.setText("Labels have already been produced for this position. Do you want to segment again?")
						msgBox.setWindowTitle("Info")
						msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
						returnValue = msgBox.exec()
						if returnValue == QMessageBox.No:
							return None

					if (self.seg_model_list.currentText()=="Threshold"):
						if self.threshold_config is None:
							msgBox = QMessageBox()
							msgBox.setIcon(QMessageBox.Warning)
							msgBox.setText("Please set a threshold configuration from the upload menu first. Abort.")
							msgBox.setWindowTitle("Warning")
							msgBox.setStandardButtons(QMessageBox.Ok)
							returnValue = msgBox.exec()
							if returnValue == QMessageBox.Ok:
								return None
						else:
							print(f"Segmentation from threshold config: {self.threshold_config}")
							process_args = {"pos": self.pos, "mode": self.mode, "n_threads": self.n_threads, "threshold_instructions": self.threshold_config, "use_gpu": self.use_gpu}
							self.job = ProgressWindow(SegmentCellThresholdProcess, parent_window=self, title="Segment", process_args = process_args)
							result = self.job.exec_()
							if result == QDialog.Accepted:
								pass
							elif result == QDialog.Rejected:
								return None
							#segment_from_threshold_at_position(self.pos, self.mode, self.threshold_config, threads=self.parent_window.parent_window.n_threads)
					else:
						# model = locate_segmentation_model(self.model_name)
						# if model is None:
						# 	process = {"output_dir": self.output_dir, "file": self.model_name}
						# 	self.download_model_job = ProgressWindow(DownloadProcess, parent_window=self, title="Download", process_args = args)

						process_args = {"pos": self.pos, "mode": self.mode, "n_threads": self.n_threads, "model_name": self.model_name, "use_gpu": self.use_gpu}
						self.job = ProgressWindow(SegmentCellDLProcess, parent_window=self, title="Segment", process_args = process_args)
						result = self.job.exec_()
						if result == QDialog.Accepted:
							pass
						elif result == QDialog.Rejected:
							return None

				if self.track_action.isChecked():
					if os.path.exists(os.sep.join([self.pos, 'output', 'tables', f'trajectories_{self.mode}.csv'])) and not self.parent_window.position_list.isMultipleSelection():
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Question)
						msgBox.setText("A trajectory set already exists. Previously annotated data for\nthis position will be lost. Do you want to proceed?")
						msgBox.setWindowTitle("Info")
						msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
						returnValue = msgBox.exec()
						if returnValue == QMessageBox.No:
							return None

					process_args = {"pos": self.pos, "mode": self.mode, "n_threads": self.n_threads}
					self.job = ProgressWindow(TrackingProcess, parent_window=self, title="Tracking", process_args=process_args)
					result = self.job.exec_()
					if result == QDialog.Accepted:
						pass
					elif result == QDialog.Rejected:
						return None
					#track_at_position(self.pos, self.mode, threads=self.parent_window.parent_window.n_threads)

				if self.measure_action.isChecked():
					process_args = {"pos": self.pos, "mode": self.mode, "n_threads": self.n_threads}
					self.job = ProgressWindow(MeasurementProcess, parent_window=self, title="Measurement", process_args=process_args)
					result = self.job.exec_()
					if result == QDialog.Accepted:
						pass
					elif result == QDialog.Rejected:
						return None
					#measure_at_position(self.pos, self.mode, threads=self.parent_window.parent_window.n_threads)

				table = os.sep.join([self.pos, 'output', 'tables', f'trajectories_{self.mode}.csv'])
				if self.signal_analysis_action.isChecked() and os.path.exists(table):
					table = pd.read_csv(table)
					cols = list(table.columns)
					if 'class_color' in cols:
						colors = list(table['class_color'].to_numpy())
						if 'tab:orange' in colors or 'tab:cyan' in colors:
							if not self.parent_window.position_list.isMultipleSelection():
								msgBox = QMessageBox()
								msgBox.setIcon(QMessageBox.Question)
								msgBox.setText("The signals of the cells in the position appear to have been annotated... Do you want to proceed?")
								msgBox.setWindowTitle("Info")
								msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
								returnValue = msgBox.exec()
								if returnValue == QMessageBox.No:
									return None
					analyze_signals_at_position(self.pos, self.signal_models_list.currentText(), self.mode)


			# self.stack = None
		self.parent_window.update_position_options()
		for action in [self.segment_action, self.track_action, self.measure_action, self.signal_analysis_action]:
			if action.isChecked():
				action.setChecked(False)

		self.reset_generalist_setup(0)

	def open_napari_tracking(self):
		print(f'View the tracks before post-processing for position {self.parent_window.pos} in napari...')
		control_tracks(self.parent_window.pos, prefix=self.parent_window.movie_prefix, population=self.mode, threads=self.parent_window.parent_window.n_threads)

	def view_table_ui(self):

		print('Load table...')
		self.load_available_tables()

		if self.df is not None:
			plot_mode = 'plot_track_signals'
			if 'TRACK_ID' not in list(self.df.columns):
				plot_mode = 'static'
			self.tab_ui = TableUI(self.df, f"{self.parent_window.well_list.currentText()}; Position {self.parent_window.position_list.currentText()}", population=self.mode, plot_mode=plot_mode, save_inplace_option=True)
			self.tab_ui.show()
		else:
			print('Table could not be loaded...')
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table could be loaded...")
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

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=self.well_option, position_option=self.position_option, population=self.mode, return_pos_info=True)
		if self.df is None:
			print('No table could be found...')

	def set_cellpose_scale(self):

		scale = self.parent_window.PxToUm * float(self.diamWidget.diameter_le.get_threshold()) / 30.0
		if self.model_name=="CP_nuclei":
			scale = self.parent_window.PxToUm * float(self.diamWidget.diameter_le.get_threshold()) / 17.0
		flow_thresh = self.diamWidget.flow_slider.value()
		cellprob_thresh = self.diamWidget.cellprob_slider.value()
		model_complete_path = locate_segmentation_model(self.model_name)
		input_config_path = model_complete_path+"config_input.json"
		new_channels = [self.diamWidget.cellpose_channel_cb[i].currentText() for i in range(2)]
		print(new_channels)
		with open(input_config_path) as config_file:
			input_config = json.load(config_file)

		input_config['spatial_calibration'] = scale
		input_config['channels'] = new_channels
		input_config['flow_threshold'] = flow_thresh
		input_config['cellprob_threshold'] = cellprob_thresh
		with open(input_config_path, 'w') as f:
			json.dump(input_config, f, indent=4)

		self.cellpose_calibrated = True
		print('model scale automatically computed: ', scale)
		self.diamWidget.close()
		self.process_population()

	def set_stardist_scale(self):

		model_complete_path = locate_segmentation_model(self.model_name)
		input_config_path = model_complete_path+"config_input.json"
		new_channels = [self.diamWidget.stardist_channel_cb[i].currentText() for i in range(len(self.diamWidget.stardist_channel_cb))]
		with open(input_config_path) as config_file:
			input_config = json.load(config_file)

		input_config['channels'] = new_channels
		with open(input_config_path, 'w') as f:
			json.dump(input_config, f, indent=4)

		self.stardist_calibrated = True
		self.diamWidget.close()
		self.process_population()



class NeighPanel(QFrame, Styles):
	def __init__(self, parent_window):

		super().__init__()
		self.parent_window = parent_window
		self.exp_channels = self.parent_window.exp_channels
		self.exp_dir = self.parent_window.exp_dir
		self.wells = np.array(self.parent_window.wells,dtype=str)
		self.protocols = []
		self.mode='neighborhood'

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
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.populate_contents()

		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):

		effector_open = not self.parent_window.ProcessEffectors.ContentsFrame.isHidden()
		targets_open = not self.parent_window.ProcessTargets.ContentsFrame.isHidden()
		interactions_open = not self.parent_window.NeighPanel.ContentsFrame.isHidden()
		preprocessing_open = not self.parent_window.PreprocessingPanel.ContentsFrame.isHidden()
		is_open = np.array([effector_open, targets_open, interactions_open, preprocessing_open])

		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.parent_window.scroll.setMinimumHeight(int(550))
				self.parent_window.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent_window.scroll.setMinimumHeight(min(int(1000), int(0.9*self.parent_window.screen_height)))


	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)
		self.grid_contents.setContentsMargins(0,0,0,0)
		self.grid_contents.setSpacing(3)

		# Button to compute the neighborhoods
		neigh_option_hbox = QHBoxLayout()
		self.neigh_action = QCheckBox('NEIGHBORHOODS')
		self.neigh_action.setStyleSheet(self.menu_check_style)

		#self.neigh_action.setIcon(icon(MDI6.eyedropper, color="black"))
		#self.neigh_action.setIconSize(QSize(20, 20))
		self.neigh_action.setToolTip(
			"Compute neighborhoods in list below.")

		neigh_option_hbox.addWidget(self.neigh_action,90)


		self.help_neigh_btn = QPushButton()
		self.help_neigh_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_neigh_btn.setIconSize(QSize(20, 20))
		self.help_neigh_btn.clicked.connect(self.help_neighborhood)
		self.help_neigh_btn.setStyleSheet(self.button_select_all)
		self.help_neigh_btn.setToolTip("Help.")
		neigh_option_hbox.addWidget(self.help_neigh_btn,5,alignment=Qt.AlignRight)

		self.grid_contents.addLayout(neigh_option_hbox, 1,0,1,4)


		neigh_options_layout = QVBoxLayout()

		neigh_options_vbox = QVBoxLayout()

		# DISTANCE NEIGHBORHOOD
		dist_neigh_hbox = QHBoxLayout()
		dist_neigh_hbox.setContentsMargins(0,0,0,0)
		dist_neigh_hbox.setSpacing(0)

		self.dist_neigh_action = QLabel("ISOTROPIC DISTANCE THRESHOLD")
		self.dist_neigh_action.setStyleSheet(self.action_lbl_style_sheet)
		#self.dist_neigh_action.setIcon(icon(MDI6.circle_expand, color='black'))
		self.dist_neigh_action.setToolTip("")
		self.dist_neigh_action.setToolTip("Define an isotropic neighborhood between the center of mass\nof the cells, within a threshold distance.")
		#self.segment_action.toggled.connect(self.enable_segmentation_model_list)
		#self.to_disable.append(self.segment_action)

		self.config_distance_neigh_btn = QPushButton()
		self.config_distance_neigh_btn.setIcon(icon(MDI6.plus,color="black"))
		self.config_distance_neigh_btn.setIconSize(QSize(20, 20))
		self.config_distance_neigh_btn.setToolTip("Configure.")
		self.config_distance_neigh_btn.setStyleSheet(self.button_select_all)
		self.config_distance_neigh_btn.clicked.connect(self.open_config_distance_threshold_neighborhood)
		dist_neigh_hbox.addWidget(self.config_distance_neigh_btn,5)
		dist_neigh_hbox.addWidget(self.dist_neigh_action, 95)
		neigh_options_vbox.addLayout(dist_neigh_hbox)

		# CONTACT NEIGHBORHOOD
		contact_neighborhood_layout = QHBoxLayout()
		contact_neighborhood_layout.setContentsMargins(0,0,0,0)
		contact_neighborhood_layout.setSpacing(0)

		self.contact_neigh_action = QLabel("MASK CONTACT")
		self.contact_neigh_action.setToolTip("Identify touching cell masks, within a threshold edge distance.")
		self.contact_neigh_action.setStyleSheet(self.action_lbl_style_sheet)
		#self.contact_neigh_action.setIcon(icon(MDI6.transition_masked, color='black'))
		self.contact_neigh_action.setToolTip("")

		self.config_contact_neigh_btn = QPushButton()
		self.config_contact_neigh_btn.setIcon(icon(MDI6.plus,color="black"))
		self.config_contact_neigh_btn.setIconSize(QSize(20, 20))
		self.config_contact_neigh_btn.setToolTip("Configure.")
		self.config_contact_neigh_btn.setStyleSheet(self.button_select_all)
		self.config_contact_neigh_btn.clicked.connect(self.open_config_contact_neighborhood)
		contact_neighborhood_layout.addWidget(self.config_contact_neigh_btn,5)
		contact_neighborhood_layout.addWidget(self.contact_neigh_action, 95)
		neigh_options_vbox.addLayout(contact_neighborhood_layout)
		#self.grid_contents.addLayout(neigh_options_vbox, 2,0,1,4)

		#self.grid_contents.addWidget(QHSeperationLine(), 3, 0, 1, 4)

		self.delete_protocol_btn = QPushButton('')
		self.delete_protocol_btn.setStyleSheet(self.button_select_all)
		self.delete_protocol_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.delete_protocol_btn.setToolTip("Remove a neighborhood computation.")
		self.delete_protocol_btn.setIconSize(QSize(20, 20))
		self.delete_protocol_btn.clicked.connect(self.remove_protocol_from_list)

		self.protocol_list_lbl = QLabel('Neighborhoods to compute: ')
		self.protocol_list = QListWidget()
		self.protocol_list.setToolTip("Neighborhoods to compute sequentially.")

		list_header_layout = QHBoxLayout()
		list_header_layout.addWidget(self.protocol_list_lbl)
		list_header_layout.addWidget(self.delete_protocol_btn, alignment=Qt.AlignRight)
		#self.grid_contents.addLayout(list_header_layout, 4, 0, 1, 4)
		#self.grid_contents.addWidget(self.protocol_list, 5, 0, 1, 4)

		neigh_options_layout.addLayout(neigh_options_vbox)
		neigh_options_layout.addWidget(QHSeperationLine())
		neigh_options_layout.addLayout(list_header_layout)
		neigh_options_layout.addWidget(self.protocol_list)

		neigh_options_layout.setContentsMargins(30,5,30,5)
		neigh_options_layout.setSpacing(1)
		self.grid_contents.addLayout(neigh_options_layout, 5, 0, 1, 4)


		rel_layout = QHBoxLayout()
		self.measure_pairs_action = QCheckBox("MEASURE PAIRS")
		self.measure_pairs_action.setStyleSheet(self.menu_check_style)

		self.measure_pairs_action.setIcon(icon(MDI6.eyedropper, color="black"))
		self.measure_pairs_action.setIconSize(QSize(20, 20))
		self.measure_pairs_action.setToolTip("Measure the relative quantities defined for the cell pairs, for all neighborhoods.")
		rel_layout.addWidget(self.measure_pairs_action, 90)

		self.classify_pairs_btn = QPushButton()
		self.classify_pairs_btn.setIcon(icon(MDI6.scatter_plot, color="black"))
		self.classify_pairs_btn.setIconSize(QSize(20, 20))
		self.classify_pairs_btn.setToolTip("Classify data.")
		self.classify_pairs_btn.setStyleSheet(self.button_select_all)
		self.classify_pairs_btn.clicked.connect(self.open_classifier_ui_pairs)
		rel_layout.addWidget(self.classify_pairs_btn, 5) #4,2,1,1, alignment=Qt.AlignRight

		self.grid_contents.addLayout(rel_layout, 6, 0, 1, 4)

		signal_layout = QVBoxLayout()
		signal_hlayout = QHBoxLayout()
		self.signal_analysis_action = QCheckBox("DETECT PAIR EVENTS")
		self.signal_analysis_action.setStyleSheet(self.menu_check_style)

		self.signal_analysis_action.setIcon(icon(MDI6.chart_bell_curve_cumulative, color="black"))
		self.signal_analysis_action.setIconSize(QSize(20, 20))
		self.signal_analysis_action.setToolTip("Detect cell pair events using a DL model.")
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
		self.config_signal_annotator_btn.setToolTip("Configure the animation of the annotation tool.")
		self.config_signal_annotator_btn.setStyleSheet(self.button_select_all)
		self.config_signal_annotator_btn.clicked.connect(self.open_signal_annotator_configuration_ui)
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

		self.pair_train_signal_model_btn = QPushButton("TRAIN")
		self.pair_train_signal_model_btn.setToolTip("Train a cell pair event detection model.")
		self.pair_train_signal_model_btn.setIcon(icon(MDI6.redo_variant, color='black'))
		self.pair_train_signal_model_btn.setIconSize(QSize(20, 20))
		self.pair_train_signal_model_btn.setStyleSheet(self.button_style_sheet_3)
		pair_model_zoo_layout.addWidget(self.pair_train_signal_model_btn, 5)
		self.pair_train_signal_model_btn.clicked.connect(self.open_signal_model_config_ui)

		pair_signal_model_vbox.addLayout(pair_model_zoo_layout)
		pair_signal_model_vbox.addWidget(self.pair_signal_models_list)

		signal_layout.addLayout(pair_signal_model_vbox)
		self.grid_contents.addLayout(signal_layout, 7, 0, 1, 4)
		self.grid_contents.addWidget(QHSeperationLine(), 11, 0, 1, 4)

		self.view_tab_btn = QPushButton("Explore table")
		self.view_tab_btn.setStyleSheet(self.button_style_sheet_2)
		self.view_tab_btn.clicked.connect(self.view_table_ui)
		self.view_tab_btn.setToolTip('Explore table')
		self.view_tab_btn.setIcon(icon(MDI6.table,color="#1565c0"))
		self.view_tab_btn.setIconSize(QSize(20, 20))
		#self.view_tab_btn.setEnabled(False)
		self.grid_contents.addWidget(self.view_tab_btn, 12, 0, 1, 4)

		#self.grid_contents.addWidget(QLabel(''), 12, 0, 1, 4)

		self.submit_btn = QPushButton("Submit")
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.setToolTip("Compute the neighborhoods of the selected positions.")
		self.submit_btn.clicked.connect(self.process_neighborhood)
		self.grid_contents.addWidget(self.submit_btn, 14, 0, 1, 4)

		self.neigh_action.toggled.connect(self.activate_neigh_options)
		self.neigh_action.setChecked(True)
		self.neigh_action.setChecked(False)

	def open_classifier_ui_pairs(self):

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

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','neighborhood.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(f"{suggestion}\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/interactions.html#neighborhood'>here</a>.")
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

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=self.well_option, position_option=self.position_option, population="pairs", return_pos_info=True)
		if self.df is None:
			print('No table could be found...')


	def view_table_ui(self):

		print('Load table...')
		self.load_available_tables()

		if self.df is not None:
			plot_mode = 'static'
			self.tab_ui = TableUI(self.df, f"{self.parent_window.well_list.currentText()}; Position {self.parent_window.position_list.currentText()}", population='pairs', plot_mode=plot_mode, save_inplace_option=True)
			self.tab_ui.show()
		else:
			print('Table could not be loaded...')
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table could be loaded...")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None


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
		self.mode = 'pairs'
		self.ConfigSignalAnnotator = ConfigSignalAnnotator(self)
		self.ConfigSignalAnnotator.show()

	def open_signal_model_config_ui(self):
		self.ConfigSignalTrain = ConfigSignalModelTraining(self, signal_mode='pairs')
		self.ConfigSignalTrain.show()
		
	def remove_protocol_from_list(self):

		current_item = self.protocol_list.currentRow()
		if current_item > -1:
			del self.protocols[current_item]
			self.protocol_list.takeItem(current_item)

	def open_config_distance_threshold_neighborhood(self):

		self.ConfigNeigh = ConfigNeighborhoods(parent_window=self,
											   neighborhood_type='distance_threshold',
											   neighborhood_parameter_name='threshold distance',
											   )
		self.ConfigNeigh.show()

	def open_config_contact_neighborhood(self):

		self.ConfigNeigh = ConfigNeighborhoods(parent_window=self,
											   neighborhood_type='mask_contact',
											   neighborhood_parameter_name='tolerance contact distance',
											   )
		self.ConfigNeigh.show()

	def enable_signal_model_list(self):
		if self.signal_analysis_action.isChecked():
			self.pair_signal_models_list.setEnabled(True)
		else:
			self.pair_signal_models_list.setEnabled(False)

	def process_neighborhood(self):

		# if self.parent_window.well_list.currentText().startswith('Multiple'):
		# 	self.well_index = np.linspace(0,len(self.wells)-1,len(self.wells),dtype=int)
		# else:
		self.well_index = self.parent_window.well_list.getSelectedIndices()
		print(f"Processing well {self.parent_window.well_list.currentText()}...")

		# self.freeze()
		# QApplication.setOverrideCursor(Qt.WaitCursor)

		loop_iter=0

		if self.parent_window.position_list.isMultipleSelection():
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Question)
			msgBox.setText("If you continue, all positions will be processed.\nDo you want to proceed?")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None

		for w_idx in self.well_index:

			pos = self.parent_window.positions[w_idx]
			pos_indices = self.parent_window.position_list.getSelectedIndices()

			well = self.parent_window.wells[w_idx]

			for pos_idx in pos_indices:

				self.pos = natsorted(glob(well+f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*{os.sep}"))[pos_idx]
				print(f"Position {self.pos}...\nLoading stack movie...")

				if not os.path.exists(self.pos + 'output' + os.sep):
					os.mkdir(self.pos + 'output' + os.sep)
				if not os.path.exists(self.pos + os.sep.join(['output','tables'])+os.sep):
					os.mkdir(self.pos + os.sep.join(['output','tables'])+os.sep)

				if self.neigh_action.isChecked():
					for protocol in self.protocols:

						if protocol['neighborhood_type']=='distance_threshold':

							compute_neighborhood_at_position(self.pos,
															protocol['distance'],
															population=protocol['population'],
															theta_dist=None,
															img_shape=(self.parent_window.shape_x,self.parent_window.shape_y),
															return_tables=False,
															clear_neigh=protocol['clear_neigh'],
															event_time_col=protocol['event_time_col'],
															neighborhood_kwargs=protocol['neighborhood_kwargs'],
															)

						elif protocol['neighborhood_type']=='mask_contact':

							compute_contact_neighborhood_at_position(self.pos,
														protocol['distance'],
														population=protocol['population'],
														theta_dist=None,
														img_shape=(self.parent_window.shape_x,self.parent_window.shape_y),
														return_tables=False,
														clear_neigh=protocol['clear_neigh'],
														event_time_col=protocol['event_time_col'],
														neighborhood_kwargs=protocol['neighborhood_kwargs'],
														)
				if self.measure_pairs_action.isChecked():
					rel_measure_at_position(self.pos)

				if self.signal_analysis_action.isChecked():

					analyze_pair_signals_at_position(self.pos, self.pair_signal_models_list.currentText(), use_gpu=self.parent_window.parent_window.use_gpu)

		self.parent_window.update_position_options()
		print('Done.')

	def check_signals2(self):

		test = self.parent_window.locate_selected_position()
		if test:
			self.SignalAnnotator2 = SignalAnnotator2(self)
			self.SignalAnnotator2.show()

	def check_measurements2(self):

		test = self.parent_window.locate_selected_position()
		if test:
			self.MeasurementAnnotator2 = MeasureAnnotator2(self)
			self.MeasurementAnnotator2.show()



class PreprocessingPanel(QFrame, Styles):

	def __init__(self, parent_window):

		super().__init__()
		self.parent_window = parent_window
		self.exp_channels = self.parent_window.exp_channels
		self.exp_dir = self.parent_window.exp_dir
		self.wells = np.array(self.parent_window.wells,dtype=str)
		exp_config = self.exp_dir + "config.ini"
		self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
		self.channel_names = np.array(self.channel_names)
		self.background_correction = []
		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()

		self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.grid = QGridLayout(self)

		self.generate_header()

	def generate_header(self):

		"""
		Read the mode and prepare a collapsable block to process a specific cell population.

		"""

		panel_title = QLabel(f"PREPROCESSING")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		self.grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.select_all_btn = QPushButton()
		self.select_all_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_all_btn.setIconSize(QSize(20, 20))
		self.all_ticked = False
		#self.select_all_btn.clicked.connect(self.tick_all_actions)
		self.select_all_btn.setStyleSheet(self.button_select_all)
		self.grid.addWidget(self.select_all_btn, 0, 0, 1, 4, alignment=Qt.AlignLeft)
		#self.to_disable.append(self.all_tc_actions)

		self.collapse_btn = QPushButton()
		self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_btn.setIconSize(QSize(25, 25))
		self.collapse_btn.setStyleSheet(self.button_select_all)
		self.grid.addWidget(self.collapse_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.populate_contents()

		self.grid.addWidget(self.ContentsFrame, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_btn.clicked.connect(lambda: self.ContentsFrame.setHidden(not self.ContentsFrame.isHidden()))
		self.collapse_btn.clicked.connect(self.collapse_advanced)
		self.ContentsFrame.hide()

	def collapse_advanced(self):

		effector_open = not self.parent_window.ProcessEffectors.ContentsFrame.isHidden()
		targets_open = not self.parent_window.ProcessTargets.ContentsFrame.isHidden()
		interactions_open = not self.parent_window.NeighPanel.ContentsFrame.isHidden()
		preprocessing_open = not self.parent_window.PreprocessingPanel.ContentsFrame.isHidden()
		is_open = np.array([effector_open, targets_open, interactions_open, preprocessing_open])

		if self.ContentsFrame.isHidden():
			self.collapse_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.parent_window.scroll.setMinimumHeight(int(550))
				self.parent_window.adjustSize()
		else:
			self.collapse_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_btn.setIconSize(QSize(20, 20))
			self.parent_window.scroll.setMinimumHeight(min(int(930), int(0.9*self.parent_window.screen_height)))

	def populate_contents(self):

		self.ContentsFrame = QFrame()
		self.grid_contents = QGridLayout(self.ContentsFrame)

		self.model_free_correction_layout = BackgroundModelFreeCorrectionLayout(self)
		self.fit_correction_layout = BackgroundFitCorrectionLayout(self)

		self.protocol_layout = ProtocolDesignerLayout(parent_window=self,
											  tab_layouts=[self.fit_correction_layout, self.model_free_correction_layout],
											  tab_names=['Fit', 'Model-free'],
											  title='BACKGROUND CORRECTION',
											  list_title='Corrections to apply:')
		
		self.help_background_btn = QPushButton()
		self.help_background_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_background_btn.setIconSize(QSize(20, 20))
		self.help_background_btn.clicked.connect(self.help_background)
		self.help_background_btn.setStyleSheet(self.button_select_all)
		self.help_background_btn.setToolTip("Help.")

		self.protocol_layout.title_layout.addWidget(self.help_background_btn, 5, alignment=Qt.AlignRight)

		
		self.channel_offset_correction_layout = QVBoxLayout()

		self.channel_shift_lbl = QLabel("CHANNEL OFFSET CORRECTION")
		self.channel_shift_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		self.channel_offset_correction_layout.addWidget(self.channel_shift_lbl, alignment=Qt.AlignCenter)

		self.channel_offset_options_layout = ChannelOffsetOptionsLayout(self)
		self.channel_offset_correction_layout.addLayout(self.channel_offset_options_layout)
		
		self.protocol_layout.correction_layout.addWidget(QLabel(''))
		self.protocol_layout.correction_layout.addLayout(self.channel_offset_correction_layout)

		self.grid_contents.addLayout(self.protocol_layout,0,0,1,4)

		self.submit_preprocessing_btn = QPushButton("Submit")
		self.submit_preprocessing_btn.setStyleSheet(self.button_style_sheet)
		self.submit_preprocessing_btn.clicked.connect(self.launch_preprocessing)
		
		self.grid_contents.addWidget(self.submit_preprocessing_btn, 1,0,1,4)

	def add_offset_instructions_to_parent_list(self):
		print('adding instructions')


	def launch_preprocessing(self):

		msgBox1 = QMessageBox()
		msgBox1.setIcon(QMessageBox.Question)
		msgBox1.setText("Do you want to apply the preprocessing\nto all wells and positions?")
		msgBox1.setWindowTitle("Selection")
		msgBox1.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
		returnValue = msgBox1.exec()
		if returnValue == QMessageBox.Cancel:
			return None
		elif returnValue == QMessageBox.Yes:
			self.parent_window.well_list.selectAll()
			self.parent_window.position_list.selectAll()
		elif returnValue == QMessageBox.No:
			msgBox2 = QMessageBox()
			msgBox2.setIcon(QMessageBox.Question)
			msgBox2.setText("Do you want to apply the preprocessing\nto the positions selected at the top only?")
			msgBox2.setWindowTitle("Selection")
			msgBox2.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
			returnValue = msgBox2.exec()
			if returnValue == QMessageBox.Cancel:
				return None
			if returnValue == QMessageBox.No:
				return None

		print('Proceed with correction...')

		# if self.parent_window.well_list.currentText()=='*':
		# 	well_option = "*"
		# else:
		well_option = self.parent_window.well_list.getSelectedIndices()
		position_option = self.parent_window.position_list.getSelectedIndices()

		for k,correction_protocol in enumerate(self.protocol_layout.protocols):

			movie_prefix = None
			export_prefix = 'Corrected'
			if k>0:
				# switch source stack to cumulate multi-channel preprocessing
				movie_prefix = 'Corrected'
				export_prefix = None

			if correction_protocol['correction_type']=='model-free':
				print(f'Model-free correction; {movie_prefix=} {export_prefix=}')
				correct_background_model_free(self.exp_dir, 
								   well_option=well_option,
								   position_option=position_option,
								   export = True,
								   return_stacks=False,
								   show_progress_per_well = True,
								   show_progress_per_pos = True,
								   movie_prefix = movie_prefix,
								   export_prefix = export_prefix,
								   **correction_protocol,
								)
			
			elif correction_protocol['correction_type']=='fit':
				print(f'Fit correction; {movie_prefix=} {export_prefix=} {correction_protocol=}')
				correct_background_model(self.exp_dir,
								   well_option=well_option,
								   position_option=position_option,
								   export= True,
								   return_stacks=False,
								   show_progress_per_well = True,
								   show_progress_per_pos = True,
								   movie_prefix = movie_prefix,
								   export_prefix = export_prefix,
								   **correction_protocol,
								)
			elif correction_protocol['correction_type']=='offset':
				print(f'Offset correction; {movie_prefix=} {export_prefix=} {correction_protocol=}')
				correct_channel_offset(self.exp_dir,
								   well_option=well_option,
								   position_option=position_option,
								   export= True,
								   return_stacks=False,
								   show_progress_per_well = True,
								   show_progress_per_pos = True,
								   movie_prefix = movie_prefix,
								   export_prefix = export_prefix,
								   **correction_protocol,
								)
		print('Done.')


	def locate_image(self):

		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		print(f"{self.parent_window.pos}")
		movies = glob(self.parent_window.pos + os.sep.join(['movie', f"{self.parent_window.movie_prefix}*.tif"]))

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a position containing a movie...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.current_stack = None
				return None
		else:
			self.current_stack = movies[0]

	def help_background(self):

		"""
		Helper to choose a proper cell population structure.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','preprocessing.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(suggestion)
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			
