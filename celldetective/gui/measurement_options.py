
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, \
	QGridLayout, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QDoubleValidator, QIntValidator

from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, \
	GeometryChoice, OperationChoice
from superqt import QLabeledDoubleSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

#from celldetective.gui.thresholds_gui import ThresholdSpot
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import load_frames, auto_load_number_of_frames
from celldetective.measure import compute_haralick_features
import numpy as np
from tifffile import imread
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from natsort import natsorted
from tifffile import imread
from pathlib import Path

from celldetective.gui.viewers import CellEdgeVisualizer, SpotDetectionVisualizer
from celldetective.gui.layouts import ProtocolDesignerLayout, BackgroundFitCorrectionLayout, LocalCorrectionLayout
from celldetective.gui.gui_utils import ThresholdLineEdit, PreprocessingLayout2
from celldetective.gui import Styles


class ConfigMeasurements(QMainWindow, Styles):
	"""
	UI to set measurement instructions.

	"""

	def __init__(self, parent_window=None):

		super().__init__()

		self.parent_window = parent_window
		self.setWindowTitle("Configure measurements")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective', 'icons', 'mexican-hat.png'])))
		self.mode = self.parent_window.mode
		self.exp_dir = self.parent_window.exp_dir
		self.background_correction = []
		if self.mode == "targets":
			self.config_name = "btrack_config_targets.json"
			self.measure_instructions_path = self.parent_window.exp_dir + "configs/measurement_instructions_targets.json"
		elif self.mode == "effectors":
			self.config_name = "btrack_config_effectors.json"
			self.measure_instructions_path = self.parent_window.exp_dir + "configs/measurement_instructions_effectors.json"
		self.soft_path = get_software_location()
		self.clear_previous = False

		exp_config = self.exp_dir + "config.ini"
		self.config_path = self.exp_dir + self.config_name
		self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
		self.channel_names = np.array(self.channel_names)
		self.channels = np.array(self.channels)

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		center_window(self)

		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()

		self.setMinimumWidth(500)
		self.setMinimumHeight(int(0.3 * self.screen_height))
		self.setMaximumHeight(int(0.8 * self.screen_height))
		self.populate_widget()
		self.load_previous_measurement_instructions()

	def populate_widget(self):

		"""
		Create the multibox design.

		"""

		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()
		main_layout = QVBoxLayout()
		self.button_widget.setLayout(main_layout)
		main_layout.setContentsMargins(30, 30, 30, 30)

		self.normalisation_frame = QFrame()
		self.normalisation_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		
		self.local_correction_layout = LocalCorrectionLayout(self)
		self.fit_correction_layout = BackgroundFitCorrectionLayout(self)
		
		self.protocol_layout = ProtocolDesignerLayout(parent_window=self,
													  tab_layouts=[ self.local_correction_layout, self.fit_correction_layout],
													  tab_names=['Local', 'Fit'],
													  title='BACKGROUND CORRECTION',
													  list_title='Corrections to apply:'
													  )
		
		self.normalisation_frame.setLayout(self.protocol_layout)

		#self.populate_normalisation_tabs()
		main_layout.addWidget(self.normalisation_frame)

		# first frame for FEATURES
		self.features_frame = QFrame()
		self.features_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_features_frame()
		main_layout.addWidget(self.features_frame)

		# second frame for ISOTROPIC MEASUREMENTS
		self.iso_frame = QFrame()
		self.iso_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_iso_frame()
		main_layout.addWidget(self.iso_frame)

		self.spot_detection_frame = QFrame()
		self.spot_detection_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_spot_detection()
		main_layout.addWidget(self.spot_detection_frame)

		self.clear_previous_btn = QCheckBox('clear previous measurements')
		main_layout.addWidget(self.clear_previous_btn)

		self.submit_btn = QPushButton('Save')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.write_instructions)
		main_layout.addWidget(self.submit_btn)

		# self.populate_left_panel()
		# grid.addLayout(self.left_side, 0, 0, 1, 1)
		self.button_widget.adjustSize()

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(self.button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		QApplication.processEvents()
		self.adjustScrollArea()

	def populate_iso_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.iso_frame)

		self.iso_lbl = QLabel("Position-based measurements".upper())
		self.iso_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.iso_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_iso_contents()
		grid.addWidget(self.ContentsIso, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def populate_features_frame(self):

		"""
		Add widgets and layout in the FEATURES frame.
		"""

		grid = QGridLayout(self.features_frame)

		self.feature_lbl = QLabel("Mask-based measurements".upper())
		self.feature_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.feature_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.generate_feature_panel_contents()
		grid.addWidget(self.ContentsFeatures, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def generate_iso_contents(self):

		self.ContentsIso = QFrame()
		layout = QVBoxLayout(self.ContentsIso)
		layout.setContentsMargins(0, 0, 0, 0)

		radii_layout = QHBoxLayout()
		self.radii_lbl = QLabel('Measurement radii (from center):')
		self.radii_lbl.setToolTip('Define radii or donughts for intensity measurements.')
		radii_layout.addWidget(self.radii_lbl, 90)

		self.del_radius_btn = QPushButton("")
		self.del_radius_btn.setStyleSheet(self.button_select_all)
		self.del_radius_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.del_radius_btn.setToolTip("Remove radius")
		self.del_radius_btn.setIconSize(QSize(20, 20))
		radii_layout.addWidget(self.del_radius_btn, 5)

		self.add_radius_btn = QPushButton("")
		self.add_radius_btn.setStyleSheet(self.button_select_all)
		self.add_radius_btn.setIcon(icon(MDI6.plus, color="black"))
		self.add_radius_btn.setToolTip("Add radius")
		self.add_radius_btn.setIconSize(QSize(20, 20))
		radii_layout.addWidget(self.add_radius_btn, 5)
		layout.addLayout(radii_layout)

		self.radii_list = ListWidget(GeometryChoice, initial_features=["10"], dtype=int)
		layout.addWidget(self.radii_list)

		self.del_radius_btn.clicked.connect(self.radii_list.removeSel)
		self.add_radius_btn.clicked.connect(self.radii_list.addItem)

		# Operation
		operation_layout = QHBoxLayout()
		self.op_lbl = QLabel('Operation to perform:')
		self.op_lbl.setToolTip('Set the operations to perform inside the ROI.')
		operation_layout.addWidget(self.op_lbl, 90)

		self.del_op_btn = QPushButton("")
		self.del_op_btn.setStyleSheet(self.button_select_all)
		self.del_op_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.del_op_btn.setToolTip("Remove operation")
		self.del_op_btn.setIconSize(QSize(20, 20))
		operation_layout.addWidget(self.del_op_btn, 5)

		self.add_op_btn = QPushButton("")
		self.add_op_btn.setStyleSheet(self.button_select_all)
		self.add_op_btn.setIcon(icon(MDI6.plus, color="black"))
		self.add_op_btn.setToolTip("Add operation")
		self.add_op_btn.setIconSize(QSize(20, 20))
		operation_layout.addWidget(self.add_op_btn, 5)
		layout.addLayout(operation_layout)

		self.operations_list = ListWidget(OperationChoice, initial_features=["mean"])
		layout.addWidget(self.operations_list)

		self.del_op_btn.clicked.connect(self.operations_list.removeSel)
		self.add_op_btn.clicked.connect(self.operations_list.addItem)

	def generate_feature_panel_contents(self):

		self.ContentsFeatures = QFrame()
		layout = QVBoxLayout(self.ContentsFeatures)
		layout.setContentsMargins(0, 0, 0, 0)

		feature_layout = QHBoxLayout()
		feature_layout.setContentsMargins(0, 0, 0, 0)

		self.feature_lbl = QLabel("Add features:")
		self.del_feature_btn = QPushButton("")
		self.del_feature_btn.setStyleSheet(self.button_select_all)
		self.del_feature_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.del_feature_btn.setToolTip("Remove feature")
		self.del_feature_btn.setIconSize(QSize(20, 20))

		self.add_feature_btn = QPushButton("")
		self.add_feature_btn.setStyleSheet(self.button_select_all)
		self.add_feature_btn.setIcon(icon(MDI6.filter_plus, color="black"))
		self.add_feature_btn.setToolTip("Add feature")
		self.add_feature_btn.setIconSize(QSize(20, 20))

		self.features_list = ListWidget(FeatureChoice, initial_features=['area', 'intensity_mean', ])

		self.del_feature_btn.clicked.connect(self.features_list.removeSel)
		self.add_feature_btn.clicked.connect(self.features_list.addItem)

		feature_layout.addWidget(self.feature_lbl, 90)
		feature_layout.addWidget(self.del_feature_btn, 5)
		feature_layout.addWidget(self.add_feature_btn, 5)
		layout.addLayout(feature_layout)
		layout.addWidget(self.features_list)

		self.feat_sep2 = QHSeperationLine()
		layout.addWidget(self.feat_sep2)

		contour_layout = QHBoxLayout()
		self.border_dist_lbl = QLabel('Contour measurements (from edge of mask):')
		self.border_dist_lbl.setToolTip(
			'Apply the intensity measurements defined above\nto a slice of each cell mask, defined as distance\nfrom the edge.')
		contour_layout.addWidget(self.border_dist_lbl, 90)

		self.del_contour_btn = QPushButton("")
		self.del_contour_btn.setStyleSheet(self.button_select_all)
		self.del_contour_btn.setIcon(icon(MDI6.trash_can, color="black"))
		self.del_contour_btn.setToolTip("Remove distance")
		self.del_contour_btn.setIconSize(QSize(20, 20))
		contour_layout.addWidget(self.del_contour_btn, 5)

		self.add_contour_btn = QPushButton("")
		self.add_contour_btn.setStyleSheet(self.button_select_all)
		self.add_contour_btn.setIcon(icon(MDI6.plus, color="black"))
		self.add_contour_btn.setToolTip("Add distance")
		self.add_contour_btn.setIconSize(QSize(20, 20))
		contour_layout.addWidget(self.add_contour_btn, 5)

		self.view_contour_btn = QPushButton("")
		self.view_contour_btn.setStyleSheet(self.button_select_all)
		self.view_contour_btn.setIcon(icon(MDI6.eye_plus_outline, color="black"))
		self.view_contour_btn.setToolTip("View contour")
		self.view_contour_btn.setIconSize(QSize(20, 20))
		contour_layout.addWidget(self.view_contour_btn, 5)

		layout.addLayout(contour_layout)

		self.contours_list = ListWidget(GeometryChoice, initial_features=[], dtype=int)
		layout.addWidget(self.contours_list)

		self.del_contour_btn.clicked.connect(self.contours_list.removeSel)
		self.add_contour_btn.clicked.connect(self.contours_list.addItem)
		self.view_contour_btn.clicked.connect(self.view_selected_contour)

		self.feat_sep3 = QHSeperationLine()
		layout.addWidget(self.feat_sep3)

		# Haralick features parameters
		self.activate_haralick_btn = QCheckBox('Measure Haralick texture features')
		self.activate_haralick_btn.toggled.connect(self.show_haralick_options)

		self.haralick_channel_choice = QComboBox()
		self.haralick_channel_choice.addItems(self.channel_names)
		self.haralick_channel_lbl = QLabel('Target channel: ')

		self.haralick_distance_le = QLineEdit("1")
		self.haralick_distance_lbl = QLabel('Distance: ')

		self.haralick_n_gray_levels_le = QLineEdit("256")
		self.haralick_n_gray_levels_lbl = QLabel('# gray levels: ')

		# Slider to set vmin & vmax
		self.haralick_scale_slider = QLabeledDoubleSlider()
		self.haralick_scale_slider.setSingleStep(0.05)
		self.haralick_scale_slider.setTickInterval(0.05)
		self.haralick_scale_slider.setSingleStep(1)
		self.haralick_scale_slider.setOrientation(1)
		self.haralick_scale_slider.setRange(0, 1)
		self.haralick_scale_slider.setValue(0.5)
		self.haralick_scale_lbl = QLabel('Scale: ')

		self.haralick_percentile_min_le = QLineEdit('0.01')
		self.haralick_percentile_max_le = QLineEdit('99.9')
		self.haralick_normalization_mode_btn = QPushButton()
		self.haralick_normalization_mode_btn.setStyleSheet(self.button_select_all)
		self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle, color="black"))
		self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
		self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
		self.percentile_mode = True

		min_percentile_hbox = QHBoxLayout()
		min_percentile_hbox.addWidget(self.haralick_percentile_min_le, 90)
		min_percentile_hbox.addWidget(self.haralick_normalization_mode_btn, 10)
		min_percentile_hbox.setContentsMargins(0, 0, 0, 0)

		self.haralick_percentile_min_lbl = QLabel('Min percentile: ')
		self.haralick_percentile_max_lbl = QLabel('Max percentile: ')

		self.haralick_hist_btn = QPushButton()
		self.haralick_hist_btn.clicked.connect(self.control_haralick_intensity_histogram)
		self.haralick_hist_btn.setIcon(icon(MDI6.poll, color="k"))
		self.haralick_hist_btn.setStyleSheet(self.button_select_all)

		self.haralick_digit_btn = QPushButton()
		self.haralick_digit_btn.clicked.connect(self.control_haralick_digitalization)
		self.haralick_digit_btn.setIcon(icon(MDI6.image_check, color="k"))
		self.haralick_digit_btn.setStyleSheet(self.button_select_all)

		self.haralick_layout = QVBoxLayout()
		self.haralick_layout.setContentsMargins(20, 20, 20, 20)

		activate_layout = QHBoxLayout()
		activate_layout.addWidget(self.activate_haralick_btn, 80)
		activate_layout.addWidget(self.haralick_hist_btn, 10)
		activate_layout.addWidget(self.haralick_digit_btn, 10)
		self.haralick_layout.addLayout(activate_layout)

		channel_layout = QHBoxLayout()
		channel_layout.addWidget(self.haralick_channel_lbl, 40)
		channel_layout.addWidget(self.haralick_channel_choice, 60)
		self.haralick_layout.addLayout(channel_layout)

		distance_layout = QHBoxLayout()
		distance_layout.addWidget(self.haralick_distance_lbl, 40)
		distance_layout.addWidget(self.haralick_distance_le, 60)
		self.haralick_layout.addLayout(distance_layout)

		gl_layout = QHBoxLayout()
		gl_layout.addWidget(self.haralick_n_gray_levels_lbl, 40)
		gl_layout.addWidget(self.haralick_n_gray_levels_le, 60)
		self.haralick_layout.addLayout(gl_layout)

		slider_layout = QHBoxLayout()
		slider_layout.addWidget(self.haralick_scale_lbl, 40)
		slider_layout.addWidget(self.haralick_scale_slider, 60)
		self.haralick_layout.addLayout(slider_layout)

		slider_min_percentile_layout = QHBoxLayout()
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_lbl, 40)
		# slider_min_percentile_layout.addWidget(self.haralick_percentile_min_le,55)
		slider_min_percentile_layout.addLayout(min_percentile_hbox, 60)
		# slider_min_percentile_layout.addWidget(self.haralick_normalization_mode_btn, 5)
		self.haralick_layout.addLayout(slider_min_percentile_layout)

		slider_max_percentile_layout = QHBoxLayout()
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_lbl, 40)
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_le, 60)
		self.haralick_layout.addLayout(slider_max_percentile_layout)

		self.haralick_to_hide = [self.haralick_hist_btn, self.haralick_digit_btn, self.haralick_channel_lbl,
								 self.haralick_channel_choice,
								 self.haralick_distance_le, self.haralick_distance_lbl, self.haralick_n_gray_levels_le,
								 self.haralick_n_gray_levels_lbl,
								 self.haralick_scale_lbl, self.haralick_scale_slider, self.haralick_percentile_min_lbl,
								 self.haralick_percentile_min_le,
								 self.haralick_percentile_max_lbl, self.haralick_percentile_max_le,
								 self.haralick_normalization_mode_btn]

		self.features_to_disable = [self.feature_lbl, self.del_feature_btn, self.add_feature_btn, self.features_list,
									self.activate_haralick_btn]

		self.activate_haralick_btn.setChecked(False)
		for f in self.haralick_to_hide:
			f.setEnabled(False)

		self.haralick_normalization_mode_btn.clicked.connect(self.switch_to_absolute_normalization_mode)
		layout.addLayout(self.haralick_layout)

	def switch_to_absolute_normalization_mode(self):

		if self.percentile_mode:
			self.percentile_mode = False
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle_outline, color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
			self.haralick_normalization_mode_btn.setToolTip("Switch to percentile normalization values.")
			self.haralick_percentile_min_lbl.setText('Min value: ')
			self.haralick_percentile_max_lbl.setText('Max value: ')
			self.haralick_percentile_min_le.setText('0')
			self.haralick_percentile_max_le.setText('10000')

		else:
			self.percentile_mode = True
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle, color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
			self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
			self.haralick_percentile_min_lbl.setText('Min percentile: ')
			self.haralick_percentile_max_lbl.setText('Max percentile: ')
			self.haralick_percentile_min_le.setText('0.01')
			self.haralick_percentile_max_le.setText('99.99')

	def show_haralick_options(self):

		"""
		Show the Haralick texture options.
		"""

		if self.activate_haralick_btn.isChecked():
			for element in self.haralick_to_hide:
				element.setEnabled(True)
		else:
			for element in self.haralick_to_hide:
				element.setEnabled(False)

	def adjustScrollArea(self):

		"""
		Auto-adjust scroll area to fill space
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def write_instructions(self):

		"""
		Write the selected options in a json file for later reading by the software.
		"""

		print(f"{self.spot_preprocessing.list.items=}")

		print('Writing instructions...')
		measurement_options = {}
		background_correction = self.protocol_layout.protocols
		if not background_correction:
			background_correction = None
		measurement_options.update({'background_correction': background_correction})
		features = self.features_list.getItems()
		if not features:
			features = None
		measurement_options.update({'features': features})

		border_distances = self.contours_list.getItems()
		if not border_distances:
			border_distances = None
		measurement_options.update({'border_distances': border_distances})

		self.extract_haralick_options()
		measurement_options.update({'haralick_options': self.haralick_options})

		intensity_measurement_radii = self.radii_list.getItems()
		if not intensity_measurement_radii:
			intensity_measurement_radii = None

		isotropic_operations = self.operations_list.getItems()
		if not isotropic_operations:
			isotropic_operations = None
			intensity_measurement_radii = None
		measurement_options.update({'intensity_measurement_radii': intensity_measurement_radii,
									'isotropic_operations': isotropic_operations})
		spot_detection = None
		if self.spot_check.isChecked():
			image_preprocessing = self.spot_preprocessing.list.items
			if image_preprocessing==[]:
				image_preprocessing = None
			spot_detection = {'channel': self.spot_channel.currentText(), 'diameter': float(self.diameter_value.text().replace(',','.')),
							  'threshold': float(self.threshold_value.text().replace(',','.')), 'image_preprocessing': image_preprocessing}
		measurement_options.update({'spot_detection': spot_detection})
		if self.clear_previous_btn.isChecked():
			self.clear_previous = True
		else:
			self.clear_previous = False
		measurement_options.update({'clear_previous': self.clear_previous})


		print('Measurement instructions: ', measurement_options)
		file_name = self.measure_instructions_path
		with open(file_name, 'w') as f:
			json.dump(measurement_options, f, indent=4)

		print('Done.')
		self.close()

	def extract_haralick_options(self):

		if self.activate_haralick_btn.isChecked():
			self.haralick_options = {"target_channel": self.haralick_channel_choice.currentIndex(),
									 "scale_factor": float(self.haralick_scale_slider.value()),
									 "n_intensity_bins": int(self.haralick_n_gray_levels_le.text()),
									 "distance": int(self.haralick_distance_le.text()),
									 }
			if self.percentile_mode:
				self.haralick_options.update({"percentiles": (
					float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text())),
					"clip_values": None})
			else:
				self.haralick_options.update({"percentiles": None, "clip_values": (
					float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text()))})

		else:
			self.haralick_options = None

	def load_previous_measurement_instructions(self):

		"""
		Read the measurmeent options from a previously written json file and format properly for the UI.
		"""

		print('Reading instructions..')
		if os.path.exists(self.measure_instructions_path):
			with open(self.measure_instructions_path, 'r') as f:
				measurement_instructions = json.load(f)
				print(measurement_instructions)
				if 'background_correction' in measurement_instructions:
					self.protocol_layout.protocols = measurement_instructions['background_correction']
					if self.protocol_layout.protocols is None:
						self.protocol_layout.protocols = []
					if (self.protocol_layout.protocols is not None) and len(self.protocol_layout.protocols) > 0:
						self.protocol_layout.protocol_list.clear()
						for norm_params in self.protocol_layout.protocols:
							normalisation_description = ""
							for index, (key, value) in enumerate(norm_params.items()):
								if index > 0:
									normalisation_description += ", "
								normalisation_description += str(key) + " : " + str(value)
							self.protocol_layout.protocol_list.addItem(normalisation_description)
					else:
						self.protocol_layout.protocol_list.clear()
				if 'features' in measurement_instructions:
					features = measurement_instructions['features']
					if (features is not None) and len(features) > 0:
						self.features_list.list_widget.clear()
						self.features_list.list_widget.addItems(features)
					else:
						self.features_list.list_widget.clear()

				if 'spot_detection' in measurement_instructions:
					spot_detection = measurement_instructions['spot_detection']
					if spot_detection is not None:
						self.spot_check.setChecked(True)
						if 'channel' in spot_detection:
							idx = spot_detection['channel']
							self.spot_channel.setCurrentText(idx)
						self.diameter_value.setText(str(spot_detection['diameter']))
						self.threshold_value.setText(str(spot_detection['threshold']))
						if 'image_preprocessing' in spot_detection:
							items = spot_detection['image_preprocessing']
							if items is not None:
								items_for_list = [a[0] for a in items]
								for it in items_for_list:
									self.spot_preprocessing.list.addItemToList(it)
								self.spot_preprocessing.list.items = items

				if 'border_distances' in measurement_instructions:
					border_distances = measurement_instructions['border_distances']
					if border_distances is not None:
						if isinstance(border_distances, int):
							distances = [border_distances]
						elif isinstance(border_distances, list):
							distances = []
							for d in border_distances:
								if isinstance(d, int) | isinstance(d, float):
									distances.append(str(int(d)))
								elif isinstance(d, list):
									distances.append(str(int(d[0])) + '-' + str(int(d[1])))
						self.contours_list.list_widget.clear()
						self.contours_list.list_widget.addItems(distances)

				if 'haralick_options' in measurement_instructions:
					haralick_options = measurement_instructions['haralick_options']
					if haralick_options is None:
						self.activate_haralick_btn.setChecked(False)
						self.show_haralick_options()
					else:
						self.activate_haralick_btn.setChecked(True)
						self.show_haralick_options()
						if 'target_channel' in haralick_options:
							idx = haralick_options['target_channel']
							self.haralick_channel_choice.setCurrentIndex(idx)
						if 'scale_factor' in haralick_options:
							self.haralick_scale_slider.setValue(float(haralick_options['scale_factor']))
						if ('percentiles' in haralick_options) and (haralick_options['percentiles'] is not None):
							perc = list(haralick_options['percentiles'])
							self.haralick_percentile_min_le.setText(str(perc[0]))
							self.haralick_percentile_max_le.setText(str(perc[1]))
						if ('clip_values' in haralick_options) and (haralick_options['clip_values'] is not None):
							values = list(haralick_options['clip_values'])
							self.percentile_mode = True
							self.switch_to_absolute_normalization_mode()
							self.haralick_percentile_min_le.setText(str(values[0]))
							self.haralick_percentile_max_le.setText(str(values[1]))
						if 'n_intensity_bins' in haralick_options:
							self.haralick_n_gray_levels_le.setText(str(haralick_options['n_intensity_bins']))
						if 'distance' in haralick_options:
							self.haralick_distance_le.setText(str(haralick_options['distance']))

				if 'intensity_measurement_radii' in measurement_instructions:
					intensity_measurement_radii = measurement_instructions['intensity_measurement_radii']
					if intensity_measurement_radii is not None:
						if isinstance(intensity_measurement_radii, int):
							radii = [intensity_measurement_radii]
						elif isinstance(intensity_measurement_radii, list):
							radii = []
							for r in intensity_measurement_radii:
								if isinstance(r, int) | isinstance(r, float):
									radii.append(str(int(r)))
								elif isinstance(r, list):
									radii.append(str(int(r[0])) + '-' + str(int(r[1])))
						self.radii_list.list_widget.clear()
						self.radii_list.list_widget.addItems(radii)
					else:
						self.radii_list.list_widget.clear()

				if 'isotropic_operations' in measurement_instructions:
					isotropic_operations = measurement_instructions['isotropic_operations']
					if (isotropic_operations is not None) and len(isotropic_operations) > 0:
						self.operations_list.list_widget.clear()
						self.operations_list.list_widget.addItems(isotropic_operations)
					else:
						self.operations_list.list_widget.clear()

				# if 'radial_intensity' in measurement_instructions:
				#     radial_intensity = measurement_instructions['radial_intensity']
				#     if radial_intensity is not None:
				#         self.radial_intensity_btn.setChecked(True)
				#         self.step_size.setText(str(radial_intensity['radial_step']))
				#         self.step_size.setEnabled(True)
				#         self.step_lbl.setEnabled(True)
				#         for checkbox in self.channel_chechkboxes:
				#             checkbox.setEnabled(True)
				#             if checkbox.text() in radial_intensity['radial_channels']:
				#                 checkbox.setChecked(True)

				if 'clear_previous' in measurement_instructions:
					self.clear_previous = measurement_instructions['clear_previous']
					self.clear_previous_btn.setChecked(self.clear_previous)

	def locate_image(self):

		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		movies = glob(self.parent_window.parent_window.pos + os.sep.join(['movie', f"{self.parent_window.movie_prefix}*.tif"]))

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
			self.stack_length = auto_load_number_of_frames(self.current_stack)
			self.mid_time = self.stack_length // 2
			indices = len(self.channel_names) * self.mid_time + np.arange(len(self.channel_names))
			self.test_frame = load_frames(list(indices.astype(int)),self.current_stack, normalize_input=False)

	def control_haralick_digitalization(self):

		"""
		Load an image for the first experiment movie found.
		Apply the Haralick parameters and check the result of the digitization (normalization + binning of intensities).

		"""

		self.locate_image() # pb here
		self.extract_haralick_options()
		if self.test_frame is not None:

			digitized_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]),
													  channels=self.channel_names, return_digit_image_only=True,
													  **self.haralick_options
													  )

			self.fig, self.ax = plt.subplots()
			divider = make_axes_locatable(self.ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)

			self.imshow_digit_window = FigureCanvas(self.fig, title="Haralick: control digitization")
			self.ax.clear()
			im = self.ax.imshow(digitized_img, cmap='gray')
			self.fig.colorbar(im, cax=cax, orientation='vertical')
			self.ax.set_xticks([])
			self.ax.set_yticks([])
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.imshow_digit_window.canvas.draw()
			self.imshow_digit_window.show()

	def control_haralick_intensity_histogram(self):

		"""
		Load an image for the first experiment movie found.
		Apply the Haralick normalization parameters and check the normalized intensity histogram.

		"""

		self.locate_image()
		self.extract_haralick_options()
		if self.test_frame is not None:
			norm_img = compute_haralick_features(self.test_frame, np.zeros(self.test_frame.shape[:2]),
												 channels=self.channel_names, return_norm_image_only=True,
												 **self.haralick_options
												 )
			self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
			self.hist_window = FigureCanvas(self.fig, title="Haralick: control digitized histogram")
			self.ax.clear()
			flat = norm_img.flatten()
			self.ax.hist(flat[flat==flat], bins=self.haralick_options['n_intensity_bins'])
			self.ax.set_xlabel('gray level value')
			self.ax.set_ylabel('#')
			plt.tight_layout()
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.hist_window.canvas.draw()
			self.hist_window.show()

	def view_selected_contour(self):

		"""
		Show the ROI for the selected contour measurement on experimental data.

		"""
		self.locate_image()

		if self.current_stack is not None:

			self.viewer = CellEdgeVisualizer(cell_type=self.mode,
											 stack_path=self.current_stack,
											 parent_list_widget=self.contours_list.list_widget,
											 n_channels=len(self.channel_names),
											 target_channel=0,
											 window_title='Set an edge measurement',
											 channel_cb=True,
											 channel_names = self.channel_names,
											 PxToUm = 1,
											 )
			self.viewer.show()

	def locate_mask(self):

		"""
		Load the first mask of the detected movie.
		"""

		labels_path = str(Path(self.current_stack).parent.parent) + os.sep+f'labels_{self.mode}'+os.sep
		masks = natsorted(glob(labels_path + '*.tif'))
		if len(masks) == 0:
			print('no mask found')
			self.test_mask = None
		else:
			self.test_mask = imread(masks[self.mid_time])

	def switch_channel_contour(self, value):

		"""
		Adjust intensity values when changing channels in the contour visualizer.

		"""

		self.im_contour.set_array(self.test_frame[:, :, value])
		self.im_contour.set_clim(vmin=np.percentile(self.test_frame[:, :, value].flatten(), 1),
								 vmax=np.percentile(self.test_frame[:, :, value].flatten(), 99.99))
		self.contrast_slider_contour.setRange(np.amin(self.test_frame[:, :, value]),
											  np.amax(self.test_frame[:, :, value]))
		self.contrast_slider_contour.setValue([np.percentile(self.test_frame[:, :, value].flatten(), 1),
											   np.percentile(self.test_frame[:, :, value].flatten(), 99.99)])
		self.fig_contour.canvas.draw_idle()

	def contrast_im_contour(self, value):
		vmin = value[0]
		vmax = value[1]
		self.im_contour.set_clim(vmin=vmin, vmax=vmax)
		self.fig_contour.canvas.draw_idle()

	def make_contour_transparent(self, value):

		self.im_mask.set_alpha(value)
		self.fig_contour.canvas.draw_idle()

	def remove_item_from_list(self):
		current_item = self.normalisation_list.currentRow()
		if current_item > -1:
			del self.background_correction[current_item]
			self.normalisation_list.takeItem(current_item)

	def check_the_information(self):
		
		if self.tabs.currentIndex() == 0:
			if self.background_correction  is None:
				self.background_correction  = []
			for index, normalisation_opt in enumerate(self.background_correction ):
				if self.tab1_channel_dropdown.currentText() in normalisation_opt['target channel']:
					result = self.channel_already_in_list()
					if result != QMessageBox.Yes:
						return False
					else:
						self.background_correction .remove(normalisation_opt)
						self.normalisation_list.takeItem(index)
						return True

	def display_message_box(self, missing_info):
		QMessageBox.about(self, "Message Box Title", "Please " + missing_info + " for background correction")

	def channel_already_in_list(self):
		response = QMessageBox.question(self, "Message Box Title",
										"The background correction parameters for this channel already exist, "
										"continuing will erase the previous configurations. Are you sure you want to "
										"proceed?",
										QMessageBox.No | QMessageBox.Yes, QMessageBox.No)
		return response

	def fun(self, x, y):
		return x ** 2 + y

	def view_normalisation_contour(self):

		"""
		Show the ROI for the selected contour measurement on experimental data.

		"""

		self.locate_image()

		if self.current_stack is not None:

			self.viewer = CellEdgeVisualizer(cell_type=self.mode,
											 stack_path=self.current_stack,
											 parent_le = self.tab1_txt_distance,
											 n_channels=len(self.channel_names),
											 target_channel=self.tab1_channel_dropdown.currentIndex(),
											 edge_range = (0,30),
											 initial_edge= self.tab1_txt_distance.get_threshold(),
											 invert=True,
											 window_title='Set an edge distance to estimate local intensity',
											 channel_cb=False,
											 PxToUm = 1,
											 )
			self.viewer.show()


	def populate_spot_detection(self):

		layout = QVBoxLayout(self.spot_detection_frame)

		self.spot_detection_lbl = QLabel("SPOT DETECTION")
		self.spot_detection_lbl.setStyleSheet("""font-weight: bold;padding: 0px;""")
		layout.addWidget(self.spot_detection_lbl, alignment=Qt.AlignCenter)

		perform_hbox = QHBoxLayout()
		self.spot_check = QCheckBox('Perform spot detection')
		self.spot_check.toggled.connect(self.enable_spot_detection)
		perform_hbox.addWidget(self.spot_check, 95)
		
		self.spot_viewer_btn = QPushButton()
		self.spot_viewer_btn.clicked.connect(self.spot_preview)
		self.spot_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
		self.spot_viewer_btn.setStyleSheet(self.button_select_all)
		self.spot_viewer_btn.setToolTip('Set detection parameters visually.')
		perform_hbox.addWidget(self.spot_viewer_btn, 5)
		layout.addLayout(perform_hbox)

		channel_hbox = QHBoxLayout()
		self.spot_channel_lbl = QLabel("Channel: ")
		self.spot_channel = QComboBox()
		self.spot_channel.addItems(self.channel_names)
		channel_hbox.addWidget(self.spot_channel_lbl, 30)
		channel_hbox.addWidget(self.spot_channel, 70)
		layout.addLayout(channel_hbox)

		self.spot_preprocessing = PreprocessingLayout2(fraction=30, parent_window=self)
		layout.addLayout(self.spot_preprocessing)

		# continue switching to VBox + HBox down
		diam_hbox = QHBoxLayout()
		self.diameter_lbl = QLabel('Spot diameter: ')
		self.diameter_value = QLineEdit()
		self.diameter_value.setValidator(self.onlyFloat)
		self.diameter_value.setText('7')
		self.diameter_value.textChanged.connect(self.enable_spot_preview)

		diam_hbox.addWidget(self.diameter_lbl,30)
		diam_hbox.addWidget(self.diameter_value, 70)
		layout.addLayout(diam_hbox)

		thresh_hbox = QHBoxLayout()
		self.threshold_lbl = QLabel('Spot threshold: ')
		self.threshold_value = QLineEdit()
		self.threshold_value.setValidator(self.onlyFloat)
		self.threshold_value.setText('0')
		self.threshold_value.textChanged.connect(self.enable_spot_preview)

		thresh_hbox.addWidget(self.threshold_lbl, 30)
		thresh_hbox.addWidget(self.threshold_value, 70)
		layout.addLayout(thresh_hbox)

		# #invert_layout = QHBoxLayout()
		# self.invert_check = QCheckBox('invert')
		# self.invert_value_le = QLineEdit('65535')
		# self.invert_value_le.setValidator(self.onlyFloat)
		# layout.addWidget(self.invert_check, 6, 0)
		# layout.addWidget(self.invert_value_le, 6, 1)
		# #layout.addLayout(invert_layout, 5, 1, 1, 1)

		self.spot_detection_widgets = [self.spot_channel, 
									   self.spot_channel_lbl,
									   self.diameter_value,
									   self.diameter_lbl, 
									   self.threshold_value,
									   self.threshold_lbl,
									   self.spot_viewer_btn,  
									   #self.invert_check, 
									   #self.invert_value_le, 
									   self.spot_preprocessing.list, 
									   self.spot_preprocessing.add_filter_btn, 
									   self.spot_preprocessing.delete_filter_btn, 
									   self.spot_preprocessing.preprocess_lbl
									  ]
		for wg in self.spot_detection_widgets:
			wg.setEnabled(False)

	def enable_spot_preview(self):

		diam = self.diameter_value.text().replace(',','').replace('.','')
		thresh = self.threshold_value.text().replace(',','').replace('.','')
		if diam.isnumeric() and thresh.isnumeric():
			self.spot_viewer_btn.setEnabled(True)
		else:
			self.spot_viewer_btn.setEnabled(False)

	def spot_preview(self):
		self.locate_image()
		if self.test_frame is not None:
			self.locate_mask()
			if self.test_mask is not None:

				# invert_value = self.invert_value_le.text().replace(',','.')
				# if invert_value != '':
				# 	invert_value = float(invert_value)
				# else:
				# 	invert_value = None

				self.spot_visual = SpotDetectionVisualizer(frame_slider=True,
														   contrast_slider=True,
														   cell_type=self.mode,
														   channel_cb=True,
														   channel_names = self.channel_names,
														   stack_path=self.current_stack,
														   n_channels=len(self.channel_names),
														   target_channel=self.spot_channel.currentIndex(),
														   window_title='Detect spots',
														   parent_channel_cb=self.spot_channel,
														   parent_diameter_le=self.diameter_value,
														   parent_threshold_le=self.threshold_value,
														   parent_preprocessing_list=self.spot_preprocessing.list,
														   #parent_invert_check=self.invert_check,
														   #invert = self.invert_check.isChecked(),
														   #invert_value = self.invert_value_le.text().replace(',','.'),
														   PxToUm = 1,)
				self.spot_visual.show()
				#self.spot_visual = ThresholdSpot(current_channel=self.spot_channel.currentIndex(), img=self.test_frame,
				#								 mask=self.test_mask, parent_window=self)

	def enable_spot_detection(self):

		if self.spot_check.isChecked():
			for wg in self.spot_detection_widgets:
				wg.setEnabled(True)
		else:
			for wg in self.spot_detection_widgets:
				wg.setEnabled(False)
