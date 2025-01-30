from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QMainWindow, QApplication, QMessageBox, QScrollArea, QComboBox, QFrame, QCheckBox, QFileDialog, QGridLayout, QTextEdit, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator

from celldetective.gui.gui_utils import center_window, FeatureChoice, ListWidget, QHSeperationLine, FigureCanvas, help_generic
from superqt import QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import extract_experiment_channels, get_software_location
from celldetective.io import interpret_tracking_configuration, load_frames
from celldetective.measure import compute_haralick_features
import numpy as np
import json
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob
from celldetective.gui import Styles

class ConfigTracking(QMainWindow, Styles):
	
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Configure tracking")
		self.mode = self.parent_window.mode
		self.exp_dir = self.parent_window.exp_dir
		self.floatValidator = QDoubleValidator()

		if self.mode=="targets":
			self.config_name = os.sep.join(["configs", "btrack_config_targets.json"])
			self.track_instructions_write_path = self.parent_window.exp_dir + os.sep.join(["configs","tracking_instructions_targets.json"])
		elif self.mode=="effectors":
			self.config_name = os.sep.join(["configs","btrack_config_effectors.json"])
			self.track_instructions_write_path = self.parent_window.exp_dir + os.sep.join(["configs", "tracking_instructions_effectors.json"])
		self.soft_path = get_software_location()
		
		exp_config = self.exp_dir +"config.ini"
		self.config_path = self.exp_dir + self.config_name
		self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
		self.channel_names = np.array(self.channel_names)
		self.channels = np.array(self.channels)
		self.screen_height = self.parent_window.parent_window.parent_window.screen_height

		center_window(self)
		self.setMinimumWidth(540)
		self.minimum_height = 300
		# self.setMinimumHeight(int(0.3*self.screen_height))
		# self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		self.load_previous_tracking_instructions()

	def populate_widget(self):

		"""
		Create the multibox design with collapsable frames.

		"""
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()
		main_layout = QVBoxLayout()
		self.button_widget.setLayout(main_layout)
		main_layout.setContentsMargins(30,30,30,30)

		# First collapsable Frame CONFIG

		self.btrack_option = QRadioButton('bTrack')
		self.btrack_option.setChecked(True)

		self.trackpy_option = QRadioButton('trackpy')
		self.tracker_option_group = QButtonGroup()
		self.tracker_option_group.addButton(self.btrack_option)
		self.tracker_option_group.addButton(self.trackpy_option)

		self.config_frame = QFrame()
		self.config_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_config_frame()

		tracker_hbox = QHBoxLayout()
		tracker_hbox.setContentsMargins(15,15,15,15)
		tracker_hbox.addWidget(self.btrack_option, 50, alignment=Qt.AlignCenter)
		tracker_hbox.addWidget(self.trackpy_option, 50, alignment=Qt.AlignCenter)
		main_layout.addLayout(tracker_hbox)

		main_layout.addWidget(self.config_frame)

		# Second collapsable frame FEATURES
		self.features_frame = QFrame()
		self.features_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_features_frame()
		main_layout.addWidget(self.features_frame)

		self.config_trackpy_frame = QFrame()
		self.config_trackpy_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_config_trackpy_frame()
		main_layout.addWidget(self.config_trackpy_frame)
		self.config_trackpy_frame.hide()

		# Third collapsable frame POST-PROCESSING
		self.post_proc_frame = QFrame()
		self.post_proc_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_post_proc_frame()
		main_layout.addWidget(self.post_proc_frame)

		self.submit_btn = QPushButton('Save')
		self.submit_btn.setStyleSheet(self.parent_window.parent_window.parent_window.button_style_sheet)
		self.submit_btn.clicked.connect(self.write_instructions)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)
		self.button_widget.adjustSize()

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(self.button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		self.btrack_option.toggled.connect(self.show_tracking_options)
		self.trackpy_option.toggled.connect(self.show_tracking_options)

		QApplication.processEvents()
		self.adjustScrollArea()

	def show_tracking_options(self):

		if self.btrack_option.isChecked():
			self.config_frame.show()
			self.features_frame.show()
			self.config_trackpy_frame.hide()
			#self.scroll_area.setMinimumHeight(min(int(930), int(0.9*self.screen_height)))
			#self.adjustSize()
		else:
			self.config_frame.hide()
			self.features_frame.hide()
			self.config_trackpy_frame.show()
			#self.scroll_area.setMinimumHeight(self.minimum_height)
			#self.adjustSize()


	def populate_post_proc_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.post_proc_frame)

		self.select_post_proc_btn = QPushButton()
		self.select_post_proc_btn.clicked.connect(self.activate_post_proc_options)
		self.select_post_proc_btn.setStyleSheet(self.button_select_all)

		self.post_proc_lbl = QLabel("POST-PROCESSING")
		self.post_proc_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.post_proc_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		title_hbox = QHBoxLayout()

		self.collapse_post_proc_btn = QPushButton()
		self.collapse_post_proc_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_post_proc_btn.setIconSize(QSize(20, 20))
		self.collapse_post_proc_btn.setStyleSheet(self.button_select_all)
		#grid.addWidget(self.collapse_post_proc_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.help_post_btn = QPushButton()
		self.help_post_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_post_btn.setIconSize(QSize(20, 20))
		self.help_post_btn.clicked.connect(self.help_post)
		self.help_post_btn.setStyleSheet(self.button_select_all)
		self.help_post_btn.setToolTip("Help.")

		title_hbox.addWidget(self.select_post_proc_btn, 5)
		title_hbox.addWidget(QLabel(), 85, alignment=Qt.AlignCenter)
		title_hbox.addWidget(self.help_post_btn, 5)
		title_hbox.addWidget(self.collapse_post_proc_btn, 5)
		grid.addLayout(title_hbox, 0,0,1,4)

		self.generate_post_proc_panel_contents()
		grid.addWidget(self.ContentsPostProc, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_post_proc_btn.clicked.connect(lambda: self.ContentsPostProc.setHidden(not self.ContentsPostProc.isHidden()))
		self.collapse_post_proc_btn.clicked.connect(self.collapse_post_advanced)
		self.ContentsPostProc.hide()
		self.uncheck_post_proc()

	def collapse_post_advanced(self):

		features_open = not self.ContentsFeatures.isHidden()
		config_open = not self.ContentsConfig.isHidden()
		post_open = not self.ContentsPostProc.isHidden()
		is_open = np.array([features_open, config_open, post_open])

		if self.ContentsPostProc.isHidden():
			self.collapse_post_proc_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_post_proc_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.scroll_area.setMinimumHeight(int(self.minimum_height))
				self.adjustSize()
		else:
			self.collapse_post_proc_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_post_proc_btn.setIconSize(QSize(20, 20))
			self.scroll_area.setMinimumHeight(min(int(930), int(0.9*self.screen_height)))


	def help_post(self):
		
		"""
		Helper for track post-processing strategy.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','track-postprocessing.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(rf"{suggestion}")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None		

	def help_feature(self):
		
		"""
		Helper for track post-processing strategy.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','feature-btrack.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(rf"{suggestion}")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

	def populate_features_frame(self):

		"""
		Add widgets and layout in the FEATURES frame.
		"""

		grid = QGridLayout(self.features_frame)
		title_hbox = QHBoxLayout()
		
		self.select_features_btn = QPushButton()
		self.select_features_btn.clicked.connect(self.activate_feature_options)
		self.select_features_btn.setStyleSheet(self.button_select_all)

		self.feature_lbl = QLabel("FEATURES")
		self.feature_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.feature_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_features_btn = QPushButton()
		self.collapse_features_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_features_btn.setIconSize(QSize(20, 20))
		self.collapse_features_btn.setStyleSheet(self.button_select_all)

		self.help_feature_btn = QPushButton()
		self.help_feature_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_feature_btn.setIconSize(QSize(20, 20))
		self.help_feature_btn.clicked.connect(self.help_feature)
		self.help_feature_btn.setStyleSheet(self.button_select_all)
		self.help_feature_btn.setToolTip("Help.")

		title_hbox.addWidget(self.select_features_btn, 5)
		title_hbox.addWidget(QLabel(), 85, alignment=Qt.AlignCenter)
		title_hbox.addWidget(self.help_feature_btn, 5)
		title_hbox.addWidget(self.collapse_features_btn, 5)
		grid.addLayout(title_hbox, 0,0,1,4)		

		self.generate_feature_panel_contents()
		grid.addWidget(self.ContentsFeatures, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_features_btn.clicked.connect(lambda: self.ContentsFeatures.setHidden(not self.ContentsFeatures.isHidden()))
		self.collapse_features_btn.clicked.connect(self.collapse_features_advanced)
		#self.ContentsFeatures.hide()
		self.check_features()

	def collapse_features_advanced(self):

		"""
		Switch the chevron icon and adjust the size for the FEATURES frame.
		"""

		features_open = not self.ContentsFeatures.isHidden()
		config_open = not self.ContentsConfig.isHidden()
		post_open = not self.ContentsPostProc.isHidden()
		is_open = np.array([features_open, config_open, post_open])

		if self.ContentsFeatures.isHidden():
			self.collapse_features_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_features_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.scroll_area.setMinimumHeight(int(self.minimum_height))
				self.adjustSize()
		else:
			self.collapse_features_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_features_btn.setIconSize(QSize(20, 20))
			self.scroll_area.setMinimumHeight(min(int(930), int(0.9*self.screen_height)))


	def generate_post_proc_panel_contents(self):

		self.ContentsPostProc = QFrame()
		layout = QVBoxLayout(self.ContentsPostProc)
		layout.setContentsMargins(0,0,0,0)

		post_proc_layout  = QHBoxLayout()
		self.post_proc_lbl = QLabel("Post processing on the tracks:")
		post_proc_layout.addWidget(self.post_proc_lbl, 90)
		layout.addLayout(post_proc_layout)

		clean_traj_sublayout = QVBoxLayout()
		clean_traj_sublayout.setContentsMargins(15,15,15,15)

		tracklength_layout = QHBoxLayout()
		self.min_tracklength_slider = QLabeledSlider()
		self.min_tracklength_slider.setSingleStep(1)
		self.min_tracklength_slider.setTickInterval(1)
		self.min_tracklength_slider.setSingleStep(1)
		self.min_tracklength_slider.setOrientation(1)
		self.min_tracklength_slider.setRange(0,self.parent_window.parent_window.len_movie)
		self.min_tracklength_slider.setValue(0)
		tracklength_layout.addWidget(QLabel('Min. tracklength: '),40)
		tracklength_layout.addWidget(self.min_tracklength_slider, 60)
		clean_traj_sublayout.addLayout(tracklength_layout)

		self.remove_not_in_first_checkbox = QCheckBox('Remove tracks that do not start at the beginning')
		self.remove_not_in_first_checkbox.setIcon(icon(MDI6.arrow_expand_right,color="k"))
		clean_traj_sublayout.addWidget(self.remove_not_in_first_checkbox)

		self.remove_not_in_last_checkbox = QCheckBox('Remove tracks that do not end at the end')
		self.remove_not_in_last_checkbox.setIcon(icon(MDI6.arrow_expand_left,color="k"))
		clean_traj_sublayout.addWidget(self.remove_not_in_last_checkbox)

		self.interpolate_gaps_checkbox = QCheckBox('Interpolate missed detections within tracks')
		self.interpolate_gaps_checkbox.setIcon(icon(MDI6.chart_timeline_variant_shimmer,color="k"))
		clean_traj_sublayout.addWidget(self.interpolate_gaps_checkbox)

		self.extrapolate_post_checkbox = QCheckBox('Sustain last position until the end of the movie')
		self.extrapolate_post_checkbox.setIcon(icon(MDI6.repeat,color="k"))

		self.extrapolate_pre_checkbox = QCheckBox('Sustain first position from the beginning of the movie')
		self.extrapolate_pre_checkbox.setIcon(icon(MDI6.repeat,color="k"))

		clean_traj_sublayout.addWidget(self.extrapolate_post_checkbox)
		clean_traj_sublayout.addWidget(self.extrapolate_pre_checkbox)

		# self.interpolate_na_features_checkbox = QCheckBox('Interpolate features of missed detections')
		# self.interpolate_na_features_checkbox.setIcon(icon(MDI6.format_color_fill,color="k"))

		# clean_traj_sublayout.addWidget(self.interpolate_na_features_checkbox)
		clean_traj_sublayout.addStretch()

		self.post_proc_options_to_disable = [self.post_proc_lbl, self.min_tracklength_slider, self.remove_not_in_first_checkbox,
											self.remove_not_in_last_checkbox, self.interpolate_gaps_checkbox, self.extrapolate_post_checkbox,
											self.extrapolate_pre_checkbox] #, self.interpolate_na_features_checkbox

		layout.addLayout(clean_traj_sublayout)


	def generate_feature_panel_contents(self):
		
		self.ContentsFeatures = QFrame()
		layout = QVBoxLayout(self.ContentsFeatures)
		layout.setContentsMargins(0,0,0,0)

		feature_layout = QHBoxLayout()
		feature_layout.setContentsMargins(0,0,0,0)


		self.feature_lbl = QLabel("Add features:")
		self.del_feature_btn = QPushButton("")
		self.del_feature_btn.setStyleSheet(self.button_select_all)
		self.del_feature_btn.setIcon(icon(MDI6.trash_can,color="black"))
		self.del_feature_btn.setToolTip("Remove feature")
		self.del_feature_btn.setIconSize(QSize(20, 20))

		self.add_feature_btn = QPushButton("")
		self.add_feature_btn.setStyleSheet(self.button_select_all)
		self.add_feature_btn.setIcon(icon(MDI6.filter_plus,color="black"))
		self.add_feature_btn.setToolTip("Add feature")
		self.add_feature_btn.setIconSize(QSize(20, 20))		

		self.features_list = ListWidget(FeatureChoice, initial_features=['area','intensity_mean',])

		self.del_feature_btn.clicked.connect(self.features_list.removeSel)
		self.add_feature_btn.clicked.connect(self.features_list.addItem)

		feature_layout.addWidget(self.feature_lbl, 90)
		feature_layout.addWidget(self.del_feature_btn, 5)
		feature_layout.addWidget(self.add_feature_btn, 5)
		layout.addLayout(feature_layout)
		layout.addWidget(self.features_list)

		self.feat_sep2 = QHSeperationLine()
		layout.addWidget(self.feat_sep2)

		self.use_channel_lbl = QLabel('Use channel:')
		layout.addWidget(self.use_channel_lbl)
		self.mask_channels_cb = [QCheckBox() for i in range(len(self.channels))]
		for cb,chn in zip(self.mask_channels_cb,self.channel_names):
			cb.setText(chn)
			cb.setChecked(True)
			layout.addWidget(cb)

		self.feat_sep1 = QHSeperationLine()
		layout.addWidget(self.feat_sep1)

		self.activate_haralick_btn = QCheckBox('activate Haralick texture features')
		self.activate_haralick_btn.toggled.connect(self.show_haralick_options)
		# Haralick features parameters

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
		self.haralick_scale_slider.setRange(0,1)
		self.haralick_scale_slider.setValue(0.5)
		self.haralick_scale_lbl = QLabel('Scale: ')

		self.haralick_percentile_min_le = QLineEdit('0.01')
		self.haralick_percentile_max_le = QLineEdit('99.9')
		self.haralick_normalization_mode_btn = QPushButton()
		self.haralick_normalization_mode_btn.setStyleSheet(self.button_select_all)
		self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle,color="black"))
		self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))		
		self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")
		self.percentile_mode = True

		self.haralick_percentile_min_lbl = QLabel('Min percentile: ')
		self.haralick_percentile_max_lbl = QLabel('Max percentile: ')

		self.haralick_hist_btn = QPushButton()
		self.haralick_hist_btn.clicked.connect(self.control_haralick_intensity_histogram)
		self.haralick_hist_btn.setIcon(icon(MDI6.poll,color="k"))
		self.haralick_hist_btn.setStyleSheet(self.button_select_all)

		self.haralick_digit_btn = QPushButton()
		self.haralick_digit_btn.clicked.connect(self.control_haralick_digitalization)
		self.haralick_digit_btn.setIcon(icon(MDI6.image_check,color="k"))
		self.haralick_digit_btn.setStyleSheet(self.button_select_all)

		self.haralick_layout = QVBoxLayout()
		self.haralick_layout.setContentsMargins(20,20,20,20)

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
		distance_layout.addWidget(self.haralick_distance_lbl,40)
		distance_layout.addWidget(self.haralick_distance_le, 60)
		self.haralick_layout.addLayout(distance_layout)

		gl_layout = QHBoxLayout()
		gl_layout.addWidget(self.haralick_n_gray_levels_lbl,40)
		gl_layout.addWidget(self.haralick_n_gray_levels_le,60)
		self.haralick_layout.addLayout(gl_layout)

		slider_layout = QHBoxLayout()
		slider_layout.addWidget(self.haralick_scale_lbl,40)
		slider_layout.addWidget(self.haralick_scale_slider,60)
		self.haralick_layout.addLayout(slider_layout)

		slider_min_percentile_layout = QHBoxLayout()
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_lbl,40)
		slider_min_percentile_layout.addWidget(self.haralick_percentile_min_le,55)
		slider_min_percentile_layout.addWidget(self.haralick_normalization_mode_btn, 5)
		self.haralick_layout.addLayout(slider_min_percentile_layout)

		slider_max_percentile_layout = QHBoxLayout()
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_lbl,40)
		slider_max_percentile_layout.addWidget(self.haralick_percentile_max_le,60)
		self.haralick_layout.addLayout(slider_max_percentile_layout)

		self.haralick_to_hide = [self.haralick_hist_btn, self.haralick_digit_btn, self.haralick_channel_lbl, self.haralick_channel_choice,
								self.haralick_distance_le, self.haralick_distance_lbl, self.haralick_n_gray_levels_le, self.haralick_n_gray_levels_lbl,
								self.haralick_scale_lbl, self.haralick_scale_slider, self.haralick_percentile_min_lbl, self.haralick_percentile_min_le,
								self.haralick_percentile_max_lbl, self.haralick_percentile_max_le, self.haralick_normalization_mode_btn]
		for element in self.haralick_to_hide:
			element.hide()

		self.features_to_disable = [self.feature_lbl, self.del_feature_btn, self.add_feature_btn, self.features_list, 
									self.use_channel_lbl, *self.mask_channels_cb, self.activate_haralick_btn]

		self.haralick_normalization_mode_btn.clicked.connect(self.switch_to_absolute_normalization_mode)
		layout.addLayout(self.haralick_layout)

	def switch_to_absolute_normalization_mode(self):
		if self.percentile_mode:
			self.percentile_mode = False
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle_outline,color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))		
			self.haralick_normalization_mode_btn.setToolTip("Switch to percentile normalization values.")			
			self.haralick_percentile_min_lbl.setText('Min value: ')
			self.haralick_percentile_max_lbl.setText('Max value: ')
			self.haralick_percentile_min_le.setText('0')
			self.haralick_percentile_max_le.setText('10000')

		else:
			self.percentile_mode = True
			self.haralick_normalization_mode_btn.setIcon(icon(MDI6.percent_circle,color="black"))
			self.haralick_normalization_mode_btn.setIconSize(QSize(20, 20))
			self.haralick_normalization_mode_btn.setToolTip("Switch to absolute normalization values.")			
			self.haralick_percentile_min_lbl.setText('Min percentile: ')
			self.haralick_percentile_max_lbl.setText('Max percentile: ')			
			self.haralick_percentile_min_le.setText('0.01')
			self.haralick_percentile_max_le.setText('99.99')

	def populate_config_frame(self):

		grid = QGridLayout(self.config_frame)
		panel_title = QLabel(f"CONFIGURATION")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_config_btn = QPushButton()
		self.collapse_config_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_config_btn.setIconSize(QSize(20, 20))
		self.collapse_config_btn.setStyleSheet(self.button_select_all)
		grid.addWidget(self.collapse_config_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)

		self.generate_config_panel_contents()
		grid.addWidget(self.ContentsConfig, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_config_btn.clicked.connect(lambda: self.ContentsConfig.setHidden(not self.ContentsConfig.isHidden()))
		self.collapse_config_btn.clicked.connect(self.collapse_config_advanced)
		#self.ContentsConfig.hide()

	def populate_config_trackpy_frame(self):

		grid = QGridLayout(self.config_trackpy_frame)
		panel_title = QLabel(f"CONFIGURATION")
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")

		grid.addWidget(panel_title, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.collapse_config_trackpy_btn = QPushButton()
		self.collapse_config_trackpy_btn.setIcon(icon(MDI6.chevron_down,color="black"))
		self.collapse_config_trackpy_btn.setIconSize(QSize(20, 20))
		self.collapse_config_trackpy_btn.setStyleSheet(self.button_select_all)
		grid.addWidget(self.collapse_config_trackpy_btn, 0, 0, 1, 4, alignment=Qt.AlignRight)
		self.generate_config_trackpy_panel_contents()
		grid.addWidget(self.ContentsConfigTrackpy, 1, 0, 1, 4, alignment=Qt.AlignTop)
		self.collapse_config_trackpy_btn.clicked.connect(lambda: self.ContentsConfigTrackpy.setHidden(not self.ContentsConfigTrackpy.isHidden()))
		self.collapse_config_trackpy_btn.clicked.connect(self.collapse_config_trackpy_advanced)
		#self.ContentsConfig.hide()


	def collapse_config_advanced(self):

		"""
		Switch the chevron icon and adjust the size for the CONFIG frame.
		"""

		features_open = not self.ContentsFeatures.isHidden()
		config_open = not self.ContentsConfig.isHidden()
		post_open = not self.ContentsPostProc.isHidden()
		is_open = np.array([features_open, config_open, post_open])

		if self.ContentsConfig.isHidden():
			self.collapse_config_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_config_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.scroll_area.setMinimumHeight(int(self.minimum_height))
				self.adjustSize()
		else:
			self.collapse_config_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_config_btn.setIconSize(QSize(20, 20))
			self.scroll_area.setMinimumHeight(min(int(930), int(0.9*self.screen_height)))

	def collapse_config_trackpy_advanced(self):

		"""
		Switch the chevron icon and adjust the size for the CONFIG frame.
		"""


		post_open = not self.ContentsPostProc.isHidden()
		is_open = np.array([post_open])

		if self.ContentsConfigTrackpy.isHidden():
			self.collapse_config_trackpy_btn.setIcon(icon(MDI6.chevron_down,color="black"))
			self.collapse_config_trackpy_btn.setIconSize(QSize(20, 20))
			if len(is_open[is_open])==0:
				self.scroll_area.setMinimumHeight(int(self.minimum_height))
				self.adjustSize()
		else:
			self.collapse_config_trackpy_btn.setIcon(icon(MDI6.chevron_up,color="black"))
			self.collapse_config_trackpy_btn.setIconSize(QSize(20, 20))
			self.scroll_area.setMinimumHeight(min(int(930), int(0.9*self.screen_height)))


	def generate_config_trackpy_panel_contents(self):

		self.ContentsConfigTrackpy = QFrame()
		layout = QVBoxLayout(self.ContentsConfigTrackpy)
		layout.setContentsMargins(0,0,0,0)

		sr_layout = QHBoxLayout()
		self.search_range_lbl = QLabel("search range [px]: ")
		self.search_range_le = QLineEdit('30')
		self.search_range_le.setPlaceholderText('search distance in pixels')
		self.search_range_le.setValidator(self.floatValidator)
		sr_layout.addWidget(self.search_range_lbl, 30)
		sr_layout.addWidget(self.search_range_le, 70)
		layout.addLayout(sr_layout)

		memory_layout = QHBoxLayout()
		self.memory_lbl = QLabel("memory [# frames]: ")
		self.memory_slider = QLabeledSlider()
		self.memory_slider.setSingleStep(1)
		self.memory_slider.setTickInterval(1)
		self.memory_slider.setSingleStep(1)
		self.memory_slider.setOrientation(1)
		self.memory_slider.setRange(0,self.parent_window.parent_window.len_movie)
		self.memory_slider.setValue(0)
		memory_layout.addWidget(self.memory_lbl, 30)
		memory_layout.addWidget(self.memory_slider, 70)
		layout.addLayout(memory_layout)


	def generate_config_panel_contents(self):
		
		self.ContentsConfig = QFrame()
		layout = QVBoxLayout(self.ContentsConfig)
		layout.setContentsMargins(0,0,0,0)

		btrack_config_layout = QHBoxLayout()
		self.config_lbl = QLabel("bTrack configuration: ")
		btrack_config_layout.addWidget(self.config_lbl, 90)

		self.upload_btrack_config_btn = QPushButton()
		self.upload_btrack_config_btn.setIcon(icon(MDI6.plus,color="black"))
		self.upload_btrack_config_btn.setIconSize(QSize(20, 20))
		self.upload_btrack_config_btn.setToolTip("Upload a new bTrack configuration.")
		self.upload_btrack_config_btn.setStyleSheet(self.button_select_all)
		self.upload_btrack_config_btn.clicked.connect(self.upload_btrack_config)
		btrack_config_layout.addWidget(self.upload_btrack_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

		self.reset_config_btn = QPushButton()
		self.reset_config_btn.setIcon(icon(MDI6.arrow_u_right_top,color="black"))
		self.reset_config_btn.setIconSize(QSize(20, 20))
		self.reset_config_btn.setToolTip("Reset the configuration to the default bTrack config.")
		self.reset_config_btn.setStyleSheet(self.button_select_all)
		self.reset_config_btn.clicked.connect(self.reset_btrack_config)
		btrack_config_layout.addWidget(self.reset_config_btn, 5)  #4,3,1,1, alignment=Qt.AlignLeft

		layout.addLayout(btrack_config_layout)

		self.config_le = QTextEdit()
		self.config_le.setMinimumHeight(150)
		#self.config_le.setStyleSheet("""
		#							background: #EEEDEB;
		#							border: 2px solid black;
		#							""")
		layout.addWidget(self.config_le)
		self.load_cell_config()


	def show_haralick_options(self):

		"""
		Show the Haralick texture options.
		"""

		if self.activate_haralick_btn.isChecked():
			for element in self.haralick_to_hide:
				element.show()
		else:
			for element in self.haralick_to_hide:
				element.hide()   		

	def upload_btrack_config(self):

		"""
		Upload a specific bTrack config to the experiment folder for the cell population.
		"""

		self.file_dialog = QFileDialog()
		try:
			modelpath = os.sep.join([self.soft_path, "celldetective","models","tracking_configs"]) + os.sep
			print("Track config path: ", modelpath)
			self.filename = self.file_dialog.getOpenFileName(None, "Load config", modelpath, "json files (*.json)")[0]
			if self.filename!=self.config_path:
				copyfile(self.filename, self.config_path)
			self.load_cell_config()
		except Exception as e:
			print(e, modelpath)
			return None

	def reset_btrack_config(self):

		"""
		Set the bTrack config to the default bTrack config.
		"""

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Question)
		msgBox.setText("You are about to revert to the default bTrack configuration? Do you want to proceed?")
		msgBox.setWindowTitle("Confirm")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.Yes:
			config = interpret_tracking_configuration(None)
			if config!=self.config_path:
				copyfile(config, self.config_path)
			self.load_cell_config()
		else:
			return None

	def activate_feature_options(self):

		"""
		Tick the features option.
		"""

		self.switch_feature_option()
		if self.features_ticked:
			for element in self.features_to_disable:
				element.setEnabled(True)
			self.select_features_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_features_btn.setIconSize(QSize(20, 20))
		else:
			for element in self.features_to_disable:
				element.setEnabled(False)
			self.select_features_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_features_btn.setIconSize(QSize(20, 20))
			self.features_list.list_widget.clearSelection()
			self.activate_haralick_btn.setChecked(False)


	def activate_post_proc_options(self):

		"""
		Tick the features option.
		"""

		self.switch_post_proc_option()
		if self.post_proc_ticked:
			for element in self.post_proc_options_to_disable:
				element.setEnabled(True)
			self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
			self.select_post_proc_btn.setIconSize(QSize(20, 20))
		else:
			for element in self.post_proc_options_to_disable:
				element.setEnabled(False)
			self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
			self.select_post_proc_btn.setIconSize(QSize(20, 20))

	def switch_feature_option(self):

		"""
		Switch the feature option.
		"""

		if self.features_ticked == True:
			self.features_ticked = False
		else:
			self.features_ticked = True

	def switch_post_proc_option(self):

		"""
		Switch the feature option.
		"""

		if self.post_proc_ticked == True:
			self.post_proc_ticked = False
		else:
			self.post_proc_ticked = True	

	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def load_cell_config(self):

		"""
		Load the cell configuration and write in the QLineEdit.
		"""

		file_name = interpret_tracking_configuration(self.config_path)
		with open(file_name, 'r') as f:
			json_data = json.load(f)
			self.config_le.setText(json.dumps(json_data, indent=4))

	def write_instructions(self):

		"""
		Write the selected options in a json file for later reading by the software.
		"""

		print('Writing instructions...')
		
		if self.btrack_option.isChecked():
			btrack_option = True
		else:
			btrack_option = False

		# Fetch trackpky params
		if not btrack_option:
			search_range = int(self.search_range_le.text().replace(',','.'))
			memory = self.memory_slider.value()
		else:
			search_range = None
			memory = None	

		tracking_options = {'btrack_config_path': self.config_path}
		tracking_options.update({'btrack_option': btrack_option, 'search_range': search_range, 'memory': memory})

		if not self.features_ticked:
			features = None
			masked_channels = None
		else:
			features = self.features_list.getItems()
			masked_channels = self.channel_names[np.array([not cb.isChecked() for cb in self.mask_channels_cb])]
			if len(masked_channels)==0:
				masked_channels = None
			else:
				masked_channels = list(masked_channels)

		tracking_options.update({'features': features, 'mask_channels': masked_channels})
		
		self.extract_haralick_options()
		tracking_options.update({'haralick_options': self.haralick_options})

		if self.post_proc_ticked:
			post_processing_options = {"minimum_tracklength": int(self.min_tracklength_slider.value()),
									   "remove_not_in_first": self.remove_not_in_first_checkbox.isChecked(), 
									   "remove_not_in_last": self.remove_not_in_last_checkbox.isChecked(),
									   "interpolate_position_gaps": self.interpolate_gaps_checkbox.isChecked(), 
									   "extrapolate_tracks_pre": self.extrapolate_pre_checkbox.isChecked(),
									   "extrapolate_tracks_post": self.extrapolate_post_checkbox.isChecked(),
									   'interpolate_na': False, #self.interpolate_na_features_checkbox.isChecked()
									   }
		else:

			post_processing_options = None

		tracking_options.update({'post_processing_options': post_processing_options})
		file_name = self.track_instructions_write_path
		with open(file_name, 'w') as f:
			json.dump(tracking_options, f, indent=4)

		# Save the JSON data to the file
		file_name = self.config_path
		with open(file_name, 'w') as f:
			f.write(self.config_le.toPlainText())
		print('Done.')
		self.close()

	def uncheck_post_proc(self):
		self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_post_proc_btn.setIconSize(QSize(20, 20))
		self.post_proc_ticked = False
		for element in self.post_proc_options_to_disable:
			element.setEnabled(False)

	def check_post_proc(self):
		self.select_post_proc_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
		self.select_post_proc_btn.setIconSize(QSize(20, 20))
		self.post_proc_ticked = True		
		for element in self.post_proc_options_to_disable:
			element.setEnabled(True)

	def uncheck_features(self):
		self.select_features_btn.setIcon(icon(MDI6.checkbox_blank_outline,color="black"))
		self.select_features_btn.setIconSize(QSize(20, 20))
		self.features_ticked = False
		for element in self.features_to_disable:
			element.setEnabled(False)

	def check_features(self):
		self.select_features_btn.setIcon(icon(MDI6.checkbox_outline,color="black"))
		self.select_features_btn.setIconSize(QSize(20, 20))
		self.features_ticked = True		
		for element in self.features_to_disable:
			element.setEnabled(True)

	def extract_haralick_options(self):

		if self.activate_haralick_btn.isChecked():
			self.haralick_options = {"target_channel": self.haralick_channel_choice.currentIndex(),
								"scale_factor": float(self.haralick_scale_slider.value()),
								"n_intensity_bins": int(self.haralick_n_gray_levels_le.text()),
								"distance" : int(self.haralick_distance_le.text()),
								}
			if self.percentile_mode:
				self.haralick_options.update({"percentiles": (float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text())), "clip_values": None})
			else:
				self.haralick_options.update({"percentiles": None, "clip_values": (float(self.haralick_percentile_min_le.text()), float(self.haralick_percentile_max_le.text()))})

		else:
			self.haralick_options = None		

	def load_previous_tracking_instructions(self):

		"""
		Read the tracking options from a previously written json file.
		"""

		print('Reading instructions..')
		if os.path.exists(self.track_instructions_write_path):
			with open(self.track_instructions_write_path, 'r') as f:
				tracking_instructions = json.load(f)
				print(tracking_instructions)
				
				# Features
				features = tracking_instructions['features']
				if (features is not None) and len(features)>0:
					self.check_features()
					self.ContentsFeatures.show()
					self.features_list.list_widget.clear()
					self.features_list.list_widget.addItems(features)
				else:
					self.ContentsFeatures.hide()
					self.uncheck_features()

				btrack_option = True
				if 'btrack_option' in tracking_instructions:
					btrack_option = tracking_instructions['btrack_option']
				if btrack_option:
					self.btrack_option.click()
				else:
					self.trackpy_option.click()

				if 'search_range' in tracking_instructions:
					search_range = tracking_instructions['search_range']
					if search_range is not None:
						self.search_range_le.setText(str(search_range).replace('.',','))				
				if 'memory' in tracking_instructions:
					memory = tracking_instructions['memory']
					if memory is not None:
						self.memory_slider.setValue(memory)

				# Uncheck channels that are masked
				mask_channels = tracking_instructions['mask_channels']
				if (mask_channels is not None) and len(mask_channels)>0:
					for ch in mask_channels:
						for cb in self.mask_channels_cb:
							if cb.text()==ch:
								cb.setChecked(False)

				haralick_options = tracking_instructions['haralick_options']
				if haralick_options is None:
					self.activate_haralick_btn.setChecked(False)
					self.show_haralick_options()
				else:
					self.activate_haralick_btn.setChecked(True)
					self.show_haralick_options()
					if 'target_channel' in haralick_options:
						idx = haralick_options['target_channel']
						#idx = self.haralick_channel_choice.findText(text_to_find)
						self.haralick_channel_choice.setCurrentIndex(idx)
					if 'scale_factor' in haralick_options:
						self.haralick_scale_slider.setValue(float(haralick_options['scale_factor']))
					if ('percentiles' in haralick_options) and (haralick_options['percentiles'] is not None):
						perc = list(haralick_options['percentiles'])
						self.haralick_percentile_min_le.setText(str(perc[0]))
						self.haralick_percentile_max_le.setText(str(perc[1]))
					if ('clip_values' in haralick_options) and (haralick_options['clip_values'] is not None):
						values = list(haralick_options['clip_values'])
						self.haralick_percentile_min_le.setText(str(values[0]))
						self.haralick_percentile_max_le.setText(str(values[1]))
						self.percentile_mode=True
						self.switch_to_absolute_normalization_mode()
					if 'n_intensity_bins' in haralick_options:
						self.haralick_n_gray_levels_le.setText(str(haralick_options['n_intensity_bins']))
					if 'distance' in haralick_options:
						self.haralick_distance_le.setText(str(haralick_options['distance']))
		


				# Post processing options
				post_processing_options = tracking_instructions['post_processing_options']
				if post_processing_options is None:
					self.uncheck_post_proc()
					self.ContentsPostProc.hide()
					for element in [self.remove_not_in_last_checkbox, self.remove_not_in_first_checkbox, self.interpolate_gaps_checkbox, 
									self.extrapolate_post_checkbox, self.extrapolate_pre_checkbox]: #self.interpolate_na_features_checkbox
						element.setChecked(False)
					self.min_tracklength_slider.setValue(0)

				else:
					self.check_post_proc()
					self.ContentsPostProc.show()
					if "minimum_tracklength" in post_processing_options:
						self.min_tracklength_slider.setValue(int(post_processing_options["minimum_tracklength"]))
					if "remove_not_in_first" in post_processing_options:
						self.remove_not_in_first_checkbox.setChecked(post_processing_options["remove_not_in_first"])
					if "remove_not_in_last" in post_processing_options:
						self.remove_not_in_last_checkbox.setChecked(post_processing_options["remove_not_in_last"])
					if "interpolate_position_gaps" in post_processing_options:
						self.interpolate_gaps_checkbox.setChecked(post_processing_options["interpolate_position_gaps"])
					if "extrapolate_tracks_pre" in post_processing_options:
						self.extrapolate_pre_checkbox.setChecked(post_processing_options["extrapolate_tracks_pre"])
					if "extrapolate_tracks_post" in post_processing_options:
						self.extrapolate_post_checkbox.setChecked(post_processing_options["extrapolate_tracks_post"])
					# if "interpolate_na" in post_processing_options:
					# 	self.interpolate_na_features_checkbox.setChecked(post_processing_options["interpolate_na"])

	def locate_image(self):
		
		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		movies = glob(self.parent_window.parent_window.exp_dir + os.sep.join(["*","*","movie",self.parent_window.parent_window.movie_prefix+"*.tif"]))
		if len(movies)==0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movies are detected in the experiment folder. Cannot load an image to test Haralick.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.test_frame = None
				return None
		else:
			stack0 = movies[0]
			n_channels = len(self.channels)
			self.test_frame = load_frames(np.arange(n_channels), stack0, scale=None, normalize_input=False)


	def control_haralick_digitalization(self):

		"""
		Load an image for the first experiment movie found. 
		Apply the Haralick parameters and check the result of the digitization (normalization + binning of intensities).

		"""

		self.locate_image()
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
			self.fig, self.ax = plt.subplots(1,1,figsize=(4,3))
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





