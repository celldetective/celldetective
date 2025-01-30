from PyQt5.QtWidgets import QAction, QMenu, QMainWindow, QMessageBox, QLabel, QWidget, QFileDialog, QHBoxLayout, \
	QGridLayout, QLineEdit, QScrollArea, QVBoxLayout, QComboBox, QPushButton, QApplication, QPushButton, QRadioButton, QButtonGroup
from PyQt5.QtGui import QDoubleValidator, QIntValidator

from celldetective.filters import std_filter, gauss_filter
from celldetective.gui.gui_utils import center_window, FigureCanvas, color_from_class, help_generic
from celldetective.gui.gui_utils import PreprocessingLayout
from celldetective.utils import get_software_location, extract_experiment_channels, rename_intensity_column, estimate_unreliable_edge
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.segmentation import threshold_image, identify_markers_from_binary, apply_watershed
import scipy.ndimage as ndi
from PyQt5.QtCore import Qt, QSize
from glob import glob
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
import matplotlib.pyplot as plt
from superqt import QLabeledSlider, QLabeledDoubleRangeSlider, QLabeledDoubleSlider
from celldetective.segmentation import filter_image
import pandas as pd
from skimage.measure import regionprops_table
import json
import os

from celldetective.gui import Styles


class ThresholdConfigWizard(QMainWindow, Styles):
	"""
	UI to create a threshold pipeline for segmentation.

	"""

	def __init__(self, parent_window=None):

		super().__init__()
		self.parent_window = parent_window
		self.screen_height = self.parent_window.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.parent_window.screen_width
		self.setMinimumWidth(int(0.8 * self.screen_width))
		self.setMinimumHeight(int(0.8 * self.screen_height))
		self.setWindowTitle("Threshold configuration wizard")
		center_window(self)
		self.setWindowIcon(self.celldetective_icon)
		self._createActions()
		self._createMenuBar()

		self.mode = self.parent_window.mode
		self.pos = self.parent_window.parent_window.parent_window.pos
		self.exp_dir = self.parent_window.parent_window.exp_dir
		self.soft_path = get_software_location()
		self.footprint = 30
		self.min_dist = 30
		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()
		self.cell_properties = ['centroid', 'area', 'perimeter', 'eccentricity', 'intensity_mean', 'solidity']
		self.edge = None

		if self.mode == "targets":
			self.config_out_name = "threshold_targets.json"
		elif self.mode == "effectors":
			self.config_out_name = "threshold_effectors.json"

		self.locate_stack()
		if self.img is not None:
			self.threshold_slider = QLabeledDoubleRangeSlider()
			self.initialize_histogram()
			self.show_image()
			self.initalize_props_scatter()
			self.prep_cell_properties()
			self.populate_widget()
			self.setAttribute(Qt.WA_DeleteOnClose)

	def _createMenuBar(self):
		menuBar = self.menuBar()
		# Creating menus using a QMenu object
		fileMenu = QMenu("&File", self)
		fileMenu.addAction(self.openAction)
		menuBar.addMenu(fileMenu)

	# Creating menus using a title
	# editMenu = menuBar.addMenu("&Edit")
	# helpMenu = menuBar.addMenu("&Help")

	def _createActions(self):
		# Creating action using the first constructor
		# self.newAction = QAction(self)
		# self.newAction.setText("&New")
		# Creating actions using the second constructor
		self.openAction = QAction(icon(MDI6.folder), "&Open...", self)
		self.openAction.triggered.connect(self.load_previous_config)

	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		self.button_widget = QWidget()
		main_layout = QHBoxLayout()
		self.button_widget.setLayout(main_layout)

		main_layout.setContentsMargins(30, 30, 30, 30)

		self.scroll_area = QScrollArea()
		self.scroll_container = QWidget()
		self.scroll_area.setWidgetResizable(True)
		self.scroll_area.setWidget(self.scroll_container)

		self.left_panel = QVBoxLayout(self.scroll_container)
		self.left_panel.setContentsMargins(30, 30, 30, 30)
		self.left_panel.setSpacing(10)
		self.populate_left_panel()

		# Right panel
		self.right_panel = QVBoxLayout()
		self.populate_right_panel()

		# threhsold options
		# self.left_panel.addWidget(self.cell_fcanvas)

		# Animation
		# self.right_panel.addWidget(self.fcanvas)

		# self.populate_left_panel()
		# grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.scroll_area.setAlignment(Qt.AlignCenter)
		# self.scroll_area.setLayout(self.left_panel)
		# self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		# self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		# self.scroll_area.setWidgetResizable(True)

		main_layout.addWidget(self.scroll_area, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def populate_left_panel(self):

		self.preprocessing = PreprocessingLayout(self)
		self.left_panel.addLayout(self.preprocessing)

		###################
		# THRESHOLD SECTION
		###################

		grid_threshold = QGridLayout()
		grid_threshold.setContentsMargins(20, 20, 20, 20)
		idx = 0

		threshold_title_grid = QHBoxLayout()
		section_threshold = QLabel("Threshold")
		section_threshold.setStyleSheet("font-weight: bold;")
		threshold_title_grid.addWidget(section_threshold, 90, alignment=Qt.AlignCenter)

		self.ylog_check = QPushButton("")
		self.ylog_check.setIcon(icon(MDI6.math_log, color="black"))
		self.ylog_check.setStyleSheet(self.button_select_all)
		self.ylog_check.clicked.connect(self.switch_to_log)
		threshold_title_grid.addWidget(self.ylog_check, 5)

		self.equalize_option_btn = QPushButton("")
		self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="black"))
		self.equalize_option_btn.setIconSize(QSize(20, 20))
		self.equalize_option_btn.setStyleSheet(self.button_select_all)
		self.equalize_option_btn.setToolTip("Enable histogram matching")
		self.equalize_option_btn.clicked.connect(self.activate_histogram_equalizer)
		self.equalize_option = False
		threshold_title_grid.addWidget(self.equalize_option_btn, 5)

		grid_threshold.addLayout(threshold_title_grid, idx, 0, 1, 2)

		idx += 1

		# Slider to set vmin & vmax
		self.threshold_slider.setSingleStep(0.00001)
		self.threshold_slider.setTickInterval(0.00001)
		self.threshold_slider.setOrientation(1)
		self.threshold_slider.setDecimals(5)
		self.threshold_slider.setRange(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.threshold_slider.setValue([np.percentile(self.img.flatten(), 90), np.amax(self.img)])
		self.threshold_slider.valueChanged.connect(self.threshold_changed)

		# self.initialize_histogram()
		grid_threshold.addWidget(self.canvas_hist, idx, 0, 1, 3)

		idx += 1

		# self.threshold_contrast_range.valueChanged.connect(self.set_clim_thresh)

		grid_threshold.addWidget(self.threshold_slider, idx, 1, 1, 1)
		self.canvas_hist.setMinimumHeight(self.screen_height // 6)
		self.left_panel.addLayout(grid_threshold)

		self.generate_marker_contents()
		self.generate_props_contents()

		#################
		# FINAL SAVE BTN#
		#################

		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.button_style_sheet)
		self.save_btn.clicked.connect(self.write_instructions)
		self.left_panel.addWidget(self.save_btn)

		self.properties_box_widgets = [self.propscanvas, *self.features_cb,
									   self.property_query_le, self.submit_query_btn, self.save_btn]
		for p in self.properties_box_widgets:
			p.setEnabled(False)

	def help_prefilter(self):

		"""
		Helper for prefiltering strategy
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','prefilter-for-segmentation.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(f"The suggested technique is to {suggestion}.\nSee a tutorial <a href='https://celldetective.readthedocs.io/en/latest/segment.html'>here</a>.")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None		

	def generate_marker_contents(self):

		marker_box = QVBoxLayout()
		marker_box.setContentsMargins(30, 30, 30, 30)

		marker_lbl = QLabel('Objects')
		marker_lbl.setStyleSheet("font-weight: bold;")
		marker_box.addWidget(marker_lbl, alignment=Qt.AlignCenter)

		object_option_hbox = QHBoxLayout()
		self.marker_option = QRadioButton('markers')
		self.all_objects_option = QRadioButton('all non-contiguous objects')
		self.marker_option_group = QButtonGroup()
		self.marker_option_group.addButton(self.marker_option)
		self.marker_option_group.addButton(self.all_objects_option)
		object_option_hbox.addWidget(self.marker_option, 50, alignment=Qt.AlignCenter)
		object_option_hbox.addWidget(self.all_objects_option, 50, alignment=Qt.AlignCenter)
		marker_box.addLayout(object_option_hbox)

		hbox_footprint = QHBoxLayout()
		hbox_footprint.addWidget(QLabel('Footprint: '), 20)
		self.footprint_slider = QLabeledSlider()
		self.footprint_slider.setSingleStep(1)
		self.footprint_slider.setOrientation(1)
		self.footprint_slider.setRange(1, self.binary.shape[0] // 4)
		self.footprint_slider.setValue(self.footprint)
		self.footprint_slider.valueChanged.connect(self.set_footprint)
		hbox_footprint.addWidget(self.footprint_slider, 30)
		hbox_footprint.addWidget(QLabel(''), 50)
		marker_box.addLayout(hbox_footprint)

		hbox_distance = QHBoxLayout()
		hbox_distance.addWidget(QLabel('Min distance: '), 20)
		self.min_dist_slider = QLabeledSlider()
		self.min_dist_slider.setSingleStep(1)
		self.min_dist_slider.setOrientation(1)
		self.min_dist_slider.setRange(0, self.binary.shape[0] // 4)
		self.min_dist_slider.setValue(self.min_dist)
		self.min_dist_slider.valueChanged.connect(self.set_min_dist)
		hbox_distance.addWidget(self.min_dist_slider, 30)
		hbox_distance.addWidget(QLabel(''), 50)
		marker_box.addLayout(hbox_distance)

		hbox_marker_btns = QHBoxLayout()

		self.markers_btn = QPushButton("Run")
		self.markers_btn.clicked.connect(self.detect_markers)
		self.markers_btn.setStyleSheet(self.button_style_sheet)
		hbox_marker_btns.addWidget(self.markers_btn)

		self.watershed_btn = QPushButton("Watershed")
		self.watershed_btn.setIcon(icon(MDI6.waves_arrow_up, color="white"))
		self.watershed_btn.setIconSize(QSize(20, 20))
		self.watershed_btn.clicked.connect(self.apply_watershed_to_selection)
		self.watershed_btn.setStyleSheet(self.button_style_sheet)
		self.watershed_btn.setEnabled(False)
		hbox_marker_btns.addWidget(self.watershed_btn)
		marker_box.addLayout(hbox_marker_btns)

		self.marker_option.clicked.connect(self.enable_marker_options)
		self.all_objects_option.clicked.connect(self.enable_marker_options)
		self.marker_option.click()

		self.left_panel.addLayout(marker_box)

	def enable_marker_options(self):
		if self.marker_option.isChecked():
			self.footprint_slider.setEnabled(True)
			self.min_dist_slider.setEnabled(True)
			self.markers_btn.setEnabled(True)
		else:
			self.footprint_slider.setEnabled(False)
			self.min_dist_slider.setEnabled(False)
			self.markers_btn.setEnabled(False)
			self.watershed_btn.setEnabled(True)	

	def generate_props_contents(self):

		properties_box = QVBoxLayout()
		properties_box.setContentsMargins(30, 30, 30, 30)

		properties_lbl = QLabel('Filter on properties')
		properties_lbl.setStyleSheet('font-weight: bold;')
		properties_box.addWidget(properties_lbl, alignment=Qt.AlignCenter)

		properties_box.addWidget(self.propscanvas)

		self.features_cb = [QComboBox() for i in range(2)]
		for i in range(2):
			hbox_feat = QHBoxLayout()
			hbox_feat.addWidget(QLabel(f'feature {i}: '), 20)
			hbox_feat.addWidget(self.features_cb[i], 80)
			properties_box.addLayout(hbox_feat)

		hbox_classify = QHBoxLayout()
		hbox_classify.addWidget(QLabel('remove: '), 10)
		self.property_query_le = QLineEdit()
		self.property_query_le.setPlaceholderText(
			'eliminate points using a query such as: area > 100 or eccentricity > 0.95')
		hbox_classify.addWidget(self.property_query_le, 70)
		self.submit_query_btn = QPushButton('Submit...')
		self.submit_query_btn.setStyleSheet(self.button_style_sheet)
		self.submit_query_btn.clicked.connect(self.apply_property_query)
		hbox_classify.addWidget(self.submit_query_btn, 20)
		properties_box.addLayout(hbox_classify)

		self.left_panel.addLayout(properties_box)

	def populate_right_panel(self):

		self.right_panel.addWidget(self.fcanvas, 70)

		channel_hbox = QHBoxLayout()
		channel_hbox.setContentsMargins(150, 30, 150, 5)
		self.channels_cb = QComboBox()
		self.channels_cb.addItems(self.channel_names)
		self.channels_cb.currentTextChanged.connect(self.reload_frame)
		channel_hbox.addWidget(QLabel('channel: '), 10)
		channel_hbox.addWidget(self.channels_cb, 90)
		self.right_panel.addLayout(channel_hbox)

		frame_hbox = QHBoxLayout()
		frame_hbox.setContentsMargins(150, 5, 150, 5)
		self.frame_slider = QLabeledSlider()
		self.frame_slider.setSingleStep(1)
		self.frame_slider.setOrientation(1)
		self.frame_slider.setRange(0, self.len_movie - 1)
		self.frame_slider.setValue(0)
		self.frame_slider.valueChanged.connect(self.reload_frame)
		frame_hbox.addWidget(QLabel('frame: '), 10)
		frame_hbox.addWidget(self.frame_slider, 90)
		self.right_panel.addLayout(frame_hbox)

		contrast_hbox = QHBoxLayout()
		contrast_hbox.setContentsMargins(150, 5, 150, 5)
		self.contrast_slider = QLabeledDoubleRangeSlider()
		self.contrast_slider.setSingleStep(0.00001)
		self.contrast_slider.setTickInterval(0.00001)
		self.contrast_slider.setOrientation(1)
		self.contrast_slider.setRange(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.contrast_slider.setValue([np.percentile(self.img.flatten(), 1), np.percentile(self.img.flatten(), 99.99)])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
		contrast_hbox.addWidget(QLabel('contrast: '))
		contrast_hbox.addWidget(self.contrast_slider, 90)
		self.right_panel.addLayout(contrast_hbox)

	def locate_stack(self):

		"""
		Locate the target movie.

		"""

		if isinstance(self.pos, str):
			movies = glob(self.pos + f"movie/{self.parent_window.parent_window.parent_window.movie_prefix}*.tif")

		else:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a unique position before launching the wizard...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.img = None
			self.close()
			return None

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movies are detected in the experiment folder. Cannot load an image to test Haralick.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.img = None
			self.close()
		else:
			self.stack_path = movies[0]
			print(f'Attempt to read stack {os.path.split(self.stack_path)[-1]}')
			self.len_movie = self.parent_window.parent_window.parent_window.len_movie
			len_movie_auto = auto_load_number_of_frames(self.stack_path)
			if len_movie_auto is not None:
				self.len_movie = len_movie_auto
			exp_config = self.exp_dir + "config.ini"
			self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
			self.channel_names = np.array(self.channel_names)
			self.channels = np.array(self.channels)
			self.nbr_channels = len(self.channels)
			self.current_channel = 0
			self.img = load_frames(0, self.stack_path, normalize_input=False)
			print(f'Detected image shape: {self.img.shape}...')

	def show_image(self):

		"""
		Load an image.

		"""

		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		self.im = self.ax.imshow(self.img, cmap='gray', interpolation='none')

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1],
									  foreground_value=1., fill_holes=True, edge_exclusion=None)
		self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)
		self.image_thresholded = self.ax.imshow(self.thresholded_image, cmap="viridis", alpha=0.5, interpolation='none')

		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")
		self.scat_markers = self.ax.scatter([], [], color="tab:red")

		self.fcanvas.canvas.draw()

	def initalize_props_scatter(self):

		"""
		Define properties scatter.
		"""

		self.fig_props, self.ax_props = plt.subplots(tight_layout=True)
		self.propscanvas = FigureCanvas(self.fig_props, interactive=True)
		self.fig_props.set_facecolor('none')
		self.fig_props.canvas.setStyleSheet("background-color: transparent;")
		self.scat_props = self.ax_props.scatter([], [], color='k', alpha=0.75)
		self.propscanvas.canvas.draw_idle()
		self.propscanvas.canvas.setMinimumHeight(self.screen_height // 5)

	def initialize_histogram(self):

		self.fig_hist, self.ax_hist = plt.subplots(tight_layout=True)
		self.canvas_hist = FigureCanvas(self.fig_hist, interactive=False)
		self.fig_hist.set_facecolor('none')
		self.fig_hist.canvas.setStyleSheet("background-color: transparent;")

		# self.ax_hist.clear()
		# self.ax_hist.cla()
		self.ax_hist.patch.set_facecolor('none')
		self.hist_y, x, _ = self.ax_hist.hist(self.img.flatten(), density=True, bins=300, color="k")
		# self.ax_hist.set_xlim(np.amin(self.img),np.amax(self.img))
		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		# self.ax_hist.set_yticks([])
		self.ax_hist.set_xlim(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.ax_hist.set_ylim(0, self.hist_y.max())

		self.threshold_slider.setRange(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.threshold_slider.setValue([np.nanpercentile(self.img.flatten(), 90), np.amax(self.img)])
		self.add_hist_threshold()

		self.canvas_hist.canvas.draw_idle()
		self.canvas_hist.canvas.setMinimumHeight(self.screen_height // 8)

	def update_histogram(self):

		"""
		Redraw the histogram after an update on the image.
		Move the threshold slider accordingly.

		"""

		self.ax_hist.clear()
		self.ax_hist.patch.set_facecolor('none')
		self.hist_y, x, _ = self.ax_hist.hist(self.img.flatten(), density=True, bins=300, color="k")
		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		# self.ax_hist.set_yticks([])
		self.ax_hist.set_xlim(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.ax_hist.set_ylim(0, self.hist_y.max())
		self.add_hist_threshold()
		self.canvas_hist.canvas.draw()

		self.threshold_slider.setRange(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.threshold_slider.setValue([np.nanpercentile(self.img.flatten(), 90), np.amax(self.img)])
		self.threshold_changed(self.threshold_slider.value())

	def add_hist_threshold(self):

		ymin, ymax = self.ax_hist.get_ylim()
		self.min_intensity_line, = self.ax_hist.plot(
			[self.threshold_slider.value()[0], self.threshold_slider.value()[0]], [0, ymax], c="tab:purple")
		self.max_intensity_line, = self.ax_hist.plot(
			[self.threshold_slider.value()[1], self.threshold_slider.value()[1]], [0, ymax], c="tab:purple")

	# self.canvas_hist.canvas.draw_idle()

	def reload_frame(self):

		"""
		Load the frame from the current channel and time choice. Show imshow, update histogram.
		"""

		self.clear_post_threshold_options()

		self.current_channel = self.channels_cb.currentIndex()
		t = int(self.frame_slider.value())
		idx = t * self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		if self.img is not None:
			self.refresh_imshow()
			self.update_histogram()
		# self.redo_histogram()
		else:
			print('Frame could not be loaded...')

	# def redo_histogram(self):
	# 	self.ax_hist.clear()
	# 	self.canvas_hist.canvas.draw()

	def contrast_slider_action(self):

		"""
		Recontrast the imshow as the contrast slider is moved.
		"""

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)

		self.fcanvas.canvas.draw_idle()

	def refresh_imshow(self):

		"""

		Update the imshow based on the current frame selection.

		"""

		self.vmin = np.nanpercentile(self.img.flatten(), 1)
		self.vmax = np.nanpercentile(self.img.flatten(), 99.)

		self.contrast_slider.disconnect()
		self.contrast_slider.setRange(np.amin(self.img[self.img==self.img]), np.amax(self.img[self.img==self.img]))
		self.contrast_slider.setValue([self.vmin, self.vmax])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

		self.im.set_data(self.img)
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	# self.initialize_histogram()

	def preprocess_image(self):

		"""
		Reload the frame, apply the filters, update imshow and histogram.

		"""

		self.reload_frame()
		filters = self.preprocessing.list.items
		self.edge = estimate_unreliable_edge(filters)
		self.img = filter_image(self.img, filters)
		self.refresh_imshow()
		self.update_histogram()

	def threshold_changed(self, value):

		"""
		Move the threshold values on histogram, when slider is moved.
		"""

		self.clear_post_threshold_options()

		self.thresh_min = value[0]
		self.thresh_max = value[1]
		ymin, ymax = self.ax_hist.get_ylim()
		self.min_intensity_line.set_data([self.thresh_min, self.thresh_min], [0, ymax])
		self.max_intensity_line.set_data([self.thresh_max, self.thresh_max], [0, ymax])
		self.canvas_hist.canvas.draw_idle()
		# update imshow threshold
		self.update_threshold()

	def switch_to_log(self):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if self.ax_hist.get_yscale() == 'linear':
			self.ax_hist.set_yscale('log')
		else:
			self.ax_hist.set_yscale('linear')

		# self.ax_hist.autoscale()
		self.ax_hist.set_ylim(0, self.hist_y.max())
		self.canvas_hist.canvas.draw_idle()

	def update_threshold(self):

		"""

		Threshold and binarize the image based on the min/max threshold values
		and display on imshow.

		"""

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1],
									  foreground_value=1., fill_holes=True, edge_exclusion=self.edge)
		self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)
		self.image_thresholded.set_data(self.thresholded_image)
		self.fcanvas.canvas.draw_idle()

	def set_footprint(self):
		self.footprint = self.footprint_slider.value()

	# print(f"Setting footprint to {self.footprint}")

	def set_min_dist(self):
		self.min_dist = self.min_dist_slider.value()

	# print(f"Setting min distance to {self.min_dist}")

	def detect_markers(self):

		self.clear_post_threshold_options()

		if self.binary.ndim == 3:
			self.binary = np.squeeze(self.binary)
		#self.binary = binary_fill_holes(self.binary)
		self.coords, self.edt_map = identify_markers_from_binary(self.binary, self.min_dist,
																 footprint_size=self.footprint, footprint=None,
																 return_edt=True)
		if len(self.coords) > 0:
			self.scat_markers.set_offsets(self.coords[:, [1, 0]])
			self.scat_markers.set_visible(True)
			self.fcanvas.canvas.draw()
			self.scat_props.set_visible(True)
			self.watershed_btn.setEnabled(True)
		else:
			self.watershed_btn.setEnabled(False)

	def apply_watershed_to_selection(self):

		if self.marker_option.isChecked():
			self.labels = apply_watershed(self.binary, self.coords, self.edt_map)
		else:
			self.labels,_ = ndi.label(self.binary.astype(int))

		self.current_channel = self.channels_cb.currentIndex()
		t = int(self.frame_slider.value())
		idx = t * self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		self.refresh_imshow()

		self.image_thresholded.set_cmap('tab20c')
		self.image_thresholded.set_data(np.ma.masked_where(self.labels == 0., self.labels))
		self.image_thresholded.autoscale()
		self.fcanvas.canvas.draw_idle()

		self.compute_features()
		for p in self.properties_box_widgets:
			p.setEnabled(True)

		for i in range(2):
			self.features_cb[i].currentTextChanged.connect(self.update_props_scatter)

	def compute_features(self):

		# Run regionprops to have properties for filtering
		intensity_image_idx = [self.nbr_channels * self.frame_slider.value()]
		for i in range(self.nbr_channels - 1):
			intensity_image_idx += [intensity_image_idx[-1] + 1]

		# Load channels at time t
		multichannel = load_frames(intensity_image_idx, self.stack_path, normalize_input=False)
		self.props = pd.DataFrame(
			regionprops_table(self.labels, intensity_image=multichannel, properties=self.cell_properties))
		self.props = rename_intensity_column(self.props, self.channel_names)
		self.props['radial_distance'] = np.sqrt((self.props['centroid-1'] - self.img.shape[0] / 2) ** 2 + (
				self.props['centroid-0'] - self.img.shape[1] / 2) ** 2)

		for i in range(2):
			self.features_cb[i].clear()
			self.features_cb[i].addItems(list(self.props.columns))
			self.features_cb[i].setCurrentIndex(i)
		self.props["class"] = 1

		self.update_props_scatter()

	def update_props_scatter(self):

		self.scat_props.set_offsets(
			self.props[[self.features_cb[1].currentText(), self.features_cb[0].currentText()]].to_numpy())
		self.scat_props.set_facecolor([color_from_class(c) for c in self.props['class'].to_numpy()])
		self.ax_props.set_xlabel(self.features_cb[1].currentText())
		self.ax_props.set_ylabel(self.features_cb[0].currentText())

		self.scat_markers.set_offsets(self.props[['centroid-1', 'centroid-0']].to_numpy())
		self.scat_markers.set_color(['k'] * len(self.props))
		self.scat_markers.set_facecolor([color_from_class(c) for c in self.props['class'].to_numpy()])

		self.ax_props.set_xlim(0.75 * self.props[self.features_cb[1].currentText()].min(),
							   1.05 * self.props[self.features_cb[1].currentText()].max())
		self.ax_props.set_ylim(0.75 * self.props[self.features_cb[0].currentText()].min(),
							   1.05 * self.props[self.features_cb[0].currentText()].max())
		self.propscanvas.canvas.draw_idle()
		self.fcanvas.canvas.draw_idle()

	def prep_cell_properties(self):

		self.cell_properties_options = list(np.copy(self.cell_properties))
		self.cell_properties_options.remove("centroid")
		for k in range(self.nbr_channels):
			self.cell_properties_options.append(f'intensity_mean-{k}')
		self.cell_properties_options.remove('intensity_mean')

	def apply_property_query(self):
		query = self.property_query_le.text()
		self.props['class'] = 1

		if query == '':
			print('empty query')
		else:
			try:
				self.selection = self.props.query(query).index
				print(self.selection)
				self.props.loc[self.selection, 'class'] = 0
			except Exception as e:
				print(e)
				print(self.props.columns)
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText(f"The query could not be understood. No filtering was applied. {e}")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None

		self.update_props_scatter()

	def clear_post_threshold_options(self):

		self.watershed_btn.setEnabled(False)

		for p in self.properties_box_widgets:
			p.setEnabled(False)

		for i in range(2):
			try:
				self.features_cb[i].disconnect()
			except:
				pass
			self.features_cb[i].clear()

		self.property_query_le.setText('')

		self.binary = threshold_image(self.img, self.threshold_slider.value()[0], self.threshold_slider.value()[1],
									  foreground_value=1., fill_holes=True, edge_exclusion=self.edge)
		self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)

		self.scat_markers.set_color('tab:red')
		self.scat_markers.set_visible(False)
		self.image_thresholded.set_data(self.thresholded_image)
		self.image_thresholded.set_cmap('viridis')
		self.image_thresholded.autoscale()

	def write_instructions(self):

		instructions = {
			"target_channel": self.channels_cb.currentText(),  # for now index but would be more universal to use name
			"thresholds": self.threshold_slider.value(),
			"filters": self.preprocessing.list.items,
			"marker_min_distance": self.min_dist,
			"marker_footprint_size": self.footprint,
			"feature_queries": [self.property_query_le.text()],
			"equalize_reference": [self.equalize_option, self.frame_slider.value()],
			"do_watershed": self.marker_option.isChecked(),
		}

		print('The following instructions will be written: ', instructions)
		self.instruction_file = \
			QFileDialog.getSaveFileName(self, "Save File", self.exp_dir + f'configs/threshold_config_{self.mode}.json',
										'.json')[0]
		if self.instruction_file != '':
			json_object = json.dumps(instructions, indent=4)
			with open(self.instruction_file, "w") as outfile:
				outfile.write(json_object)
			print("Configuration successfully written in ", self.instruction_file)

			self.parent_window.filename = self.instruction_file
			self.parent_window.file_label.setText(self.instruction_file[:16] + '...')
			self.parent_window.file_label.setToolTip(self.instruction_file)

			self.close()
		else:
			print('The instruction file could not be written...')

	def activate_histogram_equalizer(self):

		if not self.equalize_option:
			self.equalize_option = True
			self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="#1f77b4"))
			self.equalize_option_btn.setIconSize(QSize(20, 20))
		else:
			self.equalize_option = False
			self.equalize_option_btn.setIcon(icon(MDI6.equalizer, color="black"))
			self.equalize_option_btn.setIconSize(QSize(20, 20))

	def load_previous_config(self):
		self.previous_instruction_file = \
			QFileDialog.getOpenFileName(self, "Load config",
										self.exp_dir + f'configs/threshold_config_{self.mode}.json',
										"JSON (*.json)")[0]
		with open(self.previous_instruction_file, 'r') as f:
			threshold_instructions = json.load(f)

		target_channel = threshold_instructions['target_channel']
		index = self.channels_cb.findText(target_channel)
		self.channels_cb.setCurrentIndex(index)

		filters = threshold_instructions['filters']
		items_to_add = [f[0] + '_filter' for f in filters]
		self.preprocessing.list.list_widget.clear()
		self.preprocessing.list.list_widget.addItems(items_to_add)
		self.preprocessing.list.items = filters

		self.preprocessing.apply_btn.click()
		#self.apply_filters_btn.click()

		thresholds = threshold_instructions['thresholds']
		self.threshold_slider.setValue(thresholds)

		marker_footprint_size = threshold_instructions['marker_footprint_size']
		self.footprint_slider.setValue(marker_footprint_size)

		marker_min_dist = threshold_instructions['marker_min_distance']
		self.min_dist_slider.setValue(marker_min_dist)

		self.markers_btn.click()
		self.watershed_btn.click()

		feature_queries = threshold_instructions['feature_queries']
		self.property_query_le.setText(feature_queries[0])
		self.submit_query_btn.click()

		if 'do_watershed' in threshold_instructions:
			do_watershed = threshold_instructions['do_watershed']
			if do_watershed:
				self.marker_option.click()
			else:
				self.all_objects_option.click()


class ThresholdNormalisation(ThresholdConfigWizard):
	def __init__(self, min_threshold, current_channel, parent_window=None):
		QMainWindow.__init__(self)
		self.parent_window = parent_window
		self.screen_height = self.parent_window.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.parent_window.screen_width
		self.setMaximumWidth(int(self.screen_width // 3))
		self.setMinimumHeight(int(0.8 * self.screen_height))
		self.setWindowTitle("Normalisation threshold preview")
		center_window(self)
		self.img = None
		self.min_threshold = min_threshold
		self.current_channel = current_channel
		self.mode = self.parent_window.mode
		self.pos = self.parent_window.parent_window.parent_window.pos
		self.exp_dir = self.parent_window.parent_window.exp_dir
		self.soft_path = get_software_location()
		self.auto_close = False

		self.locate_stack()
		if not self.auto_close:
			if self.img is not None:
				self.test_frame = np.squeeze(self.img)
				self.frame = std_filter(gauss_filter(self.test_frame, 2), 4)
				self.threshold_slider = QLabeledDoubleSlider()
				self.threshold_slider.setOrientation(1)
				self.initialize_histogram()
				self.show_threshold_image()
				self.populate_norm_widget()
				self.threshold_changed(self.threshold_slider.value())
				self.setAttribute(Qt.WA_DeleteOnClose)
		else:
			self.close()

	def show_threshold_image(self):
		if self.test_frame is not None:
			self.fig, self.ax = plt.subplots()
			self.fcanvas = FigureCanvas(self.fig, title="Normalisation: control threshold", interactive=True)
			self.ax.clear()
			self.im = self.ax.imshow(self.frame, cmap='gray')
			self.binary = threshold_image(self.test_frame, float(self.min_threshold), self.test_frame.max(),
										  foreground_value=255., fill_holes=False)
			self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)

			self.image_thresholded = self.ax.imshow(self.thresholded_image, cmap="viridis", alpha=0.5,
													interpolation='none')
			self.fcanvas.setMinimumSize(int(self.screen_height * 0.25), int(self.screen_width * 0.25))
			self.ax.set_xticks([])
			self.ax.set_yticks([])
			self.fig.set_facecolor('none')  # or 'None'
			self.fig.canvas.setStyleSheet("background-color: transparent;")
			self.fcanvas.canvas.draw()

	def populate_norm_widget(self):
		"""
		Create the multibox design.

		"""
		self.button_widget = QWidget()
		layout = QVBoxLayout()
		self.button_widget.setLayout(layout)
		self.setCentralWidget(self.button_widget)
		threshold_slider_layout = QHBoxLayout()
		threshold_label = QLabel("Threshold: ")
		# self.threshold_slider = QLabeledDoubleSlider()
		threshold_slider_layout.addWidget(threshold_label)
		threshold_slider_layout.addWidget(self.threshold_slider)
		histogram = self.initialize_histogram()
		self.threshold_slider.valueChanged.connect(self.threshold_changed)
		contrast_slider_layout = QHBoxLayout()
		self.contrast_slider = QLabeledDoubleRangeSlider()
		self.contrast_slider.setSingleStep(0.00001)
		self.contrast_slider.setTickInterval(0.00001)
		self.contrast_slider.setOrientation(1)
		self.contrast_slider.setRange(np.amin(self.frame[self.frame==self.frame]), np.amax(self.frame[self.frame==self.frame]))
		self.contrast_slider.setValue(
			[np.percentile(self.frame.flatten(), 1), np.percentile(self.frame.flatten(), 99.99)])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
		contrast_label = QLabel("Contrast: ")
		contrast_slider_layout.addWidget(contrast_label)
		contrast_slider_layout.addWidget(self.contrast_slider)

		self.submit_threshold_btn = QPushButton('Submit')
		self.submit_threshold_btn.setStyleSheet(self.button_style_sheet_2)
		self.submit_threshold_btn.clicked.connect(self.get_threshold)
		self.ylog_check = QPushButton("")
		self.ylog_check.setIcon(icon(MDI6.math_log, color="black"))
		self.ylog_check.setStyleSheet(self.button_select_all)
		self.ylog_check.clicked.connect(self.switch_to_log)
		self.ylog_check.setMaximumWidth(30)
		log_button = QHBoxLayout()
		log_button.addStretch(1)
		log_button.addWidget(self.ylog_check)
		log_button.addSpacing(25)
		layout.addWidget(self.fcanvas)
		layout.addLayout(log_button)
		layout.addWidget(histogram)
		layout.addLayout(threshold_slider_layout)
		layout.addLayout(contrast_slider_layout)
		layout.addWidget(self.submit_threshold_btn)

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def initialize_histogram(self):

		self.fig_hist, self.ax_hist = plt.subplots(tight_layout=True)
		self.canvas_hist = FigureCanvas(self.fig_hist, interactive=False)
		self.fig_hist.set_facecolor('none')
		self.fig_hist.canvas.setStyleSheet("background-color: transparent;")

		self.ax_hist.patch.set_facecolor('none')
		self.hist_y, x, _ = self.ax_hist.hist(self.frame.flatten(), density=True, bins=300, color="k")
		self.ax_hist.set_xlabel('intensity [a.u.]')
		self.ax_hist.spines['top'].set_visible(False)
		self.ax_hist.spines['right'].set_visible(False)
		self.ax_hist.set_xlim(np.amin(self.frame[self.frame==self.frame]), np.amax(self.frame[self.frame==self.frame]))
		self.ax_hist.set_ylim(0, self.hist_y.max())
		self.threshold_slider.setSingleStep(0.001)
		self.threshold_slider.setTickInterval(0.001)
		self.threshold_slider.setRange(np.amin(self.frame[self.frame==self.frame]), np.amax(self.frame[self.frame==self.frame]))
		self.threshold_slider.setValue(np.percentile(self.frame,90))
		self.add_hist_threshold()

		self.canvas_hist.canvas.draw_idle()
		self.canvas_hist.canvas.setMinimumHeight(int(self.screen_height // 6))
		return self.canvas_hist

	def clear_post_threshold_options(self):

		self.binary = threshold_image(self.test_frame, self.threshold_slider.value(), self.test_frame.max(),
									  foreground_value=255., fill_holes=False)
		self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)

		self.image_thresholded.set_data(self.thresholded_image)
		self.image_thresholded.set_cmap('viridis')
		self.image_thresholded.autoscale()

	def add_hist_threshold(self):

		ymin, ymax = self.ax_hist.get_ylim()
		self.min_intensity_line, = self.ax_hist.plot(
			[self.threshold_slider.value(), self.threshold_slider.value()], [0, ymax], c="tab:purple")

	def threshold_changed(self, value):

		"""
		Move the threshold values on histogram, when slider is moved.
		"""

		self.clear_post_threshold_options()

		self.thresh_min = value
		ymin, ymax = self.ax_hist.get_ylim()
		self.min_intensity_line.set_data([self.thresh_min, self.thresh_min], [0, ymax])
		self.canvas_hist.canvas.draw_idle()
		self.update_threshold()

	def update_threshold(self):

		"""

		Threshold and binarize the image based on the min/max threshold values
		and display on imshow.

		"""

		self.binary = threshold_image(self.frame, self.threshold_slider.value(), self.test_frame.max(),
									  foreground_value=255., fill_holes=False)
		self.thresholded_image = np.ma.masked_where(self.binary == 0., self.binary)
		self.image_thresholded.set_data(self.thresholded_image)
		self.fcanvas.canvas.draw_idle()

	def locate_stack(self):

		"""
		Locate the target movie.

		"""

		if isinstance(self.pos, str):
			movies = glob(self.pos + f"movie/{self.parent_window.parent_window.parent_window.movie_prefix}*.tif")
		else:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a unique position before launching the wizard...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.img = None
				self.auto_close = True
				return None

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(
				"No movies are detected in the experiment folder. Cannot load an image to check normalisation.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.auto_close = True
				return None
		else:
			self.stack_path = movies[0]
			self.len_movie = self.parent_window.parent_window.parent_window.len_movie
			len_movie_auto = auto_load_number_of_frames(self.stack_path)
			if len_movie_auto is not None:
				self.len_movie = len_movie_auto
			exp_config = self.exp_dir + "config.ini"
			self.channel_names, self.channels = extract_experiment_channels(exp_config)
			self.channel_names = np.array(self.channel_names)
			self.channels = np.array(self.channels)
			self.nbr_channels = len(self.channels)
			t = 1
			idx = t * self.nbr_channels + self.current_channel
			self.img = load_frames(idx, self.stack_path, normalize_input=False)
			print(self.img.shape)
			print(f'{self.stack_path} successfully located.')

	def get_threshold(self):
		self.parent_window.tab2_txt_threshold.setText(str(self.threshold_slider.value()))
		self.close()


# class ThresholdSpot(ThresholdConfigWizard):
# 	def __init__(self, current_channel, img, mask, parent_window=None):
# 		QMainWindow.__init__(self)
# 		self.parent_window = parent_window
# 		self.screen_height = self.parent_window.parent_window.parent_window.parent_window.screen_height
# 		self.screen_width = self.parent_window.parent_window.parent_window.parent_window.screen_width
# 		self.setMinimumHeight(int(0.8 * self.screen_height))
# 		self.setWindowTitle("Spot threshold preview")
# 		center_window(self)
# 		self.img = img
# 		self.current_channel = current_channel
# 		self.mode = self.parent_window.mode
# 		self.pos = self.parent_window.parent_window.parent_window.pos
# 		self.exp_dir = self.parent_window.parent_window.exp_dir
# 		self.onlyFloat = QDoubleValidator()
# 		self.onlyInt = QIntValidator()
# 		self.soft_path = get_software_location()
# 		self.auto_close = False

# 		if self.img is not None:
# 			print(self.img.shape)
# 			#self.test_frame = self.img
# 			self.frame = self.img
# 			self.test_mask = mask
# 			self.populate_all()
# 			self.setAttribute(Qt.WA_DeleteOnClose)
# 		if self.auto_close:
# 			self.close()

# 	def populate_left_panel(self):

# 		self.left_layout = QVBoxLayout()
# 		diameter_layout=QHBoxLayout()
# 		self.diameter_lbl = QLabel('Spot diameter: ')
# 		self.diameter_value = QLineEdit()
# 		self.diameter_value.setText(self.parent_window.diameter_value.text())
# 		self.diameter_value.setValidator(self.onlyFloat)
# 		diameter_layout.addWidget(self.diameter_lbl, alignment=Qt.AlignCenter)
# 		diameter_layout.addWidget(self.diameter_value, alignment=Qt.AlignCenter)
# 		self.left_layout.addLayout(diameter_layout)
# 		threshold_layout=QHBoxLayout()
# 		self.threshold_lbl = QLabel('Spot threshold: ')
# 		self.threshold_value = QLineEdit()
# 		self.threshold_value.setValidator(self.onlyFloat)
# 		self.threshold_value.setText(self.parent_window.threshold_value.text())
# 		threshold_layout.addWidget(self.threshold_lbl, alignment=Qt.AlignCenter)
# 		threshold_layout.addWidget(self.threshold_value, alignment=Qt.AlignCenter)
# 		self.left_layout.addLayout(threshold_layout)
# 		self.left_panel.addLayout(self.left_layout)

# 	def enable_preview(self):

# 		diam = self.diameter_value.text().replace(',','').replace('.','')
# 		thresh = self.threshold_value.text().replace(',','').replace('.','')
# 		if diam.isnumeric() and thresh.isnumeric():
# 			self.preview_button.setEnabled(True)
# 		else:
# 			self.preview_button.setEnabled(False)

# 	def draw_spot_preview(self):

# 		try:
# 			diameter_value = float(self.parent_window.diameter_value.text().replace(',','.'))
# 		except:
# 			print('Diameter could not be converted to float... Abort.')
# 			self.auto_close = True
# 			return None

# 		try:
# 			threshold_value = float(self.parent_window.threshold_value.text().replace(',','.'))
# 		except:
# 			print('Threshold could not be converted to float... Abort.')
# 			self.auto_close = True
# 			return None

# 		lbl = self.test_mask
# 		blobs = self.blob_preview(image=self.img[:, :, self.current_channel], label=lbl, threshold=threshold_value,
# 								  diameter=diameter_value)
# 		mask = np.array([lbl[int(y), int(x)] != 0 for y, x, r in blobs])
# 		if np.any(mask):
# 			blobs_filtered = blobs[mask]
# 		else:
# 			blobs_filtered=[]

# 		self.fig_contour, self.ax_contour = plt.subplots(figsize=(4, 6))
# 		self.fcanvas = FigureCanvas(self.fig_contour, title="Blob measurement", interactive=True)
# 		self.ax_contour.clear()
# 		self.im = self.ax_contour.imshow(self.img[:, :, self.current_channel], cmap='gray')
# 		self.circles = [Circle((x, y), r, color='red', fill=False, alpha=0.3) for y, x, r in blobs_filtered]
# 		for circle in self.circles:
# 			self.ax_contour.add_artist(circle)
# 		self.ax_contour.set_xticks([])
# 		self.ax_contour.set_yticks([])
# 		self.fig_contour.set_facecolor('none')  # or 'None'
# 		self.fig_contour.canvas.setStyleSheet("background-color: transparent;")
# 		self.fcanvas.canvas.draw()

# 	def populate_all(self):
# 		self.button_widget = QWidget()
# 		main_layout = QVBoxLayout()
# 		self.button_widget.setLayout(main_layout)
# 		self.right_panel = QVBoxLayout()
# 		self.left_panel = QVBoxLayout()
# 		self.left_panel.setContentsMargins(30, 30, 30, 30)
# 		self.populate_left_panel()
# 		self.draw_spot_preview()
# 		self.setCentralWidget(self.button_widget)
# 		contrast_slider_layout = QHBoxLayout()
# 		self.contrast_slider = QLabeledDoubleRangeSlider()
# 		self.contrast_slider.setSingleStep(0.00001)
# 		self.contrast_slider.setTickInterval(0.00001)
# 		self.contrast_slider.setOrientation(1)
# 		selection = self.frame[:, :, self.current_channel]
# 		self.contrast_slider.setRange(np.amin(selection[selection==selection]), np.amax(selection[selection==selection]))
# 		self.contrast_slider.setValue(
# 			[np.percentile(self.frame[:, :, self.current_channel].flatten(), 1), np.percentile(self.frame[:, :, self.current_channel].flatten(), 99.99)])
# 		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
# 		contrast_label = QLabel("Contrast: ")
# 		contrast_slider_layout.addWidget(contrast_label)
# 		contrast_slider_layout.addWidget(self.contrast_slider)
# 		self.preview_button=QPushButton("Preview")
# 		self.preview_button.clicked.connect(self.update_spots)
# 		self.preview_button.setStyleSheet(self.button_style_sheet_2)
# 		self.apply_changes=QPushButton("Apply")
# 		self.apply_changes.setStyleSheet(self.button_style_sheet)
# 		self.apply_changes.clicked.connect(self.apply)

# 		self.diameter_value.textChanged.connect(self.enable_preview)
# 		self.threshold_value.textChanged.connect(self.enable_preview)

# 		self.right_panel.addWidget(self.fcanvas.canvas)
# 		self.right_panel.addWidget(self.fcanvas.toolbar)

# 		main_layout.addLayout(self.right_panel)
# 		main_layout.addLayout(self.left_panel)
# 		main_layout.addLayout(contrast_slider_layout)
# 		main_layout.addWidget(self.preview_button)
# 		main_layout.addWidget(self.apply_changes)
# 		self.show()

# 	def blob_preview(self, image, label, threshold, diameter):
# 		removed_background = image.copy()
# 		dilated_image = ndimage.grey_dilation(label, footprint=disk(10))
# 		removed_background[np.where(dilated_image == 0)] = 0
# 		min_sigma = (1 / (1 + math.sqrt(2))) * diameter
# 		max_sigma = math.sqrt(2) * min_sigma
# 		blobs = skimage.feature.blob_dog(removed_background, threshold=threshold, min_sigma=min_sigma,
# 										 max_sigma=max_sigma, overlap=0.75)
# 		return blobs

# 	def update_spots(self):

# 		try:
# 			diameter_value = float(self.diameter_value.text().replace(',','.'))
# 		except:
# 			print('Diameter could not be converted to float... Abort.')
# 			return None

# 		try:
# 			threshold_value = float(self.threshold_value.text().replace(',','.'))
# 		except:
# 			print('Threshold could not be converted to float... Abort.')
# 			return None
# 		xlim = self.ax_contour.get_xlim()
# 		ylim = self.ax_contour.get_ylim()
# 		contrast_levels = self.contrast_slider.value()
# 		blobs = self.blob_preview(image=self.frame[:, :, self.current_channel], label=self.test_mask,
# 								  threshold=threshold_value,
# 								  diameter=diameter_value)
# 		mask = np.array([self.test_mask[int(y), int(x)] != 0 for y, x, r in blobs])
# 		if np.any(mask):
# 			blobs_filtered = blobs[mask]
# 		else:
# 			blobs_filtered = []
# 		self.ax_contour.clear()
# 		self.im = self.ax_contour.imshow(self.frame[:, :, self.current_channel], cmap='gray')
# 		self.ax_contour.set_xticks([])
# 		self.ax_contour.set_yticks([])
# 		self.circles = [Circle((x, y), r, color='red', fill=False, alpha=0.3) for y, x, r in blobs_filtered]
# 		for circle in self.circles:
# 			self.ax_contour.add_artist(circle)
# 		self.ax_contour.set_xlim(xlim)
# 		self.ax_contour.set_ylim(ylim)

# 		self.im.set_data(self.frame[:, :, self.current_channel])
# 		self.fig_contour.canvas.draw()
# 		self.contrast_slider.setValue(contrast_levels)




# 	def apply(self):
# 		self.parent_window.threshold_value.setText(self.threshold_value.text())
# 		self.parent_window.diameter_value.setText(self.diameter_value.text())
# 		self.close()