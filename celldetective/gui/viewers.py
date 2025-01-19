import numpy as np
from celldetective.io import auto_load_number_of_frames, load_frames
from celldetective.filters import *
from celldetective.segmentation import filter_image, threshold_image
from celldetective.measure import contour_of_instance_segmentation, extract_blobs_in_image
from celldetective.utils import _get_img_num_per_channel, estimate_unreliable_edge
from tifffile import imread
import matplotlib.pyplot as plt 
from pathlib import Path
from natsort import natsorted
from glob import glob
import os

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QListWidget, QShortcut
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QDoubleValidator
from celldetective.gui.gui_utils import FigureCanvas, center_window, QuickSliderLayout, QHSeperationLine, ThresholdLineEdit, PreprocessingLayout2
from celldetective.gui import Styles
from superqt import QLabeledDoubleSlider, QLabeledSlider, QLabeledDoubleRangeSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from matplotlib_scalebar.scalebar import ScaleBar
import gc
from celldetective.utils import mask_edges
from scipy.ndimage import shift

class StackVisualizer(QWidget, Styles):

	"""
	A widget for visualizing image stacks with interactive sliders and channel selection.

	Parameters:
	- stack (numpy.ndarray or None): The stack of images.
	- stack_path (str or None): The path to the stack of images if provided as a file.
	- frame_slider (bool): Enable frame navigation slider.
	- contrast_slider (bool): Enable contrast adjustment slider.
	- channel_cb (bool): Enable channel selection dropdown.
	- channel_names (list or None): Names of the channels if `channel_cb` is True.
	- n_channels (int): Number of channels.
	- target_channel (int): Index of the target channel.
	- window_title (str): Title of the window.
	- PxToUm (float or None): Pixel to micrometer conversion factor.
	- background_color (str): Background color of the widget.
	- imshow_kwargs (dict): Additional keyword arguments for imshow function.

	Methods:
	- show(): Display the widget.
	- load_stack(): Load the stack of images.
	- locate_image_virtual(): Locate the stack of images if provided as a file.
	- generate_figure_canvas(): Generate the figure canvas for displaying images.
	- generate_channel_cb(): Generate the channel dropdown if enabled.
	- generate_contrast_slider(): Generate the contrast slider if enabled.
	- generate_frame_slider(): Generate the frame slider if enabled.
	- set_target_channel(value): Set the target channel.
	- change_contrast(value): Change contrast based on slider value.
	- set_channel_index(value): Set the channel index based on dropdown value.
	- change_frame(value): Change the displayed frame based on slider value.
	- closeEvent(event): Event handler for closing the widget.

	Notes:
	- This class provides a convenient interface for visualizing image stacks with frame navigation,
	  contrast adjustment, and channel selection functionalities.
	"""
	
	def __init__(self, stack=None, stack_path=None, frame_slider=True, contrast_slider=True, channel_cb=False, channel_names=None, n_channels=1, target_channel=0, window_title='View', PxToUm=None, background_color='transparent',imshow_kwargs={}):
		super().__init__()

		#self.setWindowTitle(window_title)
		self.window_title = window_title

		self.stack = stack
		self.stack_path = stack_path
		self.create_frame_slider = frame_slider
		self.background_color = background_color
		self.create_contrast_slider = contrast_slider
		self.create_channel_cb = channel_cb
		self.n_channels = n_channels
		self.channel_names = channel_names
		self.target_channel = target_channel
		self.imshow_kwargs = imshow_kwargs
		self.PxToUm = PxToUm
		self.init_contrast = False
		self.channel_trigger = False

		self.load_stack() # need to get stack, frame etc
		self.generate_figure_canvas()
		if self.create_channel_cb:
			self.generate_channel_cb()
		if self.create_contrast_slider:
			self.generate_contrast_slider()
		if self.create_frame_slider:
			self.generate_frame_slider()

		self.canvas.layout.setContentsMargins(15,15,15,30)
		self.setAttribute(Qt.WA_DeleteOnClose)
		center_window(self)

	def show(self):
		# Display the widget
		self.canvas.show()

	def load_stack(self):
		# Load the stack of images
		if self.stack is not None:

			if isinstance(self.stack, list):
				self.stack = np.array(self.stack)

			if self.stack.ndim==3:
				print('No channel axis found...')
				self.stack = self.stack[:,:,:,np.newaxis]
				self.target_channel = 0
			
			self.mode = 'direct'
			self.stack_length = len(self.stack)
			self.mid_time = self.stack_length // 2
			self.init_frame = self.stack[self.mid_time,:,:,self.target_channel]
			self.last_frame = self.stack[-1,:,:,self.target_channel]
		else:
			self.mode = 'virtual'
			assert isinstance(self.stack_path, str)
			assert self.stack_path.endswith('.tif')
			self.locate_image_virtual()

	def locate_image_virtual(self):
		# Locate the stack of images if provided as a file

		self.stack_length = auto_load_number_of_frames(self.stack_path)
		self.mid_time = self.stack_length // 2
		self.img_num_per_channel = _get_img_num_per_channel(np.arange(self.n_channels), self.stack_length, self.n_channels)

		self.init_frame = load_frames(self.img_num_per_channel[self.target_channel, self.mid_time], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
		self.last_frame = load_frames(self.img_num_per_channel[self.target_channel, self.stack_length-1], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]

	def generate_figure_canvas(self):
		# Generate the figure canvas for displaying images

		self.fig, self.ax = plt.subplots(figsize=(5,5),tight_layout=True) #figsize=(5, 5)
		self.canvas = FigureCanvas(self.fig, title=self.window_title, interactive=True)
		self.ax.clear()
		self.im = self.ax.imshow(self.init_frame, cmap='gray', interpolation='none', zorder=0, **self.imshow_kwargs)
		if self.PxToUm is not None:
			scalebar = ScaleBar(self.PxToUm,
								"um",
								length_fraction=0.25,
								location='upper right',
								border_pad=0.4,
								box_alpha=0.95,
								color='white',
								box_color='black',
								)
			if self.PxToUm==1:
				scalebar = ScaleBar(1,
								"px",
								dimension="pixel-length",
								length_fraction=0.25,
								location='upper right',
								border_pad=0.4,
								box_alpha=0.95,
								color='white',
								box_color='black',
								)
			self.ax.add_artist(scalebar)
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet(f"background-color: {self.background_color};")
		self.canvas.canvas.draw()

	def generate_channel_cb(self):
		# Generate the channel dropdown if enabled

		assert self.channel_names is not None
		assert len(self.channel_names)==self.n_channels

		channel_layout = QHBoxLayout()
		channel_layout.setContentsMargins(15,0,15,0)
		channel_layout.addWidget(QLabel('Channel: '), 25)

		self.channels_cb = QComboBox()
		self.channels_cb.addItems(self.channel_names)
		self.channels_cb.currentIndexChanged.connect(self.set_channel_index)
		channel_layout.addWidget(self.channels_cb, 75)
		self.canvas.layout.addLayout(channel_layout)

	def generate_contrast_slider(self):
		# Generate the contrast slider if enabled
		
		self.contrast_slider = QLabeledDoubleRangeSlider()
		contrast_layout = QuickSliderLayout(
											label='Contrast: ',
											slider=self.contrast_slider,
											slider_initial_value=[np.nanpercentile(self.init_frame, 0.1),np.nanpercentile(self.init_frame, 99.99)],
											slider_range=(np.nanmin(self.init_frame),np.nanmax(self.init_frame)),
											decimal_option=True,
											precision=1.0E-05,
											)
		contrast_layout.setContentsMargins(15,0,15,0)
		self.im.set_clim(vmin=np.nanpercentile(self.init_frame, 0.1),vmax=np.nanpercentile(self.init_frame, 99.99))
		self.contrast_slider.valueChanged.connect(self.change_contrast)
		self.canvas.layout.addLayout(contrast_layout)



	def generate_frame_slider(self):
		# Generate the frame slider if enabled
	
		self.frame_slider = QLabeledSlider()
		frame_layout = QuickSliderLayout(
										label='Frame: ',
										slider=self.frame_slider,
										slider_initial_value=int(self.mid_time),
										slider_range=(0,self.stack_length-1),
										decimal_option=False,
										)
		frame_layout.setContentsMargins(15,0,15,0)
		self.frame_slider.valueChanged.connect(self.change_frame)
		self.canvas.layout.addLayout(frame_layout)

	def set_target_channel(self, value):
		# Set the target channel
		
		self.target_channel = value
		self.change_frame(self.frame_slider.value())

	def change_contrast(self, value):
		# Change contrast based on slider value

		vmin = value[0]
		vmax = value[1]
		self.im.set_clim(vmin=vmin, vmax=vmax)
		self.fig.canvas.draw_idle()

	def set_channel_index(self, value):
		# Set the channel index based on dropdown value

		self.target_channel = value
		self.init_contrast = True
		if self.mode == 'direct':
			self.last_frame = self.stack[-1,:,:,self.target_channel]
		elif self.mode == 'virtual':
			self.last_frame = load_frames(self.img_num_per_channel[self.target_channel, self.stack_length-1], 
										  self.stack_path,
										  normalize_input=False).astype(float)[:,:,0]
		self.change_frame_from_channel_switch(self.frame_slider.value())
		self.channel_trigger = False
		self.init_contrast = False

	def change_frame_from_channel_switch(self, value):
		
		self.channel_trigger = True
		self.change_frame(value)

	def change_frame(self, value):

		# Change the displayed frame based on slider value
		if self.channel_trigger:
			self.switch_from_channel = True
		else:
			self.switch_from_channel = False

		if self.mode=='virtual':

			self.init_frame = load_frames(self.img_num_per_channel[self.target_channel, value], 
								self.stack_path,
								normalize_input=False
								).astype(float)[:,:,0]
		elif self.mode=='direct':
			self.init_frame = self.stack[value,:,:,self.target_channel].copy()
		
		self.im.set_data(self.init_frame)
		
		if self.init_contrast:
			self.im.autoscale()
			I_min, I_max = self.im.get_clim()
			self.contrast_slider.setRange(np.nanmin([self.init_frame,self.last_frame]),np.nanmax([self.init_frame,self.last_frame]))
			self.contrast_slider.setValue((I_min,I_max))

		if self.create_contrast_slider:
			self.change_contrast(self.contrast_slider.value())

	
	def closeEvent(self, event):
		# Event handler for closing the widget
		self.canvas.close()	


class ThresholdedStackVisualizer(StackVisualizer):
	
	"""
	A widget for visualizing thresholded image stacks with interactive sliders and channel selection.

	Parameters:
	- preprocessing (list or None): A list of preprocessing filters to apply to the image before thresholding.
	- parent_le: The parent QLineEdit instance to set the threshold value.
	- initial_threshold (float): Initial threshold value.
	- initial_mask_alpha (float): Initial mask opacity value.
	- args, kwargs: Additional arguments to pass to the parent class constructor.

	Methods:
	- generate_apply_btn(): Generate the apply button to set the threshold in the parent QLineEdit.
	- set_threshold_in_parent_le(): Set the threshold value in the parent QLineEdit.
	- generate_mask_imshow(): Generate the mask imshow.
	- generate_threshold_slider(): Generate the threshold slider.
	- generate_opacity_slider(): Generate the opacity slider for the mask.
	- change_mask_opacity(value): Change the opacity of the mask.
	- change_threshold(value): Change the threshold value.
	- change_frame(value): Change the displayed frame and update the threshold.
	- compute_mask(threshold_value): Compute the mask based on the threshold value.
	- preprocess_image(): Preprocess the image before thresholding.

	Notes:
	- This class extends the functionality of StackVisualizer to visualize thresholded image stacks
	  with interactive sliders for threshold and mask opacity adjustment.
	"""

	def __init__(self, preprocessing=None, parent_le=None, initial_threshold=5, initial_mask_alpha=0.5, *args, **kwargs):
		# Initialize the widget and its attributes		
		super().__init__(*args, **kwargs)
		self.preprocessing = preprocessing
		self.thresh = initial_threshold
		self.mask_alpha = initial_mask_alpha
		self.parent_le = parent_le
		self.compute_mask(self.thresh)
		self.generate_mask_imshow()
		self.generate_threshold_slider()
		self.generate_opacity_slider()
		if isinstance(self.parent_le, QLineEdit):
			self.generate_apply_btn()

	def generate_apply_btn(self):
		# Generate the apply button to set the threshold in the parent QLineEdit		
		apply_hbox = QHBoxLayout()
		self.apply_threshold_btn = QPushButton('Apply')
		self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
		self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
		apply_hbox.addWidget(QLabel(''),33)
		apply_hbox.addWidget(self.apply_threshold_btn, 33)
		apply_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(apply_hbox)

	def set_threshold_in_parent_le(self):
		# Set the threshold value in the parent QLineEdit
		self.parent_le.set_threshold(self.threshold_slider.value())
		self.close()

	def generate_mask_imshow(self):
		# Generate the mask imshow		
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.mask==0, self.mask), alpha=self.mask_alpha, interpolation='none')
		self.canvas.canvas.draw()

	def generate_threshold_slider(self):
		# Generate the threshold slider
		self.threshold_slider = QLabeledDoubleSlider()
		thresh_layout = QuickSliderLayout(label='Threshold: ',
										slider=self.threshold_slider,
										slider_initial_value=self.thresh,
										slider_range=(0,30),
										decimal_option=True,
										precision=1.0E-05,
										)
		thresh_layout.setContentsMargins(15,0,15,0)
		self.threshold_slider.valueChanged.connect(self.change_threshold)
		self.canvas.layout.addLayout(thresh_layout)

	def generate_opacity_slider(self):
		# Generate the opacity slider for the mask
		self.opacity_slider = QLabeledDoubleSlider()
		opacity_layout = QuickSliderLayout(label='Opacity: ',
										slider=self.opacity_slider,
										slider_initial_value=0.5,
										slider_range=(0,1),
										decimal_option=True,
										precision=1.0E-03
										)
		opacity_layout.setContentsMargins(15,0,15,0)
		self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
		self.canvas.layout.addLayout(opacity_layout)

	def change_mask_opacity(self, value):
		# Change the opacity of the mask
		self.mask_alpha = value
		self.im_mask.set_alpha(self.mask_alpha)
		self.canvas.canvas.draw_idle()

	def change_threshold(self, value):
		# Change the threshold value		
		self.thresh = value
		self.compute_mask(self.thresh)
		mask = np.ma.masked_where(self.mask == 0, self.mask)
		self.im_mask.set_data(mask)
		self.canvas.canvas.draw_idle()

	def change_frame(self, value):
		# Change the displayed frame and update the threshold		
		super().change_frame(value)
		self.change_threshold(self.threshold_slider.value())

	def compute_mask(self, threshold_value):
		# Compute the mask based on the threshold value
		self.preprocess_image()
		edge = estimate_unreliable_edge(self.preprocessing)
		self.mask = threshold_image(self.processed_image, threshold_value, np.inf, foreground_value=1, edge_exclusion=edge).astype(int)

	def preprocess_image(self):
		# Preprocess the image before thresholding		
		if self.preprocessing is not None:

			assert isinstance(self.preprocessing, list)
			self.processed_image = filter_image(self.init_frame.copy(),filters=self.preprocessing)


class CellEdgeVisualizer(StackVisualizer):

	"""
	A widget for visualizing cell edges with interactive sliders and channel selection.

	Parameters:
	- cell_type (str): Type of cells ('effectors' by default).
	- edge_range (tuple): Range of edge sizes (-30, 30) by default.
	- invert (bool): Flag to invert the edge size (False by default).
	- parent_list_widget: The parent QListWidget instance to add edge measurements.
	- parent_le: The parent QLineEdit instance to set the edge size.
	- labels (array or None): Array of labels for cell segmentation.
	- initial_edge (int): Initial edge size (5 by default).
	- initial_mask_alpha (float): Initial mask opacity value (0.5 by default).
	- args, kwargs: Additional arguments to pass to the parent class constructor.

	Methods:
	- load_labels(): Load the cell labels.
	- locate_labels_virtual(): Locate virtual labels.
	- generate_add_to_list_btn(): Generate the add to list button.
	- generate_add_to_le_btn(): Generate the set measurement button for QLineEdit.
	- set_measurement_in_parent_le(): Set the edge size in the parent QLineEdit.
	- set_measurement_in_parent_list(): Add the edge size to the parent QListWidget.
	- generate_label_imshow(): Generate the label imshow.
	- generate_edge_slider(): Generate the edge size slider.
	- generate_opacity_slider(): Generate the opacity slider for the mask.
	- change_mask_opacity(value): Change the opacity of the mask.
	- change_edge_size(value): Change the edge size.
	- change_frame(value): Change the displayed frame and update the edge labels.
	- compute_edge_labels(): Compute the edge labels.

	Notes:
	- This class extends the functionality of StackVisualizer to visualize cell edges
	  with interactive sliders for edge size adjustment and mask opacity control.
	"""

	def __init__(self, cell_type="effectors", edge_range=(-30,30), invert=False, parent_list_widget=None, parent_le=None, labels=None, initial_edge=5, initial_mask_alpha=0.5, *args, **kwargs):
		
		# Initialize the widget and its attributes
		super().__init__(*args, **kwargs)
		self.edge_size = initial_edge
		self.mask_alpha = initial_mask_alpha
		self.cell_type = cell_type
		self.labels = labels
		self.edge_range = edge_range
		self.invert = invert
		self.parent_list_widget = parent_list_widget
		self.parent_le = parent_le

		self.load_labels()
		self.generate_label_imshow()
		self.generate_edge_slider()
		self.generate_opacity_slider()
		if isinstance(self.parent_list_widget, QListWidget):
			self.generate_add_to_list_btn()
		if isinstance(self.parent_le, QLineEdit):
			self.generate_add_to_le_btn()

	def load_labels(self):
		# Load the cell labels

		if self.labels is not None:

			if isinstance(self.labels, list):
				self.labels = np.array(self.labels)

			assert self.labels.ndim==3,'Wrong dimensions for the provided labels, expect TXY'
			assert len(self.labels)==self.stack_length

			self.mode = 'direct'
			self.init_label = self.labels[self.mid_time,:,:]
		else:
			self.mode = 'virtual'
			assert isinstance(self.stack_path, str)
			assert self.stack_path.endswith('.tif')
			self.locate_labels_virtual()
		
		self.compute_edge_labels()

	def locate_labels_virtual(self):
		# Locate virtual labels

		labels_path = str(Path(self.stack_path).parent.parent) + os.sep + f'labels_{self.cell_type}' + os.sep
		self.mask_paths = natsorted(glob(labels_path + '*.tif'))
		
		if len(self.mask_paths) == 0:

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("No labels were found for the selected cells. Abort.")
			msgBox.setWindowTitle("Critical")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.close()

		self.init_label = imread(self.mask_paths[self.frame_slider.value()])

	def generate_add_to_list_btn(self):
		# Generate the add to list button
		
		add_hbox = QHBoxLayout()
		self.add_measurement_btn = QPushButton('Add measurement')
		self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
		self.add_measurement_btn.setIcon(icon(MDI6.plus,color="white"))
		self.add_measurement_btn.setIconSize(QSize(20, 20))
		self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.add_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def generate_add_to_le_btn(self):
		# Generate the set measurement button for QLineEdit
		
		add_hbox = QHBoxLayout()
		self.set_measurement_btn = QPushButton('Set')
		self.set_measurement_btn.clicked.connect(self.set_measurement_in_parent_le)
		self.set_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.set_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def set_measurement_in_parent_le(self):
		# Set the edge size in the parent QLineEdit
		
		self.parent_le.setText(str(int(self.edge_slider.value())))
		self.close()

	def set_measurement_in_parent_list(self):
		# Add the edge size to the parent QListWidget
		
		self.parent_list_widget.addItems([str(self.edge_slider.value())])
		self.close()

	def generate_label_imshow(self):
		# Generate the label imshow
		
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.edge_labels==0, self.edge_labels), alpha=self.mask_alpha, interpolation='none', cmap="viridis")
		self.canvas.canvas.draw()

	def generate_edge_slider(self):
		# Generate the edge size slider

		self.edge_slider = QLabeledSlider()
		edge_layout = QuickSliderLayout(label='Edge: ',
										slider=self.edge_slider,
										slider_initial_value=self.edge_size,
										slider_range=self.edge_range,
										decimal_option=False,
										)
		edge_layout.setContentsMargins(15,0,15,0)
		self.edge_slider.valueChanged.connect(self.change_edge_size)
		self.canvas.layout.addLayout(edge_layout)

	def generate_opacity_slider(self):
		# Generate the opacity slider for the mask

		self.opacity_slider = QLabeledDoubleSlider()
		opacity_layout = QuickSliderLayout(label='Opacity: ',
										slider=self.opacity_slider,
										slider_initial_value=0.5,
										slider_range=(0,1),
										decimal_option=True,
										precision=1.0E-03
										)
		opacity_layout.setContentsMargins(15,0,15,0)
		self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
		self.canvas.layout.addLayout(opacity_layout)

	def change_mask_opacity(self, value):
		# Change the opacity of the mask

		self.mask_alpha = value
		self.im_mask.set_alpha(self.mask_alpha)
		self.canvas.canvas.draw_idle()

	def change_edge_size(self, value):
		# Change the edge size
		
		self.edge_size = value
		self.compute_edge_labels()
		mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
		self.im_mask.set_data(mask)
		self.canvas.canvas.draw_idle()

	def change_frame(self, value):
		# Change the displayed frame and update the edge labels
		
		super().change_frame(value)

		if self.mode=='virtual':
			self.init_label = imread(self.mask_paths[value])
		elif self.mode=='direct':
			self.init_label = self.labels[value,:,:]
		
		self.compute_edge_labels()
		mask = np.ma.masked_where(self.edge_labels == 0, self.edge_labels)
		self.im_mask.set_data(mask)

	def compute_edge_labels(self):
		# Compute the edge labels
		
		if self.invert:
			edge_size = - self.edge_size
		else:
			edge_size = self.edge_size

		self.edge_labels = contour_of_instance_segmentation(self.init_label, edge_size)

class SpotDetectionVisualizer(StackVisualizer):
	
	def __init__(self, parent_channel_cb=None, parent_diameter_le=None, parent_threshold_le=None, parent_preprocessing_list=None, cell_type='targets', labels=None, *args, **kwargs):

		super().__init__(*args, **kwargs)
		
		self.cell_type = cell_type
		self.labels = labels
		self.detection_channel = self.target_channel

		self.parent_channel_cb = parent_channel_cb
		self.parent_diameter_le = parent_diameter_le
		self.parent_threshold_le = parent_threshold_le
		self.parent_preprocessing_list = parent_preprocessing_list

		self.spot_sizes = []
		self.floatValidator = QDoubleValidator()
		self.init_scatter()
		
		self.generate_detection_channel()		
		self.detection_channel = self.detection_channel_cb.currentIndex()

		self.generate_spot_detection_params()
		self.generate_add_measurement_btn()
		self.load_labels()
		self.change_frame(self.mid_time)

		self.ax.callbacks.connect('xlim_changed', self.update_marker_sizes)
		self.ax.callbacks.connect('ylim_changed', self.update_marker_sizes)

		self.apply_diam_btn.clicked.connect(self.detect_and_display_spots)
		self.apply_thresh_btn.clicked.connect(self.detect_and_display_spots)
		
		self.channels_cb.setCurrentIndex(self.target_channel)
		self.detection_channel_cb.setCurrentIndex(self.target_channel)

	def update_marker_sizes(self, event=None):
		
		# Get axis bounds
		xlim = self.ax.get_xlim()
		ylim = self.ax.get_ylim()

		# Data-to-pixel scale
		ax_width_in_pixels = self.ax.bbox.width
		ax_height_in_pixels =self.ax.bbox.height

		x_scale = (float(xlim[1]) - float(xlim[0])) / ax_width_in_pixels
		y_scale = (float(ylim[1]) - float(ylim[0])) / ax_height_in_pixels

		# Choose the smaller scale for square pixels
		scale = min(x_scale, y_scale)

		# Convert radius_px to data units
		if len(self.spot_sizes)>0:

			radius_data_units = self.spot_sizes / float(scale)

			# Convert to scatter `s` size (points squared)
			radius_pts = radius_data_units * (72. / self.fig.dpi )
			size = np.pi * (radius_pts ** 2)

			# Update scatter sizes
			self.spot_scat.set_sizes(size)
			self.fig.canvas.draw_idle()

	def init_scatter(self):
		self.spot_scat = self.ax.scatter([],[], s=50, facecolors='none', edgecolors='tab:red',zorder=100)
		self.canvas.canvas.draw()

	def change_frame(self, value):

		super().change_frame(value)
		if not self.switch_from_channel:
			self.reset_detection()

		if self.mode=='virtual':
			self.init_label = imread(self.mask_paths[value])
			self.target_img = load_frames(self.img_num_per_channel[self.detection_channel, value], 
								self.stack_path,normalize_input=False).astype(float)[:,:,0]
		elif self.mode=='direct':
			self.init_label = self.labels[value,:,:]
			self.target_img = self.stack[value,:,:,self.detection_channel].copy()

	def detect_and_display_spots(self):

		self.reset_detection()
		self.control_valid_parameters() # set current diam and threshold
		#self.change_frame(self.frame_slider.value())
		#self.set_detection_channel_index(self.detection_channel_cb.currentIndex())

		image_preprocessing = self.preprocessing.list.items
		if image_preprocessing==[]:
			image_preprocessing = None

		blobs_filtered = extract_blobs_in_image(self.target_img, self.init_label,threshold=self.thresh, diameter=self.diameter, image_preprocessing=image_preprocessing)
		if blobs_filtered is not None:
			self.spot_positions = np.array([[x,y] for y,x,_ in blobs_filtered])
			if len(self.spot_positions)>0:
				self.spot_sizes = np.sqrt(2)*np.array([sig for _,_,sig in blobs_filtered])
			#radius_pts = self.spot_sizes * (self.fig.dpi / 72.0)
			#sizes = np.pi*(radius_pts**2)
			if len(self.spot_positions)>0:
				self.spot_scat.set_offsets(self.spot_positions)
			else:
				empty_offset = np.ma.masked_array([0, 0], mask=True)
				self.spot_scat.set_offsets(empty_offset)
			#self.spot_scat.set_sizes(sizes)
			if len(self.spot_positions)>0:
				self.update_marker_sizes()
			self.canvas.canvas.draw()

	def reset_detection(self):
		
		self.ax.scatter([], []).get_offsets()
		empty_offset = np.ma.masked_array([0, 0], mask=True)
		self.spot_scat.set_offsets(empty_offset)
		self.canvas.canvas.draw()

	def load_labels(self):

		# Load the cell labels
		if self.labels is not None:

			if isinstance(self.labels, list):
				self.labels = np.array(self.labels)

			assert self.labels.ndim==3,'Wrong dimensions for the provided labels, expect TXY'
			assert len(self.labels)==self.stack_length

			self.mode = 'direct'
			self.init_label = self.labels[self.mid_time,:,:]
		else:
			self.mode = 'virtual'
			assert isinstance(self.stack_path, str)
			assert self.stack_path.endswith('.tif')
			self.locate_labels_virtual()
	
	def locate_labels_virtual(self):
		# Locate virtual labels

		labels_path = str(Path(self.stack_path).parent.parent) + os.sep + f'labels_{self.cell_type}' + os.sep
		self.mask_paths = natsorted(glob(labels_path + '*.tif'))
		
		if len(self.mask_paths) == 0:

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("No labels were found for the selected cells. Abort.")
			msgBox.setWindowTitle("Critical")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			self.close()

		self.init_label = imread(self.mask_paths[self.frame_slider.value()])
	
	def generate_detection_channel(self):

		assert self.channel_names is not None
		assert len(self.channel_names)==self.n_channels

		channel_layout = QHBoxLayout()
		channel_layout.setContentsMargins(15,0,15,0)
		channel_layout.addWidget(QLabel('Detection\nchannel: '), 25)

		self.detection_channel_cb = QComboBox()
		self.detection_channel_cb.addItems(self.channel_names)
		self.detection_channel_cb.currentIndexChanged.connect(self.set_detection_channel_index)
		channel_layout.addWidget(self.detection_channel_cb, 75)

		# self.invert_check = QCheckBox('invert')
		# if self.invert:
		# 	self.invert_check.setChecked(True)
		# self.invert_check.toggled.connect(self.set_invert)
		# channel_layout.addWidget(self.invert_check, 10)

		self.canvas.layout.addLayout(channel_layout)
		
		self.preprocessing = PreprocessingLayout2(fraction=25, parent_window=self)
		self.preprocessing.setContentsMargins(15,0,15,0)
		self.canvas.layout.addLayout(self.preprocessing)


	# def set_invert(self):
	# 	if self.invert_check.isChecked():
	# 		self.invert = True
	# 	else:
	# 		self.invert = False

	def set_detection_channel_index(self, value):

		self.detection_channel = value
		if self.mode == 'direct':
			self.target_img = self.stack[-1,:,:,self.detection_channel]
		elif self.mode == 'virtual':
			self.target_img = load_frames(self.img_num_per_channel[self.detection_channel, self.frame_slider.value()], 
										  self.stack_path,normalize_input=False).astype(float)[:,:,0]

	def generate_spot_detection_params(self):

		self.spot_diam_le = QLineEdit('1')
		self.spot_diam_le.setValidator(self.floatValidator)
		self.apply_diam_btn = QPushButton('Set')
		self.apply_diam_btn.setStyleSheet(self.button_style_sheet_2)

		self.spot_thresh_le = QLineEdit('0')
		self.spot_thresh_le.setValidator(self.floatValidator)
		self.apply_thresh_btn = QPushButton('Set')
		self.apply_thresh_btn.setStyleSheet(self.button_style_sheet_2)		

		self.spot_diam_le.textChanged.connect(self.control_valid_parameters)
		self.spot_thresh_le.textChanged.connect(self.control_valid_parameters)

		spot_diam_layout = QHBoxLayout()
		spot_diam_layout.setContentsMargins(15,0,15,0)
		spot_diam_layout.addWidget(QLabel('Spot diameter: '), 25)
		spot_diam_layout.addWidget(self.spot_diam_le, 65)
		spot_diam_layout.addWidget(self.apply_diam_btn, 10)
		self.canvas.layout.addLayout(spot_diam_layout)

		spot_thresh_layout = QHBoxLayout()
		spot_thresh_layout.setContentsMargins(15,0,15,0)
		spot_thresh_layout.addWidget(QLabel('Detection\nthreshold: '), 25)
		spot_thresh_layout.addWidget(self.spot_thresh_le, 65)
		spot_thresh_layout.addWidget(self.apply_thresh_btn, 10)
		self.canvas.layout.addLayout(spot_thresh_layout)

	def generate_add_measurement_btn(self):

		add_hbox = QHBoxLayout()
		self.add_measurement_btn = QPushButton('Add measurement')
		self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
		self.add_measurement_btn.setIcon(icon(MDI6.plus,color="white"))
		self.add_measurement_btn.setIconSize(QSize(20, 20))
		self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.add_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def control_valid_parameters(self):
		
		valid_diam = False
		try:
			self.diameter = float(self.spot_diam_le.text().replace(',','.'))
			valid_diam = True
		except:
			valid_diam = False

		valid_thresh = False
		try:
			self.thresh = float(self.spot_thresh_le.text().replace(',','.'))
			valid_thresh = True
		except:
			valid_thresh = False

		if valid_diam and valid_thresh:
			self.apply_diam_btn.setEnabled(True)
			self.apply_thresh_btn.setEnabled(True)
			self.add_measurement_btn.setEnabled(True)
		else:
			self.apply_diam_btn.setEnabled(False)
			self.apply_thresh_btn.setEnabled(False)
			self.add_measurement_btn.setEnabled(False)

	def set_measurement_in_parent_list(self):

		if self.parent_channel_cb is not None:
			self.parent_channel_cb.setCurrentIndex(self.detection_channel)
		if self.parent_diameter_le is not None:
			self.parent_diameter_le.setText(self.spot_diam_le.text())
		if self.parent_threshold_le is not None:
			self.parent_threshold_le.setText(self.spot_thresh_le.text())
		if self.parent_preprocessing_list is not None:
			self.parent_preprocessing_list.clear()
			items = self.preprocessing.list.getItems()
			for item in items:
				self.parent_preprocessing_list.addItemToList(item)
			self.parent_preprocessing_list.items = self.preprocessing.list.items
		self.close()

class CellSizeViewer(StackVisualizer):

	"""
	A widget for visualizing cell size with interactive sliders and circle display.

	Parameters:
	- initial_diameter (int): Initial diameter of the circle (40 by default).
	- set_radius_in_list (bool): Flag to set radius instead of diameter in the list (False by default).
	- diameter_slider_range (tuple): Range of the diameter slider (0, 200) by default.
	- parent_le: The parent QLineEdit instance to set the diameter.
	- parent_list_widget: The parent QListWidget instance to add diameter measurements.
	- args, kwargs: Additional arguments to pass to the parent class constructor.

	Methods:
	- generate_circle(): Generate the circle for visualization.
	- generate_add_to_list_btn(): Generate the add to list button.
	- set_measurement_in_parent_list(): Add the diameter to the parent QListWidget.
	- on_xlims_or_ylims_change(event_ax): Update the circle position on axis limits change.
	- generate_set_btn(): Generate the set button for QLineEdit.
	- set_threshold_in_parent_le(): Set the diameter in the parent QLineEdit.
	- generate_diameter_slider(): Generate the diameter slider.
	- change_diameter(value): Change the diameter of the circle.

	Notes:
	- This class extends the functionality of StackVisualizer to visualize cell size
	  with interactive sliders for diameter adjustment and circle display.
	"""

	def __init__(self, initial_diameter=40, set_radius_in_list=False, diameter_slider_range=(0,500), parent_le=None, parent_list_widget=None, *args, **kwargs):
		# Initialize the widget and its attributes
		
		super().__init__(*args, **kwargs)
		self.diameter = initial_diameter
		self.parent_le = parent_le
		self.diameter_slider_range = diameter_slider_range
		self.parent_list_widget = parent_list_widget
		self.set_radius_in_list = set_radius_in_list
		self.generate_circle()
		self.generate_diameter_slider()

		if isinstance(self.parent_le, QLineEdit):
			self.generate_set_btn()
		if isinstance(self.parent_list_widget, QListWidget):
			self.generate_add_to_list_btn()

	def generate_circle(self):
		# Generate the circle for visualization

		self.circ = plt.Circle((self.init_frame.shape[0]//2,self.init_frame.shape[1]//2), self.diameter//2, ec="tab:red",fill=False)
		self.ax.add_patch(self.circ)

		self.ax.callbacks.connect('xlim_changed',self.on_xlims_or_ylims_change)
		self.ax.callbacks.connect('ylim_changed', self.on_xlims_or_ylims_change)

	def generate_add_to_list_btn(self):
		# Generate the add to list button
		
		add_hbox = QHBoxLayout()
		self.add_measurement_btn = QPushButton('Add measurement')
		self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
		self.add_measurement_btn.setIcon(icon(MDI6.plus,color="white"))
		self.add_measurement_btn.setIconSize(QSize(20, 20))
		self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.add_measurement_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def set_measurement_in_parent_list(self):
		# Add the diameter to the parent QListWidget

		if self.set_radius_in_list:
			val = int(self.diameter_slider.value()//2)
		else:
			val = int(self.diameter_slider.value())
		
		self.parent_list_widget.addItems([str(val)])
		self.close()

	def on_xlims_or_ylims_change(self, event_ax):
		# Update the circle position on axis limits change

		xmin,xmax = event_ax.get_xlim()
		ymin,ymax = event_ax.get_ylim()
		self.circ.center = np.mean([xmin,xmax]), np.mean([ymin,ymax])

	def generate_set_btn(self):
		# Generate the set button for QLineEdit
		
		apply_hbox = QHBoxLayout()
		self.apply_threshold_btn = QPushButton('Set')
		self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
		self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
		apply_hbox.addWidget(QLabel(''),33)
		apply_hbox.addWidget(self.apply_threshold_btn, 33)
		apply_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(apply_hbox)

	def set_threshold_in_parent_le(self):
		# Set the diameter in the parent QLineEdit

		self.parent_le.set_threshold(self.diameter_slider.value())
		self.close()

	def generate_diameter_slider(self):
		# Generate the diameter slider

		self.diameter_slider = QLabeledDoubleSlider()
		diameter_layout = QuickSliderLayout(label='Diameter: ',
										slider=self.diameter_slider,
										slider_initial_value=self.diameter,
										slider_range=self.diameter_slider_range,
										decimal_option=True,
										precision=1.0E-05,
										)
		diameter_layout.setContentsMargins(15,0,15,0)
		self.diameter_slider.valueChanged.connect(self.change_diameter)
		self.canvas.layout.addLayout(diameter_layout)

	def change_diameter(self, value):
		# Change the diameter of the circle
		
		self.diameter = value
		self.circ.set_radius(self.diameter//2)
		self.canvas.canvas.draw_idle()


class ChannelOffsetViewer(StackVisualizer):

	def __init__(self, parent_window=None, *args, **kwargs):
		
		self.parent_window = parent_window
		self.overlay_target_channel = -1
		self.shift_vertical = 0
		self.shift_horizontal = 0
		super().__init__(*args, **kwargs)

		self.load_stack()
		self.canvas.layout.addWidget(QHSeperationLine())
		
		self.generate_overlay_channel_cb()
		self.generate_overlay_imshow()
		
		self.generate_overlay_alpha_slider()
		self.generate_overlay_contrast_slider()

		self.generate_overlay_shift()
		self.generate_add_to_parent_btn()

		if self.overlay_target_channel==-1:
			index = len(self.channel_names) -1
		else:
			index = self.overlay_target_channel
		self.channels_overlay_cb.setCurrentIndex(index)
		self.frame_slider.valueChanged.connect(self.change_overlay_frame)

		self.define_keyboard_shortcuts()

		self.channels_overlay_cb.setCurrentIndex(self.parent_window.channels_cb.currentIndex())
		self.set_channel_index(0)

		self.setAttribute(Qt.WA_DeleteOnClose)

	def generate_overlay_imshow(self):
		self.im_overlay = self.ax.imshow(self.overlay_init_frame, cmap='Blues', interpolation='none',alpha=0.5, **self.imshow_kwargs)

	def generate_overlay_alpha_slider(self):
		# Generate the contrast slider if enabled
		
		self.overlay_alpha_slider = QLabeledDoubleSlider()
		alpha_layout = QuickSliderLayout(
											label='Overlay\ntransparency: ',
											slider=self.overlay_alpha_slider,
											slider_initial_value=0.5,
											slider_range=(0,1.0),
											decimal_option=True,
											precision=1.0E-05,
											)
		alpha_layout.setContentsMargins(15,0,15,0)
		self.overlay_alpha_slider.valueChanged.connect(self.change_alpha_overlay)
		self.canvas.layout.addLayout(alpha_layout)


	def generate_overlay_contrast_slider(self):
		# Generate the contrast slider if enabled
		
		self.overlay_contrast_slider = QLabeledDoubleRangeSlider()
		contrast_layout = QuickSliderLayout(
											label='Overlay contrast: ',
											slider=self.overlay_contrast_slider,
											slider_initial_value=[np.nanpercentile(self.overlay_init_frame, 0.1),np.nanpercentile(self.overlay_init_frame, 99.99)],
											slider_range=(np.nanmin(self.overlay_init_frame),np.nanmax(self.overlay_init_frame)),
											decimal_option=True,
											precision=1.0E-05,
											)
		contrast_layout.setContentsMargins(15,0,15,0)
		self.im_overlay.set_clim(vmin=np.nanpercentile(self.overlay_init_frame, 0.1),vmax=np.nanpercentile(self.overlay_init_frame, 99.99))
		self.overlay_contrast_slider.valueChanged.connect(self.change_contrast_overlay)
		self.canvas.layout.addLayout(contrast_layout)

	def set_overlay_channel_index(self, value):
		# Set the channel index based on dropdown value

		self.overlay_target_channel = value
		self.overlay_init_contrast = True
		if self.mode == 'direct':
			self.overlay_last_frame = self.stack[-1,:,:,self.overlay_target_channel]
		elif self.mode == 'virtual':
			self.overlay_last_frame = load_frames(self.img_num_per_channel[self.overlay_target_channel, self.stack_length-1], 
										  self.stack_path,
										  normalize_input=False).astype(float)[:,:,0]
		self.change_overlay_frame(self.frame_slider.value())
		self.overlay_init_contrast = False

	def generate_overlay_channel_cb(self):

		assert self.channel_names is not None
		assert len(self.channel_names)==self.n_channels

		channel_layout = QHBoxLayout()
		channel_layout.setContentsMargins(15,0,15,0)
		channel_layout.addWidget(QLabel('Overlay channel: '), 25)

		self.channels_overlay_cb = QComboBox()
		self.channels_overlay_cb.addItems(self.channel_names)
		self.channels_overlay_cb.currentIndexChanged.connect(self.set_overlay_channel_index)
		channel_layout.addWidget(self.channels_overlay_cb, 75)
		self.canvas.layout.addLayout(channel_layout)

	def generate_overlay_shift(self):

		shift_layout = QHBoxLayout()
		shift_layout.setContentsMargins(15,0,15,0)
		shift_layout.addWidget(QLabel('shift (h): '), 20, alignment=Qt.AlignRight)

		self.apply_shift_btn = QPushButton('Apply')
		self.apply_shift_btn.setStyleSheet(self.button_style_sheet_2)
		self.apply_shift_btn.setToolTip('Apply the shift to the overlay channel.')
		self.apply_shift_btn.clicked.connect(self.shift_generic)

		self.set_shift_btn = QPushButton('Set')

		self.horizontal_shift_le = ThresholdLineEdit(init_value=self.shift_horizontal, connected_buttons=[self.apply_shift_btn, self.set_shift_btn],placeholder='horizontal shift [pixels]', value_type='float')
		shift_layout.addWidget(self.horizontal_shift_le, 20)

		shift_layout.addWidget(QLabel('shift (v): '), 20, alignment=Qt.AlignRight)

		self.vertical_shift_le = ThresholdLineEdit(init_value=self.shift_vertical, connected_buttons=[self.apply_shift_btn, self.set_shift_btn],placeholder='vertical shift [pixels]', value_type='float')
		shift_layout.addWidget(self.vertical_shift_le, 20)

		shift_layout.addWidget(self.apply_shift_btn, 20)

		self.canvas.layout.addLayout(shift_layout)


	def change_overlay_frame(self, value):
		# Change the displayed frame based on slider value
		
		if self.mode=='virtual':

			self.overlay_init_frame = load_frames(self.img_num_per_channel[self.overlay_target_channel, value], 
								self.stack_path,
								normalize_input=False
								).astype(float)[:,:,0]
		elif self.mode=='direct':
			self.overlay_init_frame = self.stack[value,:,:,self.overlay_target_channel].copy()
		
		self.im_overlay.set_data(self.overlay_init_frame)
		
		if self.overlay_init_contrast:
			self.im_overlay.autoscale()
			I_min, I_max = self.im_overlay.get_clim()
			self.overlay_contrast_slider.setRange(np.nanmin([self.overlay_init_frame,self.overlay_last_frame]),np.nanmax([self.overlay_init_frame,self.overlay_last_frame]))
			self.overlay_contrast_slider.setValue((I_min,I_max))

		if self.create_contrast_slider:
			self.change_contrast_overlay(self.overlay_contrast_slider.value())
	
	def locate_image_virtual(self):
		# Locate the stack of images if provided as a file
		self.stack_length = auto_load_number_of_frames(self.stack_path)
		if self.stack_length is None:
			stack = imread(self.stack_path)
			self.stack_length = len(stack)
			del stack
			gc.collect()

		self.mid_time = self.stack_length // 2
		self.img_num_per_channel = _get_img_num_per_channel(np.arange(self.n_channels), self.stack_length, self.n_channels)

		self.init_frame = load_frames(self.img_num_per_channel[self.target_channel, self.mid_time], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
		self.last_frame = load_frames(self.img_num_per_channel[self.target_channel, self.stack_length-1], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
		self.overlay_init_frame = load_frames(self.img_num_per_channel[self.overlay_target_channel, self.mid_time], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
		self.overlay_last_frame = load_frames(self.img_num_per_channel[self.overlay_target_channel, self.stack_length-1], 
									  self.stack_path,
									  normalize_input=False).astype(float)[:,:,0]
	
	def change_contrast_overlay(self, value):
		# Change contrast based on slider value

		vmin = value[0]
		vmax = value[1]
		self.im_overlay.set_clim(vmin=vmin, vmax=vmax)
		self.fig.canvas.draw_idle()

	def change_alpha_overlay(self, value):
		# Change contrast based on slider value

		alpha = value
		self.im_overlay.set_alpha(alpha)
		self.fig.canvas.draw_idle()


	def define_keyboard_shortcuts(self):
		
		self.shift_up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self.canvas)
		self.shift_up_shortcut.activated.connect(self.shift_overlay_up)
		
		self.shift_down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self.canvas)
		self.shift_down_shortcut.activated.connect(self.shift_overlay_down)

		self.shift_left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self.canvas)
		self.shift_left_shortcut.activated.connect(self.shift_overlay_left)
		
		self.shift_right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self.canvas)
		self.shift_right_shortcut.activated.connect(self.shift_overlay_right)


	def shift_overlay_up(self):
		self.shift_vertical -= 2
		self.vertical_shift_le.set_threshold(self.shift_vertical)
		#self.shift_generic()
		self.apply_shift_btn.click()

	def shift_overlay_down(self):
		self.shift_vertical += 2
		self.vertical_shift_le.set_threshold(self.shift_vertical)
		#self.shift_generic()
		self.apply_shift_btn.click()

	def shift_overlay_left(self):
		self.shift_horizontal -= 2
		self.horizontal_shift_le.set_threshold(self.shift_horizontal)
		#self.shift_generic()
		self.apply_shift_btn.click()

	def shift_overlay_right(self):
		self.shift_horizontal += 2
		self.horizontal_shift_le.set_threshold(self.shift_horizontal)
		#self.shift_generic()
		self.apply_shift_btn.click()

	def shift_generic(self):
		self.shift_vertical = self.vertical_shift_le.get_threshold()
		self.shift_horizontal = self.horizontal_shift_le.get_threshold()
		self.shifted_frame = shift(self.overlay_init_frame, [self.shift_vertical, self.shift_horizontal],prefilter=False)
		self.im_overlay.set_data(self.shifted_frame)
		self.fig.canvas.draw_idle()

	def generate_add_to_parent_btn(self):
		
		add_hbox = QHBoxLayout()
		add_hbox.setContentsMargins(0,5,0,5)
		self.set_shift_btn.clicked.connect(self.set_parent_attributes)
		self.set_shift_btn.setStyleSheet(self.button_style_sheet)
		add_hbox.addWidget(QLabel(''),33)
		add_hbox.addWidget(self.set_shift_btn, 33)
		add_hbox.addWidget(QLabel(''),33)		
		self.canvas.layout.addLayout(add_hbox)

	def set_parent_attributes(self):

		idx = self.channels_overlay_cb.currentIndex()
		self.parent_window.channels_cb.setCurrentIndex(idx)
		self.parent_window.vertical_shift_le.set_threshold(self.vertical_shift_le.get_threshold())
		self.parent_window.horizontal_shift_le.set_threshold(self.horizontal_shift_le.get_threshold())
		self.close()
