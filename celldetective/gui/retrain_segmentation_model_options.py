from PyQt5.QtWidgets import QDialog, QMainWindow, QApplication,QRadioButton, QMessageBox, QScrollArea, QComboBox, QFrame, QFileDialog, QGridLayout, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon
from celldetective.gui.gui_utils import center_window
from celldetective.gui.layouts import ChannelNormGenerator

from superqt import QLabeledDoubleSlider,QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import get_software_location
from celldetective.io import get_segmentation_datasets_list, locate_segmentation_dataset, get_segmentation_models_list
from celldetective.segmentation import train_segmentation_model
from celldetective.gui.layouts import CellposeParamsWidget
import numpy as np
import json
import os
from glob import glob
from datetime import datetime
from celldetective.gui import Styles
from celldetective.gui.processes.train_segmentation_model import TrainSegModelProcess
from celldetective.gui.workers import ProgressWindow

class ConfigSegmentationModelTraining(QMainWindow, Styles):
	
	"""
	UI to set segmentation model training instructions.

	"""

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.use_gpu = self.parent_window.use_gpu
		self.setWindowTitle("Train segmentation model")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))
		self.mode = self.parent_window.mode
		self.exp_dir = self.parent_window.exp_dir
		self.soft_path = get_software_location()
		self.pretrained_model = None 
		self.dataset_folder = None
		self.software_models_dir = os.sep.join([self.soft_path, 'celldetective', 'models', f'segmentation_{self.mode}'])

		self.onlyFloat = QDoubleValidator()
		self.onlyInt = QIntValidator()

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		center_window(self)

		self.setMinimumWidth(500)
		self.setMinimumHeight(int(0.3*self.screen_height))
		self.setMaximumHeight(int(0.8*self.screen_height))
		self.populate_widget()
		#self.load_previous_measurement_instructions()

	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		self.button_widget = QWidget()
		self.main_layout = QVBoxLayout()
		self.button_widget.setLayout(self.main_layout)
		self.main_layout.setContentsMargins(30,30,30,30)

		# first frame for FEATURES
		self.model_frame = QFrame()
		self.model_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_model_frame()
		self.main_layout.addWidget(self.model_frame)

		self.data_frame = QFrame()
		self.data_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_data_frame()
		self.main_layout.addWidget(self.data_frame)

		self.hyper_frame = QFrame()
		self.hyper_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
		self.populate_hyper_frame()
		self.main_layout.addWidget(self.hyper_frame)

		self.submit_btn = QPushButton('Train')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.prep_model)
		self.main_layout.addWidget(self.submit_btn)
		self.submit_btn.setEnabled(False)
		self.submit_warning = QLabel('')
		self.main_layout.addWidget(self.submit_warning)

		self.spatial_calib_le.textChanged.connect(self.activate_train_btn)
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")

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

		QApplication.processEvents()
		self.adjustScrollArea()

	def populate_hyper_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.hyper_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.hyper_lbl = QLabel("HYPERPARAMETERS")
		self.hyper_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.hyper_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_hyper_contents()
		grid.addWidget(self.ContentsHyper, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def generate_hyper_contents(self):

		self.ContentsHyper = QFrame()
		layout = QVBoxLayout(self.ContentsHyper)
		layout.setContentsMargins(0,0,0,0)

		lr_layout = QHBoxLayout()
		lr_layout.addWidget(QLabel('learning rate: '),30)
		self.lr_le = QLineEdit('0,0003')
		self.lr_le.setValidator(self.onlyFloat)
		lr_layout.addWidget(self.lr_le, 70)
		layout.addLayout(lr_layout)

		bs_layout = QHBoxLayout()
		bs_layout.addWidget(QLabel('batch size: '),30)
		self.bs_le = QLineEdit('8')
		self.bs_le.setValidator(self.onlyInt)
		bs_layout.addWidget(self.bs_le, 70)
		layout.addLayout(bs_layout)

		epochs_layout = QHBoxLayout()
		epochs_layout.addWidget(QLabel('# epochs: '), 30)
		self.epochs_slider = QLabeledSlider()
		self.epochs_slider.setRange(1,300)
		self.epochs_slider.setSingleStep(1)
		self.epochs_slider.setTickInterval(1)		
		self.epochs_slider.setOrientation(1)
		self.epochs_slider.setValue(100)
		epochs_layout.addWidget(self.epochs_slider, 70)
		layout.addLayout(epochs_layout)

		self.stardist_model.clicked.connect(self.rescale_slider)
		self.cellpose_model.clicked.connect(self.rescale_slider)

	def populate_data_frame(self):

		"""
		Add widgets and layout in the POST-PROCESSING frame.
		"""

		grid = QGridLayout(self.data_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.data_lbl = QLabel("DATA")
		self.data_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.data_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)
		self.generate_data_contents()
		grid.addWidget(self.ContentsData, 1, 0, 1, 4, alignment=Qt.AlignTop)

	def populate_model_frame(self):

		"""
		Add widgets and layout in the FEATURES frame.
		"""

		grid = QGridLayout(self.model_frame)
		grid.setContentsMargins(30,30,30,30)
		grid.setSpacing(30)

		self.model_lbl = QLabel("MODEL")
		self.model_lbl.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		grid.addWidget(self.model_lbl, 0, 0, 1, 4, alignment=Qt.AlignCenter)

		self.generate_model_panel_contents()
		grid.addWidget(self.ContentsModel, 1, 0, 1, 4, alignment=Qt.AlignTop)


	def generate_data_contents(self):

		self.ContentsData = QFrame()
		layout = QVBoxLayout(self.ContentsData)
		layout.setContentsMargins(0,0,0,0)

		train_data_layout = QHBoxLayout()
		train_data_layout.addWidget(QLabel('Training data: '), 30)
		self.select_data_folder_btn = QPushButton('Choose folder')
		self.select_data_folder_btn.clicked.connect(self.showDialog_dataset)
		self.data_folder_label = QLabel('No folder chosen')
		train_data_layout.addWidget(self.select_data_folder_btn, 35)
		train_data_layout.addWidget(self.data_folder_label, 30)

		self.cancel_dataset = QPushButton()
		self.cancel_dataset.setIcon(icon(MDI6.close,color="black"))
		self.cancel_dataset.clicked.connect(self.clear_dataset)
		self.cancel_dataset.setStyleSheet(self.button_select_all)
		self.cancel_dataset.setIconSize(QSize(20, 20))
		self.cancel_dataset.setVisible(False)
		train_data_layout.addWidget(self.cancel_dataset, 5)


		layout.addLayout(train_data_layout)

		include_dataset_layout = QHBoxLayout()
		include_dataset_layout.addWidget(QLabel('include dataset: '),30)
		self.dataset_cb = QComboBox()
		available_datasets, self.datasets_path = get_segmentation_datasets_list(return_path=True)
		signal_datasets = ['--'] + available_datasets #[d.split('/')[-2] for d in available_datasets]
		self.dataset_cb.addItems(signal_datasets)
		include_dataset_layout.addWidget(self.dataset_cb, 70)
		layout.addLayout(include_dataset_layout)

		augmentation_hbox = QHBoxLayout()
		augmentation_hbox.addWidget(QLabel('augmentation\nfactor: '), 30)
		self.augmentation_slider = QLabeledDoubleSlider()
		self.augmentation_slider.setSingleStep(0.01)
		self.augmentation_slider.setTickInterval(0.01)		
		self.augmentation_slider.setOrientation(1)
		self.augmentation_slider.setRange(0.01, 3)
		self.augmentation_slider.setValue(2.0)

		augmentation_hbox.addWidget(self.augmentation_slider, 70)
		layout.addLayout(augmentation_hbox)

		validation_split_layout = QHBoxLayout()
		validation_split_layout.addWidget(QLabel('validation split: '),30)
		self.validation_slider = QLabeledDoubleSlider()
		self.validation_slider.setSingleStep(0.01)
		self.validation_slider.setTickInterval(0.01)		
		self.validation_slider.setOrientation(1)
		self.validation_slider.setRange(0,0.9)
		self.validation_slider.setValue(0.2)		
		validation_split_layout.addWidget(self.validation_slider, 70)
		layout.addLayout(validation_split_layout)


	def generate_model_panel_contents(self):
		
		self.ContentsModel = QFrame()
		layout = QVBoxLayout(self.ContentsModel)
		layout.setContentsMargins(0,0,0,0)

		model_type_layout = QHBoxLayout()
		model_type_layout.setContentsMargins(30,5,30,15)
		self.cellpose_model = QRadioButton('Cellpose')
		self.stardist_model = QRadioButton('StarDist')
		self.stardist_model.setChecked(True)
		model_type_layout.addWidget(self.stardist_model,50, alignment=Qt.AlignCenter)
		model_type_layout.addWidget(self.cellpose_model,50, alignment=Qt.AlignCenter)
		layout.addLayout(model_type_layout)

		modelname_layout = QHBoxLayout()
		modelname_layout.addWidget(QLabel('Model name: '), 30)
		self.modelname_le = QLineEdit()
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")
		self.modelname_le.textChanged.connect(self.activate_train_btn)
		modelname_layout.addWidget(self.modelname_le, 70)
		layout.addLayout(modelname_layout)

		pretrained_layout = QHBoxLayout()
		pretrained_layout.setContentsMargins(0,0,0,0)
		pretrained_layout.addWidget(QLabel('Pretrained model: '), 30)

		self.browse_pretrained_btn = QPushButton('Choose folder')
		self.browse_pretrained_btn.clicked.connect(self.showDialog_pretrained)
		pretrained_layout.addWidget(self.browse_pretrained_btn, 35)

		self.pretrained_lbl = QLabel('No folder chosen')
		pretrained_layout.addWidget(self.pretrained_lbl, 30)

		self.cancel_pretrained = QPushButton()
		self.cancel_pretrained.setIcon(icon(MDI6.close,color="black"))
		self.cancel_pretrained.clicked.connect(self.clear_pretrained)
		self.cancel_pretrained.setStyleSheet(self.button_select_all)
		self.cancel_pretrained.setIconSize(QSize(20, 20))
		self.cancel_pretrained.setVisible(False)
		pretrained_layout.addWidget(self.cancel_pretrained, 5)

		layout.addLayout(pretrained_layout)

		# recompile_layout = QHBoxLayout()
		# recompile_layout.addWidget(QLabel('Recompile: '), 30)
		# self.recompile_option = QCheckBox()
		# self.recompile_option.setEnabled(False)
		# recompile_layout.addWidget(self.recompile_option, 70)
		# layout.addLayout(recompile_layout)

		self.max_nbr_channels = 5
		self.ch_norm = ChannelNormGenerator(self, mode='channels')
		layout.addLayout(self.ch_norm)

		spatial_calib_layout = QHBoxLayout()
		spatial_calib_layout.addWidget(QLabel('input spatial\ncalibration'), 30)
		parent_pxtoum = f"{self.parent_window.parent_window.PxToUm}"
		self.spatial_calib_le = QLineEdit(parent_pxtoum.replace('.',','))
		self.spatial_calib_le.setPlaceholderText('e.g. 0.1 µm per pixel')
		self.spatial_calib_le.setValidator(self.onlyFloat)
		spatial_calib_layout.addWidget(self.spatial_calib_le, 70)
		layout.addLayout(spatial_calib_layout)

	def activate_train_btn(self):

		current_name = self.modelname_le.text()
		models = get_segmentation_models_list(mode=self.mode, return_path=False)
		if not current_name in models and not self.spatial_calib_le.text()=='' and not np.all([cb.currentText()=='--' for cb in self.ch_norm.channel_cbs]):
			self.submit_btn.setEnabled(True)
			self.submit_warning.setText('')
		else:
			self.submit_btn.setEnabled(False)
			if current_name in models:
				self.submit_warning.setText('A model with this name already exists... Please pick another.')
			elif self.spatial_calib_le.text()=='':
				self.submit_warning.setText('Please provide a valid spatial calibration...')
			elif np.all([cb.currentText()=='--' for cb in self.ch_norm.channel_cbs]):
				self.submit_warning.setText('Please provide valid channels...')

	def rescale_slider(self):
		if self.stardist_model.isChecked():
			self.epochs_slider.setRange(1,500)
			self.lr_le.setText('0,0003')
		else:
			self.epochs_slider.setRange(1,10000)
			self.lr_le.setText('0,01')


	def showDialog_pretrained(self):

		# try:
		# 	self.cancel_pretrained.click()
		# except Exception as e:
		# 	print(e)
		# 	pass

		self.clear_pretrained()
		self.pretrained_model = None
		self.pretrained_model = QFileDialog.getExistingDirectory(
						self, "Open Directory",
						os.sep.join([self.soft_path, 'celldetective', 'models', f'segmentation_generic','']),
						QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
						)
		
		if self.pretrained_model=='':
			return None
		
		if self.pretrained_model is None:
			return None

		else:
			self.pretrained_model = self.pretrained_model.replace('\\','/')
			self.pretrained_model = rf"{self.pretrained_model}"
			
			subfiles = glob(os.sep.join([self.pretrained_model,"*"]))
			subfiles = [s.replace('\\','/') for s in subfiles]
			subfiles = [rf"{s}" for s in subfiles]

			if "/".join([self.pretrained_model,"config_input.json"]) in subfiles:
				self.load_pretrained_config()
				self.pretrained_lbl.setText(self.pretrained_model.split("/")[-1])
				self.cancel_pretrained.setVisible(True)
				#self.recompile_option.setEnabled(True)
				self.modelname_le.setText(f"{self.pretrained_model.split('/')[-1]}_{datetime.today().strftime('%Y-%m-%d')}")
			else:
				self.pretrained_model = None
				self.pretrained_lbl.setText('No folder chosen')	
				#self.recompile_option.setEnabled(False)	
				self.cancel_pretrained.setVisible(False)
				return None

		self.seg_folder = self.pretrained_model.split('/')[-2]
		self.model_name = self.pretrained_model.split('/')[-1]
		if self.model_name.startswith('CP') and self.seg_folder=='segmentation_generic':

			self.diamWidget = CellposeParamsWidget(self, model_name=self.model_name)
			self.diamWidget.show()

	def set_cellpose_scale(self):

		scale = self.parent_window.parent_window.PxToUm * float(self.diamWidget.diameter_le.text().replace(',','.')) / 30.0
		if self.model_name=="CP_nuclei":
			scale = self.parent_window.parent_window.PxToUm * float(self.diamWidget.diameter_le.text().replace(',','.')) / 17.0
		self.spatial_calib_le.setText(str(scale).replace('.',','))

		for k in range(len(self.diamWidget.cellpose_channel_cb)):
			ch = self.diamWidget.cellpose_channel_cb[k].currentText()
			idx = self.ch_norm.channel_cbs[k].findText(ch)
			self.ch_norm.channel_cbs[k].setCurrentIndex(idx)

		self.diamWidget.close()


	def showDialog_dataset(self):

		self.dataset_folder = QFileDialog.getExistingDirectory(
						self, "Open Directory",
						self.exp_dir,
						QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
						)
		if self.dataset_folder is not None:

			subfiles = glob(self.dataset_folder+os.sep+"*.tif")
			if len(subfiles)>0:
				print(f'found {len(subfiles)} files in folder')
				self.data_folder_label.setText(self.dataset_folder[:16]+'...')
				self.data_folder_label.setToolTip(self.dataset_folder)
				self.cancel_dataset.setVisible(True)
			else:
				self.data_folder_label.setText('No folder chosen')
				self.data_folder_label.setToolTip('')
				self.dataset_folder = None
				self.cancel_dataset.setVisible(False)

	def clear_pretrained(self):
		
		self.pretrained_model = None
		self.pretrained_lbl.setText('No folder chosen')
		for i in range(len(self.ch_norm.channel_cbs)):
			self.ch_norm.channel_cbs[i].setEnabled(True)
			self.ch_norm.normalization_mode_btns[i].setEnabled(True)
			self.ch_norm.normalization_max_value_le[i].setEnabled(True)
			self.ch_norm.normalization_min_value_le[i].setEnabled(True)
			self.ch_norm.normalization_clip_btns[i].setEnabled(True)
			self.ch_norm.normalization_min_value_lbl[i].setEnabled(True)
			self.ch_norm.normalization_max_value_lbl[i].setEnabled(True)
		self.ch_norm.add_col_btn.setEnabled(True)

		self.cancel_pretrained.setVisible(False)
		self.modelname_le.setText(f"Untitled_model_{datetime.today().strftime('%Y-%m-%d')}")

	def clear_dataset(self):

		self.dataset_folder = None
		self.data_folder_label.setText('No folder chosen')
		self.data_folder_label.setToolTip('')
		self.cancel_dataset.setVisible(False)

	def load_stardist_train_config(self):
		
		config = os.sep.join([self.pretrained_model,"config.json"])
		if os.path.exists(config):
			with open(config, 'r') as f:
				config = json.load(f)
				if 'train_batch_size' in config:
					bs = config['train_batch_size']
					self.bs_le.setText(str(bs).replace('.',','))
				if 'train_learning_rate' in config:
					lr = config['train_learning_rate']
					self.lr_le.setText(str(lr).replace('.',','))

	def load_pretrained_config(self):

		f = open(os.sep.join([self.pretrained_model,"config_input.json"]))
		data = json.load(f)
		channels = data["channels"]
		self.seg_folder = self.pretrained_model.split('/')[-2]
		self.model_name = self.pretrained_model.split('/')[-1]
		if self.model_name.startswith('CP') and self.seg_folder=='segmentation_generic':
			channels = ['brightfield_channel', 'live_nuclei_channel']
			if self.model_name=="CP_nuclei":
				channels = ['live_nuclei_channel', 'None']
		if self.model_name.startswith('SD') and self.seg_folder=='segmentation_generic':
			channels = ['live_nuclei_channel']
			if self.model_name=="SD_versatile_he":
				channels = ["H&E_1","H&E_2","H&E_3"]

		normalization_percentile = data['normalization_percentile']
		normalization_clip = data['normalization_clip']
		normalization_values = data['normalization_values']
		spatial_calib = data['spatial_calibration']
		model_type = data['model_type']
		if model_type=='stardist':
			self.stardist_model.setChecked(True)
			self.cellpose_model.setChecked(False)
			self.load_stardist_train_config()
		else:
			self.stardist_model.setChecked(False)
			self.cellpose_model.setChecked(True)			

		for c,cb in zip(channels, self.ch_norm.channel_cbs):
			index = cb.findText(c)
			cb.setCurrentIndex(index)

		for i in range(len(channels)):

			to_clip = normalization_clip[i]
			if self.ch_norm.clip_option[i] != to_clip:
				self.ch_norm.normalization_clip_btns[i].click()

			use_percentile = normalization_percentile[i]
			if self.ch_norm.normalization_mode[i] != use_percentile:
				self.ch_norm.normalization_mode_btns[i].click()

			self.ch_norm.normalization_min_value_le[i].setText(str(normalization_values[i][0]))
			self.ch_norm.normalization_max_value_le[i].setText(str(normalization_values[i][1]))
			

		if len(channels)<len(self.ch_norm.channel_cbs):
			for k in range(len(self.ch_norm.channel_cbs)-len(channels)):
				self.ch_norm.channel_cbs[len(channels)+k].setCurrentIndex(0)
				self.ch_norm.channel_cbs[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_mode_btns[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_max_value_le[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_min_value_le[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_min_value_lbl[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_max_value_lbl[len(channels)+k].setEnabled(False)
				self.ch_norm.normalization_clip_btns[len(channels)+k].setEnabled(False)
		self.ch_norm.add_col_btn.setEnabled(False)

		self.spatial_calib_le.setText(str(spatial_calib).replace('.',','))

	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def prep_model(self):

		model_name = self.modelname_le.text()
		pretrained_model = self.pretrained_model

		channels = []
		for i in range(len(self.ch_norm.channel_cbs)):
			channels.append(self.ch_norm.channel_cbs[i].currentText())

		slots_to_keep = np.where(np.array(channels)!='--')[0]
		while '--' in channels:
			channels.remove('--')

		norm_values = np.array([[float(a.replace(',','.')),float(b.replace(',','.'))] for a,b in zip([l.text() for l in self.ch_norm.normalization_min_value_le],
											[l.text() for l in self.ch_norm.normalization_max_value_le])])
		norm_values = norm_values[slots_to_keep]
		norm_values = [list(v) for v in norm_values]

		clip_values = np.array(self.ch_norm.clip_option)
		clip_values = list(clip_values[slots_to_keep])
		clip_values = [bool(c) for c in clip_values]

		normalization_mode = np.array(self.ch_norm.normalization_mode)
		normalization_mode = list(normalization_mode[slots_to_keep])
		normalization_mode = [bool(m) for m in normalization_mode]

		data_folders = []
		if self.dataset_folder is not None:
			data_folders.append(self.dataset_folder)
		if self.dataset_cb.currentText()!='--':
			dataset = locate_segmentation_dataset(self.dataset_cb.currentText()) #glob(self.soft_path+'/celldetective/datasets/signals/*/')[self.dataset_cb.currentIndex()-1]
			data_folders.append(dataset)

		aug_factor = round(self.augmentation_slider.value(),2)
		val_split = round(self.validation_slider.value(),2)
		if self.stardist_model.isChecked():
			model_type = 'stardist'
		else:
			model_type = 'cellpose'

		try:
			lr = float(self.lr_le.text().replace(',','.'))
		except:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Invalid value encountered for the learning rate.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			
		
		bs = int(self.bs_le.text())
		epochs = self.epochs_slider.value()
		spatial_calib = float(self.spatial_calib_le.text().replace(',','.'))

		self.training_instructions = {'model_name': model_name,'model_type': model_type, 'pretrained': pretrained_model, 'spatial_calibration': spatial_calib, 'channel_option': channels, 'normalization_percentile': normalization_mode,
		'normalization_clip': clip_values,'normalization_values': norm_values, 'ds': data_folders, 'augmentation_factor': aug_factor, 'validation_split': val_split,
		'learning_rate': lr, 'batch_size': bs, 'epochs': epochs}


		model_folder = os.sep.join([self.software_models_dir,model_name, ''])
		print(model_folder)
		if not os.path.exists(model_folder):
			os.mkdir(model_folder)

		self.training_instructions.update({'target_directory': self.software_models_dir})

		print(f"Set of instructions: {self.training_instructions}")

		self.instructions = model_folder+"training_instructions.json"

		with open(model_folder+"training_instructions.json", 'w') as f:
			json.dump(self.training_instructions, f, indent=4)
		
		# process_args = {"instructions": self.instructions, "use_gpu": self.use_gpu}
		# self.job = ProgressWindow(TrainSegModelProcess, parent_window=self, title="Training", position_info=False, process_args=process_args)
		# result = self.job.exec_()
		# if result == QDialog.Accepted:
		# 	pass
		# elif result == QDialog.Rejected:
		# 	return None		

		train_segmentation_model(self.instructions, use_gpu=self.parent_window.parent_window.parent_window.use_gpu)

		self.parent_window.init_seg_model_list()
		idx = self.parent_window.seg_model_list.findText(model_name)
		self.parent_window.seg_model_list.setCurrentIndex(idx)