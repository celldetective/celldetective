from PyQt5.QtWidgets import QWidget, QGridLayout, QComboBox, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QRadioButton, QFileDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window
from celldetective.gui.layouts import ChannelNormGenerator
from celldetective.gui import ThresholdConfigWizard
from PyQt5.QtGui import QDoubleValidator
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import numpy as np
from glob import glob
import os
import json
import shutil
from celldetective.gui import Styles
import gc
from cellpose.models import CellposeModel


class SegmentationModelLoader(QWidget, Styles):
	
	"""
	Upload a segmentation model or define a Threshold pipeline.
	"""

	def __init__(self, parent_window):
		
		super().__init__()
		self.parent_window = parent_window
		self.mode = self.parent_window.mode
		if self.mode=="targets":
			self.target_folder = "segmentation_targets"
		elif self.mode=="effectors":
			self.target_folder = "segmentation_effectors"
		self.setWindowTitle('Upload model')
		self.generate_content()
		self.setWindowIcon(self.celldetective_icon)
		center_window(self)
		self.setAttribute(Qt.WA_DeleteOnClose)

	def generate_content(self):

		self.layout = QGridLayout(self)
		self.layout.addWidget(QLabel('Select:'), 0, 0, 1, 1)
		self.layout.setSpacing(5)

		option_layout = QHBoxLayout()
		option_layout.setContentsMargins(10,10,10,10)

		# Create radio buttons
		self.stardist_button = QRadioButton('StarDist')
		#self.stardist_button.setIcon(QIcon(abs_path+f"/icons/star.png"))

		self.cellpose_button = QRadioButton('Cellpose')
		#self.cellpose_button.setIcon(QIcon(abs_path+f"/icons/cellpose.png"))

		self.threshold_button = QRadioButton('Threshold')

		option_layout.addWidget(self.threshold_button)
		option_layout.addWidget(self.stardist_button)
		option_layout.addWidget(self.cellpose_button)

		self.layout.addLayout(option_layout, 1,0,1,2, alignment=Qt.AlignCenter)
		self.generate_base_block()
		self.layout.addLayout(self.base_block, 2,0,1,2)
		# self.combos = self.channel_layout.channel_cbs
		# self.cb_labels = self.channel_layout.channel_labels

		self.generate_cellpose_options()
		self.layout.addLayout(self.cellpose_block, 3,0,1,2)
		self.generate_threshold_options()

		# Create file dialog
		self.file_dialog = QFileDialog()

		# Create button to open file dialog
		self.open_dialog_button = QPushButton('Choose File')
		self.open_dialog_button.clicked.connect(self.showDialog)
		self.file_label = QLabel('No file chosen', self)
		self.layout.addWidget(self.open_dialog_button, 9, 0, 1, 1)
		self.layout.addWidget(self.file_label, 9, 1, 1, 1)


		self.merge_lbl = QLabel('Merging option: ')
		self.merge_cb = QComboBox()
		self.merge_cb.addItems(['OR'])
		merge_hbox = QHBoxLayout()
		merge_hbox.addWidget(self.merge_lbl, 33)
		merge_hbox.addWidget(self.merge_cb, 66)
		self.layout.addLayout(merge_hbox, 10, 0, 1, 2)
		self.merge_lbl.hide()
		self.merge_cb.hide()

		self.upload_button = QPushButton("Upload")
		self.upload_button.clicked.connect(self.upload_model)
		self.upload_button.setIcon(icon(MDI6.upload,color="white"))
		self.upload_button.setIconSize(QSize(25, 25))
		self.upload_button.setStyleSheet(self.button_style_sheet)
		self.upload_button.setEnabled(False)
		self.layout.addWidget(self.upload_button, 11, 0, 1, 1)
		
		self.base_block_options = [self.calibration_label, self.spatial_calib_le,*self.channel_layout.channel_cbs,*self.channel_layout.channel_labels,*self.channel_layout.normalization_mode_btns, *self.channel_layout.normalization_clip_btns, *self.channel_layout.normalization_min_value_lbl,
								   *self.channel_layout.normalization_min_value_le, *self.channel_layout.normalization_max_value_lbl,*self.channel_layout.normalization_max_value_le,
								   self.channel_layout.add_col_btn]

		self.stardist_button.toggled.connect(self.show_seg_options)
		self.cellpose_button.toggled.connect(self.show_seg_options)
		self.threshold_button.toggled.connect(self.show_seg_options)

		for cb in self.channel_layout.channel_cbs:
			cb.activated.connect(self.unlock_upload)

		self.setLayout(self.layout)

		self.threshold_button.setChecked(True)
		self.show()

	def unlock_upload(self):
		if self.stardist_button.isChecked():
			if np.any([c.currentText()!='--' for c in self.channel_layout.channel_cbs]):
				self.upload_button.setEnabled(True)
			else:
				self.upload_button.setEnabled(False)
		elif self.cellpose_button.isChecked():
			if np.any([c.currentText()!='--' for c in self.channel_layout.channel_cbs]):
				self.upload_button.setEnabled(True)
			else:
				self.upload_button.setEnabled(False)

	def generate_base_block(self):

		"""
		Create widgets common to StarDist and Cellpose.
		"""
		
		self.base_block = QVBoxLayout()
		
		pixel_calib_layout = QHBoxLayout()
		self.calibration_label = QLabel("input spatial\ncalibration: ")
		self.spatial_calib_le = QLineEdit("0,1")
		self.qdv = QDoubleValidator(0.0, np.amax([self.parent_window.parent_window.shape_x, self.parent_window.parent_window.shape_y]), 8, notation=QDoubleValidator.StandardNotation)
		self.spatial_calib_le.setValidator(self.qdv)
		pixel_calib_layout.addWidget(self.calibration_label, 30)
		pixel_calib_layout.addWidget(self.spatial_calib_le, 70)
		self.base_block.addLayout(pixel_calib_layout)

		self.channel_layout = ChannelNormGenerator(self, mode='channels',init_n_channels=2)
		self.channel_layout.setContentsMargins(0,0,0,0)
		self.base_block.addLayout(self.channel_layout)


	def generate_cellpose_options(self):

		"""
		Create Cellpose specific fields to use the model properly when calling it from the app.
		"""

		self.cellpose_block = QVBoxLayout()
		self.cp_diameter_label = QLabel('cell\ndiameter:   ')
		self.cp_diameter_le = QLineEdit("30,0")
		self.cp_diameter_le.setValidator(self.qdv)
		diam_hbox = QHBoxLayout()
		diam_hbox.addWidget(self.cp_diameter_label, 30)
		diam_hbox.addWidget(self.cp_diameter_le, 70)
		self.cellpose_block.addLayout(diam_hbox)

		qdv_prob = QDoubleValidator(-6, 6, 8, notation=QDoubleValidator.StandardNotation)
		self.cp_cellprob_label = QLabel('cellprob\nthreshold:   ')
		self.cp_cellprob_le = QLineEdit('0,0')
		self.cp_cellprob_le.setValidator(qdv_prob)
		cellprob_hbox = QHBoxLayout()
		cellprob_hbox.addWidget(self.cp_cellprob_label, 30)
		cellprob_hbox.addWidget(self.cp_cellprob_le, 70)
		self.cellpose_block.addLayout(cellprob_hbox)

		self.cp_flow_label = QLabel('flow threshold:   ')
		self.cp_flow_le = QLineEdit('0,4')
		self.cp_flow_le.setValidator(qdv_prob)
		flow_hbox = QHBoxLayout()
		flow_hbox.addWidget(self.cp_flow_label, 30)
		flow_hbox.addWidget(self.cp_flow_le, 70)
		self.cellpose_block.addLayout(flow_hbox)

		self.cellpose_options = [self.cp_diameter_label,self.cp_diameter_le,self.cp_cellprob_label,
			self.cp_cellprob_le, self.cp_flow_label, self.cp_flow_le]
		for c in self.cellpose_options:
			c.hide()

	def generate_threshold_options(self):

		"""
		Show the threshold config pipeline button.
		"""

		self.threshold_config_button = QPushButton("Threshold Config Wizard")
		self.threshold_config_button.setIcon(icon(MDI6.auto_fix,color="#1565c0"))
		self.threshold_config_button.setIconSize(QSize(20, 20))
		self.threshold_config_button.setVisible(False)
		self.threshold_config_button.setStyleSheet(self.button_style_sheet_2)
		self.threshold_config_button.clicked.connect(self.open_threshold_config_wizard)
		self.layout.addWidget(self.threshold_config_button,3,0,1,2)
		self.threshold_config_button.hide()

	def showDialog(self):

		"""
		Import the model in the proper folder.
		"""

		# Set the file dialog according to the option chosen
		if self.stardist_button.isChecked():
			self.seg_mode = "stardist"
			self.file_dialog.setFileMode(QFileDialog.Directory)
		elif self.cellpose_button.isChecked():
			self.seg_mode = "cellpose"
			self.file_dialog.setFileMode(QFileDialog.ExistingFile)
		elif self.threshold_button.isChecked():
			self.seg_mode = "threshold"
			self.file_dialog.setFileMode(QFileDialog.ExistingFile)

		# If accepted check validity of data
		if self.seg_mode!='threshold':
			if self.file_dialog.exec_() == QFileDialog.Accepted:
				self.filename = self.file_dialog.selectedFiles()[0]
				if self.seg_mode=="stardist":
					subfiles = glob(self.filename+"/*")
					subfiles = [s.replace('\\','/') for s in subfiles]
					if self.filename+"/thresholds.json" in subfiles:
						self.file_label.setText(self.filename.split("/")[-1])
						self.modelname = self.filename.split("/")[-1]
						self.destination = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+f"/models/{self.target_folder}/"+self.modelname
						self.folder_dest = self.destination
					else:
						msgBox = QMessageBox()
						msgBox.setIcon(QMessageBox.Warning)
						msgBox.setText("StarDist model not recognized... Please ensure that it contains a thresholds.json file or that it is a valid StarDist model...")
						msgBox.setWindowTitle("Warning")
						msgBox.setStandardButtons(QMessageBox.Ok)
						returnValue = msgBox.exec()
						if returnValue == QMessageBox.Ok:
							return None					
				
				if self.seg_mode=="cellpose":
					self.file_label.setText(self.filename.split("/")[-1])
					self.modelname = self.filename.split("/")[-1]
					print(f"Transferring Cellpose model {self.filename}...")
					self.folder_dest = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]+f"/models/{self.target_folder}/"+self.modelname
					self.destination = self.folder_dest+f"/{self.modelname}"

		else:
			self.filename, _ = QFileDialog.getOpenFileNames(
														None,
														"Load threshold configuration(s)...",
														"",
														"Json Configs (*.json)",
														)
			if self.filename:
				n_files = len(self.filename)
				print(f'You loaded {n_files} threshold configuration files...')
				for i,filename in enumerate(self.filename):
					print(f"Config {i}: ",filename,"...")

				if n_files==1:
					self.merge_cb.hide()
					self.merge_lbl.hide()
					self.file_label.setText(self.filename[0].split("/")[-1])
				else:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("You selected more than one pipeline. Please set a merging procedure for the resulting masks...")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						pass

					self.merge_cb.show()
					self.merge_lbl.show()
					self.file_label.setText(f"{n_files} configs loaded...")

	def show_seg_options(self):

		"""
		Show the relevant widgets, mask the others.
		"""

		self.filename = None
		self.file_label.setText('No file chosen')
		self.base_block_options = [self.calibration_label, self.spatial_calib_le,*self.channel_layout.channel_cbs,*self.channel_layout.channel_labels,*self.channel_layout.normalization_mode_btns, *self.channel_layout.normalization_clip_btns, *self.channel_layout.normalization_min_value_lbl,
								   *self.channel_layout.normalization_min_value_le, *self.channel_layout.normalization_max_value_lbl,*self.channel_layout.normalization_max_value_le,
								   self.channel_layout.add_col_btn]

		if self.cellpose_button.isChecked():
			self.spatial_calib_le.setToolTip('Cellpose rescales the images such that the cells are 30.0 pixels. You can compute the scale from the training data as\n(pixel calibration [µm] in training images)*(cell diameter [px] in training images)/(30 [px]).\nIf you pass images with a different calibration to the model, they will be rescaled automatically.\nThe rescaling is ignored if you pass a diameter different from 30 px below.')
			self.channel_layout.channel_labels[0].setText('cyto: ')
			self.channel_layout.channel_labels[1].setText('nuclei: ')
			for c in [self.threshold_config_button, self.merge_lbl, self.merge_cb]:
				c.hide()
			for c in self.cellpose_options+self.base_block_options:
				c.show()
			self.unlock_upload()
		elif self.stardist_button.isChecked():
			self.spatial_calib_le.setToolTip('')
			self.channel_layout.channel_labels[0].setText('channel 1: ')
			self.channel_layout.channel_labels[1].setText('channel 2: ')
			for c in self.base_block_options:
				c.show()
			for c in self.cellpose_options+[self.threshold_config_button, self.merge_lbl, self.merge_cb]:
				c.hide()
			self.unlock_upload()
		else:
			for c in self.base_block_options + self.cellpose_options:
				c.hide()
			self.threshold_config_button.show()
			self.upload_button.setEnabled(True)

		self.resize(self.sizeHint());
		#self.adjustSize()

	def upload_model(self):

		"""
		Upload the model.
		"""
		
		channels = []
		for i in range(len(self.channel_layout.channel_cbs)):
			channels.append(self.channel_layout.channel_cbs[i].currentText())

		if self.file_label.text()=='No file chosen':
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a model first.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		if not self.threshold_button.isChecked():
			
			if self.stardist_button.isChecked():
				
				try:
					shutil.copytree(self.filename, self.destination)
				except FileExistsError:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("A model with the same name already exists in the models folder. Please rename it.")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None

			elif self.cellpose_button.isChecked():

				try:
					if not os.path.exists(self.folder_dest):
						os.mkdir(self.folder_dest)
					shutil.copy(self.filename, self.destination)
				except FileExistsError:
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Warning)
					msgBox.setText("A model with the same name already exists in the models folder. Please rename it.")
					msgBox.setWindowTitle("Warning")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None
				
				try:
					model = CellposeModel(pretrained_model=self.destination, model_type=None, nchan=len(channels))
					self.scale_model = model.diam_mean
					print(f'{self.scale_model=}')
					del model
					gc.collect()
				except Exception as e:
					print(e)
					msgBox = QMessageBox()
					msgBox.setIcon(QMessageBox.Critical)
					msgBox.setText(f"Cellpose model could not be loaded...")
					msgBox.setWindowTitle("Error")
					msgBox.setStandardButtons(QMessageBox.Ok)
					returnValue = msgBox.exec()
					if returnValue == QMessageBox.Ok:
						return None					

			self.generate_input_config()
			self.parent_window.init_seg_model_list()
			self.close()
		else:
			if not isinstance(self.filename, list):
				if not self.filename is None:
					self.filename = [self.filename]
			if self.mode=="targets":
				self.parent_window.threshold_config_targets = self.filename
				self.parent_window.seg_model_list.setCurrentText('Threshold')
				print('Path to the traditional segmentation pipeline successfully set in celldetective...')
				self.close()
			elif self.mode=="effectors":
				self.parent_window.threshold_config_effectors = self.filename
				self.parent_window.seg_model_list.setCurrentText('Threshold')
				print('Path to the traditional segmentation pipeline successfully set in celldetective...')
				self.close()

	def generate_input_config(self):

		"""
		Check the ticked options and input parameters to create
		a configuration to use the uploaded model properly.
		"""
		
		if self.threshold_button.isChecked():
			model_type = "threshold"
			return None

		channels = []
		for i in range(len(self.channel_layout.channel_cbs)):
			channels.append(self.channel_layout.channel_cbs[i].currentText())

		slots_to_keep = np.where(np.array(channels)!='--')[0]
		while '--' in channels:
			channels.remove('--')

		norm_values = np.array([[float(a.replace(',','.')),float(b.replace(',','.'))] for a,b in zip([l.text() for l in self.channel_layout.normalization_min_value_le],
											[l.text() for l in self.channel_layout.normalization_max_value_le])])
		norm_values = norm_values[slots_to_keep]
		norm_values = [list(v) for v in norm_values]

		clip_values = np.array(self.channel_layout.clip_option)
		clip_values = list(clip_values[slots_to_keep])
		clip_values = [bool(c) for c in clip_values]

		normalization_mode = np.array(self.channel_layout.normalization_mode)
		normalization_mode = list(normalization_mode[slots_to_keep])
		normalization_mode = [bool(m) for m in normalization_mode]

		dico = {}

		# Check model option
		if self.stardist_button.isChecked():
			
			# Get channels
			# channels = []
			# for c in self.combos:
			# 	if c.currentText()!="--":
			# 		channels.append(c.currentText())
			model_type = "stardist"
			spatial_calib = float(self.spatial_calib_le.text().replace(',','.'))
			#normalize = self.normalize_checkbox.isChecked()
			dico.update({"channels": channels,
						 #"normalize": normalize,
						 "spatial_calibration": spatial_calib,
						 'normalization_percentile': normalization_mode,
						 'normalization_clip': clip_values,
						 'normalization_values': norm_values,
						 })

		elif self.cellpose_button.isChecked():
			
			# Get channels (cyto and nucleus)
			# channels = []
			# for c in self.combos[:2]:
			# 	if c.currentText()!="--":
			# 		channels.append(c.currentText())

			model_type = "cellpose"
			#cellpose requires at least two channels
			if len(channels)==1:
				channels += ['None']
				normalization_mode += [True]
				clip_values += [True]
				norm_values += [[1,99]]

			diameter = float(self.cp_diameter_le.text().replace(',','.'))
			cellprob_threshold = float(self.cp_cellprob_le.text().replace(',','.'))
			flow_threshold = float(self.cp_flow_le.text().replace(',','.'))
			#normalize = self.normalize_checkbox.isChecked()
			spatial_calib = float(self.spatial_calib_le.text().replace(',','.'))
			# assume 30 px diameter in cellpose model
			spatial_calib = spatial_calib * diameter / self.scale_model
			diameter = 30.0

			dico.update({"channels": channels,
						 "diameter": diameter,
						 "cellprob_threshold": cellprob_threshold,
						 "flow_threshold": flow_threshold,
						 #"normalize": normalize,
						 "spatial_calibration": spatial_calib,
						 'normalization_percentile': normalization_mode,
						 'normalization_clip': clip_values,
						 'normalization_values': norm_values,
						 })


		dico.update({"model_type": model_type})
		print(f"{dico=}")
		json_object = json.dumps(dico, indent=4)

		# Writing to sample.json
		# if os.path.exists(self.folder_dest):

		# 	msgBox = QMessageBox()
		# 	msgBox.setIcon(QMessageBox.Warning)
		# 	msgBox.setText("A model with the same name already exists. Do you want to replace it?")
		# 	msgBox.setWindowTitle("Confirm")
		# 	msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		# 	returnValue = msgBox.exec()
		# 	if returnValue == QMessageBox.No:
		# 		return None
		# 	elif returnValue == QMessageBox.Yes:
		# 		shutil.rmtree(self.folder_dest)
		#os.mkdir(self.folder_dest)

		print("Configuration successfully written in ",self.folder_dest+os.sep+"config_input.json")
		with open(self.folder_dest+os.sep+"config_input.json", "w") as outfile:
			outfile.write(json_object)

	def open_threshold_config_wizard(self):

		self.parent_window.parent_window.locate_image()
		if self.parent_window.parent_window.current_stack is None:
			return None
		else:
			self.ThreshWizard = ThresholdConfigWizard(self)
			self.ThreshWizard.show()
