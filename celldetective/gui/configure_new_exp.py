from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QHBoxLayout, QFileDialog, QVBoxLayout, QScrollArea, QCheckBox, QGridLayout, QLabel, QLineEdit, QPushButton, QWidget
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from celldetective.gui.gui_utils import center_window, help_generic
from celldetective.gui.styles import Styles
from celldetective.utils import get_software_location
import json

from superqt import QLabeledSlider
from PyQt5.QtCore import Qt, QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from configparser import ConfigParser
import os
from functools import partial
import numpy as np
from celldetective.gui import Styles

class ConfigNewExperiment(QMainWindow, Styles):

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("New experiment")
		center_window(self)
		self.setFixedWidth(500)
		self.setMaximumHeight(int(0.8*self.parent_window.screen_height))
		self.onlyFloat = QDoubleValidator()
		
		self.newExpFolder = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		self.populate_widget()

	def populate_widget(self):
		
		# Create button widget and layout
		self.scroll_area = QScrollArea(self)
		button_widget = QWidget()
		self.grid = QGridLayout()
		button_widget.setLayout(self.grid)

		self.grid.setContentsMargins(30,30,30,30)
		self.grid.addWidget(QLabel("Folder:"), 0, 0, 1, 3)
		self.supFolder = QLineEdit()
		self.supFolder.setAlignment(Qt.AlignLeft)	
		self.supFolder.setEnabled(True)
		self.supFolder.setText(self.newExpFolder)
		self.grid.addWidget(self.supFolder, 1, 0, 1, 1)

		self.browse_button = QPushButton("Browse...")
		self.browse_button.clicked.connect(self.browse_experiment_folder)
		self.browse_button.setIcon(icon(MDI6.folder, color="white"))
		self.browse_button.setStyleSheet(self.button_style_sheet)
		#self.browse_button.setIcon(QIcon_from_svg(abs_path+f"/icons/browse.svg", color='white'))
		self.grid.addWidget(self.browse_button, 1, 1, 1, 1)

		self.grid.addWidget(QLabel("Experiment name:"), 2, 0, 1, 3)
		
		self.expName = QLineEdit()
		self.expName.setPlaceholderText('folder_name_for_the_experiment')
		self.expName.setAlignment(Qt.AlignLeft)	
		self.expName.setEnabled(True)
		self.expName.setFixedWidth(400)
		self.expName.setText("Untitled_Experiment")
		self.grid.addWidget(self.expName, 3, 0, 1, 3)

		self.generate_movie_settings()
		self.grid.addLayout(self.ms_grid,29,0,1,3)

		self.generate_channel_params_box()
		self.grid.addLayout(self.channel_grid,30,0,1,3)

		self.validate_button = QPushButton("Submit")
		self.validate_button.clicked.connect(self.create_config)
		self.validate_button.setStyleSheet(self.button_style_sheet)
		#self.validate_button.setIcon(QIcon_from_svg(abs_path+f"/icons/process.svg", color='white'))

		self.grid.addWidget(self.validate_button, 31, 0, 1, 3, alignment = Qt.AlignBottom)		
		button_widget.adjustSize()

		self.scroll_area.setAlignment(Qt.AlignCenter)
		self.scroll_area.setWidget(button_widget)
		self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self.scroll_area)
		self.show()

		QApplication.processEvents()
		self.adjustScrollArea()

	def adjustScrollArea(self):
		
		"""
		Auto-adjust scroll area to fill space 
		(from https://stackoverflow.com/questions/66417576/make-qscrollarea-use-all-available-space-of-qmainwindow-height-axis)
		"""

		step = 5
		while self.scroll_area.verticalScrollBar().isVisible() and self.height() < self.maximumHeight():
			self.resize(self.width(), self.height() + step)

	def generate_movie_settings(self):
		
		"""
		Parameters related to the movie parameters
		"""

		onlyInt = QIntValidator()
		onlyInt.setRange(0, 100000)

		self.ms_grid = QGridLayout()
		self.ms_grid.setContentsMargins(21,30,20,30)

		ms_lbl = QLabel("MOVIE SETTINGS")
		ms_lbl.setStyleSheet("""
			font-weight: bold;
			""")
		self.ms_grid.addWidget(ms_lbl, 0,0,1,3, alignment=Qt.AlignCenter)

		self.number_of_wells = QLabel("Number of wells:")
		self.ms_grid.addWidget(self.number_of_wells, 1, 0, 1, 3)

		self.help_btn = QPushButton()
		self.help_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_btn.setIconSize(QSize(20, 20))
		self.help_btn.clicked.connect(self.help_structure)
		self.help_btn.setStyleSheet(self.button_select_all)
		self.help_btn.setToolTip("Help.")
		self.ms_grid.addWidget(self.help_btn, 1, 0, 1, 3, alignment=Qt.AlignRight)


		self.SliderWells = QLabeledSlider(Qt.Horizontal, self)
		self.SliderWells.setMinimum(1)
		self.SliderWells.setMaximum(32)
		self.ms_grid.addWidget(self.SliderWells, 2, 0, 1, 3, alignment=Qt.AlignTop)

		self.number_of_positions = QLabel("Number of positions per well:")
		self.ms_grid.addWidget(self.number_of_positions, 3, 0, 1, 3)

		self.SliderPos = QLabeledSlider(Qt.Horizontal, self)
		self.SliderPos.setMinimum(1)
		self.SliderPos.setMaximum(50)

		self.ms_grid.addWidget(self.SliderPos, 4, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Calibration from pixel to µm:"), 5, 0, 1, 3)
		self.PxToUm_field = QLineEdit()
		self.PxToUm_field.setValidator(self.onlyFloat)
		self.PxToUm_field.setPlaceholderText('1 px = XXX µm')
		self.PxToUm_field.setAlignment(Qt.AlignLeft)	
		self.PxToUm_field.setEnabled(True)
		self.PxToUm_field.setFixedWidth(400)
		self.PxToUm_field.setText("1,0")
		self.ms_grid.addWidget(self.PxToUm_field, 6, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Calibration from frame to minutes:"), 7, 0, 1, 3)
		self.FrameToMin_field = QLineEdit()
		self.FrameToMin_field.setAlignment(Qt.AlignLeft)	
		self.FrameToMin_field.setEnabled(True)
		self.FrameToMin_field.setFixedWidth(400)
		self.FrameToMin_field.setValidator(self.onlyFloat)
		self.FrameToMin_field.setPlaceholderText('1 frame = XXX min')
		self.FrameToMin_field.setText("1,0")
		self.ms_grid.addWidget(self.FrameToMin_field, 8, 0, 1, 3)

		self.movie_length = QLabel("Number of frames:")
		self.movie_length.setToolTip('Optional: depending on how the movies are encoded, the automatic extraction of the number of frames can be difficult.\nThe software will then rely on this value.')
		self.ms_grid.addWidget(self.movie_length,9, 0, 1, 3)
		self.MovieLengthSlider = QLabeledSlider(Qt.Horizontal, self)
		self.MovieLengthSlider.setMinimum(2)
		self.MovieLengthSlider.setMaximum(128)
		self.ms_grid.addWidget(self.MovieLengthSlider, 10, 0, 1, 3)

		self.prefix_lbl = QLabel("Prefix for the movies:")
		self.prefix_lbl.setToolTip('The stack file name must start with this prefix to be properly loaded.')
		self.ms_grid.addWidget(self.prefix_lbl, 11, 0, 1, 3)
		self.movie_prefix_field = QLineEdit()
		self.movie_prefix_field.setAlignment(Qt.AlignLeft)	
		self.movie_prefix_field.setEnabled(True)
		self.movie_prefix_field.setFixedWidth(400)
		self.movie_prefix_field.setText("")
		self.ms_grid.addWidget(self.movie_prefix_field, 12, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("X shape in pixels:"), 13, 0, 1, 3)
		self.shape_x_field = QLineEdit()
		self.shape_x_field.setValidator(onlyInt)
		self.shape_x_field.setAlignment(Qt.AlignLeft)	
		self.shape_x_field.setEnabled(True)
		self.shape_x_field.setFixedWidth(400)
		self.shape_x_field.setText("2048")
		self.ms_grid.addWidget(self.shape_x_field, 14, 0, 1, 3)

		self.ms_grid.addWidget(QLabel("Y shape in pixels:"), 15, 0, 1, 3)
		self.shape_y_field = QLineEdit()
		self.shape_y_field.setValidator(onlyInt)
		self.shape_y_field.setAlignment(Qt.AlignLeft)	
		self.shape_y_field.setEnabled(True)
		self.shape_y_field.setFixedWidth(400)
		self.shape_y_field.setText("2048")
		self.ms_grid.addWidget(self.shape_y_field, 16, 0, 1, 3)

	def help_structure(self):

		"""
		Helper to choose an experiment structure.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','exp-structure.json'])

		with open(dict_path) as f:
			d = json.load(f)

		suggestion = help_generic(d)
		if isinstance(suggestion, str):
			print(f"{suggestion=}")
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Information)
			msgBox.setTextFormat(Qt.RichText)
			msgBox.setText(suggestion+"\nSee <a href='https://celldetective.readthedocs.io/en/latest/get-started.html#data-organization'>the docs</a> for more information.")
			msgBox.setWindowTitle("Info")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None		

	def generate_channel_params_box(self):

		"""
		Parameters related to the movie channels
		Rewrite all of it

		"""

		self.channel_grid = QGridLayout()
		self.channel_grid.setContentsMargins(21,30,20,30)

		channel_lbl = QLabel("CHANNELS")
		channel_lbl.setStyleSheet("""
			font-weight: bold;
			""")
		self.channel_grid.addWidget(channel_lbl, 0,0,1,3, alignment=Qt.AlignCenter)


		self.channels = ['brightfield', "live nuclei channel\n(Hoechst, NucSpot®)", "dead nuclei channel\n(PI)", "effector fluorescence\n(CFSE)",
						"adhesion\n(RICM, IRM)", "fluorescence miscellaneous 1\n(LAMP-1, Actin...)", "fluorescence miscellaneous 2\n(LAMP-1, Actin...)"]
		self.channel_mapping = ['brightfield_channel', 'live_nuclei_channel', 'dead_nuclei_channel', 'effector_fluo_channel',
								'adhesion_channel', 'fluo_channel_1', 'fluo_channel_2']
		self.checkBoxes = [QCheckBox() for i in range(len(self.channels))]
		self.sliders = [QLabeledSlider(Qt.Orientation.Horizontal) for i in range(len(self.channels))]

		for i in range(len(self.channels)):

			self.checkBoxes[i].setText(self.channels[i])
			self.checkBoxes[i].toggled.connect(partial(self.show_slider, i))

			self.channel_grid.addWidget(self.checkBoxes[i], i+1, 0, 1, 1)

			self.sliders[i].setMinimum(0)
			self.sliders[i].setMaximum(len(self.channels))
			self.sliders[i].setEnabled(False)

			self.channel_grid.addWidget(self.sliders[i], i+1, 1, 1, 2, alignment = Qt.AlignRight)

		# Add channel button
		self.addChannelBtn = QPushButton('Add channel')
		self.addChannelBtn.setIcon(icon(MDI6.plus,color="white"))
		self.addChannelBtn.setIconSize(QSize(25, 25))
		self.addChannelBtn.setStyleSheet(self.button_style_sheet)
		self.addChannelBtn.clicked.connect(self.add_custom_channel)
		self.channel_grid.addWidget(self.addChannelBtn, 1000, 0, 1, 1)

	def add_custom_channel(self):

		self.CustomChannelWidget = QWidget()
		self.CustomChannelWidget.setWindowTitle("Custom channel")
		layout = QVBoxLayout()
		self.CustomChannelWidget.setLayout(layout)

		self.name_le = QLineEdit('custom_channel')
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('channel name: '), 33)
		hbox.addWidget(self.name_le, 66)
		layout.addLayout(hbox)

		self.createBtn = QPushButton('create')
		self.createBtn.setStyleSheet(self.button_style_sheet)
		self.createBtn.clicked.connect(self.write_custom_channel)
		layout.addWidget(self.createBtn)
		center_window(self.CustomChannelWidget)
		self.CustomChannelWidget.show()

		print('new channel added')

	def write_custom_channel(self):
		
		self.new_channel_name = self.name_le.text()
		name_map = self.new_channel_name
		name_map = name_map.replace('_channel','')
		name_map = name_map.replace('channel','')
		name_map = name_map.replace(' ','')
		if not name_map.endswith('_channel'):
			name_map += '_channel'

		self.channels.append(self.new_channel_name)
		self.channel_mapping.append(name_map)
		self.checkBoxes.append(QCheckBox())
		self.sliders.append(QLabeledSlider(Qt.Orientation.Horizontal))
		self.CustomChannelWidget.close()

		self.checkBoxes[-1].setText(self.channels[-1])
		self.checkBoxes[-1].toggled.connect(partial(self.show_slider, len(self.channels)-1))
		self.channel_grid.addWidget(self.checkBoxes[-1], len(self.channels)+1, 0, 1, 1)

		self.sliders[-1].setMinimum(0)
		for i in range(len(self.channels)):
			self.sliders[i].setMaximum(len(self.channels))
		self.sliders[-1].setEnabled(False)
		self.channel_grid.addWidget(self.sliders[-1], len(self.channels)+1, 1, 1, 2, alignment = Qt.AlignRight)

	def show_slider(self, index):
		if self.checkBoxes[index].isChecked():
			self.sliders[index].setEnabled(True)
		else:
			self.sliders[index].setEnabled(False)

	def browse_experiment_folder(self):

		"""
		Set a new base directory.
		"""

		self.newExpFolder = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		self.supFolder.setText(self.newExpFolder)

	def create_config(self):

		"""
		Create the folder tree, the config, issue a warning if the experiment folder already exists.
		"""

		channel_indices = []
		for i in range(len(self.channels)):
			if self.checkBoxes[i].isChecked():
				channel_indices.append(self.sliders[i].value())
		if not channel_indices:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please set at least one channel before proceeding.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		if len(channel_indices) != len(set(channel_indices)):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Some channel indices are repeated. Please check your configuration.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

		sorted_list = set(channel_indices)
		expected_list = set(np.arange(max(sorted_list)+1))
		print(sorted_list, expected_list, sorted_list==expected_list)

		if not sorted_list==expected_list:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("There is a gap in your channel indices. Please check your configuration.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None			
		
		try:
			
			folder = self.supFolder.text()
			folder = folder.replace('\\','/')
			folder = rf"{folder}"

			name = str(self.expName.text())
			name = name.replace(' ','')

			self.directory = os.sep.join([folder,name])
			os.mkdir(self.directory)
			os.chdir(self.directory)
			self.create_subfolders()
			self.annotate_wells()

		except FileExistsError:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("This experiment already exists... Please select another name.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return None

	def create_subfolders(self):

		"""
		Create the well folders and position folders within the wells.
		"""
		
		self.nbr_wells = self.SliderWells.value()
		self.nbr_positions = self.SliderPos.value()
		for k in range(self.nbr_wells):
			well_name = f"W{k+1}"+os.sep
			os.mkdir(well_name)
			for p in range(self.nbr_positions):
				position_name = well_name+f"{k+1}0{p}"+os.sep
				os.mkdir(position_name)
				os.mkdir(position_name+os.sep+"movie"+os.sep)

	def annotate_wells(self):
		self.w = SetupConditionLabels(self, self.nbr_wells)
		self.w.show()


	def create_config_file(self):

		"""

		Write all user input parameters to a configuration file associated to an experiment.
		"""

		config = ConfigParser(interpolation=None)

		# add a new section and some values
		config.add_section('MovieSettings')
		config.set('MovieSettings', 'PxToUm', self.PxToUm_field.text().replace(',','.'))
		config.set('MovieSettings', 'FrameToMin', self.FrameToMin_field.text().replace(',','.'))
		config.set('MovieSettings', 'len_movie', str(self.MovieLengthSlider.value()))
		config.set('MovieSettings', 'shape_x', self.shape_x_field.text())
		config.set('MovieSettings', 'shape_y', self.shape_y_field.text())
		config.set('MovieSettings', 'movie_prefix', self.movie_prefix_field.text())

		config.add_section('Channels')
		for i in range(len(self.channels)):
			if self.checkBoxes[i].isChecked():
				config.set('Channels', self.channel_mapping[i], str(self.sliders[i].value()))
			else:
				config.set('Channels', self.channel_mapping[i], "nan")

		config.add_section('Labels')
		config.set('Labels', 'cell_types', self.cell_types)
		config.set('Labels', 'antibodies', self.antibodies)
		config.set('Labels', 'concentrations', self.concentrations)
		config.set('Labels', 'pharmaceutical_agents', self.pharmaceutical_agents)

		config.add_section('Metadata')
		config.set('Metadata', 'concentration_units', self.concentration_units)

		# save to a file
		with open('config.ini', 'w') as configfile:
			config.write(configfile)

		self.parent_window.set_experiment_path(self.directory)
		print(f'New experiment successfully configured in folder {self.directory}...')
		self.close()

class SetupConditionLabels(QWidget, Styles):
	def __init__(self, parent_window, n_wells):
		super().__init__()
		self.parent_window = parent_window
		self.n_wells = n_wells
		self.setWindowTitle("Well conditions")
		self.layout = QVBoxLayout()
		self.layout.setContentsMargins(30,30,30,30)
		self.setLayout(self.layout)
		self.onlyFloat = QDoubleValidator()
		self.populate()
		center_window(self)

	def populate(self):

		self.cell_type_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.antibodies_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.concentrations_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.pharmaceutical_agents_cbs = [QLineEdit() for i in range(self.n_wells)]
		self.concentration_units_le = QLineEdit('pM')
		self.concentration_units_le.setPlaceholderText('concentration units')

		for i in range(self.n_wells):
			hbox = QHBoxLayout()
			hbox.setContentsMargins(15,5,15,5)
			hbox.addWidget(QLabel(f'well {i+1}'), 5, alignment=Qt.AlignLeft)
			hbox.addWidget(QLabel('cell type: '), 5)
			hbox.addWidget(self.cell_type_cbs[i], 10)
			self.cell_type_cbs[i].setPlaceholderText('e.g. T-cell, NK')

			hbox.addWidget(QLabel('antibody: '), 5)
			hbox.addWidget(self.antibodies_cbs[i], 10)
			self.antibodies_cbs[i].setPlaceholderText('e.g. anti-CD4')
			
			hbox.addWidget(QLabel('concentration: '), 5)
			hbox.addWidget(self.concentrations_cbs[i], 10)
			self.concentrations_cbs[i].setPlaceholderText('e.g. 100 (pM)')
			self.concentrations_cbs[i].setValidator(self.onlyFloat)
			
			hbox.addWidget(QLabel('pharmaceutical agents: '), 5)
			hbox.addWidget(self.pharmaceutical_agents_cbs[i], 10)
			self.pharmaceutical_agents_cbs[i].setPlaceholderText('e.g. dextran')

			self.layout.addLayout(hbox)

		concentration_units_layout = QHBoxLayout()
		concentration_units_layout.addWidget(QLabel('concentration\nunits: '),5,alignment=Qt.AlignLeft)
		concentration_units_layout.addWidget(self.concentration_units_le,10)
		concentration_units_layout.addWidget(QLabel(''), 85)
		self.layout.addLayout(concentration_units_layout)

		btn_hbox = QHBoxLayout()
		btn_hbox.setContentsMargins(0,20,0,0)
		self.skip_btn = QPushButton('Skip')
		self.skip_btn.setStyleSheet(self.button_style_sheet_2)
		self.skip_btn.clicked.connect(self.set_default_values)
		btn_hbox.addWidget(self.skip_btn)

		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.set_user_values)
		btn_hbox.addWidget(self.submit_btn)
		self.layout.addLayout(btn_hbox)

	def set_default_values(self):
		
		for i in range(self.n_wells):
			self.cell_type_cbs[i].setText(str(i))
			self.antibodies_cbs[i].setText(str(i))
			self.concentrations_cbs[i].setText(str(i))
			self.pharmaceutical_agents_cbs[i].setText(str(i))
		self.set_attributes()
		self.parent_window.create_config_file()
		self.close()

	def set_user_values(self):
		for i in range(self.n_wells):
			if self.cell_type_cbs[i].text()=='':
				self.cell_type_cbs[i].setText(str(i))
			if self.antibodies_cbs[i].text()=='':
				self.antibodies_cbs[i].setText(str(i))
			if self.concentrations_cbs[i].text()=='':
				self.concentrations_cbs[i].setText(str(i))
			if self.pharmaceutical_agents_cbs[i].text()=='':
				self.pharmaceutical_agents_cbs[i].setText(str(i))
		self.set_attributes()
		self.parent_window.create_config_file()
		self.close()

	def set_attributes(self):

		cell_type_text = [c.text() for c in self.cell_type_cbs]
		self.parent_window.cell_types = ','.join(cell_type_text)

		antibodies_text = [c.text() for c in self.antibodies_cbs]
		self.parent_window.antibodies = ','.join(antibodies_text)

		concentrations_text = [c.text() for c in self.concentrations_cbs]
		self.parent_window.concentrations = ','.join(concentrations_text)

		pharamaceutical_text = [c.text() for c in self.pharmaceutical_agents_cbs]
		self.parent_window.pharmaceutical_agents = ','.join(pharamaceutical_text)

		self.parent_window.concentration_units = self.concentration_units_le.text()





