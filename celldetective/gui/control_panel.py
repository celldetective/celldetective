from PyQt5.QtWidgets import QMainWindow, QPushButton, QHBoxLayout, QLabel, QWidget, QGridLayout, QFrame, \
	QTabWidget, QVBoxLayout, QMessageBox, QScrollArea, QDesktopWidget
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.gui_utils import center_window, QHSeperationLine, QCheckableComboBox
from celldetective.utils import _extract_labels_from_config, ConfigSectionMap, extract_experiment_channels, extract_identity_col
from celldetective.gui import ConfigEditor, ProcessPanel, PreprocessingPanel, AnalysisPanel, NeighPanel
from celldetective.io import get_experiment_wells, get_config, get_spatial_calibration, get_temporal_calibration, get_experiment_concentrations, get_experiment_cell_types, get_experiment_antibodies, get_experiment_pharmaceutical_agents
from natsort import natsorted
from glob import glob
import os
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import gc
import subprocess
from celldetective.gui.viewers import StackVisualizer
from celldetective.utils import extract_experiment_channels
from celldetective.gui import Styles
import pandas as pd


class ControlPanel(QMainWindow, Styles):

	def __init__(self, parent_window=None, exp_dir=""):
		
		super().__init__()
		
		self.exp_dir = exp_dir
		if not self.exp_dir.endswith(os.sep):
			self.exp_dir = self.exp_dir+os.sep
		self.setWindowTitle("celldetective")
		self.setWindowIcon(self.celldetective_icon)
		self.parent_window = parent_window

		self.init_wells_and_positions()
		self.load_configuration()

		self.w = QWidget()
		self.grid = QGridLayout(self.w)
		self.grid.setSpacing(5)
		self.grid.setContentsMargins(10,10,10,10) #left top right bottom

		self.to_disable = []
		self.generate_header()
		self.ProcessEffectors = ProcessPanel(self,'effectors')
		self.ProcessTargets = ProcessPanel(self,'targets')
		self.NeighPanel = NeighPanel(self)
		self.PreprocessingPanel = PreprocessingPanel(self)
		
		ProcessFrame = QFrame()
		grid_process = QVBoxLayout(ProcessFrame)
		grid_process.setContentsMargins(15,30,15,15)

		AnalyzeFrame = QFrame()
		grid_analyze = QVBoxLayout(AnalyzeFrame)
		grid_analyze.setContentsMargins(15,30,15,15)
		self.SurvivalBlock = AnalysisPanel(self,title='Survival')

		grid_process.addWidget(self.PreprocessingPanel)
		grid_process.addWidget(self.ProcessEffectors)
		grid_process.addWidget(self.ProcessTargets)
		grid_process.addWidget(self.NeighPanel)
		grid_analyze.addWidget(self.SurvivalBlock)

		self.scroll=QScrollArea()
		self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.scroll.setWidgetResizable(True)
		desktop = QDesktopWidget()
		self.scroll.setMinimumHeight(550)
		#self.scroll.setMinimumHeight(int(0.4*screen_height))

		tabWidget = QTabWidget()
		tab_index_process = tabWidget.addTab(ProcessFrame, "Process")
		tabWidget.setTabIcon(tab_index_process, icon(MDI6.cog_outline, color='black'))

		tab_index_analyze = tabWidget.addTab(AnalyzeFrame, "Analyze")
		tabWidget.setTabIcon(tab_index_analyze, icon(MDI6.poll, color='black'))
		tabWidget.setStyleSheet(self.qtab_style)

		self.grid.addWidget(tabWidget, 7,0,1,3, alignment=Qt.AlignTop)
		self.grid.setSpacing(5)
		self.scroll.setWidget(self.w)
		self.setCentralWidget(self.scroll)
		self.create_config_dir()
		self.update_position_options()
		#self.setMinimumHeight(int(self.sizeHint().height()))

		self.initial_height = self.size().height()
		self.initial_width = self.size().width()
		self.screen_height = desktop.screenGeometry().height()
		self.screen_width = desktop.screenGeometry().width()
		self.scroll.setMinimumWidth(440)
		
		self.setAttribute(Qt.WA_DeleteOnClose)
		center_window(self)


		self.well_list.setCurrentIndex(0)
		#self.position_list.setCurrentIndex(0)

	def init_wells_and_positions(self):

		"""
		Detect the wells in the experiment folder and the associated positions.
		"""
		
		self.wells = get_experiment_wells(self.exp_dir) #natsorted(glob(self.exp_dir + "W*" + os.sep))
		self.positions = []
		for w in self.wells:
			w = os.path.split(w[:-1])
			root = w[0]
			w = w[1]
			positions_path = natsorted(glob(os.sep.join([root, w, f"{w[1:]}*{os.sep}"])))
			self.positions.append([os.path.split(pos[:-1])[1] for pos in positions_path])

	def generate_header(self):
		
		"""
		Show the experiment name, create two QComboBox for respectively the 
		biological condition (well) and position of interest, access the experiment config.
		"""

		condition_label = QLabel("condition: ")
		position_label = QLabel("position: ")

		name = self.exp_dir.split(os.sep)[-2]
		experiment_label = QLabel(f"Experiment:")
		experiment_label.setStyleSheet("""
			font-weight: bold;
			""")

		self.folder_exp_btn = QPushButton()
		self.folder_exp_btn.setIcon(icon(MDI6.folder,color="black"))
		self.folder_exp_btn.setIconSize(QSize(20, 20))
		self.folder_exp_btn.setToolTip("Experiment folder")
		self.folder_exp_btn.clicked.connect(self.open_experiment_folder)
		self.folder_exp_btn.setStyleSheet(self.button_select_all)


		self.edit_config_button = QPushButton()
		self.edit_config_button.setIcon(icon(MDI6.cog_outline,color="black"))
		self.edit_config_button.setIconSize(QSize(20, 20))
		self.edit_config_button.setToolTip("Configuration file")
		self.edit_config_button.clicked.connect(self.open_config_editor)
		self.edit_config_button.setStyleSheet(self.button_select_all)

		self.well_list = QCheckableComboBox(obj='well', parent_window=self)
		thresh = 32
		self.well_truncated = [w[:thresh - 3]+'...' if len(w)>thresh else w for w in self.well_labels]
		for i in range(len(self.well_truncated)):
			self.well_list.addItem(self.well_truncated[i], tooltip=self.well_labels[i])

		self.position_list = QCheckableComboBox(obj='position', parent_window=self)
		self.position_list.addItems(self.positions[0])
		self.to_disable.append(self.position_list)
		#self.locate_selected_position()

		self.well_list.activated.connect(self.display_positions)

		self.position_list.activated.connect(self.update_position_options)

		self.view_stack_btn = QPushButton()
		self.view_stack_btn.setStyleSheet(self.button_select_all)
		self.view_stack_btn.setIcon(icon(MDI6.image_check, color="black"))
		self.view_stack_btn.setToolTip("View stack.")
		self.view_stack_btn.setIconSize(QSize(20, 20))
		self.view_stack_btn.clicked.connect(self.view_current_stack)
		self.view_stack_btn.setEnabled(False)

		self.select_all_wells_btn = QPushButton()
		self.select_all_wells_btn.setIcon(icon(MDI6.select_all,color="black"))
		self.select_all_wells_btn.setIconSize(QSize(20, 20))
		self.select_all_wells_btn.setToolTip("Select all wells.")
		self.select_all_wells_btn.clicked.connect(self.select_all_wells)
		self.select_all_wells_btn.setStyleSheet(self.button_select_all)
		self.select_all_wells_option = False


		self.select_all_pos_btn = QPushButton()
		self.select_all_pos_btn.setIcon(icon(MDI6.select_all,color="black"))
		self.select_all_pos_btn.setIconSize(QSize(20, 20))
		self.select_all_pos_btn.setToolTip("Select all positions.")
		self.select_all_pos_btn.clicked.connect(self.select_all_positions)
		self.select_all_pos_btn.setStyleSheet(self.button_select_all)
		self.select_all_pos_option = False


		well_lbl = QLabel('Well: ')
		well_lbl.setAlignment(Qt.AlignRight)

		pos_lbl = QLabel('Position: ')
		pos_lbl.setAlignment(Qt.AlignRight)

		hsep = QHSeperationLine()

		## LAYOUT

		# Header layout
		vbox = QVBoxLayout()
		self.grid.addLayout(vbox, 0,0,1,3)

		# Experiment row
		exp_hbox = QHBoxLayout()
		exp_hbox.addWidget(experiment_label, 25, alignment=Qt.AlignRight)
		exp_subhbox = QHBoxLayout()
		if len(name)>thresh:
			name_cut = name[:thresh - 3]+'...'
		else:
			name_cut = name
		exp_name_lbl = QLabel(name_cut)
		exp_name_lbl.setToolTip(name)
		exp_subhbox.addWidget(exp_name_lbl, 90, alignment=Qt.AlignLeft)
		exp_subhbox.addWidget(self.folder_exp_btn, 5, alignment=Qt.AlignRight)
		exp_subhbox.addWidget(self.edit_config_button, 5, alignment=Qt.AlignRight)
		exp_hbox.addLayout(exp_subhbox, 75)
		vbox.addLayout(exp_hbox)

		# Well row
		well_hbox = QHBoxLayout()
		well_hbox.addWidget(well_lbl, 25, alignment=Qt.AlignVCenter)
		well_subhbox = QHBoxLayout()
		well_subhbox.addWidget(self.well_list, 95)
		well_subhbox.addWidget(self.select_all_wells_btn, 5)
		well_hbox.addLayout(well_subhbox, 75)
		vbox.addLayout(well_hbox)

		# Position row
		position_hbox = QHBoxLayout()
		position_hbox.addWidget(pos_lbl, 25, alignment=Qt.AlignVCenter)
		pos_subhbox = QHBoxLayout()
		pos_subhbox.addWidget(self.position_list, 90)
		pos_subhbox.addWidget(self.select_all_pos_btn, 5)
		pos_subhbox.addWidget(self.view_stack_btn, 5)
		position_hbox.addLayout(pos_subhbox, 75)
		vbox.addLayout(position_hbox)

		vbox.addWidget(hsep)

	def select_all_wells(self):

		if not self.select_all_wells_option:
			self.well_list.selectAll()
			self.select_all_wells_option = True
			self.select_all_wells_btn.setIcon(icon(MDI6.select_all,color=self.celldetective_blue))
			self.select_all_wells_btn.setIconSize(QSize(20, 20))
			self.display_positions()
		else:
			self.well_list.unselectAll()
			self.select_all_wells_option = False
			self.select_all_wells_btn.setIcon(icon(MDI6.select_all,color="black"))
			self.select_all_wells_btn.setIconSize(QSize(20, 20))
			self.display_positions()

	def select_all_positions(self):

		if not self.select_all_pos_option:
			self.position_list.selectAll()
			self.select_all_pos_option = True
			self.select_all_pos_btn.setIcon(icon(MDI6.select_all,color=self.celldetective_blue))
			self.select_all_pos_btn.setIconSize(QSize(20, 20))
		else:
			self.position_list.unselectAll()
			self.select_all_pos_option = False
			self.select_all_pos_btn.setIcon(icon(MDI6.select_all,color="black"))
			self.select_all_pos_btn.setIconSize(QSize(20, 20))

	def locate_image(self):

		"""
		Load the first frame of the first movie found in the experiment folder as a sample.
		"""

		movies = glob(self.pos + os.sep.join(['movie', f"{self.movie_prefix}*.tif"]))

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

	def view_current_stack(self):
		
		self.locate_image()
		if self.current_stack is not None:
			self.viewer = StackVisualizer(
										  stack_path=self.current_stack,
										  window_title=f'Position {self.position_list.currentText()}',
										  frame_slider = True,
										  contrast_slider = True,
										  channel_cb = True,
										  channel_names = self.exp_channels,
										  n_channels = self.nbr_channels,
										  PxToUm = self.PxToUm,
										 )
			self.viewer.show()

	def open_experiment_folder(self):

		try:
			subprocess.Popen(f'explorer {os.path.realpath(self.exp_dir)}')
		except:
			try:
				os.system('xdg-open "%s"' % self.exp_dir)
			except:
				return None		

	def load_configuration(self):

		'''
		This methods load the configuration read in the config.ini file of the experiment.
		'''

		print('Reading experiment configuration...')
		self.exp_config = get_config(self.exp_dir)

		self.PxToUm = get_spatial_calibration(self.exp_dir)
		self.FrameToMin = get_temporal_calibration(self.exp_dir)

		self.len_movie = int(ConfigSectionMap(self.exp_config,"MovieSettings")["len_movie"])
		self.shape_x = int(ConfigSectionMap(self.exp_config,"MovieSettings")["shape_x"])
		self.shape_y = int(ConfigSectionMap(self.exp_config,"MovieSettings")["shape_y"])
		self.movie_prefix = ConfigSectionMap(self.exp_config,"MovieSettings")["movie_prefix"]

		# Read channels
		self.exp_channels, channel_indices = extract_experiment_channels(self.exp_dir)
		self.nbr_channels = len(self.exp_channels)

		number_of_wells = len(self.wells)
		self.well_labels = _extract_labels_from_config(self.exp_config,number_of_wells)

		self.concentrations = get_experiment_concentrations(self.exp_dir)
		self.cell_types = get_experiment_cell_types(self.exp_dir)
		self.antibodies = get_experiment_antibodies(self.exp_dir)
		self.pharmaceutical_agents = get_experiment_pharmaceutical_agents(self.exp_dir)

		self.metadata = ConfigSectionMap(self.exp_config,"Metadata")
		print('Experiment configuration successfully read...')

	def closeEvent(self, event):

		"""
		Close child windows if closed.
		"""
		
		for process_block in [self.ProcessTargets, self.ProcessEffectors]:
			try:
				if process_block.SegModelLoader:
					process_block.SegModelLoader.close()
			except:
				pass
			try:
				if process_block.ConfigTracking:
					process_block.ConfigTracking.close()
			except:
				pass
			try:
				if process_block.ConfigSignalTrain:
					process_block.ConfigSignalTrain.close()
			except:
				pass
			try:
				if process_block.ConfigMeasurements:
					process_block.ConfigMeasurements.close()
			except:
				pass
			try:
				if process_block.ConfigSignalAnnotator:
					process_block.ConfigSignalAnnotator.close()
			except:
				pass
			try:
				if process_block.tab_ui:
					process_block.tab_ui.close()
			except:
				pass

		try:
			if self.cfg_editor:
				self.cfg_editor.close()
		except:
			pass

		gc.collect()


	def display_positions(self):

		"""
		Show the positions as the well is changed.
		"""

		if self.well_list.isMultipleSelection():

			self.position_list.clear()
			position_linspace = np.linspace(0,len(self.positions[0])-1,len(self.positions[0]),dtype=int)
			position_linspace = [str(s) for s in position_linspace]
			self.position_list.addItems(position_linspace)
			if self.select_all_pos_option:
				self.select_all_pos_btn.click()
			self.select_all_pos_btn.click()
		
		elif not self.well_list.isAnySelected():

			self.position_list.unselectAll()
			if self.select_all_pos_option:
				self.select_all_pos_btn.click()

		else:
			pos_index = self.well_list.getSelectedIndices()[0]
			self.position_list.clear()
			self.position_list.addItems(self.positions[pos_index])
			if self.select_all_pos_option:
				self.select_all_pos_btn.click()
			self.position_list.setCurrentIndex(0)
		
		self.update_position_options()
	
	def open_config_editor(self):
		self.cfg_editor = ConfigEditor(self)
		self.cfg_editor.show()

	def locate_selected_position(self):

		"""
		Set the current position if the option one well, one positon is selected
		Display error messages otherwise.

		"""

		if self.well_list.isMultipleSelection():
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Critical)
			msgBox.setText("Please select a single well...")
			msgBox.setWindowTitle("Error")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				return False
		else:
			self.well_index = self.well_list.getSelectedIndices() #[self.well_list.currentIndex()]

		for w_idx in self.well_index:

			pos = self.positions[w_idx]
			if not self.position_list.isSingleSelection():
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Critical)
				msgBox.setText("Please select a single position...")
				msgBox.setWindowTitle("Error")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return False
			else:
				pos_indices = self.position_list.getSelectedIndices()

			well = self.wells[w_idx]

			for pos_idx in pos_indices:

				self.pos = natsorted(glob(well+f"{os.path.split(well)[-1].replace('W','').replace(os.sep,'')}*{os.sep}"))[pos_idx]
				if not os.path.exists(self.pos + 'output'):
					os.mkdir(self.pos + 'output')
				if not os.path.exists(self.pos + os.sep.join(['output','tables'])):
					os.mkdir(self.pos + os.sep.join(['output','tables']))

		return True

	def create_config_dir(self):

		self.config_folder = self.exp_dir+'configs'+os.sep
		if not os.path.exists(self.config_folder):
			os.mkdir(self.config_folder)

	def update_position_options(self):
		
		self.pos = self.position_list.currentText()
		panels = [self.ProcessEffectors, self.ProcessTargets]

		if self.position_list.isMultipleSelection() or not self.position_list.isAnySelected():
			
			for p in panels:
				p.check_seg_btn.setEnabled(False)
				p.check_tracking_result_btn.setEnabled(False)

			self.ProcessTargets.view_tab_btn.setEnabled(True)
			self.ProcessEffectors.view_tab_btn.setEnabled(True)
			self.NeighPanel.view_tab_btn.setEnabled(True)
			self.ProcessEffectors.signal_analysis_action.setEnabled(True)
			self.ProcessTargets.signal_analysis_action.setEnabled(True)

			self.ProcessTargets.check_seg_btn.setEnabled(False)
			self.ProcessEffectors.check_seg_btn.setEnabled(False)

			self.ProcessTargets.check_tracking_result_btn.setEnabled(False)
			self.ProcessEffectors.check_tracking_result_btn.setEnabled(False)

			self.ProcessEffectors.check_measurements_btn.setEnabled(False)
			self.ProcessTargets.check_measurements_btn.setEnabled(False)
			#self.ProcessTargets.signal_analysis_action.setEnabled(False)
			#self.ProcessEffectors.signal_analysis_action.setEnabled(False)

			self.ProcessTargets.check_signals_btn.setEnabled(False)
			self.ProcessEffectors.check_signals_btn.setEnabled(False)
			self.NeighPanel.check_signals_btn.setEnabled(False)
			self.ProcessTargets.delete_tracks_btn.hide()
			self.ProcessEffectors.delete_tracks_btn.hide()

			self.view_stack_btn.setEnabled(False)

		elif self.well_list.isMultipleSelection():

			self.ProcessTargets.view_tab_btn.setEnabled(True)
			self.ProcessEffectors.view_tab_btn.setEnabled(True)	
			self.NeighPanel.view_tab_btn.setEnabled(True)
			self.view_stack_btn.setEnabled(False)
			self.ProcessEffectors.signal_analysis_action.setEnabled(True)
			self.ProcessTargets.signal_analysis_action.setEnabled(True)
			if hasattr(self,'delete_tracks_btn'):
				self.delete_tracks_btn.hide()
			self.ProcessTargets.delete_tracks_btn.hide()
			self.ProcessEffectors.delete_tracks_btn.hide()
		else:

			if self.well_list.isAnySelected() and self.position_list.isAnySelected():

				self.locate_selected_position()
				self.view_stack_btn.setEnabled(True)
				# if os.path.exists(os.sep.join([self.pos,'labels_effectors', os.sep])):
				self.ProcessEffectors.check_seg_btn.setEnabled(True)
				# if os.path.exists(os.sep.join([self.pos,'labels_targets', os.sep])):
				self.ProcessTargets.check_seg_btn.setEnabled(True)
				
				# if os.path.exists(os.sep.join([self.pos,'output','tables','napari_target_trajectories.npy'])):
				# 	self.ProcessTargets.check_tracking_result_btn.setEnabled(True)
				# else:
				# 	self.ProcessTargets.check_tracking_result_btn.setEnabled(False)
				# if os.path.exists(os.sep.join([self.pos,'output','tables','napari_effector_trajectories.npy'])):
				# 	self.ProcessEffectors.check_tracking_result_btn.setEnabled(True)
				# else:
				# 	self.ProcessEffectors.check_tracking_result_btn.setEnabled(False)

				if os.path.exists(os.sep.join([self.pos,'output','tables','trajectories_effectors.csv'])):
					df = pd.read_csv(os.sep.join([self.pos,'output','tables','trajectories_effectors.csv']), nrows=1)
					id_col = extract_identity_col(df)
					self.ProcessEffectors.check_measurements_btn.setEnabled(True)
					if id_col=='TRACK_ID':
						self.ProcessEffectors.check_signals_btn.setEnabled(True)
						self.ProcessEffectors.delete_tracks_btn.show()
						self.ProcessEffectors.signal_analysis_action.setEnabled(True)
						self.ProcessEffectors.check_tracking_result_btn.setEnabled(True)
					else:
						self.ProcessEffectors.signal_analysis_action.setEnabled(False)
						self.ProcessEffectors.check_tracking_result_btn.setEnabled(False)			

					#self.ProcessEffectors.signal_analysis_action.setEnabled(True)
					self.ProcessEffectors.view_tab_btn.setEnabled(True)
					self.ProcessEffectors.classify_btn.setEnabled(True)
				else:
					self.ProcessEffectors.check_measurements_btn.setEnabled(False)
					self.ProcessEffectors.check_signals_btn.setEnabled(False)
					#self.ProcessEffectors.signal_analysis_action.setEnabled(False)
					self.ProcessEffectors.view_tab_btn.setEnabled(False)
					self.ProcessEffectors.classify_btn.setEnabled(False)
					self.ProcessEffectors.delete_tracks_btn.hide()
					self.ProcessEffectors.signal_analysis_action.setEnabled(False)

				if os.path.exists(os.sep.join([self.pos,'output','tables','trajectories_targets.csv'])):
					df = pd.read_csv(os.sep.join([self.pos,'output','tables','trajectories_targets.csv']), nrows=1)
					id_col = extract_identity_col(df)
					self.ProcessTargets.check_measurements_btn.setEnabled(True)
					if id_col=='TRACK_ID':
						self.ProcessTargets.check_signals_btn.setEnabled(True)
						self.ProcessTargets.signal_analysis_action.setEnabled(True)
						self.ProcessTargets.check_tracking_result_btn.setEnabled(True)
						self.ProcessTargets.delete_tracks_btn.show()
					else:
						self.ProcessTargets.signal_analysis_action.setEnabled(False)
						self.ProcessTargets.check_tracking_result_btn.setEnabled(False)

					#self.ProcessTargets.signal_analysis_action.setEnabled(True)
					self.ProcessTargets.view_tab_btn.setEnabled(True)
					self.ProcessTargets.classify_btn.setEnabled(True)
				else:
					self.ProcessTargets.check_measurements_btn.setEnabled(False)
					self.ProcessTargets.check_signals_btn.setEnabled(False)
					#self.ProcessTargets.signal_analysis_action.setEnabled(False)
					self.ProcessTargets.view_tab_btn.setEnabled(False)
					self.ProcessTargets.classify_btn.setEnabled(False)
					self.ProcessTargets.signal_analysis_action.setEnabled(False)				
					self.ProcessTargets.delete_tracks_btn.hide()
					self.ProcessTargets.signal_analysis_action.setEnabled(False)

				if os.path.exists(os.sep.join([self.pos,'output','tables','trajectories_pairs.csv'])):
					self.NeighPanel.view_tab_btn.setEnabled(True)
					self.NeighPanel.check_signals_btn.setEnabled(True)
				else:
					self.NeighPanel.view_tab_btn.setEnabled(False)
					self.NeighPanel.check_signals_btn.setEnabled(False)


