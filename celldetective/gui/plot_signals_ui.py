from PyQt5.QtWidgets import QMessageBox, QComboBox, \
	QCheckBox, QLineEdit, QVBoxLayout, QWidget, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QDoubleValidator

from celldetective.gui.gui_utils import center_window
from celldetective.gui.generic_signal_plot import GenericSignalPlotWidget
from superqt import QLabeledSlider, QColormapComboBox, QSearchableComboBox
from celldetective.utils import get_software_location, _extract_labels_from_config, extract_cols_from_table_list
from celldetective.io import load_experiment_tables
from celldetective.signals import mean_signal
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from glob import glob
from natsort import natsorted
import math
from celldetective.gui import Styles
from matplotlib import colormaps
import matplotlib.cm


class ConfigSignalPlot(QWidget, Styles):
	
	"""
	UI to set survival instructions.

	"""

	def __init__(self, parent_window=None):
		
		super().__init__()
		self.parent_window = parent_window
		self.setWindowTitle("Configure signal plot")
		self.setWindowIcon(QIcon(os.sep.join(['celldetective','icons','mexican-hat.png'])))
		self.exp_dir = self.parent_window.exp_dir
		self.soft_path = get_software_location()		
		self.exp_config = self.exp_dir +"config.ini"
		self.wells = np.array(self.parent_window.parent_window.wells,dtype=str)
		self.well_labels = _extract_labels_from_config(self.exp_config,len(self.wells))
		self.FrameToMin = self.parent_window.parent_window.FrameToMin
		self.float_validator = QDoubleValidator()
		self.target_class = [0,1]
		self.show_ci = True
		self.show_cell_lines = False
		self.ax2=None
		self.auto_close = False

		self.well_option = self.parent_window.parent_window.well_list.getSelectedIndices()
		self.position_option = self.parent_window.parent_window.position_list.getSelectedIndices()
		self.interpret_pos_location()

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		center_window(self)
		self.populate_widget()

		if self.auto_close:
			self.close()
		self.setAttribute(Qt.WA_DeleteOnClose)

	def interpret_pos_location(self):
		
		"""
		Read the well/position selection from the control panel to decide which data to load
		Set position_indices to None if all positions must be taken

		"""
		
		self.well_indices = self.parent_window.parent_window.well_list.getSelectedIndices()
		self.position_indices = self.parent_window.parent_window.position_list.getSelectedIndices()
		if not self.parent_window.parent_window.position_list.isAnySelected():
			self.position_indices = None


	def populate_widget(self):

		"""
		Create the multibox design.

		"""
		
		# Create button widget and layout
		main_layout = QVBoxLayout()
		self.setLayout(main_layout)
		main_layout.setContentsMargins(30,30,30,30)
		panel_title = QLabel('Options')
		panel_title.setStyleSheet("""
			font-weight: bold;
			padding: 0px;
			""")
		main_layout.addWidget(panel_title, alignment=Qt.AlignCenter)

		pops = []
		for population in ['effectors','targets','pairs']:
			tables = glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_{population}.csv']))
			if len(tables)>0:
				pops.append(population)

		labels = [QLabel('population: '), QLabel('class: '), QLabel('time of\ninterest: '), QLabel('cmap: ')]
		self.cb_options = [pops,[], [], []]
		self.cbs = [QComboBox() for i in range(len(labels))]
		self.cbs[-1] = QColormapComboBox()
		
		self.cbs[0].currentIndexChanged.connect(self.set_classes_and_times)

		choice_layout = QVBoxLayout()
		choice_layout.setContentsMargins(20,20,20,20)
		for i in range(len(labels)):
			hbox = QHBoxLayout()
			hbox.addWidget(labels[i], 33)
			hbox.addWidget(self.cbs[i],66)
			if i < len(labels)-1:
				self.cbs[i].addItems(self.cb_options[i])
			choice_layout.addLayout(hbox)
		
		all_cms = list(colormaps)
		for cm in all_cms:
			if hasattr(matplotlib.cm, str(cm).lower()):
				try:
					self.cbs[-1].addColormap(cm.lower())
				except:
					pass

		self.cbs[0].setCurrentIndex(1)
		self.cbs[0].setCurrentIndex(0)

		self.abs_time_checkbox = QCheckBox('absolute time')
		self.frame_slider = QLabeledSlider()
		self.frame_slider.setSingleStep(1)
		self.frame_slider.setOrientation(1)
		self.frame_slider.setRange(0,self.parent_window.parent_window.len_movie)
		self.frame_slider.setValue(0)
		self.frame_slider.setEnabled(False)
		slider_hbox = QHBoxLayout()
		slider_hbox.addWidget(self.abs_time_checkbox, 33)
		slider_hbox.addWidget(self.frame_slider, 66)
		choice_layout.addLayout(slider_hbox)
		main_layout.addLayout(choice_layout)

		self.abs_time_checkbox.stateChanged.connect(self.switch_ref_time_mode)

		select_layout = QHBoxLayout()
		select_layout.setContentsMargins(20,3,20,3)
		select_layout.addWidget(QLabel('select cells\nwith query: '), 33)
		self.query_le = QLineEdit()
		select_layout.addWidget(self.query_le, 66)
		main_layout.addLayout(select_layout)

		time_calib_layout = QHBoxLayout()
		time_calib_layout.setContentsMargins(20,3,20,3)
		time_calib_layout.addWidget(QLabel('time calibration\n(frame to min)'), 33)
		self.time_calibration_le = QLineEdit(str(self.FrameToMin).replace('.',','))
		self.time_calibration_le.setValidator(self.float_validator)
		time_calib_layout.addWidget(self.time_calibration_le, 66)
		#time_calib_layout.addWidget(QLabel(' min'))
		main_layout.addLayout(time_calib_layout)

		pool_layout = QHBoxLayout()
		pool_layout.setContentsMargins(20,3,20,3)
		self.pool_option_cb = QComboBox()
		self.pool_option_cb.addItems(['mean','median'])
		pool_layout.addWidget(QLabel('pool\nprojection:'), 33)
		pool_layout.addWidget(self.pool_option_cb, 66)
		main_layout.addLayout(pool_layout)

		n_cells_layout = QHBoxLayout()
		n_cells_layout.setContentsMargins(20,3,20,3)
		self.n_cells_slider = QLabeledSlider()
		self.n_cells_slider.setSingleStep(1)
		self.n_cells_slider.setOrientation(1)
		self.n_cells_slider.setRange(1,100)
		self.n_cells_slider.setValue(2)
		n_cells_layout.addWidget(QLabel('min # cells\nfor pool:'), 33)
		n_cells_layout.addWidget(self.n_cells_slider, 66)
		main_layout.addLayout(n_cells_layout)


		self.submit_btn = QPushButton('Submit')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.process_signal)
		main_layout.addWidget(self.submit_btn)

		#self.populate_left_panel()
		#grid.addLayout(self.left_side, 0, 0, 1, 1)

		# self.setCentralWidget(self.scroll_area)
		# self.show()

	def set_classes_and_times(self):

		# Look for all classes and times
		population = self.cbs[0].currentText()
		tables = natsorted(glob(self.exp_dir+os.sep.join(['W*','*','output','tables',f'trajectories_{population}.csv'])))
		
		self.all_columns = extract_cols_from_table_list(tables)

		class_idx = np.array([s.startswith('class_') for s in self.all_columns])
		time_idx = np.array([s.startswith('t_') for s in self.all_columns])
		
		try:
			class_columns = list(self.all_columns[class_idx])
			time_columns = list(self.all_columns[time_idx])
		except:
			print('columns not found')
			self.auto_close = True
			return None
		
		if 'class' in self.all_columns:
			class_columns.append("class")
		if 't0' in self.all_columns:
			time_columns.append('t0')
		
		self.class_columns = np.unique(class_columns)
		self.time_columns = np.unique(time_columns)
		thresh = 18
		self.class_truncated = [w[:thresh - 3]+'...' if len(w)>thresh else w for w in self.class_columns]
		self.time_truncated = [w[:thresh - 3]+'...' if len(w)>thresh else w for w in self.time_columns]

		self.cbs[2].clear()
		self.cbs[2].addItems(self.time_truncated)
		for i in range(len(self.time_columns)):
			self.cbs[2].setItemData(i, self.time_columns[i], Qt.ToolTipRole)

		self.cbs[1].clear()
		self.cbs[1].addItems(self.class_truncated)
		for i in range(len(self.class_columns)):
			self.cbs[1].setItemData(i, self.class_columns[i], Qt.ToolTipRole)

	def ask_for_feature(self):

		cols = np.array(list(self.df.columns))
		is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		feats = cols[is_number(self.df.dtypes)]

		self.feature_choice_widget = QWidget()
		self.feature_choice_widget.setWindowTitle("Select numeric feature")
		layout = QVBoxLayout()
		self.feature_choice_widget.setLayout(layout)
		self.feature_cb = QComboBox()
		self.feature_cb.addItems(feats)
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('feature: '), 33)
		hbox.addWidget(self.feature_cb, 66)
		layout.addLayout(hbox)

		self.set_feature_btn = QPushButton('set')
		self.set_feature_btn.clicked.connect(self.compute_signals)
		layout.addWidget(self.set_feature_btn)
		self.feature_choice_widget.show()
		center_window(self.feature_choice_widget)

	def ask_for_features(self):

		cols = np.array(list(self.df.columns))
		is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		feats = cols[is_number(self.df.dtypes)]

		self.feature_choice_widget = QWidget()
		self.feature_choice_widget.setWindowTitle("Select numeric feature")
		layout = QVBoxLayout()
		self.feature_choice_widget.setLayout(layout)
		self.feature_cb = QSearchableComboBox()
		self.feature_cb.addItems(feats)
		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('feature: '), 33)
		hbox.addWidget(self.feature_cb, 66)
		#hbox.addWidget((QLabel('Plot two features')))
		layout.addLayout(hbox)

		self.set_feature_btn = QPushButton('set')
		self.set_feature_btn.setStyleSheet(self.button_style_sheet)
		self.set_feature_btn.clicked.connect(self.compute_signals)
		layout.addWidget(self.set_feature_btn)
		self.feature_choice_widget.show()
		center_window(self.feature_choice_widget)

	# def enable_second_feature(self):
	# 	if self.checkBox_feature.isChecked():
	# 		self.feature_two_cb.setEnabled(True)
	# 	else:
	# 		self.feature_two_cb.setEnabled(False)
	
	def compute_signals(self):

		if self.df is not None:

			try:
				query_text = self.query_le.text()
				if query_text != '':
					self.df = self.df.query(query_text)
			except Exception as e:
				print(e, ' The query is misunderstood and will not be applied...')

			self.feature_selected = self.feature_cb.currentText()
			self.feature_choice_widget.close()
			self.compute_signal_functions()
			if self.open_widget:
				self.interpret_pos_location()
				self.plot_window = GenericSignalPlotWidget(parent_window=self, df=self.df, df_pos_info = self.df_pos_info, df_well_info = self.df_well_info, feature_selected=self.feature_selected, title='plot signals')
				self.plot_window.show()

	def process_signal(self):

		self.FrameToMin = float(self.time_calibration_le.text().replace(',','.'))
		print(f'Time calibration set to 1 frame =  {self.FrameToMin} min...')

		# read instructions from combobox options
		self.load_available_tables()
		class_col = self.class_columns[self.cbs[1].currentIndex()]
		print(f"{class_col=}")	

		if self.df is not None:

			if class_col not in list(self.df.columns):
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText("The class of interest could not be found in the data. Abort.")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None
			else:
				self.ask_for_features()
		else:
			return None

		#self.plotvbox.addWidget(self.line_choice_widget, alignment=Qt.AlignCenter)

	def load_available_tables(self):

		"""
		Load the tables of the selected wells/positions from the control Panel for the population of interest

		"""

		self.well_option = self.parent_window.parent_window.well_list.getSelectedIndices()		
		self.position_option = self.parent_window.parent_window.position_list.getSelectedIndices()

		self.df, self.df_pos_info = load_experiment_tables(self.exp_dir, well_option=self.well_option, position_option=self.position_option, population=self.cbs[0].currentText(), return_pos_info=True)

		if self.df is None:
			
			print('No table could be found...')
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No table could be found to compute survival...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.close()
				return None
			else:
				self.close()
				return None
		else:
			self.df_well_info = self.df_pos_info.loc[:,['well_path', 'well_index', 'well_name', 'well_number', 'well_alias']].drop_duplicates()

	def compute_signal_functions(self):

		# Check to move at the beginning
		self.open_widget = True
		if len(self.time_columns)==0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No synchronizing time is available...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				pass
			self.open_widget = False
			return None

		# Per position signal
		self.df = self.df.dropna(subset=['FRAME'])
		if len(self.df)==0:
			print('Warning... The dataset is empty. Please check your filters. Abort...')
			return None

		max_time = int(self.df.FRAME.max()) + 1
		class_col = self.class_columns[self.cbs[1].currentIndex()]
		time_col = self.time_columns[self.cbs[2].currentIndex()]
		if self.abs_time_checkbox.isChecked():
			time_col = self.frame_slider.value()

		for block,movie_group in self.df.groupby(['well','position']):

			well_signal_mean, well_std_mean, timeline_all, matrix_all = mean_signal(movie_group, self.feature_selected, class_col, time_col=time_col, class_value=None, return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())
			well_signal_event, well_std_event, timeline_event, matrix_event = mean_signal(movie_group, self.feature_selected, class_col, time_col=time_col, class_value=[0], return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())
			well_signal_no_event, well_std_no_event, timeline_no_event, matrix_no_event = mean_signal(movie_group, self.feature_selected, class_col, time_col=time_col, class_value=[1], return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())
			self.mean_plots_timeline = timeline_all

			self.df_pos_info.loc[self.df_pos_info['pos_path'] == block[1], 'signal'] = [
				{'mean_all': well_signal_mean, 'std_all': well_std_mean, 'matrix_all': matrix_all,
				 'mean_event': well_signal_event, 'std_event': well_std_event,
				 'matrix_event': matrix_event, 'mean_no_event': well_signal_no_event,
				 'std_no_event': well_std_no_event, 'matrix_no_event': matrix_no_event,
				 'timeline': self.mean_plots_timeline}]

		# Per well
		for well,well_group in self.df.groupby('well'):

			well_signal_mean, well_std_mean, timeline_all, matrix_all = mean_signal(well_group, self.feature_selected, class_col, time_col=time_col, class_value=None, return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())
			well_signal_event, well_std_event, timeline_event, matrix_event = mean_signal(well_group, self.feature_selected, class_col, time_col=time_col, class_value=[0], return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())			
			well_signal_no_event, well_std_no_event, timeline_no_event, matrix_no_event = mean_signal(well_group, self.feature_selected, class_col, time_col=time_col, class_value=[1], return_matrix=True, forced_max_duration=max_time, projection=self.pool_option_cb.currentText(), min_nbr_values=self.n_cells_slider.value())
			
			self.df_well_info.loc[self.df_well_info['well_path']==well,'signal'] = [{'mean_all': well_signal_mean, 'std_all': well_std_mean,'matrix_all': matrix_all,'mean_event': well_signal_event, 'std_event': well_std_event,
																					'matrix_event': matrix_event,'mean_no_event': well_signal_no_event, 'std_no_event': well_std_no_event, 'matrix_no_event': matrix_no_event, 'timeline':  self.mean_plots_timeline}]

		self.df_pos_info.loc[:,'select'] = True
		self.df_well_info.loc[:,'select'] = True


	def generate_synchronized_matrix(self, well_group, feature_selected, cclass, max_time):
		
		if isinstance(cclass,int):
			cclass = [cclass]

		class_col = self.class_columns[self.cbs[1].currentIndex()]
		time_col = self.time_columns[self.cbs[2].currentIndex()]

		n_cells = len(well_group.groupby(['position','TRACK_ID']))
		depth = int(2*max_time + 3)
		matrix = np.zeros((n_cells, depth))
		matrix[:,:] = np.nan
		mapping = np.arange(-max_time-1, max_time+2)
		cid=0
		for block,movie_group in well_group.groupby('position'):
			for tid,track_group in movie_group.loc[movie_group[class_col].isin(cclass)].groupby('TRACK_ID'):
				try:
					timeline = track_group['FRAME'].to_numpy().astype(int)
					feature = track_group[feature_selected].to_numpy()
					if self.checkBox_feature.isChecked():
						second_feature=track_group[self.second_feature_selected].to_numpy()
					if self.cbs[2].currentText().startswith('t') and not self.abs_time_checkbox.isChecked():
						t0 = math.floor(track_group[time_col].to_numpy()[0])
						timeline -= t0
					elif self.cbs[2].currentText()=='first detection' and not self.abs_time_checkbox.isChecked():

						if 'area' in list(track_group.columns):
							print('area in list')
							feat = track_group['area'].values
						else:
							feat = feature

						first_detection = timeline[feat==feat][0]
						timeline -= first_detection

					elif self.abs_time_checkbox.isChecked():
						timeline -= int(self.frame_slider.value())

					loc_t = [np.where(mapping==t)[0][0] for t in timeline]
					matrix[cid,loc_t] = feature
					if second_feature:
						matrix[cid,loc_t+1]=second_feature

					cid+=1
				except:
					pass
		return matrix

	def switch_ref_time_mode(self):
		if self.abs_time_checkbox.isChecked():
			self.frame_slider.setEnabled(True)
			self.cbs[-2].setEnabled(False)
		else:
			self.frame_slider.setEnabled(False)
			self.cbs[-2].setEnabled(True)