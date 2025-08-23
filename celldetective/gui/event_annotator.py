from PyQt5.QtWidgets import QComboBox, QLabel, QRadioButton, QFileDialog, QApplication, \
	QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QShortcut, QLineEdit, QSlider
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QIntValidator

from celldetective.gui.gui_utils import center_window, color_from_state
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider, QSearchableComboBox
from celldetective.utils import _get_img_num_per_channel
from celldetective.io import load_frames, get_experiment_metadata, get_experiment_labels, locate_labels
from celldetective.gui.gui_utils import FigureCanvas, color_from_status, color_from_class
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from matplotlib.animation import FuncAnimation
from matplotlib.cm import tab10
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from celldetective.gui import CelldetectiveWidget
from celldetective.measure import contour_of_instance_segmentation
from celldetective.utils import pretty_table
from celldetective.gui.base_annotator import BaseAnnotator

class EventAnnotator(BaseAnnotator):
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, *args, **kwargs):

		super().__init__(*args, **kwargs)
		self.setWindowTitle("Signal annotator")

		# default params
		self.class_name = 'class'
		self.time_name = 't0'
		self.status_name = 'status'

		# self.locate_stack()
		if not self.proceed:
			self.close()
		else:
			#self.load_annotator_config()
			#self.locate_tracks()
			self.prepare_stack()

			self.frame_lbl = QLabel('frame: ')
			self.looped_animation()
			self.init_event_buttons()
			self.populate_window()
			
			self.outliers_check.hide()
			if hasattr(self, "contrast_slider"):
				self.im.set_clim(self.contrast_slider.value()[0], self.contrast_slider.value()[1])

	def init_event_buttons(self):
	
		self.event_btn = QRadioButton('event')
		self.event_btn.setStyleSheet(self.button_style_sheet_2)
		self.event_btn.toggled.connect(self.enable_time_of_interest)
	
		self.no_event_btn = QRadioButton('no event')
		self.no_event_btn.setStyleSheet(self.button_style_sheet_2)
		self.no_event_btn.toggled.connect(self.enable_time_of_interest)
	
		self.else_btn = QRadioButton('else')
		self.else_btn.setStyleSheet(self.button_style_sheet_2)
		self.else_btn.toggled.connect(self.enable_time_of_interest)
	
		self.suppr_btn = QRadioButton('remove')
		self.suppr_btn.setToolTip('Mark for deletion. Upon saving, the cell\nwill be removed from the tables.')
		self.suppr_btn.setStyleSheet(self.button_style_sheet_2)
		self.suppr_btn.toggled.connect(self.enable_time_of_interest)
	
		self.time_of_interest_label = QLabel('time of interest: ')
		self.time_of_interest_le = QLineEdit()

	
	def populate_options_layout(self):
		
		# clear options hbox
		for i in reversed(range(self.options_hbox.count())):
			self.options_hbox.itemAt(i).widget().setParent(None)
	
		options_layout = QVBoxLayout()
		# add new widgets

		btn_hbox = QHBoxLayout()
		btn_hbox.addWidget(self.event_btn, 25, alignment=Qt.AlignCenter)
		btn_hbox.addWidget(self.no_event_btn, 25, alignment=Qt.AlignCenter)
		btn_hbox.addWidget(self.else_btn, 25, alignment=Qt.AlignCenter)
		btn_hbox.addWidget(self.suppr_btn, 25, alignment=Qt.AlignCenter)
	
		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(0, 5, 100, 10)
		time_option_hbox.addWidget(self.time_of_interest_label, 10)
		time_option_hbox.addWidget(self.time_of_interest_le, 15)
		time_option_hbox.addWidget(QLabel(''), 75)
	
		options_layout.addLayout(btn_hbox)
		options_layout.addLayout(time_option_hbox)
		
		self.options_hbox.addLayout(options_layout)
	
		self.annotation_btns_to_hide = [self.event_btn, self.no_event_btn,
										self.else_btn, self.time_of_interest_label,
										self.time_of_interest_le, self.suppr_btn]
		self.hide_annotation_buttons()
	
	def populate_window(self):

		"""
		Create the multibox design.

		"""
		
		super().populate_window()
		self.populate_options_layout()
	
		self.del_shortcut = QShortcut(Qt.Key_Delete, self)  # QKeySequence("s")
		self.del_shortcut.activated.connect(self.shortcut_suppr)
		self.del_shortcut.setEnabled(False)

		self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
		self.no_event_shortcut.activated.connect(self.shortcut_no_event)
		self.no_event_shortcut.setEnabled(False)

		# Right side
		animation_buttons_box = QHBoxLayout()
		animation_buttons_box.addWidget(self.frame_lbl, 20, alignment=Qt.AlignLeft)

		self.first_frame_btn = QPushButton()
		self.first_frame_btn.clicked.connect(self.set_first_frame)
		self.first_frame_btn.setShortcut(QKeySequence('f'))
		self.first_frame_btn.setIcon(icon(MDI6.page_first, color="black"))
		self.first_frame_btn.setStyleSheet(self.button_select_all)
		self.first_frame_btn.setFixedSize(QSize(60, 60))
		self.first_frame_btn.setIconSize(QSize(30, 30))

		self.last_frame_btn = QPushButton()
		self.last_frame_btn.clicked.connect(self.set_last_frame)
		self.last_frame_btn.setShortcut(QKeySequence('l'))
		self.last_frame_btn.setIcon(icon(MDI6.page_last, color="black"))
		self.last_frame_btn.setStyleSheet(self.button_select_all)
		self.last_frame_btn.setFixedSize(QSize(60, 60))
		self.last_frame_btn.setIconSize(QSize(30, 30))

		self.stop_btn = QPushButton()
		self.stop_btn.clicked.connect(self.stop)
		self.stop_btn.setIcon(icon(MDI6.stop, color="black"))
		self.stop_btn.setStyleSheet(self.button_select_all)
		self.stop_btn.setFixedSize(QSize(60, 60))
		self.stop_btn.setIconSize(QSize(30, 30))

		self.start_btn = QPushButton()
		self.start_btn.clicked.connect(self.start)
		self.start_btn.setIcon(icon(MDI6.play, color="black"))
		self.start_btn.setFixedSize(QSize(60, 60))
		self.start_btn.setStyleSheet(self.button_select_all)
		self.start_btn.setIconSize(QSize(30, 30))
		self.start_btn.hide()

		animation_buttons_box.addWidget(self.first_frame_btn, 5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.stop_btn, 5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.start_btn, 5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.last_frame_btn, 5, alignment=Qt.AlignRight)

		self.right_panel.addLayout(animation_buttons_box, 5)
		self.right_panel.addWidget(self.fcanvas, 90)

		if not self.rgb_mode:
			contrast_hbox = QHBoxLayout()
			contrast_hbox.setContentsMargins(150, 5, 150, 5)
			self.contrast_slider = QLabeledDoubleRangeSlider()
			self.contrast_slider.setSingleStep(0.001)
			self.contrast_slider.setTickInterval(0.001)
			self.contrast_slider.setOrientation(Qt.Horizontal)
			self.contrast_slider.setRange(
				*[np.nanpercentile(self.stack, 0.001), np.nanpercentile(self.stack, 99.999)])
			self.contrast_slider.setValue(
				[np.nanpercentile(self.stack, 1), np.nanpercentile(self.stack, 99.99)])
			self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
			contrast_hbox.addWidget(QLabel('contrast: '))
			contrast_hbox.addWidget(self.contrast_slider, 90)
			self.right_panel.addLayout(contrast_hbox, 5)
		
		if self.class_choice_cb.currentText()!="":
			self.compute_status_and_colors(0)
		
		self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
		self.no_event_shortcut.activated.connect(self.shortcut_no_event)
		self.no_event_shortcut.setEnabled(False)

		QApplication.processEvents()

	def write_new_event_class(self):

		if self.class_name_le.text() == '':
			self.target_class = 'class'
			self.target_time = 't0'
		else:
			self.target_class = 'class_' + self.class_name_le.text()
			self.target_time = 't_' + self.class_name_le.text()

		if self.target_class in list(self.df_tracks.columns):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(
				"This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				pass

		fill_option = np.where([c.isChecked() for c in self.class_option_rb])[0][0]
		self.df_tracks.loc[:, self.target_class] = fill_option
		if fill_option == 0:
			self.df_tracks.loc[:, self.target_time] = 0.1
		else:
			self.df_tracks.loc[:, self.target_time] = -1

		self.class_choice_cb.clear()
		cols = np.array(self.df_tracks.columns)
		self.class_cols = np.array([c.startswith('class') for c in list(self.df_tracks.columns)])
		self.class_cols = list(cols[self.class_cols])
		if 'class_id' in self.class_cols:
			self.class_cols.remove('class_id')
		if 'class_color' in self.class_cols:
			self.class_cols.remove('class_color')
		self.class_choice_cb.addItems(self.class_cols)
		idx = self.class_choice_cb.findText(self.target_class)
		self.class_choice_cb.setCurrentIndex(idx)

		self.newClassWidget.close()

	# def close_without_new_class(self):
	# 	self.newClassWidget.close()

	def compute_status_and_colors(self, i):

		self.class_name = self.class_choice_cb.currentText()
		self.expected_status = 'status'
		suffix = self.class_name.replace('class', '').replace('_', '', 1)
		if suffix != '':
			self.expected_status += '_' + suffix
			self.expected_time = 't_' + suffix
		else:
			self.expected_time = 't0'
		self.time_name = self.expected_time
		self.status_name = self.expected_status

		cols = list(self.df_tracks.columns)

		if self.time_name in cols and self.class_name in cols and not self.status_name in cols:
			# only create the status column if it does not exist to not erase static classification results
			self.make_status_column()
		elif self.time_name in cols and self.class_name in cols and self.df_tracks[self.status_name].isnull().all():
			self.make_status_column()
		elif self.time_name in cols and self.class_name in cols:
			# all good, do nothing
			pass
		else:
			if not self.status_name in self.df_tracks.columns:
				self.df_tracks[self.status_name] = 0
				self.df_tracks['status_color'] = color_from_status(0)
				self.df_tracks['class_color'] = color_from_class(1)

		if not self.class_name in self.df_tracks.columns:
			self.df_tracks[self.class_name] = 1
		if not self.time_name in self.df_tracks.columns:
			self.df_tracks[self.time_name] = -1

		self.df_tracks['status_color'] = [color_from_status(i) for i in self.df_tracks[self.status_name].to_numpy()]
		self.df_tracks['class_color'] = [color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()]

		self.extract_scatter_from_trajectories()
		if len(self.selection)>0:
			self.select_single_cell(self.selection[0][0], self.selection[0][1])

		self.fcanvas.canvas.draw()

	def cancel_selection(self):
		
		super().cancel_selection()
		try:
			for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
				self.colors[t][idx, 1] = self.previous_color[k][1]
		except Exception as e:
			pass

	def hide_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.hide()
		for b in [self.event_btn, self.no_event_btn, self.else_btn, self.suppr_btn]:
			b.setChecked(False)
		self.time_of_interest_label.setEnabled(False)
		self.time_of_interest_le.setText('')
		self.time_of_interest_le.setEnabled(False)

	def enable_time_of_interest(self):

		if self.event_btn.isChecked():
			self.time_of_interest_label.setEnabled(True)
			self.time_of_interest_le.setEnabled(True)
		else:
			self.time_of_interest_label.setEnabled(False)
			self.time_of_interest_le.setEnabled(False)

	def show_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.show()

		cclass = self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.class_name].to_numpy()[0]
		t0 = self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.time_name].to_numpy()[0]

		if cclass == 0:
			self.event_btn.setChecked(True)
			self.time_of_interest_le.setText(str(t0))
		elif cclass == 1:
			self.no_event_btn.setChecked(True)
		elif cclass == 2:
			self.else_btn.setChecked(True)
		elif cclass > 2:
			self.suppr_btn.setChecked(True)

		self.enable_time_of_interest()
		self.correct_btn.setText('submit')

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.apply_modification)

	def apply_modification(self):

		t0 = -1
		if self.event_btn.isChecked():
			cclass = 0
			try:
				t0 = float(self.time_of_interest_le.text().replace(',', '.'))
				self.line_dt.set_xdata([t0, t0])
				self.cell_fcanvas.canvas.draw_idle()
			except Exception as e:
				print(f"L 598 {e=}")
				t0 = -1
				cclass = 2
		elif self.no_event_btn.isChecked():
			cclass = 1
		elif self.else_btn.isChecked():
			cclass = 2
		elif self.suppr_btn.isChecked():
			cclass = 42

		self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.class_name] = cclass
		self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.time_name] = t0

		indices = self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.class_name].index
		timeline = self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, 'FRAME'].to_numpy()
		status = np.zeros_like(timeline)
		if t0 > 0:
			status[timeline >= t0] = 1.
		if cclass == 2:
			status[:] = 2
		if cclass > 2:
			status[:] = 42

		status_color = [color_from_status(s, recently_modified=True) for s in status]
		class_color = [color_from_class(cclass, recently_modified=True) for i in range(len(status))]


		# self.df_tracks['status_color'] = [color_from_status(i) for i in self.df_tracks[self.status_name].to_numpy()]
		# self.df_tracks['class_color'] = [color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()]

		self.df_tracks.loc[indices, self.status_name] = status
		self.df_tracks.loc[indices, 'status_color'] = status_color
		self.df_tracks.loc[indices, 'class_color'] = class_color

		# self.make_status_column()
		self.extract_scatter_from_trajectories()
		self.give_cell_information()

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		# self.cancel_btn.click()

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.del_shortcut.setEnabled(False)
		self.no_event_shortcut.setEnabled(False)

		self.selection.pop(0)


	def make_status_column(self):

		print(f'Generating status information for class `{self.class_name}` and time `{self.time_name}`...')
		for tid, group in self.df_tracks.groupby('TRACK_ID'):

			indices = group.index
			t0 = group[self.time_name].to_numpy()[0]
			cclass = group[self.class_name].to_numpy()[0]
			timeline = group['FRAME'].to_numpy()
			status = np.zeros_like(timeline)
			
			if t0 > 0:
				status[timeline >= t0] = 1.			
			# if cclass == 2:
			# 	status[:] = 1.
			if cclass > 2:
				status[:] = 42

			status_color = [color_from_status(s) for s in status]
			class_color = [color_from_class(cclass) for i in range(len(status))]

			self.df_tracks.loc[indices, self.status_name] = status
			self.df_tracks.loc[indices, 'status_color'] = status_color
			self.df_tracks.loc[indices, 'class_color'] = class_color

	def generate_signal_choices(self):

		self.signal_choice_cb = [QSearchableComboBox() for i in range(self.n_signals)]
		self.signal_choice_label = [QLabel(f'signal {i + 1}: ') for i in range(self.n_signals)]
		# self.log_btns = [QPushButton() for i in range(self.n_signals)]

		signals = list(self.df_tracks.columns)

		to_remove = ['TRACK_ID', 'FRAME', 'x_anim', 'y_anim', 't', 'state', 'generation', 'root', 'parent', 'class_id',
					 'class', 't0', 'POSITION_X', 'POSITION_Y', 'position', 'well', 'well_index', 'well_name',
					 'pos_name', 'index','class_color','status_color','dummy','group_color']

		meta = get_experiment_metadata(self.exp_dir)
		if meta is not None:
			keys = list(meta.keys())
			to_remove.extend(keys)

		labels = get_experiment_labels(self.exp_dir)
		if labels is not None:
			keys = list(labels.keys())
			to_remove.extend(labels)

		for c in to_remove:
			if c in signals:
				signals.remove(c)

		for i in range(len(self.signal_choice_cb)):
			self.signal_choice_cb[i].addItems(['--'] + signals)
			self.signal_choice_cb[i].setCurrentIndex(i + 1)
			self.signal_choice_cb[i].currentIndexChanged.connect(self.plot_signals)

	def plot_signals(self):

		range_values = []

		try:
			yvalues = []
			for i in range(len(self.signal_choice_cb)):

				signal_choice = self.signal_choice_cb[i].currentText()
				lbl = signal_choice
				n_cut = 35
				if len(lbl)>n_cut:
					lbl = lbl[:(n_cut-3)]+'...'
				self.lines[i].set_label(lbl)

				if signal_choice == "--":
					self.lines[i].set_xdata([])
					self.lines[i].set_ydata([])
				else:
					xdata = self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, 'FRAME'].to_numpy()
					ydata = self.df_tracks.loc[
						self.df_tracks['TRACK_ID'] == self.track_of_interest, signal_choice].to_numpy()
					
					range_values.extend(ydata)

					xdata = xdata[ydata == ydata]  # remove nan
					ydata = ydata[ydata == ydata]

					yvalues.extend(ydata)
					self.lines[i].set_xdata(xdata)
					self.lines[i].set_ydata(ydata)
					self.lines[i].set_color(tab10(i / 3.))

			self.configure_ylims()

			min_val, max_val = self.cell_ax.get_ylim()
			t0 = \
				self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.expected_time].to_numpy()[
					0]
			self.line_dt.set_xdata([t0, t0])
			self.line_dt.set_ydata([min_val, max_val])

			self.cell_ax.legend(fontsize=8)
			self.cell_fcanvas.canvas.draw()
		except Exception as e:
			print(e)
			pass
		
		if len(range_values)>0:
			range_values = np.array(range_values)
			if len(range_values[range_values==range_values])>0:
				if len(range_values[range_values>0])>0:
					self.value_magnitude = np.nanpercentile(range_values, 1)
				else:
					self.value_magnitude = 1
				self.non_log_ymin = 0.98*np.nanmin(range_values)
				self.non_log_ymax = np.nanmax(range_values)*1.02
				if self.cell_ax.get_yscale()=='linear':
					self.cell_ax.set_ylim(self.non_log_ymin, self.non_log_ymax)
				else:
					self.cell_ax.set_ylim(self.value_magnitude, self.non_log_ymax)					

	def extract_scatter_from_trajectories(self):

		self.positions = []
		self.colors = []
		self.tracks = []

		for t in np.arange(self.len_movie):
			self.positions.append(self.df_tracks.loc[self.df_tracks['FRAME'] == t, ['x_anim', 'y_anim']].to_numpy())
			self.colors.append(
				self.df_tracks.loc[self.df_tracks['FRAME'] == t, ['class_color', 'status_color']].to_numpy())
			self.tracks.append(self.df_tracks.loc[self.df_tracks['FRAME'] == t, 'TRACK_ID'].to_numpy())

	# def load_annotator_config(self):
	#
	# 	"""
	# 	Load settings from config or set default values.
	# 	"""
	#
	# 	if os.path.exists(self.instructions_path):
	# 		with open(self.instructions_path, 'r') as f:
	#
	# 			instructions = json.load(f)
	#
	# 			if 'rgb_mode' in instructions:
	# 				self.rgb_mode = instructions['rgb_mode']
	# 			else:
	# 				self.rgb_mode = False
	#
	# 			if 'percentile_mode' in instructions:
	# 				self.percentile_mode = instructions['percentile_mode']
	# 			else:
	# 				self.percentile_mode = True
	#
	# 			if 'channels' in instructions:
	# 				self.target_channels = instructions['channels']
	# 			else:
	# 				self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
	#
	# 			if 'fraction' in instructions:
	# 				self.fraction = float(instructions['fraction'])
	# 			else:
	# 				self.fraction = 0.25
	#
	# 			if 'interval' in instructions:
	# 				self.anim_interval = int(instructions['interval'])
	# 			else:
	# 				self.anim_interval = 1
	#
	# 			if 'log' in instructions:
	# 				self.log_option = instructions['log']
	# 			else:
	# 				self.log_option = False
	# 	else:
	# 		self.rgb_mode = False
	# 		self.log_option = False
	# 		self.percentile_mode = True
	# 		self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
	# 		self.fraction = 0.25
	# 		self.anim_interval = 1

	def prepare_stack(self):

		self.img_num_channels = _get_img_num_per_channel(self.channels, self.len_movie, self.nbr_channels)
		self.stack = []
		disable_tqdm = not len(self.target_channels)>1
		for ch in tqdm(self.target_channels, desc="channel",disable=disable_tqdm):
			target_ch_name = ch[0]
			if self.percentile_mode:
				normalize_kwargs = {"percentiles": (ch[1], ch[2]), "values": None}
			else:
				normalize_kwargs = {"values": (ch[1], ch[2]), "percentiles": None}

			if self.rgb_mode:
				normalize_kwargs.update({'amplification': 255., 'clip': True})

			chan = []
			indices = self.img_num_channels[self.channels[np.where(self.channel_names == target_ch_name)][0]]
			for t in tqdm(range(len(indices)), desc='frame'):
				if self.rgb_mode:
					f = load_frames(indices[t], self.stack_path, scale=self.fraction, normalize_input=True,
									normalize_kwargs=normalize_kwargs)
					f = f.astype(np.uint8)
				else:
					f = load_frames(indices[t], self.stack_path, scale=self.fraction, normalize_input=False)

				chan.append(f[:, :, 0])

			self.stack.append(chan)

		self.stack = np.array(self.stack)
		if self.rgb_mode:
			self.stack = np.moveaxis(self.stack, 0, -1)
		else:
			self.stack = self.stack[0]
			if self.log_option:
				self.stack[np.where(self.stack > 0.)] = np.log(self.stack[np.where(self.stack > 0.)])

	def closeEvent(self, event):
		
		try:
			self.stop()
			del self.stack
			gc.collect()
		except:
			pass

	def looped_animation(self):

		"""
		Load an image.

		"""

		self.framedata = 0

		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()

		self.im = self.ax.imshow(self.stack[0], cmap='gray', interpolation='none')
		self.status_scatter = self.ax.scatter(self.positions[0][:, 0], self.positions[0][:, 1], marker="x",
											  c=self.colors[0][:, 1], s=50, picker=True, pickradius=100)
		self.class_scatter = self.ax.scatter(self.positions[0][:, 0], self.positions[0][:, 1], marker='o',
											 facecolors='none', edgecolors=self.colors[0][:, 0], s=200)


		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.anim = FuncAnimation(
			self.fig,
			self.draw_frame,
			frames=self.len_movie,  # better would be to cast np.arange(len(movie)) in case frame column is incomplete
			interval=self.anim_interval,  # in ms
			blit=True,
		)

		self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)
		self.fcanvas.canvas.draw()

	def select_single_cell(self, index, timepoint):

		self.correct_btn.setEnabled(True)
		self.cancel_btn.setEnabled(True)
		self.del_shortcut.setEnabled(True)
		self.no_event_shortcut.setEnabled(True)

		self.track_of_interest = self.tracks[timepoint][index]
		print(f'You selected cell #{self.track_of_interest}...')
		self.give_cell_information()
		self.plot_signals()

		self.loc_t = []
		self.loc_idx = []
		for t in range(len(self.tracks)):
			indices = np.where(self.tracks[t] == self.track_of_interest)[0]
			if len(indices) > 0:
				self.loc_t.append(t)
				self.loc_idx.append(indices[0])

		self.previous_color = []
		for t, idx in zip(self.loc_t, self.loc_idx):
			self.previous_color.append(self.colors[t][idx].copy())
			self.colors[t][idx] = 'lime'


	def shortcut_no_event(self):
		self.correct_btn.click()
		self.no_event_btn.click()
		self.correct_btn.click()

	def configure_ylims(self):

		try:
			min_values = []
			max_values = []
			feats = []
			for i in range(len(self.signal_choice_cb)):
				signal = self.signal_choice_cb[i].currentText()
				if signal == '--':
					continue
				else:
					maxx = np.nanpercentile(self.df_tracks.loc[:, signal].to_numpy().flatten(), 99)
					minn = np.nanpercentile(self.df_tracks.loc[:, signal].to_numpy().flatten(), 1)
					min_values.append(minn)
					max_values.append(maxx)
					feats.append(signal)

			smallest_value = np.amin(min_values)
			feat_smallest_value = feats[np.argmin(min_values)]
			min_feat = self.df_tracks[feat_smallest_value].min()
			max_feat = self.df_tracks[feat_smallest_value].max()
			pad_small = (max_feat - min_feat) * 0.05
			if pad_small==0:
				pad_small = 0.05

			largest_value = np.amax(max_values)
			feat_largest_value = feats[np.argmax(max_values)]
			min_feat = self.df_tracks[feat_largest_value].min()
			max_feat = self.df_tracks[feat_largest_value].max()
			pad_large = (max_feat - min_feat) * 0.05
			if pad_large==0:
				pad_large = 0.05

			if len(min_values) > 0:
				self.cell_ax.set_ylim(smallest_value - pad_small, largest_value + pad_large)
		except Exception as e:
			print(f"L1170 {e=}")
			pass

	def draw_frame(self, framedata):

		"""
		Update plot elements at each timestep of the loop.
		"""

		self.framedata = framedata
		self.frame_lbl.setText(f'frame: {self.framedata}')
		self.im.set_array(self.stack[self.framedata])
		self.status_scatter.set_offsets(self.positions[self.framedata])
		self.status_scatter.set_color(self.colors[self.framedata][:, 1])

		self.class_scatter.set_offsets(self.positions[self.framedata])
		self.class_scatter.set_edgecolor(self.colors[self.framedata][:, 0])

		return (self.im, self.status_scatter, self.class_scatter,)

	def stop(self):
		# # On stop we disconnect all of our events.
		self.stop_btn.hide()
		self.start_btn.show()
		self.anim.pause()
		self.stop_btn.clicked.connect(self.start)

	def give_cell_information(self):

		cell_selected = f"cell: {self.track_of_interest}\n"
		cell_class = f"class: {self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.class_name].to_numpy()[0]}\n"
		cell_time = f"time of interest: {self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.time_name].to_numpy()[0]}\n"
		self.cell_info.setText(cell_selected + cell_class + cell_time)

	def save_trajectories(self):

		if self.normalized_signals:
			self.normalize_features_btn.click()
		if self.selection:
			self.cancel_selection()

		self.df_tracks = self.df_tracks.drop(self.df_tracks[self.df_tracks[self.class_name] > 2].index)
		self.df_tracks.to_csv(self.trajectories_path, index=False)
		print('Table successfully exported...')
		if self.class_choice_cb.currentText()!="":
			self.compute_status_and_colors(0)
		self.extract_scatter_from_trajectories()


	def set_last_frame(self):

		self.last_frame_btn.setEnabled(False)
		self.last_frame_btn.disconnect()

		self.last_key = len(self.stack) - 1
		while len(np.where(self.stack[self.last_key].flatten() == 0)[0]) > 0.99 * len(
				self.stack[self.last_key].flatten()):
			self.last_key -= 1
		self.anim._drawn_artists = self.draw_frame(self.last_key)
		self.anim._drawn_artists = sorted(self.anim._drawn_artists, key=lambda x: x.get_zorder())
		for a in self.anim._drawn_artists:
			a.set_visible(True)

		self.fig.canvas.draw()
		self.anim.event_source.stop()

		# self.cell_plot.draw()
		self.stop_btn.hide()
		self.start_btn.show()
		self.stop_btn.clicked.connect(self.start)
		self.start_btn.setShortcut(QKeySequence("l"))

	def set_first_frame(self):

		self.first_frame_btn.setEnabled(False)
		self.first_frame_btn.disconnect()

		self.first_key = 0
		self.anim._drawn_artists = self.draw_frame(0)
		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)


		self.anim._drawn_artists = sorted(self.anim._drawn_artists, key=lambda x: x.get_zorder())
		for a in self.anim._drawn_artists:
			a.set_visible(True)

		self.fig.canvas.draw()
		self.anim.event_source.stop()

		# self.cell_plot.draw()
		self.stop_btn.hide()
		self.start_btn.show()
		self.stop_btn.clicked.connect(self.start)
		self.start_btn.setShortcut(QKeySequence("f"))


class MeasureAnnotator(BaseAnnotator):

	def __init__(self, *args, **kwargs):
		
		super().__init__(read_config=False, *args, **kwargs)

		self.setWindowTitle("Static annotator")

		self.int_validator = QIntValidator()
		self.current_alpha=0.5
		self.value_magnitude = 1

		epsilon = 0.01
		self.observed_min_intensity = 0
		self.observed_max_intensity = 0 + epsilon
		
		self.current_frame = 0
		self.show_fliers = False
		self.status_name = 'group'

		if self.proceed:
			
			self.labels = locate_labels(self.pos, population=self.mode)
	
			self.current_channel = 0
			self.frame_lbl = QLabel('position: ')
			self.static_image()
	
			self.populate_window()
			self.changed_class()
	
			self.previous_index = None
			if hasattr(self, "contrast_slider"):
				self.im.set_clim(self.contrast_slider.value()[0], self.contrast_slider.value()[1])

		else:
			self.close()
	
	def locate_tracks(self):

		"""
		Locate the tracks.
		"""

		if not os.path.exists(self.trajectories_path):

			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("The trajectories cannot be detected.")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Yes:
				self.close()
		else:

			# Load and prep tracks
			self.df_tracks = pd.read_csv(self.trajectories_path)
			if 'TRACK_ID' in self.df_tracks.columns:
				self.df_tracks = self.df_tracks.sort_values(by=['TRACK_ID', 'FRAME'])
			else:
				self.df_tracks = self.df_tracks.sort_values(by=['ID', 'FRAME'])

			cols = np.array(self.df_tracks.columns)
			self.class_cols = np.array([c.startswith('group') or c.startswith('class') for c in list(self.df_tracks.columns)])
			self.class_cols = list(cols[self.class_cols])

			to_remove = ['class_id','group_color','class_color']
			for col in to_remove:
				try:
					self.class_cols.remove(col)
				except:
					pass
			
			if len(self.class_cols) > 0:
				self.status_name = self.class_cols[0]
			else:
				self.status_name = 'group'

			if self.status_name not in self.df_tracks.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_status_column()
			else:
				# all good, do nothing
				pass

			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)

			self.df_tracks = self.df_tracks.dropna(subset=['POSITION_X', 'POSITION_Y'])
			self.df_tracks['x_anim'] = self.df_tracks['POSITION_X']
			self.df_tracks['y_anim'] = self.df_tracks['POSITION_Y']
			self.df_tracks['x_anim'] = self.df_tracks['x_anim'].astype(int)
			self.df_tracks['y_anim'] = self.df_tracks['y_anim'].astype(int)

			self.extract_scatter_from_trajectories()
			if 'TRACK_ID' in self.df_tracks.columns:
				self.track_of_interest = self.df_tracks.dropna(subset='TRACK_ID')['TRACK_ID'].min()
			else:
				self.track_of_interest = self.df_tracks.dropna(subset='ID')['ID'].min()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.tracks)):
				indices = np.where(self.tracks[t] == self.track_of_interest)[0]
				if len(indices) > 0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])

			self.MinMaxScaler = MinMaxScaler()
			self.columns_to_rescale = list(self.df_tracks.columns)

			cols_to_remove = ['group', 'group_color', 'status', 'status_color', 'class_color', 'TRACK_ID', 'FRAME',
							  'x_anim', 'y_anim', 't','dummy','group_color',
							  'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X',
							  'POSITION_Y', 'position', 'well', 'well_index', 'well_name', 'pos_name', 'index',
							  'concentration', 'cell_type', 'antibody', 'pharmaceutical_agent', 'ID'] + self.class_cols

			meta = get_experiment_metadata(self.exp_dir)
			if meta is not None:
				keys = list(meta.keys())
				cols_to_remove.extend(keys)

			labels = get_experiment_labels(self.exp_dir)
			if labels is not None:
				keys = list(labels.keys())
				cols_to_remove.extend(labels)

			for tr in cols_to_remove:
				try:
					self.columns_to_rescale.remove(tr)
				except:
					pass

			x = self.df_tracks[self.columns_to_rescale].values
			self.MinMaxScaler.fit(x)

	
	def populate_options_layout(self):
		# clear options hbox
		for i in reversed(range(self.options_hbox.count())):
			self.options_hbox.itemAt(i).widget().setParent(None)
		
		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(100, 0, 100, 0)
		time_option_hbox.setSpacing(0)

		self.time_of_interest_label = QLabel('phenotype: ')
		time_option_hbox.addWidget(self.time_of_interest_label, 30)
		
		self.time_of_interest_le = QLineEdit()
		self.time_of_interest_le.setValidator(self.int_validator)
		time_option_hbox.addWidget(self.time_of_interest_le)
		
		self.suppr_btn = QPushButton('')
		self.suppr_btn.setStyleSheet(self.button_select_all)
		self.suppr_btn.setIcon(icon(MDI6.delete, color="black"))
		self.suppr_btn.setToolTip("Delete cell")
		self.suppr_btn.setIconSize(QSize(20, 20))
		self.suppr_btn.clicked.connect(self.del_cell)
		time_option_hbox.addWidget(self.suppr_btn)
		
		self.options_hbox.addLayout(time_option_hbox)
		
	def update_widgets(self):
		
		self.class_label.setText('characteristic \n group: ')
		self.update_class_cb()
		self.add_class_btn.setToolTip("Add a new characteristic group")
		self.del_class_btn.setToolTip("Delete a characteristic group")
		
		self.export_btn.disconnect()
		self.export_btn.clicked.connect(self.export_measurements)
		
	def update_class_cb(self):
		
		self.class_choice_cb.disconnect()
		self.class_choice_cb.clear()
		cols = np.array(self.df_tracks.columns)
		self.class_cols = np.array([c.startswith('group') or c.startswith('status') for c in list(self.df_tracks.columns)])
		self.class_cols = list(cols[self.class_cols])

		to_remove = ['group_id', 'group_color', 'class_id', 'class_color', 'status_color']
		for col in to_remove:
			try:
				self.class_cols.remove(col)
			except Exception:
				pass

		self.class_choice_cb.addItems(self.class_cols)
		self.class_choice_cb.currentIndexChanged.connect(self.changed_class)

	def populate_window(self):
		
		super().populate_window()
		# Left panel updates
		self.populate_options_layout()
		self.update_widgets()
		
		self.annotation_btns_to_hide = [self.time_of_interest_label,
										self.time_of_interest_le,
										self.suppr_btn]
		self.hide_annotation_buttons()
		
		# Right panel
		animation_buttons_box = QHBoxLayout()
		animation_buttons_box.addWidget(self.frame_lbl, 20, alignment=Qt.AlignLeft)

		self.first_frame_btn = QPushButton()
		self.first_frame_btn.clicked.connect(self.set_previous_frame)
		self.first_frame_btn.setShortcut(QKeySequence('f'))
		self.first_frame_btn.setIcon(icon(MDI6.page_first, color="black"))
		self.first_frame_btn.setStyleSheet(self.button_select_all)
		self.first_frame_btn.setFixedSize(QSize(60, 60))
		self.first_frame_btn.setIconSize(QSize(30, 30))

		self.last_frame_btn = QPushButton()
		self.last_frame_btn.clicked.connect(self.set_next_frame)
		self.last_frame_btn.setShortcut(QKeySequence('l'))
		self.last_frame_btn.setIcon(icon(MDI6.page_last, color="black"))
		self.last_frame_btn.setStyleSheet(self.button_select_all)
		self.last_frame_btn.setFixedSize(QSize(60, 60))
		self.last_frame_btn.setIconSize(QSize(30, 30))

		self.frame_slider = QSlider(Qt.Horizontal)
		self.frame_slider.setFixedSize(200, 30)
		self.frame_slider.setRange(0, self.len_movie - 1)
		self.frame_slider.setValue(0)
		self.frame_slider.valueChanged.connect(self.update_frame)

		self.start_btn = QPushButton()
		self.start_btn.clicked.connect(self.start)
		self.start_btn.setIcon(icon(MDI6.play, color="black"))
		self.start_btn.setFixedSize(QSize(60, 60))
		self.start_btn.setStyleSheet(self.button_select_all)
		self.start_btn.setIconSize(QSize(30, 30))
		self.start_btn.hide()

		animation_buttons_box.addWidget(self.first_frame_btn, 5, alignment=Qt.AlignRight)
		animation_buttons_box.addWidget(self.frame_slider, 5, alignment=Qt.AlignCenter)
		animation_buttons_box.addWidget(self.last_frame_btn, 5, alignment=Qt.AlignLeft)

		self.right_panel.addLayout(animation_buttons_box, 5)

		self.right_panel.addWidget(self.fcanvas, 90)

		contrast_hbox = QHBoxLayout()
		contrast_hbox.setContentsMargins(150, 5, 150, 5)
		self.contrast_slider = QLabeledDoubleRangeSlider()

		self.contrast_slider.setSingleStep(0.001)
		self.contrast_slider.setTickInterval(0.001)
		self.contrast_slider.setOrientation(Qt.Horizontal)
		self.contrast_slider.setRange(
			*[np.nanpercentile(self.img, 0.001), np.nanpercentile(self.img, 99.999)])
		self.contrast_slider.setValue(
			[np.nanpercentile(self.img, 1), np.nanpercentile(self.img, 99.99)])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
		contrast_hbox.addWidget(QLabel('contrast: '))
		contrast_hbox.addWidget(self.contrast_slider, 90)
		self.right_panel.addLayout(contrast_hbox, 5)
		self.alpha_slider = QLabeledDoubleSlider()
		self.alpha_slider.setSingleStep(0.001)
		self.alpha_slider.setOrientation(Qt.Horizontal)
		self.alpha_slider.setRange(0, 1)
		self.alpha_slider.setValue(self.current_alpha)
		self.alpha_slider.setDecimals(3)
		self.alpha_slider.valueChanged.connect(self.set_transparency)

		slider_alpha_hbox = QHBoxLayout()
		slider_alpha_hbox.setContentsMargins(150, 5, 150, 5)
		slider_alpha_hbox.addWidget(QLabel('transparency: '), 10)
		slider_alpha_hbox.addWidget(self.alpha_slider, 90)
		self.right_panel.addLayout(slider_alpha_hbox)

		channel_hbox = QHBoxLayout()
		self.choose_channel = QComboBox()
		self.choose_channel.addItems(self.channel_names)
		self.choose_channel.currentIndexChanged.connect(self.changed_channel)
		channel_hbox.addWidget(self.choose_channel)
		self.right_panel.addLayout(channel_hbox, 5)
	
		self.draw_frame(0)
		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		
		self.fcanvas.canvas.draw()
		self.plot_signals()
	
	def static_image(self):

		"""
		Load an image.

		"""

		self.framedata = 0
		self.current_label=self.labels[self.current_frame]
		self.fig, self.ax = plt.subplots(tight_layout=True)
		self.fcanvas = FigureCanvas(self.fig, interactive=True)
		self.ax.clear()
		# print(self.current_stack.shape)
		self.im = self.ax.imshow(self.img, cmap='gray')
		self.status_scatter = self.ax.scatter(self.positions[0][:, 0], self.positions[0][:, 1], marker="o",
											  facecolors='none', edgecolors=self.colors[0][:, 0], s=200, picker=True)
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.current_label == 0, self.current_label),
											  cmap='viridis', interpolation='none',alpha=self.current_alpha,vmin=0,vmax=np.nanmax(self.labels.flatten()))
		self.ax.set_xticks([])
		self.ax.set_yticks([])
		self.ax.set_aspect('equal')

		self.fig.set_facecolor('none')  # or 'None'
		self.fig.canvas.setStyleSheet("background-color: black;")

		self.fig.canvas.mpl_connect('pick_event', self.on_scatter_pick)
		self.fcanvas.canvas.draw()

		
	def plot_signals(self):

		#try:
		current_frame = self.current_frame  # Assuming you have a variable for the current frame

		yvalues = []
		all_yvalues = []
		current_yvalues = []
		labels = []
		range_values = []

		for i in range(len(self.signal_choice_cb)):

			signal_choice = self.signal_choice_cb[i].currentText()

			if signal_choice != "--":
				if 'TRACK_ID' in self.df_tracks.columns:
					ydata = self.df_tracks.loc[
						(self.df_tracks['TRACK_ID'] == self.track_of_interest) &
						(self.df_tracks['FRAME'] == current_frame), signal_choice].to_numpy()
				else:
					ydata = self.df_tracks.loc[
						(self.df_tracks['ID'] == self.track_of_interest), signal_choice].to_numpy()
				all_ydata = self.df_tracks.loc[:, signal_choice].to_numpy()
				ydataNaN = ydata
				ydata = ydata[ydata == ydata]  # remove nan

				current_ydata = self.df_tracks.loc[
					(self.df_tracks['FRAME'] == current_frame), signal_choice].to_numpy()
				current_ydata = current_ydata[current_ydata == current_ydata]
				all_ydata = all_ydata[all_ydata == all_ydata]
				yvalues.extend(ydataNaN)
				current_yvalues.append(current_ydata)
				all_yvalues.append(all_ydata)
				range_values.extend(all_ydata)
				labels.append(signal_choice)

		self.cell_ax.clear()
		if self.log_scale:
			self.cell_ax.set_yscale('log')
		else:
			self.cell_ax.set_yscale('linear')

		if len(yvalues) > 0:
			try:
				self.cell_ax.boxplot(all_yvalues, showfliers=self.show_fliers)
			except Exception as e:
				print(f"{e=}")
			
			x_pos = np.arange(len(all_yvalues)) + 1
			for index, feature in enumerate(current_yvalues):
				x_values_strip = (index + 1) + np.random.normal(0, 0.04, size=len(
					feature))
				self.cell_ax.plot(x_values_strip, feature, marker='o', linestyle='None', color=tab10.colors[0],
								  alpha=0.1)
			self.cell_ax.plot(x_pos, yvalues, marker='H', linestyle='None', color=tab10.colors[3], alpha=1)
			range_values = np.array(range_values)
			range_values = range_values[range_values==range_values]
			
			if len(range_values[range_values > 0]) > 0:
				self.value_magnitude = np.nanmin(range_values[range_values > 0]) - 0.03 * (
							np.nanmax(range_values[range_values > 0]) - np.nanmin(range_values[range_values > 0]))
			else:
				self.value_magnitude = 1
			
			self.non_log_ymin = np.nanmin(range_values) - 0.03 * (np.nanmax(range_values) - np.nanmin(range_values))
			self.non_log_ymax = np.nanmax(range_values) + 0.03 * (np.nanmax(range_values) - np.nanmin(range_values))
			if self.cell_ax.get_yscale() == 'linear':
				self.cell_ax.set_ylim(self.non_log_ymin, self.non_log_ymax)
			else:
				self.cell_ax.set_ylim(self.value_magnitude, self.non_log_ymax)
		else:
			self.cell_ax.text(0.5, 0.5, "No data available", horizontalalignment='center',
							  verticalalignment='center', transform=self.cell_ax.transAxes)

		self.cell_fcanvas.canvas.draw()

	def plot_red_points(self, ax):
		yvalues = []
		current_frame = self.current_frame
		for i in range(len(self.signal_choice_cb)):
			signal_choice = self.signal_choice_cb[i].currentText()
			if signal_choice != "--":
				#print(f'plot signal {signal_choice} for cell {self.track_of_interest} at frame {current_frame}')
				if 'TRACK_ID' in self.df_tracks.columns:
					ydata = self.df_tracks.loc[
						(self.df_tracks['TRACK_ID'] == self.track_of_interest) &
						(self.df_tracks['FRAME'] == current_frame), signal_choice].to_numpy()
				else:
					ydata = self.df_tracks.loc[
						(self.df_tracks['ID'] == self.track_of_interest) &
						(self.df_tracks['FRAME'] == current_frame), signal_choice].to_numpy()
				ydata = ydata[ydata == ydata]  # remove nan
				yvalues.extend(ydata)
		x_pos = np.arange(len(yvalues)) + 1
		ax.plot(x_pos, yvalues, marker='H', linestyle='None', color=tab10.colors[3],
				alpha=1)  # Plot red points representing cells
		self.cell_fcanvas.canvas.draw()

	def select_single_cell(self, index, timepoint):
		
		self.correct_btn.setEnabled(True)
		self.cancel_btn.setEnabled(True)
		self.del_shortcut.setEnabled(True)
	
		self.track_of_interest = self.tracks[timepoint][index]
		print(f'You selected cell #{self.track_of_interest}...')
		self.give_cell_information()
	
		if len(self.cell_ax.lines) > 0:
			self.cell_ax.lines[-1].remove()  # Remove the last line (red points) from the plot
			self.plot_red_points(self.cell_ax)
		else:
			self.plot_signals()
		
		self.loc_t = []
		self.loc_idx = []
		for t in range(len(self.tracks)):
			indices = np.where(self.tracks[t] == self.track_of_interest)[0]
			if len(indices) > 0:
				self.loc_t.append(t)
				self.loc_idx.append(indices[0])
			
		self.previous_color = []
		for t, idx in zip(self.loc_t, self.loc_idx):
			self.previous_color.append(self.colors[t][idx].copy())
			self.colors[t][idx] = 'lime'
	
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def cancel_selection(self):
		super().cancel_selection()
		self.event = None
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def export_measurements(self):

		auto_dataset_name = self.pos.split(os.sep)[-4] + '_' + self.pos.split(os.sep)[-2] + f'_{str(self.current_frame).zfill(3)}' + f'_{self.status_name}.npy'

		if self.normalized_signals:
			self.normalize_features_btn.click()

		subdf = self.df_tracks.loc[self.df_tracks['FRAME']==self.current_frame,:]
		subdf['class'] = subdf[self.status_name]
		dico = subdf.to_dict('records')

		pathsave = QFileDialog.getSaveFileName(self, "Select file name", self.exp_dir + auto_dataset_name, ".npy")[0]
		if pathsave != '':
			if not pathsave.endswith(".npy"):
				pathsave += ".npy"
			try:
				np.save(pathsave, dico)
				print(f'File successfully written in {pathsave}.')
			except Exception as e:
				print(f"Error {e}...")

	def set_next_frame(self):

		self.current_frame = self.current_frame + 1
		if self.current_frame > self.len_movie - 1:
			self.current_frame == self.len_movie - 1
		self.frame_slider.setValue(self.current_frame)
		self.update_frame()
		self.start_btn.setShortcut(QKeySequence("f"))

	def set_previous_frame(self):

		self.current_frame = self.current_frame - 1
		if self.current_frame < 0:
			self.current_frame == 0
		self.frame_slider.setValue(self.current_frame)
		self.update_frame()

		self.start_btn.setShortcut(QKeySequence("l"))

	def write_new_event_class(self):

		if self.class_name_le.text() == '':
			self.target_class = 'group'
		else:
			self.target_class = 'group_' + self.class_name_le.text()

		if self.target_class in list(self.df_tracks.columns):
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(
				"This characteristic group name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.No:
				return None
			else:
				pass

		self.df_tracks.loc[:, self.target_class] = 0

		self.update_class_cb()

		idx = self.class_choice_cb.findText(self.target_class)
		self.status_name = self.target_class
		self.class_choice_cb.setCurrentIndex(idx)
		self.newClassWidget.close()

	def hide_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.hide()
		self.time_of_interest_label.setEnabled(False)
		self.time_of_interest_le.setText('')
		self.time_of_interest_le.setEnabled(False)

	def set_transparency(self):
		self.current_alpha = self.alpha_slider.value()
		self.im_mask.set_alpha(self.current_alpha)
		self.fcanvas.canvas.draw()

	def show_annotation_buttons(self):

		for a in self.annotation_btns_to_hide:
			a.show()

		self.time_of_interest_label.setEnabled(True)
		self.time_of_interest_le.setEnabled(True)
		self.correct_btn.setText('submit')

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.apply_modification)

	def give_cell_information(self):

		try:
			cell_selected = f"cell: {self.track_of_interest}\n"
			if 'TRACK_ID' in self.df_tracks.columns:
				cell_status = f"phenotype: {self.df_tracks.loc[(self.df_tracks['FRAME']==self.current_frame)&(self.df_tracks['TRACK_ID'] == self.track_of_interest), self.status_name].to_numpy()[0]}\n"
			else:
				cell_status = f"phenotype: {self.df_tracks.loc[self.df_tracks['ID'] == self.track_of_interest, self.status_name].to_numpy()[0]}\n"
			self.cell_info.setText(cell_selected + cell_status)
		except Exception as e:
			print(e)

	def create_new_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = CelldetectiveWidget()
		self.newClassWidget.setWindowTitle('Create new characteristic group')

		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('group name: '), 25)
		self.class_name_le = QLineEdit('group')
		name_hbox.addWidget(self.class_name_le, 75)
		layout.addLayout(name_hbox)

		btn_hbox = QHBoxLayout()
		submit_btn = QPushButton('submit')
		cancel_btn = QPushButton('cancel')
		btn_hbox.addWidget(cancel_btn, 50)
		btn_hbox.addWidget(submit_btn, 50)
		layout.addLayout(btn_hbox)

		submit_btn.clicked.connect(self.write_new_event_class)
		cancel_btn.clicked.connect(self.close_without_new_class)

		self.newClassWidget.show()
		center_window(self.newClassWidget)

	def apply_modification(self):
		if self.time_of_interest_le.text() != "":
			status = int(self.time_of_interest_le.text())
		else:
			status = 0
		if "TRACK_ID" in self.df_tracks.columns:
			self.df_tracks.loc[(self.df_tracks['TRACK_ID'] == self.track_of_interest) & (
					self.df_tracks['FRAME'] == self.current_frame), self.status_name] = status

			indices = self.df_tracks.index[(self.df_tracks['TRACK_ID'] == self.track_of_interest) & (
					self.df_tracks['FRAME'] == self.current_frame)]
		else:
			self.df_tracks.loc[(self.df_tracks['ID'] == self.track_of_interest) & (
					self.df_tracks['FRAME'] == self.current_frame), self.status_name] = status

			indices = self.df_tracks.index[(self.df_tracks['ID'] == self.track_of_interest) & (
					self.df_tracks['FRAME'] == self.current_frame)]

		self.df_tracks.loc[indices, self.status_name] = status
		all_states = self.df_tracks.loc[:, self.status_name].tolist()
		all_states = np.array(all_states)
		self.state_color_map = color_from_state(all_states, recently_modified=False)

		self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)
		self.extract_scatter_from_trajectories()
		self.give_cell_information()

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)
		self.del_shortcut.setEnabled(False)

		if len(self.selection) > 0:
			self.selection.pop(0)
		
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def assign_color_state(self, state):

		if np.isnan(state):
			state = "nan"
		return self.state_color_map[state]

	def draw_frame(self, framedata):

		"""
		Update plot elements at each timestep of the loop.
		"""
		self.framedata = framedata
		self.frame_lbl.setText(f'position: {self.framedata}')
		self.im.set_array(self.img)
		self.status_scatter.set_offsets(self.positions[self.framedata])
		# try:
		self.status_scatter.set_edgecolors(self.colors[self.framedata][:, 0])
		# except Exception as e:
		# 	pass

		self.current_label = self.labels[self.current_frame]
		self.current_label = contour_of_instance_segmentation(self.current_label, 5)

		self.im_mask.remove()
		self.im_mask = self.ax.imshow(np.ma.masked_where(self.current_label == 0, self.current_label),
											   cmap='viridis', interpolation='none',alpha=self.current_alpha,vmin=0,vmax=np.nanmax(self.labels.flatten()))

		return (self.im, self.status_scatter,self.im_mask,)

	def compute_status_and_colors(self):

		self.cancel_selection()
		
		if self.class_choice_cb.currentText() == '':
			pass
		else:
			self.status_name = self.class_choice_cb.currentText()

		if self.status_name not in self.df_tracks.columns:
			print('Creating a new status for visualization...')
			self.make_status_column()
		else:
			print(f'Generating per-state colors for the status "{self.status_name}"...')
			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			print(f'Color mapping for "{self.status_name}":')
			pretty_table(self.state_color_map)
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)


	def make_status_column(self):
		if self.status_name == "state_firstdetection":
			pass
		else:
			self.df_tracks.loc[:, self.status_name] = 0
			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)


	def extract_scatter_from_trajectories(self):

		self.positions = []
		self.colors = []
		self.tracks = []

		for t in np.arange(self.len_movie):
			self.positions.append(self.df_tracks.loc[self.df_tracks['FRAME'] == t, ['POSITION_X', 'POSITION_Y']].to_numpy())
			self.colors.append(
				self.df_tracks.loc[self.df_tracks['FRAME'] == t, ['group_color']].to_numpy())
			if 'TRACK_ID' in self.df_tracks.columns:
				self.tracks.append(self.df_tracks.loc[self.df_tracks['FRAME'] == t, 'TRACK_ID'].to_numpy())
			else:
				self.tracks.append(self.df_tracks.loc[self.df_tracks['FRAME'] == t, 'ID'].to_numpy())

	def changed_class(self):
		self.status_name = self.class_choice_cb.currentText()
		if self.status_name!="":
			self.compute_status_and_colors()
			self.modify()
			self.draw_frame(self.current_frame)
			self.fcanvas.canvas.draw()

	def update_frame(self):
		"""
		Update the displayed frame.
		"""
		self.current_frame = self.frame_slider.value()
		self.reload_frame()
		if 'TRACK_ID' in list(self.df_tracks.columns):
			pass
		elif 'ID' in list(self.df_tracks.columns):
			print('ID in cols... change class of interest... ')
			self.track_of_interest = self.df_tracks[self.df_tracks['FRAME'] == self.current_frame]['ID'].min()
			self.modify()

		self.draw_frame(self.current_frame)
		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.give_cell_information()

		self.fcanvas.canvas.draw()
		self.plot_signals()

	def changed_channel(self):

		self.reload_frame()
		self.contrast_slider.setRange(
			*[np.nanpercentile(self.img, 0.001),
			  np.nanpercentile(self.img, 99.999)])
		self.contrast_slider.setValue(
			[np.nanpercentile(self.img, 0.1), np.nanpercentile(self.img, 99.99)])
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def save_trajectories(self):
		print(f"Saving trajectories !!")
		if self.normalized_signals:
			self.normalize_features_btn.click()
		if self.selection:
			self.cancel_selection()
		self.df_tracks = self.df_tracks.drop(self.df_tracks[self.df_tracks[self.status_name] == 99].index)
		#color_column = str(self.status_name) + "_color"
		try:
			self.df_tracks.drop(columns='', inplace=True)
		except:
			pass
		try:
			self.df_tracks.drop(columns='group_color', inplace=True)
		except:
			pass
		try:
			self.df_tracks.drop(columns='x_anim', inplace=True)
		except:
			pass
		try:
			self.df_tracks.drop(columns='y_anim', inplace=True)
		except:
			pass

		self.df_tracks.to_csv(self.trajectories_path, index=False)
		print('Table successfully exported...')
		
		self.locate_tracks()
		self.changed_class()

	def modify(self):
		all_states = self.df_tracks.loc[:, self.status_name].tolist()
		all_states = np.array(all_states)
		self.state_color_map = color_from_state(all_states, recently_modified=False)

		self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)

		self.extract_scatter_from_trajectories()
		self.give_cell_information()

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)

	def reload_frame(self):

		"""
		Load the frame from the current channel and time choice. Show imshow, update histogram.
		"""

		# self.clear_post_threshold_options()
		self.previous_channel = self.current_channel
		self.current_channel = self.choose_channel.currentIndex()

		t = int(self.frame_slider.value())
		idx = t * self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)

		if self.previous_channel != self.current_channel:
			# reinitialize intensity bounds
			epsilon = 0.01
			self.observed_min_intensity = 0
			self.observed_max_intensity = 0 + epsilon

		if self.img is not None:
			max_img = np.nanmax(self.img)
			min_img = np.nanmin(self.img)
			if max_img > self.observed_max_intensity:
				self.observed_max_intensity = max_img
			if min_img < self.observed_min_intensity:
				self.observed_min_intensity = min_img
			self.refresh_imshow()
		# self.redo_histogram()
		else:
			print('Frame could not be loaded...')

	def refresh_imshow(self):

		"""

		Update the imshow based on the current frame selection.

		"""

		if self.previous_channel != self.current_channel:

			self.vmin = np.nanpercentile(self.img.flatten(), 0.1)
			self.vmax = np.nanpercentile(self.img.flatten(), 99.99)

			self.contrast_slider.disconnect()
			self.contrast_slider.setRange(np.nanmin(self.img), np.nanmax(self.img))
			self.contrast_slider.setValue([self.vmin, self.vmax])
			self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
		else:
			#self.contrast_slider.disconnect()
			self.contrast_slider.setRange(self.observed_min_intensity, self.observed_max_intensity)
			#self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

		self.im.set_data(self.img)

	def del_cell(self):
		self.time_of_interest_le.setEnabled(False)
		self.time_of_interest_le.setText("99")
		self.apply_modification()
