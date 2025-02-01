from PyQt5.QtWidgets import QMainWindow, QComboBox, QLabel, QRadioButton, QLineEdit, QFileDialog, QApplication, \
	QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QShortcut, QLineEdit, QSlider, QCheckBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QIntValidator

from celldetective.gui.gui_utils import center_window, color_from_state
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider, QSearchableComboBox
from celldetective.utils import extract_experiment_channels, get_software_location, _get_img_num_per_channel
from celldetective.io import auto_load_number_of_frames, load_frames, \
	load_napari_data, get_experiment_metadata, get_experiment_labels
from celldetective.gui.gui_utils import FigureCanvas, color_from_status, color_from_class, ExportPlotBtn
import json
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import gc
from matplotlib.animation import FuncAnimation
from matplotlib.cm import tab10
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from celldetective.gui import Styles
from celldetective.measure import contour_of_instance_segmentation

class SignalAnnotator(QMainWindow, Styles):
	"""
	UI to set tracking parameters for bTrack.

	"""

	def __init__(self, parent_window=None):

		super().__init__()
		
		center_window(self)
		self.proceed = True
		self.setAttribute(Qt.WA_DeleteOnClose)

		self.parent_window = parent_window
		self.setWindowTitle("Signal annotator")
		self.mode = self.parent_window.mode
		self.pos = self.parent_window.parent_window.pos
		self.exp_dir = self.parent_window.exp_dir
		self.PxToUm = self.parent_window.parent_window.PxToUm
		self.n_signals = 3
		self.soft_path = get_software_location()
		self.recently_modified = False
		self.selection = []

		if self.mode == "targets":
			self.instructions_path = self.exp_dir + os.sep.join(['configs', 'signal_annotator_config_targets.json'])
			self.trajectories_path = self.pos + os.sep.join(['output','tables','trajectories_targets.csv'])
		elif self.mode == "effectors":
			self.instructions_path = self.exp_dir + os.sep.join(['configs', 'signal_annotator_config_effectors.json'])
			self.trajectories_path = self.pos + os.sep.join(['output','tables','trajectories_effectors.csv'])

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.screen_width
		#self.setMinimumHeight(int(0.8*self.screen_height))
		self.value_magnitude = 1

		# default params
		self.class_name = 'class'
		self.time_name = 't0'
		self.status_name = 'status'

		self.locate_stack()
		if not self.proceed:
			self.close()
		else:
			self.load_annotator_config()
			self.locate_tracks()
			self.prepare_stack()

			self.generate_signal_choices()
			self.frame_lbl = QLabel('frame: ')
			self.looped_animation()
			self.create_cell_signal_canvas()

			self.populate_widget()

	def resizeEvent(self, event):
		
		super().resizeEvent(event)

		try:
			self.cell_fig.tight_layout()
		except:
			pass

	def populate_widget(self):

		"""
		Create the multibox design.

		"""

		self.button_widget = QWidget()
		main_layout = QHBoxLayout()
		self.button_widget.setLayout(main_layout)

		main_layout.setContentsMargins(30, 30, 30, 30)
		self.left_panel = QVBoxLayout()
		self.left_panel.setContentsMargins(30, 5, 30, 5)
		self.left_panel.setSpacing(3)

		self.right_panel = QVBoxLayout()

		class_hbox = QHBoxLayout()
		class_hbox.setContentsMargins(0,0,0,0)
		class_hbox.addWidget(QLabel('event: '), 25)
		self.class_choice_cb = QComboBox()

		cols = np.array(self.df_tracks.columns)
		self.class_cols = np.array([c.startswith('class') for c in list(self.df_tracks.columns)])
		self.class_cols = list(cols[self.class_cols])
		try:
			self.class_cols.remove('class_id')
		except Exception:
			pass
		try:
			self.class_cols.remove('class_color')
		except Exception:
			pass

		self.class_choice_cb.addItems(self.class_cols)
		self.class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors)

		class_hbox.addWidget(self.class_choice_cb, 70)

		self.add_class_btn = QPushButton('')
		self.add_class_btn.setStyleSheet(self.button_select_all)
		self.add_class_btn.setIcon(icon(MDI6.plus, color="black"))
		self.add_class_btn.setToolTip("Add a new event class")
		self.add_class_btn.setIconSize(QSize(20, 20))
		self.add_class_btn.clicked.connect(self.create_new_event_class)
		class_hbox.addWidget(self.add_class_btn, 5)

		self.del_class_btn = QPushButton('')
		self.del_class_btn.setStyleSheet(self.button_select_all)
		self.del_class_btn.setIcon(icon(MDI6.delete, color="black"))
		self.del_class_btn.setToolTip("Delete an event class")
		self.del_class_btn.setIconSize(QSize(20, 20))
		self.del_class_btn.clicked.connect(self.del_event_class)
		class_hbox.addWidget(self.del_class_btn, 5)

		self.left_panel.addLayout(class_hbox,5)

		self.cell_info = QLabel('')
		self.left_panel.addWidget(self.cell_info,10)

		# Annotation buttons
		options_hbox = QHBoxLayout()
		options_hbox.setContentsMargins(0, 0, 0, 0)
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

		options_hbox.addWidget(self.event_btn, 25, alignment=Qt.AlignCenter)
		options_hbox.addWidget(self.no_event_btn, 25, alignment=Qt.AlignCenter)
		options_hbox.addWidget(self.else_btn, 25, alignment=Qt.AlignCenter)
		options_hbox.addWidget(self.suppr_btn, 25, alignment=Qt.AlignCenter)
		self.left_panel.addLayout(options_hbox,5)

		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(0, 5, 100, 10)
		self.time_of_interest_label = QLabel('time of interest: ')
		time_option_hbox.addWidget(self.time_of_interest_label, 10)
		self.time_of_interest_le = QLineEdit()
		time_option_hbox.addWidget(self.time_of_interest_le, 15)
		time_option_hbox.addWidget(QLabel(''), 75)
		self.left_panel.addLayout(time_option_hbox,5)

		main_action_hbox = QHBoxLayout()
		main_action_hbox.setContentsMargins(0,0,0,0)
		self.correct_btn = QPushButton('correct')
		self.correct_btn.setIcon(icon(MDI6.redo_variant, color="white"))
		self.correct_btn.setIconSize(QSize(20, 20))
		self.correct_btn.setStyleSheet(self.button_style_sheet)
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		self.correct_btn.setEnabled(False)
		main_action_hbox.addWidget(self.correct_btn)

		self.cancel_btn = QPushButton('cancel')
		self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
		self.cancel_btn.setShortcut(QKeySequence("Esc"))
		self.cancel_btn.setEnabled(False)
		self.cancel_btn.clicked.connect(self.cancel_selection)
		main_action_hbox.addWidget(self.cancel_btn)
		self.left_panel.addLayout(main_action_hbox,5)

		self.annotation_btns_to_hide = [self.event_btn, self.no_event_btn,
										self.else_btn, self.time_of_interest_label,
										self.time_of_interest_le, self.suppr_btn]
		self.hide_annotation_buttons()
		#### End of annotation buttons

		self.del_shortcut = QShortcut(Qt.Key_Delete, self)  # QKeySequence("s")
		self.del_shortcut.activated.connect(self.shortcut_suppr)
		self.del_shortcut.setEnabled(False)

		self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
		self.no_event_shortcut.activated.connect(self.shortcut_no_event)
		self.no_event_shortcut.setEnabled(False)

		# Cell signals

		self.left_panel.addWidget(self.cell_fcanvas, 45)

		plot_buttons_hbox = QHBoxLayout()
		plot_buttons_hbox.setContentsMargins(0, 0, 0, 0)
		self.normalize_features_btn = QPushButton('')
		self.normalize_features_btn.setStyleSheet(self.button_select_all)
		self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical, color="black"))
		self.normalize_features_btn.setIconSize(QSize(25, 25))
		self.normalize_features_btn.setFixedSize(QSize(30, 30))
		# self.normalize_features_btn.setShortcut(QKeySequence('n'))
		self.normalize_features_btn.clicked.connect(self.normalize_features)

		plot_buttons_hbox.addWidget(QLabel(''), 90)
		plot_buttons_hbox.addWidget(self.normalize_features_btn, 5)
		self.normalized_signals = False

		self.log_btn = QPushButton()
		self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
		self.log_btn.setStyleSheet(self.button_select_all)
		self.log_btn.clicked.connect(self.switch_to_log)
		plot_buttons_hbox.addWidget(self.log_btn, 5)

		self.export_plot_btn = ExportPlotBtn(self.cell_fig, export_dir = self.exp_dir)
		plot_buttons_hbox.addWidget(self.export_plot_btn, 5)		

		self.left_panel.addLayout(plot_buttons_hbox,5)

		signal_choice_vbox = QVBoxLayout()
		signal_choice_vbox.setContentsMargins(30, 0, 30, 0)
		for i in range(len(self.signal_choice_cb)):
			hlayout = QHBoxLayout()
			hlayout.addWidget(self.signal_choice_label[i], 20)
			hlayout.addWidget(self.signal_choice_cb[i], 75)
			# hlayout.addWidget(self.log_btns[i], 5)
			signal_choice_vbox.addLayout(hlayout)

		# self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
		# self.log_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		self.left_panel.addLayout(signal_choice_vbox,15)

		btn_hbox = QHBoxLayout()
		btn_hbox.setContentsMargins(0,10,0,0)
		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.button_style_sheet)
		self.save_btn.clicked.connect(self.save_trajectories)
		btn_hbox.addWidget(self.save_btn, 90)

		self.export_btn = QPushButton('')
		self.export_btn.setStyleSheet(self.button_select_all)
		self.export_btn.clicked.connect(self.export_signals)
		self.export_btn.setIcon(icon(MDI6.export, color="black"))
		self.export_btn.setIconSize(QSize(25, 25))
		btn_hbox.addWidget(self.export_btn, 10)
		self.left_panel.addLayout(btn_hbox,5)

		# Animation
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
			self.contrast_slider.setOrientation(1)
			self.contrast_slider.setRange(
				*[np.nanpercentile(self.stack, 0.001), np.nanpercentile(self.stack, 99.999)])
			self.contrast_slider.setValue(
				[np.nanpercentile(self.stack, 1), np.nanpercentile(self.stack, 99.99)])
			self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
			contrast_hbox.addWidget(QLabel('contrast: '))
			contrast_hbox.addWidget(self.contrast_slider, 90)
			self.right_panel.addLayout(contrast_hbox, 5)

		# speed_hbox = QHBoxLayout()
		# speed_hbox.setContentsMargins(150,5,150,5)
		# self.interval_slider = QLabeledSlider()
		# self.interval_slider.setSingleStep(1)
		# self.interval_slider.setTickInterval(1)
		# self.interval_slider.setOrientation(1)
		# self.interval_slider.setRange(1, 10000)
		# self.interval_slider.setValue(self.speed)
		# self.interval_slider.valueChanged.connect(self.interval_slider_action)
		# speed_hbox.addWidget(QLabel('interval (ms): '))
		# speed_hbox.addWidget(self.interval_slider,90)
		# self.right_panel.addLayout(speed_hbox, 10)

		# self.populate_left_panel()
		# grid.addLayout(self.left_side, 0, 0, 1, 1)

		main_layout.addLayout(self.left_panel, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.compute_status_and_colors(0)

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def del_event_class(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Warning)
		msgBox.setText(
			f"You are about to delete event class {self.class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?")
		msgBox.setWindowTitle("Warning")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:
			class_to_delete = self.class_choice_cb.currentText()
			time_to_delete = class_to_delete.replace('class', 't')
			status_to_delete = class_to_delete.replace('class', 'status')
			cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
			for c in cols_to_delete:
				try:
					self.df_tracks = self.df_tracks.drop([c], axis=1)
				except Exception as e:
					print(e)
			item_idx = self.class_choice_cb.findText(class_to_delete)
			self.class_choice_cb.removeItem(item_idx)

	def create_new_event_class(self):

		# display qwidget to name the event
		self.newClassWidget = QWidget()
		self.newClassWidget.setWindowTitle('Create new event class')

		layout = QVBoxLayout()
		self.newClassWidget.setLayout(layout)
		name_hbox = QHBoxLayout()
		name_hbox.addWidget(QLabel('event name: '), 25)
		self.class_name_le = QLineEdit('event')
		name_hbox.addWidget(self.class_name_le, 75)
		layout.addLayout(name_hbox)

		class_labels = ['event', 'no event', 'else']
		layout.addWidget(QLabel('prefill: '))
		radio_box = QHBoxLayout()
		self.class_option_rb = [QRadioButton() for i in range(3)]
		for i, c in enumerate(self.class_option_rb):
			if i == 0:
				c.setChecked(True)
			c.setText(class_labels[i])
			radio_box.addWidget(c, 33, alignment=Qt.AlignCenter)
		layout.addLayout(radio_box)

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

	# Prefill with class value
	# write in table

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

	def close_without_new_class(self):
		self.newClassWidget.close()

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


	def contrast_slider_action(self):

		"""
		Recontrast the imshow as the contrast slider is moved.
		"""

		self.vmin = self.contrast_slider.value()[0]
		self.vmax = self.contrast_slider.value()[1]
		self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fcanvas.canvas.draw_idle()

	def cancel_selection(self):

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)

		try:
			self.selection.pop(0)
		except Exception as e:
			print(f"L 536 {e=}")

		try:
			for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
				self.colors[t][idx, 0] = self.previous_color[k][0]
				self.colors[t][idx, 1] = self.previous_color[k][1]
		except Exception as e:
			print(f'L 543 {e=}')

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

	# self.fcanvas.canvas.draw()

	def locate_stack(self):

		"""
		Locate the target movie.

		"""

		movies = glob(self.pos + os.sep.join(["movie", f"{self.parent_window.parent_window.movie_prefix}*.tif"]))

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movie is detected in the experiment folder.\nPlease check the stack prefix...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.proceed = False
				self.close()
			else:
				self.close()
		else:
			self.stack_path = movies[0]
			self.len_movie = self.parent_window.parent_window.len_movie
			len_movie_auto = auto_load_number_of_frames(self.stack_path)
			if len_movie_auto is not None:
				self.len_movie = len_movie_auto
			exp_config = self.exp_dir + "config.ini"
			self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
			self.channel_names = np.array(self.channel_names)
			self.channels = np.array(self.channels)
			self.nbr_channels = len(self.channels)

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
			self.df_tracks = self.df_tracks.sort_values(by=['TRACK_ID', 'FRAME'])

			cols = np.array(self.df_tracks.columns)
			self.class_cols = np.array([c.startswith('class') for c in list(self.df_tracks.columns)])
			self.class_cols = list(cols[self.class_cols])
			try:
				self.class_cols.remove('class_id')
			except:
				pass
			try:
				self.class_cols.remove('class_color')
			except:
				pass
			if len(self.class_cols) > 0:
				self.class_name = self.class_cols[0]
				self.expected_status = 'status'
				suffix = self.class_name.replace('class', '').replace('_', '')
				if suffix != '':
					self.expected_status += '_' + suffix
					self.expected_time = 't_' + suffix
				else:
					self.expected_time = 't0'
				self.time_name = self.expected_time
				self.status_name = self.expected_status
			else:
				self.class_name = 'class'
				self.time_name = 't0'
				self.status_name = 'status'

			if self.time_name in self.df_tracks.columns and self.class_name in self.df_tracks.columns and not self.status_name in self.df_tracks.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_status_column()
			elif self.time_name in self.df_tracks.columns and self.class_name in self.df_tracks.columns:
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

			self.df_tracks = self.df_tracks.dropna(subset=['POSITION_X', 'POSITION_Y'])
			self.df_tracks['x_anim'] = self.df_tracks['POSITION_X'] * self.fraction
			self.df_tracks['y_anim'] = self.df_tracks['POSITION_Y'] * self.fraction
			self.df_tracks['x_anim'] = self.df_tracks['x_anim'].astype(int)
			self.df_tracks['y_anim'] = self.df_tracks['y_anim'].astype(int)

			self.extract_scatter_from_trajectories()
			self.track_of_interest = self.df_tracks['TRACK_ID'].min()

			self.loc_t = []
			self.loc_idx = []
			for t in range(len(self.tracks)):
				indices = np.where(self.tracks[t] == self.track_of_interest)[0]
				if len(indices) > 0:
					self.loc_t.append(t)
					self.loc_idx.append(indices[0])

			self.MinMaxScaler = MinMaxScaler()
			self.columns_to_rescale = list(self.df_tracks.columns)
			#self.columns_to_rescale = self.df_tracks.select_dtypes(exclude=['object']).columns

			# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# is_number_test = is_number(self.df_tracks.dtypes)
			# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# print(self.columns_to_rescale)

			cols_to_remove = ['status', 'status_color', 'class_color', 'TRACK_ID', 'FRAME', 'x_anim', 'y_anim', 't',
							  'state', 'generation', 'root', 'parent', 'class_id', 'class', 't0', 'POSITION_X',
							  'POSITION_Y', 'position', 'well', 'well_index', 'well_name', 'pos_name', 'index',] + self.class_cols
			
			meta = get_experiment_metadata(self.exp_dir)
			if meta is not None:
				keys = list(meta.keys())
				cols_to_remove.extend(keys)

			labels = get_experiment_labels(self.exp_dir)
			if labels is not None:
				keys = list(labels.keys())
				cols_to_remove.extend(labels)

			cols = np.array(list(self.df_tracks.columns))
			time_cols = np.array([c.startswith('t_') for c in cols])
			time_cols = list(cols[time_cols])
			cols_to_remove += time_cols
			#cols_to_remove.extend(self.df_tracks.select_dtypes(include=['object']).columns)

			print(f"{cols_to_remove=}")

			for tr in cols_to_remove:
				try:
					self.columns_to_rescale.remove(tr)
				except:
					pass
			# print(f'column {tr} could not be found...')

			x = self.df_tracks[self.columns_to_rescale].values
			self.MinMaxScaler.fit(x)

	# self.loc_t, self.loc_idx = np.where(self.tracks==self.track_of_interest)

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
			if cclass == 2:
				status[:] = 2
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
					 'pos_name', 'index','class_color','status_color']

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
				self.lines[i].set_label(signal_choice)

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

			self.cell_ax.legend()
			self.cell_fcanvas.canvas.draw()
		except Exception as e:
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

	def load_annotator_config(self):

		"""
		Load settings from config or set default values.
		"""

		if os.path.exists(self.instructions_path):
			with open(self.instructions_path, 'r') as f:

				instructions = json.load(f)

				if 'rgb_mode' in instructions:
					self.rgb_mode = instructions['rgb_mode']
				else:
					self.rgb_mode = False

				if 'percentile_mode' in instructions:
					self.percentile_mode = instructions['percentile_mode']
				else:
					self.percentile_mode = True

				if 'channels' in instructions:
					self.target_channels = instructions['channels']
				else:
					self.target_channels = [[self.channel_names[0], 0.01, 99.99]]

				if 'fraction' in instructions:
					self.fraction = float(instructions['fraction'])
				else:
					self.fraction = 0.25

				if 'interval' in instructions:
					self.anim_interval = int(instructions['interval'])
				else:
					self.anim_interval = 1

				if 'log' in instructions:
					self.log_option = instructions['log']
				else:
					self.log_option = False
		else:
			self.rgb_mode = False
			self.log_option = False
			self.percentile_mode = True
			self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
			self.fraction = 0.25
			self.anim_interval = 1

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

	def create_cell_signal_canvas(self):

		self.cell_fig, self.cell_ax = plt.subplots(tight_layout=True)
		self.cell_fcanvas = FigureCanvas(self.cell_fig, interactive=True)
		self.cell_ax.clear()

		spacing = 0.5
		minorLocator = MultipleLocator(1)
		self.cell_ax.xaxis.set_minor_locator(minorLocator)
		self.cell_ax.xaxis.set_major_locator(MultipleLocator(5))
		self.cell_ax.grid(which='major')
		self.cell_ax.set_xlabel("time [frame]")
		self.cell_ax.set_ylabel("signal")

		self.cell_fig.set_facecolor('none')  # or 'None'
		self.cell_fig.canvas.setStyleSheet("background-color: transparent;")

		self.lines = [
			self.cell_ax.plot([np.linspace(0, self.len_movie - 1, self.len_movie)], [np.zeros((self.len_movie))])[0] for
			i in range(len(self.signal_choice_cb))]
		for i in range(len(self.lines)):
			self.lines[i].set_label(f'signal {i}')

		min_val, max_val = self.cell_ax.get_ylim()
		self.line_dt, = self.cell_ax.plot([-1, -1], [min_val, max_val], c="k", linestyle="--")

		self.cell_ax.set_xlim(0, self.len_movie)
		self.cell_ax.legend()
		self.cell_fcanvas.canvas.draw()

		self.plot_signals()

	def on_scatter_pick(self, event):

		self.event = event

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)

		ind = event.ind

		if len(ind) > 1:
			# More than one point in vicinity
			datax, datay = [self.positions[self.framedata][i, 0] for i in ind], [self.positions[self.framedata][i, 1]
																				 for i in ind]
			msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
			dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
			ind = [ind[np.argmin(dist)]]

		if len(ind) > 0 and (len(self.selection) == 0):
			
			self.selection.append([ind[0],self.framedata])
			self.select_single_cell(ind[0], self.framedata)

		elif len(ind) > 0 and len(self.selection) == 1:
			self.cancel_btn.click()
		else:
			pass

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


	def shortcut_suppr(self):
		self.correct_btn.click()
		self.suppr_btn.click()
		self.correct_btn.click()

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

	def start(self):
		'''
		Starts interactive animation. Adds the draw frame command to the GUI
		handler, calls show to start the event loop.
		'''
		self.start_btn.setShortcut(QKeySequence(""))

		self.last_frame_btn.setEnabled(True)
		self.last_frame_btn.clicked.connect(self.set_last_frame)

		self.first_frame_btn.setEnabled(True)
		self.first_frame_btn.clicked.connect(self.set_first_frame)

		self.start_btn.hide()
		self.stop_btn.show()

		self.anim.event_source.start()
		self.stop_btn.clicked.connect(self.stop)

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
		self.compute_status_and_colors(0)
		self.extract_scatter_from_trajectories()

	# self.give_cell_information()

	# def interval_slider_action(self):

	# 	print(dir(self.anim.event_source))

	# 	self.anim.event_source.interval = self.interval_slider.value()
	# 	self.anim.event_source._timer_set_interval()

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

	def export_signals(self):

		auto_dataset_name = self.pos.split(os.sep)[-4] + '_' + self.pos.split(os.sep)[-2] + '.npy'

		if self.normalized_signals:
			self.normalize_features_btn.click()

		training_set = []
		cols = self.df_tracks.columns
		tracks = np.unique(self.df_tracks["TRACK_ID"].to_numpy())

		for track in tracks:
			# Add all signals at given track
			signals = {}
			for c in cols:
				signals.update({c: self.df_tracks.loc[self.df_tracks["TRACK_ID"] == track, c].to_numpy()})
			time_of_interest = self.df_tracks.loc[self.df_tracks["TRACK_ID"] == track, self.time_name].to_numpy()[0]
			cclass = self.df_tracks.loc[self.df_tracks["TRACK_ID"] == track, self.class_name].to_numpy()[0]
			signals.update({"time_of_interest": time_of_interest, "class": cclass})
			# Here auto add all available channels
			training_set.append(signals)

		pathsave = QFileDialog.getSaveFileName(self, "Select file name", self.exp_dir + auto_dataset_name, ".npy")[0]
		if pathsave != '':
			if not pathsave.endswith(".npy"):
				pathsave += ".npy"
			try:
				np.save(pathsave, training_set)
				print(f'File successfully written in {pathsave}.')
			except Exception as e:
				print(f"Error {e}...")

	def normalize_features(self):

		x = self.df_tracks[self.columns_to_rescale].values

		if not self.normalized_signals:
			x = self.MinMaxScaler.transform(x)
			self.df_tracks[self.columns_to_rescale] = x
			self.plot_signals()
			self.normalized_signals = True
			self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical, color="#1565c0"))
			self.normalize_features_btn.setIconSize(QSize(25, 25))
		else:
			x = self.MinMaxScaler.inverse_transform(x)
			self.df_tracks[self.columns_to_rescale] = x
			self.plot_signals()
			self.normalized_signals = False
			self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical, color="black"))
			self.normalize_features_btn.setIconSize(QSize(25, 25))

	def switch_to_log(self):

		"""
		Better would be to create a log(quantity) and plot it...
		"""

		try:
			if self.cell_ax.get_yscale()=='linear':
				ymin,ymax = self.cell_ax.get_ylim()
				self.cell_ax.set_yscale('log')
				self.log_btn.setIcon(icon(MDI6.math_log,color="#1565c0"))
				self.cell_ax.set_ylim(self.value_magnitude, ymax)
			else:
				self.cell_ax.set_yscale('linear')
				self.log_btn.setIcon(icon(MDI6.math_log,color="black"))
		except Exception as e:
			print(e)

		#self.cell_ax.autoscale()
		self.cell_fcanvas.canvas.draw_idle()

class MeasureAnnotator(SignalAnnotator):

	def __init__(self, parent_window=None):

		QMainWindow.__init__(self)
		self.parent_window = parent_window
		self.setWindowTitle("Signal annotator")
		self.mode = self.parent_window.mode
		self.pos = self.parent_window.parent_window.pos
		self.exp_dir = self.parent_window.exp_dir
		self.n_signals = 3
		self.soft_path = get_software_location()
		self.recently_modified = False
		self.selection = []
		self.int_validator = QIntValidator()
		self.current_alpha=0.5
		if self.mode == "targets":
			self.instructions_path = self.exp_dir + os.sep.join(['configs','signal_annotator_config_targets.json'])
			self.trajectories_path = self.pos + os.sep.join(['output','tables','trajectories_targets.csv'])
		elif self.mode == "effectors":
			self.instructions_path = self.exp_dir + os.sep.join(['configs','signal_annotator_config_effectors.json'])
			self.trajectories_path = self.pos + os.sep.join(['output','tables','trajectories_effectors.csv'])

		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.screen_width
		self.current_frame = 0
		self.show_fliers = False
		self.status_name = 'group'

		center_window(self)

		self.locate_stack()

		data, properties, graph, labels, _ = load_napari_data(self.pos, prefix=None, population=self.mode,return_stack=False)
		# if data is not None:
		# 	self.labels = relabel_segmentation(labels,data,properties)
		# else:
		self.labels = labels

		self.current_channel = 0

		self.locate_tracks()

		self.generate_signal_choices()
		self.frame_lbl = QLabel('position: ')
		self.static_image()
		self.create_cell_signal_canvas()

		self.populate_widget()
		self.changed_class()

		self.setMinimumWidth(int(0.8 * self.screen_width))
		# self.setMaximumHeight(int(0.8*self.screen_height))
		self.setMinimumHeight(int(0.8 * self.screen_height))
		# self.setMaximumHeight(int(0.8*self.screen_height))

		self.setAttribute(Qt.WA_DeleteOnClose)
		self.previous_index = None

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

	def create_cell_signal_canvas(self):

		self.cell_fig, self.cell_ax = plt.subplots()
		self.cell_fcanvas = FigureCanvas(self.cell_fig, interactive=False)
		self.cell_ax.clear()

		# spacing = 0.5
		# minorLocator = MultipleLocator(1)
		# self.cell_ax.xaxis.set_minor_locator(minorLocator)
		# self.cell_ax.xaxis.set_major_locator(MultipleLocator(5))
		self.cell_ax.grid(which='major')
		self.cell_ax.set_xlabel("time [frame]")
		self.cell_ax.set_ylabel("signal")

		self.cell_fig.set_facecolor('none')  # or 'None'
		self.cell_fig.canvas.setStyleSheet("background-color: transparent;")

		self.lines = [
			self.cell_ax.plot([np.linspace(0, self.len_movie - 1, self.len_movie)], [np.zeros(self.len_movie)])[0] for
			i in range(len(self.signal_choice_cb))]
		for i in range(len(self.lines)):
			self.lines[i].set_label(f'signal {i}')

		min_val, max_val = self.cell_ax.get_ylim()
		self.line_dt, = self.cell_ax.plot([-1, -1], [min_val, max_val], c="k", linestyle="--")

		self.cell_ax.set_xlim(0, self.len_movie)
		self.cell_fcanvas.canvas.draw()

		self.plot_signals()

	def plot_signals(self):
		
		#try:
		current_frame = self.current_frame  # Assuming you have a variable for the current frame
		
		yvalues = []
		all_yvalues = []
		current_yvalues = []
		all_median_values = []
		labels = []

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
				labels.append(signal_choice)

		self.cell_ax.clear()

		if len(yvalues) > 0:
			self.cell_ax.boxplot(all_yvalues, showfliers=self.show_fliers)
			ylim = self.cell_ax.get_ylim()
			self.cell_ax.set_ylim(ylim)
			x_pos = np.arange(len(all_yvalues)) + 1

			for index, feature in enumerate(current_yvalues):
				x_values_strip = (index + 1) + np.random.normal(0, 0.04, size=len(
					feature))
				self.cell_ax.plot(x_values_strip, feature, marker='o', linestyle='None', color=tab10.colors[0],
								  alpha=0.1)
			
			self.cell_ax.plot(x_pos, yvalues, marker='H', linestyle='None', color=tab10.colors[3], alpha=1)


		else:
			self.cell_ax.text(0.5, 0.5, "No data available", horizontalalignment='center',
							  verticalalignment='center', transform=self.cell_ax.transAxes)

		self.cell_fcanvas.canvas.draw()

		# except Exception as e:
		# 	print("plot_signals: ",f"{e=}")

	def configure_ylims(self):

		try:
			min_values = []
			max_values = []
			for i in range(len(self.signal_choice_cb)):
				signal = self.signal_choice_cb[i].currentText()
				if signal == '--':
					continue
				else:
					maxx = np.max(self.df_tracks.loc[:, signal].to_numpy().flatten())
					minn = np.min(self.df_tracks.loc[:, signal].to_numpy().flatten())
					min_values.append(minn)
					max_values.append(maxx)

			if len(min_values) > 0:
				self.cell_ax.set_ylim(np.amin(min_values), np.amax(max_values))
		except Exception as e:
			print(e)

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

	def on_scatter_pick(self, event):
		ind = event.ind
		if len(ind) > 1:
			# More than one point in vicinity
			datax, datay = [self.positions[self.framedata][i, 0] for i in ind], [self.positions[self.framedata][i, 1]
																				 for i in ind]
			msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
			dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
			ind = [ind[np.argmin(dist)]]

		if len(ind) > 0 and (len(self.selection) == 0):
			ind = ind[0]
			self.selection.append(ind)
			self.correct_btn.setEnabled(True)
			self.cancel_btn.setEnabled(True)
			self.del_shortcut.setEnabled(True)
			self.no_event_shortcut.setEnabled(True)
			self.track_of_interest = self.tracks[self.framedata][ind]
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

		elif len(ind) > 0 and len(self.selection) == 1:
			self.cancel_btn.click()
		else:
			pass
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def populate_widget(self):

		"""
		Create the multibox design.

		"""

		self.button_widget = QWidget()
		main_layout = QHBoxLayout()
		self.button_widget.setLayout(main_layout)

		main_layout.setContentsMargins(30, 30, 30, 30)

		self.left_panel = QVBoxLayout()
		self.left_panel.setContentsMargins(30, 5, 30, 5)
		#self.left_panel.setSpacing(3)

		self.right_panel = QVBoxLayout()

		class_hbox = QHBoxLayout()
		class_hbox.setContentsMargins(0,0,0,0)
		class_hbox.setSpacing(0)

		class_hbox.addWidget(QLabel('characteristic \n group: '), 25)
		self.class_choice_cb = QComboBox()

		cols = np.array(self.df_tracks.columns)
		self.class_cols = np.array([c.startswith('group') or c.startswith('status') for c in list(self.df_tracks.columns)])
		self.class_cols = list(cols[self.class_cols])

		try:
			self.class_cols.remove('group_id')
		except Exception:
			pass
		try:
			self.class_cols.remove('group_color')
		except Exception:
			pass

		self.class_choice_cb.addItems(self.class_cols)
		self.class_choice_cb.currentIndexChanged.connect(self.changed_class)
		class_hbox.addWidget(self.class_choice_cb, 70)

		self.add_class_btn = QPushButton('')
		self.add_class_btn.setStyleSheet(self.button_select_all)
		self.add_class_btn.setIcon(icon(MDI6.plus, color="black"))
		self.add_class_btn.setToolTip("Add a new characteristic group")
		self.add_class_btn.setIconSize(QSize(20, 20))
		self.add_class_btn.clicked.connect(self.create_new_event_class)
		class_hbox.addWidget(self.add_class_btn, 5)

		self.del_class_btn = QPushButton('')
		self.del_class_btn.setStyleSheet(self.button_select_all)
		self.del_class_btn.setIcon(icon(MDI6.delete, color="black"))
		self.del_class_btn.setToolTip("Delete a characteristic group")
		self.del_class_btn.setIconSize(QSize(20, 20))
		self.del_class_btn.clicked.connect(self.del_event_class)
		class_hbox.addWidget(self.del_class_btn, 5)

		self.left_panel.addLayout(class_hbox,5)

		self.cell_info = QLabel('')
		self.left_panel.addWidget(self.cell_info,5)

		time_option_hbox = QHBoxLayout()
		time_option_hbox.setContentsMargins(100, 0, 100, 0)
		time_option_hbox.setSpacing(0)

		self.time_of_interest_label = QLabel('phenotype: ')
		time_option_hbox.addWidget(self.time_of_interest_label, 30)
		self.time_of_interest_le = QLineEdit()
		self.time_of_interest_le.setValidator(self.int_validator)
		time_option_hbox.addWidget(self.time_of_interest_le)
		self.del_cell_btn = QPushButton('')
		self.del_cell_btn.setStyleSheet(self.button_select_all)
		self.del_cell_btn.setIcon(icon(MDI6.delete, color="black"))
		self.del_cell_btn.setToolTip("Delete cell")
		self.del_cell_btn.setIconSize(QSize(20, 20))
		self.del_cell_btn.clicked.connect(self.del_cell)
		time_option_hbox.addWidget(self.del_cell_btn)
		self.left_panel.addLayout(time_option_hbox,5)

		main_action_hbox = QHBoxLayout()
		main_action_hbox.setContentsMargins(0,0,0,0)
		main_action_hbox.setSpacing(0)

		self.correct_btn = QPushButton('correct')
		self.correct_btn.setIcon(icon(MDI6.redo_variant, color="white"))
		self.correct_btn.setIconSize(QSize(20, 20))
		self.correct_btn.setStyleSheet(self.button_style_sheet)
		self.correct_btn.clicked.connect(self.show_annotation_buttons)
		self.correct_btn.setEnabled(False)
		main_action_hbox.addWidget(self.correct_btn)

		self.cancel_btn = QPushButton('cancel')
		self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
		self.cancel_btn.setShortcut(QKeySequence("Esc"))
		self.cancel_btn.setEnabled(False)
		self.cancel_btn.clicked.connect(self.cancel_selection)
		main_action_hbox.addWidget(self.cancel_btn)
		self.left_panel.addLayout(main_action_hbox,5)

		self.annotation_btns_to_hide = [self.time_of_interest_label,
										self.time_of_interest_le,
										self.del_cell_btn]
		self.hide_annotation_buttons()
		#### End of annotation buttons

		self.del_shortcut = QShortcut(Qt.Key_Delete, self)  # QKeySequence("s")
		self.del_shortcut.activated.connect(self.shortcut_suppr)
		self.del_shortcut.setEnabled(False)

		self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
		self.no_event_shortcut.activated.connect(self.shortcut_no_event)
		self.no_event_shortcut.setEnabled(False)

		# Cell signals
		self.cell_fcanvas.setMinimumHeight(int(0.2*self.screen_height))
		self.left_panel.addWidget(self.cell_fcanvas,90)

		plot_buttons_hbox = QHBoxLayout()
		plot_buttons_hbox.setContentsMargins(0, 0, 0, 0)
		self.outliers_check = QCheckBox('Show outliers')
		self.outliers_check.toggled.connect(self.show_outliers)

		self.normalize_features_btn = QPushButton('')
		self.normalize_features_btn.setStyleSheet(self.button_select_all)
		self.normalize_features_btn.setIcon(icon(MDI6.arrow_collapse_vertical, color="black"))
		self.normalize_features_btn.setIconSize(QSize(25, 25))
		self.normalize_features_btn.setFixedSize(QSize(30, 30))
		# self.normalize_features_btn.setShortcut(QKeySequence('n'))
		self.normalize_features_btn.clicked.connect(self.normalize_features)

		plot_buttons_hbox.addWidget(QLabel(''), 90)
		plot_buttons_hbox.addWidget(self.outliers_check)
		plot_buttons_hbox.addWidget(self.normalize_features_btn, 5)
		self.normalized_signals = False

		self.log_btn = QPushButton()
		self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
		self.log_btn.setStyleSheet(self.button_select_all)
		self.log_btn.clicked.connect(self.switch_to_log)
		plot_buttons_hbox.addWidget(self.log_btn, 5)

		self.left_panel.addLayout(plot_buttons_hbox,5)

		signal_choice_vbox = QVBoxLayout()
		signal_choice_vbox.setContentsMargins(30, 0, 30, 50)
		for i in range(len(self.signal_choice_cb)):
			hlayout = QHBoxLayout()
			hlayout.addWidget(self.signal_choice_label[i], 20)
			hlayout.addWidget(self.signal_choice_cb[i], 75)
			# hlayout.addWidget(self.log_btns[i], 5)
			signal_choice_vbox.addLayout(hlayout)

		# self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
		# self.log_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)
		# self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		self.left_panel.addLayout(signal_choice_vbox,10)

		btn_hbox = QHBoxLayout()
		self.save_btn = QPushButton('Save')
		self.save_btn.setStyleSheet(self.button_style_sheet)
		self.save_btn.clicked.connect(self.save_trajectories)
		btn_hbox.addWidget(self.save_btn, 90)

		self.export_btn = QPushButton('')
		self.export_btn.setStyleSheet(self.button_select_all)
		self.export_btn.clicked.connect(self.export_measurements)
		self.export_btn.setIcon(icon(MDI6.export, color="black"))
		self.export_btn.setIconSize(QSize(25, 25))
		btn_hbox.addWidget(self.export_btn, 10)
		self.left_panel.addLayout(btn_hbox,5)

		# Animation
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
		self.contrast_slider.setOrientation(1)
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
		self.alpha_slider.setOrientation(1)
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

		main_layout.addLayout(self.left_panel, 35)
		main_layout.addLayout(self.right_panel, 65)
		self.button_widget.adjustSize()

		self.setCentralWidget(self.button_widget)
		self.show()

		QApplication.processEvents()

	def closeEvent(self, event):
		# result = QMessageBox.question(self,
		# 			  "Confirm Exit...",
		# 			  "Are you sure you want to exit ?",
		# 			  QMessageBox.Yes| QMessageBox.No,
		# 			  )
		# del self.img
		gc.collect()


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
		self.class_choice_cb.clear()
		cols = np.array(self.df_tracks.columns)
		self.class_cols = np.array([c.startswith('group') for c in list(self.df_tracks.columns)])
		self.class_cols = list(cols[self.class_cols])
		self.class_cols.remove('group_color')
		self.class_choice_cb.addItems(self.class_cols)
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
		self.newClassWidget = QWidget()
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
		self.no_event_shortcut.setEnabled(False)
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
		print('compute status and colors!')
		if self.class_choice_cb.currentText() == '':
			self.status_name=self.target_class
		else:
			self.status_name = self.class_choice_cb.currentText()

		print(f'{self.status_name=}')
		if self.status_name not in self.df_tracks.columns:
			print('not in df, make column')
			self.make_status_column()
		else:
			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			print(f'{self.state_color_map=}')
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)
			print(self.df_tracks['group_color'])

	def del_event_class(self):

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Warning)
		msgBox.setText(
			f"You are about to delete characteristic group {self.class_choice_cb.currentText()}. Do you still want to proceed?")
		msgBox.setWindowTitle("Warning")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:
			class_to_delete = self.class_choice_cb.currentText()
			cols_to_delete = [class_to_delete]
			for c in cols_to_delete:
				try:
					self.df_tracks = self.df_tracks.drop([c], axis=1)
				except Exception as e:
					print(e)
			item_idx = self.class_choice_cb.findText(class_to_delete)
			self.class_choice_cb.removeItem(item_idx)

	def make_status_column(self):
		if self.status_name == "state_firstdetection":
			pass
		else:
			self.df_tracks.loc[:, self.status_name] = 0
			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)

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
			self.class_cols = np.array([c.startswith('group') for c in list(self.df_tracks.columns)])
			self.class_cols = list(cols[self.class_cols])
			try:
				self.class_cols.remove('class_id')
			except:
				pass
			try:
				self.class_cols.remove('group_color')
			except:
				pass
			if len(self.class_cols) > 0:
				self.status = self.class_cols[0]

			else:

				self.status_name = 'group'

			if self.status_name not in self.df_tracks.columns:
				# only create the status column if it does not exist to not erase static classification results
				self.make_status_column()
			else:
				# all good, do nothing
				pass
			# else:
			#     if not self.status_name in self.df_tracks.columns:
			#         self.df_tracks[self.status_name] = 0
			#         self.df_tracks['state_color'] = color_from_status(0)
			#			self.df_tracks['class_color'] = color_from_class(1)

			# if not self.class_name in self.df_tracks.columns:
			# 	self.df_tracks[self.class_name] = 1
			# if not self.time_name in self.df_tracks.columns:
			# 	self.df_tracks[self.time_name] = -1
			all_states = self.df_tracks.loc[:, self.status_name].tolist()
			all_states = np.array(all_states)
			self.state_color_map = color_from_state(all_states, recently_modified=False)
			self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)
			# self.df_tracks['status_color'] = [color_from_status(i) for i in self.df_tracks[self.status_name].to_numpy()]
			# self.df_tracks['class_color'] = [color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()]

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

			# is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
			# is_number_test = is_number(self.df_tracks.dtypes)
			# self.columns_to_rescale = [col for t,col in zip(is_number_test,self.df_tracks.columns) if t]
			# print(self.columns_to_rescale)

			cols_to_remove = ['group', 'group_color', 'status', 'status_color', 'class_color', 'TRACK_ID', 'FRAME',
							  'x_anim', 'y_anim', 't',
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

			# print(f'column {tr} could not be found...')

			x = self.df_tracks[self.columns_to_rescale].values
			self.MinMaxScaler.fit(x)

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
		self.fcanvas.canvas.draw()
		self.plot_signals()

	# def load_annotator_config(self):
	#         self.rgb_mode = False
	#         self.log_option = False
	#         self.percentile_mode = True
	#         self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
	#         self.fraction = 0.5955056179775281
	#         #self.anim_interval = 1

	def prepare_stack(self):

		self.img_num_channels = _get_img_num_per_channel(self.channels, self.len_movie, self.nbr_channels)
		self.current_stack = []
		for ch in tqdm(self.target_channels, desc="channel"):
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

			self.current_stack.append(chan)

		self.current_stack = np.array(self.current_stack)
		if self.rgb_mode:
			self.current_stack = np.moveaxis(self.current_stack, 0, -1)
		else:
			self.current_stack = self.current_stack[0]
			if self.log_option:
				self.current_stack[np.where((self.current_stack > 0.)&(self.current_stack==self.current_stack))] = np.log(
					self.current_stack[np.where((self.current_stack > 0.)&(self.current_stack==self.current_stack))])

	def changed_channel(self):

		self.reload_frame()
		self.contrast_slider.setRange(
			*[np.nanpercentile(self.img, 0.001),
			  np.nanpercentile(self.img, 99.999)])
		self.contrast_slider.setValue(
			[np.nanpercentile(self.img, 1), np.nanpercentile(self.img, 99.99)])
		self.draw_frame(self.current_frame)
		self.fcanvas.canvas.draw()

	def save_trajectories(self):

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
		self.update_frame()


	# self.extract_scatter_from_trajectories()

	def modify(self):

		all_states = self.df_tracks.loc[:, self.status_name].tolist()
		all_states = np.array(all_states)
		self.state_color_map = color_from_state(all_states, recently_modified=False)
		print(f'{self.state_color_map=}')

		self.df_tracks['group_color'] = self.df_tracks[self.status_name].apply(self.assign_color_state)

		self.extract_scatter_from_trajectories()
		self.give_cell_information()

		self.correct_btn.disconnect()
		self.correct_btn.clicked.connect(self.show_annotation_buttons)

		# self.hide_annotation_buttons()
		# self.correct_btn.setEnabled(False)
		# self.correct_btn.setText('correct')
		# self.cancel_btn.setEnabled(False)
		# self.del_shortcut.setEnabled(False)
		# self.no_event_shortcut.setEnabled(False)

	def enable_time_of_interest(self):
		if self.suppr_btn.isChecked():
			self.time_of_interest_le.setEnabled(False)

	def cancel_selection(self):

		self.hide_annotation_buttons()
		self.correct_btn.setEnabled(False)
		self.correct_btn.setText('correct')
		self.cancel_btn.setEnabled(False)

		try:
			self.selection.pop(0)
		except Exception as e:
			print('Cancel selection: ',e)

		try:
			for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
				# print(self.colors[t][idx, 1])
				self.colors[t][idx, 0] = self.previous_color[k][0]
			# self.colors[t][idx, 1] = self.previous_color[k][1]
		except Exception as e:
			print("cancel_selection: ",f'{e=}')

		self.event = None

	def locate_stack(self):

		"""
		Locate the target movie.

		"""

		if isinstance(self.pos, str):
			movies = glob(self.pos + os.sep.join(['movie',f"{self.parent_window.parent_window.movie_prefix}*.tif"]))

		else:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("Please select a unique position before launching the wizard...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.img = None
				self.close()
				return None

		if len(movies) == 0:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText("No movie is detected in the experiment folder.\nPlease check the stack prefix...")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.close()
		else:
			self.stack_path = movies[0]
			self.len_movie = self.parent_window.parent_window.len_movie
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

	def reload_frame(self):

		"""
		Load the frame from the current channel and time choice. Show imshow, update histogram.
		"""

		# self.clear_post_threshold_options()

		self.current_channel = self.choose_channel.currentIndex()

		t = int(self.frame_slider.value())
		idx = t * self.nbr_channels + self.current_channel
		self.img = load_frames(idx, self.stack_path, normalize_input=False)
		if self.img is not None:
			self.refresh_imshow()
		# self.redo_histogram()
		else:
			print('Frame could not be loaded...')

	def refresh_imshow(self):

		"""

		Update the imshow based on the current frame selection.

		"""

		self.vmin = np.nanpercentile(self.img.flatten(), 1)
		self.vmax = np.nanpercentile(self.img.flatten(), 99.)

		self.contrast_slider.disconnect()
		self.contrast_slider.setRange(np.nanmin(self.img), np.nanmax(self.img))
		self.contrast_slider.setValue([self.vmin, self.vmax])
		self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

		self.im.set_data(self.img)

	# self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
	# self.fcanvas.canvas.draw_idle()

	def show_outliers(self):
		if self.outliers_check.isChecked():
			self.show_fliers = True
			self.plot_signals()
		else:
			self.show_fliers = False
			self.plot_signals()

	def del_cell(self):
		self.time_of_interest_le.setEnabled(False)
		self.time_of_interest_le.setText("99")
		self.apply_modification()

	def shortcut_suppr(self):
		self.correct_btn.click()
		self.del_cell_btn.click()
		self.correct_btn.click()
