from PyQt5.QtWidgets import QWidget, QLineEdit, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, \
	QCheckBox, QRadioButton, QButtonGroup, QComboBox
from PyQt5.QtCore import Qt, QSize
from superqt import QLabeledSlider,QLabeledDoubleSlider, QSearchableComboBox
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

import os
import numpy as np
import matplotlib.pyplot as plt
import json

from celldetective.gui.gui_utils import FigureCanvas, center_window, color_from_status, help_generic
from celldetective.gui import Styles
from celldetective.utils import get_software_location
from celldetective.measure import classify_cells_from_query, interpret_track_classification

class ClassifierWidget(QWidget, Styles):

	def __init__(self, parent_window):

		super().__init__()

		self.parent_window = parent_window
		self.screen_height = self.parent_window.parent_window.parent_window.screen_height
		self.screen_width = self.parent_window.parent_window.parent_window.screen_width
		self.currentAlpha = 1.0

		self.setWindowTitle("Custom classification")

		self.mode = self.parent_window.mode
		self.df = self.parent_window.df

		is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
		is_number_test = is_number(self.df.dtypes)
		self.cols = [col for t,col in zip(is_number_test,self.df.columns) if t]

		self.class_name = 'custom'
		self.name_le = QLineEdit(self.class_name)
		self.init_class()

		# Create the QComboBox and add some items
		center_window(self)

		
		layout = QVBoxLayout(self)
		layout.setContentsMargins(30,30,30,30)

		name_layout = QHBoxLayout()
		name_layout.addWidget(QLabel('class name: '), 33)
		name_layout.addWidget(self.name_le, 66)
		layout.addLayout(name_layout)

		fig_btn_hbox = QHBoxLayout()
		fig_btn_hbox.addWidget(QLabel(''), 95)
		self.project_times_btn = QPushButton('')
		self.project_times_btn.setStyleSheet(self.parent_window.parent_window.parent_window.button_select_all)
		self.project_times_btn.setIcon(icon(MDI6.math_integral,color="black"))
		self.project_times_btn.setToolTip("Project measurements at all times.")
		self.project_times_btn.setIconSize(QSize(20, 20))
		self.project_times = False
		self.project_times_btn.clicked.connect(self.switch_projection)
		fig_btn_hbox.addWidget(self.project_times_btn, 5)
		layout.addLayout(fig_btn_hbox)

		# Figure
		self.initalize_props_scatter()
		layout.addWidget(self.propscanvas)

		# slider
		self.frame_slider = QLabeledSlider()
		self.frame_slider.setSingleStep(1)
		self.frame_slider.setOrientation(1)
		self.frame_slider.setRange(0,int(self.df.FRAME.max()) - 1)
		self.frame_slider.setValue(0)
		self.currentFrame = 0

		slider_hbox = QHBoxLayout()
		slider_hbox.addWidget(QLabel('frame: '), 10)
		slider_hbox.addWidget(self.frame_slider, 90)
		layout.addLayout(slider_hbox)


		# transparency slider
		self.alpha_slider = QLabeledDoubleSlider()
		self.alpha_slider.setSingleStep(0.001)
		self.alpha_slider.setOrientation(1)
		self.alpha_slider.setRange(0,1)
		self.alpha_slider.setValue(1.0)
		self.alpha_slider.setDecimals(3)

		slider_alpha_hbox = QHBoxLayout()
		slider_alpha_hbox.addWidget(QLabel('transparency: '), 10)
		slider_alpha_hbox.addWidget(self.alpha_slider, 90)
		layout.addLayout(slider_alpha_hbox)


		self.features_cb = [QSearchableComboBox() for i in range(2)]
		self.log_btns = [QPushButton() for i in range(2)]

		for i in range(2):
			hbox_feat = QHBoxLayout()
			hbox_feat.addWidget(QLabel(f'feature {i}: '), 20)
			hbox_feat.addWidget(self.features_cb[i], 75)
			hbox_feat.addWidget(self.log_btns[i], 5)
			layout.addLayout(hbox_feat)

			self.features_cb[i].clear()
			self.features_cb[i].addItems(sorted(list(self.cols),key=str.lower))
			self.features_cb[i].currentTextChanged.connect(self.update_props_scatter)
			self.features_cb[i].setCurrentIndex(i)

			self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			self.log_btns[i].setStyleSheet(self.button_select_all)
			self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		hbox_classify = QHBoxLayout()
		hbox_classify.addWidget(QLabel('classify: '), 10)
		self.property_query_le = QLineEdit()
		self.property_query_le.setPlaceholderText('classify points using a query such as: area > 100 or eccentricity > 0.95')
		self.property_query_le.setToolTip('Classify points using a query on measurements.\nYou can use "and" and "or" conditions to combine\nmeasurements (e.g. "area > 100 or eccentricity > 0.95").')
		self.property_query_le.textChanged.connect(self.activate_submit_btn)
		hbox_classify.addWidget(self.property_query_le, 70)
		self.submit_query_btn = QPushButton('Submit...')
		self.submit_query_btn.clicked.connect(self.apply_property_query)
		self.submit_query_btn.setEnabled(False)
		hbox_classify.addWidget(self.submit_query_btn, 20)
		layout.addLayout(hbox_classify)

		self.time_corr = QCheckBox('Time correlated')
		if "TRACK_ID" in self.df.columns:
			self.time_corr.setEnabled(True)
		else:
			self.time_corr.setEnabled(False)

		time_prop_hbox = QHBoxLayout()
		time_prop_hbox.addWidget(self.time_corr,alignment=Qt.AlignCenter)

		self.help_propagate_btn = QPushButton()
		self.help_propagate_btn.setIcon(icon(MDI6.help_circle,color=self.help_color))
		self.help_propagate_btn.setIconSize(QSize(20, 20))
		self.help_propagate_btn.clicked.connect(self.help_propagate)
		self.help_propagate_btn.setStyleSheet(self.button_select_all)
		self.help_propagate_btn.setToolTip("Help.")
		time_prop_hbox.addWidget(self.help_propagate_btn,5,alignment=Qt.AlignRight)

		layout.addLayout(time_prop_hbox)

		self.irreversible_event_btn = QRadioButton('irreversible event')
		self.unique_state_btn = QRadioButton('unique state')
		self.transient_event_btn = QRadioButton('transient event')
		time_corr_btn_group = QButtonGroup()
		self.unique_state_btn.click()

		time_corr_layout = QHBoxLayout()
		time_corr_layout.addWidget(self.unique_state_btn, 33, alignment=Qt.AlignCenter)
		time_corr_layout.addWidget(self.irreversible_event_btn, 33,alignment=Qt.AlignCenter)
		time_corr_layout.addWidget(self.transient_event_btn, 33,alignment=Qt.AlignCenter)
		layout.addLayout(time_corr_layout)

		self.prereq_event_check = QCheckBox('prerequisite event:')
		self.prereq_event_check.toggled.connect(self.activate_prereq_cb)
		self.prereq_event_cb = QComboBox()
		event_cols = ['--'] + [c.replace('t_','') for c in self.cols if c.startswith('t_')]
		self.prereq_event_cb.addItems(event_cols)
		self.prereq_event_check.setEnabled(False)
		self.prereq_event_cb.setEnabled(False)

		self.r2_slider = QLabeledDoubleSlider()
		self.r2_slider.setValue(0.75)
		self.r2_slider.setRange(0,1)
		self.r2_slider.setSingleStep(0.01)
		self.r2_slider.setOrientation(1)
		self.r2_label = QLabel('R2 tolerance:')
		self.r2_label.setToolTip('Minimum R2 between the fit sigmoid and the binary response to the filters to accept the event.')

		r2_threshold_layout = QHBoxLayout()
		r2_threshold_layout.addWidget(QLabel(''), 33)
		r2_threshold_layout.addWidget(self.r2_label, 13)
		r2_threshold_layout.addWidget(self.r2_slider, 20)
		r2_threshold_layout.addWidget(QLabel(''), 33)

		layout.addLayout(r2_threshold_layout)
		
		self.time_corr_options = [self.irreversible_event_btn, self.unique_state_btn, self.prereq_event_check, self.prereq_event_cb, self.transient_event_btn]
		for btn in [self.irreversible_event_btn, self.unique_state_btn, self.transient_event_btn]:
			time_corr_btn_group.addButton(btn)
			btn.setEnabled(False)
		self.time_corr.toggled.connect(self.activate_time_corr_options)

		self.irreversible_event_btn.clicked.connect(self.activate_r2)
		self.unique_state_btn.clicked.connect(self.activate_r2)
		self.transient_event_btn.clicked.connect(self.activate_r2)

		for wg in [self.r2_slider, self.r2_label]:
			wg.setEnabled(False)

		prereq_layout = QHBoxLayout()
		prereq_layout.setContentsMargins(30,0,0,0)
		prereq_layout.addWidget(self.prereq_event_check, 20)
		prereq_layout.addWidget(self.prereq_event_cb, 80)
		layout.addLayout(prereq_layout)

		layout.addWidget(QLabel())

		self.submit_btn = QPushButton('apply')
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.submit_classification)
		self.submit_btn.setEnabled(False)
		layout.addWidget(self.submit_btn, 30)

		self.frame_slider.valueChanged.connect(self.set_frame)
		self.alpha_slider.valueChanged.connect(self.set_transparency)

		self.setAttribute(Qt.WA_DeleteOnClose)

	def activate_prereq_cb(self):
		if self.prereq_event_check.isChecked():
			self.prereq_event_cb.setEnabled(True)
		else:
			self.prereq_event_cb.setEnabled(False)

	def activate_submit_btn(self):

		if self.property_query_le.text()=='':
			self.submit_query_btn.setEnabled(False)
			self.submit_btn.setEnabled(False)
		else:
			self.submit_query_btn.setEnabled(True)
			self.submit_btn.setEnabled(True)

	def activate_r2(self):
		if self.irreversible_event_btn.isChecked() and self.time_corr.isChecked():
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(True)
		else:
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(False)				

	def activate_time_corr_options(self):

		if self.time_corr.isChecked():
			for btn in self.time_corr_options:
				btn.setEnabled(True)
			if self.irreversible_event_btn.isChecked():
				for wg in [self.r2_slider, self.r2_label]:
					wg.setEnabled(True)
			else:
				for wg in [self.r2_slider, self.r2_label]:
					wg.setEnabled(False)				
		else:
			for btn in self.time_corr_options:
				btn.setEnabled(False)
			for wg in [self.r2_slider, self.r2_label]:
				wg.setEnabled(False)

	def init_class(self):

		self.class_name = 'custom'
		i=1
		while self.class_name in self.df.columns:
			self.class_name = f'custom_{i}'
			i+=1
		self.name_le.setText(self.class_name)
		self.df.loc[:,self.class_name] = 1

	def initalize_props_scatter(self):

		"""
		Define properties scatter.
		"""

		self.fig_props, self.ax_props = plt.subplots(figsize=(4,4),tight_layout=True)
		self.propscanvas = FigureCanvas(self.fig_props, interactive=True)
		self.fig_props.set_facecolor('none')
		self.fig_props.canvas.setStyleSheet("background-color: transparent;")
		self.scat_props = self.ax_props.scatter([],[], color="k", alpha=self.currentAlpha)
		self.propscanvas.canvas.draw_idle()
		self.propscanvas.canvas.setMinimumHeight(self.screen_height//5)

	def closeEvent(self, event):
		self.ax_props.cla()
		self.fig_props.clf()
		plt.close(self.fig_props)
		super().closeEvent(event) 

	def update_props_scatter(self, feature_changed=True):

		try:
			if np.any(self.df[self.features_cb[0].currentText()].to_numpy() <= 0.):
				if self.ax_props.get_yscale()=='log':
					self.log_btns[0].click()
				self.log_btns[0].setEnabled(False)
			else:
				self.log_btns[0].setEnabled(True)

			if np.any(self.df[self.features_cb[1].currentText()].to_numpy() <= 0.):
				if self.ax_props.get_xscale()=='log':
					self.log_btns[1].click()
				self.log_btns[1].setEnabled(False)
			else:
				self.log_btns[1].setEnabled(True)
		except Exception as e:
			#print(e)
			pass

		class_name = self.class_name

		try:

			if not self.project_times:
				self.scat_props.set_offsets(self.df.loc[self.df['FRAME']==self.currentFrame,[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
				colors = [color_from_status(c) for c in self.df.loc[self.df['FRAME']==self.currentFrame,class_name].to_numpy()]
				self.scat_props.set_facecolor(colors)
			else:
				self.scat_props.set_offsets(self.df[[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
				colors = [color_from_status(c) for c in self.df[class_name].to_numpy()]
				self.scat_props.set_facecolor(colors)
			
			self.scat_props.set_alpha(self.currentAlpha)

			if feature_changed:
	
				self.ax_props.set_xlabel(self.features_cb[1].currentText())
				self.ax_props.set_ylabel(self.features_cb[0].currentText())

				feat_x = self.features_cb[1].currentText()
				feat_y = self.features_cb[0].currentText()
				min_x = self.df.dropna(subset=feat_x)[feat_x].min()
				max_x = self.df.dropna(subset=feat_x)[feat_x].max()
				min_y = self.df.dropna(subset=feat_y)[feat_y].min()
				max_y = self.df.dropna(subset=feat_y)[feat_y].max()
				
				x_padding = (max_x - min_x) * 0.05
				y_padding = (max_y - min_y) * 0.05
				if x_padding==0:
					x_padding = 0.05
				if y_padding==0:
					y_padding = 0.05

				if min_x==min_x and max_x==max_x:
					if self.ax_props.get_xscale()=='linear':
						self.ax_props.set_xlim(min_x - x_padding, max_x + x_padding)
					else:
						self.ax_props.set_xlim(min_x, max_x)
				if min_y==min_y and max_y==max_y:
					if self.ax_props.get_yscale()=='linear':
						self.ax_props.set_ylim(min_y - y_padding, max_y + y_padding)
					else:
						self.ax_props.set_ylim(min_y, max_y)						
			
				self.propscanvas.canvas.toolbar.update()

			self.propscanvas.canvas.draw_idle()
		
		except Exception as e:
			pass

	def apply_property_query(self):
		
		query = self.property_query_le.text()
		
		try:
			self.df = classify_cells_from_query(self.df, self.name_le.text(), query)
		except Exception as e:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(f"The query could not be understood. No filtering was applied. {e}")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.auto_close = False
				return None

		self.class_name = "status_"+self.name_le.text()
		if self.df is None:
			msgBox = QMessageBox()
			msgBox.setIcon(QMessageBox.Warning)
			msgBox.setText(f"The query could not be understood. No filtering was applied. {e}")
			msgBox.setWindowTitle("Warning")
			msgBox.setStandardButtons(QMessageBox.Ok)
			returnValue = msgBox.exec()
			if returnValue == QMessageBox.Ok:
				self.auto_close = False
				return None

		self.update_props_scatter(feature_changed=False)

	def set_frame(self, value):
		xlim=self.ax_props.get_xlim()
		ylim=self.ax_props.get_ylim()
		self.currentFrame = value
		self.update_props_scatter(feature_changed=False)
		self.ax_props.set_xlim(xlim)
		self.ax_props.set_ylim(ylim)


	def set_transparency(self, value):
		xlim=self.ax_props.get_xlim()
		ylim=self.ax_props.get_ylim()
		self.currentAlpha = value
		#fc = self.scat_props.get_facecolors()
		#fc[:, 3] = value
		#self.scat_props.set_facecolors(fc)
		#self.propscanvas.canvas.draw_idle()
		self.update_props_scatter(feature_changed=False)
		self.ax_props.set_xlim(xlim)
		self.ax_props.set_ylim(ylim)

	def switch_projection(self):
		if self.project_times:
			self.project_times = False
			self.project_times_btn.setIcon(icon(MDI6.math_integral,color="black"))
			self.project_times_btn.setIconSize(QSize(20, 20))
			self.frame_slider.setEnabled(True)
		else:
			self.project_times = True
			self.project_times_btn.setIcon(icon(MDI6.math_integral_box,color="black"))
			self.project_times_btn.setIconSize(QSize(20, 20))
			self.frame_slider.setEnabled(False)
		self.update_props_scatter(feature_changed=False)

	def submit_classification(self):
		
		self.auto_close = True
		self.apply_property_query()
		if not self.auto_close:
			return None

		if self.time_corr.isChecked():
			self.class_name_user = 'class_'+self.name_le.text()
			print(f'User defined class name: {self.class_name_user}.')
			if self.class_name_user in self.df.columns:

				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				msgBox.setText(f"The class column {self.class_name_user} already exists in the table.\nProceeding will reclassify. Do you want to continue?")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					pass
				else:
					return None

			name_map = {self.class_name: self.class_name_user}
			print(f"{name_map=}")
			self.df = self.df.drop(list(set(name_map.values()) & set(self.df.columns)), axis=1).rename(columns=name_map)
			self.df.reset_index(inplace=True, drop=True)

			pre_event = None
			if self.prereq_event_check.isChecked() and "t_"+self.prereq_event_cb.currentText() in self.cols:
				pre_event = self.prereq_event_cb.currentText()

			self.df = interpret_track_classification(self.df, self.class_name_user, irreversible_event=self.irreversible_event_btn.isChecked(), unique_state=self.unique_state_btn.isChecked(), transient_event=self.transient_event_btn.isChecked(),r2_threshold=self.r2_slider.value(), pre_event=pre_event)
		
		else:
			self.group_name_user = 'group_' + self.name_le.text()
			print(f'User defined characteristic group name: {self.group_name_user}.')
			if self.group_name_user in self.df.columns:

				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				msgBox.setText(
					f"The group column {self.group_name_user} already exists in the table.\nProceeding will reclassify. Do you want to continue?")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Yes:
					pass
				else:
					return None

			name_map = {self.class_name: self.group_name_user}
			self.df = self.df.drop(list(set(name_map.values()) & set(self.df.columns)), axis=1).rename(columns=name_map)
			print(self.df.columns)
			#self.df[self.group_name_user] = self.df[self.group_name_user].replace({0: 1, 1: 0})
			self.df.reset_index(inplace=True, drop=True)

		if 'custom' in list(self.df.columns):
			self.df = self.df.drop(['custom'],axis=1)

		for pos,pos_group in self.df.groupby('position'):
			pos_group.to_csv(pos+os.sep.join(['output', 'tables', f'trajectories_{self.mode}.csv']), index=False)

		self.parent_window.parent_window.update_position_options()
		self.close()


	def help_propagate(self):

		"""
		Helper for segmentation strategy between threshold-based and Deep learning.
		"""

		dict_path = os.sep.join([get_software_location(),'celldetective','gui','help','propagate-classification.json'])

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

	def switch_to_log(self, i):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if i==1:
			try:
				feat_x = self.features_cb[1].currentText()
				min_x = self.df.dropna(subset=feat_x)[feat_x].min()
				max_x = self.df.dropna(subset=feat_x)[feat_x].max()
				x_padding = (max_x - min_x) * 0.05
				if x_padding==0:
					x_padding = 0.05

				if self.ax_props.get_xscale()=='linear':
					self.ax_props.set_xlim(min_x, max_x)
					self.ax_props.set_xscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_xscale('linear')
					self.ax_props.set_xlim(min_x - x_padding, max_x + x_padding)
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)
		elif i==0:
			try:
				feat_y = self.features_cb[0].currentText()
				min_y = self.df.dropna(subset=feat_y)[feat_y].min()
				max_y = self.df.dropna(subset=feat_y)[feat_y].max()
				y_padding = (max_y - min_y) * 0.05
				if y_padding==0:
					y_padding = 0.05

				if self.ax_props.get_yscale()=='linear':
					self.ax_props.set_ylim(min_y, max_y)
					self.ax_props.set_yscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_yscale('linear')
					self.ax_props.set_ylim(min_y - y_padding, max_y + y_padding)
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)

		self.ax_props.autoscale()
		self.propscanvas.canvas.draw_idle()

		print('Done.')







