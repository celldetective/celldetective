from PyQt5.QtWidgets import QWidget, QLineEdit, QMessageBox, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QComboBox
from celldetective.gui.gui_utils import FigureCanvas, center_window, color_from_class
import numpy as np
import matplotlib.pyplot as plt
from superqt import QLabeledSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from PyQt5.QtCore import Qt, QSize
import os

class ClassifierWidget(QWidget):

	def __init__(self, parent):

		super().__init__()

		self.parent = parent
		self.screen_height = self.parent.parent.parent.screen_height
		self.screen_width = self.parent.parent.parent.screen_width
		self.setWindowTitle("Custom classification")
		self.parent.load_available_tables()
		self.mode = self.parent.mode
		self.df = self.parent.df

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
		self.project_times_btn.setStyleSheet(self.parent.parent.parent.button_select_all)
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


		self.features_cb = [QComboBox() for i in range(2)]
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
			self.log_btns[i].setStyleSheet(self.parent.parent.parent.button_select_all)
			self.log_btns[i].clicked.connect(lambda ch, i=i: self.switch_to_log(i))

		hbox_classify = QHBoxLayout()
		hbox_classify.addWidget(QLabel('classify: '), 10)
		self.property_query_le = QLineEdit()
		self.property_query_le.setPlaceholderText('classify points using a query such as: area > 100 or eccentricity > 0.95')
		hbox_classify.addWidget(self.property_query_le, 70)
		self.submit_query_btn = QPushButton('Submit...')
		self.submit_query_btn.clicked.connect(self.apply_property_query)
		hbox_classify.addWidget(self.submit_query_btn, 20)
		layout.addLayout(hbox_classify)

		self.submit_btn = QPushButton('apply')
		self.submit_btn.clicked.connect(self.submit_classification)
		layout.addWidget(self.submit_btn, 30)

		self.frame_slider.valueChanged.connect(self.set_frame)

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
		self.scat_props = self.ax_props.scatter([],[], color='k', alpha=0.75)
		self.propscanvas.canvas.draw_idle()
		self.propscanvas.canvas.setMinimumHeight(self.screen_height//5)

	def update_props_scatter(self):

		if not self.project_times:
			self.scat_props.set_offsets(self.df.loc[self.df['FRAME']==self.currentFrame,[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
			self.scat_props.set_facecolor([color_from_class(c) for c in self.df.loc[self.df['FRAME']==self.currentFrame,self.class_name].to_numpy()])
			self.ax_props.set_xlabel(self.features_cb[1].currentText())
			self.ax_props.set_ylabel(self.features_cb[0].currentText())
		else:

			self.scat_props.set_offsets(self.df[[self.features_cb[1].currentText(),self.features_cb[0].currentText()]].to_numpy())
			self.scat_props.set_facecolor([color_from_class(c) for c in self.df[self.class_name].to_numpy()])
			self.ax_props.set_xlabel(self.features_cb[1].currentText())
			self.ax_props.set_ylabel(self.features_cb[0].currentText())

		self.ax_props.set_xlim(1*self.df[self.features_cb[1].currentText()].min(),1.0*self.df[self.features_cb[1].currentText()].max())
		self.ax_props.set_ylim(1*self.df[self.features_cb[0].currentText()].min(),1.0*self.df[self.features_cb[0].currentText()].max())
		self.propscanvas.canvas.draw_idle()

	def apply_property_query(self):
		query = self.property_query_le.text()
		self.df[self.class_name] = 1

		print(query, self.class_name)

		if query=='':
			print('empty query')
		else:
			try:
				self.selection = self.df.query(query).index
				print(self.selection)
				self.df.loc[self.selection,self.class_name] = 0
			except Exception as e:
				print(e)
				print(self.df.columns)
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Warning)
				msgBox.setText(f"The query could not be understood. No filtering was applied. {e}")
				msgBox.setWindowTitle("Warning")
				msgBox.setStandardButtons(QMessageBox.Ok)
				returnValue = msgBox.exec()
				if returnValue == QMessageBox.Ok:
					return None

		self.update_props_scatter()

	def set_frame(self, value):
		self.currentFrame = value
		self.update_props_scatter()

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
		self.update_props_scatter()

	def submit_classification(self):
		print('submit')

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
		self.df = self.df.drop(list(set(name_map.values()) & set(self.df.columns)), axis=1).rename(columns=name_map)
		if 'TRACK_ID' in list(self.df.columns):
			print('Tracks detected... save a status column...')
			stat_col = self.class_name_user.replace('class','status')
			self.df.loc[:,stat_col] = 1 - self.df[self.class_name_user].values
			for tid,track in self.df.groupby('TRACK_ID'):
				indices = track[self.class_name_user].index
				status_values = track[stat_col].to_numpy()
				if np.all([s==0 for s in status_values]):
					self.df.loc[indices, self.class_name_user] = 1
				else:
					self.df.loc[indices, self.class_name_user] = 2

		for pos,pos_group in self.df.groupby('position'):
			pos_group.to_csv(pos+os.sep.join(['output', 'tables', f'trajectories_{self.mode}.csv']), index=False)

		# reset
		#self.init_class()
		#self.update_props_scatter()
		self.close()

	def switch_to_log(self, i):

		"""
		Switch threshold histogram to log scale. Auto adjust.
		"""

		if i==1:
			try:
				if self.ax_props.get_xscale()=='linear':
					self.ax_props.set_xscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_xscale('linear')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)
		elif i==0:
			try:
				if self.ax_props.get_yscale()=='linear':
					self.ax_props.set_yscale('log')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="#1565c0"))
				else:
					self.ax_props.set_yscale('linear')
					self.log_btns[i].setIcon(icon(MDI6.math_log,color="black"))
			except Exception as e:
				print(e)

		self.ax_props.autoscale()
		self.propscanvas.canvas.draw_idle()
