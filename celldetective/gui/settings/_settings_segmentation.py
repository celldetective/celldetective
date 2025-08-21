from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.gui.settings._settings_base import CelldetectiveSettingsPanel
import json
import os

class SettingsSegmentation(CelldetectiveSettingsPanel):
	
	def __init__(self, parent_window=None):
		
		super().__init__(title="Configure segmentation")
		self.parent_window = parent_window
		self.mode = self.parent_window.mode
		self.exp_dir = self.parent_window.exp_dir
		self._instructions_path = self.parent_window.exp_dir + f"configs/segmentation_instructions_{self.mode}.json"
		self._add_to_layout()
		self._load_previous_instructions()
		
	def _create_widgets(self):
		super()._create_widgets()
		
		self.flip_segmentation_checkbox: QCheckBox = QCheckBox("Segment frames in reverse order")
		self.flip_segmentation_checkbox.setIcon(icon(MDI6.camera_flip_outline,color="black"))
		self.flip_segmentation_checkbox.setIconSize(QSize(20, 20))
		self.flip_segmentation_checkbox.setStyleSheet(self.button_select_all)
		self.flip_segmentation_checkbox.setToolTip("Flip the order of the frames for segmentation.")
	
	def _add_to_layout(self):
		self._layout.addWidget(self.flip_segmentation_checkbox)
		self._layout.addWidget(self.submit_btn)
		#self._widget.adjustSize()

	def _load_previous_instructions(self):
		if os.path.exists(self._instructions_path):
			with open(self._instructions_path, "r") as f:
				instructions = json.load(f)
			if isinstance(instructions.get("flip"),bool):
				self.flip_segmentation_checkbox.setChecked(instructions.get("flip"))
	
	def _write_instructions(self):
		instructions = {"flip": self.flip_segmentation_checkbox.isChecked()}
		print('Segmentation instructions: ', instructions)
		file_name = self._instructions_path
		with open(file_name, 'w') as f:
			json.dump(instructions, f, indent=4)
		print('Done.')
		self.close()