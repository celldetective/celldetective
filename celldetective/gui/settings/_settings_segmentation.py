from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtCore import QSize
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils import get_software_location
from celldetective.gui.settings._settings_panel import CelldetectiveSettingsPanel

class SettingsSegmentation(CelldetectiveSettingsPanel):
	flip_segmentation_checkbox: QCheckBox = QCheckBox("Segment frames in reverse order")
	
	def __init__(self, parent_window=None):
		
		super().__init__()
		
		self.parent_window = parent_window
		self.setWindowTitle("Configure segmentation")
		self.mode = self.parent_window.mode
		self.exp_dir = self.parent_window.exp_dir
		self.segmentation_instructions_path = self.parent_window.exp_dir + f"configs/segmentation_instructions_{self.mode}.json"
		self.soft_path = get_software_location()
		self.expand_widgets()
	
	def expand_widgets(self):
		self.flip_segmentation_checkbox.setIcon(icon(MDI6.camera_flip_outline,color="black"))
		self.flip_segmentation_checkbox.setIconSize(QSize(20, 20))
		#self.flip_segmentation_checkbox.clicked.connect(self.flip_segmentation)
		self.flip_segmentation_checkbox.setStyleSheet(self.button_select_all)
		self.flip_segmentation_checkbox.setToolTip("Flip the order of the frames for segmentation.")
		self._layout.insertWidget(0, self.flip_segmentation_checkbox)
	
	def load_previous_instructions(self):
		print('override of the method adapted for segmentation! to do read instructions from existing json')
	
	def write_instructions(self):
		print('to do: the json writer')
	