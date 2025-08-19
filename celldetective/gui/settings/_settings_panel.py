from abc import abstractmethod
from PyQt5.QtWidgets import QApplication, QScrollArea, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt

from celldetective.gui.gui_utils import center_window
from celldetective.gui import CelldetectiveMainWindow, CelldetectiveWidget

class CelldetectiveSettingsPanel(CelldetectiveMainWindow):
	_screen_height: int
	_layout: QVBoxLayout = QVBoxLayout()
	_widget: CelldetectiveWidget = CelldetectiveWidget()
	submit_btn: QPushButton = QPushButton("Save")
	
	def __init__(self):
		super().__init__()
		self.get_screen_height()
		self.setMinimumWidth(500)
		self.setMaximumHeight(int(0.8 * self._screen_height))
		self.populate_widget()
		self.load_previous_instructions()
		self.center_window()
	
	def center_window(self):
		return center_window(self)
	
	def get_screen_height(self):
		app = QApplication.instance()
		screen = app.primaryScreen()
		geometry = screen.availableGeometry()
		_, self._screen_height = geometry.getRect()[-2:]
	
	def populate_widget(self):
		# Create button widget and layout
		self._scroll_area = QScrollArea(self)
		self._widget.setLayout(self._layout)
		self._layout.setContentsMargins(30, 30, 30, 30)
		
		self.submit_btn.setStyleSheet(self.button_style_sheet)
		self.submit_btn.clicked.connect(self.write_instructions)
		self._layout.addWidget(self.submit_btn)
		
		self._widget.adjustSize()
		
		self._scroll_area.setAlignment(Qt.AlignCenter)
		self._scroll_area.setWidget(self._widget)
		self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
		self._scroll_area.setWidgetResizable(True)
		self.setCentralWidget(self._scroll_area)
		
		QApplication.processEvents()
	
	@abstractmethod
	def load_previous_instructions(self): pass
	
	@abstractmethod
	def write_instructions(self): pass