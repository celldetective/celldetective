import gc
import json
import os
from glob import glob
from subprocess import Popen, check_output

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QIntValidator
from PyQt5.QtWidgets import QAction, QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit, \
	QMenu, QPushButton, QVBoxLayout
from fonticon_mdi6 import MDI6
from psutil import cpu_count
from superqt.fonticon import icon

from celldetective.gui import ConfigNewExperiment, ControlPanel, CelldetectiveMainWindow, CelldetectiveWidget
from celldetective.gui.about import AboutWidget
from celldetective.gui.gui_utils import center_window, generic_message
from celldetective.gui.processes.downloader import DownloadProcess
from celldetective.gui.workers import ProgressWindow
from celldetective.io import correct_annotation, extract_well_name_and_number
from celldetective.utils import download_zenodo_file, pretty_table


class AppInitWindow(CelldetectiveMainWindow):

	"""
	Initial window to set the experiment folder or create a new one.
	"""

	def __init__(self, parent_window=None, software_location=None):

		super().__init__()

		self.parent_window = parent_window
		self.setWindowTitle("celldetective")

		self.n_threads = min([1, cpu_count()])

		try:
			check_output('nvidia-smi')
			print('NVIDIA GPU detected (activate or disable in Memory & Threads)...')
			self.use_gpu = True
		except Exception: # this command not being found can raise quite a few different errors depending on the configuration
			print('No NVIDIA GPU detected...')
			self.use_gpu = False

		self.soft_path = software_location
		self.onlyInt = QIntValidator()

		self._createActions()
		self._createMenuBar()

		app = QApplication.instance()
		self.screen = app.primaryScreen()
		self.geometry = self.screen.availableGeometry()
		self.screen_width, self.screen_height = self.geometry.getRect()[-2:]

		central_widget = CelldetectiveWidget()
		self.vertical_layout = QVBoxLayout(central_widget)
		self.vertical_layout.setContentsMargins(15, 15, 15, 15)
		self.vertical_layout.addWidget(QLabel("Experiment folder:"))
		self.create_locate_exp_hbox()
		self.create_buttons_hbox()
		self.setCentralWidget(central_widget)
		self.reload_previous_gpu_threads()
		center_window(self)

		self.show()
	
	def closeEvent(self, event):

		QApplication.closeAllWindows()
		event.accept()
		gc.collect()

	def create_locate_exp_hbox(self):

		self.locate_exp_layout = QHBoxLayout()
		self.locate_exp_layout.setContentsMargins(0, 5, 0, 0)
		self.experiment_path_selection = QLineEdit()
		self.experiment_path_selection.setAlignment(Qt.AlignLeft)	
		self.experiment_path_selection.setEnabled(True)
		self.experiment_path_selection.setDragEnabled(True)
		self.experiment_path_selection.setFixedWidth(430)
		self.experiment_path_selection.textChanged[str].connect(self.check_path_and_enable_opening)
		try:
			self.foldername = os.getcwd()
		except FileNotFoundError as e:
			self.foldername = ""
		self.experiment_path_selection.setPlaceholderText('/path/to/experiment/folder/')
		self.locate_exp_layout.addWidget(self.experiment_path_selection, 90)

		self.browse_button = QPushButton("Browse...")
		self.browse_button.clicked.connect(self.browse_experiment_folder)
		self.browse_button.setStyleSheet(self.button_style_sheet)
		self.browse_button.setIcon(icon(MDI6.folder, color="white"))
		self.locate_exp_layout.addWidget(self.browse_button, 10)
		self.vertical_layout.addLayout(self.locate_exp_layout)


	def _createMenuBar(self):

		menuBar = self.menuBar()
		menuBar.clear()
		# Creating menus using a QMenu object

		fileMenu = QMenu("File", self)
		fileMenu.clear()
		fileMenu.addAction(self.newExpAction)
		fileMenu.addAction(self.openAction)

		fileMenu.addMenu(self.OpenRecentAction)
		self.OpenRecentAction.clear()
		if len(self.recentFileActs)>0:
			for i in range(len(self.recentFileActs)):
				self.OpenRecentAction.addAction(self.recentFileActs[i])

		fileMenu.addMenu(self.openDemo)
		self.openDemo.addAction(self.openSpreadingAssayDemo)
		self.openDemo.addAction(self.openCytotoxicityAssayDemo)

		fileMenu.addAction(self.openModels)
		fileMenu.addSeparator()
		fileMenu.addAction(self.exitAction)
		menuBar.addMenu(fileMenu)

		OptionsMenu = QMenu("Options", self)
		OptionsMenu.addAction(self.MemoryAndThreadsAction)
		menuBar.addMenu(OptionsMenu)

		PluginsMenu = QMenu("Plugins", self)
		PluginsMenu.addAction(self.CorrectAnnotationAction)
		menuBar.addMenu(PluginsMenu)

		helpMenu = QMenu("Help", self)
		helpMenu.clear()
		helpMenu.addAction(self.DocumentationAction)
		#helpMenu.addAction(self.SoftwareAction)
		helpMenu.addSeparator()
		helpMenu.addAction(self.AboutAction)
		menuBar.addMenu(helpMenu)

		#editMenu = menuBar.addMenu("&Edit")
		#helpMenu = menuBar.addMenu("&Help")

	def _createActions(self):
		# Creating action using the first constructor
		#self.newAction = QAction(self)
		#self.newAction.setText("&New")
		# Creating actions using the second constructor
		self.openAction = QAction('Open Project', self)
		self.openAction.setShortcut("Ctrl+O")
		self.openAction.setShortcutVisibleInContextMenu(True)

		self.openDemo = QMenu('Open Demo')
		self.openSpreadingAssayDemo = QAction('Spreading Assay Demo', self)
		self.openCytotoxicityAssayDemo = QAction('Cytotoxicity Assay Demo', self)

		self.MemoryAndThreadsAction = QAction('Threads')

		self.CorrectAnnotationAction = QAction('Correct a segmentation annotation')

		self.newExpAction = QAction('New', self)
		self.newExpAction.setShortcut("Ctrl+N")
		self.newExpAction.setShortcutVisibleInContextMenu(True)
		self.exitAction = QAction('Exit', self)

		self.openModels = QAction('Open Models Location')
		self.openModels.setShortcut("Ctrl+L")
		self.openModels.setShortcutVisibleInContextMenu(True)

		self.OpenRecentAction = QMenu('Open Recent Project')
		self.reload_previous_experiments()

		self.DocumentationAction = QAction("Documentation", self)
		self.DocumentationAction.setShortcut("Ctrl+D")
		self.DocumentationAction.setShortcutVisibleInContextMenu(True)

		#self.SoftwareAction = QAction("Software", self) #1st arg icon(MDI6.information)
		self.AboutAction = QAction("About celldetective", self)

		#self.DocumentationAction.triggered.connect(self.load_previous_config)
		self.openAction.triggered.connect(self.open_experiment)
		self.newExpAction.triggered.connect(self.create_new_experiment)
		self.exitAction.triggered.connect(self.close)
		self.openModels.triggered.connect(self.open_models_folder)
		self.AboutAction.triggered.connect(self.open_about_window)
		self.MemoryAndThreadsAction.triggered.connect(self.set_memory_and_threads)
		self.CorrectAnnotationAction.triggered.connect(self.correct_seg_annotation)
		self.DocumentationAction.triggered.connect(self.open_documentation)

		self.openSpreadingAssayDemo.triggered.connect(self.download_spreading_assay_demo)
		self.openCytotoxicityAssayDemo.triggered.connect(self.download_cytotoxicity_assay_demo)

	def download_spreading_assay_demo(self):

		self.target_dir = str(QFileDialog.getExistingDirectory(self, 'Select Folder for Download'))
		if self.target_dir=='':
			return None

		if not os.path.exists(os.sep.join([self.target_dir,'demo_ricm'])):
			self.output_dir = self.target_dir
			self.file = 'demo_ricm'
			process_args = {"output_dir": self.output_dir, "file": self.file}
			self.job = ProgressWindow(DownloadProcess, parent_window=self, title="Download", position_info=False, process_args=process_args)
			result = self.job.exec_()
			if result == QDialog.Accepted:
				pass
			elif result == QDialog.Rejected:
				return None
			#download_zenodo_file('demo_ricm', self.target_dir)
		self.experiment_path_selection.setText(os.sep.join([self.target_dir, 'demo_ricm']))
		self.validate_button.click()

	def download_cytotoxicity_assay_demo(self):

		self.target_dir = str(QFileDialog.getExistingDirectory(self, 'Select Folder for Download'))
		if self.target_dir=='':
			return None

		if not os.path.exists(os.sep.join([self.target_dir,'demo_adcc'])):
			download_zenodo_file('demo_adcc', self.target_dir)
		self.experiment_path_selection.setText(os.sep.join([self.target_dir, 'demo_adcc']))
		self.validate_button.click()

	def reload_previous_gpu_threads(self):

		self.recentFileActs = []
		self.threads_config_path = os.sep.join([self.soft_path,'celldetective','threads.json'])
		print('Reading previous Memory & Threads settings...')
		if os.path.exists(self.threads_config_path):
			with open(self.threads_config_path, 'r') as f:
				self.threads_config = json.load(f)
			if 'use_gpu' in self.threads_config:
				self.use_gpu = bool(self.threads_config['use_gpu'])
				print(f'Use GPU: {self.use_gpu}...')
			if 'n_threads' in self.threads_config:
				self.n_threads = int(self.threads_config['n_threads'])
				print(f'Number of threads: {self.n_threads}...')


	def reload_previous_experiments(self):

		recentExps = []
		self.recentFileActs = []
		if os.path.exists(os.sep.join([self.soft_path,'celldetective','recent.txt'])):
			recentExps = open(os.sep.join([self.soft_path,'celldetective','recent.txt']), 'r')
			recentExps = recentExps.readlines()
			recentExps = [r.strip() for r in recentExps]
			recentExps.reverse()
			recentExps = list(dict.fromkeys(recentExps))
			self.recentFileActs = [QAction(r,self) for r in recentExps]
			for r in self.recentFileActs:
				r.triggered.connect(lambda checked, item=r: self.load_recent_exp(item.text()))

	def correct_seg_annotation(self):
		
		self.filename,_ = QFileDialog.getOpenFileName(self,"Open Image", "/home/", "TIF Files (*.tif)")
		if self.filename!='':
			print('Opening ',self.filename,' in napari...')
			correct_annotation(self.filename)
		else:
			return None

	def set_memory_and_threads(self):
		
		print('setting memory and threads')

		self.ThreadsWidget = CelldetectiveWidget()
		self.ThreadsWidget.setWindowTitle("Threads")
		layout = QVBoxLayout()
		self.ThreadsWidget.setLayout(layout)

		self.threads_le = QLineEdit(str(self.n_threads))
		self.threads_le.setValidator(self.onlyInt)

		hbox = QHBoxLayout()
		hbox.addWidget(QLabel('Parallel threads: '), 33)
		hbox.addWidget(self.threads_le, 66)
		layout.addLayout(hbox)

		self.use_gpu_checkbox = QCheckBox()
		hbox2 = QHBoxLayout()
		hbox2.addWidget(QLabel('Use GPU: '), 33)
		hbox2.addWidget(self.use_gpu_checkbox, 66)
		layout.addLayout(hbox2)
		if self.use_gpu:
			self.use_gpu_checkbox.setChecked(True)

		self.validateThreadBtn = QPushButton('Submit')
		self.validateThreadBtn.setStyleSheet(self.button_style_sheet)
		self.validateThreadBtn.clicked.connect(self.set_threads)
		layout.addWidget(self.validateThreadBtn)
		center_window(self.ThreadsWidget)
		self.ThreadsWidget.show()

	def set_threads(self):
		self.n_threads = int(self.threads_le.text())
		self.use_gpu = bool(self.use_gpu_checkbox.isChecked())
		dico = {"use_gpu": self.use_gpu, "n_threads": self.n_threads}
		with open(self.threads_config_path, 'w') as f:
			json.dump(dico, f, indent=4)
		self.ThreadsWidget.close()


	def open_experiment(self):

		self.browse_experiment_folder()
		if self.experiment_path_selection.text()!='':
			self.open_directory()

	def load_recent_exp(self, path):
		
		self.experiment_path_selection.setText(path)
		print(f'Attempt to load experiment folder: {path}...')
		self.open_directory()

	def open_about_window(self):
		self.about_wdw = AboutWidget()
		self.about_wdw.show()

	def open_documentation(self):
		doc_url = QUrl('https://celldetective.readthedocs.io/')
		QDesktopServices.openUrl(doc_url)

	def open_models_folder(self):

		path = os.sep.join([self.soft_path,'celldetective','models',os.sep])
		try:
			Popen(f'explorer {os.path.realpath(path)}')
		except:

			try:
				os.system('xdg-open "%s"' % path)
			except:
				return None

	def create_buttons_hbox(self):

		self.buttons_layout = QHBoxLayout()
		self.buttons_layout.setContentsMargins(30,15,30,5)
		self.new_exp_button = QPushButton("New")
		self.new_exp_button.clicked.connect(self.create_new_experiment)
		self.new_exp_button.setStyleSheet(self.button_style_sheet_2)
		self.buttons_layout.addWidget(self.new_exp_button, 50)

		self.validate_button = QPushButton("Open")
		self.validate_button.clicked.connect(self.open_directory)
		self.validate_button.setStyleSheet(self.button_style_sheet)
		self.validate_button.setEnabled(False)
		self.validate_button.setShortcut("Return")
		self.buttons_layout.addWidget(self.validate_button, 50)
		self.vertical_layout.addLayout(self.buttons_layout)

	def check_path_and_enable_opening(self):
		
		"""
		Enable 'Open' button if the text is a valid path.
		"""

		text = self.experiment_path_selection.text()
		if (os.path.exists(text)) and os.path.exists(os.sep.join([text,"config.ini"])):
			self.validate_button.setEnabled(True)
		else:
			self.validate_button.setEnabled(False)


	def set_experiment_path(self, path):
		self.experiment_path_selection.setText(path)

	def create_new_experiment(self):

		print("Configuring new experiment...")
		self.new_exp_window = ConfigNewExperiment(self)
		self.new_exp_window.show()

	def open_directory(self):

		self.exp_dir = self.experiment_path_selection.text().replace('/', os.sep)
		print(f"Setting current directory to {self.exp_dir}...")

		wells = glob(os.sep.join([self.exp_dir,"W*"]))
		self.number_of_wells = len(wells)
		if self.number_of_wells==0:
			generic_message("No well was found in the experiment folder.\nPlease respect the W*/ nomenclature...", msg_type="critical")
			return None
		else:
			if self.number_of_wells==1:
				print(f"Found {self.number_of_wells} well...")
			elif self.number_of_wells>1:
				print(f"Found {self.number_of_wells} wells...")

			number_pos = {}
			for w in wells:
				well_name, well_nbr = extract_well_name_and_number(w)
				position_folders = glob(os.sep.join([w,f"{well_nbr}*", os.sep]))
				number_pos.update({well_name: len(position_folders)})
			print(f"Number of positions per well:")
			pretty_table(number_pos)
			
			with open(os.sep.join([self.soft_path,'celldetective','recent.txt']), 'a+') as f:
				f.write(self.exp_dir+'\n')

			self.control_panel = ControlPanel(self, self.exp_dir)
			self.control_panel.show()

			self.reload_previous_experiments()
			self._createMenuBar()


	def browse_experiment_folder(self):

		"""
		Locate an experiment folder. If no configuration file is in the experiment, display a warning.
		"""

		self.foldername = str(QFileDialog.getExistingDirectory(self, 'Select directory'))
		if self.foldername!='':
			self.experiment_path_selection.setText(self.foldername)
		else:
			return None
		if not os.path.exists(os.sep.join([self.foldername,"config.ini"])):
			generic_message("No configuration can be found in the selected folder...", msg_type="warning")
			self.experiment_path_selection.setText('')
			return None
