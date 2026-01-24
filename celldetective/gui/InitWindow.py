import gc
import json
import os
import threading
import time
from glob import glob
from subprocess import Popen, check_output

from PyQt5.QtCore import QUrl, Qt, QThread
from PyQt5.QtGui import QDesktopServices, QIntValidator
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QProgressDialog,
)
from fonticon_mdi6 import MDI6
from psutil import cpu_count
from superqt.fonticon import icon
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveMainWindow,
    generic_message,
)
from celldetective.gui.base.utils import center_window, pretty_table
from celldetective.log_manager import get_logger

logger = get_logger("celldetective")


class BackgroundLoader(QThread):
    def run(self):
        logger.info("Loading background packages...")
        try:
            from celldetective.gui.control_panel import ControlPanel

            self.ControlPanel = ControlPanel
            from celldetective.gui.about import AboutWidget

            self.AboutWidget = AboutWidget
            from celldetective.processes.downloader import DownloadProcess

            self.DownloadProcess = DownloadProcess
            from celldetective.gui.configure_new_exp import ConfigNewExperiment

            self.ConfigNewExperiment = ConfigNewExperiment
            import pandas
            import matplotlib.pyplot
            import scipy.ndimage
            import tifffile
            import numpy
            import napari
            from celldetective.napari.utils import launch_napari_viewer
        except Exception:
            logger.error("Background packages not loaded...")
        logger.info("Background packages loaded...")


class AppInitWindow(CelldetectiveMainWindow):
    """
    Initial window to set the experiment folder or create a new one.
    """

    def __init__(self, parent_window=None, software_location=None):

        super().__init__()

        self.parent_window = parent_window
        self.setWindowTitle("celldetective")

        self.n_threads = min([1, cpu_count()])

        self.use_gpu = False
        self.gpu_thread = threading.Thread(target=self.check_gpu)
        self.gpu_thread.start()

        self.soft_path = software_location
        self.onlyInt = QIntValidator()

        self._create_actions()
        self._create_menu_bar()

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
        self.adjustSize()
        self.setFixedSize(self.size())
        self.show()

        self.bg_loader = BackgroundLoader()
        self.bg_loader.start()

    def closeEvent(self, event):

        QApplication.closeAllWindows()
        event.accept()
        gc.collect()

    def check_gpu(self):
        try:
            if os.name == "nt":
                kwargs = {"creationflags": 0x08000000}
            else:
                kwargs = {}
            check_output("nvidia-smi", **kwargs)
            logger.info(
                "NVIDIA GPU detected (activate or disable in Memory & Threads)..."
            )
            self.use_gpu = True
        except Exception as e:
            logger.info(f"No NVIDIA GPU detected: {e}...")
            self.use_gpu = False

    def create_locate_exp_hbox(self):

        self.locate_exp_layout = QHBoxLayout()
        self.locate_exp_layout.setContentsMargins(0, 5, 0, 0)
        self.experiment_path_selection = QLineEdit()
        self.experiment_path_selection.setAlignment(Qt.AlignLeft)
        self.experiment_path_selection.setEnabled(True)
        self.experiment_path_selection.setDragEnabled(True)
        self.experiment_path_selection.setFixedWidth(430)
        self.experiment_path_selection.textChanged[str].connect(
            self.check_path_and_enable_opening
        )
        try:
            self.folder_name = os.getcwd()
        except FileNotFoundError as _:
            self.folder_name = ""
        self.experiment_path_selection.setPlaceholderText("/path/to/experiment/folder/")
        self.locate_exp_layout.addWidget(self.experiment_path_selection, 90)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_experiment_folder)
        self.browse_button.setStyleSheet(self.button_style_sheet)
        self.browse_button.setIcon(icon(MDI6.folder, color="white"))
        self.locate_exp_layout.addWidget(self.browse_button, 10)
        self.vertical_layout.addLayout(self.locate_exp_layout)

    def _create_menu_bar(self):

        menu_bar = self.menuBar()
        menu_bar.clear()
        # Creating menus using a QMenu object

        file_menu = QMenu("File", self)
        file_menu.clear()
        file_menu.addAction(self.new_exp_action)
        file_menu.addAction(self.open_action)

        file_menu.addMenu(self.OpenRecentAction)
        self.OpenRecentAction.clear()
        if len(self.recent_file_acts) > 0:
            for i in range(len(self.recent_file_acts)):
                self.OpenRecentAction.addAction(self.recent_file_acts[i])

        file_menu.addMenu(self.openDemo)
        self.openDemo.addAction(self.open_spreading_assay_demo)
        self.openDemo.addAction(self.open_cytotoxicity_assay_demo)

        file_menu.addAction(self.open_models)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        menu_bar.addMenu(file_menu)

        options_menu = QMenu("Options", self)
        options_menu.addAction(self.memory_and_threads_action)
        menu_bar.addMenu(options_menu)

        plugins_menu = QMenu("Plugins", self)
        plugins_menu.addAction(self.correct_annotation_action)
        menu_bar.addMenu(plugins_menu)

        help_menu = QMenu("Help", self)
        help_menu.clear()
        help_menu.addAction(self.documentation_action)
        # help_menu.addAction(self.SoftwareAction)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)
        menu_bar.addMenu(help_menu)

        # editMenu = menuBar.addMenu("&Edit")
        # help_menu = menuBar.addMenu("&Help")

    def _create_actions(self):
        # Creating action using the first constructor
        # self.newAction = QAction(self)
        # self.newAction.setText("&New")
        # Creating actions using the second constructor
        self.open_action = QAction("Open Project", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setShortcutVisibleInContextMenu(True)

        self.openDemo = QMenu("Open Demo")
        self.open_spreading_assay_demo = QAction("Spreading Assay Demo", self)
        self.open_cytotoxicity_assay_demo = QAction("Cytotoxicity Assay Demo", self)

        self.memory_and_threads_action = QAction("Threads")

        self.correct_annotation_action = QAction("Correct a segmentation annotation")

        self.new_exp_action = QAction("New", self)
        self.new_exp_action.setShortcut("Ctrl+N")
        self.new_exp_action.setShortcutVisibleInContextMenu(True)
        self.exit_action = QAction("Exit", self)

        self.open_models = QAction("Open Models Location")
        self.open_models.setShortcut("Ctrl+L")
        self.open_models.setShortcutVisibleInContextMenu(True)

        self.OpenRecentAction = QMenu("Open Recent Project")
        self.reload_previous_experiments()

        self.documentation_action = QAction("Documentation", self)
        self.documentation_action.setShortcut("Ctrl+D")
        self.documentation_action.setShortcutVisibleInContextMenu(True)

        # self.SoftwareAction = QAction("Software", self) #1st arg icon(MDI6.information)
        self.about_action = QAction("About celldetective", self)

        # self.DocumentationAction.triggered.connect(self.load_previous_config)
        self.open_action.triggered.connect(self.open_experiment)
        self.new_exp_action.triggered.connect(self.create_new_experiment)
        self.exit_action.triggered.connect(self.close)
        self.open_models.triggered.connect(self.open_models_folder)
        self.about_action.triggered.connect(self.open_about_window)
        self.memory_and_threads_action.triggered.connect(self.set_memory_and_threads)
        self.correct_annotation_action.triggered.connect(self.correct_seg_annotation)
        self.documentation_action.triggered.connect(self.open_documentation)

        self.open_spreading_assay_demo.triggered.connect(
            self.download_spreading_assay_demo
        )
        self.open_cytotoxicity_assay_demo.triggered.connect(
            self.download_cytotoxicity_assay_demo
        )

    def download_spreading_assay_demo(self):
        if self.bg_loader.isFinished() and hasattr(self.bg_loader, "DownloadProcess"):
            DownloadProcess = self.bg_loader.DownloadProcess
        else:
            from celldetective.processes.downloader import DownloadProcess
        from celldetective.gui.workers import GenericProgressWindow

        self.target_dir = str(
            QFileDialog.getExistingDirectory(self, "Select Folder for Download")
        )
        if self.target_dir == "":
            return None

        if not os.path.exists(os.sep.join([self.target_dir, "demo_ricm"])):
            self.output_dir = self.target_dir
            self.file = "demo_ricm"
            process_args = {"output_dir": self.output_dir, "file": self.file}
            self.job = GenericProgressWindow(
                DownloadProcess,
                parent_window=self,
                title="Download",
                process_args=process_args,
                label_text="Downloading demo_ricm...",
            )
            result = self.job.exec_()
            if result == QDialog.Accepted:
                pass
            elif result == QDialog.Rejected:
                return None
            # download_zenodo_file('demo_ricm', self.target_dir)
        self.experiment_path_selection.setText(
            os.sep.join([self.target_dir, "demo_ricm"])
        )
        self.validate_button.click()

    def download_cytotoxicity_assay_demo(self):
        from celldetective.utils.downloaders import download_zenodo_file

        self.target_dir = str(
            QFileDialog.getExistingDirectory(self, "Select Folder for Download")
        )
        if self.target_dir == "":
            return None

        if not os.path.exists(os.sep.join([self.target_dir, "demo_adcc"])):
            download_zenodo_file("demo_adcc", self.target_dir)
        self.experiment_path_selection.setText(
            os.sep.join([self.target_dir, "demo_adcc"])
        )
        self.validate_button.click()

    def reload_previous_gpu_threads(self):

        self.recent_file_acts = []
        self.threads_config_path = os.sep.join(
            [self.soft_path, "celldetective", "threads.json"]
        )
        logger.info("Reading previous Memory & Threads settings...")
        if os.path.exists(self.threads_config_path):
            with open(self.threads_config_path, "r") as f:
                self.threads_config = json.load(f)
            if "use_gpu" in self.threads_config:
                self.use_gpu = bool(self.threads_config["use_gpu"])
                logger.info(f"Use GPU: {self.use_gpu}...")
            if "n_threads" in self.threads_config:
                self.n_threads = int(self.threads_config["n_threads"])
                logger.info(f"Number of threads: {self.n_threads}...")

    def reload_previous_experiments(self):

        self.recent_file_acts = []
        recent_path = os.sep.join([self.soft_path, "celldetective", "recent.txt"])
        if os.path.exists(recent_path):
            with open(recent_path, "r") as f:
                recent_exps = [r.strip() for r in f.readlines()]
            recent_exps.reverse()
            recent_exps = list(dict.fromkeys(recent_exps))[:10]

            # Auto-clean the file as well
            with open(recent_path, "w") as f:
                for r in reversed(recent_exps):
                    f.write(r + "\n")

            self.recent_file_acts = [QAction(r, self) for r in recent_exps]
            for r in self.recent_file_acts:
                r.triggered.connect(
                    lambda checked, item=r: self.load_recent_exp(item.text())
                )

    def correct_seg_annotation(self):

        from celldetective.napari.utils import correct_annotation

        self.filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "/home/", "TIF Files (*.tif)"
        )

        if self.filename != "":
            # 2. Show progress bar for opening the image in Napari
            progress_open = QProgressDialog(
                f"Opening {os.path.basename(self.filename)} in napari...",
                None,
                0,
                0,
                self,
            )
            progress_open.setWindowTitle("Please wait")
            progress_open.setWindowModality(Qt.WindowModal)
            progress_open.setMinimumDuration(0)
            progress_open.show()
            QApplication.processEvents()

            logger.info(f"Opening {self.filename} in napari...")
            try:
                correct_annotation(self.filename)
            finally:
                progress_open.close()
        else:
            return None

    def set_memory_and_threads(self):

        logger.info("Opening Memory & Threads...")

        self.threads_widget = CelldetectiveWidget()
        self.threads_widget.setWindowTitle("Threads")
        layout = QVBoxLayout()
        self.threads_widget.setLayout(layout)

        self.threads_le = QLineEdit(str(self.n_threads))
        self.threads_le.setValidator(self.onlyInt)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Parallel threads: "), 33)
        hbox.addWidget(self.threads_le, 66)
        layout.addLayout(hbox)

        self.use_gpu_checkbox = QCheckBox()
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel("Use GPU: "), 33)
        hbox2.addWidget(self.use_gpu_checkbox, 66)
        layout.addLayout(hbox2)
        if self.use_gpu:
            self.use_gpu_checkbox.setChecked(True)

        self.validate_thread_btn = QPushButton("Submit")
        self.validate_thread_btn.setStyleSheet(self.button_style_sheet)
        self.validate_thread_btn.clicked.connect(self.set_threads)
        layout.addWidget(self.validate_thread_btn)
        self.threads_widget.show()
        center_window(self.threads_widget)

    def set_threads(self):
        self.n_threads = int(self.threads_le.text())
        self.use_gpu = bool(self.use_gpu_checkbox.isChecked())
        dico = {"use_gpu": self.use_gpu, "n_threads": self.n_threads}
        with open(self.threads_config_path, "w") as f:
            json.dump(dico, f, indent=4)
        self.threads_widget.close()

    def open_experiment(self):

        self.browse_experiment_folder()
        if self.experiment_path_selection.text() != "":
            self.open_directory()

    def load_recent_exp(self, path):

        self.experiment_path_selection.setText(path)
        logger.info(f"Attempt to load experiment folder: {path}...")
        self.open_directory()

    def open_about_window(self):
        if self.bg_loader.isFinished() and hasattr(self.bg_loader, "AboutWidget"):
            AboutWidget = self.bg_loader.AboutWidget
        else:
            from celldetective.gui.about import AboutWidget

        self.about_wdw = AboutWidget()
        self.about_wdw.show()

    @staticmethod
    def open_documentation():
        doc_url = QUrl("https://celldetective.readthedocs.io/")
        QDesktopServices.openUrl(doc_url)

    def open_models_folder(self):

        path = os.sep.join([self.soft_path, "celldetective", "models", os.sep])
        try:
            Popen(f"explorer {os.path.realpath(path)}")
        except Exception as e:
            logger.warning(f"{e}")
            try:
                os.system('xdg-open "%s"' % path)
            except Exception as e:
                logger.error(f"Error {e}...")
                return None

    def create_buttons_hbox(self):

        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setContentsMargins(30, 15, 30, 5)
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
        if (os.path.exists(text)) and os.path.exists(os.sep.join([text, "config.ini"])):
            self.validate_button.setEnabled(True)
        else:
            self.validate_button.setEnabled(False)

    def set_experiment_path(self, path):
        self.experiment_path_selection.setText(path)

    def create_new_experiment(self):

        if self.bg_loader.isFinished() and hasattr(
            self.bg_loader, "ConfigNewExperiment"
        ):
            ConfigNewExperiment = self.bg_loader.ConfigNewExperiment
        else:
            from celldetective.gui.configure_new_exp import ConfigNewExperiment

        logger.info("Configuring new experiment...")
        self.new_exp_window = ConfigNewExperiment(self)
        self.new_exp_window.show()
        center_window(self.new_exp_window)

    def open_directory(self):
        self.t_ref = time.time()

        from celldetective.utils.experiment import extract_well_name_and_number

        self.exp_dir = self.experiment_path_selection.text().replace("/", os.sep)
        logger.info(f"Setting current directory to {self.exp_dir}...")

        wells = glob(os.sep.join([self.exp_dir, "W*"]))
        self.number_of_wells = len(wells)
        if self.number_of_wells == 0:
            generic_message(
                "No well was found in the experiment folder.\nPlease respect the W*/ nomenclature...",
                msg_type="critical",
            )
            return None
        else:
            if self.number_of_wells == 1:
                logger.info(f"Found {self.number_of_wells} well...")
            elif self.number_of_wells > 1:
                logger.info(f"Found {self.number_of_wells} wells...")

            def log_position_stats(wells_list):
                number_pos = {}
                for w in wells_list:
                    well_name, well_nbr = extract_well_name_and_number(w)
                    position_folders = glob(os.sep.join([w, f"{well_nbr}*", os.sep]))
                    number_pos.update({well_name: len(position_folders)})
                logger.info(f"Number of positions per well:")
                pretty_table(number_pos)

                recent_path = os.sep.join(
                    [self.soft_path, "celldetective", "recent.txt"]
                )
                recent_exps = []
                if os.path.exists(recent_path):
                    with open(recent_path, "r") as f:
                        recent_exps = [r.strip() for r in f.readlines()]

                recent_exps.append(self.exp_dir)
                # Deduplicate (keep latest)
                recent_exps = list(dict.fromkeys(reversed(recent_exps)))
                recent_exps.reverse()  # Back to original order (latest at end)
                recent_exps = recent_exps[-10:]  # Keep only last 10

                with open(recent_path, "w") as f:
                    for r in recent_exps:
                        f.write(r + "\n")

            threading.Thread(
                target=log_position_stats, args=(wells,), daemon=True
            ).start()

            if self.bg_loader.isFinished() and hasattr(self.bg_loader, "ControlPanel"):
                ControlPanel = self.bg_loader.ControlPanel
            else:
                from celldetective.gui.control_panel import ControlPanel

            try:
                self.control_panel = ControlPanel(self, self.exp_dir)
                self.control_panel.adjustSize()
                self.control_panel.setFixedSize(self.control_panel.size())
                self.control_panel.show()
                center_window(self.control_panel)
            except (AssertionError, FileNotFoundError) as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Experiment",
                    f"Could not load experiment configuration.\n\nError: {str(e)}\n\nPlease ensure 'config.ini' exists in the selected folder.",
                )
                return

            self.reload_previous_experiments()
            self._create_menu_bar()

    def browse_experiment_folder(self):
        """
        Locate an experiment folder. If no configuration file is in the experiment, display a warning.
        """

        self.folder_name = str(
            QFileDialog.getExistingDirectory(self, "Select directory")
        )
        if self.folder_name != "":
            self.experiment_path_selection.setText(self.folder_name)
        else:
            return None
        if not os.path.exists(os.sep.join([self.folder_name, "config.ini"])):
            generic_message(
                "No configuration can be found in the selected folder...",
                msg_type="warning",
            )
            self.experiment_path_selection.setText("")
            return None
