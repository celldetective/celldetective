from multiprocessing import Queue
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QSize, Qt

from celldetective.gui.base_components import CelldetectiveDialog
from celldetective.gui.gui_utils import center_window
import math
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


class ProgressWindow(CelldetectiveDialog):

    def __init__(
        self,
        process=None,
        parent_window=None,
        title="",
        position_info=True,
        process_args=None,
    ):

        super().__init__()
        # QDialog.__init__(self)

        self.setWindowTitle(f"{title}")
        self.__process = process
        self.parent_window = parent_window

        self.position_info = position_info
        if self.position_info:
            self.pos_name = getattr(self.parent_window, "pos_name", "Batch")

        # self.__btn_run = QPushButton("Start")
        self.__btn_stp = QPushButton("Cancel")
        if self.position_info:
            self.position_label = QLabel(f"Processing position {self.pos_name}...")
        self.__label = QLabel("Idle")
        self.time_left_lbl = QLabel("")

        self.well_time_lbl = QLabel("Well progress:")
        self.well_progress_bar = QProgressBar()
        self.well_progress_bar.setValue(0)
        self.well_progress_bar.setFormat("Total (Wells): %p%")

        self.pos_time_lbl = QLabel("Position progress:")
        self.pos_progress_bar = QProgressBar()
        self.pos_progress_bar.setValue(0)
        self.pos_progress_bar.setFormat("Current Well (Positions): %p%")

        self.frame_time_lbl = QLabel("Frame progress:")
        self.frame_progress_bar = QProgressBar()
        self.frame_progress_bar.setValue(0)
        self.frame_progress_bar.setFormat("Current Position (Frames): %p%")

        self.__runner = Runner(
            process=self.__process,
            process_args=process_args,
        )
        logger.info("Runner initialized...")
        self.pool = QThreadPool.globalInstance()

        self.__btn_stp.clicked.connect(self.__stp_net)
        self.__runner.signals.finished.connect(self.__on_finished)
        self.__runner.signals.finished.connect(self.__on_error)

        self.__runner.signals.update_well.connect(self.well_progress_bar.setValue)
        self.__runner.signals.update_well_time.connect(self.well_time_lbl.setText)

        self.__runner.signals.update_pos.connect(self.pos_progress_bar.setValue)
        self.__runner.signals.update_pos_time.connect(self.pos_time_lbl.setText)

        self.__runner.signals.update_frame.connect(self.frame_progress_bar.setValue)
        self.__runner.signals.update_frame_time.connect(self.frame_time_lbl.setText)
        self.__runner.signals.update_status.connect(self.__label.setText)

        self.__btn_stp.setDisabled(True)

        self.layout = QVBoxLayout()
        if self.position_info:
            self.layout.addWidget(self.position_label)

        self.layout.addWidget(self.well_time_lbl)
        self.layout.addWidget(self.well_progress_bar)

        self.layout.addWidget(self.pos_time_lbl)
        self.layout.addWidget(self.pos_progress_bar)

        self.layout.addWidget(self.frame_time_lbl)
        self.layout.addWidget(self.frame_progress_bar)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.__btn_stp)
        self.btn_layout.addWidget(self.__label)
        self.layout.addLayout(self.btn_layout)

        self.setLayout(self.layout)
        self.setFixedSize(QSize(400, 250))
        self.__run_net()
        self.setModal(True)
        center_window(self)

    def closeEvent(self, evnt):
        evnt.ignore()
        self.setWindowState(Qt.WindowMinimized)

    def __run_net(self):
        # self.__btn_run.setDisabled(True)
        self.__btn_stp.setEnabled(True)
        self.__label.setText("Running...")
        self.pool.start(self.__runner)

    def __stp_net(self):
        self.__runner.close()
        logger.info("\n Job cancelled... Abort.")
        self.reject()

    def __on_finished(self):
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nFinished!")
        self.__runner.close()
        self.accept()

    def __on_error(self):
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nError")
        self.__runner.close()
        self.accept()


class Runner(QRunnable):

    def __init__(
        self,
        process=None,
        process_args=None,
    ):
        QRunnable.__init__(self)

        logger.info(f"{process_args=}")
        self.__queue = Queue()
        self.__process = process(self.__queue, process_args=process_args)
        self.signals = RunnerSignal()

    def run(self):
        logger.info("Starting Process (runner-side)...")
        self.__process.start()
        while True:
            try:
                data = self.__queue.get()

                # Handle dictionary for triple progress
                if isinstance(data, dict):
                    if "well_progress" in data:
                        self.signals.update_well.emit(int(data["well_progress"]))
                    if "well_time" in data:
                        self.signals.update_well_time.emit(data["well_time"])

                    if "pos_progress" in data:
                        self.signals.update_pos.emit(int(data["pos_progress"]))
                    if "pos_time" in data:
                        self.signals.update_pos_time.emit(data["pos_time"])

                    if "frame_progress" in data:
                        self.signals.update_frame.emit(int(data["frame_progress"]))
                    if "frame_time" in data:
                        self.signals.update_frame_time.emit(data["frame_time"])

                    if "status" in data:
                        if data["status"] == "finished":
                            self.signals.finished.emit()
                            break
                        elif data["status"] == "error":
                            self.signals.error.emit()
                        else:
                            self.signals.update_status.emit(data["status"])

                # Simple fallback for legacy list [progress, time] -> map to POS progress
                elif isinstance(data, list) and len(data) == 2:
                    progress, time = data
                    self.signals.update_pos.emit(math.ceil(progress))

                elif data == "finished":
                    self.signals.finished.emit()
                    break
                elif data == "error":
                    self.signals.error.emit()

            except Exception as e:
                logger.error(e)
                pass

    def close(self):
        self.__process.end_process()


class RunnerSignal(QObject):

    update_well = pyqtSignal(int)
    update_well_time = pyqtSignal(str)

    update_pos = pyqtSignal(int)
    update_pos_time = pyqtSignal(str)

    update_frame = pyqtSignal(int)
    update_frame_time = pyqtSignal(str)
    update_status = pyqtSignal(str)

    finished = pyqtSignal()
    error = pyqtSignal()
