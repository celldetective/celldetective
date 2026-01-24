from multiprocessing import Queue
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QSize, Qt
from PyQt5.QtGui import QPixmap, QImage
import math
import numpy as np

from celldetective.gui.base.components import CelldetectiveDialog
from celldetective.gui.base.utils import center_window
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
        well_label="Well progress:",
        pos_label="Position progress:",
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

        self.well_time_lbl = QLabel(well_label)
        self.well_progress_bar = QProgressBar()
        self.well_progress_bar.setValue(0)
        self.well_progress_bar.setFormat("Total (Wells): %p%")

        self.pos_time_lbl = QLabel(pos_label)
        self.pos_progress_bar = QProgressBar()
        self.pos_progress_bar.setValue(0)
        self.pos_progress_bar.setFormat("Current Well (Positions): %p%")

        if "show_frame_progress" in process_args:
            self.show_frame_progress = process_args["show_frame_progress"]
        else:
            self.show_frame_progress = True

        if self.show_frame_progress:
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
        self.__runner.signals.error.connect(self.__on_error)

        self.__runner.signals.update_well.connect(self.well_progress_bar.setValue)
        self.__runner.signals.update_well_time.connect(self.well_time_lbl.setText)

        self.__runner.signals.update_pos.connect(self.pos_progress_bar.setValue)
        self.__runner.signals.update_pos_time.connect(self.pos_time_lbl.setText)

        if self.show_frame_progress:
            self.__runner.signals.update_frame.connect(self.frame_progress_bar.setValue)
            self.__runner.signals.update_frame_time.connect(self.frame_time_lbl.setText)

        self.__runner.signals.update_status.connect(self.__label.setText)
        self.__runner.signals.update_image.connect(self.update_image)

        self.image_label = QLabel()
        self.image_label.setFixedSize(250, 250)
        self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label.setScaledContents(True)
        self.image_label.hide()

        self.__btn_stp.setDisabled(True)

        self.progress_layout = QVBoxLayout()
        if self.position_info:
            self.progress_layout.addWidget(self.position_label)

        self.progress_layout.addWidget(self.well_time_lbl)
        self.progress_layout.addWidget(self.well_progress_bar)

        self.progress_layout.addWidget(self.pos_time_lbl)
        self.progress_layout.addWidget(self.pos_progress_bar)

        if self.show_frame_progress:
            self.progress_layout.addWidget(self.frame_time_lbl)
            self.progress_layout.addWidget(self.frame_progress_bar)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.__btn_stp)
        self.btn_layout.addWidget(self.__label)

        # Left Column Layout (Bars + Buttons)
        self.left_layout = QVBoxLayout()
        self.left_layout.addLayout(self.progress_layout)
        self.left_layout.addLayout(self.btn_layout)

        # Main Root Layout (Left Column + Image)
        self.root_layout = QHBoxLayout()
        self.root_layout.addLayout(self.left_layout)
        self.root_layout.addWidget(self.image_label)

        self.setLayout(self.root_layout)
        self.setFixedSize(QSize(400, 220))
        self.show()
        self.raise_()
        self.activateWindow()
        logger.info("ProgressWindow initialized and shown.")
        self.__run_net()
        self.setModal(True)
        # center_window(self)

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

    def __on_error(self, message="Error"):
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nError")
        self.__runner.close()

        # Show error in a message box to ensure it's seen
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Process failed")
        msg.setInformativeText(str(message))
        msg.setWindowTitle("Error")
        msg.exec_()

        self.reject()

    def update_image(self, img_data):
        try:
            if img_data is None:
                return

            if self.image_label.isHidden():
                self.image_label.show()
                # Expand window width and height to accommodate image
                self.setFixedSize(QSize(750, 320))
            # Normalize for display
            img = img_data.astype(float)
            img = np.nan_to_num(img)

            min_val = np.min(img)
            max_val = np.max(img)

            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val) * 255
            else:
                img = np.zeros_like(img)

            img = img.astype(np.uint8)
            img = np.require(
                img, np.uint8, "C"
            )  # Maintain strict C-contiguity for Qt stability

            height, width = img.shape[:2]

            # Grayscale or RGB
            if img.ndim == 3:
                # RGB
                bytes_per_line = 3 * width
                q_img = QImage(
                    img.data, width, height, bytes_per_line, QImage.Format_RGB888
                )
            else:
                # Grayscale
                bytes_per_line = width
                q_img = QImage(
                    img.data, width, height, bytes_per_line, QImage.Format_Grayscale8
                )

            # Use .copy() ensures deep copy of data into QPixmap so we don't depend on volatile memory
            pixmap = QPixmap.fromImage(q_img.copy())
            # Scale with Aspect Ratio preserved to avoid cutting or distortion
            scaled_pixmap = pixmap.scaled(
                self.image_label.size()
                - QSize(20, 20),  # Add 10px padding on all sides
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Image update failed: {e}")


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

                    if "image_preview" in data:
                        self.signals.update_image.emit(data["image_preview"])
                    elif "bg_image" in data:  # Backward compatibility
                        self.signals.update_image.emit(data["bg_image"])

                    if "plot_data" in data:
                        self.signals.update_plot.emit(data["plot_data"])

                    if "training_result" in data:
                        self.signals.training_result.emit(data["training_result"])

                    if "result" in data:
                        self.signals.result.emit(data["result"])

                    if "status" in data:  # Moved this block out of frame_time check
                        logger.info(
                            f"Runner received status: {data['status']}"
                        )  # New log as per instruction
                        if data["status"] == "finished":
                            self.signals.finished.emit()
                            break
                        elif data["status"] == "error":
                            msg = data.get("message", "Unknown error")
                            logger.error(f"Runner received error: {msg}")
                            self.signals.error.emit(str(msg))
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
                    self.signals.error.emit("Unknown error")

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
    update_image = pyqtSignal(object)
    update_plot = pyqtSignal(dict)
    training_result = pyqtSignal(dict)
    result = pyqtSignal(object)
    update_status = pyqtSignal(str)

    finished = pyqtSignal()
    error = pyqtSignal(str)


class GenericProgressWindow(CelldetectiveDialog):

    def __init__(
        self,
        process=None,
        parent_window=None,
        title="",
        process_args=None,
        label_text="Progress:",
    ):

        super().__init__()

        self.setWindowTitle(f"{title}")
        self.__process = process
        self.parent_window = parent_window

        self.__btn_stp = QPushButton("Cancel")
        self.__label = QLabel("Idle")
        self.progress_label = QLabel(label_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.__runner = Runner(
            process=self.__process,
            process_args=process_args,
        )
        logger.info("Runner initialized...")
        self.pool = QThreadPool.globalInstance()

        self.__btn_stp.clicked.connect(self.__stp_net)
        self.__runner.signals.finished.connect(self.__on_finished)
        self.__runner.signals.error.connect(self.__on_error)
        self.__runner.signals.update_status.connect(self.__label.setText)

        # Connect update_pos for generic progress (Runner maps generic list progress to update_pos)
        self.__runner.signals.update_pos.connect(self.progress_bar.setValue)

        self.__btn_stp.setDisabled(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.__btn_stp)
        self.btn_layout.addWidget(self.__label)

        self.layout.addLayout(self.btn_layout)

        self.setLayout(self.layout)
        self.setFixedSize(QSize(400, 150))
        self.show()
        self.raise_()
        self.activateWindow()
        logger.info("GenericProgressWindow initialized and shown.")
        self.__run_net()
        self.setModal(True)

    def closeEvent(self, evnt):
        evnt.ignore()
        self.setWindowState(Qt.WindowMinimized)

    def __run_net(self):
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

    def __on_error(self, message="Error"):
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nError")
        self.__runner.close()

        # Show error in a message box to ensure it's seen
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Process failed")
        msg.setInformativeText(str(message))
        msg.setWindowTitle("Error")
        msg.exec_()

        self.reject()
