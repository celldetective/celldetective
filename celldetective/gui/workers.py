from multiprocessing import Queue
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QApplication, QComboBox
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QSize, Qt
from PyQt5.QtGui import QPixmap, QImage
from typing import Optional, Any, Dict
import math
import numpy as np

from celldetective.gui.base.components import CelldetectiveDialog
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


class ProgressWindow(CelldetectiveDialog):

    def __init__(
        self,
        process: Optional[Any] = None,
        parent_window: Optional[Any] = None,
        title: str = "",
        position_info: bool = True,
        process_args: Optional[Dict[str, Any]] = None,
        well_label: str = "Well progress:",
        pos_label: str = "Position progress:",
    ) -> None:
        """
        Initialize the ProgressWindow.

        Parameters
        ----------
        process : class, optional
            The process class to run.
        parent_window : QMainWindow, optional
            The parent window.
        title : str, optional
            The window title.
        position_info : bool, optional
            Whether to show position info.
        process_args : dict, optional
            Arguments for the process.
        well_label : str, optional
            Label for well progress.
        pos_label : str, optional
            Label for position progress.
        """

        super().__init__()
        # QDialog.__init__(self)

        self.setWindowTitle(f"{title}")
        self.__process = process
        self.parent_window = parent_window
        self.plot_data = {}

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
        self.__runner.signals.update_plot.connect(self.on_update_plot)

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

    def closeEvent(self, evnt: Any) -> None:
        """
        Handle close event.

        Parameters
        ----------
        evnt : QCloseEvent
            The close event.
        """
        if QApplication.closingDown():
            # App is shutting down — stop the job and allow the close.
            self.__runner.close()
            evnt.accept()
        else:
            # Accidental X-button press while job is running — minimize instead.
            evnt.ignore()
            self.setWindowState(Qt.WindowMinimized)

    def __run_net(self) -> None:
        """Start the runner."""
        # self.__btn_run.setDisabled(True)
        self.__btn_stp.setEnabled(True)
        self.__label.setText("Running...")
        self.pool.start(self.__runner)

    def __stp_net(self) -> None:
        """Stop the runner."""
        self.__runner.close()
        logger.info("\n Job cancelled... Abort.")
        self.reject()

    def __on_finished(self) -> None:
        """Handle process completion."""
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nFinished!")
        self.__runner.close()
        self.accept()

    def __on_error(self, message: str = "Error") -> None:
        """
        Handle process error.

        Parameters
        ----------
        message : str, optional
            The error message.
        """
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

    def update_image(self, img_data: Optional[np.ndarray]) -> None:
        """
        Update the image preview.

        Parameters
        ----------
        img_data : ndarray
            The image data.
        """
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

    def on_update_plot(self, plot_data: dict) -> None:
        """Cache the received plot data by its stack path."""
        if isinstance(plot_data, dict) and "stack_path" in plot_data:
            path = plot_data["stack_path"]
            self.plot_data[path] = plot_data


class Runner(QRunnable):

    def __init__(
        self,
        process: Optional[Any] = None,
        process_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Runner.

        Parameters
        ----------
        process : class
            The process class to run.
        process_args : dict, optional
            Arguments for the process.
        """
        QRunnable.__init__(self)

        logger.info(f"{process_args=}")
        self.__queue = Queue()
        self.__process = process(self.__queue, process_args=process_args)
        self.signals = RunnerSignal()

    def run(self) -> None:
        """Run the process."""
        logger.info("Starting Process (runner-side)...")
        self.__process.start()
        while True:
            try:
                data = self.__queue.get(timeout=2)
            except Exception:
                # Timeout — check if the subprocess died without sending "finished"
                if not self.__process.is_alive():
                    logger.error("Subprocess exited without sending a status message.")
                    self.signals.error.emit("Process exited unexpectedly.")
                    break
                continue
            try:

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
                logger.error(f"{e}")

    def close(self) -> None:
        """Close the process."""
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
        process: Optional[Any] = None,
        parent_window: Optional[Any] = None,
        title: str = "",
        process_args: Optional[Dict[str, Any]] = None,
        label_text: str = "Progress:",
    ) -> None:
        """
        Initialize the GenericProgressWindow.

        Parameters
        ----------
        process : class, optional
            The process class to run.
        parent_window : QMainWindow, optional
            The parent window.
        title : str, optional
            The window title.
        process_args : dict, optional
            Arguments for the process.
        label_text : str, optional
            The label text.
        """

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

    def closeEvent(self, evnt: Any) -> None:
        """
        Handle close event.

        Parameters
        ----------
        evnt : QCloseEvent
            The close event.
        """
        if QApplication.closingDown():
            # App is shutting down — stop the job and allow the close.
            self.__runner.close()
            evnt.accept()
        else:
            # Accidental X-button press while job is running — minimize instead.
            evnt.ignore()
            self.setWindowState(Qt.WindowMinimized)

    def __run_net(self) -> None:
        """Start the runner."""
        self.__btn_stp.setEnabled(True)
        self.__label.setText("Running...")
        self.pool.start(self.__runner)

    def __stp_net(self) -> None:
        """Stop the runner."""
        self.__runner.close()
        logger.info("\n Job cancelled... Abort.")
        self.reject()

    def __on_finished(self) -> None:
        """Handle process completion."""
        self.__btn_stp.setDisabled(True)
        self.__label.setText("\nFinished!")
        self.__runner.close()
        self.accept()

    def __on_error(self, message: str = "Error") -> None:
        """
        Handle process error.

        Parameters
        ----------
        message : str, optional
            The error message.
        """
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


class DriftTrajectoryPlotDialog(CelldetectiveDialog):
    def __init__(self, plot_data: dict, parent_window: Optional[Any] = None) -> None:
        """
        Initialize the interactive DriftTrajectoryPlotDialog.
        
        Parameters
        ----------
        plot_data : dict
            A dictionary mapping stack paths to drift registration metadata dictionaries.
        parent_window : QMainWindow, optional
            The parent window.
        """
        super().__init__()
        self.setWindowTitle("Drift Trajectory Plotter")
        self.plot_data = plot_data
        self.parent_window = parent_window

        # UI Layout
        layout = QVBoxLayout()
        
        # Header layout for dropdown
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Select Registered Movie/Position: "))
        
        self.position_cb = QComboBox()
        # Populate the combobox with nice display names
        import os
        for path in self.plot_data.keys():
            display_name = os.path.basename(path)
            self.position_cb.addItem(display_name, path)
            
        header_layout.addWidget(self.position_cb, 1)
        layout.addLayout(header_layout)

        # Matplotlib Figure & Canvas
        import matplotlib
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        # Premium dark/blue styling
        self.fig = Figure(figsize=(9, 7), dpi=100, facecolor="#f5f5f5")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        # Bottom Close button
        btn_layout = QHBoxLayout()
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565c0;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.resize(950, 750)

        # Connect position change event
        self.position_cb.currentIndexChanged.connect(self.update_plot)
        
        # Initial plot
        self.update_plot()

    def update_plot(self) -> None:
        """Redraw drift trajectory and quality metrics subplots."""
        import numpy as np
        
        path = self.position_cb.currentData()
        if not path or path not in self.plot_data:
            return
            
        data = self.plot_data[path]
        shifts = np.array(data["shifts"])  # (N, 2) -> [dy, dx]
        raw_shifts = np.array(data["raw_shifts"]) # (N, 2)
        fallbacks = data.get("fallbacks", [])
        sift_inliers = np.array(data.get("sift_inliers", []))
        max_shift_limit = data.get("max_shift_limit", 0.0)

        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212, sharex=ax1)

        # Premium Color Palette
        color_y = "#1565c0"  # Vibrant blue for Y drift
        color_x = "#7b1fa2"  # Vibrant purple for X drift
        color_raw = "#9e9e9e"  # Soft gray for raw shifts
        color_outlier = "#d32f2f"  # Red for outliers
        color_fallback = "#fbc02d"  # Amber for SIFT fallback

        # Plot Subplot 1: Accumulated Translation Drift
        frames = np.arange(len(shifts))
        
        # Raw shifts (dashed)
        ax1.plot(frames, raw_shifts[:, 0], color=color_y, linestyle="--", alpha=0.4, label="Raw dy (Unfiltered)")
        ax1.plot(frames, raw_shifts[:, 1], color=color_x, linestyle="--", alpha=0.4, label="Raw dx (Unfiltered)")
        
        # Filtered shifts (solid)
        ax1.plot(frames, shifts[:, 0], color=color_y, linestyle="-", linewidth=2.0, label="Filtered dy (Median)")
        ax1.plot(frames, shifts[:, 1], color=color_x, linestyle="-", linewidth=2.0, label="Filtered dx (Median)")

        # Highlight outliers where raw != filtered
        outliers = np.where(np.any(raw_shifts != shifts, axis=1))[0]
        if len(outliers) > 0:
            ax1.scatter(outliers, raw_shifts[outliers, 0], color=color_outlier, marker="o", s=30, zorder=5, label="Filtered Outlier")
            ax1.scatter(outliers, raw_shifts[outliers, 1], color=color_outlier, marker="o", s=30, zorder=5)

        # Highlight SIFT/Fourier fallback events
        if len(fallbacks) > 0:
            first = True
            for fb in fallbacks:
                ax1.axvline(x=fb, color=color_fallback, linestyle=":", alpha=0.7, linewidth=1.5,
                            label="Fallback Event" if first else "")
                first = False

        # Max shift threshold lines
        if max_shift_limit > 0:
            ax1.axhline(y=max_shift_limit, color="#b71c1c", linestyle="-.", alpha=0.5, label=f"Max Shift Limit ({max_shift_limit} px)")
            ax1.axhline(y=-max_shift_limit, color="#b71c1c", linestyle="-.", alpha=0.5)

        ax1.set_title("Drift Trajectory Analysis (Accumulated Translation shifts)", fontsize=11, fontweight="bold", color="#333333")
        ax1.set_ylabel("Drift Translation (pixels)", fontsize=10, fontweight="semibold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9)

        # Plot Subplot 2: SIFT keypoints / Quality metrics
        if len(sift_inliers) > 0 and np.any(sift_inliers > 0):
            ax2.plot(frames, sift_inliers, color="#2e7d32", linestyle="-", linewidth=1.5, marker=".", markersize=4, label="SIFT Inliers (RANSAC)")
            ax2.set_ylabel("RANSAC Inliers Count", fontsize=10, fontweight="semibold")
            ax2.axhline(y=3, color=color_outlier, linestyle="--", alpha=0.5, label="Min SIFT Inliers (3)")
            ax2.grid(True, linestyle=":", alpha=0.6)
            ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
            ax2.set_title("SIFT Feature Matching Quality Metrics", fontsize=11, fontweight="bold", color="#333333")
        else:
            # Fourier phase cross-correlation quality placeholder
            ax2.text(0.5, 0.5, "Fourier Phase Cross-Correlation Mode\n(No SIFT features extracted)",
                     horizontalalignment="center", verticalalignment="center",
                     transform=ax2.transAxes, fontsize=10, color="#666666", style="italic")
            ax2.set_title("Matching Confidence / Quality Metrics", fontsize=11, fontweight="bold", color="#333333")
            ax2.set_ylabel("Quality Index", fontsize=10, fontweight="semibold")
            ax2.grid(True, linestyle=":", alpha=0.6)

        ax2.set_xlabel("Frame Index (Time)", fontsize=10, fontweight="semibold")
        
        # Adjust subplot margins nicely
        self.fig.tight_layout()
        self.canvas.draw()
