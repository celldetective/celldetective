import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, \
	QSizePolicy, QVBoxLayout
from fonticon_mdi6 import MDI6
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective import get_logger

logger = get_logger(__name__)

class DynamicProgressDialog(QDialog, Styles):
    canceled = pyqtSignal()
    interrupted = pyqtSignal()

    def __init__(
        self,
        title="Training Progress",
        label_text="Launching the training script...",
        minimum=0,
        maximum=100,
        max_epochs=100,
        parent=None,
    ):
        super().__init__(parent)
        Styles.__init__(self)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowModality(Qt.ApplicationModal)

        self.resize(600, 500)  # Standard size

        self.max_epochs = max_epochs  # Keep this from original __init__
        self.current_epoch = 0  # Keep this from original __init__
        self.metrics_history = (  # Keep this from original __init__
            {}
        )  # Struct: {metric_name: {train: [], val: [], epochs: []}}
        self.current_model_name = None  # Keep this from original __init__
        self.last_update_time = 0  # Keep this from original __init__
        self.log_scale = False  # Keep this from original __init__
        self.user_interrupted = False
        self.is_percentile_scaled = False

        # Layouts
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        # Labels
        self.status_label = QLabel(label_text)
        # self.status_label.setStyleSheet("color: #333; font-size: 14px;")
        layout.addWidget(self.status_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(minimum, maximum)
        self.progress_bar.setStyleSheet(self.progress_bar_style)
        layout.addWidget(self.progress_bar)

        # Plot Canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.figure.patch.set_alpha(0.0)  # Transparent figure
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.apply_plot_style()

        # Toolbar / Controls
        controls_layout = QHBoxLayout()

        # Log Scale Button
        self.btn_log = QPushButton("")
        self.btn_log.setCheckable(True)
        self.btn_log.setIcon(icon(MDI6.math_log, color="black"))
        self.btn_log.clicked.connect(self.toggle_log_scale)
        self.btn_log.setStyleSheet(self.button_select_all)
        self.btn_log.setEnabled(False)

        # Auto Scale Button
        # self.btn_auto_scale = QPushButton("Auto Contrast")
        # self.btn_auto_scale.clicked.connect(self.auto_scale)
        # self.btn_auto_scale.setStyleSheet(self.button_style_sheet)
        # self.btn_auto_scale.setEnabled(False)
        # controls_layout.addWidget(self.btn_auto_scale)

        # Metric Selector
        self.metric_label = QLabel("Metric: ")
        self.metric_combo = QComboBox()
        # self.metric_combo.setStyleSheet(self.combo_style)
        self.metric_combo.currentIndexChanged.connect(self.force_update_plot)

        controls_layout.addWidget(self.metric_label, 10)
        controls_layout.addWidget(self.metric_combo, 85)
        controls_layout.addWidget(self.btn_log, 5, alignment=Qt.AlignRight)
        layout.addLayout(controls_layout)

        # Add Canvas
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.canvas)

        # Buttons Layout
        btn_layout = QHBoxLayout()

        # Skip Button
        self.skip_btn = QPushButton("Interrupt && Skip")
        self.skip_btn.setStyleSheet(self.button_style_sheet_2)
        self.skip_btn.setIcon(icon(MDI6.skip_next, color=self.celldetective_blue))
        self.skip_btn.clicked.connect(self.on_skip)
        self.skip_btn.setEnabled(False)
        btn_layout.addWidget(self.skip_btn, 50)

        # Cancel Button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(self.button_style_sheet)
        self.cancel_btn.clicked.connect(self.on_cancel)
        btn_layout.addWidget(self.cancel_btn, 50)

        layout.addLayout(btn_layout)
        self._get_screen_height()
        self.adjustSize()
        new_width = int(self.width() * 1.01)
        self.resize(new_width, int(self._screen_height * 0.7))
        self.setMinimumWidth(new_width)

    def _get_screen_height(self):
        app = QApplication.instance()
        screen = app.primaryScreen()
        geometry = screen.availableGeometry()
        self._screen_width, self._screen_height = geometry.getRect()[-2:]

    def on_skip(self):
        self.interrupted.emit()
        self.skip_btn.setDisabled(True)
        self.user_interrupted = True
        self.status_label.setText(
            "Interrupting current model training [effective at the end of the current epoch]..."
        )

    def apply_plot_style(self):
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.patch.set_alpha(0.0)
        self.ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        self.ax.minorticks_on()
        if getattr(self, "log_scale", False):
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")

    def show_result(self, results):
        """Display final results (Confusion Matrix or Regression Plot)"""
        self.ax.clear()
        self.apply_plot_style()
        self.ax.set_yscale("linear")
        self.ax.set_xscale("linear")
        self.metric_combo.hide()
        self.metric_label.hide()
        self.btn_log.hide()
        # self.btn_auto_scale.hide()

        # Regression
        if "val_predictions" in results and "val_ground_truth" in results:
            preds = results["val_predictions"]
            gt = results["val_ground_truth"]

            self.ax.scatter(gt, preds, alpha=0.5, c="white", edgecolors="C0")

            min_val = min(gt.min(), preds.min())
            max_val = max(gt.max(), preds.max())
            self.ax.plot([min_val, max_val], [min_val, max_val], "r--")

            self.ax.set_xlabel("Ground Truth")
            self.ax.set_ylabel("Predictions")
            val_mse = results.get("val_mse", "N/A")
            if isinstance(val_mse, (int, float)):
                title_str = f"Regression Result (MSE: {val_mse:.4f})"
            else:
                title_str = f"Regression Result (MSE: {val_mse})"
            self.ax.set_title(title_str)
            self.ax.set_aspect("equal", adjustable="box")

        # Classification (Confusion Matrix)
        elif "val_confusion" in results or "test_confusion" in results:
            cm = results.get("val_confusion", results.get("test_confusion"))
            norm_cm = cm / cm.sum(axis=1)[:, np.newaxis]

            im = self.ax.imshow(
                norm_cm, interpolation="nearest", cmap=plt.cm.Blues, aspect="equal"
            )
            self.ax.set_title("Confusion Matrix (Normalized)")
            self.ax.set_ylabel("True label")
            self.ax.set_xlabel("Predicted label")

            # Custom ticks
            tick_marks = np.arange(len(norm_cm))
            self.ax.set_xticks(tick_marks)
            self.ax.set_yticks(tick_marks)

            if len(norm_cm) == 3:
                labels = ["event", "no event", "else"]
                self.ax.set_xticklabels(labels)
                self.ax.set_yticklabels(labels)

            self.ax.grid(False)

            fmt = ".2f"
            thresh = norm_cm.max() / 2.0
            for i in range(norm_cm.shape[0]):
                for j in range(norm_cm.shape[1]):
                    self.ax.text(
                        j,
                        i,
                        format(norm_cm[i, j], fmt),
                        ha="center",
                        va="center",
                        color="white" if norm_cm[i, j] > thresh else "black",
                    )

        else:
            self.ax.text(
                0.5,
                0.5,
                "No visualization data found.",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
        self.canvas.draw()

    def toggle_log_scale(self):
        self.log_scale = self.btn_log.isChecked()
        self.update_plot_display()
        self.figure.tight_layout()
        if self.ax.get_yscale() == "linear":
            self.btn_log.setIcon(icon(MDI6.math_log, color="black"))
            try:
                QTimer.singleShot(
                    100, lambda: self.resize(self.width() - 1, self.height() - 1)
                )
            except:
                pass
        else:
            self.btn_log.setIcon(icon(MDI6.math_log, color="white"))
            try:
                QTimer.singleShot(
                    100, lambda: self.resize(self.width() + 1, self.height() + 1)
                )
            except:
                pass

    def auto_scale(self):
        target_metric = self.metric_combo.currentText()
        if not target_metric or target_metric not in self.metrics_history:
            return

        # Get data once
        data = self.metrics_history[target_metric]
        y_values = []
        if "train" in data:
            y_values.extend([v for v in data["train"] if v is not None])
        if "val" in data:
            y_values.extend([v for v in data["val"] if v is not None])

        y_values = np.array(y_values)
        if len(y_values) == 0:
            return

        if not getattr(self, "is_percentile_scaled", False):
            # Mode: Percentile 1-99
            try:
                p1, p99 = np.nanpercentile(y_values, [1, 99])
                if p1 != p99:
                    self.ax.set_ylim(p1, p99)
                    self.is_percentile_scaled = True
            except Exception as e:
                logger.warning(f"Could not compute percentiles: {e}")
        else:
            # Mode: Min/Max (Standard Autoscale)
            try:
                min_val, max_val = np.nanmin(y_values), np.nanmax(y_values)
                # Add a small padding (5%)
                margin = (max_val - min_val) * 0.05
                if margin == 0:
                    margin = 0.1  # default padding if constant
                self.ax.set_ylim(min_val - margin, max_val + margin)
                self.is_percentile_scaled = False
            except Exception as e:
                logger.warning(f"Could not compute min/max: {e}")
                self.ax.relim()
                self.ax.autoscale_view()

        self.canvas.draw()

    def force_update_plot(self):
        self.update_plot_display()

    def on_cancel(self):
        self.canceled.emit()
        self.reject()

    def update_progress(self, value, text=None):
        self.progress_bar.setValue(value)
        if text:
            self.status_label.setText(text)

    def update_plot(self, epoch_data):
        import time

        """
        epoch_data: dict with keys 'epoch', 'metrics' (dict), 'val_metrics' (dict), 'model_name', 'total_epochs'
        """
        model_name = epoch_data.get("model_name", "Unknown")
        total_epochs = epoch_data.get("total_epochs", 100)
        epoch = epoch_data.get("epoch", 0)
        metrics = epoch_data.get("metrics", {})
        val_metrics = epoch_data.get("val_metrics", {})

        # Handle Model Switch
        if model_name != self.current_model_name:
            self.metrics_history = {}  # Clear history
            self.current_model_name = model_name
            self.user_interrupted = False
            self.metric_combo.blockSignals(True)
            self.metric_combo.clear()
            # Populate combos with keys present in metrics (assuming val_metrics shares keys usually)
            # Find common keys or just use metrics keys for simplicity
            potential_metrics = list(metrics.keys())
            # Prioritize 'iou' or 'loss' if present
            potential_metrics.sort(
                key=lambda x: 0 if x in ["iou", "loss", "mse"] else 1
            )
            self.metric_combo.addItems(potential_metrics)
            self.metric_combo.blockSignals(False)

            self.status_label.setText(f"Training {model_name}...")
            self.ax.clear()
            self.apply_plot_style()
            self.metric_combo.show()
            self.metric_label.show()
            self.btn_log.show()
            # self.btn_auto_scale.show()
            self.btn_log.setEnabled(True)
            # self.btn_auto_scale.setEnabled(True)
            self.ax.set_aspect("auto")
            self.current_plot_metric = None
            self.update_plot_display()

        # Update History
        # Initialize keys if new
        for k, v in metrics.items():
            if k not in self.metrics_history:
                self.metrics_history[k] = {"train": [], "val": [], "epochs": []}

            self.metrics_history[k]["epochs"].append(epoch)
            self.metrics_history[k]["train"].append(v)

            # Find corresponding val metric
            val_key = f"val_{k}"
            if val_key in val_metrics:
                self.metrics_history[k]["val"].append(val_metrics[val_key])
            else:
                self.metrics_history[k]["val"].append(None)

        # Store total epochs for limits
        self.current_total_epochs = total_epochs

        # Throttle Update (3 seconds) OR if explicit end
        current_time = time.time()

        if epoch > -1 and not self.user_interrupted:
            self.skip_btn.setEnabled(True)

        if (current_time - self.last_update_time > 3.0) or (epoch >= total_epochs):
            self.update_plot_display()
            self.last_update_time = current_time

    def update_plot_display(self):
        target_metric = self.metric_combo.currentText()
        if not target_metric or target_metric not in self.metrics_history:
            return

        data = self.metrics_history[target_metric]

        # Check if we need to initialize the plot (new metric or first time)
        if getattr(self, "current_plot_metric", None) != target_metric:
            self.ax.clear()
            self.apply_plot_style()
            # self.ax.set_title(f"Training {self.current_model_name} - {target_metric}")
            self.ax.set_xlabel("Epoch")
            self.ax.set_ylabel(target_metric)

            # Initial X limits
            if hasattr(self, "current_total_epochs"):
                self.ax.set_xlim(0, self.current_total_epochs)

            # Initialize lines
            (self.train_line,) = self.ax.plot(
                [], [], label="Train", marker=".", color="tab:blue"
            )
            (self.val_line,) = self.ax.plot(
                [], [], label="Validation", marker=".", color="tab:orange"
            )
            self.ax.legend()
            self.current_plot_metric = target_metric

        # Update data
        if any(v is not None for v in data["train"]):
            self.train_line.set_data(data["epochs"], data["train"])

        if any(v is not None for v in data["val"]):
            self.val_line.set_data(data["epochs"], data["val"])

        # Update limits without resetting zoom if user zoomed
        if getattr(self, "log_scale", False):
            self.ax.set_yscale("log")
        else:
            self.ax.set_yscale("linear")

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        if max(data["epochs"]) % 2:
            try:
                QTimer.singleShot(
                    100, lambda: self.resize(self.width() + 1, self.height() + 1)
                )
            except:
                pass
        else:
            try:
                QTimer.singleShot(
                    100, lambda: self.resize(self.width() - 1, self.height() - 1)
                )
            except:
                pass

    def update_status(self, text):
        self.status_label.setText(text)
        if "Loading" in text and "librar" in text.lower():
            try:
                QTimer.singleShot(
                    100, lambda: self.status_label.setText("Training model...")
                )
            except:
                pass
