import pandas as pd
import matplotlib.pyplot as plt
from fonticon_mdi6 import MDI6
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QComboBox,
    QProgressBar,
    QSizePolicy,
    QApplication,
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles
from celldetective.gui.gui_utils import FigureCanvas
import logging
import numpy as np

logger = logging.getLogger(__name__)


class InteractiveEventViewer(QDialog, Styles):
    def __init__(
        self,
        table_path,
        signal_name=None,
        event_label=None,
        df=None,
        callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self.table_path = table_path

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(table_path)

        self.signal_name = signal_name
        self.event_label = event_label
        self.callback = callback
        self.selected_tracks = set()
        self.setWindowTitle("Interactive Event Viewer")
        self.resize(800, 600)

        # Analyze columns to identify signal, class, time columns
        self.detect_columns()

        self.init_ui()
        self.plot_signals()

    def notify_update(self):
        if self.callback:
            self.callback()

    def detect_columns(self):
        self.event_types = {}
        cols = self.df.columns

        # If explicit label is provided, prioritize it
        if self.event_label is not None:
            label = self.event_label
            if label == "":  # No label
                c_col, t_col, s_col = "class", "t0", "status"
            else:
                c_col, t_col, s_col = f"class_{label}", f"t_{label}", f"status_{label}"

            if c_col in cols and t_col in cols:
                self.event_types[label if label else "Default"] = {
                    "class": c_col,
                    "time": t_col,
                    "status": s_col if s_col in cols else None,
                }

        # If no label provided or columns not found (safety), fall back to scan
        if not self.event_types:
            # Check for default
            if "class" in cols and "t0" in cols:
                self.event_types["Default"] = {
                    "class": "class",
                    "time": "t0",
                    "status": "status" if "status" in cols else None,
                }

            # Check for labeled events
            # Find all columns starting with class_
            for c in cols:
                if c.startswith("class_") and c not in ["class_id", "class_color"]:
                    suffix = c[len("class_") :]
                    # Avoid duplication if label was provided but somehow not matched above
                    if suffix == self.event_label:
                        continue

                    t_col = f"t_{suffix}"
                    if t_col in cols:
                        status_col = f"status_{suffix}"
                        self.event_types[suffix] = {
                            "class": c,
                            "time": t_col,
                            "status": status_col if status_col in cols else None,
                        }

        if not self.event_types:
            # Fallback if no pairs found (maybe just class exists?)
            # Use heuristics from before but valid only if one exists
            self.event_types["Unknown"] = {
                "class": next(
                    (
                        c
                        for c in cols
                        if c.startswith("class")
                        and c not in ["class_id", "class_color"]
                    ),
                    "class",
                ),
                "time": next(
                    (c for c in cols if c.startswith("t_") or c == "t0"), "t0"
                ),
                "status": next((c for c in cols if c.startswith("status")), "status"),
            }

        # Set current active columns to first found
        self.set_active_event_type(next(iter(self.event_types)))

        self.time_axis_col = next((c for c in cols if c in ["FRAME", "time"]), "FRAME")
        self.track_col = next(
            (c for c in cols if c in ["TRACK_ID", "track"]), "TRACK_ID"
        )

        # Signal name detection
        if self.signal_name and self.signal_name not in cols:
            # Try to find a match (e.g. if config has 'dead_nuclei_channel' but table has 'dead_nuclei_channel_mean')
            potential = [c for c in cols if c.startswith(self.signal_name)]
            if potential:
                logger.info(
                    f"Signal '{self.signal_name}' not found. Using '{potential[0]}' instead."
                )
                self.signal_name = potential[0]
            else:
                logger.info(
                    f"Signal '{self.signal_name}' not found and no partial match. Falling back to auto-detection."
                )
                self.signal_name = None

        if self.signal_name is None:
            excluded = {
                "class_id",
                "class_color",
                "None",
                self.track_col,
                self.time_axis_col,
            }
            for info in self.event_types.values():
                excluded.update(info.values())

            candidates = [
                c
                for c in cols
                if c not in excluded
                and pd.api.types.is_numeric_dtype(self.df[c])
                and not c.startswith("class")
                and not c.startswith("t_")
                and not c.startswith("status")
            ]
            if candidates:
                self.signal_name = candidates[0]
            else:
                self.signal_name = cols[0]

    def set_active_event_type(self, type_name):
        self.current_event_type = type_name
        info = self.event_types[type_name]
        self.class_col = info["class"]
        self.time_col = info["time"]
        self.status_col = info["status"]

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Top controls
        top_layout = QHBoxLayout()

        # Event Type Selector
        if len(self.event_types) > 1:
            top_layout.addWidget(QLabel("Event Type:"))
            self.event_combo = QComboBox()
            self.event_combo.addItems(list(self.event_types.keys()))
            self.event_combo.currentTextChanged.connect(self.change_event_type)
            top_layout.addWidget(self.event_combo)

        top_layout.addWidget(QLabel("Signal:"))
        self.signal_combo = QComboBox()

        # Populate signal combo
        excluded = {
            "class_id",
            "class_color",
            "None",
            self.track_col,
            self.time_axis_col,
        }
        for info in self.event_types.values():
            excluded.update({v for k, v in info.items() if v})

        candidates = [
            c
            for c in self.df.columns
            if c not in excluded and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        self.signal_combo.addItems(candidates)
        if self.signal_name in candidates:
            self.signal_combo.setCurrentText(self.signal_name)
        self.signal_combo.currentTextChanged.connect(self.change_signal)
        top_layout.addWidget(self.signal_combo)

        top_layout.addWidget(QLabel("Filter:"))
        self.event_filter_combo = QComboBox()
        self.event_filter_combo.addItems(
            ["All", "Events (0)", "No Events (1)", "Else (2)"]
        )
        self.event_filter_combo.currentTextChanged.connect(self.plot_signals)
        top_layout.addWidget(self.event_filter_combo)

        self.event_btn = QPushButton("Event")
        self.event_btn.clicked.connect(lambda: self.set_class(0))
        top_layout.addWidget(self.event_btn)

        self.reject_btn = QPushButton("No Event")
        self.reject_btn.clicked.connect(lambda: self.set_class(1))
        top_layout.addWidget(self.reject_btn)

        self.else_btn = QPushButton("Left-censored/Else")
        self.else_btn.clicked.connect(lambda: self.set_class(2))
        top_layout.addWidget(self.else_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda: self.set_class(3))
        top_layout.addWidget(self.delete_btn)

        self.save_btn = QPushButton("Save Changes")
        self.save_btn.clicked.connect(self.save_changes)
        top_layout.addWidget(self.save_btn)

        for btn in [self.event_btn, self.reject_btn, self.else_btn, self.delete_btn]:
            btn.setStyleSheet(self.button_style_sheet_2)
        for btn in [self.save_btn]:
            btn.setStyleSheet(self.button_style_sheet)

        layout.addLayout(top_layout)

        # Plot
        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig, interactive=True)
        layout.addWidget(self.canvas)

        # Tooltip/Info
        self.info_label = QLabel(
            "Select (Box): Drag mouse. | Shift Time: Left/Right Arrows. | Set Class: Buttons above."
        )
        layout.addWidget(self.info_label)

    def change_event_type(self, text):
        self.set_active_event_type(text)
        self.plot_signals()

    def change_signal(self, text):
        self.signal_name = text
        self.plot_signals()

    def keyPressEvent(self, event):
        if not self.selected_tracks:
            super().keyPressEvent(event)
            return

        step = 0.5
        mask = self.df[self.track_col].isin(self.selected_tracks)

        if event.key() == Qt.Key_Left:
            # Shift curve LEFT: Increase t0 -> x decreases
            self.df.loc[mask, self.time_col] += step

            # Recompute status if column exists
            if self.status_col and self.status_col in self.df.columns:
                # status is 1 if time >= t0, else 0
                self.df.loc[mask, self.status_col] = (
                    self.df.loc[mask, self.time_axis_col]
                    >= self.df.loc[mask, self.time_col]
                ).astype(int)

            self.plot_signals()
            self.notify_update()
        elif event.key() == Qt.Key_Right:
            # Shift curve RIGHT: Decrease t0 -> x increases
            self.df.loc[mask, self.time_col] -= step

            # Recompute status if column exists
            if self.status_col and self.status_col in self.df.columns:
                # status is 1 if time >= t0, else 0
                self.df.loc[mask, self.status_col] = (
                    self.df.loc[mask, self.time_axis_col]
                    >= self.df.loc[mask, self.time_col]
                ).astype(int)

            self.plot_signals()
            self.notify_update()
        else:
            super().keyPressEvent(event)

    def plot_signals(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.lines = {}  # map line -> track_id

        # Filter based on combo box
        filter_choice = self.event_filter_combo.currentText()
        if "All" in filter_choice:
            # Exclude deleted (3) usually? Or allow all? Only exclude 3 if it means strictly delete.
            valid_mask = self.df[self.class_col] != 3
        elif "Events (0)" in filter_choice:
            valid_mask = self.df[self.class_col] == 0
        elif "No Events (1)" in filter_choice:
            valid_mask = self.df[self.class_col] == 1
        elif "Else (2)" in filter_choice:
            valid_mask = self.df[self.class_col] == 2
        else:
            valid_mask = ~self.df[self.class_col].isin([1, 3])

        if not valid_mask.any():
            # If nothing left, show empty or message?
            pass

        tracks = self.df[valid_mask][self.track_col].unique()

        for tid in tracks:
            group = self.df[self.df[self.track_col] == tid]
            t0 = group[self.time_col].iloc[0]
            # Handle NaN t0 if necessary
            if pd.isna(t0):
                continue

            time = group[self.time_axis_col].values
            signal = group[self.signal_name].values

            # Center time
            x = time - t0

            # Color coding
            # Class 0: Blue, Class 1: Gray, Class 2: Orange
            c_val = group[self.class_col].iloc[0]
            color = "tab:red"
            if c_val == 1:
                color = "tab:blue"
            elif c_val == 2:
                color = "yellow"

            (line,) = self.ax.plot(x, signal, picker=True, alpha=0.95, color=color)
            self.lines[line] = tid

            # Highlight if selected (persist selection)
            if tid in self.selected_tracks:
                line.set_color("red")
                line.set_alpha(1.0)

        self.ax.set_title(f"Centered Signals: {self.signal_name}")
        self.ax.set_xlabel("Time from Event (t - t0)")
        self.ax.set_ylabel("Signal Intensity")

        # Setup selector
        self.selector = RectangleSelector(
            self.ax,
            self.on_select_rect,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        self.ax.grid(True)

        self.canvas.draw()

    def on_select_rect(self, eclick, erelease):
        # Find lines intersecting the rectangle
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        self.selected_tracks.clear()

        for line, tid in self.lines.items():
            xdata = line.get_xdata()
            ydata = line.get_ydata()

            # Check if any point is in rect
            mask = (xdata >= xmin) & (xdata <= xmax) & (ydata >= ymin) & (ydata <= ymax)
            if mask.any():
                self.selected_tracks.add(tid)
                line.set_color("red")
                line.set_alpha(1.0)
            else:
                # Reset color based on class
                # Need to look up class again or store it in line metadata?
                # Just redraw is safer/easier or lookup df
                # Optimization: store class in lines map? self.lines[line] = (tid, class)
                # For now just set to blue/orange heuristic
                c_val = self.df.loc[
                    self.df[self.track_col] == tid, self.class_col
                ].iloc[0]
                color = "tab:red"
                if c_val == 1:
                    color = "tab:blue"
                elif c_val == 2:
                    color = "yellow"
                line.set_color(color)
                line.set_alpha(0.5)

        self.canvas.draw()
        self.info_label.setText(f"Selected {len(self.selected_tracks)} tracks.")

    def set_class(self, class_val):
        """Set class for selected tracks."""
        if not self.selected_tracks:
            return

        count = len(self.selected_tracks)
        # direct update without confirmation for speed, or maybe optional?
        # User wants interactive flow.

        mask = self.df[self.track_col].isin(self.selected_tracks)
        self.df.loc[mask, self.class_col] = class_val

        # Clear selection after action? Or keep it?
        # Usually better to clear or refresh.
        # Since we filter out Class 1/3, the lines will disappear.

        self.selected_tracks.clear()
        self.plot_signals()
        self.info_label.setText(f"Set {count} tracks to Class {class_val}.")

        self.notify_update()

    def reject_selection(self):
        self.set_class(1)

    def save_changes(self):
        try:
            self.df.to_csv(self.table_path, index=False)
            QMessageBox.information(self, "Saved", "Table saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save table: {e}")


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
            QTimer.singleShot(
                100, lambda: self.resize(self.width() - 1, self.height() - 1)
            )
        else:
            self.btn_log.setIcon(icon(MDI6.math_log, color="white"))
            QTimer.singleShot(
                100, lambda: self.resize(self.width() + 1, self.height() + 1)
            )

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
            QTimer.singleShot(
                100, lambda: self.resize(self.width() + 1, self.height() + 1)
            )
        else:
            QTimer.singleShot(
                100, lambda: self.resize(self.width() - 1, self.height() - 1)
            )

    def update_status(self, text):
        self.status_label.setText(text)
        if "Loading" in text and "librar" in text.lower():
            QTimer.singleShot(
                100, lambda: self.status_label.setText("Training model...")
            )
