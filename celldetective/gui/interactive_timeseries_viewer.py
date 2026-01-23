import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector

from celldetective.gui.base.styles import Styles
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective import get_logger

logger = get_logger(__name__)


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
        self.fig.patch.set_alpha(0)
        self.canvas = FigureCanvas(self.fig, interactive=True)
        self.canvas.setStyleSheet("background-color:transparent;")
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
