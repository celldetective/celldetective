from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QShortcut,
    QTabWidget,
    QWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from matplotlib.axes import Axes
from typing import List, Optional, Union, Any
import numpy as np
import pandas as pd
import os
from matplotlib.cm import tab10
from celldetective.gui.base.sliders import QLabeledDoubleSlider
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6

import matplotlib.pyplot as plt
from celldetective.gui.annotator_common import (
    NON_SIGNAL_COLUMNS,
    closest_picked_index,
    confirm,
    default_dataset_name,
    experiment_columns,
    fit_feature_scaler,
    icon_button,
    new_class_dialog,
    save_dataset,
    set_toggle_icon,
    split_by_frame,
    warn,
)
from celldetective.gui.base_annotator import BaseAnnotator
from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer
from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.gui_utils import color_from_state, ExportPlotBtn
from celldetective.utils.image_loaders import locate_labels
from celldetective.utils.data_cleaning import extract_identity_col
from celldetective import get_logger

logger = get_logger(__name__)

# Columns of the annotation bookkeeping rather than characteristic groups.
NON_GROUP_COLUMNS = (
    "class_id",
    "group_color",
    "class_color",
    "group_id",
    "status_color",
    "status_id",
)


class AnnotatorStackVisualizer(CellEdgeVisualizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the AnnotatorStackVisualizer.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to CellEdgeVisualizer.
        **kwargs : dict
            Keyword arguments passed to CellEdgeVisualizer.
        """
        self.scat_markers = None
        super().__init__(*args, **kwargs)
        self.compact_layout()

    def generate_figure_canvas(self) -> None:
        """
        Generate the figure canvas and initialize custom overlays.
        """
        super().generate_figure_canvas()
        self.generate_custom_overlays()

    def generate_custom_overlays(self) -> None:
        """Initialize scatter artists."""
        self.scat_markers = self.ax.scatter([], [], color="tab:red", picker=True)
        # CellEdgeVisualizer handles self.im_mask

    def update_overlays(self, positions: np.ndarray, colors: np.ndarray) -> None:
        """
        Update the scatter plot overlays with new positions and colors.

        Parameters
        ----------
        positions : array-like
            The positions of the points (N, 2).
        colors : array-like
            The colors of the points.
        """
        self.scat_markers.set_offsets(positions)
        if len(positions) > 0 and len(colors) > 0:
            self.scat_markers.set_edgecolors(colors)
            self.scat_markers.set_facecolors("none")
            self.scat_markers.set_picker(10)
            self.scat_markers.set_linewidths(2)
            self.scat_markers.set_sizes([200] * len(positions))

        self.canvas.canvas.draw_idle()

    def generate_edge_slider(self) -> None:
        """No edge slider: the contours are not tuned here."""

    def generate_opacity_slider(self) -> None:
        """
        Generate a compact opacity slider for the mask.
        """
        self.opacity_slider = QLabeledDoubleSlider()
        self.opacity_slider.setOrientation(Qt.Horizontal)
        self.opacity_slider.setRange(0, 1)
        self.opacity_slider.setValue(self.mask_alpha)
        self.opacity_slider.setDecimals(2)
        self.opacity_slider.valueChanged.connect(self.change_mask_opacity)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(QLabel("Opacity:"), 15)
        layout.addWidget(self.opacity_slider, 85)
        self.canvas.layout.addLayout(layout)

    def compact_layout(self) -> None:
        """
        Reduce margins and spacing for a more compact layout.
        """
        self.canvas.layout.setSpacing(0)
        self.canvas.layout.setContentsMargins(0, 5, 0, 5)
        for i in range(self.canvas.layout.count()):
            layout = self.canvas.layout.itemAt(i).layout()
            if layout:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)


class MeasureAnnotator(BaseAnnotator):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the MeasureAnnotator.
        """
        self.status_name = "group"
        super().__init__(read_config=False, *args, **kwargs)

        self.setWindowTitle("Phenotype Annotator")

        self.int_validator = QIntValidator()
        self.current_alpha = 0.5
        self.value_magnitude = 1

        epsilon = 0.01
        self.observed_min_intensity = 0
        self.observed_max_intensity = 0 + epsilon

        self.current_frame = 0
        self.show_fliers = False

        if not self.proceed:
            self.close()
            return

        from celldetective.utils.image_loaders import fix_missing_labels

        # Ensure labels match stack length
        self.labels = locate_labels(self.pos, population=self.mode)
        if self.len_movie > 0:
            if self.labels is None or len(self.labels) < self.len_movie:
                fix_missing_labels(
                    self.pos,
                    population=self.mode,
                    prefix=self.parent_window.movie_prefix,
                )
                self.labels = locate_labels(self.pos, population=self.mode)
            elif len(self.labels) > self.len_movie:
                self.labels = self.labels[: self.len_movie]

        self.current_channel = 0
        self.frame_lbl = QLabel("position: ")

        self.populate_window()
        self.changed_class()

        self.previous_index = None

    @staticmethod
    def group_columns(columns) -> List[str]:
        """The characteristic groups: group, group_* and status_* columns."""
        return [
            c
            for c in columns
            if (c == "group" or c.startswith(("group_", "status_")))
            and c not in NON_GROUP_COLUMNS
        ]

    def locate_tracks(self) -> None:
        """
        Locate the tracks.
        """

        if not os.path.exists(self.trajectories_path):
            warn("The trajectories cannot be detected.")
            self.close()
            return

        self.annotation_log = []

        # Load and prep tracks
        self.df_tracks = pd.read_csv(self.trajectories_path)
        id_col = extract_identity_col(self.df_tracks)
        self.df_tracks = self.df_tracks.sort_values(by=[id_col, "FRAME"])

        # Generate missing status columns from class columns
        for c in self.df_tracks.columns:
            if not c.startswith("class_") or c in NON_GROUP_COLUMNS:
                continue
            status_col = c.replace("class_", "status_")
            if status_col in self.df_tracks.columns:
                continue
            if c == "class_firstdetection":
                try:
                    from celldetective.tracking import write_first_detection_class

                    self.df_tracks = write_first_detection_class(self.df_tracks)
                except Exception as e:
                    logger.error(f"Could not generate status_firstdetection: {e}")
                    self.df_tracks[status_col] = self.df_tracks[c]
            else:
                self.df_tracks[status_col] = self.df_tracks[c]

        self.class_cols = [
            c
            for c in self.df_tracks.columns
            if c.startswith(("group", "status")) and c not in NON_GROUP_COLUMNS
        ]
        if self.class_cols:
            if self.status_name not in self.class_cols:
                self.status_name = self.class_cols[0]
        else:
            self.status_name = "group"

        if self.status_name not in self.df_tracks.columns:
            # only create the status column if it does not exist to not erase static classification results
            self.make_status_column()

        self.color_states()

        self.df_tracks = self.df_tracks.dropna(subset=["POSITION_X", "POSITION_Y"])
        self.df_tracks["x_anim"] = self.df_tracks["POSITION_X"].astype(int)
        self.df_tracks["y_anim"] = self.df_tracks["POSITION_Y"].astype(int)

        self.extract_scatter_from_trajectories()
        self.track_of_interest = self.df_tracks.dropna(subset=id_col)[id_col].min()
        self.locate_track_of_interest()

        self.columns_to_rescale, self.feature_scaler = fit_feature_scaler(
            self.df_tracks,
            excluded=list(NON_SIGNAL_COLUMNS)
            + self.class_cols
            + experiment_columns(self.exp_dir),
        )

    def color_states(self) -> None:
        """Color the cells by their state in the current group."""

        all_states = np.array(self.df_tracks.loc[:, self.status_name].tolist())
        self.state_color_map = color_from_state(all_states, recently_modified=False)
        self.df_tracks["group_color"] = self.df_tracks[self.status_name].apply(
            self.assign_color_state
        )

    def populate_options_layout(self) -> None:
        """Populate the options layout."""
        # clear options hbox
        for i in reversed(range(self.options_hbox.count())):
            self.options_hbox.itemAt(i).widget().setParent(None)

        time_option_hbox = QHBoxLayout()
        time_option_hbox.setContentsMargins(100, 0, 100, 0)
        time_option_hbox.setSpacing(0)

        self.time_of_interest_label = QLabel("phenotype: ")
        time_option_hbox.addWidget(self.time_of_interest_label, 30)

        self.time_of_interest_le = QLineEdit()
        self.time_of_interest_le.setValidator(self.int_validator)
        time_option_hbox.addWidget(self.time_of_interest_le)

        self.suppr_btn = icon_button(
            MDI6.delete, self.button_select_all, tooltip="Delete cell"
        )
        self.suppr_btn.clicked.connect(self.del_cell)
        time_option_hbox.addWidget(self.suppr_btn)

        self.options_hbox.addLayout(time_option_hbox)

    def update_widgets(self) -> None:
        """Update widgets."""
        self.class_label.setText("characteristic \n group: ")
        self.update_class_cb()
        self.add_class_btn.setToolTip("Add a new characteristic group")
        self.del_class_btn.setToolTip("Delete a characteristic group")

        self.export_btn.disconnect()
        self.export_btn.clicked.connect(self.export_measurements)

    def update_class_cb(self) -> None:
        """Update class combo box."""
        self.class_choice_cb.disconnect()
        self.class_choice_cb.clear()
        self.class_cols = self.group_columns(self.df_tracks.columns)
        self.class_choice_cb.addItems(self.class_cols)
        if self.status_name in self.class_cols:
            self.class_choice_cb.setCurrentText(self.status_name)
        self.class_choice_cb.currentIndexChanged.connect(self.changed_class)

    def populate_window(self) -> None:
        """Build the full window layout with a tabbed left panel."""
        # --- Main scaffold (same as base) ---
        self.button_widget = CelldetectiveWidget()
        self.main_layout = QHBoxLayout()
        self.button_widget.setLayout(self.main_layout)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # --- Left panel ---
        self.left_panel = QVBoxLayout()
        self.left_panel.setContentsMargins(10, 10, 10, 10)
        self.left_panel.setSpacing(6)

        # Class selection
        self.init_class_selection_block()
        # The rows keep their natural height (stretch 0): only the tabs holding
        # the plot take the room left, instead of it opening gaps between rows.
        self.left_panel.addLayout(self.class_hbox)
        self.left_panel.addWidget(self.cell_info)

        # Options & correction buttons
        self.init_options_block()
        self.populate_options_layout()
        self.update_widgets()
        self.init_correction_block()
        self.left_panel.addLayout(self.options_hbox)
        self.left_panel.addLayout(self.action_hbox)

        self.annotation_btns_to_hide = [
            self.time_of_interest_label,
            self.time_of_interest_le,
            self.suppr_btn,
        ]
        self.hide_annotation_buttons()
        self.left_panel.addSpacing(10)

        # --- Tabbed area ---
        self.init_plot_buttons_block()
        self.create_cell_histogram()
        self._build_left_panel_tabs()
        self.left_panel.addWidget(self.left_tabs, 65)

        # Save / Export at the very bottom
        self.init_save_btn_block()
        self.left_panel.addLayout(self.btn_hbox, 5)

        # --- Right panel ---
        self.right_panel = QVBoxLayout()

        self.viewer = AnnotatorStackVisualizer(
            stack_path=self.stack_path,
            labels=self.labels,
            frame_slider=True,
            channel_cb=True,
            channel_names=self.channel_names,
            n_channels=self.nbr_channels,
            target_channel=0,
            window_title="Stack Viewer",
        )
        self.viewer.frame_slider.valueChanged.connect(self.sync_frame)
        self.viewer.channel_cb.currentIndexChanged.connect(self.plot_signals)
        self.viewer.channel_cb.currentIndexChanged.connect(self._on_channel_changed)
        self.cid_pick = self.viewer.fig.canvas.mpl_connect(
            "pick_event", self.on_scatter_pick
        )
        self.right_panel.addWidget(self.viewer.canvas)

        # --- Assemble ---
        self.main_layout.addLayout(self.left_panel, 35)
        self.main_layout.addLayout(self.right_panel, 65)
        self.button_widget.adjustSize()
        self.setCentralWidget(self.button_widget)

        self.del_shortcut = QShortcut(Qt.Key_Delete, self)
        self.del_shortcut.activated.connect(self.shortcut_suppr)
        self.del_shortcut.setEnabled(False)

        # Force start at frame 0
        self.viewer.frame_slider.setValue(0)
        self.plot_signals()
        self.compact_layout_main()

    def compact_layout_main(self) -> None:
        """Compact the main layout."""
        if hasattr(self, "viewer"):
            self.viewer.compact_layout()

    def _build_left_panel_tabs(self) -> None:
        """Build the QTabWidget with signals and histogram tabs."""
        # --- Tab 1: Signals (dropdowns + boxplot + toolbar) ---
        signals_tab = QWidget()
        sig_layout = QVBoxLayout(signals_tab)
        sig_layout.setContentsMargins(10, 10, 10, 5)
        sig_layout.setSpacing(6)

        for label, combo in zip(self.signal_choice_label, self.signal_choice_cb):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(label, 20)
            row.addWidget(combo, 80)
            sig_layout.addLayout(row)

        # The plot outweighs the stretch below it: it takes the room until its
        # height is capped, and only what it leaves collects under it, rather
        # than spreading the signal rows above apart.
        sig_layout.addWidget(self.cell_fcanvas, 100)
        sig_layout.addStretch(1)

        # Customise the signals matplotlib toolbar
        if hasattr(self.cell_fcanvas, "toolbar"):
            tb = self.cell_fcanvas.toolbar
            self._strip_toolbar(tb)
            tb.addSeparator()
            tb.addWidget(self.outliers_check)
            self._sig_norm_action = tb.addAction(
                icon(MDI6.arrow_collapse_vertical, color="black"),
                "Normalize", self.normalize_features,
            )
            self._sig_log_action = tb.addAction(
                icon(MDI6.math_log, color="black"),
                "Log scale", self.switch_to_log,
            )
            tb.addAction(
                icon(MDI6.content_save, color="black"),
                "Save figure", lambda: self.export_plot_btn.save_plot(),
            )

        # --- Tab 2: Cell Histogram ---
        hist_tab = QWidget()
        hist_layout = QVBoxLayout(hist_tab)
        hist_layout.setContentsMargins(10, 10, 10, 5)
        hist_layout.setSpacing(6)
        hist_layout.addWidget(self.hist_canvas, 1)

        # Customise the histogram matplotlib toolbar
        if hasattr(self.hist_canvas, "toolbar"):
            tb = self.hist_canvas.toolbar
            self._strip_toolbar(tb)
            tb.addSeparator()
            self._hist_density_action = tb.addAction(
                icon(MDI6.chart_bell_curve_cumulative, color="black"),
                "Density", self._toggle_hist_density,
            )
            self._hist_log_action = tb.addAction(
                icon(MDI6.math_log, color="black"),
                "Log scale", self._toggle_hist_log,
            )
            tb.addAction(
                icon(MDI6.content_save, color="black"),
                "Save figure", lambda: self.hist_export_btn.save_plot(),
            )

        # --- Assemble ---
        self.left_tabs = QTabWidget()
        self.left_tabs.setDocumentMode(True)
        self.left_tabs.addTab(signals_tab, "Signals")
        self.left_tabs.addTab(hist_tab, "Cell Histogram")

    @staticmethod
    def _strip_toolbar(toolbar) -> None:
        """Remove Subplots, Customize, and Save from a matplotlib toolbar."""
        for action in toolbar.actions():
            if action.text() in ("Subplots", "Customize", "Save"):
                toolbar.removeAction(action)

    def switch_to_log(self):
        """Override to also update the toolbar action icon."""
        super().switch_to_log()
        if hasattr(self, "_sig_log_action"):
            set_toggle_icon(self._sig_log_action, MDI6.math_log, self.log_scale)

    def normalize_features(self):
        """Override to also update the toolbar action icon."""
        super().normalize_features()
        if hasattr(self, "_sig_norm_action"):
            set_toggle_icon(
                self._sig_norm_action,
                MDI6.arrow_collapse_vertical,
                self.normalized_signals,
            )

    # ------------------------------------------------------------------
    # Cell intensity histogram
    # ------------------------------------------------------------------

    def create_cell_histogram(self) -> None:
        """Create an empty histogram canvas for single-cell intensity."""
        self.hist_fig, self.hist_ax = plt.subplots(tight_layout=True)
        self.hist_canvas = FigureCanvas(self.hist_fig, interactive=True)
        self.hist_log_scale = False
        self.hist_density = False
        self._hist_ymax = 0  # remembered y-limit for density mode

        # Hidden ExportPlotBtn used for its save_plot() method
        self.hist_export_btn = ExportPlotBtn(self.hist_fig, export_dir=self.exp_dir)
        self.hist_export_btn.hide()

        self._style_hist_ax()
        self.hist_fig.set_facecolor("none")
        self.hist_fig.canvas.setStyleSheet("background-color: transparent;")
        self.hist_fig.canvas.draw()

    def _style_hist_ax(self) -> None:
        """Apply consistent styling to the histogram axes."""
        self.hist_ax.spines["top"].set_visible(False)
        self.hist_ax.spines["right"].set_visible(False)
        self.hist_ax.set_xlabel("intensity [a.u.]")
        ylabel = "density" if self.hist_density else "count"
        self.hist_ax.set_ylabel(ylabel)

    def _fit_hist_ylim(self) -> None:
        """Fit the y-axis of the histogram to its highest bar, from 0.1 in log scale."""
        heights = [p.get_height() for p in self.hist_ax.patches if p.get_height() > 0]
        if heights:
            self.hist_ax.set_ylim(1e-1 if self.hist_log_scale else 0, max(heights))

    def _toggle_hist_log(self) -> None:
        """Toggle log scale on the histogram y-axis (like ThresholdConfigurationWizard)."""
        self.hist_log_scale = self.hist_ax.get_yscale() == "linear"
        self.hist_ax.set_yscale("log" if self.hist_log_scale else "linear")

        if hasattr(self, "_hist_log_action"):
            set_toggle_icon(self._hist_log_action, MDI6.math_log, self.hist_log_scale)

        self._fit_hist_ylim()
        self.hist_fig.tight_layout()
        self.hist_fig.canvas.draw_idle()

    def _toggle_hist_density(self) -> None:
        """Toggle density normalization on the histogram."""
        self._hist_ymax = 0
        self.hist_density = not self.hist_density
        if hasattr(self, "_hist_density_action"):
            set_toggle_icon(
                self._hist_density_action,
                MDI6.chart_bell_curve_cumulative,
                self.hist_density,
            )
        if self.selection:
            self.update_cell_histogram()

    def update_cell_histogram(self) -> None:
        """Redraw the histogram for the currently selected cell."""
        if not hasattr(self, "hist_ax"):
            return
        if not hasattr(self, "track_of_interest"):
            return

        frame = self.current_frame
        image = self.viewer.init_frame  # raw pixels of the displayed channel

        # Find the label id for the selected track in this frame
        label_id = self._label_id_for_track(self.track_of_interest, frame)

        self.hist_ax.clear()

        if label_id is not None and label_id != 0 and frame < len(self.labels):
            mask = self.labels[frame] == label_id
            pixels = image[mask].ravel()

            if pixels.size > 0:
                counts, bin_edges, _ = self.hist_ax.hist(
                    pixels, bins=60, color="tab:blue", alpha=0.7,
                    density=self.hist_density,
                )

                mean_val = np.nanmean(pixels)
                median_val = np.nanmedian(pixels)
                mode_idx = np.argmax(counts)
                mode_val = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

                self.hist_ax.axvline(mean_val, color="tab:blue", lw=1.2, label="mean")
                self.hist_ax.axvline(median_val, color="tab:red", lw=1.2, label="median")
                self.hist_ax.axvline(mode_val, color="purple", lw=1.2, label="mode")
                self.hist_ax.legend(loc="upper right")

                # In density mode, remember the max y across frames
                current_ymax = counts.max()
                if self.hist_density and current_ymax > self._hist_ymax:
                    self._hist_ymax = current_ymax
                if self.hist_density and self._hist_ymax > 0:
                    self.hist_ax.set_ylim(0, self._hist_ymax * 1.05)

        # Sync x-limits with the contrast slider's discovered range
        if hasattr(self.viewer, "contrast_slider"):
            lo, hi = self.viewer.contrast_slider.minimum(), self.viewer.contrast_slider.maximum()
            if lo < hi:
                self.hist_ax.set_xlim(lo, hi)

        self._style_hist_ax()

        # Re-apply log scale if toggled (clear() resets to linear)
        if self.hist_log_scale:
            self.hist_ax.set_yscale("log")
            self._fit_hist_ylim()

        self.hist_fig.tight_layout()
        self.hist_fig.canvas.draw()

    def _label_id_for_track(self, track_id, frame: int) -> Optional[int]:
        """Return the segmentation label id for *track_id* at *frame*."""
        rows = self.df_tracks.loc[self.cell_rows(track_id, frame)]
        if rows.empty:
            return None
        if "class_id" in rows.columns:
            val = rows.iloc[0]["class_id"]
            if pd.notna(val):
                return int(val)
        return None

    def _on_channel_changed(self, index: int) -> None:
        """Update the cell histogram when the viewer channel changes."""
        self._hist_ymax = 0  # reset remembered y-limit for new channel
        if self.selection:
            self.update_cell_histogram()

    def sync_frame(self, value: int) -> None:
        """
        Callback when StackVisualizer frame changes.

        Parameters
        ----------
        value : int
            The new frame index.
        """

        self.current_frame = value
        self.update_frame_logic()
        if self.selection:
            self.update_cell_histogram()

    def cell_rows(self, track_id=None, frame: Optional[int] = None) -> pd.Series:
        """Rows of a cell (the cell of interest by default) at a frame (the current one)."""

        track_id = self.track_of_interest if track_id is None else track_id
        frame = self.current_frame if frame is None else frame
        id_col = extract_identity_col(self.df_tracks)
        return (self.df_tracks[id_col] == track_id) & (self.df_tracks["FRAME"] == frame)

    def cell_values(self) -> List[float]:
        """Values of the chosen signals for the cell of interest, at the current frame."""

        rows = self.cell_rows()
        values = []
        for combo in self.signal_choice_cb:
            signal = combo.currentText()
            if signal != "--":
                values.extend(self.df_tracks.loc[rows, signal].to_numpy())
        return values

    def plot_signals(self) -> None:
        """Plot the chosen signals of all the cells at the current frame, the cell of interest in red."""
        if not hasattr(self, "viewer"):
            return

        signals = [
            combo.currentText()
            for combo in self.signal_choice_cb
            if combo.currentText() != "--"
        ]
        frame_rows = self.df_tracks["FRAME"] == self.current_frame

        def not_nan(values):
            return values[values == values]

        all_yvalues = [not_nan(self.df_tracks[s].to_numpy()) for s in signals]
        current_yvalues = [
            not_nan(self.df_tracks.loc[frame_rows, s].to_numpy()) for s in signals
        ]
        yvalues = self.cell_values()

        self.cell_ax.clear()
        self.cell_ax.set_yscale("log" if self.log_scale else "linear")

        if len(yvalues) > 0:
            try:
                self.cell_ax.boxplot(all_yvalues, showfliers=self.show_fliers)
            except Exception as e:
                logger.error(f"{e}")

            for index, feature in enumerate(current_yvalues):
                x_values_strip = (index + 1) + np.random.normal(
                    0, 0.04, size=len(feature)
                )
                self.cell_ax.plot(
                    x_values_strip,
                    feature,
                    marker="o",
                    linestyle="None",
                    color=tab10.colors[0],
                    alpha=0.1,
                )
            self.plot_red_points(self.cell_ax, yvalues, draw=False)

            range_values = np.concatenate(all_yvalues) if all_yvalues else np.array([])
            # Filter out non-positive values if log scale is active to prevent warnings
            if self.log_scale:
                range_values = range_values[range_values > 0]

            if len(range_values) > 0:
                low, high = np.nanmin(range_values), np.nanmax(range_values)
                margin = 0.03 * (high - low)
                self.value_magnitude = low - margin
                self.non_log_ymin = low - margin
                self.non_log_ymax = high + margin
                if self.cell_ax.get_yscale() == "linear":
                    self.cell_ax.set_ylim(self.non_log_ymin, self.non_log_ymax)
                else:
                    self.cell_ax.set_ylim(self.value_magnitude, self.non_log_ymax)
            else:
                self.value_magnitude = 1
        else:
            self.cell_ax.text(
                0.5,
                0.5,
                "No data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.cell_ax.transAxes,
            )

        self.cell_fcanvas.canvas.draw()

    def plot_red_points(
        self, ax: Axes, yvalues: Optional[List[float]] = None, draw: bool = True
    ) -> None:
        """
        Plot the values of the cell of interest in red, over the box plots.

        Parameters
        ----------
        ax : Axes
            The matplotlib axes.
        yvalues : list of float, optional
            The values, those of :meth:`cell_values` by default.
        draw : bool, optional
            Whether to redraw the plot.
        """
        if yvalues is None:
            yvalues = [v for v in self.cell_values() if v == v]
        x_pos = np.arange(len(yvalues)) + 1
        ax.plot(
            x_pos, yvalues, marker="H", linestyle="None", color=tab10.colors[3], alpha=1
        )
        if draw:
            self.cell_fcanvas.canvas.draw()

    def select_single_cell(self, index: int, timepoint: int) -> None:
        """
        Select a single cell.

        Parameters
        ----------
        index : int
            The cell index.
        timepoint : int
            The timepoint.
        """

        self.enable_correction()

        self.track_of_interest = self.tracks[timepoint][index]
        logger.info(f"You selected cell #{self.track_of_interest}...")
        self.give_cell_information()

        if len(self.cell_ax.lines) > 0:
            # The last line holds the red points of the previous cell.
            self.cell_ax.lines[-1].remove()
            self.plot_red_points(self.cell_ax)
        else:
            self.plot_signals()

        self.highlight_track_of_interest()
        self.draw_frame(self.current_frame)

    def cancel_selection(self) -> None:
        """Cancel selection."""
        super().cancel_selection()
        self.draw_frame(self.current_frame)
        # Clear the histogram
        if hasattr(self, "hist_ax"):
            self.hist_ax.clear()
            self._style_hist_ax()
            self.hist_fig.tight_layout()
            self.hist_fig.canvas.draw()

    def export_measurements(self) -> None:
        """Export measurements to file."""
        logger.info("User interactions: Exporting measurements...")
        if self.normalized_signals:
            self.normalize_features_btn.click()

        cells = self.df_tracks.loc[self.df_tracks["FRAME"] == self.current_frame].copy()
        cells["class"] = cells[self.status_name]

        name = default_dataset_name(
            self.pos, suffix=f"_{str(self.current_frame).zfill(3)}_{self.status_name}"
        )
        save_dataset(self, cells.to_dict("records"), self.exp_dir + name)

    def write_new_event_class(self) -> None:
        """Write new event class."""
        name = self.class_name_le.text()
        self.target_class = f"group_{name}" if name else "group"

        logger.info(
            f"User interactions: Creating new characteristic group '{self.target_class}'"
        )

        if self.target_class in self.df_tracks.columns and not confirm(
            "This characteristic group name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?"
        ):
            return

        self.df_tracks.loc[:, self.target_class] = 0

        self.update_class_cb()

        idx = self.class_choice_cb.findText(self.target_class)
        self.status_name = self.target_class
        self.class_choice_cb.setCurrentIndex(idx)
        self.newClassWidget.close()

    def show_annotation_buttons(self) -> None:
        """Show annotation buttons."""

        for a in self.annotation_btns_to_hide:
            a.show()

        self.time_of_interest_label.setEnabled(True)
        self.time_of_interest_le.setEnabled(True)
        self.correct_btn.setText("submit")
        self.set_correct_action(self.apply_modification)

    def give_cell_information(self) -> None:
        """Display cell information."""

        try:
            cell_selected = f"cell: {self.track_of_interest}\n"
            if self.status_name in self.df_tracks.columns:
                val = self.df_tracks.loc[self.cell_rows(), self.status_name].to_numpy()
                if len(val) > 0:
                    cell_status = f"phenotype: {val[0]}\n"
                else:
                    cell_status = "phenotype: N/A\n"
            else:
                cell_status = f"phenotype: N/A (col '{self.status_name}' missing)\n"
            self.cell_info.setText(cell_selected + cell_status)
        except Exception as e:
            logger.error(f"Error in give_cell_information: {e}")

    def create_new_event_class(self) -> None:
        """Create a new event class."""

        self.newClassWidget, self.class_name_le, _ = new_class_dialog(
            "Create new characteristic group",
            "group name: ",
            "group",
            self.write_new_event_class,
            self.close_without_new_class,
            prefill=False,
        )

    def apply_modification(self) -> None:
        """Apply modification to the cell."""
        if self.time_of_interest_le.text() != "":
            status = int(self.time_of_interest_le.text())
        else:
            status = 0

        logger.info(
            f"User interactions: Reclassifying cell #{self.track_of_interest} at frame {self.current_frame} to status {status}"
        )
        # Record the previous group/status (df still holds the old value here) so the log
        # shows which cells actually changed and how
        rows = self.cell_rows()
        old_status = self.df_tracks.loc[rows, self.status_name].to_numpy()
        old_status = old_status[0] if len(old_status) else None
        self.annotation_log.append(
            f"cell {self.track_of_interest} @ FRAME {self.current_frame}: "
            f"{self.status_name} {old_status}->{status}"
        )
        self.df_tracks.loc[rows, self.status_name] = status

        self.color_states()
        self.extract_scatter_from_trajectories()
        self.give_cell_information()

        self.reset_correction()

        if len(self.selection) > 0:
            self.selection.pop(0)

        self.draw_frame(self.current_frame)

    def assign_color_state(self, state: Union[int, str]) -> list:
        """
        Assign color based on state.

        Parameters
        ----------
        state : int or str
            The state.

        Returns
        -------
        color
            The assigned color.
        """
        try:
            if np.isnan(state):
                state = "nan"
        except TypeError:
            pass
        return self.state_color_map[state]

    def on_scatter_pick(self, event: "matplotlib.backend_bases.PickEvent") -> None:
        """
        Handle pick event on scatter plot.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event.
        """
        idx = closest_picked_index(
            event.ind, self.positions[self.current_frame], event.mouseevent
        )
        if idx is None:
            return

        # Enforce single selection: picking the selected cell again unselects it.
        if len(self.selection) > 0:
            prev_idx, _ = self.selection[0]
            self.cancel_selection()
            if prev_idx == idx:
                return

        self.selection = [[idx, self.current_frame]]
        self._hist_ymax = 0  # reset y-limit memory for new cell
        self.select_single_cell(idx, self.current_frame)
        self.update_cell_histogram()

    def draw_frame(self, framedata: int) -> None:
        """
        Update plot elements at each timestep of the loop.
        Using StackVisualizer overlay update.

        Parameters
        ----------
        framedata : int
            The current frame index.
        """
        self.framedata = framedata

        # The mask is updated by the change_frame of the viewer.
        if self.framedata < len(self.positions):
            positions = self.positions[self.framedata]
            colors = self.colors[self.framedata][:, 0]
        else:
            positions, colors = [], []
        self.viewer.update_overlays(positions=positions, colors=colors)

    def make_status_column(self) -> None:
        """Create the status column."""
        if self.status_name != "state_firstdetection":
            self.df_tracks.loc[:, self.status_name] = 0
            self.color_states()

    def extract_scatter_from_trajectories(self) -> None:
        """Extract scatter data from trajectories."""

        id_col = extract_identity_col(self.df_tracks)
        frames = split_by_frame(self.df_tracks, self.len_movie)
        self.positions = [cells[["POSITION_X", "POSITION_Y"]].to_numpy() for cells in frames]
        # A copy: the colors are written to highlight the selected cell.
        self.colors = [cells[["group_color"]].to_numpy().copy() for cells in frames]
        self.tracks = [cells[id_col].to_numpy() for cells in frames]

    def compute_status_and_colors(self, index: Optional[int] = None) -> None:
        """
        Compute status and colors.

        Parameters
        ----------
        index : int, optional
            The index to compute for. Default is 0.
        """
        self.changed_class()

    def changed_class(self):
        """Handle class change."""
        self.status_name = self.class_choice_cb.currentText()
        if self.status_name != "":
            self.modify()
            self.draw_frame(self.current_frame)

    def update_frame_logic(self):
        """
        Logic to execute when frame changes.
        """
        # Auto-switch track of interest if ID mode
        if "TRACK_ID" not in self.df_tracks.columns and "ID" in self.df_tracks.columns:
            candidates = self.df_tracks.loc[
                self.df_tracks["FRAME"] == self.current_frame, "ID"
            ]
            if not candidates.empty:
                self.track_of_interest = candidates.min()
            self.modify()

        self.draw_frame(self.current_frame)
        self.give_cell_information()
        self.plot_signals()

    def save_trajectories(self):
        """Save trajectories to file."""
        logger.info("Saving trajectories...")
        if self.normalized_signals:
            self.normalize_features_btn.click()
        if self.selection:
            self.cancel_selection()

        self.df_tracks.drop(columns=["", "group_color", "x_anim", "y_anim"], errors="ignore", inplace=True)

        self.df_tracks.to_csv(self.trajectories_path, index=False)
        logger.info("Table successfully exported...")
        self.write_annotation_log(
            "MEASUREMENT ANNOTATION (manual)", f"group: {self.status_name}"
        )

        self.locate_tracks()
        self.changed_class()

    def modify(self):
        """Apply modification (update colors/scatter)."""
        if self.status_name not in self.df_tracks.columns:
            logger.warning(
                f"Column '{self.status_name}' not found in df_tracks. Skipping modify."
            )
            return

        self.color_states()
        self.extract_scatter_from_trajectories()
        self.give_cell_information()
        self.set_correct_action(self.show_annotation_buttons)

    def del_cell(self):
        """Delete the selected cell."""
        logger.info(
            f"User interactions: Deleting cell #{self.track_of_interest} (setting status to 99)"
        )
        self.time_of_interest_le.setEnabled(False)
        self.time_of_interest_le.setText("99")
        self.apply_modification()
