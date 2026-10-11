from typing import Any, Optional

import gc
import os
import numpy as np
import pandas as pd
from PyQt5.QtGui import QCloseEvent, QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QShortcut,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QMainWindow,
)
from PyQt5.QtCore import QSize, Qt
from superqt import QSearchableComboBox

from fonticon_mdi6 import MDI6

from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveMainWindow,
)
from celldetective.gui.gui_utils import ExportPlotBtn
from celldetective.gui.annotator_common import (
    AnnotatorMixin,
    NON_SIGNAL_COLUMNS,
    closest_picked_index,
    color_events,
    complete_event_columns,
    confirm,
    default_dataset_name,
    event_class_columns,
    event_columns,
    experiment_columns,
    fit_feature_scaler,
    icon_button,
    new_class_dialog,
    save_dataset,
    set_toggle_icon,
    warn,
)
from celldetective.gui.base.utils import flush_layout_events
from celldetective import get_logger

logger = get_logger(__name__)


class BaseAnnotator(AnnotatorMixin, CelldetectiveMainWindow):

    def __init__(
        self, parent_window: Optional[QMainWindow] = None, read_config: bool = True
    ) -> None:
        """
        Initialize the BaseAnnotator.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The parent window.
        read_config : bool, optional
            Whether to read the configuration file, default is True.
        """

        super().__init__()
        self.parent_window = parent_window
        self.read_config = read_config

        self.mode = self.parent_window.mode
        self.pos = self.parent_window.parent_window.pos
        self.exp_dir = self.parent_window.exp_dir
        self.n_signals = 3
        self.selection = []
        self.fraction = 1
        self.log_scale = False

        self.instructions_path = self.exp_dir + os.sep.join(
            ["configs", f"signal_annotator_config_{self.mode}.json"]
        )
        self.trajectories_path = self.pos + os.sep.join(
            ["output", "tables", f"trajectories_{self.mode}.csv"]
        )

        self.screen_height = (
            self.parent_window.parent_window.parent_window.screen_height
        )
        self.screen_width = self.parent_window.parent_window.parent_window.screen_width
        self.value_magnitude = 1

        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        self.proceed = True
        self.locate_stack()
        if not self.proceed:
            self.close()
        else:
            if self.read_config:
                self.load_annotator_config()
            self.locate_tracks()
            self._init_base_widgets()

    def _init_base_widgets(self):
        """Initialize base widgets."""
        self._init_base_widgets_left()

    def _init_base_widgets_left(self):
        """Initialize left panel widgets."""

        self.class_label = QLabel("event: ")
        self.class_choice_cb = QComboBox()
        self.class_choice_cb.addItems(self.class_cols)
        self.class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors)

        self.add_class_btn = icon_button(
            MDI6.plus, self.button_select_all, tooltip="Add a new event class"
        )
        self.add_class_btn.clicked.connect(self.create_new_event_class)

        self.del_class_btn = icon_button(
            MDI6.delete, self.button_select_all, tooltip="Delete an event class"
        )
        self.del_class_btn.clicked.connect(self.del_event_class)

        self.cell_info = QLabel("")

        self.correct_btn = icon_button(
            MDI6.redo_variant, self.button_style_sheet, color="white", text="correct"
        )
        self.correct_btn.clicked.connect(self.show_annotation_buttons)
        self.correct_btn.setEnabled(False)

        self.cancel_btn = QPushButton("cancel")
        self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
        self.cancel_btn.setShortcut(QKeySequence("Esc"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_selection)

        self._init_cell_fig_widgets()

        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet(self.button_style_sheet)
        self.save_btn.clicked.connect(self.save_trajectories)

        self.export_btn = icon_button(MDI6.export, self.button_select_all, icon_size=25)
        self.export_btn.clicked.connect(self.export_signals)

    def _init_cell_fig_widgets(self):
        """Initialize cell figure widgets."""

        self.generate_signal_choices()
        self.create_cell_signal_canvas()
        self.cell_fcanvas.setMinimumHeight(int(0.2 * self.screen_height))
        # The plot takes the height the left panel has left, but no more than
        # a little under its width: a tall window made it a narrow strip.
        self.cell_fcanvas.limit_aspect(0.85)

        self.outliers_check = QCheckBox("Show outliers")
        self.outliers_check.toggled.connect(self.show_outliers)

        self.normalize_features_btn = icon_button(
            MDI6.arrow_collapse_vertical, self.button_select_all, icon_size=25
        )
        self.normalize_features_btn.setFixedSize(QSize(30, 30))
        self.normalize_features_btn.clicked.connect(self.normalize_features)
        self.normalized_signals = False

        self.log_btn = icon_button(MDI6.math_log, self.button_select_all)
        self.log_btn.clicked.connect(self.switch_to_log)

        self.export_plot_btn = ExportPlotBtn(self.cell_fig, export_dir=self.exp_dir)

    def init_class_selection_block(self):
        """Initialize the class selection block."""

        self.class_hbox = QHBoxLayout()
        self.class_hbox.setContentsMargins(0, 0, 0, 0)

        self.class_hbox.addWidget(self.class_label, 25)
        self.class_hbox.addWidget(self.class_choice_cb, 70)
        self.class_hbox.addWidget(self.add_class_btn, 5)
        self.class_hbox.addWidget(self.del_class_btn, 5)

    def init_options_block(self):
        """Initialize the options block."""
        self.options_hbox = QHBoxLayout()
        self.options_hbox.setContentsMargins(0, 0, 0, 0)

    def init_correction_block(self):
        """Initialize the correction block."""
        self.action_hbox = QHBoxLayout()
        self.action_hbox.setContentsMargins(0, 0, 0, 0)

        self.action_hbox.addWidget(self.correct_btn)
        self.action_hbox.addWidget(self.cancel_btn)

    def init_plot_buttons_block(self):
        """Initialize the plot buttons block."""
        self.plot_buttons_hbox = QHBoxLayout()
        self.plot_buttons_hbox.setContentsMargins(0, 0, 0, 0)
        self.plot_buttons_hbox.addWidget(QLabel(""), 90)
        self.plot_buttons_hbox.addWidget(self.outliers_check, 5)
        self.plot_buttons_hbox.addWidget(self.normalize_features_btn, 5)
        self.plot_buttons_hbox.addWidget(self.log_btn, 5)
        self.plot_buttons_hbox.addWidget(self.export_plot_btn, 5)

    def init_save_btn_block(self):
        """Initialize the save button block."""

        self.btn_hbox = QHBoxLayout()
        self.btn_hbox.setContentsMargins(0, 10, 0, 0)
        self.btn_hbox.addWidget(self.save_btn, 90)
        self.btn_hbox.addWidget(self.export_btn, 10)

    def populate_window(self):
        """
        Create the multibox design.

        """

        # Main layout
        self.button_widget = CelldetectiveWidget()
        self.main_layout = QHBoxLayout()
        self.button_widget.setLayout(self.main_layout)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Left panel
        self.left_panel = QVBoxLayout()
        self.left_panel.setContentsMargins(30, 5, 30, 5)
        self.left_panel.setSpacing(3)

        self.init_class_selection_block()
        # The rows keep their natural height (stretch 0): only the plot takes
        # the room left, instead of it opening gaps between the rows.
        self.left_panel.addLayout(self.class_hbox)
        self.left_panel.addWidget(self.cell_info)

        self.init_options_block()
        self.init_correction_block()
        self.left_panel.addLayout(self.options_hbox)
        self.left_panel.addLayout(self.action_hbox)

        # Outweighs the stretch above the save button: the plot takes the room
        # until its height is capped, and only the rest goes there.
        self.left_panel.addWidget(self.cell_fcanvas, 100)

        self.init_plot_buttons_block()
        self.left_panel.addLayout(self.plot_buttons_hbox)

        signal_choice_vbox = QVBoxLayout()
        signal_choice_vbox.setContentsMargins(30, 0, 30, 0)
        for lbl, cb in zip(self.signal_choice_label, self.signal_choice_cb):
            hlayout = QHBoxLayout()
            hlayout.addWidget(lbl, 20)
            hlayout.addWidget(cb, 75)
            signal_choice_vbox.addLayout(hlayout)

        self.left_panel.addLayout(signal_choice_vbox, 15)

        # Room the capped plot does not take collects above the save button,
        # rather than between the rows.
        self.left_panel.addStretch(1)

        self.init_save_btn_block()
        self.left_panel.addLayout(self.btn_hbox)

        # Right panel
        self.right_panel = QVBoxLayout()

        # add left/right to main layout
        self.main_layout.addLayout(self.left_panel, 35)
        self.main_layout.addLayout(self.right_panel, 65)
        self.button_widget.adjustSize()

        self.setCentralWidget(self.button_widget)
        self.resize(int(0.8 * self.screen_width), int(0.8 * self.screen_height))

        self.del_shortcut = QShortcut(Qt.Key_Delete, self)
        self.del_shortcut.activated.connect(self.shortcut_suppr)
        self.del_shortcut.setEnabled(False)

        flush_layout_events(self)

    def generate_signal_choices(self):
        """Generate signal choice combos."""

        self.signal_choice_cb = [QSearchableComboBox() for _ in range(self.n_signals)]
        self.signal_choice_label = [
            QLabel(f"signal {i + 1}: ") for i in range(self.n_signals)
        ]

        excluded = set(NON_SIGNAL_COLUMNS).union(experiment_columns(self.exp_dir))
        signals = [
            c
            for c in self.df_tracks.columns
            if pd.api.types.is_numeric_dtype(self.df_tracks[c]) and c not in excluded
        ]

        for i, cb in enumerate(self.signal_choice_cb):
            cb.addItems(["--"] + signals)
            cb.setCurrentIndex(i + 1 if i + 1 < cb.count() else 0)
            cb.currentIndexChanged.connect(self.plot_signals)

    def on_scatter_pick(self, event: Any) -> None:
        """
        Handle scatter plot pick events.

        Parameters
        ----------
        event : PickEvent
            The pick event.
        """
        self.set_correct_action(self.show_annotation_buttons)

        index = closest_picked_index(
            event.ind, self.positions[self.framedata], event.mouseevent
        )
        if index is None:
            return
        if not self.selection:
            self.selection.append([index, self.framedata])
            self.select_single_cell(index, self.framedata)
        elif len(self.selection) == 1:
            self.cancel_btn.click()

    def locate_stack(self):
        """Locate the target movie."""

        experiment = self.parent_window.parent_window
        if not self.find_movie(experiment.movie_prefix, experiment.len_movie):
            warn(
                "No movie is detected in the experiment folder.\nPlease check the stack prefix..."
            )
            self.proceed = False
            self.close()

    def locate_tracks(self):
        """Locate the trajectories file."""

        if not os.path.exists(self.trajectories_path):
            warn("The trajectories cannot be detected.")
            self.close()
            return

        # Reset the per-position record of manual modifications
        self.annotation_log = []

        # Load and prep tracks
        self.df_tracks = pd.read_csv(self.trajectories_path)
        self.df_tracks = self.df_tracks.sort_values(by=["TRACK_ID", "FRAME"])
        self.df_tracks.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.class_cols = event_class_columns(self.df_tracks.columns)
        self.class_name = self.class_cols[0] if self.class_cols else "class"
        self.time_name, self.status_name = event_columns(self.class_name)

        complete_event_columns(
            self.df_tracks,
            self.class_name,
            self.time_name,
            self.status_name,
            make_status=self.make_status_column,
        )
        color_events(self.df_tracks, self.class_name, self.status_name)

        self.df_tracks = self.df_tracks.dropna(subset=["POSITION_X", "POSITION_Y"])
        self.df_tracks["x_anim"] = (self.df_tracks["POSITION_X"] * self.fraction).astype(int)
        self.df_tracks["y_anim"] = (self.df_tracks["POSITION_Y"] * self.fraction).astype(int)

        self.extract_scatter_from_trajectories()
        self.track_of_interest = self.df_tracks["TRACK_ID"].min()
        self.locate_track_of_interest()

        self.columns_to_rescale, self.feature_scaler = fit_feature_scaler(
            self.df_tracks,
            excluded=list(NON_SIGNAL_COLUMNS)
            + self.class_cols
            + experiment_columns(self.exp_dir),
            excluded_prefixes=("t_", "status_"),
        )

    def highlight_track_of_interest(self) -> None:
        """Paint the track of interest in lime, keeping its colors to restore them."""

        self.locate_track_of_interest()
        self.previous_color = []
        for t, idx in zip(self.loc_t, self.loc_idx):
            self.previous_color.append(self.colors[t][idx].copy())
            self.colors[t][idx] = "lime"

    def locate_track_of_interest(self) -> None:
        """Find the track of interest in each frame: ``loc_t``, ``loc_idx``."""

        self.loc_t = []
        self.loc_idx = []
        for t, tracks in enumerate(self.tracks):
            indices = np.where(tracks == self.track_of_interest)[0]
            if len(indices) > 0:
                self.loc_t.append(t)
                self.loc_idx.append(indices[0])

    def del_event_class(self):
        """Delete an event class."""

        class_to_delete = self.class_choice_cb.currentText()
        if not confirm(
            f"You are about to delete the class {class_to_delete}. The associated time and\nstatus will also be deleted. Do you still want to proceed?"
        ):
            return
        self.df_tracks = self.df_tracks.drop(
            columns=[
                class_to_delete,
                class_to_delete.replace("class", "t"),
                class_to_delete.replace("class", "status"),
            ],
            errors="ignore",
        )
        self.class_choice_cb.removeItem(self.class_choice_cb.findText(class_to_delete))

    def normalize_features(self):
        """Normalize or denormalize features."""

        x = self.df_tracks[self.columns_to_rescale].values
        if self.normalized_signals:
            x = self.feature_scaler.inverse_transform(x)
        else:
            x = self.feature_scaler.transform(x)
        self.df_tracks[self.columns_to_rescale] = x
        self.plot_signals()
        self.normalized_signals = not self.normalized_signals
        set_toggle_icon(
            self.normalize_features_btn,
            MDI6.arrow_collapse_vertical,
            self.normalized_signals,
            size=25,
        )

    def show_outliers(self):
        """Toggle outliers visibility."""
        self.show_fliers = self.outliers_check.isChecked()
        self.plot_signals()

    def export_signals(self):
        """Export signals to a file."""

        if self.normalized_signals:
            self.normalize_features_btn.click()

        training_set = []
        for _, track in self.df_tracks.groupby("TRACK_ID"):
            signals = {c: track[c].to_numpy() for c in track.columns}
            signals["time_of_interest"] = track[self.time_name].to_numpy()[0]
            signals["class"] = track[self.class_name].to_numpy()[0]
            training_set.append(signals)

        save_dataset(self, training_set, self.exp_dir + default_dataset_name(self.pos))

    def create_new_event_class(self):
        """Create a new event class."""

        self.newClassWidget, self.class_name_le, self.class_option_rb = new_class_dialog(
            "Create new event class",
            "event name: ",
            "event",
            self.write_new_event_class,
            self.close_without_new_class,
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle close event.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        gc.collect()
        super().closeEvent(event)

    def save_trajectories(self):
        """Save trajectories (not implemented)."""
        # specific to event annotator
        logger.info(
            "Save trajectory function not implemented for BaseAnnotator class..."
        )

    def cancel_selection(self):
        """Cancel the current selection."""
        self.reset_correction()

        if self.selection:
            self.selection.pop(0)

        try:
            for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
                self.colors[t][idx, 0] = self.previous_color[k][0]
        except Exception as e:
            logger.debug(f"Could not revert colors on cancel: {e}")
