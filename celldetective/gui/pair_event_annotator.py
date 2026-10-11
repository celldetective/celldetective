from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QComboBox,
    QLabel,
    QRadioButton,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QButtonGroup,
    QWidget,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence
from celldetective.gui.base.components import (
    CelldetectiveMainWindow,
    CelldetectiveWidget,
)
from celldetective.gui.base.utils import (
    center_window,
    flush_layout_events,
)
from superqt import QSearchableComboBox
from celldetective.gui.annotator_common import (
    AnimationMixin,
    AnnotatorMixin,
    EventAnnotationMixin,
    NON_SIGNAL_COLUMNS,
    PAIR_COLUMNS,
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
    prefill_class,
    save_dataset,
    signal_limits,
    split_by_frame,
    warn,
    write_event_status,
)
from celldetective.gui.gui_utils import color_from_status
import numpy as np
from fonticon_mdi6 import MDI6
import os
from matplotlib.cm import tab10
import pandas as pd
from functools import partial
from pandas.api.types import is_numeric_dtype
import logging

logger = logging.getLogger("celldetective")

# Signal sources of the signal choices, as ids of their button groups.
REFERENCE, NEIGHBOR, PAIR = 0, 1, 2


class PairEventAnnotator(
    EventAnnotationMixin, AnimationMixin, AnnotatorMixin, CelldetectiveMainWindow
):
    """
    UI to explore cell interactions.

    """

    # The lines to the neighbors drawn on the current frame.
    lines_list = ()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the annotator.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget. Default is None.

        Raises
        ------
        ValueError
            If the movie, the tables, the tracks or the neighborhoods are
            missing: the caller shows the message.
        """
        super().__init__()
        self.parent_window = parent
        self.setWindowTitle("Interaction Annotator")

        experiment = self.parent_window.parent_window
        self.pos = experiment.pos
        self.exp_dir = self.parent_window.exp_dir
        self.populations = experiment.populations

        self.n_signals = 3
        self.annotation_log = []

        self.reference_selection = []
        self.neighbor_selection = []
        self.pair_selection = []
        self.neigh_cell_loc_t = []
        self.neigh_cell_loc_idx = []
        self.selected_population = None

        self.reference_track_of_interest = None
        self.neighbor_track_of_interest = None
        self.value_magnitude = 1

        self.cols_to_remove = (
            list(NON_SIGNAL_COLUMNS)
            + list(PAIR_COLUMNS)
            + experiment_columns(self.exp_dir)
        )

        # Read instructions from target block for now...
        self.mode = "neighborhood"
        self.instructions_path = self.exp_dir + os.sep.join(
            ["configs", "signal_annotator_config_neighborhood.json"]
        )

        center_window(self)

        if not self.find_movie(experiment.movie_prefix, experiment.len_movie):
            raise ValueError(
                "No movie is detected in the experiment folder.\nPlease check the stack prefix..."
            )
        self.load_annotator_config()

        self.locate_all_tracks()
        self.extract_scatter_from_trajectories()
        self.locate_neighborhoods()
        self.locate_relative_tracks()

        self.prepare_stack()

        self.generate_signal_choices()
        self.frame_lbl = QLabel("frame: ")
        self.looped_animation()
        self.create_cell_signal_canvas()

        # Widget settings. Before populate_widget, which sizes the window from
        # the screen: set after it, the annotator failed to open.
        self.screen_height = experiment.parent_window.screen_height
        self.screen_width = experiment.parent_window.screen_width
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        self.populate_widget()
        self.fill_signal_choices()

        self.reference_pop_option_buttons[0].setChecked(True)
        self.neighbor_pop_option_buttons[1].setChecked(True)
        self.relative_pop_option_buttons[2].setChecked(True)

        self.give_reference_cell_information()
        self.give_neighbor_cell_information()
        self.give_pair_information()

        # The plot takes the height the left panel has left, but no more than
        # a little under its width, as in the other annotators.
        self.cell_fcanvas.limit_aspect(0.85)

    def locate_neighborhoods(self) -> None:
        """
        List the neighborhoods computed for the populations.

        Raises
        ------
        ValueError
            If there is none.
        """

        self.neighborhood_cols = [
            f"{pop}_ref_{c}"
            for pop, df in self.dataframes.items()
            if df is not None
            for c in df.columns
            if c.startswith("neighborhood")
        ]
        logger.debug(f"The following neighborhoods were detected: neighborhood_cols={self.neighborhood_cols}...")
        if not self.neighborhood_cols:
            raise ValueError(
                "No neighborhoods detected. Please compute neighborhoods first."
            )

    def populate_widget(self):
        """
        Create the multibox design.

        """

        self.button_widget = CelldetectiveWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(30)

        self.button_widget.setLayout(main_layout)

        main_layout.setContentsMargins(30, 30, 30, 30)
        self.left_panel = QVBoxLayout()
        self.left_panel.setContentsMargins(5, 5, 5, 5)
        self.left_panel.setSpacing(5)

        self.right_panel = QVBoxLayout()
        self.right_panel.setSpacing(0)
        self.right_panel.setContentsMargins(5, 5, 5, 5)

        # NEIGHBORHOOD
        neigh_hbox = QHBoxLayout()
        neigh_hbox.setContentsMargins(0, 0, 0, 0)
        neigh_hbox.addWidget(QLabel("neighborhood: "), 25)
        self.neighborhood_choice_cb = QComboBox()
        self.neighborhood_choice_cb.addItems(self.neighborhood_cols)
        self.neighborhood_choice_cb.setCurrentIndex(0)
        neigh_hbox.addWidget(self.neighborhood_choice_cb, 75)
        self.left_panel.addLayout(neigh_hbox)

        self.reference_cell_info = QLabel("")
        self.reference_cell_info.setStyleSheet("color: #1f77b4;")
        self.pair_info = QLabel("")
        self.neighbor_cell_info = QLabel("")
        self.neighbor_cell_info.setStyleSheet("color: #d62728;")

        class_hbox = QHBoxLayout()
        class_hbox.addWidget(QLabel("interaction event: "), 25)

        subclass_hbox = QHBoxLayout()
        self.relative_class_choice_cb = QComboBox()
        self.relative_class_choice_cb.addItems(self.relative_class_cols)
        self.relative_class_choice_cb.currentIndexChanged.connect(
            self.compute_status_and_colors_pair
        )
        self.relative_class_choice_cb.setCurrentIndex(0)

        subclass_hbox.addWidget(self.relative_class_choice_cb, 90)

        self.set_reference_and_neighbor_populations()
        self.neighborhood_choice_cb.currentIndexChanged.connect(
            self.neighborhood_changed
        )
        self.compute_status_and_colors_pair()

        self.relative_add_class_btn = icon_button(
            MDI6.plus,
            self.button_select_all,
            tooltip="Add a new interaction event class",
        )
        self.relative_add_class_btn.clicked.connect(
            self.create_new_relative_event_class
        )
        subclass_hbox.addWidget(self.relative_add_class_btn, 5)

        self.relative_del_class_btn = icon_button(
            MDI6.delete,
            self.button_select_all,
            tooltip="Delete an interaction event class",
        )
        self.relative_del_class_btn.clicked.connect(self.del_relative_event_class)
        subclass_hbox.addWidget(self.relative_del_class_btn, 5)
        class_hbox.addLayout(subclass_hbox, 75)
        self.left_panel.addLayout(class_hbox)

        self.cell_events_hbox = QHBoxLayout()
        self.cell_events_hbox.setContentsMargins(0, 0, 0, 0)
        self.cell_events_hbox.addWidget(QLabel("Event: "), 25)
        self.reference_event_choice_cb = QComboBox()
        self.cell_events_hbox.addWidget(self.reference_event_choice_cb, 75)

        self.neigh_cell_events_hbox = QHBoxLayout()
        self.neigh_cell_events_hbox.setContentsMargins(0, 0, 0, 0)
        self.neigh_lab = QLabel("Event: ")
        self.neigh_cell_events_hbox.addWidget(self.neigh_lab, 25)
        self.neighbor_event_choice_cb = QComboBox()
        self.neigh_cell_events_hbox.addWidget(self.neighbor_event_choice_cb, 75)
        self.fill_class_cbs()

        # Text information about selected cells
        self.cell_info_hbox = QHBoxLayout()
        self.cell_info_hbox.setContentsMargins(30, 0, 30, 0)

        reference_layout = QVBoxLayout()
        reference_layout.addWidget(self.reference_cell_info)
        reference_layout.addLayout(self.cell_events_hbox)

        neighbor_layout = QVBoxLayout()
        neighbor_layout.addWidget(self.neighbor_cell_info, alignment=Qt.AlignRight)
        neighbor_layout.addLayout(self.neigh_cell_events_hbox)

        self.cell_info_hbox.addLayout(reference_layout, 33)
        self.cell_info_hbox.addWidget(self.pair_info, 33, alignment=Qt.AlignCenter)
        self.cell_info_hbox.addLayout(neighbor_layout, 33)

        self.left_panel.addLayout(self.cell_info_hbox)

        # Annotation buttons
        self.init_event_buttons(remove_label="mark for\nsuppression")
        options_hbox = QHBoxLayout()
        options_hbox.setContentsMargins(150, 0, 50, 0)
        for button in self.class_options:
            options_hbox.addWidget(button, 25)
        self.left_panel.addLayout(options_hbox)

        time_option_hbox = QHBoxLayout()
        time_option_hbox.setContentsMargins(100, 0, 100, 0)
        time_option_hbox.addWidget(self.time_of_interest_label, 30)
        time_option_hbox.addWidget(self.time_of_interest_le, 70)
        self.left_panel.addLayout(time_option_hbox)

        main_action_hbox = QHBoxLayout()
        main_action_hbox.setContentsMargins(100, 0, 100, 0)
        self.correct_btn = icon_button(
            MDI6.redo_variant, self.button_style_sheet, color="white", text="correct"
        )
        self.correct_btn.clicked.connect(self.show_annotation_buttons)
        self.correct_btn.setEnabled(False)
        main_action_hbox.addWidget(self.correct_btn)

        self.cancel_btn = QPushButton("cancel")
        self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
        self.cancel_btn.setShortcut(QKeySequence("Esc"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_selection)
        main_action_hbox.addWidget(self.cancel_btn)
        self.left_panel.addLayout(main_action_hbox)

        self.hide_annotation_buttons()

        self.del_shortcut = self.add_selection_shortcut(Qt.Key_Delete, self.shortcut_suppr)
        self.no_event_shortcut = self.add_selection_shortcut(
            QKeySequence("n"), self.shortcut_no_event
        )

        # Cell signals
        # Outweighs the stretch above the save button: the plot takes the room
        # until its height is capped, and only the rest goes there.
        self.left_panel.addWidget(self.cell_fcanvas, 100)

        self.normalize_features_btn = icon_button(
            MDI6.arrow_collapse_vertical, self.button_select_all, icon_size=25
        )
        self.normalize_features_btn.setFixedSize(QSize(30, 30))
        self.normalize_features_btn.clicked.connect(self.normalize_features)
        self.normalized_signals = False

        self.log_btn = icon_button(MDI6.math_log, self.button_select_all)
        self.log_btn.clicked.connect(self.switch_to_log)

        self.log_toggles = [self.log_btn]
        self.normalize_toggles = [self.normalize_features_btn]

        # In the toolbar of the plot rather than on a row of their own, as in
        # the phenotype annotator: the row cost the plot its height in a short
        # window.
        toolbar = getattr(self.cell_fcanvas, "toolbar", None)
        if toolbar is not None:
            toolbar.addSeparator()
            toolbar.addWidget(self.normalize_features_btn)
            toolbar.addWidget(self.log_btn)
        else:
            plot_buttons_hbox = QHBoxLayout()
            plot_buttons_hbox.setContentsMargins(0, 0, 0, 0)
            plot_buttons_hbox.addStretch(1)
            plot_buttons_hbox.addWidget(self.normalize_features_btn)
            plot_buttons_hbox.addWidget(self.log_btn)
            self.left_panel.addLayout(plot_buttons_hbox)

        signal_choice_grid = QVBoxLayout()
        signal_choice_grid.setContentsMargins(30, 0, 70, 5)

        header_layout = QHBoxLayout()
        for header in ("reference", "neighbor", "pair"):
            header_layout.addWidget(QLabel(header), 23, alignment=Qt.AlignCenter)
        header_layout.addWidget(QLabel(""), 30, alignment=Qt.AlignCenter)
        signal_choice_grid.addLayout(header_layout)

        for i in range(self.n_signals):
            h_layout = QHBoxLayout()
            if i == (self.n_signals - 1):
                h_layout.setContentsMargins(0, 0, 0, 0)
            for buttons in (
                self.reference_pop_option_buttons,
                self.neighbor_pop_option_buttons,
                self.relative_pop_option_buttons,
            ):
                h_layout.addWidget(buttons[i], 23, alignment=Qt.AlignCenter)
            h_layout.addWidget(self.signal_choices[i], 30)
            signal_choice_grid.addLayout(h_layout)

        self.left_panel.addLayout(signal_choice_grid)

        btn_hbox = QHBoxLayout()
        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet(self.button_style_sheet)
        self.save_btn.clicked.connect(self.save_trajectories)
        btn_hbox.addWidget(self.save_btn, 90)

        self.export_btn = icon_button(MDI6.export, self.button_select_all, icon_size=25)
        self.export_btn.clicked.connect(self.export_signals)
        btn_hbox.addWidget(self.export_btn, 10)
        # Room the capped plot does not take collects above the save button,
        # rather than between the rows.
        self.left_panel.addStretch(1)
        self.left_panel.addLayout(btn_hbox)

        # Animation
        self.right_panel.addLayout(self.build_animation_controls(), 5)
        self.right_panel.addWidget(self.fcanvas, 90)
        contrast_controls = self.build_contrast_controls()
        if contrast_controls is not None:
            self.right_panel.addLayout(contrast_controls, 5)

        main_layout.addLayout(self.left_panel, 35)
        main_layout.addLayout(self.right_panel, 65)
        self.button_widget.adjustSize()
        self.compute_status_and_colors_reference()
        self.compute_status_and_colors_neighbor()

        self.extract_relevant_events()

        self.setCentralWidget(self.button_widget)
        self.show()
        self.resize(int(0.8 * self.screen_width), int(0.8 * self.screen_height))
        flush_layout_events(self)

    # ------------------------------------------------------------------
    # Rows of the pair table
    # ------------------------------------------------------------------

    def neighborhood_rows(self) -> pd.Series:
        """Pairs in the current neighborhood, from the reference population."""

        df = self.df_relative
        return (~df["status_" + self.current_neighborhood].isnull()) & (
            df["reference_population"] == self.reference_population
        )

    def population_pair_rows(self) -> pd.Series:
        """Pairs between the reference and the neighbor populations."""

        df = self.df_relative
        return (df["reference_population"] == self.reference_population) & (
            df["neighbor_population"] == self.neighbor_population
        )

    def selected_pair_rows(self, in_neighborhood: bool = True) -> pd.Series:
        """Rows of the selected pair, in the current neighborhood or not."""

        df = self.df_relative
        rows = (
            (df["REFERENCE_ID"] == self.reference_track_of_interest)
            & (df["NEIGHBOR_ID"] == self.neighbor_track_of_interest)
            & self.population_pair_rows()
        )
        if in_neighborhood:
            rows &= ~df["status_" + self.current_neighborhood].isnull()
        return rows

    # ------------------------------------------------------------------
    # Event classes
    # ------------------------------------------------------------------

    def fill_class_cbs(self):
        """Fill class combo boxes."""

        for combo, population, slot in (
            (
                self.reference_event_choice_cb,
                self.reference_population,
                self.compute_status_and_colors_reference,
            ),
            (
                self.neighbor_event_choice_cb,
                self.neighbor_population,
                self.compute_status_and_colors_neighbor,
            ),
        ):
            try:
                combo.disconnect()
            except TypeError:
                pass  # No connection to remove.
            combo.clear()
            combo.addItems(event_class_columns(self.dataframes[population].columns))
            combo.currentIndexChanged.connect(slot)

    def del_relative_event_class(self):
        """Delete relative event class."""

        class_to_delete = self.relative_class_choice_cb.currentText()
        if not confirm(
            f"You are about to delete event class {class_to_delete}. The associated time and\nstatus will also be deleted. Do you still want to proceed?"
        ):
            return
        self.df_relative = self.df_relative.drop(
            columns=[
                class_to_delete,
                class_to_delete.replace("class", "t"),
                class_to_delete.replace("class", "status"),
            ],
            errors="ignore",
        )
        self.relative_class_choice_cb.removeItem(
            self.relative_class_choice_cb.findText(class_to_delete)
        )

    def create_new_relative_event_class(self):
        """Create a new relative event class."""

        (
            self.newClassWidget,
            self.relative_class_name_le,
            self.class_option_rb,
        ) = new_class_dialog(
            "Create new event class",
            "event name: ",
            "event",
            self.write_new_relative_event_class,
            self.close_without_new_class,
        )

    def write_new_relative_event_class(self):
        """Write new relative event class."""

        name = self.relative_class_name_le.text()
        if name:
            class_name, time_name, status_name = (
                f"class_{name}",
                f"t0_{name}",
                f"status_{name}",
            )
        else:
            class_name, time_name, status_name = "class", "t0", "status"

        if class_name in self.df_relative.columns and not confirm(
            "This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?"
        ):
            return

        fill_option = prefill_class(self.class_option_rb)
        rows = self.neighborhood_rows()
        self.df_relative.loc[rows, class_name] = fill_option
        self.df_relative.loc[rows, time_name] = 0.1 if fill_option == 0 else -1
        # A rewritten class keeps its status column, which selecting it would
        # not remake: derive it from the new class and time now.
        write_event_status(
            self.df_relative,
            class_name,
            time_name,
            status_name,
            keys=PAIR_COLUMNS,
            rows=rows,
        )

        # Selecting the class in the list sets the pair columns, so that the
        # annotations go to the class shown.
        self.extract_relevant_events()
        index = self.relative_class_choice_cb.findText(class_name)
        if index >= 0:
            self.relative_class_choice_cb.setCurrentIndex(index)

        self.newClassWidget.close()

    def compute_cell_status(self, role: str, combo: QComboBox) -> bool:
        """
        Prepare the event class chosen for the reference or the neighbor cells:
        sets ``<role>_class_name``, ``<role>_time_name`` and ``<role>_status_name``.

        Parameters
        ----------
        role : str
            "reference" or "neighbor".
        combo : QComboBox
            The event choice of the role.

        Returns
        -------
        bool
            Whether an event class is chosen.
        """

        population = getattr(self, f"{role}_population")
        df = self.dataframes[population]
        class_name = combo.currentText()
        setattr(self, f"{role}_class_name", class_name)
        if class_name == "":
            return False

        time_name, status_name = event_columns(class_name)
        setattr(self, f"{role}_time_name", time_name)
        setattr(self, f"{role}_status_name", status_name)

        # The status is made only if missing, not to erase static classifications.
        complete_event_columns(
            df,
            class_name,
            time_name,
            status_name,
            make_status=lambda: write_event_status(
                df, class_name, time_name, status_name
            ),
        )
        color_events(df, class_name, status_name)
        self.refresh_scatter_from_trajectories(population)
        return True

    def compute_status_and_colors_reference(self):
        """Compute status and colors for reference."""

        if not self.compute_cell_status("reference", self.reference_event_choice_cb):
            return
        if self.reference_track_of_interest is not None:
            df_reference = self.dataframes[self.reference_population]
            t_reference = df_reference.loc[
                df_reference["TRACK_ID"] == self.reference_track_of_interest,
                self.reference_time_name,
            ].values
            if len(t_reference) > 0:
                self.set_time_line(self.line_dt_reference, t_reference[0])
                self.cell_fcanvas.canvas.draw()

    def compute_status_and_colors_neighbor(self):
        """Compute status and colors for neighbor."""
        self.compute_cell_status("neighbor", self.neighbor_event_choice_cb)

    def compute_status_and_colors_pair(self):
        """Compute status and colors for pair."""

        self.pair_class_name = self.relative_class_choice_cb.currentText()
        if self.pair_class_name == "":
            return

        self.pair_time_name, self.pair_status_name = event_columns(self.pair_class_name)
        if self.pair_time_name.startswith("t_"):
            # The pair tables name the time t0_<event>, or t_<event> as the cells.
            legacy_time = "t0_" + self.pair_time_name[len("t_") :]
            if legacy_time in self.df_relative.columns:
                self.pair_time_name = legacy_time

        # The status is made only if missing, not to erase static classifications.
        complete_event_columns(
            self.df_relative,
            self.pair_class_name,
            self.pair_time_name,
            self.pair_status_name,
            make_status=self.make_relative_status_column,
        )
        color_events(self.df_relative, self.pair_class_name, self.pair_status_name)

        self.extract_scatter_from_lines()
        self.give_pair_information()
        self.plot_signals()

    def make_relative_status_column(self):
        """Make relative status column."""

        write_event_status(
            self.df_relative,
            self.pair_class_name,
            self.pair_time_name,
            self.pair_status_name,
            keys=PAIR_COLUMNS,
            rows=self.neighborhood_rows(),
        )

    def extract_relevant_events(self):
        """List the pair event classes holding values for the current populations."""

        try:
            self.relative_class_choice_cb.disconnect()
        except TypeError:
            pass  # No connection to remove.

        self.relative_class_choice_cb.clear()

        self.relative_class_cols = event_class_columns(self.df_relative.columns)
        rows = self.population_pair_rows()
        filled_class_cols = [
            c
            for c in self.relative_class_cols
            if self.df_relative.loc[rows, c].notnull().any()
        ]

        if filled_class_cols:
            self.relative_class_choice_cb.currentIndexChanged.connect(
                self.compute_status_and_colors_pair
            )
            self.relative_class_choice_cb.addItems(filled_class_cols)
            self.relative_class_choice_cb.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Selection and annotation
    # ------------------------------------------------------------------

    def cancel_selection(self):
        """Cancel selection."""

        logger.debug("Canceling selection...")

        self.reset_correction()

        self.reference_selection = []
        self.reference_track_of_interest = None
        self.give_reference_cell_information()

        if len(self.pair_selection) > 0:
            self.cancel_pair_selection()

        # Restore the colors of all the cells.
        _, _, neighbor_colors, initial_neighbor_colors = self.get_neighbor_sets()
        _, _, _, reference_colors, initial_reference_colors = self.get_reference_sets()
        for colors, initial_colors in (
            (neighbor_colors, initial_neighbor_colors),
            (reference_colors, initial_reference_colors),
        ):
            for frame_colors, frame_initial_colors in zip(colors, initial_colors):
                frame_colors[:] = frame_initial_colors

        # Their lines go at the next frame drawn.
        self.lines_data = {}
        self.selected_population = None

        for button in self.reference_pop_option_buttons:
            button.setEnabled(False)

        self.plot_signals()

    def cancel_pair_selection(self):
        """Cancel pair selection."""

        # Unselect and recolor pair line
        self.pair_selection = []
        for t, status_colors in enumerate(self.lines_colors_status):
            selected = status_colors[:, 2] == "lime"
            status_colors[selected, 2] = self.initial_lines_colors_status[t][selected, 2]
            self.lines_colors_class[t][selected, 2] = self.initial_lines_colors_class[t][selected, 2]

        # Unselect and recolor neighbor
        self.neighbor_selection = []
        self.neighbor_track_of_interest = None
        _, _, colors_neigh, _ = self.get_neighbor_sets()
        for k, (t, idx) in enumerate(
            zip(self.neigh_cell_loc_t, self.neigh_cell_loc_idx)
        ):
            colors_neigh[t][idx, :2] = self.neigh_previous_color[k][:2]
        self.give_neighbor_cell_information()
        self.give_pair_information()

        cleared = False
        for i in range(self.n_signals):
            self.neighbor_pop_option_buttons[i].setEnabled(False)
            self.relative_pop_option_buttons[i].setEnabled(False)
            if self.signal_pop_button_groups[i].checkedId() != REFERENCE:
                self.lines[i].set_xdata([])
                self.lines[i].set_ydata([])
                self.lines[i].set_label("")
                cleared = True
        if cleared:
            self.clear_time_lines(reference=self.reference_track_of_interest is None)

        self.reset_correction()

    def current_annotation(self) -> Tuple[int, float]:
        """Class and time of the selected pair."""

        pair = self.df_relative.loc[self.selected_pair_rows()]
        return (
            pair[self.pair_class_name].to_numpy()[0],
            pair[self.pair_time_name].to_numpy()[0],
        )

    def apply_modification(self):
        """Apply modification."""

        annotation = self.chosen_annotation()
        if annotation is None:
            return
        cclass, t0 = annotation
        pair_filter = self.selected_pair_rows()

        # Record the previous class/time so the log shows what actually changed
        old_class, old_t0 = self.current_annotation()

        self.df_relative.loc[pair_filter, self.pair_class_name] = cclass
        self.df_relative.loc[pair_filter, self.pair_time_name] = t0

        self.annotation_log.append(
            f"REFERENCE_ID {self.reference_track_of_interest} - NEIGHBOR_ID "
            f"{self.neighbor_track_of_interest} ({self.reference_population}-"
            f"{self.neighbor_population}): {self.pair_class_name} {old_class}->{cclass}, "
            f"{self.pair_time_name} {old_t0}->{t0}"
        )

        write_event_status(
            self.df_relative,
            self.pair_class_name,
            self.pair_time_name,
            self.pair_status_name,
            keys=PAIR_COLUMNS,
            rows=pair_filter,
            recently_modified=True,
        )

        self.extract_scatter_from_lines()
        self.give_reference_cell_information()
        self.give_neighbor_cell_information()
        self.give_pair_information()

        self.reset_correction()

        # Unselect the pair, but keep the reference
        self.pair_selection = []
        self.neighbor_selection = []
        self.neighbor_track_of_interest = None

        self.refresh_scatter_from_trajectories(self.reference_population)
        self.refresh_scatter_from_trajectories(self.neighbor_population)

        self.recolor_selection()
        self.trace_neighbors()

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def load_population_table(self, population: str) -> Optional[pd.DataFrame]:
        """
        The trajectories of a population, None (after a warning) if missing.

        Raises
        ------
        ValueError
            If the population is not tracked.
        """

        path = self.pos + os.sep.join(
            ["output", "tables", f"trajectories_{population}.pkl"]
        )
        if not os.path.exists(path):
            path = path.replace(".pkl", ".csv")
        if not os.path.exists(path):
            warn(f"The trajectories for the {population} population cannot be detected...")
            return None

        df = pd.read_pickle(path) if path.endswith(".pkl") else pd.read_csv(path)
        if "TRACK_ID" not in df.columns:
            raise ValueError(
                f"The '{population}' population has not been tracked (no TRACK_ID found).\nPlease track this population before analyzing interactions."
            )
        return df

    def locate_all_tracks(self):
        """Load the trajectories of all the populations."""

        self.dataframes = {}
        self.population_columns = {}
        self.MinMaxScaler_per_pop = {}

        for population in self.populations:
            df = self.load_population_table(population)
            class_cols, columns, scaler = [], [], None

            if df is not None:
                df = df.sort_values(by=["TRACK_ID", "FRAME"])
                class_cols = event_class_columns(df.columns)
                class_name = class_cols[0] if class_cols else "class"
                time_name, status_name = event_columns(class_name)
                # The status is left as is, not to erase static classifications.
                complete_event_columns(df, class_name, time_name, status_name)

                # All the cells in grey until an event is chosen.
                df["status_color"] = color_from_status(2)
                df["class_color"] = color_from_status(2)

                df = df.dropna(subset=["POSITION_X", "POSITION_Y"])
                df["x_anim"] = (df["POSITION_X"] * self.fraction).astype(int)
                df["y_anim"] = (df["POSITION_Y"] * self.fraction).astype(int)

                columns, scaler = fit_feature_scaler(
                    df,
                    excluded=self.cols_to_remove + class_cols,
                    excluded_prefixes=("t_", "neighborhood_"),
                )

            self.dataframes[population] = df
            self.population_columns[population] = columns
            self.MinMaxScaler_per_pop[population] = scaler

    def locate_relative_tracks(self):
        """Locate relative tracks."""

        self.relative_trajectories_path = self.pos + os.sep.join(
            ["output", "tables", "trajectories_pairs.csv"]
        )

        if not os.path.exists(self.relative_trajectories_path):
            # Raised rather than closing the window: __init__ carried on after
            # the message and crashed on the missing table. The caller shows it.
            raise ValueError(
                "The pair measurements cannot be detected... Please measure the pairs first."
            )

        self.df_relative = pd.read_csv(self.relative_trajectories_path)
        self.df_relative = self.df_relative.astype(
            {"reference_population": str, "neighbor_population": str}
        )
        self.df_relative = self.df_relative.sort_values(
            by=[
                "REFERENCE_ID",
                "NEIGHBOR_ID",
                "reference_population",
                "neighbor_population",
                "FRAME",
            ]
        )
        self.relative_cols = np.array(self.df_relative.columns)
        self.relative_class_cols = event_class_columns(self.df_relative.columns)

        for col in self.df_relative.columns:
            if (
                self.df_relative[col].isnull().all()
                or self.df_relative[col].dtype == object
            ):
                self.cols_to_remove.append(col)

        self.pair_columns, self.MinMaxScaler_pairs = fit_feature_scaler(
            self.df_relative,
            excluded=self.cols_to_remove + self.relative_class_cols,
            excluded_prefixes=("t0_", "t_", "neighborhood_"),
        )

    def set_reference_and_neighbor_populations(self):
        """Set reference and neighbor populations."""

        neigh = self.neighborhood_choice_cb.currentText()

        self.current_neighborhood = neigh
        for pop in self.dataframes.keys():
            self.current_neighborhood = self.current_neighborhood.replace(
                f"{pop}_ref_", ""
            )

        self.reference_population = neigh.split("_")[0]
        self.neighbor_population = self.reference_population

        if "_(" in self.current_neighborhood and ")_" in self.current_neighborhood:
            populations = (
                self.current_neighborhood.split("_(")[-1].split(")_")[0].split("-")
            )
            self.reference_population = populations[0]
            self.neighbor_population = populations[-1]

        logger.debug(f"Current neighborhood: {self.current_neighborhood}; reference={self.reference_population}; neighbor={self.neighbor_population}")

        self.relative_class_choice_cb.setCurrentIndex(
            self.relative_class_choice_cb.findText("class_" + self.current_neighborhood)
        )

    def neighborhood_changed(self):
        """Handle neighborhood change."""
        self.cancel_selection()
        self.set_reference_and_neighbor_populations()
        # Update reference classes and neighbor classes
        self.extract_relevant_events()
        self.fill_class_cbs()
        self.extract_scatter_from_lines()
        self.plot_signals()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def fill_signal_choices(self):
        """Fill signal choices."""

        self.reference_signals = list(
            self.dataframes[self.reference_population].columns
        )
        self.neighbor_signals = list(self.dataframes[self.neighbor_population].columns)
        self.relative_signals = list(self.relative_cols)

        self.cols_to_remove.extend(
            c
            for c in self.reference_signals + self.neighbor_signals
            if c.startswith("neighborhood")
        )

        removed = set(self.cols_to_remove)
        for signals in (
            self.reference_signals,
            self.neighbor_signals,
            self.relative_signals,
        ):
            signals[:] = [c for c in signals if c not in removed]

        for i in range(self.n_signals):
            self.update_signal_choices(i)

    def update_signal_choices(self, index: int) -> None:
        """
        Update signal choices.

        Parameters
        ----------
        index : int
            The index of the signal choice.
        """

        combo = self.signal_choices[index]
        try:
            combo.currentIndexChanged.disconnect()
        except TypeError:
            pass  # No connection to remove.

        current_idx = max(combo.currentIndex(), 0)
        combo.clear()
        signals = {
            REFERENCE: self.reference_signals,
            NEIGHBOR: self.neighbor_signals,
            PAIR: self.relative_signals,
        }.get(self.signal_pop_button_groups[index].checkedId())
        if signals is not None:
            combo.addItems(["--"] + signals)
            combo.setCurrentIndex(current_idx)

        combo.currentIndexChanged.connect(self.plot_signals)
        self.plot_signals()

    def generate_signal_choices(self):
        """Generate signal choices."""

        self.signal_choices = [QSearchableComboBox() for _ in range(self.n_signals)]
        self.signal_pop_button_groups = [QButtonGroup() for _ in range(self.n_signals)]
        self.reference_pop_option_buttons = [
            QRadioButton() for _ in range(self.n_signals)
        ]
        self.neighbor_pop_option_buttons = [
            QRadioButton() for _ in range(self.n_signals)
        ]
        self.relative_pop_option_buttons = [
            QRadioButton() for _ in range(self.n_signals)
        ]

        for i in range(self.n_signals):
            self.signal_choices[i].currentIndexChanged.connect(self.plot_signals)
            for source, buttons in (
                (REFERENCE, self.reference_pop_option_buttons),
                (NEIGHBOR, self.neighbor_pop_option_buttons),
                (PAIR, self.relative_pop_option_buttons),
            ):
                self.signal_pop_button_groups[i].addButton(buttons[i], source)
                buttons[i].toggled.connect(partial(self.update_signal_choices, i))
                buttons[i].setEnabled(False)

    @staticmethod
    def set_time_line(line, t: float, ylim: Optional[Tuple[float, float]] = None) -> None:
        """Place a vertical time line of the signal plot."""

        line.set_xdata([t, t])
        line.set_ydata(list(ylim if ylim is not None else line.axes.get_ylim()))

    def clear_time_lines(self, reference: bool = True) -> None:
        """Hide the time lines of the pair and the neighbor, and of the reference."""

        lines = [self.line_dt, self.line_dt_neighbor]
        if reference:
            lines.append(self.line_dt_reference)
        for line in lines:
            line.set_xdata([])
            line.set_ydata([])

    def plot_signals(self):
        """Plot signals."""

        for t in list(self.cell_ax.texts):
            t.remove()

        if (
            self.reference_track_of_interest is None
            and self.neighbor_track_of_interest is None
        ):
            # No cell selected, plot nothing
            self.cell_ax.text(
                0.5,
                0.5,
                "No data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=self.cell_ax.transAxes,
            )
            for line in self.lines:
                line.set_xdata([])
                line.set_ydata([])
                line.set_label("")
            self.clear_time_lines()
            self.cell_fcanvas.canvas.draw()
            return

        range_values = []
        for i, line in enumerate(self.lines):

            signal_txt = self.signal_choices[i].currentText()
            option = self.signal_pop_button_groups[i].checkedId()

            n_cut = 25
            shown = signal_txt
            if len(signal_txt) > n_cut:
                shown = signal_txt[: (n_cut - 3)] + "..."

            df, rows, label = None, None, ""
            if signal_txt not in ("--", ""):
                if option == REFERENCE and self.reference_track_of_interest is not None:
                    df = self.dataframes[self.reference_population]
                    rows = df["TRACK_ID"] == self.reference_track_of_interest
                    all_values = df[signal_txt]
                    label = f"reference ({self.reference_population}) {shown}"
                elif option == NEIGHBOR and self.neighbor_track_of_interest is not None:
                    df = self.dataframes[self.neighbor_population]
                    rows = df["TRACK_ID"] == self.neighbor_track_of_interest
                    all_values = df[signal_txt]
                    label = f"neighbor ({self.neighbor_population}) {shown}"
                elif (
                    option == PAIR
                    and self.reference_track_of_interest is not None
                    and self.neighbor_track_of_interest is not None
                ):
                    df = self.df_relative
                    rows = self.selected_pair_rows(in_neighborhood=False)
                    all_values = df.loc[self.population_pair_rows(), signal_txt]
                    label = f"pair {shown}"

            if df is None:
                timeline, signal = [], []
            else:
                timeline = df.loc[rows, "FRAME"].to_numpy()
                signal = df.loc[rows, signal_txt].to_numpy()
                range_values.extend(all_values.values)

            line.set_label(label)
            line.set_xdata(timeline)
            line.set_ydata(signal)
            line.set_color(tab10(i / float(self.n_signals)))

        limits = signal_limits(pd.to_numeric(range_values, errors="coerce"))
        if limits is not None:
            self.set_signal_ylim(*limits)

        df_reference = self.dataframes[self.reference_population]
        t_reference = df_reference.loc[
            df_reference["TRACK_ID"] == self.reference_track_of_interest,
            self.reference_time_name,
        ].values
        if len(t_reference) > 0:
            self.set_time_line(self.line_dt_reference, t_reference[0])

        if (
            self.reference_track_of_interest is not None
            and self.neighbor_track_of_interest is not None
        ):
            t0 = (
                self.df_relative.loc[
                    self.selected_pair_rows(in_neighborhood=False), self.pair_time_name
                ]
                .dropna()
                .values
            )
            if len(t0) > 0:
                self.set_time_line(self.line_dt, t0[0])

            df_neighbor = self.dataframes[self.neighbor_population]
            t_neighbor = df_neighbor.loc[
                df_neighbor["TRACK_ID"] == self.neighbor_track_of_interest,
                self.neighbor_time_name,
            ].values
            if len(t_neighbor) > 0:
                self.set_time_line(self.line_dt_neighbor, t_neighbor[0])

        self.cell_ax.legend(fontsize=8)
        self.cell_fcanvas.canvas.draw()

    # ------------------------------------------------------------------
    # Scatters
    # ------------------------------------------------------------------

    def extract_scatter_from_lines(self):
        """Extract scatter from lines."""

        self.lines_tracks = []
        self.lines_colors_status = []
        self.initial_lines_colors_status = []
        self.lines_colors_class = []
        self.initial_lines_colors_class = []

        pairs = self.df_relative.loc[
            self.neighborhood_rows(),
            ["FRAME", "REFERENCE_ID", "NEIGHBOR_ID", "status_color", "class_color"],
        ]
        for frame_pairs in split_by_frame(pairs, self.len_movie):
            data = frame_pairs.iloc[:, 1:].to_numpy()
            self.lines_tracks.append(data[:, [0, 1]])

            self.initial_lines_colors_status.append(data[:, [0, 1, 2]])
            self.lines_colors_status.append(data[:, [0, 1, 2]])

            self.initial_lines_colors_class.append(data[:, [0, 1, 3]])
            self.lines_colors_class.append(data[:, [0, 1, 3]])

    def extract_scatter_from_trajectories(self):
        """Extract scatter from trajectories."""

        self.tracks = {}
        self.positions = {}
        self.colors = {}
        self.initial_colors = {}
        self.timeline = {}

        for population in self.dataframes.keys():
            self.refresh_scatter_from_trajectories(population)

    def refresh_scatter_from_trajectories(self, population: str) -> None:
        """
        Refresh scatter from trajectories.

        Parameters
        ----------
        population : str
            The population to refresh.
        """

        df = self.dataframes[population]
        positions = []
        tracks = []
        timeline = []
        colors = []
        initial_colors = []

        if df is not None:
            for t, cells in enumerate(split_by_frame(df, self.len_movie)):
                positions.append(cells[["x_anim", "y_anim"]].to_numpy())
                # Copies: both are written (selection highlight, restore), and
                # under copy-on-write pandas hands out read-only arrays.
                cell_colors = cells[["class_color", "status_color"]].to_numpy()
                colors.append(cell_colors.copy())
                initial_colors.append(cell_colors.copy())
                tracks.append(cells["TRACK_ID"].to_numpy())
                timeline.append(t)

        self.tracks[population] = tracks
        self.timeline[population] = timeline
        self.positions[population] = positions
        self.colors[population] = colors
        self.initial_colors[population] = initial_colors

        if (
            self.reference_track_of_interest is not None
            and self.neighbor_track_of_interest is not None
        ):
            self.highlight_the_pair()

    def release_resources(self) -> None:
        """Free the tables, with the rest of the window."""

        super().release_resources()
        self.dataframes.clear()
        # A window may be closed twice.
        if hasattr(self, "df_relative"):
            del self.df_relative

    def looped_animation(self):
        """Show the movie with the cells of each population marked, playing in a loop."""

        self.create_movie_axes()
        self.lines_data = {}

        self.status_scatter = {}
        self.class_scatter = {}

        markers = ["x", "+", "*", ".", "X"]
        while len(markers) < len(self.dataframes):
            markers = markers * 2

        for i, (pop, df) in enumerate(self.dataframes.items()):
            if df is not None:
                status_scatter = self.ax.scatter(
                    self.positions[pop][0][:, 0],
                    self.positions[pop][0][:, 1],
                    c=self.colors[pop][0][:, 1],
                    s=50,
                    picker=True,
                    pickradius=150,
                    zorder=10,
                    marker=markers[i],
                )
                class_scatter = self.ax.scatter(
                    self.positions[pop][0][:, 0],
                    self.positions[pop][0][:, 1],
                    marker="o",
                    facecolors="none",
                    edgecolors=self.colors[pop][0][:, 0],
                    s=200,
                    zorder=10,
                )
            else:
                status_scatter = self.ax.scatter(
                    [], [], s=50, picker=True, pickradius=10, marker=markers[i]
                )
                class_scatter = self.ax.scatter(
                    [], [], marker="o", facecolors="none", s=200
                )
            self.status_scatter[pop] = status_scatter
            self.class_scatter[pop] = class_scatter

        self.points = self.ax.scatter(
            [], [], marker="$\\Join$", s=100, picker=True, pickradius=10, zorder=10
        )
        self.start_animation()

    def create_cell_signal_canvas(self):
        """Create the canvas for plotting signals, with the time of each cell."""

        super().create_cell_signal_canvas()
        min_val, max_val = self.cell_ax.get_ylim()
        (self.line_dt_reference,) = self.cell_ax.plot(
            [-1, -1], [min_val, max_val], c="tab:blue", linestyle="--"
        )
        (self.line_dt_neighbor,) = self.cell_ax.plot(
            [-1, -1], [min_val, max_val], c="tab:red", linestyle="--"
        )
        self.cell_fcanvas.canvas.draw()

    def on_scatter_pick(self, event: "matplotlib.backend_bases.PickEvent") -> None:
        """
        Handle scatter pick event.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event.
        """
        self.identify_closest_marker(event)

        if self.selected_population == self.reference_population:

            _, tracks, _, _, _ = self.get_reference_sets()
            if self.index is None:
                return
            toi = tracks[self.framedata][self.index]

            if len(self.reference_selection) == 0:

                self.reference_track_of_interest = toi
                self.reference_selection.append(self.reference_track_of_interest)

                self.get_neighbors_of_selected_cell(self.reference_track_of_interest)
                logger.info(f"You selected track {self.reference_track_of_interest} with {len(self.neighbors)} neighbors...")

                self.give_reference_cell_information()
                self.give_neighbor_cell_information()
                self.give_pair_information()

                self.recolor_selection()
                self.trace_neighbors()

                for button in self.reference_pop_option_buttons:
                    button.setEnabled(True)

                self.plot_signals()

            elif toi in self.reference_selection and not self.pair_selected:
                self.cancel_btn.click()
                self.cancel_selection()

            elif (
                toi in self.neighbors
                and self.neighbor_population == self.reference_population
                and not self.pair_selected
            ):
                self.pick_neighbor(toi)
            else:
                logger.debug("One cell already selected, skip.")

        elif len(self.reference_selection) > 0 and not self.pair_selected:

            logger.debug("Picking a cell from the neighbor population...")
            _, tracks, _, _ = self.get_neighbor_sets()
            if self.index is None:
                return
            toi = tracks[self.framedata][self.index]
            if toi in self.neighbors:
                self.pick_neighbor(toi)
            else:
                self.cancel_pair_selection()

        if self.pair_selected and len(self.reference_selection) > 0:

            logger.debug("A pair has been selected.")
            if self.index is not None and len(self.pair_selection) == 0:
                try:
                    selected_point = event.artist.get_offsets()[self.index]
                except Exception as e:
                    logger.warning(f"Failed to get selected point offsets: {e}")
                    return

                connection = self.connections.get(
                    (selected_point[0], selected_point[1])
                )
                if connection is not None:
                    self.neighbor_track_of_interest = connection[0][1]
                    self.highlight_the_pair()
                else:
                    self.cancel_pair_selection()
            else:
                self.cancel_pair_selection()

        logger.debug(f"pair_selection={self.pair_selection}")

    def pick_neighbor(self, track: int) -> None:
        """Select the pair with a neighbor, or unselect the pair selected."""

        if len(self.pair_selection) == 0:
            self.neighbor_track_of_interest = track
            self.highlight_the_pair()
        else:
            self.cancel_pair_selection()

    def highlight_the_pair(self):
        """Highlight the selected pair."""
        # 1) recolor the neighbor marker
        logger.debug(f"Reference cell: {self.reference_track_of_interest}, neighbor cell: {self.neighbor_track_of_interest}")

        _, tracks, colors, _ = self.get_neighbor_sets()
        self.neigh_cell_loc_idx = []
        self.neigh_cell_loc_t = []
        self.neigh_previous_color = []

        for t, frame_tracks in enumerate(tracks):
            indices_picked = np.where(frame_tracks == self.neighbor_track_of_interest)[0]
            if len(indices_picked) > 0:
                self.neigh_cell_loc_t.append(t)
                self.neigh_cell_loc_idx.append(indices_picked[0])

        for t, idx in zip(self.neigh_cell_loc_t, self.neigh_cell_loc_idx):
            self.neigh_previous_color.append(colors[t][idx].copy())
            colors[t][idx] = "lime"

        # 2) identify the pair line and recolor it
        for frame_colors in self.lines_colors_status + self.lines_colors_class:
            ids = frame_colors[:, :2].astype(float)
            pair = (ids[:, 0] == self.reference_track_of_interest) & (
                ids[:, 1] == self.neighbor_track_of_interest
            )
            frame_colors[pair, 2] = "lime"

        self.pair_selection.append(
            (self.reference_track_of_interest, self.neighbor_track_of_interest)
        )
        self.neighbor_selection.append(self.neighbor_track_of_interest)
        self.give_neighbor_cell_information()
        self.give_pair_information()

        # Allow pair signal options
        for i in range(self.n_signals):
            self.neighbor_pop_option_buttons[i].setEnabled(True)
            self.relative_pop_option_buttons[i].setEnabled(True)

        self.plot_signals()

        # Allow pair annotation
        self.enable_correction()

    def get_neighbor_sets(self):
        """
        Get neighbor sets.

        Returns
        -------
        tuple
            Neighbor positions, tracks, colors, and initial colors.
        """
        return (
            self.positions[self.neighbor_population],
            self.tracks[self.neighbor_population],
            self.colors[self.neighbor_population],
            self.initial_colors[self.neighbor_population],
        )

    def get_reference_sets(self):
        """
        Get reference sets.

        Returns
        -------
        tuple
            Reference positions, tracks, timeline, colors, and initial colors.
        """
        return (
            self.positions[self.reference_population],
            self.tracks[self.reference_population],
            self.timeline[self.reference_population],
            self.colors[self.reference_population],
            self.initial_colors[self.reference_population],
        )

    def trace_neighbors(self):
        """Trace the lines from the selected cell to its neighbors."""
        self.lines_data = {}
        self.points_data = {}
        self.connections = {}
        self.line_connections = {}

        positions, tracks, _, _ = self.get_neighbor_sets()
        _, tracks_reference, _, _, _ = self.get_reference_sets()

        for neigh in self.neighbors:
            for t, frame_tracks in enumerate(tracks):
                indices = np.where(frame_tracks == neigh)[0]
                if len(indices) == 0 or not np.any(
                    tracks_reference[t] == self.reference_track_of_interest
                ):
                    continue

                t_ref_idx = np.where(self.reference_timeline == t)[0]
                if len(t_ref_idx) == 0:
                    continue
                t_ref_idx = t_ref_idx[0]

                ref_x = self.reference_x[t_ref_idx]
                ref_y = self.reference_y[t_ref_idx]
                neigh_x, neigh_y = positions[t][indices[0]]
                midpoint = ((ref_x + neigh_x) / 2, (ref_y + neigh_y) / 2)

                self.lines_data.setdefault(t, []).append(
                    ([ref_x, neigh_x], [ref_y, neigh_y])
                )
                self.points_data.setdefault(t, []).append(midpoint)

                pair = [(self.reference_track_of_interest, neigh)]
                self.connections[midpoint] = pair
                self.line_connections[(ref_x, neigh_x, ref_y, neigh_y)] = pair

    def recolor_selection(self):
        """Recolor the selected cell, and all the other reference cells in black."""
        positions, tracks, timelines, colors, init_colors = self.get_reference_sets()

        self.reference_x = []
        self.reference_y = []
        self.reference_timeline = []

        for t, frame_tracks in enumerate(tracks):
            picked = frame_tracks == self.reference_track_of_interest

            indices_picked = np.where(picked)[0]
            if len(indices_picked) > 0:
                idx = indices_picked[0]
                self.reference_x.append(positions[t][idx, 0])
                self.reference_y.append(positions[t][idx, 1])
                self.reference_timeline.append(timelines[t])
                colors[t][idx] = "lime"

            # Remember the colors of the other cells, then blacken them
            not_picked = np.where(~picked)[0]
            init_colors[t][not_picked] = colors[t][not_picked].copy()
            colors[t][not_picked] = "black"

        self.reference_timeline = np.array(self.reference_timeline)

    def get_neighbors_of_selected_cell(self, selected_cell: int) -> None:
        """
        Get neighbors of the selected cell.

        Parameters
        ----------
        selected_cell : int
            The ID of the selected cell.
        """
        self.neighbors = np.unique(
            self.df_relative.loc[
                (self.df_relative["REFERENCE_ID"] == selected_cell)
                & self.neighborhood_rows(),
                "NEIGHBOR_ID",
            ]
        )
        self.neighbor_track_of_interest = None

    def identify_closest_marker(
        self, event: "matplotlib.backend_bases.PickEvent"
    ) -> None:
        """
        Identify the closest marker to the pick event.

        Parameters
        ----------
        event : matplotlib.backend_bases.PickEvent
            The pick event.
        """
        ind = event.ind
        populations = list(self.dataframes.keys())

        # The artists are numbered in their order of creation: the image, then
        # two scatters per population, then the pair midpoints.
        number = int(event.artist.get_label().split("_child")[1])
        self.pair_selected = number > len(populations) * 2

        if self.pair_selected:
            logger.debug("A pair is selected...")
            self.selected_population = None
            self.index = ind[0] if len(ind) > 0 else None
            return

        self.selected_population = populations[(number - 1) // 2]
        try:
            self.index = closest_picked_index(
                ind,
                self.positions[self.selected_population][self.framedata],
                event.mouseevent,
            )
        except Exception as e:
            logger.warning(f"Failed to find closest marker: {e}")
            self.index = None

    def draw_frame(self, framedata: int) -> List:
        """
        Update plot elements at each timestep of the loop.

        Parameters
        ----------
        framedata : int
            The current frame index.
        """

        self.framedata = framedata
        self.frame_lbl.setText(f"frame: {self.framedata}")
        self.im.set_array(self.stack[self.framedata])

        for pop, df in self.dataframes.items():
            if df is None:
                continue
            positions = self.positions[pop][self.framedata]
            colors = self.colors[pop][self.framedata]
            self.status_scatter[pop].set_visible(True)
            self.status_scatter[pop].set_picker(True)
            self.status_scatter[pop].set_offsets(positions)
            self.status_scatter[pop].set_color(colors[:, 1])
            self.class_scatter[pop].set_visible(True)
            self.class_scatter[pop].set_offsets(positions)
            self.class_scatter[pop].set_edgecolor(colors[:, 0])

        # The lines of the previous frame would otherwise pile up on the axes.
        for line in self.lines_list:
            line.remove()
        self.lines_list = []

        for x_coords, y_coords in self.lines_data.get(self.framedata, []):
            class_colors = self.lines_colors_class[self.framedata]
            pair = self.line_connections[
                x_coords[0], x_coords[1], y_coords[0], y_coords[1]
            ][0]
            this_pair = class_colors[
                (class_colors[:, 0] == pair[0]) & (class_colors[:, 1] == pair[1])
            ]
            if len(this_pair) > 0:
                (line,) = self.ax.plot(
                    x_coords, y_coords, alpha=1, linewidth=2, color=this_pair[0][2]
                )
                self.lines_list.append(line)

        artists = (
            [self.im]
            + list(self.status_scatter.values())
            + list(self.class_scatter.values())
        )
        if not self.lines_list:
            return artists

        points = self.points_data[self.framedata]
        status_colors = self.lines_colors_status[self.framedata]
        colors = []
        for point in points:
            reference, neighbor = self.connections[point][0]
            match = status_colors[
                (status_colors[:, 0] == reference) & (status_colors[:, 1] == neighbor)
            ]
            colors.append(match[0][2] if len(match) > 0 else "black")
        self.points.set_offsets(points)
        self.points.set_color(colors)

        return artists + self.lines_list + [self.points]

    # ------------------------------------------------------------------
    # Information, saving and export
    # ------------------------------------------------------------------

    def give_reference_cell_information(self):
        """Display information about the selected reference cell."""

        if self.reference_track_of_interest is not None:
            selected = f"Ref #{self.reference_track_of_interest}\n"
        else:
            selected = "Ref: None\n"
        self.reference_cell_info.setText(
            f"{selected}Pop: {self.reference_population}\n"
        )

    def give_neighbor_cell_information(self):
        """Display information about the selected neighbor cell."""

        if self.neighbor_track_of_interest is not None:
            selected = f"Neigh #{self.neighbor_track_of_interest}\n"
        else:
            selected = "Neigh: None\n"
        self.neighbor_cell_info.setText(
            f"{selected}Pop: {self.neighbor_population}\n"
        )

    def give_pair_information(self):
        """Display information about the selected pair."""

        if (
            self.neighbor_track_of_interest is None
            or self.reference_track_of_interest is None
        ):
            self.pair_info.setText("Pair: None\n")
            return

        pair_selected = f"Pair: ({self.reference_track_of_interest},{self.neighbor_track_of_interest})\n"
        pair_subset = self.df_relative.loc[self.selected_pair_rows()]
        if len(pair_subset) > 0:
            current_class = self.relative_class_choice_cb.currentText()
            pair_class = f"Event class: {pair_subset[current_class].values[0]}\n"
            pair_time = f"Time: {pair_subset[self.pair_time_name].values[0]}\n"
        else:
            pair_class = "Event class: N/A\n"
            pair_time = "Time: N/A\n"
        self.pair_info.setText(pair_selected + pair_class + pair_time)

    def save_trajectories(self):
        """Save relative trajectories."""
        if self.normalized_signals:
            self.normalize_features_btn.click()
        self.cancel_selection()

        self.relative_class_name = self.relative_class_choice_cb.currentText()
        self.df_relative = self.df_relative.drop(
            self.df_relative[self.df_relative[self.relative_class_name] > 2].index
        )
        self.df_relative.to_csv(self.relative_trajectories_path, index=False)
        logger.info("Relative table saved.")
        self.write_annotation_log(
            "PAIR EVENT ANNOTATION (manual)",
            f"relative_class_name: {self.relative_class_name}",
        )

    def export_signals(self):
        """Export signals to a .npy file."""

        if self.normalized_signals:
            self.normalize_features_btn.click()

        pair_filter = self.population_pair_rows() & self.neighborhood_rows()
        df_reference = self.dataframes[self.reference_population]
        df_neighbor = self.dataframes[self.neighbor_population]

        def cell_signals(df, track, prefix):
            cell = df.loc[df["TRACK_ID"] == track]
            return {
                prefix + col: cell[col].to_numpy()
                for col in cell.columns
                if not col.startswith("neighborhood") and is_numeric_dtype(cell[col])
            }

        training_set = []
        for (reference_cell, neighbor_cell), group in self.df_relative.loc[
            pair_filter
        ].groupby(["REFERENCE_ID", "NEIGHBOR_ID"]):
            signals = {
                "time_of_interest": group[self.pair_time_name].values[0],
                "class": group[self.pair_class_name].values[0],
                "neighborhood_of_interest": self.current_neighborhood,
                "reference_population": self.reference_population,
                "neighbor_population": self.neighbor_population,
            }
            signals.update(
                {
                    "pair_" + col: group[col].to_numpy()
                    for col in group.columns
                    if is_numeric_dtype(group[col])
                }
            )
            signals.update(cell_signals(df_reference, reference_cell, "reference_"))
            signals.update(cell_signals(df_neighbor, neighbor_cell, "neighbor_"))
            training_set.append(signals)

        save_dataset(self, training_set, self.exp_dir + default_dataset_name(self.pos))

    def normalize_features(self):
        """Normalize features."""

        transform = "inverse_transform" if self.normalized_signals else "transform"
        for pop, df_pop in self.dataframes.items():
            if df_pop is not None:
                cols = self.population_columns[pop]
                scaler = self.MinMaxScaler_per_pop[pop]
                df_pop[cols] = getattr(scaler, transform)(df_pop[cols].values)

        if self.df_relative is not None:
            self.df_relative[self.pair_columns] = getattr(
                self.MinMaxScaler_pairs, transform
            )(self.df_relative[self.pair_columns].values)

        self.normalized_signals = not self.normalized_signals
        self.show_normalized()
        self.plot_signals()
