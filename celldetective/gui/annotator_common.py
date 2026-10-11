"""
What the annotator windows share: the event, interaction (pair event) and
phenotype annotators.

The functions work on the trajectory tables alone. The mixins hold the widgets
and the behavior the windows have in common: :class:`AnnotatorMixin` for all of
them, :class:`EventAnnotationMixin` for the windows annotating events (a class
and a time per track or pair), :class:`AnimationMixin` for those playing the
movie.
"""

import gc
import json
import os
from glob import glob
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fonticon_mdi6 import MDI6
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QCloseEvent, QKeySequence, QResizeEvent
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)
from sklearn.preprocessing import MinMaxScaler
from superqt.fonticon import icon
from tqdm import tqdm

from celldetective import get_logger
from celldetective.log_manager import positionlogger
from celldetective.gui.base.components import CelldetectiveWidget, generic_message
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.sliders import QLabeledDoubleRangeSlider, QLabeledSlider
from celldetective.gui.base.styles import CELLDETECTIVE_BLUE
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import color_from_class, color_from_status
from celldetective.utils.contrast import contrast_limits
from celldetective.utils.experiment import (
    extract_experiment_channels,
    get_experiment_labels,
    get_experiment_metadata,
)
from celldetective.utils.image_loaders import (
    _get_img_num_per_channel,
    auto_load_number_of_frames,
    load_frames,
)

logger = get_logger(__name__)

# Columns of a trajectory table that are not signals: identities, positions,
# annotation bookkeeping and experiment conditions.
NON_SIGNAL_COLUMNS = (
    "FRAME",
    "ID",
    "POSITION_X",
    "POSITION_Y",
    "TRACK_ID",
    "antibody",
    "cell_type",
    "class",
    "class_color",
    "class_id",
    "concentration",
    "dummy",
    "generation",
    "group",
    "group_color",
    "index",
    "parent",
    "pharmaceutical_agent",
    "pos_name",
    "position",
    "root",
    "state",
    "status",
    "status_color",
    "t",
    "t0",
    "well",
    "well_index",
    "well_name",
    "x_anim",
    "y_anim",
)

# The cells of a pair, in the pair table.
PAIR_COLUMNS = (
    "REFERENCE_ID",
    "NEIGHBOR_ID",
    "reference_population",
    "neighbor_population",
)

# Columns starting with "class" that are not event classes.
NON_EVENT_CLASS_COLUMNS = ("class_id", "class_color")

# Event classes.
EVENT, NO_EVENT, ELSE, DELETED = 0, 1, 2, 42

# Color of the icon of a toggled plot option (log scale, normalization).
ACTIVE_ICON_COLOR = CELLDETECTIVE_BLUE


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------


def experiment_columns(exp_dir: str) -> List[str]:
    """Columns the experiment metadata and condition labels add to the tables."""

    columns = []
    for mapping in (get_experiment_metadata(exp_dir), get_experiment_labels(exp_dir)):
        if mapping is not None:
            columns.extend(mapping.keys())
    return columns


def event_class_columns(columns: Iterable[str]) -> List[str]:
    """The event class columns among `columns`."""
    return [
        c
        for c in columns
        if c.startswith("class") and c not in NON_EVENT_CLASS_COLUMNS
    ]


def event_columns(class_name: str) -> Tuple[str, str]:
    """
    Time and status columns of an event class.

    ``class_<event>`` goes with ``t_<event>`` and ``status_<event>``, ``class``
    with ``t0`` and ``status``.
    """

    suffix = class_name.removeprefix("class").removeprefix("_")
    if not suffix:
        return "t0", "status"
    return f"t_{suffix}", f"status_{suffix}"


def _colors(values: np.ndarray, color_of: Callable, recently_modified: bool = False) -> list:
    """The color of each value, computed once per distinct value."""

    palette = {}
    colors = []
    for value in values:
        # NaN keys never match: their color is recomputed, as it is cheap and rare.
        if value not in palette:
            palette[value] = color_of(value, recently_modified=recently_modified)
        colors.append(palette[value])
    return colors


def write_event_status(
    df: pd.DataFrame,
    class_name: str,
    time_name: str,
    status_name: str,
    keys: Union[str, Sequence[str]] = "TRACK_ID",
    rows: Optional[pd.Series] = None,
    recently_modified: bool = False,
) -> None:
    """
    Fill the status column of an event class, and the status and class colors,
    from the class and time of each track: 0 before the event, 1 from it on, 2
    for the "else" class and 42 for a cell marked for deletion.

    Parameters
    ----------
    df : pandas.DataFrame
        The table, modified in place.
    class_name, time_name, status_name : str
        The columns of the event class.
    keys : str or sequence of str, optional
        The columns identifying a track, or a pair.
    rows : pandas.Series of bool, optional
        The rows to fill, all of them by default.
    recently_modified : bool, optional
        Whether to color the cells as just annotated.
    """

    keys = [keys] if isinstance(keys, str) else list(keys)
    subset = df if rows is None else df.loc[rows]
    subset = subset.dropna(subset=keys)
    groups = subset.groupby(keys, sort=False)
    t0 = groups[time_name].transform("first").to_numpy()
    cclass = groups[class_name].transform("first").to_numpy()
    frames = subset["FRAME"].to_numpy()

    status = np.where((t0 > 0) & (frames >= t0), 1.0, 0.0)
    status[cclass == ELSE] = ELSE
    status[cclass > ELSE] = DELETED

    df.loc[subset.index, status_name] = status
    df.loc[subset.index, "status_color"] = _colors(
        status, color_from_status, recently_modified
    )
    df.loc[subset.index, "class_color"] = _colors(
        cclass, color_from_class, recently_modified
    )


def complete_event_columns(
    df: pd.DataFrame,
    class_name: str,
    time_name: str,
    status_name: str,
    make_status: Optional[Callable[[], None]] = None,
    remake_empty_status: bool = False,
) -> None:
    """
    Add the columns an event class lacks.

    With its class and time, a missing status is derived from them by
    `make_status` (which, if None, leaves it missing); without them, a missing
    status is 0. The class defaults to 1 (no event) and the time to -1.

    Parameters
    ----------
    df : pandas.DataFrame
        The table, modified in place.
    class_name, time_name, status_name : str
        The columns of the event class.
    make_status : callable, optional
        Writes the status column from the class and time.
    remake_empty_status : bool, optional
        Whether a status column holding no value is derived again.
    """

    if time_name in df.columns and class_name in df.columns:
        missing = status_name not in df.columns or (
            remake_empty_status and df[status_name].isnull().all()
        )
        if missing and make_status is not None:
            make_status()
    elif status_name not in df.columns:
        df[status_name] = 0

    if class_name not in df.columns:
        df[class_name] = NO_EVENT
    if time_name not in df.columns:
        df[time_name] = -1


def color_events(df: pd.DataFrame, class_name: str, status_name: str) -> None:
    """Color the cells of the table by their status and class."""
    df["status_color"] = _colors(df[status_name].to_numpy(), color_from_status)
    df["class_color"] = _colors(df[class_name].to_numpy(), color_from_class)


def fit_feature_scaler(
    df: pd.DataFrame,
    excluded: Iterable[str],
    excluded_prefixes: Sequence[str] = (),
) -> Tuple[List[str], MinMaxScaler]:
    """
    Fit a min-max scaler to the features of a table, for the normalization of
    the plotted signals.

    Parameters
    ----------
    df : pandas.DataFrame
        The table.
    excluded : iterable of str
        Columns that are not features.
    excluded_prefixes : sequence of str, optional
        Prefixes of columns that are not features either.

    Returns
    -------
    columns : list of str
        The features.
    scaler : MinMaxScaler
        The scaler, fitted to them.
    """

    excluded = set(excluded)
    prefixes = tuple(excluded_prefixes)
    columns = [
        c
        for c in df.columns
        if c not in excluded and not (prefixes and c.startswith(prefixes))
    ]
    scaler = MinMaxScaler()
    scaler.fit(df[columns].values)
    return columns, scaler


def closest_picked_index(
    ind: Sequence[int], positions: np.ndarray, mouseevent: Any
) -> Optional[int]:
    """
    The point Matplotlib picked closest to the click, when it picked several.

    Parameters
    ----------
    ind : sequence of int
        Indices of the picked points.
    positions : numpy.ndarray
        Positions (x, y) of all the points of the scatter.
    mouseevent : matplotlib.backend_bases.MouseEvent
        The click.

    Returns
    -------
    int or None
        The index of the closest point, None if none was picked.
    """

    if len(ind) == 0:
        return None
    if len(ind) == 1:
        return ind[0]
    points = np.asarray(positions)[list(ind)]
    distances = np.hypot(
        points[:, 0] - mouseevent.xdata, points[:, 1] - mouseevent.ydata
    )
    return ind[int(np.argmin(distances))]


# ----------------------------------------------------------------------------
# Widgets and dialogs
# ----------------------------------------------------------------------------


def icon_button(
    icon_name: str,
    style: str,
    tooltip: Optional[str] = None,
    icon_size: int = 20,
    color: str = "black",
    text: str = "",
) -> QPushButton:
    """A push button carrying an MDI icon."""

    button = QPushButton(text)
    button.setStyleSheet(style)
    button.setIcon(icon(icon_name, color=color))
    button.setIconSize(QSize(icon_size, icon_size))
    if tooltip:
        button.setToolTip(tooltip)
    return button


def set_toggle_icon(button: Any, icon_name: str, active: bool) -> None:
    """Color the icon of a plot option button (or action) as on or off."""
    button.setIcon(icon(icon_name, color=ACTIVE_ICON_COLOR if active else "black"))


def signal_limits(values: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Y-limits of a signal plot showing `values`: a little beyond their range in
    linear scale, from their first percentile (or 1) in log scale.

    Returns
    -------
    tuple of float or None
        Linear minimum and maximum, and log minimum; None without values.
    """

    values = np.asarray(values, dtype=float)
    if not np.any(values == values):
        return None
    log_min = np.nanpercentile(values, 1) if np.any(values > 0) else 1
    return 0.98 * np.nanmin(values), np.nanmax(values) * 1.02, log_min


def confirm(text: str) -> bool:
    """Ask a yes/no question in a warning box: True unless the answer is no."""

    box = QMessageBox()
    box.setIcon(QMessageBox.Warning)
    box.setText(text)
    box.setWindowTitle("Warning")
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    return box.exec() != QMessageBox.No


def warn(text: str) -> None:
    """Show (and log) a warning box."""
    generic_message(text, msg_type="warning")


def new_class_dialog(
    title: str,
    name_label: str,
    default_name: str,
    on_submit: Callable[[], None],
    on_cancel: Callable[[], None],
    prefill: bool = True,
) -> Tuple[QWidget, QLineEdit, List[QRadioButton]]:
    """
    Show the dialog naming a new class.

    Parameters
    ----------
    title : str
        Window title.
    name_label : str
        Label of the name field.
    default_name : str
        Initial name.
    on_submit, on_cancel : callable
        Slots of the submit and cancel buttons.
    prefill : bool, optional
        Whether to offer to prefill the class with event, no event or else.

    Returns
    -------
    widget : QWidget
        The dialog.
    name_le : QLineEdit
        The name field.
    prefill_options : list of QRadioButton
        The prefill options, event first (empty without `prefill`).
    """

    widget = CelldetectiveWidget()
    widget.setWindowTitle(title)
    layout = QVBoxLayout()
    widget.setLayout(layout)

    name_hbox = QHBoxLayout()
    name_hbox.addWidget(QLabel(name_label), 25)
    name_le = QLineEdit(default_name)
    name_hbox.addWidget(name_le, 75)
    layout.addLayout(name_hbox)

    prefill_options = []
    if prefill:
        layout.addWidget(QLabel("prefill: "))
        radio_box = QHBoxLayout()
        for label in ("event", "no event", "else"):
            option = QRadioButton(label)
            radio_box.addWidget(option, 33, alignment=Qt.AlignCenter)
            prefill_options.append(option)
        prefill_options[0].setChecked(True)
        layout.addLayout(radio_box)

    btn_hbox = QHBoxLayout()
    submit_btn = QPushButton("submit")
    cancel_btn = QPushButton("cancel")
    btn_hbox.addWidget(cancel_btn, 50)
    btn_hbox.addWidget(submit_btn, 50)
    layout.addLayout(btn_hbox)
    submit_btn.clicked.connect(on_submit)
    cancel_btn.clicked.connect(on_cancel)

    widget.show()
    center_window(widget)
    return widget, name_le, prefill_options


def prefill_class(prefill_options: Sequence[QRadioButton]) -> int:
    """The class chosen in the prefill options of :func:`new_class_dialog`."""
    return [option.isChecked() for option in prefill_options].index(True)


def default_dataset_name(pos: str, suffix: str = "") -> str:
    """Default file name of a training set exported from a position."""

    parts = pos.split(os.sep)
    return f"{parts[-4]}_{parts[-2]}{suffix}.npy"


def save_dataset(parent: QWidget, data: Any, default_path: str) -> None:
    """Ask where to save a training set, and save it as a .npy file."""

    pathsave = QFileDialog.getSaveFileName(
        parent, "Select file name", default_path, ".npy"
    )[0]
    if pathsave == "":
        return
    if not pathsave.endswith(".npy"):
        pathsave += ".npy"
    try:
        np.save(pathsave, data)
        logger.info(f"File successfully written in {pathsave}.")
    except Exception as e:
        logger.error(f"Failed to write file {pathsave}: {e}")


def split_by_frame(df: pd.DataFrame, len_movie: int) -> List[pd.DataFrame]:
    """The rows of each frame of the movie, empty for a frame without cells."""

    frames = {int(frame): rows for frame, rows in df.groupby("FRAME")}
    empty = df.iloc[0:0]
    return [frames.get(t, empty) for t in range(len_movie)]


def _transport_button(icon_name: str, style: str, slot: Callable) -> QPushButton:
    """A square button of the movie controls."""

    button = icon_button(icon_name, style, icon_size=30)
    button.setFixedSize(QSize(60, 60))
    button.clicked.connect(slot)
    return button


# ----------------------------------------------------------------------------
# Mixins
# ----------------------------------------------------------------------------


class AnnotatorMixin:
    """
    The movie, the signal plot and the correction workflow of an annotator.

    The window sets ``exp_dir``, ``pos``, ``instructions_path`` and
    ``n_signals``, and builds ``correct_btn`` and ``cancel_btn``.
    """

    # Shortcuts acting on the selection, enabled only while there is one.
    selection_shortcuts = ()
    # Widgets showing whether the plot is in log scale, or its features normalized.
    log_toggles = ()
    normalize_toggles = ()

    def load_annotator_config(self) -> None:
        """Load the display settings of the annotator, or their defaults."""

        self.rgb_mode = False
        self.log_option = False
        self.percentile_mode = True
        self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
        self.fraction = 0.25
        self.anim_interval = 33

        if os.path.exists(self.instructions_path):
            with open(self.instructions_path, "r") as f:
                instructions = json.load(f)
            self.rgb_mode = instructions.get("rgb_mode", self.rgb_mode)
            self.percentile_mode = instructions.get(
                "percentile_mode", self.percentile_mode
            )
            self.target_channels = instructions.get("channels", self.target_channels)
            self.fraction = float(instructions.get("fraction", self.fraction))
            self.log_option = instructions.get("log", self.log_option)

    def find_movie(self, movie_prefix: str, len_movie: int) -> bool:
        """
        Find the movie of the position and read its channels.

        Parameters
        ----------
        movie_prefix : str
            Prefix of the movie file name.
        len_movie : int
            Number of frames, if the movie does not tell.

        Returns
        -------
        bool
            Whether a movie was found.
        """

        movies = glob(self.pos + os.sep.join(["movie", f"{movie_prefix}*.tif"]))
        if not movies:
            return False

        self.stack_path = movies[0]
        self.len_movie = auto_load_number_of_frames(self.stack_path) or len_movie
        channel_names, channels = extract_experiment_channels(self.exp_dir)
        self.channel_names = np.array(channel_names)
        self.channels = np.array(channels)
        self.nbr_channels = len(self.channels)
        return True

    def prepare_stack(self, progress_callback: Optional[Callable] = None) -> None:
        """
        Load the channels to display, rescaled by ``fraction``.

        Parameters
        ----------
        progress_callback : callable, optional
            Called with the progress in percent, and a status message when a
            channel starts; the loading stops when it returns False.
        """

        self.img_num_channels = _get_img_num_per_channel(
            self.channels, self.len_movie, self.nbr_channels
        )

        def frame_indices(channel_name):
            channel = self.channels[np.where(self.channel_names == channel_name)][0]
            return self.img_num_channels[channel]

        total_frames = sum(len(frame_indices(ch[0])) for ch in self.target_channels)
        loaded = 0

        stack = []
        for ch in tqdm(
            self.target_channels,
            desc="channel",
            disable=len(self.target_channels) <= 1,
        ):
            channel_name = ch[0]
            if self.percentile_mode:
                normalize_kwargs = {"percentiles": (ch[1], ch[2]), "values": None}
            else:
                normalize_kwargs = {"values": (ch[1], ch[2]), "percentiles": None}
            if self.rgb_mode:
                normalize_kwargs.update({"amplification": 255.0, "clip": True})

            if progress_callback and not progress_callback(
                int(loaded / total_frames * 100), f"Loading channel {channel_name}..."
            ):
                return

            frames = []
            for index in tqdm(frame_indices(channel_name), desc="frame"):
                if progress_callback:
                    if not progress_callback(int(loaded / total_frames * 100)):
                        return
                    loaded += 1

                if self.rgb_mode:
                    f = load_frames(
                        index,
                        self.stack_path,
                        scale=self.fraction,
                        normalize_input=True,
                        normalize_kwargs=normalize_kwargs,
                    ).astype(np.uint8)
                else:
                    f = load_frames(
                        index,
                        self.stack_path,
                        scale=self.fraction,
                        normalize_input=False,
                    )
                frames.append(f[:, :, 0])
            stack.append(frames)

        self.stack = np.array(stack)
        if self.rgb_mode:
            self.stack = np.moveaxis(self.stack, 0, -1)
        else:
            self.stack = self.stack[0]
            if self.log_option:
                positive = self.stack > 0.0
                self.stack[positive] = np.log(self.stack[positive])

    def create_cell_signal_canvas(self) -> None:
        """Create the plot of the signals of the selected cell."""

        self.cell_fig, self.cell_ax = plt.subplots(tight_layout=True)
        self.cell_fcanvas = FigureCanvas(self.cell_fig, interactive=True)
        self.cell_ax.clear()

        self.cell_ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.cell_ax.xaxis.set_major_locator(MultipleLocator(5))
        self.cell_ax.grid(which="major")
        self.cell_ax.set_xlabel("time [frame]")
        self.cell_ax.set_ylabel("signal")

        self.cell_fig.set_facecolor("none")
        self.cell_fig.canvas.setStyleSheet("background-color: transparent;")

        timeline = np.linspace(0, self.len_movie - 1, self.len_movie)
        self.lines = [
            self.cell_ax.plot([timeline], [np.zeros(self.len_movie)])[0]
            for _ in range(self.n_signals)
        ]
        for i, line in enumerate(self.lines):
            line.set_label(f"signal {i}")

        min_val, max_val = self.cell_ax.get_ylim()
        (self.line_dt,) = self.cell_ax.plot(
            [-1, -1], [min_val, max_val], c="k", linestyle="--"
        )

        self.cell_ax.set_xlim(0, self.len_movie)
        self.cell_ax.legend(fontsize=8)
        self.cell_fcanvas.canvas.draw()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Fit the signal plot to its new size."""

        super().resizeEvent(event)
        try:
            self.cell_fig.tight_layout()
        except Exception as e:
            logger.debug(f"tight_layout failed on resize: {e}")

    def switch_to_log(self) -> None:
        """Toggle the log scale of the signal plot."""

        try:
            if self.cell_ax.get_yscale() == "linear":
                _, ymax = self.cell_ax.get_ylim()
                self.cell_ax.set_yscale("log")
                self.cell_ax.set_ylim(self.value_magnitude, ymax)
                self.log_scale = True
            else:
                self.cell_ax.set_yscale("linear")
                self.log_scale = False
            for widget in self.log_toggles:
                set_toggle_icon(widget, MDI6.math_log, self.log_scale)
        except Exception as e:
            logger.warning(f"Failed to toggle log scale: {e}")

        self.cell_fcanvas.canvas.draw_idle()

    def show_normalized(self) -> None:
        """Show on the plot options whether the features are normalized."""
        for widget in self.normalize_toggles:
            set_toggle_icon(widget, MDI6.arrow_collapse_vertical, self.normalized_signals)

    def set_signal_ylim(self, ymin: float, ymax: float, log_min: float) -> None:
        """Set the y-limits of the signal plot, from `log_min` in log scale."""

        self.non_log_ymin, self.non_log_ymax = ymin, ymax
        self.value_magnitude = log_min
        if self.cell_ax.get_yscale() == "linear":
            self.cell_ax.set_ylim(ymin, ymax)
        else:
            self.cell_ax.set_ylim(log_min, ymax)

    def add_selection_shortcut(self, key: Any, slot: Callable) -> QShortcut:
        """A shortcut acting on the selection, disabled until there is one."""

        shortcut = QShortcut(key, self)
        shortcut.activated.connect(slot)
        shortcut.setEnabled(False)
        self.selection_shortcuts = (*self.selection_shortcuts, shortcut)
        return shortcut

    def set_correct_action(self, slot: Callable) -> None:
        """Make the correct button trigger `slot`, and only it."""

        try:
            self.correct_btn.clicked.disconnect()
        except TypeError:
            pass  # No connection to remove.
        self.correct_btn.clicked.connect(slot)

    def enable_correction(self) -> None:
        """Allow the correction of the selection, from the buttons and the shortcuts."""

        self.correct_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        for shortcut in self.selection_shortcuts:
            shortcut.setEnabled(True)

    def reset_correction(self) -> None:
        """Leave the correction: options hidden, correct and cancel disabled."""

        self.hide_annotation_buttons()
        self.correct_btn.setEnabled(False)
        self.correct_btn.setText("correct")
        self.set_correct_action(self.show_annotation_buttons)
        self.cancel_btn.setEnabled(False)
        for shortcut in self.selection_shortcuts:
            shortcut.setEnabled(False)

    def hide_annotation_buttons(self) -> None:
        """Hide the correction widgets and clear the time (or phenotype) field."""

        for widget in self.annotation_btns_to_hide:
            widget.hide()
        self.time_of_interest_label.setEnabled(False)
        self.time_of_interest_le.setText("")
        self.time_of_interest_le.setEnabled(False)

    def write_annotation_log(self, title: str, header: str) -> None:
        """Log the corrections made since the last save to the position log, and forget them."""

        with positionlogger(self.pos, filename=f"log_{self.mode}.txt"):
            logger.info(title)
            logger.info(header)
            for entry in self.annotation_log:
                logger.info(f"modified {entry}")
        self.annotation_log = []

    def release_resources(self) -> None:
        """
        Free what the window holds when it closes. Each layer extends it, the
        window releasing its tables.
        """

        figure = getattr(self, "cell_fig", None)  # Absent if closed while opening.
        if figure is not None:
            try:
                plt.close(figure)
            except Exception as e:
                logger.debug(f"Could not close cell_fig: {e}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Release the resources of the window, then close it."""

        self.release_resources()
        gc.collect()
        super().closeEvent(event)

    def close_without_new_class(self) -> None:
        """Close the new class dialog."""
        self.newClassWidget.close()

    def shortcut_suppr(self) -> None:
        """Mark the selected cell for deletion (Delete key)."""
        self.correct_btn.click()
        self.suppr_btn.click()
        self.correct_btn.click()


class EventAnnotationMixin:
    """
    The options annotating an event: event, no event, else or remove, and the
    time of the event.

    The window implements :meth:`current_annotation` and ``apply_modification``.
    """

    def init_event_buttons(self, remove_label: str = "remove") -> None:
        """Create the annotation options."""

        def option(label):
            button = QRadioButton(label)
            button.setStyleSheet(self.button_style_sheet_2)
            button.toggled.connect(self.enable_time_of_interest)
            return button

        self.event_btn = option("event")
        self.no_event_btn = option("no event")
        self.else_btn = option("else")
        self.suppr_btn = option(remove_label)
        self.suppr_btn.setToolTip(
            "Mark for deletion. Upon saving, the cell\nwill be removed from the tables."
        )

        self.time_of_interest_label = QLabel("time of interest: ")
        self.time_of_interest_le = QLineEdit()

        self.annotation_btns_to_hide = [
            self.event_btn,
            self.no_event_btn,
            self.else_btn,
            self.time_of_interest_label,
            self.time_of_interest_le,
            self.suppr_btn,
        ]

    @property
    def class_options(self) -> List[QRadioButton]:
        """The options, in the order of the classes."""
        return [self.event_btn, self.no_event_btn, self.else_btn, self.suppr_btn]

    def hide_annotation_buttons(self) -> None:
        """Hide and clear the annotation options."""

        super().hide_annotation_buttons()
        for button in self.class_options:
            button.setChecked(False)

    def enable_time_of_interest(self) -> None:
        """Let the time be set only for an event."""

        enabled = self.event_btn.isChecked()
        self.time_of_interest_label.setEnabled(enabled)
        self.time_of_interest_le.setEnabled(enabled)

    def current_annotation(self) -> Tuple[int, float]:
        """Class and time of the selected cell or pair."""
        raise NotImplementedError

    def show_annotation_buttons(self) -> None:
        """Show the options, set to the current annotation, to correct it."""

        for widget in self.annotation_btns_to_hide:
            widget.show()

        cclass, t0 = self.current_annotation()
        if cclass == EVENT:
            self.event_btn.setChecked(True)
            self.time_of_interest_le.setText(str(t0))
        elif cclass == NO_EVENT:
            self.no_event_btn.setChecked(True)
        elif cclass == ELSE:
            self.else_btn.setChecked(True)
        elif cclass > ELSE:
            self.suppr_btn.setChecked(True)

        self.enable_time_of_interest()
        self.correct_btn.setText("submit")
        self.set_correct_action(self.apply_modification)

    def chosen_annotation(self) -> Optional[Tuple[int, float]]:
        """
        Class and time chosen in the options, None if no option is chosen. An
        event without a valid time falls back to the "else" class.
        """

        if self.event_btn.isChecked():
            try:
                t0 = float(self.time_of_interest_le.text().replace(",", "."))
            except ValueError:
                return ELSE, -1
            self.line_dt.set_xdata([t0, t0])
            self.cell_fcanvas.canvas.draw_idle()
            return EVENT, t0
        if self.no_event_btn.isChecked():
            return NO_EVENT, -1
        if self.else_btn.isChecked():
            return ELSE, -1
        if self.suppr_btn.isChecked():
            return DELETED, -1
        # A class the options do not show (NaN) leaves them all unchecked:
        # submitting must not mark the cell for deletion.
        return None

    def shortcut_no_event(self) -> None:
        """Mark the selected cell as without event (N key)."""
        self.correct_btn.click()
        self.no_event_btn.click()
        self.correct_btn.click()


class AnimationMixin:
    """
    The movie playing in a loop, with its controls.

    The window implements ``draw_frame(framedata)``, returning the artists it
    updated, and sets ``stack``, ``len_movie`` and ``anim_interval``.
    """

    def create_movie_axes(self) -> None:
        """Create the figure of the movie, showing its first frame."""

        self.framedata = 0
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.fcanvas = FigureCanvas(self.fig, interactive=True)
        self.ax.clear()

        if self.rgb_mode:
            # Matplotlib ignores the limits of an RGB image.
            self.im = self.ax.imshow(self.stack[0], interpolation="none")
        else:
            # Kept for the contrast slider.
            self._contrast = contrast_limits(self.stack)
            _, (vmin, vmax) = self._contrast
            self.im = self.ax.imshow(
                self.stack[0], cmap="gray", interpolation="none", vmin=vmin, vmax=vmax
            )

    def start_animation(self) -> None:
        """Play the movie in the figure made by :meth:`create_movie_axes`."""

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")
        self.fig.set_facecolor("none")
        self.fig.canvas.setStyleSheet("background-color: black;")

        self.anim = FuncAnimation(
            self.fig,
            self.draw_frame,
            # A generator, so that the movie resumes where it was paused.
            frames=self.animation_generator,
            interval=self.anim_interval,
            blit=True,
            cache_frame_data=False,
        )
        self._pick_cid = self.fig.canvas.mpl_connect("pick_event", self.on_scatter_pick)
        self.fcanvas.canvas.draw()

    def animation_generator(self):
        """Frame indices, looping from the current frame on."""

        i = self.framedata
        while True:
            yield i
            i = (i + 1) % self.len_movie

    def build_animation_controls(self) -> QHBoxLayout:
        """
        The frame label, the framerate slider and the transport buttons, with
        their shortcuts: F first frame, L last frame, space play/pause.
        """

        style = self.button_select_all
        self.first_frame_btn = _transport_button(MDI6.page_first, style, self.set_first_frame)
        self.prev_frame_btn = _transport_button(MDI6.chevron_left, style, self.prev_frame)
        self.stop_btn = _transport_button(MDI6.stop, style, self.stop)
        self.start_btn = _transport_button(MDI6.play, style, self.start)
        self.next_frame_btn = _transport_button(MDI6.chevron_right, style, self.next_frame)
        self.last_frame_btn = _transport_button(MDI6.page_last, style, self.set_last_frame)
        # The movie plays at first.
        self.start_btn.hide()
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)

        self.first_short = QShortcut(QKeySequence("f"), self)
        self.first_short.activated.connect(self.set_first_frame)
        self.last_short = QShortcut(QKeySequence("l"), self)
        self.last_short.activated.connect(self.set_last_frame)
        self.toggle_short = QShortcut(Qt.Key_Space, self)
        self.toggle_short.activated.connect(self.toggle_animation)

        self.speed_slider = QLabeledSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 60)
        self.speed_slider.setValue(int(1000 / max(1, self.anim_interval)))
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.speed_slider.setFixedWidth(200)
        self.speed_slider.setToolTip("Adjust animation framerate (FPS)")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Framerate: "))
        speed_layout.addWidget(self.speed_slider)

        layout = QHBoxLayout()
        layout.addWidget(self.frame_lbl, 20, alignment=Qt.AlignLeft)
        layout.addLayout(speed_layout, 20)
        for button in (
            self.first_frame_btn,
            self.prev_frame_btn,
            self.stop_btn,
            self.start_btn,
            self.next_frame_btn,
            self.last_frame_btn,
        ):
            layout.addWidget(button, 5, alignment=Qt.AlignRight)
        return layout

    def build_contrast_controls(self) -> Optional[QHBoxLayout]:
        """The contrast slider of the movie, None for an RGB movie."""

        if self.rgb_mode:
            return None

        self.contrast_slider = QLabeledDoubleRangeSlider()
        self.contrast_slider.setSingleStep(0.001)
        self.contrast_slider.setTickInterval(0.001)
        self.contrast_slider.setOrientation(Qt.Horizontal)
        value_range, limits = self._contrast
        self.contrast_slider.setRange(*value_range)
        self.contrast_slider.setValue(limits)
        self.contrast_slider.valueChanged.connect(self.contrast_slider_action)

        layout = QHBoxLayout()
        layout.setContentsMargins(150, 5, 150, 5)
        layout.addWidget(QLabel("contrast: "))
        layout.addWidget(self.contrast_slider, 90)
        return layout

    def contrast_slider_action(self) -> None:
        """Recontrast the movie as the contrast slider moves."""

        self.vmin, self.vmax = self.contrast_slider.value()
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.fcanvas.canvas.draw_idle()

    def show_frame(self, framedata: int) -> None:
        """Draw a frame of the paused movie."""

        self.draw_frame(framedata)
        self.fcanvas.canvas.draw()

    def stop(self) -> None:
        """Pause the movie."""

        self.stop_btn.hide()
        self.start_btn.show()
        self.anim.pause()
        self.prev_frame_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)

    def start(self) -> None:
        """Play the movie."""

        self.start_btn.hide()
        self.stop_btn.show()
        self.prev_frame_btn.setEnabled(False)
        self.next_frame_btn.setEnabled(False)
        self.anim.resume()

    def toggle_animation(self) -> None:
        """Play or pause the movie."""

        if self.stop_btn.isVisible():
            self.stop()
        else:
            self.start()

    def next_frame(self) -> None:
        """Show the next frame, looping to the first."""
        self.show_frame((self.framedata + 1) % self.len_movie)

    def prev_frame(self) -> None:
        """Show the previous frame, looping to the last."""
        self.show_frame((self.framedata - 1) % self.len_movie)

    def set_first_frame(self) -> None:
        """Pause on the first frame."""

        self.stop()
        self.show_frame(0)

    def set_last_frame(self) -> None:
        """Pause on the last frame that is not almost black."""

        self.stop()
        frame = len(self.stack) - 1
        while frame > 0 and (
            np.count_nonzero(self.stack[frame]) < 0.01 * self.stack[frame].size
        ):
            frame -= 1
        self.show_frame(frame)

    def update_speed(self) -> None:
        """Apply the framerate of the slider to the movie."""

        fps = self.speed_slider.value()
        self.anim_interval = int(1000 / max(1, fps))
        logger.debug(f"Framerate {fps} FPS: interval of {self.anim_interval} ms.")

        # The slider is built with the movie playing.
        try:
            self.anim._interval = self.anim_interval
            self.anim.event_source.interval = self.anim_interval
            if self.stop_btn.isVisible():  # Playing: restart the timer.
                self.anim.event_source.stop()
                self.anim.event_source.start()
        except Exception as e:
            logger.debug(f"Error updating animation interval: {e}")

    def release_resources(self) -> None:
        """
        Stop the movie and free its figure and the stack, breaking the
        reference cycles of the animation. The window may close while the
        stack is loading, before any of them exist.
        """

        try:
            self.stop()
        except Exception as e:
            logger.debug(f"Could not stop annotator cleanly: {e}")

        if getattr(self, "anim", None):
            try:
                self.anim.event_source.stop()
            except Exception as e:
                logger.debug(f"Could not stop animation event source: {e}")
            del self.anim

        figure = getattr(self, "fig", None)
        if figure is not None:
            try:
                plt.close(figure)
            except Exception as e:
                logger.debug(f"Could not close fig: {e}")

        if hasattr(self, "stack"):
            del self.stack

        super().release_resources()
