from PyQt5.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QAction,
    QWidget,
)
from celldetective.gui.interactive_timeseries_viewer import InteractiveEventViewer
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence, QCloseEvent

import numpy as np
from matplotlib.cm import tab10
from typing import Tuple, Any
from celldetective.gui.annotator_common import (
    AnimationMixin,
    EventAnnotationMixin,
    complete_event_columns,
    confirm,
    event_class_columns,
    event_columns,
    prefill_class,
    signal_limits,
    split_by_frame,
)
from celldetective.utils.event_status import color_events, write_event_status
from celldetective.gui.base_annotator import BaseAnnotator
import logging
from celldetective.gui.base.threads import start_tracked
from celldetective.gui.base.utils import flush_layout_events

logger = logging.getLogger("celldetective")


class StackLoaderThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, annotator: QWidget) -> None:
        """
        Initialize the loader thread.

        Parameters
        ----------
        annotator : QWidget
            The annotator instance.
        """
        super().__init__()
        self.annotator = annotator
        self._is_cancelled = False

    def stop(self):
        """Stop the thread."""
        self._is_cancelled = True

    def run(self):
        """Run the thread."""

        def callback(progress: int, status: str = "") -> bool:
            """
            Callback to update progress.

            Parameters
            ----------
            progress : int
                Progress percentage.
            status : str, optional
                Status message. Default is "".

            Returns
            -------
            bool
                True if should continue, False if cancelled.
            """
            if self._is_cancelled:
                return False
            self.progress.emit(progress)
            if status:
                self.status_update.emit(status)
            return True

        try:
            self.annotator.prepare_stack(progress_callback=callback)
            if not self._is_cancelled:
                self.finished.emit()
        except Exception as e:
            logger.error(f"Error in loader thread: {e}")
            self.finished.emit()


class EventAnnotator(EventAnnotationMixin, AnimationMixin, BaseAnnotator):
    """
    Annotate an event per track, on the movie of a position.

    Parameters
    ----------
    *args
        Variable length argument list.
    lazy_load : bool, optional
        If True, do not start loading the stack immediately. Default is False.
    **kwargs
        Arbitrary keyword arguments.
    """

    # Set while the stack loads in the background.
    _loader_thread = None
    _progress_dialog = None
    # The interactive signal plotter, once opened.
    plotter = None

    def __init__(self, *args, lazy_load=False, **kwargs):
        """
        Initialize the annotator.

        Parameters
        ----------
        *args
            Variable length argument list.
        lazy_load : bool, optional
            Whether to load images lazily. Default is False.
        **kwargs
            Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Signal annotator")

        # default params
        self.class_name = "class"
        self.time_name = "t0"
        self.status_name = "status"

        if not self.proceed:
            self.close()
        elif not lazy_load:
            self._start_threaded_loading()

    def _start_threaded_loading(self):
        """Start loading the stack in a background thread with progress dialog."""
        from celldetective.gui.base.components import CelldetectiveProgressDialog

        self._progress_dialog = CelldetectiveProgressDialog(
            "Loading stack...", "Cancel", 0, 100, self
        )
        self._progress_dialog.setWindowTitle("Loading")
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)

        self._loader_thread = StackLoaderThread(self)
        self._loader_thread.progress.connect(self._on_load_progress)
        self._loader_thread.status_update.connect(self._on_load_status)
        self._loader_thread.finished.connect(self._on_load_finished)
        self._progress_dialog.canceled.connect(self._on_load_canceled)

        start_tracked(self._loader_thread)

    def _on_load_progress(self, value: int) -> None:
        """
        Update progress dialog.

        Parameters
        ----------
        value : int
            Progress value (0-100).
        """
        if self._progress_dialog:
            self._progress_dialog.setValue(value)

    def _on_load_status(self, status: str) -> None:
        """
        Update progress dialog label.

        Parameters
        ----------
        status : str
            Status message to display.
        """
        if self._progress_dialog:
            self._progress_dialog.setLabelText(status)

    def _on_load_canceled(self):
        """Handle cancel button click."""
        if self._loader_thread:
            self._loader_thread.stop()
            self._loader_thread.wait(3000)
        self.close()

    def _on_load_finished(self):
        """Called when loading completes."""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        self._loader_thread = None
        self.finalize_init()

    def finalize_init(self):
        """Finalize initialization after loading stack."""
        self.frame_lbl = QLabel("frame: ")
        self.looped_animation()
        self.init_event_buttons()
        self.populate_window()
        self.outliers_check.hide()

    def populate_options_layout(self):
        """Populate the options layout."""

        # clear options hbox
        for i in reversed(range(self.options_hbox.count())):
            self.options_hbox.itemAt(i).widget().setParent(None)

        btn_hbox = QHBoxLayout()
        for button in self.class_options:
            btn_hbox.addWidget(button, 25, alignment=Qt.AlignCenter)

        time_option_hbox = QHBoxLayout()
        time_option_hbox.setContentsMargins(0, 5, 100, 10)
        time_option_hbox.addWidget(self.time_of_interest_label, 10)
        time_option_hbox.addWidget(self.time_of_interest_le, 15)
        time_option_hbox.addWidget(QLabel(""), 75)

        options_layout = QVBoxLayout()
        options_layout.addLayout(btn_hbox)
        options_layout.addLayout(time_option_hbox)
        self.options_hbox.addLayout(options_layout)

        self.hide_annotation_buttons()

    def populate_window(self):
        """
        Create the multibox design.

        """

        super().populate_window()
        self.populate_options_layout()

        self.no_event_shortcut = self.add_selection_shortcut(
            QKeySequence("n"), self.shortcut_no_event
        )

        # Right side
        self.right_panel.addLayout(self.build_animation_controls(), 5)
        self.right_panel.addWidget(self.fcanvas, 90)
        contrast_controls = self.build_contrast_controls()
        if contrast_controls is not None:
            self.right_panel.addLayout(contrast_controls, 5)

        if self.class_choice_cb.currentText() != "":
            self.compute_status_and_colors(0)

        flush_layout_events(self)

        # Add Menu for Interactive Plotter
        menubar = self.menuBar()
        viewMenu = menubar.addMenu("View")

        openPlotterAct = QAction("Interactive Plotter", self)
        openPlotterAct.setShortcut("Ctrl+P")
        openPlotterAct.setStatusTip("Open interactive signal plotter for corrections")
        openPlotterAct.triggered.connect(self.launch_interactive_viewer)
        viewMenu.addAction(openPlotterAct)

    def launch_interactive_viewer(self):
        """Launch the interactive signal viewer."""
        if self.plotter is None or not self.plotter.isVisible():
            label = None
            if self.class_name.startswith("class_"):
                label = self.class_name.replace("class_", "")

            # Create with shared DF and callback
            self.plotter = InteractiveEventViewer(
                self.trajectories_path,
                df=self.df_tracks,
                event_label=label,
                callback=self.on_viewer_update,
                parent=self,
            )
        self.plotter.show()
        self.plotter.activateWindow()

    def on_viewer_update(self):
        """Callback from interactive viewer to refresh annotator."""
        self.compute_status_and_colors(0)
        self.update_scatters()
        self.fcanvas.canvas.draw_idle()

    def update_scatters(self):
        """Move and color the markers to the current frame."""
        positions = self.positions[self.framedata]
        colors = self.colors[self.framedata]
        self.status_scatter.set_offsets(positions)
        self.status_scatter.set_color(colors[:, 1])
        self.class_scatter.set_offsets(positions)
        self.class_scatter.set_edgecolor(colors[:, 0])

    def compute_status_and_colors(self, i: int) -> None:
        """
        Compute the status and colors of the cells.

        Parameters
        ----------
        i : int
            Current frame index.
        """
        self.class_name = self.class_choice_cb.currentText() or "class"
        self.time_name, self.status_name = event_columns(self.class_name)

        complete_event_columns(
            self.df_tracks,
            self.class_name,
            self.time_name,
            self.status_name,
            make_status=self.make_status_column,
            remake_empty_status=True,
        )
        color_events(self.df_tracks, self.class_name, self.status_name)

        self.extract_scatter_from_trajectories()
        if len(self.selection) > 0:
            self.select_single_cell(self.selection[0][0], self.selection[0][1])

        self.fcanvas.canvas.draw()

    def write_new_event_class(self):
        """Add the event class named in the new class dialog, prefilled as chosen."""

        name = self.class_name_le.text()
        target_class = f"class_{name}" if name else "class"
        target_time, target_status = event_columns(target_class)

        if target_class in self.df_tracks.columns and not confirm(
            "This event name already exists. If you proceed,\nall annotated data will be rewritten. Do you wish to continue?"
        ):
            return

        fill_option = prefill_class(self.class_option_rb)
        self.df_tracks.loc[:, target_class] = fill_option
        self.df_tracks.loc[:, target_time] = 0.1 if fill_option == 0 else -1
        # A rewritten class keeps its status column, which selecting it would
        # not remake: derive it from the new class and time now.
        write_event_status(self.df_tracks, target_class, target_time, target_status)

        self.class_choice_cb.clear()
        self.class_cols = event_class_columns(self.df_tracks.columns)
        self.class_choice_cb.addItems(self.class_cols)
        self.class_choice_cb.setCurrentIndex(self.class_choice_cb.findText(target_class))

        self.newClassWidget.close()

    def track_rows(self):
        """Rows of the track of interest."""
        return self.df_tracks["TRACK_ID"] == self.track_of_interest

    def current_annotation(self) -> Tuple[int, float]:
        """Class and time of the track of interest."""
        row = self.df_tracks.loc[self.track_rows()].iloc[0]
        return row[self.class_name], row[self.time_name]

    def apply_modification(self):
        """Apply modification to the event."""

        annotation = self.chosen_annotation()
        if annotation is None:
            return
        cclass, t0 = annotation
        track_mask = self.track_rows()

        # Record the previous class/time so the log shows what actually changed
        old_class, old_t0 = self.current_annotation()

        self.df_tracks.loc[track_mask, self.class_name] = cclass
        self.df_tracks.loc[track_mask, self.time_name] = t0

        self.annotation_log.append(
            f"TRACK_ID {self.track_of_interest}: {self.class_name} {old_class}->{cclass}, "
            f"{self.time_name} {old_t0}->{t0}"
        )

        write_event_status(
            self.df_tracks,
            self.class_name,
            self.time_name,
            self.status_name,
            rows=track_mask,
            recently_modified=True,
        )

        self.extract_scatter_from_trajectories()
        self.give_cell_information()

        self.reset_correction()
        self.selection.pop(0)

    def make_status_column(self):
        """Create the status column based on class and time."""

        logger.info(f"Generating status information for class `{self.class_name}` and time `{self.time_name}`...")
        write_event_status(self.df_tracks, self.class_name, self.time_name, self.status_name)

    def plot_signals(self):
        """Plot the selected signals."""

        range_values = []
        track = self.df_tracks.loc[self.track_rows()]

        try:
            for i, (cb, line) in enumerate(zip(self.signal_choice_cb, self.lines)):

                signal_choice = cb.currentText()
                n_cut = 35
                label = signal_choice
                if len(label) > n_cut:
                    label = label[: (n_cut - 3)] + "..."
                line.set_label(label)

                if signal_choice == "--":
                    line.set_xdata([])
                    line.set_ydata([])
                else:
                    xdata = track["FRAME"].to_numpy()
                    ydata = track[signal_choice].to_numpy()
                    range_values.extend(ydata)

                    valid = ydata == ydata  # not NaN
                    line.set_xdata(xdata[valid])
                    line.set_ydata(ydata[valid])
                    line.set_color(tab10(i / 3.0))

            # Fitted to the cell, before the time line spans them and the plot
            # is drawn; to the whole table only when the cell has no value.
            limits = signal_limits(range_values)
            if limits is not None:
                self.set_signal_ylim(*limits)
            else:
                self.configure_ylims()

            min_val, max_val = self.cell_ax.get_ylim()
            t0 = track[self.time_name].to_numpy()[0]
            self.line_dt.set_xdata([t0, t0])
            self.line_dt.set_ydata([min_val, max_val])

            self.cell_ax.legend(fontsize=8)
            self.cell_fcanvas.canvas.draw()
        except Exception as e:
            logger.warning(f"Failed to update cell plot: {e}")

    def extract_scatter_from_trajectories(self):
        """Extract scatter data from trajectories using efficient groupby."""

        frames = split_by_frame(self.df_tracks, self.len_movie)
        self.positions = [cells[["x_anim", "y_anim"]].to_numpy() for cells in frames]
        # A copy: the colors are written to highlight the selected cell, and
        # under copy-on-write pandas hands out read-only arrays.
        self.colors = [
            cells[["class_color", "status_color"]].to_numpy().copy() for cells in frames
        ]
        self.tracks = [cells["TRACK_ID"].to_numpy() for cells in frames]

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle close event.

        Parameters
        ----------
        event : QCloseEvent
            The close event.
        """
        # If the stack is still loading in the background, stop and join the
        # loader thread before tearing anything down. Otherwise it keeps running
        # prepare_stack against this (being-destroyed) widget — racing on
        # self.stack with the `del` below and firing finished -> finalize_init on
        # a dead window.
        loader = self._loader_thread
        if loader is not None:
            try:
                loader.progress.disconnect()
                loader.status_update.disconnect()
                loader.finished.disconnect()
            except (TypeError, RuntimeError):
                pass  # no connections / already deleted
            try:
                if loader.isRunning():
                    loader.stop()
                    loader.wait(5000)
            except RuntimeError:
                pass  # underlying C++ thread already gone
            self._loader_thread = None

        dlg = self._progress_dialog
        if dlg is not None:
            try:
                dlg.close()
            except (RuntimeError, AttributeError) as e:
                logger.debug(f"Could not close loading dialog: {e}")
            self._progress_dialog = None

        super().closeEvent(event)

    def looped_animation(self):
        """Show the movie with the cells marked, playing in a loop."""

        self.create_movie_axes()
        self.status_scatter = self.ax.scatter(
            self.positions[0][:, 0],
            self.positions[0][:, 1],
            marker="x",
            c=self.colors[0][:, 1],
            s=50,
            picker=True,
            pickradius=100,
        )
        self.class_scatter = self.ax.scatter(
            self.positions[0][:, 0],
            self.positions[0][:, 1],
            marker="o",
            facecolors="none",
            edgecolors=self.colors[0][:, 0],
            s=200,
        )
        self.start_animation()

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
        self.plot_signals()
        self.highlight_track_of_interest()

    def configure_ylims(self):
        """Configure y-axis limits for signal plots."""

        try:
            signals = [
                cb.currentText()
                for cb in self.signal_choice_cb
                if cb.currentText() != "--"
            ]
            if not signals:
                return

            def padding(signal):
                values = self.df_tracks[signal]
                return (values.max() - values.min()) * 0.05 or 0.05

            min_values = [np.nanpercentile(self.df_tracks[s].to_numpy(), 1) for s in signals]
            max_values = [np.nanpercentile(self.df_tracks[s].to_numpy(), 99) for s in signals]
            lowest = int(np.argmin(min_values))
            highest = int(np.argmax(max_values))
            self.cell_ax.set_ylim(
                min_values[lowest] - padding(signals[lowest]),
                max_values[highest] + padding(signals[highest]),
            )
        except Exception as e:
            logger.warning(f"Failed to update cell plot y-limits: {e}")

    def draw_frame(self, framedata: int) -> Tuple[Any, ...]:
        """
        Update plot elements at each timestep of the loop.

        Parameters
        ----------
        framedata : int
            The current frame index.

        Returns
        -------
        tuple
            The updated plot elements.
        """

        self.framedata = framedata
        self.frame_lbl.setText(f"frame: {self.framedata}")
        self.im.set_array(self.stack[self.framedata])
        self.update_scatters()
        return (
            self.im,
            self.status_scatter,
            self.class_scatter,
        )

    def give_cell_information(self):
        """Display cell information."""

        row = self.df_tracks.loc[self.track_rows()].iloc[0]
        cell_info = (
            f"cell: {self.track_of_interest}\n"
            f"class: {row[self.class_name]}\n"
            f"time of interest: {row[self.time_name]}\n"
        )
        self.cell_info.setText(cell_info)

    def save_trajectories(self):
        """Save trajectories to file."""

        if self.normalized_signals:
            self.normalize_features_btn.click()
        if self.selection:
            self.cancel_selection()

        self.df_tracks = self.df_tracks.drop(
            self.df_tracks[self.df_tracks[self.class_name] > 2].index
        )
        self.df_tracks.to_csv(self.trajectories_path, index=False)
        logger.info("Table successfully exported...")
        self.write_annotation_log(
            "EVENT ANNOTATION (manual)", f"class_name: {self.class_name}"
        )
        if self.class_choice_cb.currentText() != "":
            self.compute_status_and_colors(0)
        self.extract_scatter_from_trajectories()
