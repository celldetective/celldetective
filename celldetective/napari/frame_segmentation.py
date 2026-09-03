"""
Segment the frame on screen, from inside the napari correction viewer.

The panel embeds the very channel-selection widget the main window is built on
-- :class:`~celldetective.gui.base.model_channel_selection.ModelChannelSelection`,
one dropdown per input slot of the chosen model -- and adds whatever inference
parameters that model type actually takes. Values are seeded from the model's
``config_input.json`` -- including any ``selected_channels`` mapping already set
in the main window -- but edits stay local to the napari session and are never
written back, so trying settings out here cannot silently change the next
full-stack run.
"""

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PyQt5.QtCore import QEvent, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from celldetective import get_logger
from celldetective.gui.base.model_channel_selection import ModelChannelSelection
from celldetective.utils.experiment import (
    extract_experiment_channels,
    extract_experiment_from_position,
    get_spatial_calibration,
)
from celldetective.utils.model_loaders import locate_segmentation_model

logger = get_logger(__name__)

# Offered in the model dropdown when nothing is installed, so that the panel --
# and with it the whole viewer -- still builds.
NO_MODEL = "(no segmentation model found)"

# How many prepared models the panel keeps alive at once. Each one is a whole
# StarDist / Cellpose network, so the cache trades a few hundred MB for not
# reloading when the user goes back and forth between two settings.
MAX_CACHED_MODELS = 2

# Workers that have been started and not yet finished. A QThread must outlive its
# own `run()`, and destroying one that is still running is a fatal error in Qt, so
# the thread objects are parented to nothing and held here instead of on the panel
# -- that way closing the viewer cannot take a running thread down with it.
_LIVE_WORKERS: Set["_SegmentationWorker"] = set()


def _read_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Read a segmentation model's input configuration, without downloading it.

    Parameters
    ----------
    model_name : str
        Name of the segmentation model.

    Returns
    -------
    dict or None
        The parsed ``config_input.json``, or None when the model is not present
        locally (it lives in the remote repository and has not been fetched yet)
        or its configuration cannot be read.
    """

    try:
        model_path = locate_segmentation_model(model_name, download=False)
    except Exception as e:
        logger.debug(f"Could not locate model '{model_name}': {e}")
        return None
    if model_path is None:
        return None

    config_path = os.path.join(model_path, "config_input.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except Exception as e:
        logger.warning(f"Could not read the configuration of '{model_name}': {e}")
        return None


def available_segmentation_models(population: str) -> List[str]:
    """
    List the models offered in the panel: population-specific first, then generic.

    Parameters
    ----------
    population : str
        The population whose model family to list, e.g. ``"targets"``.

    Returns
    -------
    list of str
        Model names without duplicates. Never empty: a placeholder is returned
        when nothing can be listed, so the dropdown always builds.
    """

    from celldetective.utils.model_getters import get_segmentation_models_list

    # "target" / "effector" are used interchangeably with their plurals across the
    # viewer, but the model directories are only ever named in the plural.
    mode = population if population.endswith("s") else f"{population}s"

    models: List[str] = []
    for family in (mode, "generic"):
        try:
            # cleanup=False: listing must not create the category directory nor
            # delete local model folders that happen to lack a config_input.json.
            # Opening a viewer is not the moment to be rewriting the model tree.
            models.extend(
                get_segmentation_models_list(
                    mode=family, return_path=False, cleanup=False
                )
            )
        except Exception as e:
            # Listing reaches out to the model repository; being offline must not
            # stop the viewer from opening.
            logger.warning(f"Could not list the '{family}' segmentation models: {e}")

    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]
    return models or [NO_MODEL]


def _fit_to_layer_dtype(
    labels: np.ndarray, dtype: np.dtype, remedy: str = ""
) -> np.ndarray:
    """
    Cast labels to the segmentation layer's type, refusing to wrap round.

    napari keeps the labels in whatever integer type they were read as, often
    ``uint16``. A plain assignment of a larger value wraps silently, which merges
    unrelated cells under one identifier - the sort of corruption that is only
    noticed once it is in the measurements.

    Parameters
    ----------
    labels : ndarray
        The labels to write.
    dtype : numpy.dtype
        The layer's integer type.
    remedy : str, optional
        Appended to the error message to say what the user can do about it.

    Returns
    -------
    ndarray
        `labels`, cast to `dtype`.

    Raises
    ------
    ValueError
        If any label is too large for `dtype`.
    """

    info = np.iinfo(dtype)
    highest = int(labels.max()) if labels.size else 0
    if highest > info.max:
        message = (
            f"The segmentation reaches label {highest}, more than the {dtype} "
            f"segmentation layer can hold ({info.max})."
        )
        raise ValueError(f"{message} {remedy}".strip())
    return labels.astype(dtype, copy=False)


class _FloatEdit(QLineEdit):
    """A line edit accepting a single float, blank meaning "use the model's value"."""

    def __init__(
        self,
        value: Optional[float] = None,
        parent=None,
        bottom: Optional[float] = None,
    ):
        super().__init__(parent)
        validator = QDoubleValidator()
        if bottom is not None:
            # Keeps a negative out of the field entirely; a value of exactly the
            # bottom still gets through, so anything that must be strictly greater
            # is checked again before the run starts.
            validator.setBottom(bottom)
        self.setValidator(validator)
        if value is not None:
            self.setText(str(value))

    def value(self) -> Optional[float]:
        """Return the entered number, or None when the field is blank or invalid."""
        text = self.text().strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None


class _SegmentationWorker(QThread):
    """
    Load the model and segment one frame, off the GUI thread.

    Cancellation is cooperative and checked between phases: neither StarDist nor
    Cellpose exposes a hook to interrupt a forward pass, so a cancel raised once
    inference has started does not stop it -- it lets the interface go back to
    normal and discards the result when it lands. Cancelling while the model is
    still loading, which is the slow part the first time round, does take effect
    before any inference happens.

    Parameters
    ----------
    stack : ndarray or dask.array.Array
        The stack the frame is read from. It is only materialised inside
        :meth:`run`, so that a lazily loaded movie is read from disk on this
        thread rather than freezing the viewer.
    frame_index : int
        Index of the frame to segment along the stack's first axis.
    prepared : PreparedSegmentationModel or None
        An already-prepared model to reuse; when None it is built from
        ``prepare_kwargs``.
    prepare_kwargs : dict
        Arguments for :func:`prepare_segmentation_model`, used when `prepared`
        is None.
    """

    #: Emitted with ``(prepared_model, labels)`` when the frame was segmented.
    succeeded = pyqtSignal(object, object)
    #: Emitted with a message the user should see.
    failed = pyqtSignal(str)
    #: Emitted with the name of the phase being entered.
    stage = pyqtSignal(str)

    def __init__(
        self,
        stack,
        frame_index: int,
        prepared,
        prepare_kwargs: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self._stack = stack
        self._frame_index = frame_index
        self._prepared = prepared
        self._prepare_kwargs = prepare_kwargs
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the worker to stop at the next phase boundary."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return self._cancelled

    def run(self) -> None:
        """Read the frame, prepare the model if needed, then segment."""

        from celldetective.segmentation import (
            prepare_segmentation_model,
            segment_frame,
        )

        try:
            self.stage.emit("Reading the frame…")
            frame = np.asarray(self._stack[self._frame_index])

            prepared = self._prepared
            if prepared is None:
                self.stage.emit("Loading the model…")
                if self._cancelled:
                    return
                prepared = prepare_segmentation_model(**self._prepare_kwargs)
                if prepared is None:
                    model_name = self._prepare_kwargs.get("model_name", "the model")
                    self.failed.emit(
                        f"Model '{model_name}' could not be loaded. See the log for details."
                    )
                    return

            if self._cancelled:
                return

            self.stage.emit("Segmenting the frame…")
            labels = segment_frame(frame, prepared)
        except ValueError as e:
            # Raised when none of the mapped channels reach the model.
            self.failed.emit(str(e))
            return
        except Exception as e:
            logger.exception("Single-frame segmentation failed.")
            self.failed.emit(f"Segmentation failed: {e}")
            return

        if self._cancelled:
            return
        self.succeeded.emit(prepared, labels)


class FrameSegmentationPanel(QWidget):
    """
    Dock panel that runs a segmentation model on the frame currently displayed.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer holding the ``segmentation`` labels layer to write into.
    stack : ndarray
        The image stack being displayed (TYXC), used as the pixel source so the
        panel does not depend on how the image layers happen to be named.
    position : str
        The position directory, used to locate the experiment configuration.
    population : str
        The population being segmented, selecting the model family to offer.
    """

    def __init__(self, viewer, stack, position: str, population: str, parent=None):
        super().__init__(parent)

        self.viewer = viewer
        self.stack = stack
        self.position = position
        self.population = population

        # Prepared models, most recently used last. Keyed only on what actually
        # goes into building one -- the model, the channel mapping and the
        # rescaling -- because the Cellpose thresholds are inference-time
        # arguments and re-keying on them would reload a whole network every time
        # a threshold is nudged. Bounded, since each entry is a full network.
        self._prepared: "OrderedDict[Any, Any]" = OrderedDict()

        # The run in flight, and what it was asked to do.
        self._worker: Optional[_SegmentationWorker] = None
        self._pending: Optional[Dict[str, Any]] = None

        # Set once the panel is being torn down, so a result that lands late is
        # dropped instead of being written into a half-destroyed viewer.
        self._closing = False
        self._watched_window: Optional[QWidget] = None

        self.experiment = extract_experiment_from_position(position)
        try:
            self.exp_channels = list(extract_experiment_channels(self.experiment)[0])
        except Exception as e:
            logger.warning(f"Could not read the experiment channels: {e}")
            self.exp_channels = []
        try:
            self.spatial_calibration = get_spatial_calibration(self.experiment)
        except Exception as e:
            logger.warning(f"Could not read the spatial calibration: {e}")
            self.spatial_calibration = None

        self.channel_selection: Optional[ModelChannelSelection] = None
        self.diameter_le: Optional[_FloatEdit] = None
        self.cellprob_le: Optional[_FloatEdit] = None
        self.flow_le: Optional[_FloatEdit] = None
        self.cell_size_le: Optional[_FloatEdit] = None
        self.config: Optional[Dict[str, Any]] = None

        self._build()
        self._install_close_hook()

    def _install_close_hook(self) -> None:
        """
        Arrange for :meth:`closeEvent` to fire when the viewer window closes.

        Qt delivers a close event to top-level windows only, and this panel lives
        inside a dock widget, so on its own it would never see one: the viewer
        would simply be destroyed underneath it, taking a running worker with it.
        Watching the napari window for its own close event, and the application
        for its shutdown, gives the panel the chance to stop that worker first.
        """

        try:
            window = self.viewer.window._qt_window
        except Exception as e:
            logger.debug(f"Could not reach the napari window to watch it close: {e}")
            window = None

        if window is not None:
            window.installEventFilter(self)
            self._watched_window = window

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_worker)

    def eventFilter(self, watched, event) -> bool:
        """Turn the viewer window's close into this panel's own close."""
        if (
            watched is self._watched_window
            and event.type() == QEvent.Close
            and not self._closing
        ):
            # An event filter runs *before* the window's own `closeEvent`, and
            # napari's asks for confirmation: the user may still call the close
            # off, in which case the viewer stays open and this panel must stay
            # with it. Settle it on the next trip through the event loop, once
            # the window has had its say.
            QTimer.singleShot(0, self._close_if_window_closed)
        return super().eventFilter(watched, event)

    def _close_if_window_closed(self) -> None:
        """Close the panel, but only if the viewer window really did close."""

        window = self._watched_window
        if window is None:
            return
        try:
            still_open = window.isVisible()
        except RuntimeError:
            # The C++ window is already gone, so the close went through.
            still_open = False
        if not still_open:
            self.close()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Lay out the panel and fill it for the initially selected model."""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("model:"), 30)
        self.model_cb = QComboBox()
        self.model_cb.addItems(available_segmentation_models(self.population))
        self.model_cb.setToolTip(
            "Segmentation model to run on the frame currently displayed."
        )
        model_row.addWidget(self.model_cb, 70)
        outer.addLayout(model_row)

        self.channel_box = QGroupBox("channels")
        self.channel_layout = QVBoxLayout(self.channel_box)
        self.channel_layout.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self.channel_box)

        self.param_box = QGroupBox("parameters")
        self.param_form = QFormLayout(self.param_box)
        self.param_form.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self.param_box)

        self.replace_cb = QCheckBox("Replace the labels on this frame")
        self.replace_cb.setChecked(True)
        self.replace_cb.setToolTip(
            "Unticked, existing labels are kept and the new ones only fill the\n"
            "background, so manual corrections on this frame survive."
        )
        outer.addWidget(self.replace_cb)

        self.run_btn = QPushButton("Segment this frame")
        self.run_btn.clicked.connect(self._on_run_clicked)
        outer.addWidget(self.run_btn)

        # Indeterminate: there is no progress to report from inside a forward
        # pass, so the bar says "working" rather than how far along it is.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        outer.addWidget(self.progress)

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.hide()
        outer.addWidget(self.status_lbl)

        self.model_cb.currentTextChanged.connect(self._reload_model)
        self._reload_model(self.model_cb.currentText())

    def _clear(self, layout) -> None:
        """Remove every row from a layout, deleting the widgets it held."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Unparent first: `deleteLater` only schedules the deletion, and
                # until it runs the widget would still be shown by the box it was
                # just taken out of.
                widget.setParent(None)
                widget.deleteLater()

    def _reload_model(self, model_name: str) -> None:
        """
        Rebuild the channel and parameter rows for the selected model.

        Parameters
        ----------
        model_name : str
            The newly selected model.
        """

        self._clear(self.channel_layout)
        self._clear(self.param_form)
        self.channel_selection = None
        self.diameter_le = None
        self.cellprob_le = None
        self.flow_le = None
        self.cell_size_le = None

        if not model_name or model_name == NO_MODEL:
            self.config = None
            self.channel_layout.addWidget(QLabel("No model available."))
            self.run_btn.setEnabled(False)
            return

        self.run_btn.setEnabled(True)
        self.config = _read_model_config(model_name)

        if self.config is None:
            # A repository model that has not been fetched yet: its channels and
            # parameters are unknown until it lands on disk.
            self.channel_layout.addWidget(
                QLabel("Not downloaded yet.\nIt will be fetched on the first run,\nusing the model's own settings.")
            )
            return

        self._build_channel_rows()
        self._build_parameter_rows()

    def _build_channel_rows(self) -> None:
        """
        One dropdown per model input slot, seeded from the stored mapping.

        The rows are the main window's own channel-selection widget, so a model's
        inputs are mapped the same way wherever it is run from -- including the
        ``selected_channels`` mapping already saved there, which is what makes a
        model like ``CP_cyto3`` usable at all: it declares channel names no real
        experiment is named after.
        """

        stored = self.config.get("selected_channels")
        self.channel_selection = ModelChannelSelection(
            required_channels=list(self.config.get("channels", [])),
            available_channels=self.exp_channels,
            selected_channels=stored if isinstance(stored, list) else None,
        )
        self.channel_layout.addWidget(self.channel_selection)

    def _build_parameter_rows(self) -> None:
        """Expose the inference parameters that this model type actually takes."""

        model_type = self.config.get("model_type")

        if model_type == "cellpose":
            self.diameter_le = _FloatEdit(self.config.get("diameter"))
            self.diameter_le.setToolTip(
                "Cellpose object diameter, in pixels. Blank uses the model's own value."
            )
            self.param_form.addRow(QLabel("diameter [px]:"), self.diameter_le)

            self.cellprob_le = _FloatEdit(self.config.get("cellprob_threshold"))
            self.cellprob_le.setToolTip(
                "Lower to keep more objects, raise to keep fewer."
            )
            self.param_form.addRow(QLabel("cell probability:"), self.cellprob_le)

            self.flow_le = _FloatEdit(self.config.get("flow_threshold"))
            self.flow_le.setToolTip(
                "Maximum flow error allowed per mask. Raise to keep more objects."
            )
            self.param_form.addRow(QLabel("flow threshold:"), self.flow_le)

        if "cell_size_um" in self.config:
            trained = self.config["cell_size_um"]
            self.cell_size_le = _FloatEdit(
                self.config.get("target_cell_size_um", trained), bottom=0.0
            )
            self.cell_size_le.setToolTip(
                f"Typical object size in these images, in µm.\n"
                f"The model was trained on objects of about {trained} µm;\n"
                f"the frame is rescaled to match."
            )
            self.param_form.addRow(QLabel("cell size [µm]:"), self.cell_size_le)

        if self.param_form.count() == 0:
            self.param_form.addRow(
                QLabel(f"No adjustable parameters for a {model_type} model.")
            )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _selected_channels(self) -> Optional[List[str]]:
        """The channel mapping as currently set, or None when the model is unknown."""
        if self.channel_selection is None:
            return None
        return self.channel_selection.selected_channels() or None

    def _failed(self, message: str) -> None:
        """
        Report a failure without tearing down the viewer.

        Parameters
        ----------
        message : str
            Shown to the user and written to the log.
        """

        logger.error(message)
        self.viewer.status = message
        try:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText(message)
            box.setWindowTitle("Segmentation")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
        except Exception as e:
            logger.debug(f"Could not show the segmentation error dialog: {e}")

    def _set_running(self, running: bool) -> None:
        """
        Put the panel into its working or idle state.

        Parameters
        ----------
        running : bool
            True while a segmentation is in flight.
        """

        self.model_cb.setEnabled(not running)
        self.replace_cb.setEnabled(not running)
        if self.channel_selection is not None:
            self.channel_selection.setEnabled(not running)
        for edit in (
            self.diameter_le,
            self.cellprob_le,
            self.flow_le,
            self.cell_size_le,
        ):
            if edit is not None:
                edit.setEnabled(not running)

        self.progress.setVisible(running)
        self.status_lbl.setVisible(running)
        if running:
            self.run_btn.setText("Cancel")
            self.run_btn.setToolTip(
                "Stop the run. Inference cannot be interrupted once it has\n"
                "started, so the result is discarded when it lands."
            )
        else:
            self.run_btn.setText("Segment this frame")
            self.run_btn.setToolTip("")
            # Stay disabled when there is nothing to run.
            model_name = self.model_cb.currentText()
            self.run_btn.setEnabled(bool(model_name) and model_name != NO_MODEL)
            self.status_lbl.setText("")

    def _stop_worker(self) -> None:
        """
        Detach and wind down the run in flight, without ever blocking on it.

        Inference cannot be interrupted, so the worker is asked to stop, given a
        moment to reach the next phase boundary, and then simply let go of: its
        signals are disconnected so nothing lands on a panel that is going away,
        and `_LIVE_WORKERS` keeps the thread object alive until `run()` actually
        returns. Waiting any longer would freeze the close; destroying the thread
        instead would be a fatal error in Qt.
        """

        self._closing = True
        worker = self._worker
        self._worker = None
        if worker is None:
            return

        worker.cancel()
        try:
            worker.stage.disconnect()
            worker.succeeded.disconnect()
            worker.failed.disconnect()
        except (TypeError, RuntimeError) as e:
            # Already disconnected, or the C++ object is gone: nothing to undo.
            logger.debug(f"Could not disconnect the segmentation worker: {e}")

        if worker.isRunning():
            worker.wait(2000)
        if worker.isRunning():
            logger.info(
                "Leaving a single-frame segmentation to finish in the background; "
                "its result will be discarded."
            )

    def closeEvent(self, event) -> None:
        """Stop a run in flight before the panel goes away."""
        self._stop_worker()
        if self._watched_window is not None:
            try:
                self._watched_window.removeEventFilter(self)
            except RuntimeError as e:
                logger.debug(f"Watched window already destroyed: {e}")
            self._watched_window = None
        # Networks can be hundreds of MB each; do not keep them alive through a
        # dangling reference to a closed panel.
        self._prepared.clear()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        """Bring the panel back to life if it is shown again after a close."""
        # `_stop_worker` latches `_closing` so that a result landing mid-teardown
        # is dropped rather than pushed into a dying viewer. A panel that is on
        # screen again is not dying, and leaving the latch set would make every
        # later run hang with its labels silently discarded.
        self._closing = False
        super().showEvent(event)

    def _on_run_clicked(self) -> None:
        """Start a segmentation, or cancel the one in flight."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.run_btn.setEnabled(False)
            self.status_lbl.setText("Cancelling…")
            self.viewer.status = "Cancelling the segmentation…"
            return
        self.segment_current_frame()

    def segment_current_frame(self) -> None:
        """
        Run the selected model on the frame the time slider is on.

        The work happens on a worker thread, so the viewer stays usable while it
        runs. The result is written straight into the segmentation layer; nothing
        reaches disk until the labels are saved.
        """

        if self._worker is not None and self._worker.isRunning():
            return

        model_name = self.model_cb.currentText()
        if not model_name or model_name == NO_MODEL:
            self._failed(
                "No segmentation model is available. Download or train one first."
            )
            return

        selected = self._selected_channels()
        if self.channel_selection is not None and self.channel_selection.is_empty():
            self._failed(
                "Every input channel is set to None. Assign at least one experiment "
                "channel to a model input."
            )
            return

        t = int(self.viewer.dims.current_step[0])

        target_cell_size = self.cell_size_le.value() if self.cell_size_le else None
        if target_cell_size is not None and target_cell_size <= 0:
            self._failed(
                "The cell size must be greater than zero. Clear the field to use "
                "the model's own value."
            )
            return

        diameter = self.diameter_le.value() if self.diameter_le else None
        cellprob = self.cellprob_le.value() if self.cellprob_le else None
        flow = self.flow_le.value() if self.flow_le else None

        # Only what the model is built from. The Cellpose thresholds and diameter
        # are passed to inference, not to the constructor, so they are re-applied
        # to a cached model instead of forcing it to be loaded again.
        cache_key = (
            model_name,
            tuple(selected) if selected is not None else None,
            target_cell_size,
        )

        cached = self._reuse_prepared(cache_key, diameter, cellprob, flow)

        # The GPU is left to napari's renderer: a single frame is quick on CPU,
        # and a TensorFlow context would compete for the VRAM the viewer is
        # already using.
        prepare_kwargs = dict(
            model_name=model_name,
            channels=self.exp_channels or None,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
            # The panel mirrors the main window, so it opts in to the mapping the
            # channel-selection dialog stored: with no channel rows yet (a model
            # fetched on this very run) that is what the pipeline would use.
            use_stored_mapping=True,
            selected_channels=selected,
            target_cell_size=target_cell_size,
            diameter=diameter,
            cellprob_threshold=cellprob,
            flow_threshold=flow,
        )

        self._pending = {"frame": t, "cache_key": cache_key, "model_name": model_name}

        # Deliberately unparented: a QThread destroyed while running aborts the
        # process, and a panel that is closing must be able to walk away from one.
        worker = _SegmentationWorker(
            stack=self.stack,
            frame_index=t,
            prepared=cached,
            prepare_kwargs=prepare_kwargs,
        )
        _LIVE_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _LIVE_WORKERS.discard(w))
        worker.finished.connect(worker.deleteLater)
        worker.stage.connect(self._on_stage)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker

        self._set_running(True)
        self.status_lbl.setText("Starting…")
        self.viewer.status = f"Segmenting frame {t} with '{model_name}'…"
        worker.start()

    def _reuse_prepared(
        self,
        cache_key,
        diameter: Optional[float],
        cellprob: Optional[float],
        flow: Optional[float],
    ):
        """
        Fetch a prepared model for these settings and re-apply the inference values.

        The Cellpose diameter and thresholds are arguments to the forward pass, not
        to the constructor, so a change to any of them only has to be written onto
        the model rather than causing a new one to be loaded. A field left blank
        means "use the model's own value", which is what
        ``PreparedSegmentationModel.config_defaults`` holds.

        Parameters
        ----------
        cache_key : tuple
            Identity of the model to look up.
        diameter, cellprob, flow : float or None
            The values entered in the panel, None meaning the model's own.

        Returns
        -------
        PreparedSegmentationModel or None
            The cached model, updated in place, or None when there is no hit.
        """

        prepared = self._prepared.get(cache_key)
        if prepared is None:
            return None

        # Most recently used last, so the eviction below drops the coldest entry.
        self._prepared.move_to_end(cache_key)

        if prepared.model_type == "cellpose":
            defaults = prepared.config_defaults
            prepared.diameter = (
                diameter if diameter is not None else defaults.get("diameter")
            )
            prepared.cellprob_threshold = (
                cellprob
                if cellprob is not None
                else defaults.get("cellprob_threshold")
            )
            prepared.flow_threshold = (
                flow if flow is not None else defaults.get("flow_threshold")
            )
        return prepared

    def _cache_prepared(self, cache_key, prepared) -> None:
        """Store a prepared model, evicting the coldest once the cache is full."""
        self._prepared[cache_key] = prepared
        self._prepared.move_to_end(cache_key)
        while len(self._prepared) > MAX_CACHED_MODELS:
            evicted, _ = self._prepared.popitem(last=False)
            logger.debug(f"Dropped the cached segmentation model for {evicted}.")

    def _on_stage(self, message: str) -> None:
        """Show the phase the worker has entered."""
        self.status_lbl.setText(message)
        self.viewer.status = message

    def _on_failed(self, message: str) -> None:
        """Report a worker failure."""
        self._failed(message)

    def _on_worker_finished(self) -> None:
        """Return the panel to its idle state once the thread has ended."""
        if self._closing:
            return
        cancelled = self._worker is not None and self._worker.cancelled
        self._worker = None
        self._set_running(False)
        if cancelled:
            self.viewer.status = "Segmentation cancelled."
            logger.info("Single-frame segmentation cancelled.")

    def _merged_labels(self, current: np.ndarray, new_labels: np.ndarray) -> np.ndarray:
        """
        Combine new labels with the ones already drawn on the frame.

        The incoming labels are pushed past the highest existing one so the two
        sets cannot collide, and only fill background, so manual corrections
        survive.

        Parameters
        ----------
        current : ndarray
            The labels currently on the frame.
        new_labels : ndarray
            The labels the model produced.

        Returns
        -------
        ndarray
            The merged labels, in `current`'s dtype.

        Raises
        ------
        ValueError
            If the merged labels would not fit the layer's integer type. Wrapping
            round silently would merge unrelated cells under one identifier.
        """

        offset = int(current.max())
        # int64 throughout: `new_labels` is typically uint16, and adding the offset
        # in its own dtype wraps round without a word of warning.
        incoming = np.where(new_labels > 0, new_labels.astype(np.int64) + offset, 0)
        merged = np.where(current > 0, current.astype(np.int64), incoming)
        return _fit_to_layer_dtype(
            merged,
            current.dtype,
            "Tick 'Replace the labels on this frame' to segment it afresh.",
        )

    def _record_undo(self, layer, t: int, before: np.ndarray, after: np.ndarray) -> None:
        """
        Push this write onto the labels layer's undo history, if napari lets us.

        Without this the frame can be segmented but not un-segmented: napari only
        records what its own painting tools do, so a bulk write would leave Ctrl+Z
        undoing whatever the user had done before instead. Best-effort - the
        history is private API, so a napari that has moved it simply gets no undo
        step rather than an error.

        Parameters
        ----------
        layer : napari.layers.Labels
            The layer being written to.
        t : int
            Index of the frame that changed.
        before, after : ndarray
            The frame's labels either side of the write.
        """

        try:
            changed = np.nonzero(before != after)
            if len(changed[0]) == 0:
                return
            indices = (np.full(changed[0].shape, t, dtype=np.intp),) + changed
            layer._save_history((indices, before[changed], after[changed]))
        except Exception as e:
            logger.debug(f"Could not record an undo step for the segmentation: {e}")

    def _on_succeeded(self, prepared, new_labels) -> None:
        """
        Write the labels the worker produced into the segmentation layer.

        Parameters
        ----------
        prepared : PreparedSegmentationModel
            The model used, cached so the next run with the same settings reuses it.
        new_labels : ndarray
            The labels for the frame that was segmented.
        """

        if self._closing:
            return

        pending = self._pending or {}
        t = pending.get("frame", 0)
        model_name = pending.get("model_name", "")
        cache_key = pending.get("cache_key")
        if cache_key is not None:
            self._cache_prepared(cache_key, prepared)

        # A model fetched during this very run: its configuration only reached the
        # disk once the worker had started, so the panel is still showing the
        # "not downloaded yet" placeholder with no channel or parameter rows.
        # Build them now, rather than leaving the user to cycle the dropdown to
        # get at a mapping the model has had all along.
        if (
            self.config is None
            and model_name
            and model_name == self.model_cb.currentText()
        ):
            self._reload_model(model_name)
            if self.config is not None and cache_key is not None:
                # Those rows feed the cache key, so the entry just stored is keyed
                # on a state that can never come back; drop it rather than hold a
                # few hundred MB of network alive for a key nothing will ask for.
                self._prepared.pop(cache_key, None)

        # The viewer may have been closed while the worker was running.
        try:
            layer = self.viewer.layers["segmentation"]
        except Exception as e:
            logger.debug(f"Segmentation layer is gone, dropping the result: {e}")
            return

        try:
            current = layer.data[t]
            before = current.copy()
            if self.replace_cb.isChecked():
                merged = _fit_to_layer_dtype(new_labels, current.dtype)
            else:
                merged = self._merged_labels(current, new_labels)
            current[...] = merged
        except ValueError as e:
            # Raised when the labels will not fit the layer's integer type.
            self._failed(str(e))
            return
        except Exception as e:
            self._failed(f"Could not write the labels into the viewer: {e}")
            return

        self._record_undo(layer, t, before, layer.data[t])
        layer.refresh()

        # Count the objects rather than reading the highest label: in merge mode
        # the incoming labels are offset, so the maximum is not a count.
        n_objects = int(np.count_nonzero(np.unique(layer.data[t])))
        message = f"Frame {t}: {n_objects} objects after segmenting with '{model_name}'."
        self.viewer.status = message
        logger.info(message)
