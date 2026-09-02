"""
Segment the frame on screen, from inside the napari correction viewer.

The panel mirrors the channel-selection dialog of the main window: one dropdown
per input slot of the chosen model, plus whatever inference parameters that
model type actually takes. Values are seeded from the model's
``config_input.json`` -- including any ``selected_channels`` mapping already set
in the main window -- but edits stay local to the napari session and are never
written back, so trying settings out here cannot silently change the next
full-stack run.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
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
from celldetective.utils.experiment import (
    extract_experiment_channels,
    extract_experiment_from_position,
    get_spatial_calibration,
)
from celldetective.utils.model_loaders import locate_segmentation_model

logger = get_logger()

# Offered in the model dropdown when nothing is installed, so that the panel --
# and with it the whole viewer -- still builds.
NO_MODEL = "(no segmentation model found)"

# The channel-selection dialog uses this spelling for an unused input slot, and
# it is what ends up in `selected_channels`, so match it exactly.
NO_CHANNEL = "None"


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

    models: List[str] = []
    for mode in (population, "generic"):
        try:
            models.extend(get_segmentation_models_list(mode=mode, return_path=False))
        except Exception as e:
            # Listing reaches out to the model repository; being offline must not
            # stop the viewer from opening.
            logger.warning(f"Could not list the '{mode}' segmentation models: {e}")

    seen = set()
    models = [m for m in models if not (m in seen or seen.add(m))]
    return models or [NO_MODEL]


class _FloatEdit(QLineEdit):
    """A line edit accepting a single float, blank meaning "use the model's value"."""

    def __init__(self, value: Optional[float] = None, parent=None):
        super().__init__(parent)
        self.setValidator(QDoubleValidator())
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
    frame : ndarray
        The single multichannel image to segment.
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

    def __init__(self, frame, prepared, prepare_kwargs: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._frame = frame
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
        """Prepare the model if needed, then segment the frame."""

        from celldetective.segmentation import (
            prepare_segmentation_model,
            segment_frame,
        )

        try:
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
            labels = segment_frame(self._frame, prepared)
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

        # One entry per (model, channel mapping, parameters) actually used, so
        # that repeated runs with the same settings only pay for inference.
        self._prepared: Dict[Any, Any] = {}

        # The run in flight, and what it was asked to do. Kept on the panel so
        # the thread is not garbage collected while it works.
        self._worker: Optional[_SegmentationWorker] = None
        self._pending: Optional[Dict[str, Any]] = None

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

        self.channel_cbs: List[QComboBox] = []
        self.diameter_le: Optional[_FloatEdit] = None
        self.cellprob_le: Optional[_FloatEdit] = None
        self.flow_le: Optional[_FloatEdit] = None
        self.cell_size_le: Optional[_FloatEdit] = None
        self.config: Optional[Dict[str, Any]] = None

        self._build()

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
        self.channel_form = QFormLayout(self.channel_box)
        self.channel_form.setContentsMargins(8, 8, 8, 8)
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

    def _clear(self, form: QFormLayout) -> None:
        """Remove every row from a form layout."""
        while form.count():
            item = form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _reload_model(self, model_name: str) -> None:
        """
        Rebuild the channel and parameter rows for the selected model.

        Parameters
        ----------
        model_name : str
            The newly selected model.
        """

        self._clear(self.channel_form)
        self._clear(self.param_form)
        self.channel_cbs = []
        self.diameter_le = None
        self.cellprob_le = None
        self.flow_le = None
        self.cell_size_le = None

        if not model_name or model_name == NO_MODEL:
            self.config = None
            self.channel_form.addRow(QLabel("No model available."))
            self.run_btn.setEnabled(False)
            return

        self.run_btn.setEnabled(True)
        self.config = _read_model_config(model_name)

        if self.config is None:
            # A repository model that has not been fetched yet: its channels and
            # parameters are unknown until it lands on disk.
            self.channel_form.addRow(
                QLabel("Not downloaded yet.\nIt will be fetched on the first run,\nusing the model's own settings.")
            )
            return

        self._build_channel_rows()
        self._build_parameter_rows()

    def _build_channel_rows(self) -> None:
        """One dropdown per model input slot, seeded from the stored mapping."""

        required = list(self.config.get("channels", []))
        stored = self.config.get("selected_channels")
        options = list(self.exp_channels) + [NO_CHANNEL]

        for i, slot in enumerate(required):
            combo = QComboBox()
            combo.addItems(options)
            combo.setToolTip(f"Experiment channel feeding the model's '{slot}' input.")

            # Prefer the mapping the main window already saved, then a plain name
            # match, and leave the slot empty when neither is available.
            default = None
            if isinstance(stored, list) and i < len(stored):
                default = stored[i]
            if default is None or combo.findText(str(default)) < 0:
                default = slot if slot in self.exp_channels else NO_CHANNEL

            idx = combo.findText(str(default))
            combo.setCurrentIndex(idx if idx >= 0 else len(options) - 1)

            self.channel_cbs.append(combo)
            self.channel_form.addRow(QLabel(f"{slot}:"), combo)

        if not required:
            self.channel_form.addRow(QLabel("This model declares no input channels."))

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
                self.config.get("target_cell_size_um", trained)
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
        if not self.channel_cbs:
            return None
        return [combo.currentText() for combo in self.channel_cbs]

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
        for combo in self.channel_cbs:
            combo.setEnabled(not running)
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

    def closeEvent(self, event) -> None:
        """Stop a run in flight before the panel goes away."""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            # Inference cannot be interrupted, so give it a moment to reach the
            # next phase boundary rather than tearing the thread down under it.
            worker.wait(5000)
        super().closeEvent(event)

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
        if selected is not None and all(ch == NO_CHANNEL for ch in selected):
            self._failed(
                "Every input channel is set to None. Assign at least one experiment "
                "channel to a model input."
            )
            return

        t = int(self.viewer.dims.current_step[0])

        target_cell_size = self.cell_size_le.value() if self.cell_size_le else None
        diameter = self.diameter_le.value() if self.diameter_le else None
        cellprob = self.cellprob_le.value() if self.cellprob_le else None
        flow = self.flow_le.value() if self.flow_le else None

        # Settings are part of the identity of a prepared model: changing the
        # mapping or a threshold has to rebuild it, not reuse the last one.
        cache_key = (
            model_name,
            tuple(selected) if selected is not None else None,
            target_cell_size,
            diameter,
            cellprob,
            flow,
        )

        # The GPU is left to napari's renderer: a single frame is quick on CPU,
        # and a TensorFlow context would compete for the VRAM the viewer is
        # already using.
        prepare_kwargs = dict(
            model_name=model_name,
            channels=self.exp_channels or None,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
            selected_channels=selected,
            target_cell_size=target_cell_size,
            diameter=diameter,
            cellprob_threshold=cellprob,
            flow_threshold=flow,
        )

        self._pending = {"frame": t, "cache_key": cache_key, "model_name": model_name}

        worker = _SegmentationWorker(
            frame=np.asarray(self.stack[t]),
            prepared=self._prepared.get(cache_key),
            prepare_kwargs=prepare_kwargs,
            parent=self,
        )
        worker.stage.connect(self._on_stage)
        worker.succeeded.connect(self._on_succeeded)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker

        self._set_running(True)
        self.status_lbl.setText("Starting…")
        self.viewer.status = f"Segmenting frame {t} with '{model_name}'…"
        worker.start()

    def _on_stage(self, message: str) -> None:
        """Show the phase the worker has entered."""
        self.status_lbl.setText(message)
        self.viewer.status = message

    def _on_failed(self, message: str) -> None:
        """Report a worker failure."""
        self._failed(message)

    def _on_worker_finished(self) -> None:
        """Return the panel to its idle state once the thread has ended."""
        cancelled = self._worker is not None and self._worker.cancelled
        self._worker = None
        self._set_running(False)
        if cancelled:
            self.viewer.status = "Segmentation cancelled."
            logger.info("Single-frame segmentation cancelled.")

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

        pending = self._pending or {}
        t = pending.get("frame", 0)
        model_name = pending.get("model_name", "")
        cache_key = pending.get("cache_key")
        if cache_key is not None:
            self._prepared[cache_key] = prepared

        # The viewer may have been closed while the worker was running.
        try:
            layer = self.viewer.layers["segmentation"]
        except Exception as e:
            logger.debug(f"Segmentation layer is gone, dropping the result: {e}")
            return

        try:
            current = layer.data[t]
            if self.replace_cb.isChecked():
                current[...] = new_labels
            else:
                # Offset the incoming labels past the ones already drawn so the
                # two sets cannot collide, then keep whatever was already there.
                offset = int(current.max())
                incoming = np.where(new_labels > 0, new_labels + offset, 0)
                current[...] = np.where(current > 0, current, incoming)
        except Exception as e:
            self._failed(f"Could not write the labels into the viewer: {e}")
            return
        layer.refresh()

        n_objects = int(np.max(layer.data[t]))
        message = f"Frame {t}: {n_objects} objects after segmenting with '{model_name}'."
        self.viewer.status = message
        logger.info(message)
