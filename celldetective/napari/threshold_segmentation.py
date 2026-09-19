"""
Threshold-segment the frame on screen, from inside the napari correction viewer.

The counterpart of :mod:`celldetective.napari.frame_segmentation` for the classical
pipeline: one or more configurations written by the threshold configuration
wizard are applied to the frame the time slider is on, either to the whole frame
or only within regions of interest drawn as napari shapes. The configurations are
the very files the batch pipeline reads, applied the same way, so what is tried
here is what a full run would produce.

The configurations last used for a population are remembered in the experiment
(see :mod:`celldetective.utils.threshold_configs`), so they are there again the
next time the viewer is opened, whether they were picked here, in the main
window, or just written by the wizard.
"""

import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from celldetective import get_logger
from celldetective.napari.frame_segmentation import _fit_to_layer_dtype
from celldetective.utils.experiment import (
    extract_experiment_channels,
    extract_experiment_from_position,
)
from celldetective.utils.threshold_configs import (
    load_threshold_config,
    recall_threshold_configs,
    remember_threshold_configs,
)

logger = get_logger(__name__)

#: Offered in the region dropdown to threshold the frame as a whole.
WHOLE_FRAME = "whole frame"

#: Name of the shapes layer the panel creates when asked for one.
ROI_LAYER_NAME = "ROIs"

#: Configuration keys handed straight to ``segment_frame_from_thresholds``. The
#: channel and the equalization reference are resolved against the stack first.
_PASSED_KEYS = (
    "thresholds",
    "filters",
    "marker_min_distance",
    "marker_footprint_size",
    "feature_queries",
    "do_watershed",
    "edge_exclusion",
    "fill_holes",
)

# Workers that have been started and not yet finished, held here rather than on
# the panel: a QThread destroyed while running aborts the process, and closing
# the viewer must not be able to take a running one down with it.
_LIVE_WORKERS: Set["_ThresholdWorker"] = set()


def _channel_index(target, channel_names: Sequence[str]) -> int:
    """
    Resolve a configuration's target channel against the stack's channels.

    Parameters
    ----------
    target : str or int
        The channel name the wizard wrote, or an index from an older file.
    channel_names : list of str
        The channel names of the stack, in order.

    Returns
    -------
    int
        Index of the channel along the stack's last axis.

    Raises
    ------
    ValueError
        If the channel is not one of the stack's.
    """

    if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
        return int(target)
    names = list(channel_names)
    if target in names:
        return names.index(target)
    available = ", ".join(names) if names else "none known"
    raise ValueError(
        f"The configuration thresholds channel '{target}', which is not in this "
        f"image (channels: {available})."
    )


def threshold_frame(
    stack,
    frame_index: int,
    configs: Sequence[Dict[str, Any]],
    channel_names: Sequence[str],
) -> np.ndarray:
    """
    Segment one frame of a stack with threshold configurations.

    Each configuration is applied as the batch pipeline applies it, and when
    there are several their masks are merged the same way (``OR``).

    Parameters
    ----------
    stack : ndarray or dask.array.Array
        The TYXC stack the frame is read from.
    frame_index : int
        Index of the frame along the first axis.
    configs : list of dict
        Threshold configurations, as written by the wizard.
    channel_names : list of str
        The channel names of `stack`, in order.

    Returns
    -------
    ndarray
        The instance labels of the frame.
    """

    from celldetective.segmentation import (
        merge_instance_segmentation,
        segment_frame_from_thresholds,
    )

    frame = np.asarray(stack[frame_index])
    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]

    masks = []
    for config in configs:
        channel = _channel_index(config["target_channel"], channel_names)
        if not 0 <= channel < frame.shape[-1]:
            raise ValueError(
                f"The configuration thresholds channel {channel}, but the image "
                f"only has {frame.shape[-1]}."
            )

        kwargs = {k: config[k] for k in _PASSED_KEYS if k in config}
        kwargs["target_channel"] = channel
        kwargs["channel_names"] = list(channel_names) or None

        equalize = config.get("equalize_reference")
        if isinstance(equalize, (list, tuple)) and len(equalize) == 2 and equalize[0]:
            reference_index = int(equalize[1])
            if 0 <= reference_index < len(stack):
                reference = np.asarray(stack[reference_index])
                if reference.ndim == 2:
                    reference = reference[:, :, np.newaxis]
                kwargs["equalize_reference"] = reference[:, :, channel]
            else:
                logger.warning(
                    f"The equalization reference (frame {reference_index}) is not in "
                    f"this stack; thresholding without equalization."
                )

        masks.append(segment_frame_from_thresholds(frame, **kwargs))

    if len(masks) > 1:
        return merge_instance_segmentation(masks, mode="OR")
    return masks[0]


def _polygon_mask(vertices: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon given as (y, x) vertices."""
    from skimage.draw import polygon

    mask = np.zeros(shape, dtype=bool)
    rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape=shape)
    mask[rr, cc] = True
    return mask


def _ellipse_vertices(corners: np.ndarray, n: int = 64) -> np.ndarray:
    """
    Outline of the ellipse napari stores as the corners of its bounding box.

    Built from the box's two half-axes, so a rotated ellipse comes out rotated.
    """
    center = corners.mean(axis=0)
    u = (corners[1] - corners[0]) / 2
    v = (corners[3] - corners[0]) / 2
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)


def _shape_frame(vertices: np.ndarray) -> Optional[int]:
    """The frame a shape is drawn on, None for a 2D shape or one across frames."""
    if vertices.shape[1] <= 2:
        return None
    frames = np.unique(np.round(vertices[:, 0]))
    return int(frames[0]) if len(frames) == 1 else None


def shapes_region(
    layer, frame_index: int, shape: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], int, Optional[int]]:
    """
    The area covered by a shapes layer on one frame.

    Rectangles, polygons and ellipses count; lines and paths enclose nothing and
    are skipped. A 2D layer applies to every frame. On a layer with a time axis
    the shapes drawn on this frame are taken, as for the annotation export --
    or, when there are none, those of the closest earlier frame that has some,
    so regions drawn once carry on through the movie without being redrawn.

    Parameters
    ----------
    layer : napari.layers.Shapes
        The layer holding the regions of interest.
    frame_index : int
        The frame being segmented.
    shape : tuple of int
        The (Y, X) shape of the frame.

    Returns
    -------
    region : ndarray of bool or None
        The union of the shapes, or None when no shape covers this frame.
    n_shapes : int
        How many shapes make up the region.
    source_frame : int or None
        The frame the shapes were drawn on, None for 2D shapes.
    """

    shape_types = list(getattr(layer, "shape_type", []))
    candidates = []
    for i, vertices in enumerate(layer.data):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] < 2 or len(vertices) < 3:
            continue
        kind = shape_types[i] if i < len(shape_types) else "polygon"
        if kind in ("line", "path"):
            continue
        candidates.append((_shape_frame(vertices), kind, vertices[:, -2:]))

    # Shapes of the frame itself, else those of the closest earlier frame.
    drawn_on = {f for f, _, _ in candidates if f is not None and f <= frame_index}
    source_frame = max(drawn_on) if drawn_on else None

    region = np.zeros(shape, dtype=bool)
    n_shapes = 0
    for frame, kind, yx in candidates:
        if frame is not None and frame != source_frame:
            continue
        if kind == "ellipse" and len(yx) == 4:
            yx = _ellipse_vertices(yx)
        region |= _polygon_mask(yx, shape)
        n_shapes += 1

    if n_shapes == 0:
        return None, 0, None
    return region, n_shapes, source_frame


def _labels_in_region(labels: np.ndarray, region: np.ndarray) -> np.ndarray:
    """
    The labels whose centroid falls within a region.

    An object is taken or left whole rather than cut along the edge of the
    region, so a cell straddling the outline of a shape keeps its full mask.
    """
    from scipy import ndimage as ndi

    ids = np.unique(labels)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return ids
    centroids = np.asarray(
        ndi.center_of_mass(np.ones_like(labels, dtype=np.uint8), labels, ids)
    )
    rows = np.clip(np.round(centroids[:, 0]).astype(int), 0, labels.shape[0] - 1)
    cols = np.clip(np.round(centroids[:, 1]).astype(int), 0, labels.shape[1] - 1)
    return ids[region[rows, cols]]


def combine_labels(
    current: np.ndarray,
    new_labels: np.ndarray,
    region: Optional[np.ndarray] = None,
    replace: bool = True,
) -> np.ndarray:
    """
    Write new labels onto a frame, optionally only within a region.

    Parameters
    ----------
    current : ndarray
        The labels currently on the frame.
    new_labels : ndarray
        The labels the thresholding produced, over the whole frame.
    region : ndarray of bool, optional
        Restricts the write to the objects whose centroid falls within it. None
        writes the whole frame.
    replace : bool, optional
        Whether the existing objects are cleared first -- all of them for the
        whole frame, those whose centroid falls within `region` otherwise. The
        others are kept, and new objects only fill the background around them,
        so manual corrections survive.

    Returns
    -------
    ndarray
        The frame's new labels, as int64.
    """

    from skimage.segmentation import relabel_sequential

    current = current.astype(np.int64)
    new_labels = np.asarray(new_labels).astype(np.int64)

    if region is None:
        if replace:
            return relabel_sequential(new_labels)[0]
        kept = current
    else:
        new_labels = np.where(
            np.isin(new_labels, _labels_in_region(new_labels, region)), new_labels, 0
        )
        kept = current
        if replace:
            kept = np.where(
                np.isin(current, _labels_in_region(current, region)), 0, current
            )

    new_labels = relabel_sequential(new_labels)[0]
    offset = int(kept.max()) if kept.size else 0
    incoming = np.where(new_labels > 0, new_labels + offset, 0)
    return np.where(kept > 0, kept, incoming)


def _experiment_movie_prefix(exp_dir: str) -> Optional[str]:
    """The movie prefix set in the experiment configuration, if it can be read."""
    try:
        from celldetective.utils.experiment import get_config
        from celldetective.utils.parsing import config_section_to_dict

        settings = config_section_to_dict(get_config(exp_dir), "MovieSettings")
        return settings.get("movie_prefix")
    except Exception as e:
        logger.debug(f"Could not read the movie prefix of {exp_dir}: {e}")
        return None


class _ThresholdWorker(QThread):
    """
    Threshold one or more frames off the GUI thread, reading them there too.

    Frames are handed back one at a time, so the viewer fills in as the run
    progresses and a cancel -- checked between frames -- keeps what is done.
    """

    #: Emitted with ``(frame_index, labels)`` for each frame thresholded.
    frame_done = pyqtSignal(int, object)
    #: Emitted with a message the user should see; the run stops there.
    failed = pyqtSignal(str)

    def __init__(self, stack, frames, configs, channel_names, parent=None):
        super().__init__(parent)
        self._stack = stack
        self._frames = list(frames)
        self._configs = configs
        self._channel_names = channel_names
        self._cancelled = False

    def cancel(self) -> None:
        """Stop before the next frame."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        for t in self._frames:
            if self._cancelled:
                return
            try:
                labels = threshold_frame(
                    self._stack, t, self._configs, self._channel_names
                )
            except ValueError as e:
                self.failed.emit(str(e))
                return
            except Exception as e:
                logger.exception(f"Threshold segmentation of frame {t} failed.")
                self.failed.emit(f"Threshold segmentation of frame {t} failed: {e}")
                return
            self.frame_done.emit(t, labels)


def _history_atom(t: int, before: np.ndarray, after: np.ndarray):
    """
    The change to one frame, as a napari labels history atom.

    Returns
    -------
    tuple or None
        ``(indices, before, after)`` over the changed pixels, None if none did.
    """
    changed = np.nonzero(before != after)
    if len(changed[0]) == 0:
        return None
    indices = (np.full(changed[0].shape, t, dtype=np.intp),) + changed
    return indices, before[changed], after[changed]


def _record_run_undo(layer, atoms: Sequence[tuple]) -> None:
    """
    Push every frame a run changed as a single undo step.

    One Ctrl+Z then takes the whole run back, rather than a frame at a time.
    Best-effort, like :func:`~celldetective.napari.frame_segmentation.record_undo`:
    the history is private napari API.
    """
    if not atoms:
        return
    try:
        n_axes = len(atoms[0][0])
        indices = tuple(
            np.concatenate([atom[0][axis] for atom in atoms]) for axis in range(n_axes)
        )
        layer._save_history(
            (
                indices,
                np.concatenate([atom[1] for atom in atoms]),
                np.concatenate([atom[2] for atom in atoms]),
            )
        )
    except Exception as e:
        logger.debug(f"Could not record an undo step for the thresholding: {e}")


class ThresholdSegmentationPanel(QWidget):
    """
    Dock panel applying threshold configurations to the frame currently displayed.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer holding the ``segmentation`` labels layer to write into.
    stack : ndarray
        The image stack being displayed (TYXC).
    position : str or None
        The position directory. Used to find the experiment -- where the last
        configurations are remembered -- and the movie the wizard opens on.
    population : str
        The population being segmented.
    channels : list of str, optional
        The channel names of `stack`, overriding the experiment's. Required when
        there is no position.
    exp_dir : str, optional
        The experiment directory, when there is no position to find it from.
    """

    def __init__(
        self,
        viewer,
        stack,
        position: Optional[str] = None,
        population: str = "targets",
        parent=None,
        channels: Optional[List[str]] = None,
        exp_dir: Optional[str] = None,
    ):
        super().__init__(parent)

        self.viewer = viewer
        self.stack = stack
        self.population = population
        self.position = position
        if position and not position.endswith(("/", os.sep)):
            self.position = position + os.sep

        if exp_dir is None and position:
            exp_dir = extract_experiment_from_position(position)
        self.exp_dir = exp_dir

        if channels is not None:
            self.exp_channels = list(channels)
        elif self.exp_dir:
            try:
                self.exp_channels = list(extract_experiment_channels(self.exp_dir)[0])
            except Exception as e:
                logger.warning(f"Could not read the experiment channels: {e}")
                self.exp_channels = []
        else:
            self.exp_channels = []

        self.config_paths: List[str] = []
        self.configs: List[Dict[str, Any]] = []
        self._worker: Optional[_ThresholdWorker] = None
        self._pending: Dict[str, Any] = {}
        self._wizard = None

        self._build()
        self._set_configs(recall_threshold_configs(self.exp_dir, population), quiet=True)
        self._connect_layer_events()
        self._refresh_regions()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        config_row = QHBoxLayout()
        config_row.addWidget(QLabel("config:"), 30)
        self.config_lbl = QLabel("none")
        self.config_lbl.setWordWrap(True)
        config_row.addWidget(self.config_lbl, 70)
        outer.addLayout(config_row)

        buttons = QHBoxLayout()
        self.load_btn = QPushButton("Load…")
        self.load_btn.setToolTip(
            "Pick one or more threshold configurations written by the wizard.\n"
            "Several are merged, as in the batch pipeline. The choice is\n"
            "remembered for this experiment."
        )
        self.load_btn.clicked.connect(self._on_load_clicked)
        buttons.addWidget(self.load_btn)

        self.wizard_btn = QPushButton("Wizard…")
        self.wizard_btn.clicked.connect(self.open_wizard)
        buttons.addWidget(self.wizard_btn)
        outer.addLayout(buttons)

        # Looked up once: it takes reading the experiment configuration.
        self._movie = self._wizard_movie()
        if self._movie is None:
            self.wizard_btn.setEnabled(False)
            self.wizard_btn.setToolTip(
                "The wizard needs the movie of a position, which this image has not."
            )
        else:
            self.wizard_btn.setToolTip(
                "Open the threshold configuration wizard on this frame.\n"
                "The configuration it saves is loaded here."
            )

        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("region:"), 30)
        self.region_cb = QComboBox()
        self.region_cb.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.region_cb.setMinimumContentsLength(10)
        self.region_cb.setToolTip(
            "Threshold the whole frame, or only keep the objects whose centre\n"
            "lies within the shapes of a shapes layer drawn on this frame\n"
            "(or, if none, on the closest earlier frame that has some)."
        )
        region_row.addWidget(self.region_cb, 60)
        self.add_roi_btn = QPushButton("+")
        self.add_roi_btn.setFixedWidth(28)
        self.add_roi_btn.setToolTip(
            f"Add a '{ROI_LAYER_NAME}' shapes layer to draw regions of interest in."
        )
        self.add_roi_btn.clicked.connect(self.add_roi_layer)
        region_row.addWidget(self.add_roi_btn, 10)
        outer.addLayout(region_row)

        self.replace_cb = QCheckBox("Replace the labels in the region")
        self.replace_cb.setChecked(True)
        self.replace_cb.setToolTip(
            "Unticked, existing labels are kept and the new ones only fill the\n"
            "background, so manual corrections survive."
        )
        outer.addWidget(self.replace_cb)

        self.following_cb = QCheckBox("Also the following frames")
        self.following_cb.setToolTip(
            "Threshold every frame from this one to the end of the movie.\n"
            "The region drawn on this frame is used on all of them. Frames are\n"
            "written as they are done; one Ctrl+Z takes the whole run back."
        )
        self.following_cb.toggled.connect(lambda _: self._set_running(False))
        outer.addWidget(self.following_cb)

        self.run_btn = QPushButton("Threshold this frame")
        self.run_btn.clicked.connect(self._on_run_clicked)
        outer.addWidget(self.run_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.hide()
        outer.addWidget(self.status_lbl)

    def _connect_layer_events(self) -> None:
        """Keep the region dropdown in step with the shapes layers of the viewer."""
        try:
            events = self.viewer.layers.events
            events.inserted.connect(self._refresh_regions)
            events.removed.connect(self._refresh_regions)
        except Exception as e:
            logger.debug(f"Could not follow the viewer's layers: {e}")

    def _shapes_layers(self) -> List[Any]:
        try:
            layers = list(self.viewer.layers)
        except Exception:
            return []
        return [layer for layer in layers if type(layer).__name__ == "Shapes"]

    def _refresh_regions(self, *_: Any) -> None:
        """List the shapes layers as regions, keeping the current choice if it is still there."""
        try:
            current = self.region_cb.currentText()
        except RuntimeError:
            # The panel is gone; napari still holds the connection.
            return
        names = [WHOLE_FRAME] + [layer.name for layer in self._shapes_layers()]
        self.region_cb.blockSignals(True)
        self.region_cb.clear()
        self.region_cb.addItems(names)
        if current in names:
            self.region_cb.setCurrentText(current)
        self.region_cb.blockSignals(False)

    def add_roi_layer(self) -> None:
        """Add a shapes layer for regions of interest and select it as the region."""
        existing = {layer.name: layer for layer in self._shapes_layers()}
        layer = existing.get(ROI_LAYER_NAME)
        if layer is None:
            layer = self.viewer.add_shapes(
                name=ROI_LAYER_NAME,
                ndim=self.viewer.layers["segmentation"].ndim,
                edge_color="yellow",
                face_color="transparent",
                edge_width=2,
            )
        self._refresh_regions()
        self.region_cb.setCurrentText(layer.name)
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "add_rectangle"
        except Exception as e:
            logger.debug(f"Could not switch to drawing rectangles: {e}")

    # ------------------------------------------------------------------
    # Configurations
    # ------------------------------------------------------------------

    def _set_configs(self, paths: Sequence[str], quiet: bool = False) -> bool:
        """
        Load configurations, keeping the previous ones if any of them is unusable.

        Parameters
        ----------
        paths : list of str
            The configuration files.
        quiet : bool, optional
            Log problems rather than showing them, for the recalled ones.

        Returns
        -------
        bool
            Whether the configurations were loaded.
        """

        if not paths:
            self._show_configs()
            return False
        try:
            configs = [load_threshold_config(p) for p in paths]
        except ValueError as e:
            if quiet:
                logger.warning(str(e))
            else:
                self._failed(str(e))
            self._show_configs()
            return False
        self.config_paths = list(paths)
        self.configs = configs
        self._show_configs()
        return True

    def _show_configs(self) -> None:
        if not self.config_paths:
            self.config_lbl.setText("none")
            self.config_lbl.setToolTip(
                "Load a configuration, or write one with the wizard."
            )
            self.run_btn.setEnabled(False)
            return
        names = [os.path.basename(p) for p in self.config_paths]
        self.config_lbl.setText(names[0] if len(names) == 1 else f"{len(names)} merged")
        self.config_lbl.setToolTip("\n".join(self.config_paths))
        self.run_btn.setEnabled(self._worker is None)

    def _on_load_clicked(self) -> None:
        start = ""
        if self.config_paths:
            start = os.path.dirname(self.config_paths[0])
        elif self.exp_dir:
            configs_dir = os.path.join(self.exp_dir, "configs")
            start = configs_dir if os.path.isdir(configs_dir) else self.exp_dir
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load threshold configuration(s)…", start, "JSON (*.json)"
        )
        if paths:
            self.load_configs(paths)

    def load_configs(self, paths: Sequence[str]) -> None:
        """
        Use these configurations from now on, and remember them for the experiment.

        Parameters
        ----------
        paths : list of str
            The configuration files, merged when there are several.
        """

        if self._set_configs(paths):
            remember_threshold_configs(self.exp_dir, self.population, list(paths))
            self.viewer.status = f"Threshold configuration: {self.config_lbl.text()}."

    # ------------------------------------------------------------------
    # Wizard
    # ------------------------------------------------------------------

    def _movie_prefix(self) -> Optional[str]:
        return _experiment_movie_prefix(self.exp_dir) if self.exp_dir else None

    def _wizard_movie(self) -> Optional[str]:
        """The movie the wizard would open on, or None when there is none."""
        if not self.position or not self.exp_dir:
            return None
        prefix = self._movie_prefix()
        if prefix is None:
            return None
        movies = glob(os.path.join(self.position, "movie", f"{prefix}*.tif"))
        return movies[0] if movies else None

    def open_wizard(self) -> None:
        """Open the threshold configuration wizard on the frame on screen."""

        if self._movie is None:
            self._failed("No movie was found for this position to open the wizard on.")
            return
        try:
            from celldetective.gui.thresholds_gui import ThresholdConfigWizard

            self._wizard = ThresholdConfigWizard(
                None,
                mode=self.population,
                pos=self.position,
                exp_dir=self.exp_dir,
                movie_prefix=self._movie_prefix(),
                initial_frame=int(self.viewer.dims.current_step[0]),
                on_saved=self._on_wizard_saved,
            )
            self._wizard.show()
        except Exception as e:
            logger.exception("Could not open the threshold configuration wizard.")
            self._failed(f"Could not open the threshold configuration wizard: {e}")

    def _on_wizard_saved(self, path: str) -> None:
        """Pick up the configuration the wizard just wrote."""
        self._wizard = None
        if self._set_configs([path]):
            remember_threshold_configs(self.exp_dir, self.population, [path])
            self.viewer.status = (
                f"Loaded {os.path.basename(path)}. Threshold this frame to apply it."
            )

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def _failed(self, message: str) -> None:
        """Report a failure without tearing down the viewer."""
        logger.error(message)
        try:
            self.viewer.status = message
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setText(message)
            box.setWindowTitle("Threshold segmentation")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
        except Exception as e:
            logger.debug(f"Could not show the threshold segmentation error: {e}")

    def _selected_region(self, t: int, shape: Tuple[int, int]):
        """
        The region chosen in the dropdown on frame `t`.

        Returns
        -------
        region : ndarray of bool or None
            None for the whole frame.
        description : str
            How the region reads in the status bar.

        Raises
        ------
        ValueError
            If a shapes layer is chosen but has nothing drawn on this frame.
        """

        name = self.region_cb.currentText()
        if not name or name == WHOLE_FRAME:
            return None, "the whole frame"
        try:
            layer = self.viewer.layers[name]
        except KeyError:
            raise ValueError(f"The shapes layer '{name}' is gone.")
        region, n_shapes, source = shapes_region(layer, t, shape)
        if region is None:
            raise ValueError(
                f"No rectangle, polygon or ellipse of '{name}' is drawn on frame {t} "
                f"or any frame before it."
            )
        description = f"{n_shapes} ROI{'s' if n_shapes > 1 else ''}"
        if source is not None and source != t:
            description += f" drawn on frame {source}"
        return region, description

    def _idle_run_text(self) -> str:
        return (
            "Threshold from this frame on"
            if self.following_cb.isChecked()
            else "Threshold this frame"
        )

    def _set_running(self, running: bool) -> None:
        for widget in (
            self.load_btn,
            self.region_cb,
            self.add_roi_btn,
            self.replace_cb,
            self.following_cb,
        ):
            widget.setEnabled(not running)
        self.wizard_btn.setEnabled(not running and self._movie is not None)
        if running:
            self.run_btn.setText("Cancel")
            self.run_btn.setEnabled(True)
        else:
            self.run_btn.setText(self._idle_run_text())
            self.run_btn.setEnabled(bool(self.configs))
        self.status_lbl.setVisible(running)
        if not running:
            self.status_lbl.setText("")

    def _on_run_clicked(self) -> None:
        """Start a run, or cancel the one in flight."""
        if self._worker is not None:
            self._worker.cancel()
            self.run_btn.setEnabled(False)
            self.status_lbl.setText("Cancelling…")
            return
        self.threshold_current_frame()

    def threshold_current_frame(self) -> None:
        """
        Apply the loaded configurations to the frame the time slider is on.

        With "Also the following frames" ticked, every frame from there to the
        end of the movie is thresholded too, within the region drawn on this
        frame. The work runs on a worker thread; each frame is written into the
        segmentation layer as it is done, the whole run is one undoable step,
        and nothing reaches disk until the labels are saved.
        """

        if self._worker is not None:
            return
        if not self.configs:
            self._failed("Load a threshold configuration first.")
            return

        t = int(self.viewer.dims.current_step[0])
        try:
            layer = self.viewer.layers["segmentation"]
            shape = tuple(layer.data.shape[-2:])
            region, description = self._selected_region(t, shape)
        except ValueError as e:
            self._failed(str(e))
            return
        except Exception as e:
            self._failed(f"Could not read the segmentation layer: {e}")
            return

        n_frames = int(layer.data.shape[0])
        frames = list(range(t, n_frames)) if self.following_cb.isChecked() else [t]

        self._pending = {
            "frames": frames,
            "region": region,
            "description": description,
            "replace": self.replace_cb.isChecked(),
            "atoms": [],
            "done": 0,
            "failed": False,
        }

        # Unparented, see `_LIVE_WORKERS`.
        worker = _ThresholdWorker(
            self.stack, frames, list(self.configs), self.exp_channels
        )
        _LIVE_WORKERS.add(worker)
        worker.finished.connect(lambda w=worker: _LIVE_WORKERS.discard(w))
        worker.finished.connect(worker.deleteLater)
        worker.frame_done.connect(self._on_frame_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker

        self._set_running(True)
        self.status_lbl.setText(f"Thresholding 0/{len(frames)}…")
        if len(frames) > 1:
            self.viewer.status = (
                f"Thresholding frames {frames[0]}–{frames[-1]} in {description}…"
            )
        else:
            self.viewer.status = f"Thresholding frame {t} in {description}…"
        worker.start()

    def _on_finished(self) -> None:
        worker = self._worker
        cancelled = worker is not None and worker.cancelled
        self._worker = None
        self._set_running(False)

        pending = self._pending
        try:
            layer = self.viewer.layers["segmentation"]
        except Exception:
            return
        _record_run_undo(layer, pending.get("atoms", []))
        pending["atoms"] = []

        frames = pending.get("frames", [])
        done = pending.get("done", 0)
        if len(frames) <= 1 or pending.get("failed"):
            return
        message = (
            f"Thresholded frames {frames[0]}–{frames[0] + done - 1} "
            f"in {pending.get('description', 'the frame')}"
        )
        if cancelled:
            message += f"; cancelled with {len(frames) - done} left."
        else:
            message += "."
        self.viewer.status = message
        logger.info(message)

    def _on_failed(self, message: str) -> None:
        self._pending["failed"] = True
        self._failed(message)

    def _on_frame_done(self, t: int, new_labels) -> None:
        """Write one frame's thresholded labels into the segmentation layer."""

        try:
            layer = self.viewer.layers["segmentation"]
        except Exception as e:
            logger.debug(f"Segmentation layer is gone, dropping the result: {e}")
            return

        try:
            current = layer.data[t]
            before = current.copy()
            combined = combine_labels(
                current,
                new_labels,
                region=self._pending.get("region"),
                replace=self._pending.get("replace", True),
            )
            current[...] = _fit_to_layer_dtype(
                combined,
                current.dtype,
                "Tick 'Replace the labels in the region' to segment it afresh.",
            )
        except Exception as e:
            # Stop there: carrying on would only fail the same way on every frame.
            if self._worker is not None:
                self._worker.cancel()
            self._pending["failed"] = True
            self._failed(
                str(e)
                if isinstance(e, ValueError)
                else f"Could not write the labels into the viewer: {e}"
            )
            return

        atom = _history_atom(t, before, layer.data[t])
        if atom is not None:
            self._pending.setdefault("atoms", []).append(atom)
        self._pending["done"] = self._pending.get("done", 0) + 1
        layer.refresh()

        frames = self._pending.get("frames", [t])
        if len(frames) > 1:
            self.status_lbl.setText(
                f"Thresholding {self._pending['done']}/{len(frames)}…"
            )
            return

        n_objects = int(np.count_nonzero(np.unique(layer.data[t])))
        message = (
            f"Frame {t}: {n_objects} objects after thresholding "
            f"{self._pending.get('description', 'the frame')}."
        )
        self.viewer.status = message
        logger.info(message)

    def closeEvent(self, event) -> None:
        """Let a run in flight finish unseen, and close the wizard opened from here."""
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.cancel()
            for signal, slot in (
                (worker.frame_done, self._on_frame_done),
                (worker.failed, self._on_failed),
                (worker.finished, self._on_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        try:
            self.viewer.layers.events.inserted.disconnect(self._refresh_regions)
            self.viewer.layers.events.removed.disconnect(self._refresh_regions)
        except Exception:
            pass
        super().closeEvent(event)
