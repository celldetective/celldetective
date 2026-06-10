import gc
import json
import os
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import napari
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout
from celldetective.utils.io import save_tiff_imagej_compatible
from magicgui import magicgui
from skimage.measure import regionprops_table
from tifffile import imread
from tqdm import tqdm

from celldetective.utils.data_cleaning import tracks_to_btrack
from celldetective.utils.mask_cleaning import auto_correct_masks, relabel_segmentation
from celldetective.utils.image_loaders import (
    locate_labels,
    locate_stack,
    locate_stack_and_labels,
    locate_stack_lazy,
    fix_missing_labels,
)
from celldetective.utils.data_loaders import get_position_table, load_tracking_data
from celldetective.utils.experiment import (
    extract_experiment_from_position,
    _get_contrast_limits,
    get_experiment_wells,
    get_experiment_labels,
    get_experiment_metadata,
    extract_experiment_channels,
)
from celldetective.utils.parsing import config_section_to_dict
from celldetective import get_logger
from celldetective.gui.base.styles import Styles

logger = get_logger()


def control_tracks(
    position: str,
    prefix: str = "Aligned",
    population: str = "target",
    relabel: bool = True,
    flush_memory: bool = True,
    threads: int = 1,
    progress_callback: Optional[Callable[[int], bool]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    prepare_only: bool = False,
) -> Optional[Union[napari.Viewer, Dict[str, Any]]]:
    """
    Controls the tracking of cells or objects within a given position by locating the relevant image stack and label data,
    and then visualizing and managing the tracks in the Napari viewer.

    Parameters
    ----------
    position : str
            The path to the directory containing the position's data. The function will ensure the path uses forward slashes.

    prefix : str, optional, default="Aligned"
            The prefix of the file names for the image stack and labels. This parameter helps locate the relevant data files.

    population : str, optional, default="target"
            The population to be tracked, typically either "target" or "effectors". This is used to identify the group of interest for tracking.

    relabel : bool, optional, default=True
            If True, will relabel the tracks, potentially assigning new track IDs to the detected objects.

    flush_memory : bool, optional, default=True
            If True, will flush memory after processing to free up resources.

    threads : int, optional, default=1
            The number of threads to use for processing. This can speed up the task in multi-threaded environments.

    progress_callback : function, optional
            A callback function to report progress.

    prepare_only : bool, optional, default=False
            If True, only prepare the data structure but do not launch the viewer.

    Returns
    -------
    None
            The function performs visualization and management of tracks in the Napari viewer. It does not return any value.

    Notes
    -----
    - This function assumes that the necessary data for tracking (stack and labels) are located in the specified position directory.
    - The `locate_stack_and_labels` function is used to retrieve the image stack and labels from the specified directory.
    - The tracks are visualized using the `view_tracks_in_napari` function, which handles the display in the Napari viewer.
    - The function can be used for tracking biological entities (e.g., cells) and their movement across time frames in an image stack.

    Example
    -------
    >>> control_tracks("/path/to/data/position_1", prefix="Aligned", population="target", relabel=True, flush_memory=True, threads=4)

    """

    if not position.endswith(os.sep):
        position += os.sep

    position = position.replace("\\", "/")
    if progress_callback:
        progress_callback(0)

    # --- Load masks (parallel, with progress) ---
    if status_callback:
        status_callback("Loading masks…")

    def _labels_progress(p: int) -> None:
        # Map mask-loading progress onto the 0–20 band of the overall bar.
        if progress_callback:
            progress_callback(int(p * 0.20))

    n_label_threads = max(int(threads), 4)
    labels = locate_labels(
        position,
        population=population,
        threads=n_label_threads,
        progress_callback=_labels_progress,
    )

    # --- Load image stack (lazily when possible, else eagerly) ---
    if status_callback:
        status_callback("Loading image stack…")
    if progress_callback:
        progress_callback(20)

    stack = locate_stack_lazy(position, prefix=prefix)
    if stack is None:
        stack = locate_stack(position, prefix=prefix)

    # Mirror locate_stack_and_labels: repair/realign label count if needed.
    if labels is None or len(labels) < len(stack):
        fix_missing_labels(position, population=population, prefix=prefix)
        labels = locate_labels(
            position, population=population, threads=n_label_threads
        )
    if len(stack) != len(labels):
        raise ValueError(
            f"The shape of the stack {getattr(stack, 'shape', None)} does not "
            f"match with the shape of the labels {getattr(labels, 'shape', None)}"
        )

    if progress_callback:
        progress_callback(25)

    return view_tracks_in_napari(
        position,
        population,
        labels=labels,
        stack=stack,
        relabel=relabel,
        flush_memory=flush_memory,
        threads=threads,
        progress_callback=progress_callback,
        status_callback=status_callback,
        prepare_only=prepare_only,
    )


def tracks_to_napari(
    df: pd.DataFrame, exclude_nans: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Convert a DataFrame of tracks to Napari-compatible format.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing tracking data.
    exclude_nans : bool, optional
        Whether to exclude NaNs from the data.

    Returns
    -------
    tuple
        A tuple containing vertices, tracks, properties, and graph.
    """

    # tracks_to_btrack mutates its input in place (inplace dropna, adds 'z'/'dummy'
    # columns). Pass a copy so the caller's working DataFrame is never silently
    # modified (otherwise displaying the tracks would permanently drop NaN rows).
    data, properties, graph = tracks_to_btrack(df.copy(), exclude_nans=exclude_nans)
    vertices = data[:, [1, -2, -1]]
    if data.shape[1] == 4:
        tracks = data
    else:
        tracks = data[:, [0, 1, 3, 4]]
    return vertices, tracks, properties, graph


def view_tracks_in_napari(
    position: str,
    population: str,
    stack: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    relabel: bool = True,
    flush_memory: bool = True,
    threads: int = 1,
    progress_callback: Optional[Callable[[int], bool]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    prepare_only: bool = False,
) -> Optional[Union[napari.Viewer, Dict[str, Any]]]:
    """
    View tracks in Napari.

    Parameters
    ----------
    position : str
        The path to the position directory.
    population : str
        The population to visualize.
    stack : numpy.ndarray, optional
        The image stack.
    labels : numpy.ndarray, optional
        The label images.
    relabel : bool, optional
        Whether to relabel the segmentation to match track IDs.
    flush_memory : bool, optional
        Whether to flush memory after visualization.
    threads : int, optional
        Number of threads for processing.
    progress_callback : function, optional
        Callback for progress updates.
    prepare_only : bool, optional
        If True, returns the data dictionary instead of launching the viewer.

    Returns
    -------
    napari.Viewer or dict or None
        The Napari viewer instance, data dictionary, or None.
    """

    logger.debug(f"view_tracks_in_napari called with pos={position}, pop={population}")
    if status_callback:
        status_callback("Reading trajectories…")
    df, df_path = get_position_table(position, population=population, return_path=True)
    logger.debug(f"get_position_table returned df={df is not None}")

    if progress_callback:
        progress_callback(50)

    if df is None:
        logger.warning("Please compute trajectories first... Abort...")
        return None
    shared_data = {
        "df": df,
        "path": df_path,
        "position": position,
        "population": population,
        "selected_frame": None,
    }

    if (labels is not None) and relabel:
        logger.info("Replacing the cell mask labels with the track ID...")
        if status_callback:
            status_callback("Relabeling masks…")

        def wrapped_callback(p: int) -> bool:
            """
            Wrap the progress callback to scale it.

            Parameters
            ----------
            p : int
                Progress value.
            """
            if progress_callback:
                return progress_callback(50 + int(p * 0.5))
            return True

        labels = relabel_segmentation(
            labels,
            df,
            exclude_nans=True,
            threads=threads,
            progress_callback=wrapped_callback,
        )
        if labels is None:
            return None

    if status_callback:
        status_callback("Preparing tracks…")
    vertices, tracks, properties, graph = tracks_to_napari(df, exclude_nans=True)

    contrast_limits = _get_contrast_limits(stack)

    data = {
        "stack": stack,
        "labels": labels,
        "vertices": vertices,
        "tracks": tracks,
        "properties": properties,
        "graph": graph,
        "shared_data": shared_data,
        "contrast_limits": contrast_limits,
        "flush_memory": flush_memory,
    }

    if prepare_only:
        return data

    return launch_napari_viewer(**data)


def launch_napari_viewer(
    stack: np.ndarray,
    labels: np.ndarray,
    vertices: np.ndarray,
    tracks: np.ndarray,
    properties: Dict[str, Any],
    graph: Dict[str, Any],
    shared_data: Dict[str, Any],
    contrast_limits: List[Tuple[float, float]],
    flush_memory: bool = True,
    block: bool = True,
    progress_callback: Optional[Callable[[int], bool]] = None,
) -> napari.Viewer:
    """
    Launch the Napari viewer with the provided data.

    Parameters
    ----------
    stack : numpy.ndarray
        The image stack.
    labels : numpy.ndarray
        The label images.
    vertices : numpy.ndarray
        The vertices of the tracks.
    tracks : numpy.ndarray
        The track data.
    properties : dict
        Properties of the tracks.
    graph : dict
        The graph of the tracks.
    shared_data : dict
        Shared data for the viewer.
    contrast_limits : list
        Contrast limits for the image.
    flush_memory : bool, optional
        Whether to flush memory after closing.
    block : bool, optional
        Whether to block execution while the viewer is open.
    progress_callback : function, optional
        Callback for progress.
    """

    viewer = napari.Viewer()

    # Prevent default double-click to zoom behavior
    for cb in list(viewer.mouse_double_click_callbacks):
        if getattr(cb, "__name__", "") == "double_click_to_zoom":
            viewer.mouse_double_click_callbacks.remove(cb)

    if stack is not None:
        viewer.add_image(
            stack,
            channel_axis=-1,
            colormap=["gray"] * stack.shape[-1],
            contrast_limits=contrast_limits,
        )

    if labels is not None:
        # Avoid a full int64 copy of the whole TYX stack when the labels are
        # already an integer type (relabel_segmentation now returns int32).
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int32)
        labels_layer = viewer.add_labels(
            labels, name="segmentation", opacity=0.4
        )
    viewer.add_points(vertices, size=4, name="points", opacity=0.3)
    viewer.add_tracks(tracks, properties=properties, graph=graph, name="tracks")

    def lock_controls(
        layer: napari.layers.Layer, widgets: Tuple[str, ...] = (), locked: bool = True
    ) -> None:
        """
        Lock or unlock controls for a layer.

        Parameters
        ----------
        layer : str
            The layer name.
        widgets : tuple, optional
            Widgets to lock/unlock.
        locked : bool, optional
            Whether to lock or unlock the widgets.
        """
        qctrl = viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widgets:
            try:
                getattr(qctrl, wdg).setEnabled(not locked)
            except Exception as e:
                logger.debug(f"Could not set {wdg} enabled state: {e}")

    label_widget_list = [
        "paint_button",
        "erase_button",
        "fill_button",
        "polygon_button",
        "transform_button",
    ]
    lock_controls(viewer.layers["segmentation"], label_widget_list)

    point_widget_list = [
        "addition_button",
        "delete_button",
        "select_button",
        "transform_button",
    ]
    lock_controls(viewer.layers["points"], point_widget_list)

    track_widget_list = ["transform_button"]
    lock_controls(viewer.layers["tracks"], track_widget_list)

    # Initialize selected frame
    selected_frame = viewer.dims.current_step[0]
    shared_data["selected_frame"] = selected_frame

    # Cache the running maximum label/track id so we never have to scan the whole
    # label stack on every correction (see on_second_click_of_double_click).
    max_label = 0
    seg_data_init = viewer.layers["segmentation"].data if "segmentation" in viewer.layers else None
    if seg_data_init is not None and seg_data_init.size:
        max_label = int(np.max(seg_data_init))
    try:
        df_max = np.nanmax(shared_data["df"]["TRACK_ID"].to_numpy())
        if np.isfinite(df_max):
            max_label = max(max_label, int(df_max))
    except (ValueError, KeyError, TypeError):
        pass
    shared_data["max_label"] = max_label

    # Bounded undo history: snapshots of (df, modified label tail, counters).
    undo_stack: List[Dict[str, Any]] = []
    MAX_UNDO = 5

    def push_undo_snapshot(frame_start: int) -> None:
        """
        Snapshot the current state before a correction so it can be reverted.

        Parameters
        ----------
        frame_start : int
            First frame index whose labels are about to be modified. Only the
            label tail from this frame onward is copied, to bound memory use.
        """
        seg = viewer.layers["segmentation"].data
        undo_stack.append(
            {
                "df": shared_data["df"].copy(),
                "frame_start": frame_start,
                "labels_tail": seg[frame_start:].copy(),
                "max_label": shared_data["max_label"],
                "selected_frame": shared_data["selected_frame"],
            }
        )
        if len(undo_stack) > MAX_UNDO:
            undo_stack.pop(0)

    def refresh_track_layers() -> None:
        """Rebuild the points/tracks layers from the current DataFrame."""
        vertices, tracks, properties, graph = tracks_to_napari(
            shared_data["df"], exclude_nans=True
        )
        viewer.layers["tracks"].data = tracks
        viewer.layers["tracks"].properties = properties
        viewer.layers["tracks"].graph = graph
        viewer.layers["points"].data = vertices
        viewer.layers["segmentation"].refresh()
        viewer.layers["tracks"].refresh()
        viewer.layers["points"].refresh()

    def undo_last_correction() -> None:
        """Revert the most recent correction, if any."""
        if not undo_stack:
            logger.info("Nothing to undo.")
            return
        snap = undo_stack.pop()
        shared_data["df"] = snap["df"]
        shared_data["max_label"] = snap["max_label"]
        shared_data["selected_frame"] = snap["selected_frame"]
        seg = viewer.layers["segmentation"].data
        seg[snap["frame_start"]:] = snap["labels_tail"]
        refresh_track_layers()
        logger.info("Reverted the last correction.")

    def export_modifications():
        """Export modified tracks."""

        from celldetective.tracking import (
            write_first_detection_class,
            clean_trajectories,
        )
        from celldetective.utils.maths import velocity_per_track

        df = shared_data["df"]
        position = shared_data["position"]
        population = shared_data["population"]
        df = velocity_per_track(df, window_size=3, mode="bi")
        df = write_first_detection_class(df, img_shape=labels[0].shape)

        experiment = extract_experiment_from_position(position)
        instruction_file = "/".join(
            [experiment, "configs", f"tracking_instructions_{population}.json"]
        )
        logger.debug(f"instruction_file={instruction_file}")
        if os.path.exists(instruction_file):
            logger.info("Tracking configuration file found...")
            with open(instruction_file, "r") as f:
                instructions = json.load(f)
                if "post_processing_options" in instructions:
                    post_processing_options = instructions["post_processing_options"]
                    logger.info(
                        f"Applying the following track postprocessing: {post_processing_options}..."
                    )
                    df = clean_trajectories(df.copy(), **post_processing_options)
        unnamed_cols = [c for c in list(df.columns) if c.startswith("Unnamed")]
        df = df.drop(unnamed_cols, axis=1)
        logger.debug(f"Columns after export: {list(df.columns)}")
        df.to_csv(shared_data["path"], index=False)
        logger.info("Track export done.")

    @magicgui(call_button="Export the modified\ntracks...")
    def export_table_widget():
        """Widget to trigger export."""
        return export_modifications()

    @magicgui(call_button="Undo last\ncorrection")
    def undo_widget():
        """Widget to revert the last correction."""
        return undo_last_correction()

    export_table_widget.native.setStyleSheet(Styles().button_style_sheet)
    undo_widget.native.setStyleSheet(Styles().button_style_sheet)

    def label_changed(event: str) -> None:
        """
        Handle label selection change.

        Parameters
        ----------
        event : str
             The event type.
        """

        value = viewer.layers["segmentation"].selected_label
        if value != 0:
            selected_frame = viewer.dims.current_step[0]
            shared_data["selected_frame"] = selected_frame

    viewer.layers["segmentation"].events.selected_label.connect(label_changed)

    track_button_container = QWidget()
    track_button_layout = QVBoxLayout(track_button_container)
    track_button_layout.setSpacing(10)
    track_button_layout.addWidget(export_table_widget.native)
    track_button_layout.addWidget(undo_widget.native)
    viewer.window.add_dock_widget(track_button_container, area="right")

    @labels_layer.mouse_double_click_callbacks.append
    def on_second_click_of_double_click(
        layer: napari.layers.Labels, event: napari.utils.events.Event
    ) -> None:
        """
        Handle double click on the labels layer.

        Parameters
        ----------
        layer : napari.layers.Labels
            The labels layer.
        event : Event
            The event object.
        """

        # Prevent double click event from propagating to the viewer and zooming
        event.handled = True

        df = shared_data["df"]
        position = shared_data["position"]
        population = shared_data["population"]

        seg_data = viewer.layers["segmentation"].data

        # event.position is (frame, row, col) in array coordinates.
        try:
            frame = int(event.position[0])
            row = int(event.position[1])
            col = int(event.position[2])
            value_under = seg_data[frame, row, col]
            if value_under == 0:
                return None
        except Exception:
            logger.warning("Invalid mask value...")
            return None

        target_track_id = viewer.layers["segmentation"].selected_label

        # Guard: with no track picked (selected_label == 0) the propagation would
        # silently delete the clicked track from this frame on. Refuse instead.
        if target_track_id == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText(
                "No track is currently selected. Pick the surviving track with "
                "the colour picker (pipette) before propagating it to a cell."
            )
            msgBox.setWindowTitle("No track selected")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
            return None

        if value_under == target_track_id:
            # Nothing to do: the cell already belongs to the selected track.
            return None

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText(
            f"Do you want to propagate track {target_track_id} to the cell under the mouse, track {value_under}?"
        )
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.No:
            return None
        else:

            # Snapshot for undo before mutating anything.
            push_undo_snapshot(frame)

            if target_track_id not in df[
                "TRACK_ID"
            ].unique() and target_track_id in np.unique(
                seg_data[shared_data["selected_frame"]]
            ):
                # the selected cell in frame -1 is not in the table... we can add it to DataFrame
                current_labelm1 = seg_data[shared_data["selected_frame"]]
                original_labelm1 = locate_labels(
                    position,
                    population=population,
                    frames=shared_data["selected_frame"],
                )
                if original_labelm1 is None:
                    logger.warning(
                        "Could not load original labels for frame "
                        f"{shared_data['selected_frame']}; skipping correction."
                    )
                    undo_stack.pop()
                    return None
                original_labelm1[current_labelm1 != target_track_id] = 0
                props = regionprops_table(
                    original_labelm1,
                    intensity_image=None,
                    properties=["centroid", "label"],
                )
                props = pd.DataFrame(props)
                new_cell = props[["centroid-1", "centroid-0", "label"]].copy()
                new_cell.rename(
                    columns={
                        "centroid-1": "POSITION_X",
                        "centroid-0": "POSITION_Y",
                        "label": "class_id",
                    },
                    inplace=True,
                )
                new_cell["FRAME"] = shared_data["selected_frame"]
                new_cell["TRACK_ID"] = target_track_id
                df = pd.concat([df, new_cell], ignore_index=True)

            if value_under not in df["TRACK_ID"].unique():
                # the cell to add is not currently part of DataFrame, need to add measurement

                current_label = seg_data[frame]
                original_label = locate_labels(
                    position, population=population, frames=frame
                )
                if original_label is None:
                    logger.warning(
                        f"Could not load original labels for frame {frame}; "
                        "skipping correction."
                    )
                    undo_stack.pop()
                    return None

                original_label[current_label != value_under] = 0

                props = regionprops_table(
                    original_label,
                    intensity_image=None,
                    properties=["centroid", "label"],
                )
                props = pd.DataFrame(props)

                new_cell = props[["centroid-1", "centroid-0", "label"]].copy()
                new_cell.rename(
                    columns={
                        "centroid-1": "POSITION_X",
                        "centroid-0": "POSITION_Y",
                        "label": "class_id",
                    },
                    inplace=True,
                )
                new_cell["FRAME"] = frame
                new_cell["TRACK_ID"] = value_under
                df = pd.concat([df, new_cell], ignore_index=True)

            # Displace the tail of the selected track onto a fresh id, then hand
            # its identity to the clicked track. Use the cached running max id to
            # avoid scanning the whole label stack, and vectorise over the tail
            # rather than looping frame by frame.
            new_track_id = shared_data["max_label"] + 1
            shared_data["max_label"] = new_track_id

            tail = seg_data[frame:]
            tail[tail == target_track_id] = new_track_id
            tail[tail == value_under] = target_track_id

            df.loc[
                (df["FRAME"] >= frame) & (df["TRACK_ID"] == target_track_id),
                "TRACK_ID",
            ] = new_track_id
            df.loc[
                (df["FRAME"] >= frame) & (df["TRACK_ID"] == value_under), "TRACK_ID"
            ] = target_track_id
            df = df.sort_values(by=["TRACK_ID", "FRAME"])

            shared_data["df"] = df
            refresh_track_layers()

    viewer.show(block=block)

    if flush_memory and block:

        # temporary fix for slight napari memory leak — pop until empty (IndexError)
        for i in range(10000):
            try:
                viewer.layers.pop()
            except Exception:
                break

        del viewer
        del stack
        del labels
        gc.collect()


def load_napari_data(
    position: str,
    prefix: str = "Aligned",
    population: str = "target",
    return_stack: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Load the necessary data for visualization in napari.

    Parameters
    ----------
    position : str
            The path to the position or experiment directory.
    prefix : str, optional
            The prefix used to identify the movie file. The default is "Aligned".
    population : str, optional
            The population type to load, either "target" or "effector". The default is "target".
    return_stack : bool, optional
            Whether to return the image stack.

    Returns
    -------
    tuple
            A tuple containing the loaded data, properties, graph, labels, and stack.

    Examples
    --------
    >>> data, properties, graph, labels, stack = load_napari_data("path/to/position")
    # Load the necessary data for visualization of target trajectories.

    """

    if not position.endswith(os.sep):
        position += os.sep

    position = position.replace("\\", "/")
    if population.lower() == "target" or population.lower() == "targets":
        if os.path.exists(
            position
            + os.sep.join(["output", "tables", "napari_target_trajectories.npy"])
        ):
            napari_data = np.load(
                position
                + os.sep.join(["output", "tables", "napari_target_trajectories.npy"]),
                allow_pickle=True,
            )
        else:
            napari_data = None
    elif population.lower() == "effector" or population.lower() == "effectors":
        if os.path.exists(
            position
            + os.sep.join(["output", "tables", "napari_effector_trajectories.npy"])
        ):
            napari_data = np.load(
                position
                + os.sep.join(["output", "tables", "napari_effector_trajectories.npy"]),
                allow_pickle=True,
            )
        else:
            napari_data = None
    else:
        if os.path.exists(
            position
            + os.sep.join(["output", "tables", f"napari_{population}_trajectories.npy"])
        ):
            napari_data = np.load(
                position
                + os.sep.join(
                    ["output", "tables", f"napari_{population}_trajectories.npy"]
                ),
                allow_pickle=True,
            )
        else:
            napari_data = None

    if napari_data is not None:
        data = napari_data.item()["data"]
        properties = napari_data.item()["properties"]
        graph = napari_data.item()["graph"]
    else:
        data = None
        properties = None
        graph = None
    if return_stack:
        stack, labels = locate_stack_and_labels(
            position, prefix=prefix, population=population
        )
    else:
        labels = locate_labels(position, population=population)
        stack = None
    return data, properties, graph, labels, stack


def control_segmentation_napari(
    position: str,
    prefix: str = "Aligned",
    population: str = "target",
    flush_memory: bool = False,
    threads: int = 1,
    progress_callback: Optional[Callable[[int], bool]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    prepare_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """

    Control the visualization of segmentation labels using the napari viewer.

    Parameters
    ----------
    position : str
            The position or directory path where the segmentation labels and stack are located.
    prefix : str, optional
            The prefix used to identify the stack. The default is 'Aligned'.
    population : str, optional
            The population type for which the segmentation is performed. The default is 'target'.
    flush_memory : bool, optional
            Pop napari layers upon closing the viewer to empty the memory footprint. The default is `False`.
    threads : int, optional
            Number of threads used to load the per-frame masks in parallel.
    progress_callback : function, optional
            Called with an int in [0, 100] as the data loads.
    status_callback : function, optional
            Called with a short phase message (e.g. "Loading masks…").
    prepare_only : bool, optional
            If True, load and prepare the data but do not create the viewer;
            return the keyword arguments for :func:`launch_segmentation_viewer`
            instead. Use this to run the slow loading phase in a worker thread
            and create the viewer on the GUI thread.

    Notes
    -----
    This function loads the segmentation labels and stack corresponding to the specified position and population.
    It then creates a napari viewer and adds the stack and labels as layers for visualization.

    Examples
    --------
    >>> control_segmentation_napari(position, prefix='Aligned', population="target")
    # Control the visualization of segmentation labels using the napari viewer.

    """

    if not position.endswith(os.sep):
        position += os.sep
    position = position.replace("\\", "/")

    if progress_callback:
        progress_callback(0)

    # --- Load masks (parallel, with progress) ---
    if status_callback:
        status_callback("Loading masks…")

    def _labels_progress(p: int) -> None:
        # Map mask-loading progress onto the 0–60 band of the overall bar.
        if progress_callback:
            progress_callback(int(p * 0.60))

    n_label_threads = max(int(threads), 4)
    labels = locate_labels(
        position,
        population=population,
        threads=n_label_threads,
        progress_callback=_labels_progress,
    )

    # --- Load image stack (lazily when possible, else eagerly) ---
    if status_callback:
        status_callback("Loading image stack…")
    if progress_callback:
        progress_callback(60)

    stack = locate_stack_lazy(position, prefix=prefix)
    if stack is None:
        stack = locate_stack(position, prefix=prefix)

    # Mirror locate_stack_and_labels: repair/realign label count if needed.
    if labels is None or len(labels) < len(stack):
        fix_missing_labels(position, population=population, prefix=prefix)
        labels = locate_labels(
            position, population=population, threads=n_label_threads
        )
    if len(stack) != len(labels):
        raise ValueError(
            f"The shape of the stack {getattr(stack, 'shape', None)} does not "
            f"match with the shape of the labels {getattr(labels, 'shape', None)}"
        )

    if status_callback:
        status_callback("Adjusting the contrast…")
    if progress_callback:
        progress_callback(90)

    contrast_limits = _get_contrast_limits(stack)

    if progress_callback:
        progress_callback(100)

    data = {
        "stack": stack,
        "labels": labels,
        "contrast_limits": contrast_limits,
        "position": position,
        "population": population,
        "flush_memory": flush_memory,
    }

    if prepare_only:
        return data

    return launch_segmentation_viewer(**data)


def launch_segmentation_viewer(
    stack: np.ndarray,
    labels: np.ndarray,
    contrast_limits: Optional[List[Tuple[float, float]]],
    position: str,
    population: str = "target",
    flush_memory: bool = False,
    block: bool = True,
) -> None:
    """
    Create the napari viewer for segmentation inspection from pre-loaded data.

    Must run on the GUI thread. The data is typically prepared by
    :func:`control_segmentation_napari` (optionally in a worker thread with
    ``prepare_only=True``).

    Parameters
    ----------
    stack : ndarray or dask.array.Array
        The image stack shaped (T, Y, X, C).
    labels : ndarray
        The label stack shaped (T, Y, X).
    contrast_limits : list of tuple or None
        Per-channel contrast limits for the image layers.
    position : str
        The position folder (with trailing separator) — used to save labels
        and annotations.
    population : str, optional
        The population whose masks are displayed. The default is 'target'.
    flush_memory : bool, optional
        Pop napari layers upon closing the viewer to empty the memory
        footprint. Only effective when ``block=True``.
    block : bool, optional
        Whether to block execution while the viewer is open.
    """

    def export_labels():
        """Export corrected labels."""
        labels_layer = viewer.layers["segmentation"].data
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im)
            except Exception as e:
                logger.warning(f"auto_correct_masks failed: {e}")

            save_tiff_imagej_compatible(
                output_folder + f"{str(t).zfill(4)}.tif", im.astype(np.int16), axes="YX"
            )
        logger.info("The labels have been successfully rewritten.")

    def export_annotation():
        """Export annotation data."""

        # Locate experiment config
        parent1 = Path(position).parent
        expfolder = parent1.parent
        config = PurePath(expfolder, Path("config.ini"))
        expfolder = str(expfolder)
        exp_name = os.path.split(expfolder)[-1]

        wells = get_experiment_wells(expfolder)
        well_idx = list(wells).index(str(parent1) + os.sep)

        label_info = get_experiment_labels(expfolder)
        metadata_info = get_experiment_metadata(expfolder)

        info = {}
        for k in list(label_info.keys()):
            values = label_info[k]
            try:
                info.update({k: values[well_idx]})
            except Exception as e:
                logger.warning(f"Failed to retrieve label info for key '{k}': {e}")

        if metadata_info is not None:
            keys = list(metadata_info.keys())
            for k in keys:
                info.update({k: metadata_info[k]})

        spatial_calibration = float(
            config_section_to_dict(config, "MovieSettings")["pxtoum"]
        )
        channel_names, channel_indices = extract_experiment_channels(expfolder)

        annotation_folder = expfolder + os.sep + f"annotations_{population}" + os.sep
        if not os.path.exists(annotation_folder):
            os.mkdir(annotation_folder)

        logger.info("Exporting annotation...")
        t = viewer.dims.current_step[0]
        labels_layer = viewer.layers["segmentation"].data[t]  # at current time

        try:
            labels_layer = auto_correct_masks(labels_layer)
        except Exception as e:
            logger.warning(f"auto_correct_masks failed: {e}")

        fov_export = True

        if "Shapes" in viewer.layers:
            squares = viewer.layers["Shapes"].data
            test_in_frame = np.array(
                [
                    squares[i][0, 0] == t and len(squares[i]) == 4
                    for i in range(len(squares))
                ]
            )
            squares = np.array(squares)
            squares = squares[test_in_frame]
            nbr_squares = len(squares)
            logger.info(f"Found {nbr_squares} ROIs...")
            if nbr_squares > 0:
                # deactivate field of view mode
                fov_export = False

            for k, sq in enumerate(squares):
                logger.debug(f"ROI: {sq}")
                pad_to_256 = False

                xmin = int(sq[0, 1])
                xmax = int(sq[2, 1])
                if xmax < xmin:
                    xmax, xmin = xmin, xmax
                ymin = int(sq[0, 2])
                ymax = int(sq[1, 2])
                if ymax < ymin:
                    ymax, ymin = ymin, ymax
                logger.debug(f"xmin={xmin};xmax={xmax};ymin={ymin};ymax={ymax}")
                frame = viewer.layers["Image"].data[t][xmin:xmax, ymin:ymax]
                if frame.shape[1] < 256 or frame.shape[0] < 256:
                    pad_to_256 = True
                    logger.warning(
                        "Crop too small! Padding with zeros to reach 256*256 pixels..."
                    )
                    # continue
                multichannel = [frame]
                for i in range(len(channel_indices) - 1):
                    try:
                        frame = viewer.layers[f"Image [{i + 1}]"].data[t][
                            xmin:xmax, ymin:ymax
                        ]
                        multichannel.append(frame)
                    except Exception as e:
                        logger.debug(f"Could not extract frame from layer Image [{i + 1}]: {e}")
                multichannel = np.array(multichannel)
                lab = labels_layer[xmin:xmax, ymin:ymax].astype(np.int16)
                if pad_to_256:
                    shape = multichannel.shape
                    pad_length_x = max([0, 256 - multichannel.shape[1]])
                    if pad_length_x > 0 and pad_length_x % 2 == 1:
                        pad_length_x += 1
                    pad_length_y = max([0, 256 - multichannel.shape[2]])
                    if pad_length_y > 0 and pad_length_y % 2 == 1:
                        pad_length_y += 1
                    padded_image = np.array(
                        [
                            np.pad(
                                im,
                                (
                                    (pad_length_x // 2, pad_length_x // 2),
                                    (pad_length_y // 2, pad_length_y // 2),
                                ),
                                mode="constant",
                            )
                            for im in multichannel
                        ]
                    )
                    padded_label = np.pad(
                        lab,
                        (
                            (pad_length_x // 2, pad_length_x // 2),
                            (pad_length_y // 2, pad_length_y // 2),
                        ),
                        mode="constant",
                    )
                    lab = padded_label
                    multichannel = padded_image

                save_tiff_imagej_compatible(
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}_labelled.tif",
                    lab,
                    axes="YX",
                )
                save_tiff_imagej_compatible(
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.tif",
                    multichannel,
                    axes="CYX",
                )

                info.update(
                    {
                        "spatial_calibration": spatial_calibration,
                        "channels": list(channel_names),
                        "frame": t,
                    }
                )

                info_name = (
                    annotation_folder
                    + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.json"
                )
                with open(info_name, "w") as f:
                    json.dump(info, f, indent=4)

        if fov_export:
            frame = viewer.layers["Image"].data[t]
            multichannel = [frame]
            for i in range(len(channel_indices) - 1):
                try:
                    frame = viewer.layers[f"Image [{i + 1}]"].data[t]
                    multichannel.append(frame)
                except Exception as e:
                    logger.debug(f"Could not extract frame from layer Image [{i + 1}] at t={t}: {e}")
            multichannel = np.array(multichannel)
            save_tiff_imagej_compatible(
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_labelled.tif",
                labels_layer,
                axes="YX",
            )
            save_tiff_imagej_compatible(
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.tif",
                multichannel,
                axes="CYX",
            )

            info.update(
                {
                    "spatial_calibration": spatial_calibration,
                    "channels": list(channel_names),
                    "frame": t,
                }
            )

            info_name = (
                annotation_folder
                + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.json"
            )
            with open(info_name, "w") as f:
                json.dump(info, f, indent=4)

        logger.info("Annotation export done.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        """
        Widget to trigger saving.

        Returns
        -------
        function
            The export function.
        """
        return export_labels()

    @magicgui(call_button="Export the annotation\nof the current frame")
    def export_widget():
        """Widget to trigger export."""
        return export_annotation()

    output_folder = position + f"labels_{population}{os.sep}"
    logger.info(f"Shape of the loaded image stack: {stack.shape}...")

    viewer = napari.Viewer()
    try:
        viewer.window._qt_window.setWindowIcon(Styles().celldetective_icon)
    except Exception as e:
        logger.debug(f"Could not set napari window icon: {e}")
    viewer.add_image(
        stack,
        channel_axis=-1,
        colormap=["gray"] * stack.shape[-1],
        contrast_limits=contrast_limits,
    )
    # Avoid a full int64 copy of the whole TYX stack when the labels are
    # already an integer type.
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int32)
    viewer.add_labels(labels, name="segmentation", opacity=0.4)

    button_container = QWidget()
    layout = QVBoxLayout(button_container)
    layout.setSpacing(10)
    layout.addWidget(save_widget.native)
    layout.addWidget(export_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

    save_widget.native.setStyleSheet(Styles().button_style_sheet)
    export_widget.native.setStyleSheet(Styles().button_style_sheet)

    def lock_controls(
        layer: napari.layers.Layer, widgets: Tuple[str, ...] = (), locked: bool = True
    ) -> None:
        """
        Lock or unlock controls.

        Parameters
        ----------
        layer : str
            The layer name.
        widgets : tuple, optional
            Widgets to lock/unlock.
        locked : bool, optional
            Whether to lock or unlock.
        """
        qctrl = viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widgets:
            try:
                getattr(qctrl, wdg).setEnabled(not locked)
            except Exception as e:
                logger.debug(f"Could not set {wdg} enabled state: {e}")

    label_widget_list = ["polygon_button", "transform_button"]
    lock_controls(viewer.layers["segmentation"], label_widget_list)

    viewer.show(block=block)

    if flush_memory:
        # temporary fix for slight napari memory leak — pop until IndexError (empty)
        for i in range(10000):
            try:
                viewer.layers.pop()
            except Exception:
                break

        del viewer
        del stack
        del labels
        gc.collect()

    logger.info("napari viewer was successfully closed...")


def correct_annotation(filename: str) -> None:
    """
    New function to reannotate an annotation image in post, using napari and save update inplace.

    Parameters
    ----------
    filename : str
        The path to the annotation file.
    """

    def export_labels():
        """Export corrected labels to file."""
        labels_layer = viewer.layers["segmentation"].data
        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im)
            except Exception as e:
                logger.warning(f"auto_correct_masks failed: {e}")

            save_tiff_imagej_compatible(existing_lbl, im.astype(np.int16), axes="YX")
        logger.info("The labels have been successfully rewritten.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        """
        Widget to trigger saving.

        Returns
        -------
        function
            The export function.
        """
        return export_labels()

    if filename.endswith("_labelled.tif"):
        filename = filename.replace("_labelled.tif", ".tif")
    if filename.endswith(".json"):
        filename = filename.replace(".json", ".tif")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Image {filename} does not seem to exist...")

    img = imread(filename.replace("\\", "/"))
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    elif img.ndim == 2:
        img = img[:, :, np.newaxis]

    existing_lbl = filename.replace(".tif", "_labelled.tif")
    if os.path.exists(existing_lbl):
        labels = imread(existing_lbl)[np.newaxis, :, :].astype(int)
    else:
        labels = np.zeros_like(img[:, :, 0]).astype(int)[np.newaxis, :, :]

    stack = img[np.newaxis, :, :, :]
    contrast_limits = _get_contrast_limits(stack)
    viewer = napari.Viewer()
    viewer.add_image(
        stack,
        channel_axis=-1,
        colormap=["gray"] * stack.shape[-1],
        contrast_limits=contrast_limits,
    )
    viewer.add_labels(labels, name="segmentation", opacity=0.4)
    viewer.window.add_dock_widget(save_widget, area="right")
    save_widget.native.setStyleSheet(Styles().button_style_sheet)

    viewer.show(block=False)


def _view_on_napari(
    tracks: Optional[pd.DataFrame] = None,
    stack: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    """

    Visualize tracks, stack, and labels using Napari.

    Parameters
    ----------
    tracks : pandas DataFrame
            DataFrame containing track information.
    stack : numpy array, optional
            Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
            and C is the number of channels. Default is None.
    labels : numpy array, optional
            Label stack with shape (T, Y, X) representing cell segmentations. Default is None.

    Returns
    -------
    None

    Notes
    -----
    This function visualizes tracks, stack, and labels using Napari, an interactive multi-dimensional image viewer.
    The tracks are represented as line segments on the viewer. If a stack is provided, it is displayed as an image.
    If labels are provided, they are displayed as a segmentation overlay on the stack.

    Examples
    --------
    >>> tracks = pd.DataFrame({'track': [1, 2, 3], 'time': [1, 1, 1],
    ...                        'x': [10, 20, 30], 'y': [15, 25, 35]})
    >>> stack = np.random.rand(100, 100, 3)
    >>> labels = np.random.randint(0, 2, (100, 100))
    >>> _view_on_napari(tracks, stack=stack, labels=labels)
    # Visualize tracks, stack, and labels using Napari.

    """

    viewer = napari.Viewer()
    contrast_limits = _get_contrast_limits(stack)
    if stack is not None:
        viewer.add_image(
            stack,
            channel_axis=-1,
            colormap=["gray"] * stack.shape[-1],
            contrast_limits=contrast_limits,
        )
    if labels is not None:
        viewer.add_labels(labels, name="segmentation", opacity=0.4)
    if tracks is not None:
        viewer.add_tracks(tracks, name="tracks")
    viewer.show(block=True)


def control_tracking_table(
    position: str,
    calibration: float = 1,
    prefix: str = "Aligned",
    population: str = "target",
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "frame": "FRAME",
        "y": "POSITION_Y",
        "x": "POSITION_X",
        "label": "class_id",
    },
) -> None:
    """

    Control the tracking table and visualize tracks using Napari.

    Parameters
    ----------
    position : str
            The position or directory of the tracking data.
    calibration : float, optional
            Calibration factor for converting pixel coordinates to physical units. Default is 1.
    prefix : str, optional
            Prefix used for the tracking data file. Default is "Aligned".
    population : str, optional
            Population type, either "target" or "effector". Default is "target".
    column_labels : dict, optional
            Dictionary containing the column labels for the tracking table. Default is
            {'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X', 'label': 'class_id'}.

    Returns
    -------
    None

    Notes
    -----
    This function loads the tracking data, applies calibration to the spatial coordinates, and visualizes the tracks
    using Napari. The tracking data is loaded from the specified `position` directory with the given `prefix` and
    `population`. The spatial coordinates (x, y) in the tracking table are divided by the `calibration` factor to
    convert them from pixel units to the specified physical units. The tracks are then visualized using Napari.

    Examples
    --------
    >>> control_tracking_table('path/to/tracking_data', calibration=0.1, prefix='Aligned', population='target')
    # Control the tracking table and visualize tracks using Napari.

    """

    position = position.replace("\\", "/")
    tracks, labels, stack = load_tracking_data(
        position, prefix=prefix, population=population
    )
    tracks = tracks.loc[
        :,
        [
            column_labels["track"],
            column_labels["frame"],
            column_labels["y"],
            column_labels["x"],
        ],
    ].to_numpy()
    tracks[:, -2:] /= calibration
    _view_on_napari(tracks, labels=labels, stack=stack)
