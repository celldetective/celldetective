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

from celldetective.utils.data_cleaning import tracks_to_btrack, extract_identity_col
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
    get_spatial_calibration,
)
from celldetective.utils.parsing import config_section_to_dict
from celldetective import get_logger
from celldetective.log_manager import positionlogger
from celldetective.gui.base.styles import Styles

logger = get_logger()


def _drop_fully_maskless_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tracks that have no mask in any frame.

    A position with no mask has a NaN ``class_id``. A track whose every position
    is maskless carries no segmentation at all and can only be a "ghost" — for
    instance one left behind when a correction reassigned all of a track's masks
    to another track. Such tracks are dropped, while any track that keeps at
    least one real detection is preserved untouched (including its interpolated
    gaps), so sparse edits don't lose data.

    Parameters
    ----------
    df : pandas.DataFrame
        Trajectory table with ``TRACK_ID`` and ``class_id`` columns.

    Returns
    -------
    pandas.DataFrame
        The table without fully-maskless tracks (index reset). Returned
        unchanged if the required columns are missing.
    """
    if "class_id" not in df.columns or "TRACK_ID" not in df.columns:
        return df
    has_mask = df["class_id"].notna().groupby(df["TRACK_ID"]).transform("any")
    n_before = df["TRACK_ID"].nunique()
    cleaned = df[has_mask].reset_index(drop=True)
    n_dropped = n_before - cleaned["TRACK_ID"].nunique()
    if n_dropped > 0:
        if cleaned.empty and not df.empty:
            # Every track is maskless: far more likely an unpopulated class_id
            # column than genuinely all-ghost data. Don't silently empty the
            # table — leave it untouched and warn instead.
            logger.warning(
                "All tracks appear maskless (class_id is entirely NaN); keeping "
                "the table unchanged. Has tracking/measurement populated class_id?"
            )
            return df
        logger.info(f"Dropped {n_dropped} fully maskless (ghost) track(s).")
    return cleaned


def control_tracks(
    position: str,
    prefix: str = "Aligned",
    population: str = "target",
    relabel: bool = True,
    flush_memory: bool = True,
    threads: int = 1,
    progress_callback: Optional[Callable[[int], bool]] = None,
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

    stack, labels = locate_stack_and_labels(
        position, prefix=prefix, population=population
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

    data, properties, graph = tracks_to_btrack(df, exclude_nans=exclude_nans)
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
    df, df_path = get_position_table(position, population=population, return_path=True)
    logger.debug(f"get_position_table returned df={df is not None}")

    if progress_callback:
        progress_callback(50)

    if df is None:
        logger.warning("Please compute trajectories first... Abort...")
        return None

    # Drop "ghost" tracks that have no mask in any frame (e.g. left behind by an
    # earlier correction that reassigned all of a track's masks). Tracks that
    # keep at least one real detection are preserved untouched — including their
    # interpolated positions — so sparse edits don't lose data. Ghosts created
    # during this session are cleaned again on export.
    df = _drop_fully_maskless_tracks(df)

    shared_data = {
        "df": df,
        "path": df_path,
        "position": position,
        "population": population,
        "selected_frame": None,
    }

    if (labels is not None) * relabel:
        logger.info("Replacing the cell mask labels with the track ID...")

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
        # already an integer type.
        if not np.issubdtype(labels.dtype, np.integer):
            labels = labels.astype(np.int32)
        labels_layer = viewer.add_labels(labels, name="segmentation", opacity=0.4)
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

    # Defaults for the post-processing options: start from clean_trajectories'
    # own defaults, then override with whatever is already configured in the
    # position's tracking instructions (so the widgets reflect the current setup).
    _pp_defaults = {
        "remove_not_in_first": False,
        "remove_not_in_last": False,
        "minimum_tracklength": 0,
        "interpolate_position_gaps": False,
        "extrapolate_tracks_post": False,
        "extrapolate_tracks_pre": False,
        "interpolate_na": False,
    }
    try:
        _experiment = extract_experiment_from_position(shared_data["position"])
        _instruction_file = "/".join(
            [
                _experiment,
                "configs",
                f"tracking_instructions_{shared_data['population']}.json",
            ]
        )
        if os.path.exists(_instruction_file):
            with open(_instruction_file, "r") as f:
                _opts = (json.load(f) or {}).get("post_processing_options") or {}
            for _k in _pp_defaults:
                if _opts.get(_k) is not None:
                    _pp_defaults[_k] = _opts[_k]
    except Exception as e:
        logger.debug(f"Could not load existing post-processing options: {e}")

    @magicgui(
        layout="vertical",
        call_button=False,
        remove_not_in_first={
            "widget_type": "CheckBox",
            "text": "Remove tracks that do not start at the beginning",
        },
        remove_not_in_last={
            "widget_type": "CheckBox",
            "text": "Remove tracks that do not end at the end",
        },
        interpolate_position_gaps={
            "widget_type": "CheckBox",
            "text": "Interpolate missed detections within tracks",
        },
        extrapolate_tracks_pre={
            "widget_type": "CheckBox",
            "text": "Sustain first position from the beginning of the movie",
        },
        extrapolate_tracks_post={
            "widget_type": "CheckBox",
            "text": "Sustain last position until the end of the movie",
        },
        interpolate_na={
            "widget_type": "CheckBox",
            "text": "Interpolate missing values",
        },
        minimum_tracklength={
            "label": "Min. tracklength",
            "min": 0,
            "max": 1_000_000,
        },
    )
    def post_processing_options(
        remove_not_in_first: bool = False,
        remove_not_in_last: bool = False,
        interpolate_position_gaps: bool = False,
        extrapolate_tracks_pre: bool = False,
        extrapolate_tracks_post: bool = False,
        interpolate_na: bool = False,
        minimum_tracklength: int = 0,
    ):
        """Track post-processing applied when exporting the corrected tracks."""

    # Seed the widgets with the currently-configured options.
    for _k, _v in _pp_defaults.items():
        try:
            getattr(post_processing_options, _k).value = _v
        except Exception as e:
            logger.debug(f"Could not set default for {_k}: {e}")

    def _post_processing_kwargs() -> Dict[str, Any]:
        """Build ``clean_trajectories`` kwargs from the option widgets."""
        return {
            "remove_not_in_first": post_processing_options.remove_not_in_first.value,
            "remove_not_in_last": post_processing_options.remove_not_in_last.value,
            "minimum_tracklength": post_processing_options.minimum_tracklength.value,
            "interpolate_position_gaps": post_processing_options.interpolate_position_gaps.value,
            "extrapolate_tracks_post": post_processing_options.extrapolate_tracks_post.value,
            "extrapolate_tracks_pre": post_processing_options.extrapolate_tracks_pre.value,
            "interpolate_na": post_processing_options.interpolate_na.value,
        }

    def export_modifications():
        """Export modified tracks, applying the chosen post-processing options."""

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

        post_processing_opts = _post_processing_kwargs()
        logger.info(
            f"Applying the following track postprocessing: {post_processing_opts}..."
        )
        df = clean_trajectories(df.copy(), **post_processing_opts)

        # Remove any ghost tracks (no mask in any frame) created by corrections
        # before writing the table.
        df = _drop_fully_maskless_tracks(df)

        unnamed_cols = [c for c in list(df.columns) if c.startswith("Unnamed")]
        df = df.drop(unnamed_cols, axis=1)
        logger.debug(f"Columns after export: {list(df.columns)}")
        df.to_csv(shared_data["path"], index=False)
        logger.info("Track export done.")

        # Reflect the post-processed tracks in the viewer so the effect of the
        # chosen options is visible in place (dropped tracks disappear, gaps get
        # interpolated, etc.).
        shared_data["df"] = df
        try:
            vertices, tracks, properties, graph = tracks_to_napari(
                df, exclude_nans=True
            )
            viewer.layers["tracks"].data = tracks
            viewer.layers["tracks"].properties = properties
            viewer.layers["tracks"].graph = graph
            viewer.layers["points"].data = vertices
            viewer.layers["tracks"].refresh()
            viewer.layers["points"].refresh()
        except Exception as e:
            logger.warning(f"Could not refresh track layers after export: {e}")

        with positionlogger(position, filename=f"log_{population}.txt"):
            logger.info("TRACK CORRECTION (manual, napari)")
            logger.info(f"population: {population}")
            logger.info(f"post_processing: {post_processing_opts}")
            try:
                id_col = extract_identity_col(df)
                n_tracks = df[id_col].nunique() if id_col is not None else None
                logger.info(f"tracks: {n_tracks}, detections: {len(df)}")
            except Exception as e:
                logger.warning(f"Could not summarise track correction: {e}")

    @magicgui(call_button="Export the modified\ntracks...")
    def export_table_widget():
        """Widget to trigger export."""
        return export_modifications()

    export_table_widget.native.setStyleSheet(Styles().button_style_sheet)
    post_processing_options.native.setStyleSheet(Styles().button_style_sheet)

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
    track_button_layout.addWidget(post_processing_options.native)
    track_button_layout.addWidget(export_table_widget.native)
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

        frame, x, y = event.position
        try:
            value_under = viewer.layers["segmentation"].data[
                int(frame), int(x), int(y)
            ]  # labels[0,int(y),int(x)]
            if value_under == 0:
                return None
        except Exception:
            logger.warning("Invalid mask value...")
            return None

        target_track_id = viewer.layers["segmentation"].selected_label

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

            if target_track_id not in df[
                "TRACK_ID"
            ].unique() and target_track_id in np.unique(
                viewer.layers["segmentation"].data[shared_data["selected_frame"]]
            ):
                # the selected cell in frame -1 is not in the table... we can add it to DataFrame
                current_labelm1 = viewer.layers["segmentation"].data[
                    shared_data["selected_frame"]
                ]
                original_labelm1 = locate_labels(
                    position,
                    population=population,
                    frames=shared_data["selected_frame"],
                )
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

                current_label = viewer.layers["segmentation"].data[int(frame)]
                original_label = locate_labels(
                    position, population=population, frames=int(frame)
                )

                new_datapoint = {
                    "TRACK_ID": value_under,
                    "FRAME": frame,
                    "POSITION_X": np.nan,
                    "POSITION_Y": np.nan,
                    "class_id": np.nan,
                }

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
                new_cell["FRAME"] = int(frame)
                new_cell["TRACK_ID"] = value_under
                df = pd.concat([df, new_cell], ignore_index=True)

            relabel = np.amax(viewer.layers["segmentation"].data) + 1
            for f in viewer.layers["segmentation"].data[int(frame) :]:
                if target_track_id != 0:
                    f[np.where(f == target_track_id)] = relabel
                f[np.where(f == value_under)] = target_track_id

            if target_track_id != 0:
                df.loc[
                    (df["FRAME"] >= frame) & (df["TRACK_ID"] == target_track_id),
                    "TRACK_ID",
                ] = relabel
            df.loc[
                (df["FRAME"] >= frame) & (df["TRACK_ID"] == value_under), "TRACK_ID"
            ] = target_track_id
            df = df.loc[~(df["TRACK_ID"] == 0), :]
            df = df.sort_values(by=["TRACK_ID", "FRAME"])

            vertices, tracks, properties, graph = tracks_to_napari(
                df, exclude_nans=True
            )

            viewer.layers["tracks"].data = tracks
            viewer.layers["tracks"].properties = properties
            viewer.layers["tracks"].graph = graph

            viewer.layers["points"].data = vertices

            viewer.layers["segmentation"].refresh()
            viewer.layers["tracks"].refresh()
            viewer.layers["points"].refresh()

        shared_data["df"] = df

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

    Load the segmentation labels and stack for a position, and (optionally) open
    them in napari for visualization and correction.

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
            Reserved for interface compatibility with the GUI loader thread. The default is `1`.
    progress_callback : callable, optional
            Callback receiving an int (0-100) while loading. The default is None.
    status_callback : callable, optional
            Callback receiving a status string while loading. The default is None.
    prepare_only : bool, optional
            If True, load the data and return it as a dict instead of opening the
            viewer (used to run the loading step off the GUI thread). The default
            is `False`.

    Returns
    -------
    dict or None
            The prepared data dict when ``prepare_only`` is True, otherwise None.

    Notes
    -----
    This function loads the segmentation labels and stack corresponding to the specified position and population.
    The viewer itself is built by :func:`launch_segmentation_viewer`.

    Examples
    --------
    >>> control_segmentation_napari(position, prefix='Aligned', population="target")
    # Control the visualization of segmentation labels using the napari viewer.

    """

    # --- Load masks (parallel, with granular progress) ---
    if status_callback:
        status_callback("Loading masks…")
    if progress_callback:
        progress_callback(0)

    def _labels_progress(p: int):
        # Masks dominate the load cost; map their progress onto the 0–90 band.
        # Returns the callback's value so a cancel request (False) aborts the load.
        if progress_callback:
            return progress_callback(int(p * 0.9))
        return True

    n_label_threads = max(int(threads), 4)
    labels = locate_labels(
        position,
        population=population,
        threads=n_label_threads,
        progress_callback=_labels_progress,
    )

    # locate_labels returns None when cancelled via the progress callback: stop
    # here so we don't waste time loading the stack / computing contrast limits.
    if labels is None:
        if status_callback:
            status_callback("Cancelled.")
        return None

    # --- Load image stack (lazily when possible, else eagerly) ---
    if status_callback:
        status_callback("Loading image stack…")
    if progress_callback:
        progress_callback(90)

    stack = locate_stack_lazy(position, prefix=prefix)
    if stack is None:
        stack = locate_stack(position, prefix=prefix)

    # Mirror locate_stack_and_labels: repair/realign label count if needed.
    if len(labels) < len(stack):
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
        progress_callback(100)

    contrast_limits = _get_contrast_limits(stack)

    data = {
        "stack": stack,
        "labels": labels,
        "position": position,
        "population": population,
        "contrast_limits": contrast_limits,
        "flush_memory": flush_memory,
    }

    if prepare_only:
        return data

    launch_segmentation_viewer(**data)
    return None


def launch_segmentation_viewer(
    stack: np.ndarray,
    labels: np.ndarray,
    position: str,
    population: str = "target",
    contrast_limits: Optional[List[Tuple[float, float]]] = None,
    flush_memory: bool = False,
    block: bool = True,
    progress_callback: Optional[Callable[[Any], Any]] = None,
) -> None:
    """
    Build the napari viewer for segmentation visualization and correction.

    Parameters
    ----------
    stack : numpy.ndarray
            The image stack (TYXC).
    labels : numpy.ndarray
            The label stack (TYX).
    position : str
            The position directory (used to locate config and write corrections).
    population : str, optional
            The population type. The default is 'target'.
    contrast_limits : list, optional
            Contrast limits for the image layers. Computed from the stack if None.
    flush_memory : bool, optional
            Pop napari layers upon closing the viewer to empty the memory footprint.
    block : bool, optional
            Whether to block while the viewer is open. The default is `True`.
    progress_callback : callable, optional
            Optional callback receiving status strings during viewer init.
    """

    @magicgui(
        layout="vertical",
        call_button=False,
        split_merged_labels={
            "widget_type": "CheckBox",
            "text": "Split merged labels",
        },
        remove_small_objects={
            "widget_type": "CheckBox",
            "text": "Remove small objects",
        },
        fill_holes={"widget_type": "CheckBox", "text": "Fill holes in masks"},
        min_area={
            "label": "Min object area (px²)",
            "min": 0,
            "max": 1_000_000,
        },
    )
    def correction_options(
        split_merged_labels: bool = True,
        remove_small_objects: bool = True,
        fill_holes: bool = False,
        min_area: int = 9,
    ):
        """Auto-fixes applied to the masks when saving."""

    def _correction_kwargs() -> Dict[str, Any]:
        """Build ``auto_correct_masks`` kwargs from the option widgets."""
        return {
            "correct_anomalies": correction_options.split_merged_labels.value,
            "fill_labels": correction_options.fill_holes.value,
            "min_area": (
                correction_options.min_area.value
                if correction_options.remove_small_objects.value
                else 0
            ),
        }

    def export_labels():
        """Export corrected labels."""
        from PyQt5.QtWidgets import QApplication, QProgressDialog
        from PyQt5.QtCore import Qt

        labels_layer = viewer.layers["segmentation"].data
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        n_total = len(labels_layer)
        # Saving can take a few seconds; show a modal, non-cancellable progress
        # bar and pump the event loop each frame so it stays responsive.
        try:
            parent = viewer.window._qt_window
        except Exception:
            parent = None
        pbar = QProgressDialog("Saving the modified labels…", None, 0, n_total, parent)
        pbar.setWindowTitle("Saving")
        pbar.setWindowModality(Qt.WindowModal)
        pbar.setMinimumDuration(0)
        pbar.setAutoClose(False)
        # Free the dialog widget when it closes so repeated saves don't accumulate
        # hidden QProgressDialog children on the napari main window.
        pbar.setAttribute(Qt.WA_DeleteOnClose)
        pbar.setValue(0)
        QApplication.processEvents()

        corrected_stack = labels_layer.copy()
        n_frames = 0
        n_objects_total = 0
        try:
            for t, im in enumerate(tqdm(labels_layer)):

                try:
                    im = auto_correct_masks(im, **_correction_kwargs())
                except Exception as e:
                    logger.warning(f"auto_correct_masks failed: {e}")

                corrected_stack[t] = im.astype(corrected_stack.dtype)
                save_tiff_imagej_compatible(
                    output_folder + f"{str(t).zfill(4)}.tif",
                    im.astype(np.int16),
                    axes="YX",
                )
                n_frames += 1
                n_objects_total += int((np.unique(im) != 0).sum())

                pbar.setValue(t + 1)
                QApplication.processEvents()

            # Reflect the auto-fixed masks back into the viewer so the result is visible
            viewer.layers["segmentation"].data = corrected_stack
            viewer.layers["segmentation"].refresh()
        finally:
            pbar.close()

        logger.info("The labels have been successfully rewritten.")
        with positionlogger(position, filename=f"log_{population}.txt"):
            logger.info("LABEL CORRECTION (manual, napari)")
            logger.info(f"population: {population}")
            logger.info(f"output_folder: {output_folder}")
            logger.info(
                f"frames written: {n_frames}, labelled objects (summed over frames): {n_objects_total}"
            )

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
            labels_layer = auto_correct_masks(labels_layer, **_correction_kwargs())
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

    # ------------------------------------------------------------------
    # Segment the frame currently on screen
    # ------------------------------------------------------------------

    # Loading a segmentation model costs seconds; keep each one alive for as
    # long as the viewer is open so that repeated calls only pay for inference.
    prepared_models: Dict[str, Any] = {}

    def _available_segmentation_models() -> List[str]:
        """
        List the models offered in the dropdown: population-specific, then generic.

        Returns
        -------
        list of str
            Model names, without duplicates, in the order they are offered.
        """

        from celldetective.utils.model_getters import get_segmentation_models_list

        models: List[str] = []
        for mode in (population, "generic"):
            try:
                models.extend(
                    get_segmentation_models_list(mode=mode, return_path=False)
                )
            except Exception as e:
                # Listing reaches out to the model repository; a network failure
                # must not stop the viewer from opening.
                logger.warning(f"Could not list the '{mode}' segmentation models: {e}")

        seen = set()
        return [m for m in models if not (m in seen or seen.add(m))]

    def _segmentation_failed(message: str) -> None:
        """
        Report a segmentation failure without tearing down the viewer.

        Parameters
        ----------
        message : str
            The message shown to the user and written to the log.
        """

        logger.error(message)
        viewer.status = message
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Warning)
            box.setText(message)
            box.setWindowTitle("Segmentation")
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
        except Exception as e:
            logger.debug(f"Could not show the segmentation error dialog: {e}")

    @magicgui(
        call_button="Segment this frame",
        model={"label": "model", "choices": _available_segmentation_models()},
        replace_existing={
            "widget_type": "CheckBox",
            "text": "Replace the labels on this frame",
        },
    )
    def segment_frame_widget(model: str, replace_existing: bool = True) -> None:
        """
        Segment the frame currently displayed, using the selected model.

        Runs on the frame the time slider is on, writes the result straight into
        the segmentation layer, and leaves every other frame untouched. Nothing
        is written to disk until the labels are saved.

        Parameters
        ----------
        model : str
            Name of the segmentation model to run.
        replace_existing : bool, optional
            If True, the labels on this frame are replaced. If False, existing
            labels are kept and the new ones only fill the background, so manual
            corrections survive. The default is True.
        """

        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        if not model:
            _segmentation_failed(
                "No segmentation model is available. Download or train one first."
            )
            return

        t = int(viewer.dims.current_step[0])
        experiment = extract_experiment_from_position(position)

        try:
            channel_names, _ = extract_experiment_channels(experiment)
            channel_names = list(channel_names)
            spatial_calibration = get_spatial_calibration(experiment)
        except Exception as e:
            _segmentation_failed(f"Could not read the experiment configuration: {e}")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        viewer.status = f"Segmenting frame {t} with '{model}'…"
        try:
            prepared = prepared_models.get(model)
            if prepared is None:
                from celldetective.segmentation import prepare_segmentation_model

                # The GPU is left to napari's renderer: a single frame is quick
                # on CPU, and a TensorFlow context would compete for the VRAM
                # the viewer is already using.
                prepared = prepare_segmentation_model(
                    model,
                    channels=channel_names,
                    spatial_calibration=spatial_calibration,
                    use_gpu=False,
                )
                if prepared is None:
                    _segmentation_failed(
                        f"Model '{model}' could not be loaded. See the log for details."
                    )
                    return
                prepared_models[model] = prepared

            from celldetective.segmentation import segment_frame

            new_labels = segment_frame(np.asarray(stack[t]), prepared)
        except Exception as e:
            logger.exception("Single-frame segmentation failed.")
            _segmentation_failed(f"Segmentation failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        layer = viewer.layers["segmentation"]
        try:
            current = layer.data[t]
            if replace_existing:
                current[...] = new_labels
            else:
                # Offset the new labels past the ones already drawn so the two
                # sets cannot collide, then keep whatever was already there.
                offset = int(current.max())
                incoming = np.where(new_labels > 0, new_labels + offset, 0)
                current[...] = np.where(current > 0, current, incoming)
        except Exception as e:
            _segmentation_failed(f"Could not write the labels into the viewer: {e}")
            return
        layer.refresh()

        n_objects = int(np.max(layer.data[t]))
        message = f"Frame {t}: {n_objects} objects after segmenting with '{model}'."
        viewer.status = message
        logger.info(message)

    if contrast_limits is None:
        contrast_limits = _get_contrast_limits(stack)

    output_folder = position + f"labels_{population}{os.sep}"
    logger.info(f"Shape of the loaded image stack: {stack.shape}...")
    if progress_callback:
        try:
            progress_callback("Initializing napari viewer…")
        except Exception:
            pass

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
    layout.addWidget(segment_frame_widget.native)
    layout.addWidget(correction_options.native)
    layout.addWidget(save_widget.native)
    layout.addWidget(export_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

    save_widget.native.setStyleSheet(Styles().button_style_sheet)
    try:
        segment_frame_widget.call_button.native.setStyleSheet(
            Styles().button_style_sheet
        )
    except Exception as e:
        logger.debug(f"Could not style the segmentation button: {e}")
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

    if not block:
        # Non-blocking mode (launched from the GUI loader thread): show() returns
        # immediately and we must NOT touch the viewer afterwards. The napari main
        # window has WA_DeleteOnClose, so closing it tears down the Qt viewer and
        # releases the label array / memmap-backed stack through normal garbage
        # collection. We deliberately do NOT pop layers from a `destroyed` handler:
        # that runs during C++ teardown of the window and segfaults.
        return

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

    @magicgui(
        layout="vertical",
        call_button=False,
        split_merged_labels={
            "widget_type": "CheckBox",
            "text": "Split merged labels",
        },
        remove_small_objects={
            "widget_type": "CheckBox",
            "text": "Remove small objects",
        },
        fill_holes={"widget_type": "CheckBox", "text": "Fill holes in masks"},
        min_area={
            "label": "Min object area (px²)",
            "min": 0,
            "max": 1_000_000,
        },
    )
    def correction_options(
        split_merged_labels: bool = True,
        remove_small_objects: bool = True,
        fill_holes: bool = False,
        min_area: int = 9,
    ):
        """Auto-fixes applied to the masks when saving."""

    def _correction_kwargs() -> Dict[str, Any]:
        """Build ``auto_correct_masks`` kwargs from the option widgets."""
        return {
            "correct_anomalies": correction_options.split_merged_labels.value,
            "fill_labels": correction_options.fill_holes.value,
            "min_area": (
                correction_options.min_area.value
                if correction_options.remove_small_objects.value
                else 0
            ),
        }

    def export_labels():
        """Export corrected labels to file."""
        labels_layer = viewer.layers["segmentation"].data
        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im, **_correction_kwargs())
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

    button_container = QWidget()
    layout = QVBoxLayout(button_container)
    layout.setSpacing(10)
    layout.addWidget(correction_options.native)
    layout.addWidget(save_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

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
