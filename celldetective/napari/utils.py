import gc
import json
import os
from pathlib import Path, PurePath

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
from celldetective.utils.image_loaders import locate_labels, locate_stack_and_labels
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
    position,
    prefix="Aligned",
    population="target",
    relabel=True,
    flush_memory=True,
    threads=1,
    progress_callback=None,
    prepare_only=False,
):
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


def tracks_to_napari(df, exclude_nans=False):

    data, properties, graph = tracks_to_btrack(df, exclude_nans=exclude_nans)
    vertices = data[:, [1, -2, -1]]
    if data.shape[1] == 4:
        tracks = data
    else:
        tracks = data[:, [0, 1, 3, 4]]
    return vertices, tracks, properties, graph


def view_tracks_in_napari(
    position,
    population,
    stack=None,
    labels=None,
    relabel=True,
    flush_memory=True,
    threads=1,
    progress_callback=None,
    prepare_only=False,
):
    """
    Updated
    """

    print(f"DEBUG: view_tracks_in_napari called with pos={position}, pop={population}")
    df, df_path = get_position_table(position, population=population, return_path=True)
    print(f"DEBUG: get_position_table returned df={df is not None}")

    if progress_callback:
        progress_callback(50)

    if df is None:
        print("Please compute trajectories first... Abort...")
        return None
    shared_data = {
        "df": df,
        "path": df_path,
        "position": position,
        "population": population,
        "selected_frame": None,
    }

    if (labels is not None) * relabel:
        print("Replacing the cell mask labels with the track ID...")

        def wrapped_callback(p):
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
    stack,
    labels,
    vertices,
    tracks,
    properties,
    graph,
    shared_data,
    contrast_limits,
    flush_memory=True,
    block=True,
    progress_callback=None,
):

    viewer = napari.Viewer()

    if stack is not None:
        viewer.add_image(
            stack,
            channel_axis=-1,
            colormap=["gray"] * stack.shape[-1],
            contrast_limits=contrast_limits,
        )

    if labels is not None:
        labels_layer = viewer.add_labels(
            labels.astype(int), name="segmentation", opacity=0.4
        )
    viewer.add_points(vertices, size=4, name="points", opacity=0.3)
    viewer.add_tracks(tracks, properties=properties, graph=graph, name="tracks")

    def lock_controls(layer, widgets=(), locked=True):
        qctrl = viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widgets:
            try:
                getattr(qctrl, wdg).setEnabled(not locked)
            except:
                pass

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

    def export_modifications():

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
        print(f"{instruction_file=}")
        if os.path.exists(instruction_file):
            print("Tracking configuration file found...")
            with open(instruction_file, "r") as f:
                instructions = json.load(f)
                if "post_processing_options" in instructions:
                    post_processing_options = instructions["post_processing_options"]
                    print(
                        f"Applying the following track postprocessing: {post_processing_options}..."
                    )
                    df = clean_trajectories(df.copy(), **post_processing_options)
        unnamed_cols = [c for c in list(df.columns) if c.startswith("Unnamed")]
        df = df.drop(unnamed_cols, axis=1)
        print(f"{list(df.columns)=}")
        df.to_csv(shared_data["path"], index=False)
        print("Done...")

    @magicgui(call_button="Export the modified\ntracks...")
    def export_table_widget():
        return export_modifications()

    export_table_widget.native.setStyleSheet(Styles().button_style_sheet)

    def label_changed(event):

        value = viewer.layers["segmentation"].selected_label
        if value != 0:
            selected_frame = viewer.dims.current_step[0]
            shared_data["selected_frame"] = selected_frame

    viewer.layers["segmentation"].events.selected_label.connect(label_changed)

    viewer.window.add_dock_widget(export_table_widget, area="right")

    @labels_layer.mouse_double_click_callbacks.append
    def on_second_click_of_double_click(layer, event):

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
        except:
            print("Invalid mask value...")
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

        # temporary fix for slight napari memory leak
        for i in range(10000):
            try:
                viewer.layers.pop()
            except:
                pass

        del viewer
        del stack
        del labels
        gc.collect()


def load_napari_data(
    position, prefix="Aligned", population="target", return_stack=True
):
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
    position, prefix="Aligned", population="target", flush_memory=False
):
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

    Notes
    -----
    This function loads the segmentation labels and stack corresponding to the specified position and population.
    It then creates a napari viewer and adds the stack and labels as layers for visualization.

    Examples
    --------
    >>> control_segmentation_napari(position, prefix='Aligned', population="target")
    # Control the visualization of segmentation labels using the napari viewer.

    """

    def export_labels():
        labels_layer = viewer.layers["segmentation"].data
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im)
            except Exception as e:
                print(e)

            save_tiff_imagej_compatible(
                output_folder + f"{str(t).zfill(4)}.tif", im.astype(np.int16), axes="YX"
            )
        print("The labels have been successfully rewritten.")

    def export_annotation():

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
                print(f"{e=}")

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

        print("Exporting!")
        t = viewer.dims.current_step[0]
        labels_layer = viewer.layers["segmentation"].data[t]  # at current time

        try:
            labels_layer = auto_correct_masks(labels_layer)
        except Exception as e:
            print(e)

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
            print(f"Found {nbr_squares} ROIs...")
            if nbr_squares > 0:
                # deactivate field of view mode
                fov_export = False

            for k, sq in enumerate(squares):
                print(f"ROI: {sq}")
                pad_to_256 = False

                xmin = int(sq[0, 1])
                xmax = int(sq[2, 1])
                if xmax < xmin:
                    xmax, xmin = xmin, xmax
                ymin = int(sq[0, 2])
                ymax = int(sq[1, 2])
                if ymax < ymin:
                    ymax, ymin = ymin, ymax
                print(f"{xmin=};{xmax=};{ymin=};{ymax=}")
                frame = viewer.layers["Image"].data[t][xmin:xmax, ymin:ymax]
                if frame.shape[1] < 256 or frame.shape[0] < 256:
                    pad_to_256 = True
                    print(
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
                    except:
                        pass
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
                except:
                    pass
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

        print("Done.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        return export_labels()

    @magicgui(call_button="Export the annotation\nof the current frame")
    def export_widget():
        return export_annotation()

    stack, labels = locate_stack_and_labels(
        position, prefix=prefix, population=population
    )
    contrast_limits = _get_contrast_limits(stack)

    output_folder = position + f"labels_{population}{os.sep}"
    logger.info(f"Shape of the loaded image stack: {stack.shape}...")

    viewer = napari.Viewer()
    try:
        viewer.window._qt_window.setWindowIcon(Styles().celldetective_icon)
    except Exception as e:
        pass
    viewer.add_image(
        stack,
        channel_axis=-1,
        colormap=["gray"] * stack.shape[-1],
        contrast_limits=contrast_limits,
    )
    viewer.add_labels(labels.astype(int), name="segmentation", opacity=0.4)

    button_container = QWidget()
    layout = QVBoxLayout(button_container)
    layout.setSpacing(10)
    layout.addWidget(save_widget.native)
    layout.addWidget(export_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

    save_widget.native.setStyleSheet(Styles().button_style_sheet)
    export_widget.native.setStyleSheet(Styles().button_style_sheet)

    def lock_controls(layer, widgets=(), locked=True):
        qctrl = viewer.window.qt_viewer.controls.widgets[layer]
        for wdg in widgets:
            try:
                getattr(qctrl, wdg).setEnabled(not locked)
            except:
                pass

    label_widget_list = ["polygon_button", "transform_button"]
    lock_controls(viewer.layers["segmentation"], label_widget_list)

    viewer.show(block=True)

    if flush_memory:
        # temporary fix for slight napari memory leak
        for i in range(10000):
            try:
                viewer.layers.pop()
            except:
                pass

        del viewer
        del stack
        del labels
        gc.collect()

    logger.info("napari viewer was successfully closed...")


def correct_annotation(filename):
    """
    New function to reannotate an annotation image in post, using napari and save update inplace.
    """

    def export_labels():
        labels_layer = viewer.layers["segmentation"].data
        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im)
            except Exception as e:
                print(e)

            save_tiff_imagej_compatible(existing_lbl, im.astype(np.int16), axes="YX")
        print("The labels have been successfully rewritten.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        return export_labels()

    if filename.endswith("_labelled.tif"):
        filename = filename.replace("_labelled.tif", ".tif")
    if filename.endswith(".json"):
        filename = filename.replace(".json", ".tif")
    assert os.path.exists(filename), f"Image {filename} does not seem to exist..."

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


def _view_on_napari(tracks=None, stack=None, labels=None):
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
    position,
    calibration=1,
    prefix="Aligned",
    population="target",
    column_labels={
        "track": "TRACK_ID",
        "frame": "FRAME",
        "y": "POSITION_Y",
        "x": "POSITION_X",
        "label": "class_id",
    },
):
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
