from PyQt5.QtCore import QSize
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from natsort import natsorted
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
from glob import glob
from tifffile import imread, TiffFile, memmap
import dask.array as da
import dask
import numpy as np
import os
import pandas as pd
import napari
import json

import gc
from tqdm import tqdm
import threading
import concurrent.futures

from csbdeep.utils import normalize_mi_ma
from csbdeep.io import save_tiff_imagej_compatible

import imageio.v2 as imageio
from skimage.measure import regionprops_table, label

from btrack.datasets import cell_config
from magicgui import magicgui
from pathlib import Path, PurePath
from shutil import copyfile, rmtree

from celldetective.utils import (
    _rearrange_multichannel_frame,
    _fix_no_contrast,
    zoom_multiframes,
    config_section_to_dict,
    extract_experiment_channels,
    _extract_labels_from_config,
    get_zenodo_files,
    download_zenodo_file,
)
from celldetective.utils import interpolate_nan_multichannel, get_config

from stardist import fill_label_holes
from skimage.transform import resize
import re

from typing import List, Tuple, Union
import numbers
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


from celldetective.utils.experiments import (
    extract_experiment_from_well,
    extract_well_from_position,
    extract_experiment_from_position,
    collect_experiment_metadata,
    get_experiment_wells,
    get_spatial_calibration,
    get_temporal_calibration,
    get_experiment_metadata,
    get_experiment_labels,
    get_experiment_concentrations,
    get_experiment_cell_types,
    get_experiment_antibodies,
    get_experiment_pharmaceutical_agents,
    get_experiment_populations,
    interpret_wells_and_positions,
    extract_well_name_and_number,
    extract_position_name,
    locate_stack,
    locate_labels,
    fix_missing_labels,
    locate_stack_and_labels,
    load_tracking_data,
    get_position_table,
    relabel_segmentation_lazy,
    _get_contrast_limits,
    view_tracks_in_napari,
    auto_correct_masks,
    _view_on_napari,
    control_tracking_table,
    auto_load_number_of_frames,
)


def get_position_pickle(pos, population, return_path=False):
    """
    Retrieves the data table for a specified population at a given position, optionally returning the table's file path.

    This function locates and loads a CSV data table associated with a specific population (e.g., 'targets', 'cells')
    from a specified position directory. The position directory should contain an 'output/tables' subdirectory where
    the CSV file named 'trajectories_{population}.csv' is expected to be found. If the file exists, it is loaded into
    a pandas DataFrame; otherwise, None is returned.

    Parameters
    ----------
    pos : str
            The path to the position directory from which to load the data table.
    population : str
            The name of the population for which the data table is to be retrieved. This name is used to construct the
            file name of the CSV file to be loaded.
    return_path : bool, optional
            If True, returns a tuple containing the loaded data table (or None) and the path to the CSV file. If False,
            only the loaded data table (or None) is returned (default is False).

    Returns
    -------
    pandas.DataFrame or None, or (pandas.DataFrame or None, str)
            If return_path is False, returns the loaded data table as a pandas DataFrame, or None if the table file does
            not exist. If return_path is True, returns a tuple where the first element is the data table (or None) and the
            second element is the path to the CSV file.

    Examples
    --------
    >>> df_pos = get_position_table('/path/to/position', 'targets')
    # This will load the 'trajectories_targets.csv' table from the specified position directory into a pandas DataFrame.

    >>> df_pos, table_path = get_position_table('/path/to/position', 'targets', return_path=True)
    # This will load the 'trajectories_targets.csv' table and also return the path to the CSV file.

    """

    if not pos.endswith(os.sep):
        table = os.sep.join([pos, "output", "tables", f"trajectories_{population}.pkl"])
    else:
        table = pos + os.sep.join(
            ["output", "tables", f"trajectories_{population}.pkl"]
        )

    if os.path.exists(table):
        df_pos = np.load(table, allow_pickle=True)
    else:
        df_pos = None

    if return_path:
        return df_pos, table
    else:
        return df_pos


def get_position_movie_path(pos, prefix=""):
    """
    Get the path of the movie file for a given position.

    This function constructs the path to a movie file within a given position directory.
    It searches for TIFF files that match the specified prefix. If multiple matching files
    are found, the first one is returned.

    Parameters
    ----------
    pos : str
            The directory path for the position.
    prefix : str, optional
            The prefix to filter movie files. Defaults to an empty string.

    Returns
    -------
    stack_path : str or None
            The path to the first matching movie file, or None if no matching file is found.

    Examples
    --------
    >>> pos_path = "path/to/position1"
    >>> get_position_movie_path(pos_path, prefix='experiment_')
    'path/to/position1/movie/experiment_001.tif'

    >>> pos_path = "another/path/positionA"
    >>> get_position_movie_path(pos_path)
    'another/path/positionA/movie/001.tif'

    >>> pos_path = "nonexistent/path"
    >>> get_position_movie_path(pos_path)
    None

    """

    if not pos.endswith(os.sep):
        pos += os.sep
    movies = glob(pos + os.sep.join(["movie", prefix + "*.tif"]))
    if len(movies) > 0:
        stack_path = movies[0]
    else:
        stack_path = None

    return stack_path


def load_experiment_tables(
    experiment,
    population="targets",
    well_option="*",
    position_option="*",
    return_pos_info=False,
    load_pickle=False,
):
    """
    Load tabular data for an experiment, optionally including position-level information.

    This function retrieves and processes tables associated with positions in an experiment.
    It supports filtering by wells and positions, and can load either CSV data or pickle files.

    Parameters
    ----------
    experiment : str
            Path to the experiment folder to load data for.
    population : str, optional
            The population to extract from the position tables (`'targets'` or `'effectors'`). Default is `'targets'`.
    well_option : str or list, optional
            Specifies which wells to include. Default is `'*'`, meaning all wells.
    position_option : str or list, optional
            Specifies which positions to include within selected wells. Default is `'*'`, meaning all positions.
    return_pos_info : bool, optional
            If `True`, also returns a DataFrame containing position-level metadata. Default is `False`.
    load_pickle : bool, optional
            If `True`, loads pre-processed pickle files for the positions instead of raw data. Default is `False`.

    Returns
    -------
    df : pandas.DataFrame or None
            A DataFrame containing aggregated data for the specified wells and positions, or `None` if no data is found.
            The DataFrame includes metadata such as well and position identifiers, concentrations, antibodies, and other
            experimental parameters.
    df_pos_info : pandas.DataFrame, optional
            A DataFrame with metadata for each position, including file paths and experimental details. Returned only
            if `return_pos_info=True`.

    Notes
    -----
    - The function assumes the experiment's configuration includes details about movie prefixes, concentrations,
      cell types, antibodies, and pharmaceutical agents.
    - Wells and positions can be filtered using `well_option` and `position_option`, respectively. If filtering
      fails or is invalid, those specific wells/positions are skipped.
    - Position-level metadata is assembled into `df_pos_info` and includes paths to data and movies.

    Examples
    --------
    Load all data for an experiment:

    >>> df = load_experiment_tables("path/to/experiment1")

    Load data for specific wells and positions, including position metadata:

    >>> df, df_pos_info = load_experiment_tables(
    ...     "experiment_01", well_option=["A1", "B1"], position_option=[0, 1], return_pos_info=True
    ... )

    Use pickle files for faster loading:

    >>> df = load_experiment_tables("experiment_01", load_pickle=True)

    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)

    movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    labels = get_experiment_labels(experiment)
    metadata = get_experiment_metadata(experiment)  # None or dict of metadata
    well_labels = _extract_labels_from_config(config, len(wells))

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )

    df = []
    df_pos_info = []
    real_well_index = 0

    for k, well_path in enumerate(tqdm(wells[well_indices])):

        any_table = False  # assume no table

        well_name, well_number = extract_well_name_and_number(well_path)
        widx = well_indices[k]
        well_alias = well_labels[widx]

        positions = get_positions_in_well(well_path)
        if position_indices is not None:
            try:
                positions = positions[position_indices]
            except Exception as e:
                logger.error(e)
                continue

        real_pos_index = 0
        for pidx, pos_path in enumerate(positions):

            pos_name = extract_position_name(pos_path)

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)

            if not load_pickle:
                df_pos, table = get_position_table(
                    pos_path, population=population, return_path=True
                )
            else:
                df_pos, table = get_position_pickle(
                    pos_path, population=population, return_path=True
                )

            if df_pos is not None:

                df_pos["position"] = pos_path
                df_pos["well"] = well_path
                df_pos["well_index"] = well_number
                df_pos["well_name"] = well_name
                df_pos["pos_name"] = pos_name

                for k in list(labels.keys()):
                    values = labels[k]
                    try:
                        df_pos[k] = values[widx]
                    except Exception as e:
                        logger.error(f"{e=}")

                if metadata is not None:
                    keys = list(metadata.keys())
                    for key in keys:
                        df_pos[key] = metadata[key]

                df.append(df_pos)
                any_table = True

                pos_dict = {
                    "pos_path": pos_path,
                    "pos_index": real_pos_index,
                    "pos_name": pos_name,
                    "table_path": table,
                    "stack_path": stack_path,
                    "well_path": well_path,
                    "well_index": real_well_index,
                    "well_name": well_name,
                    "well_number": well_number,
                    "well_alias": well_alias,
                }

                df_pos_info.append(pos_dict)

                real_pos_index += 1

        if any_table:
            real_well_index += 1

    df_pos_info = pd.DataFrame(df_pos_info)
    if len(df) > 0:
        df = pd.concat(df)
        df = df.reset_index(drop=True)
    else:
        df = None

    if return_pos_info:
        return df, df_pos_info
    else:
        return df


def parse_isotropic_radii(string):
    """
    Parse a string representing isotropic radii into a structured list.

    This function extracts integer values and ranges (denoted by square brackets)
    from a string input and returns them as a list. Single values are stored as integers,
    while ranges are represented as lists of two integers.

    Parameters
    ----------
    string : str
            The input string containing radii and ranges, separated by commas or spaces.
            Ranges should be enclosed in square brackets, e.g., `[1 2]`.

    Returns
    -------
    list
            A list of parsed radii where:
            - Single integers are included as `int`.
            - Ranges are included as two-element lists `[start, end]`.

    Examples
    --------
    Parse a string with single radii and ranges:

    >>> parse_isotropic_radii("1, [2 3], 4")
    [1, [2, 3], 4]

    Handle inputs with mixed delimiters:

    >>> parse_isotropic_radii("5 [6 7], 8")
    [5, [6, 7], 8]

    Notes
    -----
    - The function splits the input string by commas or spaces.
    - It identifies ranges using square brackets and assumes that ranges are always
      two consecutive values.
    - Non-integer sections of the string are ignored.

    """

    sections = re.split(r"[ ,]", string)
    radii = []
    for k, s in enumerate(sections):
        if s.isdigit():
            radii.append(int(s))
        if "[" in s:
            ring = [int(s.replace("[", "")), int(sections[k + 1].replace("]", ""))]
            radii.append(ring)
        else:
            pass
    return radii


def get_tracking_configs_list(return_path=False):
    """

    Retrieve a list of available tracking configurations.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available tracking configurations.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available tracking configurations by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_tracking_configs_list()
    # Retrieve a list of available tracking configurations.

    >>> models, path = get_tracking_configs_list(return_path=True)
    # Retrieve a list of available tracking configurations.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "models",
            "tracking_configs",
            os.sep,
        ]
    )
    available_models = glob(modelpath + "*.json")
    available_models = [m.replace("\\", "/").split("/")[-1] for m in available_models]
    available_models = [m.replace("\\", "/").split(".")[0] for m in available_models]

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def interpret_tracking_configuration(config):
    """
    Interpret and resolve the path for a tracking configuration file.

    This function determines the appropriate configuration file path based on the input.
    If the input is a string representing an existing path or a known configuration name,
    it resolves to the correct file path. If the input is invalid or `None`, a default
    configuration is returned.

    Parameters
    ----------
    config : str or None
            The input configuration, which can be:
            - A string representing the full path to a configuration file.
            - A short name of a configuration file without the `.json` extension.
            - `None` to use a default configuration.

    Returns
    -------
    str
            The resolved path to the configuration file.

    Notes
    -----
    - If `config` is a string and the specified path exists, it is returned as-is.
    - If `config` is a name, the function searches in the `tracking_configs` directory
      within the `celldetective` models folder.
    - If the file or name is not found, or if `config` is `None`, the function falls
      back to a default configuration using `cell_config()`.

    Examples
    --------
    Resolve a full path:

    >>> interpret_tracking_configuration("/path/to/config.json")
    '/path/to/config.json'

    Resolve a named configuration:

    >>> interpret_tracking_configuration("default_tracking")
    '/path/to/celldetective/models/tracking_configs/default_tracking.json'

    Handle `None` to return the default configuration:

    >>> interpret_tracking_configuration(None)
    '/path/to/default/config.json'

    """

    if isinstance(config, str):
        if os.path.exists(config):
            return config
        else:
            modelpath = os.sep.join(
                [
                    os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
                    "celldetective",
                    "models",
                    "tracking_configs",
                    os.sep,
                ]
            )
            if os.path.exists(modelpath + config + ".json"):
                return modelpath + config + ".json"
            else:
                config = cell_config()
    elif config is None:
        config = cell_config()

    return config


def get_signal_models_list(return_path=False):
    """

    Retrieve a list of available signal detection models.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available signal detection models.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available signal detection models by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_signal_models_list()
    # Retrieve a list of available signal detection models.

    >>> models, path = get_signal_models_list(return_path=True)
    # Retrieve a list of available signal detection models and the path to the models.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "models",
            "signal_detection",
            os.sep,
        ]
    )
    repository_models = get_zenodo_files(
        cat=os.sep.join(["models", "signal_detection"])
    )

    available_models = glob(modelpath + f"*{os.sep}")
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]
    for rm in repository_models:
        if rm not in available_models:
            available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def get_pair_signal_models_list(return_path=False):
    """

    Retrieve a list of available signal detection models.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available signal detection models.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available signal detection models by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_signal_models_list()
    # Retrieve a list of available signal detection models.

    >>> models, path = get_signal_models_list(return_path=True)
    # Retrieve a list of available signal detection models and the path to the models.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "models",
            "pair_signal_detection",
            os.sep,
        ]
    )
    # repository_models = get_zenodo_files(cat=os.sep.join(["models", "pair_signal_detection"]))

    available_models = glob(modelpath + f"*{os.sep}")
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]
    # for rm in repository_models:
    #   if rm not in available_models:
    #       available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def locate_signal_model(name, path=None, pairs=False):
    """
    Locate a signal detection model by name, either locally or from Zenodo.

    This function searches for a signal detection model with the specified name in the local
    `celldetective` directory. If the model is not found locally, it attempts to download
    the model from Zenodo.

    Parameters
    ----------
    name : str
            The name of the signal detection model to locate.
    path : str, optional
            An additional directory path to search for the model. If provided, this directory
            is also scanned for matching models. Default is `None`.
    pairs : bool, optional
            If `True`, searches for paired signal detection models in the `pair_signal_detection`
            subdirectory. If `False`, searches in the `signal_detection` subdirectory. Default is `False`.

    Returns
    -------
    str or None
            The full path to the located model directory if found, or `None` if the model is not available
            locally or on Zenodo.

    Notes
    -----
    - The function first searches in the `celldetective/models/signal_detection` or
      `celldetective/models/pair_signal_detection` directory based on the `pairs` argument.
    - If a `path` is specified, it is searched in addition to the default directories.
    - If the model is not found locally, the function queries Zenodo for the model. If available,
      the model is downloaded to the appropriate `celldetective` subdirectory.

    Examples
    --------
    Search for a signal detection model locally:

    >>> locate_signal_model("example_model")
    'path/to/celldetective/models/signal_detection/example_model/'

    Search for a paired signal detection model:

    >>> locate_signal_model("paired_model", pairs=True)
    'path/to/celldetective/models/pair_signal_detection/paired_model/'

    Include an additional search path:

    >>> locate_signal_model("custom_model", path="/additional/models/")
    '/additional/models/custom_model/'

    Handle a model available only on Zenodo:

    >>> locate_signal_model("remote_model")
    'path/to/celldetective/models/signal_detection/remote_model/'

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
    )
    modelpath = os.sep.join([main_dir, "models", "signal_detection", os.sep])
    if pairs:
        modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
    logger.info(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")
    if path is not None:
        if not path.endswith(os.sep):
            path += os.sep
        models += glob(path + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def locate_pair_signal_model(name, path=None):
    """
    Locate a pair signal detection model by name.

    This function searches for a pair signal detection model in the default
    `celldetective` directory and optionally in an additional user-specified path.

    Parameters
    ----------
    name : str
            The name of the pair signal detection model to locate.
    path : str, optional
            An additional directory path to search for the model. If provided, this directory
            is also scanned for matching models. Default is `None`.

    Returns
    -------
    str or None
            The full path to the located model directory if found, or `None` if no matching
            model is located.

    Notes
    -----
    - The function first searches in the default `celldetective/models/pair_signal_detection`
      directory.
    - If a `path` is specified, it is searched in addition to the default directory.
    - The function prints the search path and model name during execution.

    Examples
    --------
    Locate a model in the default directory:

    >>> locate_pair_signal_model("example_model")
    'path/to/celldetective/models/pair_signal_detection/example_model/'

    Include an additional search directory:

    >>> locate_pair_signal_model("custom_model", path="/additional/models/")
    '/additional/models/custom_model/'

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
    )
    modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
    logger.info(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")
    if path is not None:
        if not path.endswith(os.sep):
            path += os.sep
        models += glob(path + f"*{os.sep}")




def control_tracks(
    position,
    prefix="Aligned",
    population="target",
    relabel=True,
    flush_memory=True,
    threads=1,
    lazy=False,
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
    stack, labels = locate_stack_and_labels(
        position, prefix=prefix, population=population, lazy=lazy
    )

    view_tracks_in_napari(
        position,
        population,
        labels=labels,
        stack=stack,
        relabel=relabel,
        flush_memory=flush_memory,
        threads=threads,
        lazy=lazy,
    )


def tracks_to_btrack(df, exclude_nans=False):
    """
    Converts a dataframe of tracked objects into the bTrack output format.
    The function prepares tracking data, properties, and an empty graph structure for further processing.

    Parameters
    ----------
    df : pandas.DataFrame
            A dataframe containing tracking information. The dataframe must have columns for `TRACK_ID`,
            `FRAME`, `POSITION_Y`, `POSITION_X`, and `class_id` (among others).

    exclude_nans : bool, optional, default=False
            If True, rows with NaN values in the `class_id` column will be excluded from the dataset.
            If False, the dataframe will retain all rows, including those with NaN in `class_id`.

    Returns
    -------
    data : numpy.ndarray
            A 2D numpy array containing the tracking data with columns `[TRACK_ID, FRAME, z, POSITION_Y, POSITION_X]`.
            The `z` column is set to zero for all rows.

    properties : dict
            A dictionary where keys are property names (e.g., 'FRAME', 'state', 'generation', etc.) and values are numpy arrays
            containing the corresponding values from the dataframe.

    graph : dict
            An empty dictionary intended to store graph-related information for the tracking data. It can be extended
            later to represent relationships between different tracking objects.

    Notes
    -----
    - The function assumes that the dataframe contains specific columns: `TRACK_ID`, `FRAME`, `POSITION_Y`, `POSITION_X`,
      and `class_id`. These columns are used to construct the tracking data and properties.
    - The `z` coordinate is set to 0 for all tracks since the function does not process 3D data.
    - This function is useful for transforming tracking data into a format that can be used by tracking graph algorithms.

    Example
    -------
    >>> data, properties, graph = tracks_to_btrack(df, exclude_nans=True)

    """

    graph = {}
    if exclude_nans:
        df.dropna(subset="class_id", inplace=True)
        df.dropna(subset="TRACK_ID", inplace=True)

    df["z"] = 0.0
    data = df[["TRACK_ID", "FRAME", "z", "POSITION_Y", "POSITION_X"]].to_numpy()

    df["dummy"] = False
    prop_cols = ["FRAME", "state", "generation", "root", "parent", "dummy", "class_id"]
    properties = {}
    for col in prop_cols:
        properties.update({col: df[col].to_numpy()})

    return data, properties, graph


def tracks_to_napari(df, exclude_nans=False):

    data, properties, graph = tracks_to_btrack(df, exclude_nans=exclude_nans)
    vertices = data[:, [1, -2, -1]]
    if data.shape[1] == 4:
        tracks = data
    else:
        tracks = data[:, [0, 1, 3, 4]]
    return vertices, tracks, properties, graph


def control_segmentation_napari(
    position, prefix="Aligned", population="target", flush_memory=False, lazy=False
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
                logger.error(e)

            save_tiff_imagej_compatible(
                output_folder + f"{str(t).zfill(4)}.tif", im.astype(np.int16), axes="YX"
            )
        logger.info("The labels have been successfully rewritten.")

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
                logger.error(f"{e=}")

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

        logger.info("Exporting!")
        t = viewer.dims.current_step[0]
        labels_layer = viewer.layers["segmentation"].data[t]  # at current time

        try:
            labels_layer = auto_correct_masks(labels_layer)
        except Exception as e:
            logger.error(e)

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
                logger.info(f"ROI: {sq}")
                pad_to_256 = False

                xmin = int(sq[0, 1])
                xmax = int(sq[2, 1])
                if xmax < xmin:
                    xmax, xmin = xmin, xmax
                ymin = int(sq[0, 2])
                ymax = int(sq[1, 2])
                if ymax < ymin:
                    ymax, ymin = ymin, ymax
                logger.info(f"{xmin=};{xmax=};{ymin=};{ymax=}")
                frame = viewer.layers["Image"].data[t][xmin:xmax, ymin:ymax]
                if frame.shape[1] < 256 or frame.shape[0] < 256:
                    pad_to_256 = True
                    logger.info(
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

        logger.info("Done.")

    @magicgui(call_button="Save the modified labels")
    def save_widget():
        return export_labels()

    @magicgui(call_button="Export the annotation\nof the current frame")
    def export_widget():
        return export_annotation()

    from celldetective.gui import Styles

    stack, labels = locate_stack_and_labels(
        position, prefix=prefix, population=population, lazy=lazy
    )
    output_folder = position + f"labels_{population}{os.sep}"
    logger.info(f"Shape of the loaded image stack: {stack.shape}...")

    viewer = napari.Viewer()
    contrast_limits = _get_contrast_limits(stack)
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

    from celldetective.gui import Styles

    def export_labels():
        labels_layer = viewer.layers["segmentation"].data
        for t, im in enumerate(tqdm(labels_layer)):

            try:
                im = auto_correct_masks(im)
            except Exception as e:
                logger.error(e)

            save_tiff_imagej_compatible(existing_lbl, im.astype(np.int16), axes="YX")
        logger.info("The labels have been successfully rewritten.")

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

    viewer = napari.Viewer()
    contrast_limits = _get_contrast_limits(stack)
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
    layout.addWidget(save_widget.native)
    viewer.window.add_dock_widget(button_container, area="right")

    save_widget.native.setStyleSheet(Styles().button_style_sheet)

    viewer.show(block=True)

    # temporary fix for slight napari memory leak
    for i in range(100):
        try:
            viewer.layers.pop()
        except:
            pass
    del viewer
    del stack
    del labels
    gc.collect()


def get_segmentation_models_list(mode="targets", return_path=False):

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "models",
            f"segmentation_{mode}",
            os.sep,
        ]
    )
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
        repository_models = []
    else:
        repository_models = get_zenodo_files(
            cat=os.sep.join(["models", f"segmentation_{mode}"])
        )

    available_models = natsorted(glob(modelpath + "*/"))
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]

    # Auto model cleanup
    to_remove = []
    for model in available_models:
        path = modelpath + model
        files = glob(path + os.sep + "*")
        if path + os.sep + "config_input.json" not in files:
            rmtree(path)
            to_remove.append(model)
    for m in to_remove:
        available_models.remove(m)

    for rm in repository_models:
        if rm not in available_models:
            available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def locate_segmentation_model(name, download=True):
    """
    Locates a specified segmentation model within the local 'celldetective' directory or
    downloads it from Zenodo if not found locally.

    This function attempts to find a segmentation model by name within a predefined directory
    structure starting from the 'celldetective/models/segmentation*' path. If the model is not
    found locally, it then tries to locate and download the model from Zenodo, placing it into
    the appropriate category directory within 'celldetective'. The function prints the search
    directory path and returns the path to the found or downloaded model.

    Parameters
    ----------
    name : str
            The name of the segmentation model to locate.

    Returns
    -------
    str or None
            The full path to the located or downloaded segmentation model directory, or None if the
            model could not be found or downloaded.

    Raises
    ------
    FileNotFoundError
            If the model cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
    )
    modelpath = os.sep.join([main_dir, "models", "segmentation*"]) + os.sep
    # print(f'Looking for {name} in {modelpath}')
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    if download:
        # else no match, try zenodo
        files, categories = get_zenodo_files()
        if name in files:
            index = files.index(name)
            cat = categories[index]
            download_zenodo_file(name, os.sep.join([main_dir, cat]))
            match = os.sep.join([main_dir, cat, name]) + os.sep

    return match


def get_segmentation_datasets_list(return_path=False):
    """
    Retrieves a list of available segmentation datasets from both the local 'celldetective/datasets/segmentation_annotations'
    directory and a Zenodo repository, optionally returning the path to the local datasets directory.

    This function compiles a list of available segmentation datasets by first identifying datasets stored locally
    within a specified path related to the script's directory. It then extends this list with datasets available
    in a Zenodo repository, ensuring no duplicates are added. The function can return just the list of dataset
    names or, if specified, also return the path to the local datasets directory.

    Parameters
    ----------
    return_path : bool, optional
            If True, the function returns a tuple containing the list of available dataset names and the path to the
            local datasets directory. If False, only the list of dataset names is returned (default is False).

    Returns
    -------
    list or (list, str)
            If return_path is False, returns a list of strings, each string being the name of an available dataset.
            If return_path is True, returns a tuple where the first element is this list and the second element is a
            string representing the path to the local datasets directory.

    """

    datasets_path = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "datasets",
            "segmentation_annotations",
            os.sep,
        ]
    )
    repository_datasets = get_zenodo_files(
        cat=os.sep.join(["datasets", "segmentation_annotations"])
    )

    available_datasets = natsorted(glob(datasets_path + "*/"))
    available_datasets = [
        m.replace("\\", "/").split("/")[-2] for m in available_datasets
    ]
    for rm in repository_datasets:
        if rm not in available_datasets:
            available_datasets.append(rm)

    if not return_path:
        return available_datasets
    else:
        return available_datasets, datasets_path


def locate_segmentation_dataset(name):
    """
    Locates a specified segmentation dataset within the local 'celldetective/datasets/segmentation_annotations' directory
    or downloads it from Zenodo if not found locally.

    This function attempts to find a segmentation dataset by name within a predefined directory structure. If the dataset
    is not found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate
    category directory within 'celldetective'. The function prints the search directory path and returns the path to the
    found or downloaded dataset.

    Parameters
    ----------
    name : str
            The name of the segmentation dataset to locate.

    Returns
    -------
    str or None
            The full path to the located or downloaded segmentation dataset directory, or None if the dataset could not be
            found or downloaded.

    Raises
    ------
    FileNotFoundError
            If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
    )
    modelpath = os.sep.join([main_dir, "datasets", "segmentation_annotations", os.sep])
    logger.info(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def get_signal_datasets_list(return_path=False):
    """
    Retrieves a list of available signal datasets from both the local 'celldetective/datasets/signal_annotations' directory
    and a Zenodo repository, optionally returning the path to the local datasets directory.

    This function compiles a list of available signal datasets by first identifying datasets stored locally within a specified
    path related to the script's directory. It then extends this list with datasets available in a Zenodo repository, ensuring
    no duplicates are added. The function can return just the list of dataset names or, if specified, also return the path to
    the local datasets directory.

    Parameters
    ----------
    return_path : bool, optional
            If True, the function returns a tuple containing the list of available dataset names and the path to the local datasets
            directory. If False, only the list of dataset names is returned (default is False).

    Returns
    -------
    list or (list, str)
            If return_path is False, returns a list of strings, each string being the name of an available dataset. If return_path
            is True, returns a tuple where the first element is this list and the second element is a string representing the path
            to the local datasets directory.

    """

    datasets_path = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            "celldetective",
            "datasets",
            "signal_annotations",
            os.sep,
        ]
    )
    repository_datasets = get_zenodo_files(
        cat=os.sep.join(["datasets", "signal_annotations"])
    )

    available_datasets = natsorted(glob(datasets_path + "*/"))
    available_datasets = [
        m.replace("\\", "/").split("/")[-2] for m in available_datasets
    ]
    for rm in repository_datasets:
        if rm not in available_datasets:
            available_datasets.append(rm)

    if not return_path:
        return available_datasets
    else:
        return available_datasets, datasets_path


def locate_signal_dataset(name):
    """
    Locates a specified signal dataset within the local 'celldetective/datasets/signal_annotations' directory or downloads
    it from Zenodo if not found locally.

    This function attempts to find a signal dataset by name within a predefined directory structure. If the dataset is not
    found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate category
    directory within 'celldetective'. The function prints the search directory path and returns the path to the found or
    downloaded dataset.

    Parameters
    ----------
    name : str
            The name of the signal dataset to locate.

    Returns
    -------
    str or None
            The full path to the located or downloaded signal dataset directory, or None if the dataset could not be found or
            downloaded.

    Raises
    ------
    FileNotFoundError
            If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
    )
    modelpath = os.sep.join([main_dir, "datasets", "signal_annotations", os.sep])
    logger.info(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def normalize(
    frame,
    percentiles=(0.0, 99.99),
    values=None,
    ignore_gray_value=0.0,
    clip=False,
    amplification=None,
    dtype=float,
):
    """

    Normalize the intensity values of a frame.

    Parameters
    ----------
    frame : ndarray
            The input frame to be normalized.
    percentiles : tuple, optional
            The percentiles used to determine the minimum and maximum values for normalization. Default is (0.0, 99.99).
    values : tuple or None, optional
            The specific minimum and maximum values to use for normalization. If None, percentiles are used. Default is None.
    ignore_gray_value : float or None, optional
            The gray value to ignore during normalization. If specified, the pixels with this value will not be normalized. Default is 0.0.

    Returns
    -------
    ndarray
            The normalized frame.

    Notes
    -----
    This function performs intensity normalization on a frame. It computes the minimum and maximum values for normalization either
    using the specified values or by calculating percentiles from the frame. The frame is then normalized between the minimum and
    maximum values using the `normalize_mi_ma` function. If `ignore_gray_value` is specified, the pixels with this value will be
    left unmodified during normalization.

    Examples
    --------
    >>> frame = np.array([[10, 20, 30],
                                              [40, 50, 60],
                                              [70, 80, 90]])
    >>> normalized = normalize(frame)
    >>> normalized

    array([[0. , 0.2, 0.4],
               [0.6, 0.8, 1. ],
               [1.2, 1.4, 1.6]], dtype=float32)

    >>> normalized = normalize(frame, percentiles=(10.0, 90.0))
    >>> normalized

    array([[0.33333334, 0.44444445, 0.5555556 ],
               [0.6666667 , 0.7777778 , 0.8888889 ],
               [1.        , 1.1111112 , 1.2222222 ]], dtype=float32)

    """

    frame = frame.astype(float)

    if ignore_gray_value is not None:
        subframe = frame[frame != ignore_gray_value]
    else:
        subframe = frame.copy()

    if values is not None:
        mi = values[0]
        ma = values[1]
    else:
        mi = np.nanpercentile(subframe.flatten(), percentiles[0], keepdims=True)
        ma = np.nanpercentile(subframe.flatten(), percentiles[1], keepdims=True)

    frame0 = frame.copy()
    frame = normalize_mi_ma(frame0, mi, ma, clip=False, eps=1e-20, dtype=np.float32)
    if amplification is not None:
        frame *= amplification
    if clip:
        if amplification is None:
            amplification = 1.0
        frame[frame >= amplification] = amplification
        frame[frame <= 0.0] = 0.0
    if ignore_gray_value is not None:
        frame[np.where(frame0) == ignore_gray_value] = ignore_gray_value

    return frame.copy().astype(dtype)


def normalize_multichannel(
    multichannel_frame: np.ndarray,
    percentiles=None,
    values=None,
    ignore_gray_value=0.0,
    clip=False,
    amplification=None,
    dtype=float,
):
    """
    Normalizes a multichannel frame by adjusting the intensity values of each channel based on specified percentiles,
    direct value ranges, or amplification factors, with options to ignore a specific gray value and to clip the output.

    Parameters
    ----------
    multichannel_frame : ndarray
            The input multichannel image frame to be normalized, expected to be a 3-dimensional array where the last dimension
            represents the channels.
    percentiles : list of tuples or tuple, optional
            Percentile ranges (low, high) for each channel used to scale the intensity values. If a single tuple is provided,
            it is applied to all channels. If None, the default percentile range of (0., 99.99) is used for each channel.
    values : list of tuples or tuple, optional
            Direct value ranges (min, max) for each channel to scale the intensity values. If a single tuple is provided, it
            is applied to all channels. This parameter overrides `percentiles` if provided.
    ignore_gray_value : float, optional
            A specific gray value to ignore during normalization (default is 0.).
    clip : bool, optional
            If True, clips the output values to the range [0, 1] or the specified `dtype` range if `dtype` is not float
            (default is False).
    amplification : float, optional
            A factor by which to amplify the intensity values after normalization. If None, no amplification is applied.
    dtype : data-type, optional
            The desired data-type for the output normalized frame. The default is float, but other types can be specified
            to change the range of the output values.

    Returns
    -------
    ndarray
            The normalized multichannel frame as a 3-dimensional array of the same shape as `multichannel_frame`.

    Raises
    ------
    AssertionError
            If the input `multichannel_frame` does not have 3 dimensions, or if the length of `values` does not match the
            number of channels in `multichannel_frame`.

    Notes
    -----
    - This function provides flexibility in normalization by allowing the use of percentile ranges, direct value ranges,
      or amplification factors.
    - The function makes a copy of the input frame to avoid altering the original data.
    - When both `percentiles` and `values` are provided, `values` takes precedence for normalization.

    Examples
    --------
    >>> multichannel_frame = np.random.rand(100, 100, 3)  # Example multichannel frame
    >>> normalized_frame = normalize_multichannel(multichannel_frame, percentiles=[(1, 99), (2, 98), (0, 100)])
    # Normalizes each channel of the frame using specified percentile ranges.

    """

    mf = multichannel_frame.copy().astype(float)
    assert mf.ndim == 3, f"Wrong shape for the multichannel frame: {mf.shape}."
    if percentiles is None:
        percentiles = [(0.0, 99.99)] * mf.shape[-1]
    elif isinstance(percentiles, tuple):
        percentiles = [percentiles] * mf.shape[-1]
    if values is not None:
        if isinstance(values, tuple):
            values = [values] * mf.shape[-1]
        assert (
            len(values) == mf.shape[-1]
        ), "Mismatch between the normalization values provided and the number of channels."

    mf_new = []
    for c in range(mf.shape[-1]):
        if values is not None:
            v = values[c]
        else:
            v = None

        if np.all(mf[:, :, c] == 0.0):
            mf_new.append(mf[:, :, c].copy())
        else:
            norm = normalize(
                mf[:, :, c].copy(),
                percentiles=percentiles[c],
                values=v,
                ignore_gray_value=ignore_gray_value,
                clip=clip,
                amplification=amplification,
                dtype=dtype,
            )
            mf_new.append(norm)

    return np.moveaxis(mf_new, 0, -1)


def load_frames(
    img_nums,
    stack_path,
    scale=None,
    normalize_input=True,
    dtype=np.float64,
    normalize_kwargs={"percentiles": (0.0, 99.99)},
):
    """
    Loads and optionally normalizes and rescales specified frames from a stack located at a given path.

    This function reads specified frames from a stack file, applying systematic adjustments to ensure
    the channel axis is last. It supports optional normalization of the input frames and rescaling. An
    artificial pixel modification is applied to frames with uniform values to prevent errors during
    normalization.

    Parameters
    ----------
    img_nums : int or list of int
            The index (or indices) of the image frame(s) to load from the stack.
    stack_path : str
            The file path to the stack from which frames are to be loaded.
    scale : float, optional
            The scaling factor to apply to the frames. If None, no scaling is applied (default is None).
    normalize_input : bool, optional
            Whether to normalize the loaded frames. If True, normalization is applied according to
            `normalize_kwargs` (default is True).
    dtype : data-type, optional
            The desired data-type for the output frames (default is float).
    normalize_kwargs : dict, optional
            Keyword arguments to pass to the normalization function (default is {"percentiles": (0., 99.99)}).

    Returns
    -------
    ndarray or None
            The loaded, and possibly normalized and rescaled, frames as a NumPy array. Returns None if there
            is an error in loading the frames.

    Raises
    ------
    Exception
            Prints an error message if the specified frames cannot be loaded or if there is a mismatch between
            the provided experiment channel information and the stack format.

    Notes
    -----
    - The function uses scikit-image for reading frames and supports multi-frame TIFF stacks.
    - Normalization and scaling are optional and can be customized through function parameters.
    - A workaround is implemented for frames with uniform pixel values to prevent normalization errors by
      adding a 'fake' pixel.

    Examples
    --------
    >>> frames = load_frames([0, 1, 2], '/path/to/stack.tif', scale=0.5, normalize_input=True, dtype=np.uint8)
    # Loads the first three frames from '/path/to/stack.tif', normalizes them, rescales by a factor of 0.5,
    # and converts them to uint8 data type.

    """

    try:
        frames = imageio.imread(stack_path, key=img_nums)
    except Exception as e:
        logger.error(
            f"Error in loading the frame {img_nums} {e}. Please check that the experiment channel information is consistent with the movie being read."
        )
        return None
    try:
        frames[np.isinf(frames)] = np.nan
    except Exception as e:
        logger.error(e)

    frames = _rearrange_multichannel_frame(frames)

    if normalize_input:
        frames = normalize_multichannel(frames, **normalize_kwargs)

    if scale is not None:
        frames = zoom_multiframes(frames, scale)

    # add a fake pixel to prevent auto normalization errors on images that are uniform
    frames = _fix_no_contrast(frames)

    return frames.astype(dtype)


def get_stack_normalization_values(stack, percentiles=None, ignore_gray_value=0.0):
    """
    Computes the normalization value ranges (minimum and maximum) for each channel in a 4D stack based on specified percentiles.

    This function calculates the value ranges for normalizing each channel within a 4-dimensional stack, with dimensions
    expected to be in the order of Time (T), Y (height), X (width), and Channels (C). The normalization values are determined
    by the specified percentiles for each channel. An option to ignore a specific gray value during computation is provided,
    though its effect is not implemented in this snippet.

    Parameters
    ----------
    stack : ndarray
            The input 4D stack with dimensions TYXC from which to calculate normalization values.
    percentiles : tuple, list of tuples, optional
            The percentile values (low, high) used to calculate the normalization ranges for each channel. If a single tuple
            is provided, it is applied to all channels. If a list of tuples is provided, each tuple is applied to the
            corresponding channel. If None, defaults to (0., 99.99) for each channel.
    ignore_gray_value : float, optional
            A gray value to potentially ignore during the calculation. This parameter is provided for interface consistency
            but is not utilized in the current implementation (default is 0.).

    Returns
    -------
    list of tuples
            A list where each tuple contains the (minimum, maximum) values for normalizing each channel based on the specified
            percentiles.

    Raises
    ------
    AssertionError
            If the input stack does not have 4 dimensions, or if the length of the `percentiles` list does not match the number
            of channels in the stack.

    Notes
    -----
    - The function assumes the input stack is in TYXC format, where T is the time dimension, Y and X are spatial dimensions,
      and C is the channel dimension.
    - Memory management via `gc.collect()` is employed after calculating normalization values for each channel to mitigate
      potential memory issues with large datasets.

    Examples
    --------
    >>> stack = np.random.rand(5, 100, 100, 3)  # Example 4D stack with 3 channels
    >>> normalization_values = get_stack_normalization_values(stack, percentiles=((1, 99), (2, 98), (0, 100)))
    # Calculates normalization ranges for each channel using the specified percentiles.

    """

    assert (
        stack.ndim == 4
    ), f"Wrong number of dimensions for the stack, expect TYXC (4) got {stack.ndim}."
    if percentiles is None:
        percentiles = [(0.0, 99.99)] * stack.shape[-1]
    elif isinstance(percentiles, tuple):
        percentiles = [percentiles] * stack.shape[-1]
    elif isinstance(percentiles, list):
        assert (
            len(percentiles) == stack.shape[-1]
        ), f"Mismatch between the provided percentiles and the number of channels {stack.shape[-1]}. If you meant to apply the same percentiles to all channels, please provide a single tuple."

    values = []
    for c in range(stack.shape[-1]):
        perc = percentiles[c]
        mi = np.nanpercentile(stack[:, :, :, c].flatten(), perc[0], keepdims=True)[0]
        ma = np.nanpercentile(stack[:, :, :, c].flatten(), perc[1], keepdims=True)[0]
        values.append(tuple((mi, ma)))
        gc.collect()

    return values


def get_positions_in_well(well):
    """
    Retrieves the list of position directories within a specified well directory,
    formatted as a NumPy array of strings.

    This function identifies position directories based on their naming convention,
    which must include a numeric identifier following the well's name. The well's name
    is expected to start with 'W' (e.g., 'W1'), followed by a numeric identifier. Position
    directories are assumed to be named with this numeric identifier directly after the well
    identifier, without the 'W'. For example, positions within well 'W1' might be named
    '101', '102', etc. This function will glob these directories and return their full
    paths as a NumPy array.

    Parameters
    ----------
    well : str
            The path to the well directory from which to retrieve position directories.

    Returns
    -------
    np.ndarray
            An array of strings, each representing the full path to a position directory within
            the specified well. The array is empty if no position directories are found.

    Notes
    -----
    - This function relies on a specific naming convention for wells and positions. It assumes
      that each well directory is prefixed with 'W' followed by a numeric identifier, and
      position directories are named starting with this numeric identifier directly.

    Examples
    --------
    >>> get_positions_in_well('/path/to/experiment/W1')
    # This might return an array like array(['/path/to/experiment/W1/101', '/path/to/experiment/W1/102'])
    if position directories '101' and '102' exist within the well 'W1' directory.

    """

    if well.endswith(os.sep):
        well = well[:-1]

    w_numeric = os.path.split(well)[-1].replace("W", "")
    positions = natsorted(glob(os.sep.join([well, f"{w_numeric}*{os.sep}"])))

    return np.array(positions, dtype=str)


def extract_experiment_folder_output(experiment_folder, destination_folder):
    """
    Copies the output subfolder and associated tables from an experiment folder to a new location,
    making the experiment folder much lighter by only keeping essential data.

    This function takes the path to an experiment folder and a destination folder as input.
    It creates a copy of the experiment folder at the destination, but only includes the output subfolders
    and their associated tables for each well and position within the experiment.
    This operation significantly reduces the size of the experiment data by excluding non-essential files.

    The structure of the copied experiment folder is preserved, including the configuration file,
    well directories, and position directories within each well.
    Only the 'output' subfolder and its 'tables' subdirectory are copied for each position.

    Parameters
    ----------
    experiment_folder : str
            The path to the source experiment folder from which to extract data.
    destination_folder : str
            The path to the destination folder where the reduced copy of the experiment
            will be created.

    Notes
    -----
    - This function assumes that the structure of the experiment folder is consistent,
      with wells organized in subdirectories and each containing a position subdirectory.
      Each position subdirectory should have an 'output' folder and a 'tables' subfolder within it.

    - The function also assumes the existence of a configuration file in the root of the
      experiment folder, which is copied to the root of the destination experiment folder.

    Examples
    --------
    >>> extract_experiment_folder_output('/path/to/experiment_folder', '/path/to/destination_folder')
    # This will copy the 'experiment_folder' to 'destination_folder', including only
    # the output subfolders and their tables for each well and position.

    """

    if experiment_folder.endswith(os.sep):
        experiment_folder = experiment_folder[:-1]
    if destination_folder.endswith(os.sep):
        destination_folder = destination_folder[:-1]

    exp_name = experiment_folder.split(os.sep)[-1]
    output_path = os.sep.join([destination_folder, exp_name])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    config = get_config(experiment_folder)
    copyfile(config, os.sep.join([output_path, os.path.split(config)[-1]]))

    wells_src = get_experiment_wells(experiment_folder)
    wells = [w.split(os.sep)[-2] for w in wells_src]

    for k, w in enumerate(wells):

        well_output_path = os.sep.join([output_path, w])
        if not os.path.exists(well_output_path):
            os.mkdir(well_output_path)

        positions = get_positions_in_well(wells_src[k])

        for pos in positions:
            pos_name = extract_position_name(pos)
            output_pos = os.sep.join([well_output_path, pos_name])
            if not os.path.exists(output_pos):
                os.mkdir(output_pos)
            output_folder = os.sep.join([output_pos, "output"])
            output_tables_folder = os.sep.join([output_folder, "tables"])

            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            if not os.path.exists(output_tables_folder):
                os.mkdir(output_tables_folder)

            tab_path = glob(pos + os.sep.join(["output", "tables", f"*"]))

            for t in tab_path:
                copyfile(t, os.sep.join([output_tables_folder, os.path.split(t)[-1]]))


def _load_frames_to_segment(file, indices, scale_model=None, normalize_kwargs=None):

    frames = load_frames(
        indices,
        file,
        scale=scale_model,
        normalize_input=True,
        normalize_kwargs=normalize_kwargs,
    )
    frames = interpolate_nan_multichannel(frames)

    if np.any(indices == -1):
        frames[:, :, np.where(indices == -1)[0]] = 0.0

    return frames


def _load_frames_to_measure(file, indices):
    return load_frames(indices, file, scale=None, normalize_input=False)


def _check_label_dims(lbl, file=None, template=None):

    if file is not None:
        template = load_frames(0, file, scale=1, normalize_input=False)
    elif template is not None:
        template = template
    else:
        return lbl

    if lbl.shape != template.shape[:2]:
        lbl = resize(lbl, template.shape[:2], order=0)
    return lbl


if __name__ == "__main__":
    control_segmentation_napari(
        "/home/limozin/Documents/Experiments/MinimumJan/W4/401/",
        prefix="Aligned",
        population="target",
        flush_memory=False,
    )
