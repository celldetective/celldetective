import os
from glob import glob
from pathlib import Path, PosixPath, PurePosixPath, WindowsPath
from shutil import copyfile
from typing import Union, List, Tuple

import numpy as np
from natsort import natsorted

from celldetective.utils.parsing import (
    _extract_channels_from_config,
    config_section_to_dict,
)


def extract_well_from_position(pos_path):
    """
    Extracts the well directory path from a given position directory path.

    Parameters
    ----------
    pos_path : str
            The file system path to a position directory. The path should end with the position folder,
            but it does not need to include a trailing separator.

    Returns
    -------
    str
            The path to the well directory, which is assumed to be two levels above the position directory,
            with a trailing separator appended.

    Notes
    -----
    - This function expects the position directory to be organized such that the well directory is
      two levels above it in the file system hierarchy.
    - If the input path does not end with a file separator (`os.sep`), one is appended before processing.

    Example
    -------
    >>> pos_path = "/path/to/experiment/plate/well/position"
    >>> extract_well_from_position(pos_path)
    '/path/to/experiment/plate/well/'

    """

    if not pos_path.endswith(os.sep):
        pos_path += os.sep
    well_path_blocks = pos_path.split(os.sep)[:-2]
    well_path = os.sep.join(well_path_blocks) + os.sep
    return well_path


def extract_experiment_from_position(pos_path):
    """
    Extracts the experiment directory path from a given position directory path.

    Parameters
    ----------
    pos_path : str
            The file system path to a position directory. The path should end with the position folder,
            but it does not need to include a trailing separator.

    Returns
    -------
    str
            The path to the experiment directory, which is assumed to be three levels above the position directory.

    Notes
    -----
    - This function expects the position directory to be organized hierarchically such that the experiment directory
      is three levels above it in the file system hierarchy.
    - If the input path does not end with a file separator (`os.sep`), one is appended before processing.

    Example
    -------
    >>> pos_path = "/path/to/experiment/plate/well/position"
    >>> extract_experiment_from_position(pos_path)
    '/path/to/experiment'

    """

    pos_path = pos_path.replace(os.sep, "/")
    if not pos_path.endswith("/"):
        pos_path += "/"
    exp_path_blocks = pos_path.split("/")[:-3]
    experiment = os.sep.join(exp_path_blocks)

    return experiment


def get_experiment_wells(experiment):
    """
    Retrieves the list of well directories from a given experiment directory, sorted
    naturally and returned as a NumPy array of strings.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory from which to retrieve well directories.

    Returns
    -------
    np.ndarray
            An array of strings, each representing the full path to a well directory within the specified
            experiment. The array is empty if no well directories are found.

    Notes
    -----
    - The function assumes well directories are prefixed with 'W' and uses this to filter directories
      within the experiment folder.

    - Natural sorting is applied to the list of wells to ensure that the order is intuitive (e.g., 'W2'
      comes before 'W10'). This sorting method is especially useful when dealing with numerical sequences
      that are part of the directory names.

    """

    if not experiment.endswith(os.sep):
        experiment += os.sep

    wells = natsorted(glob(experiment + "W*" + os.sep))
    return np.array(wells, dtype=str)


def extract_well_name_and_number(well):
    """
    Extract the well name and number from a given well path.

    This function takes a well path string, splits it by the OS-specific path separator,
    and extracts the well name and number. The well name is the last component of the path,
    and the well number is derived by removing the 'W' prefix and converting the remaining
    part to an integer.

    Parameters
    ----------
    well : str
            The well path string, where the well name is the last component.

    Returns
    -------
    well_name : str
            The name of the well, extracted from the last component of the path.
    well_number : int
            The well number, obtained by stripping the 'W' prefix from the well name
            and converting the remainder to an integer.

    Examples
    --------
    >>> well_path = "path/to/W23"
    >>> extract_well_name_and_number(well_path)
    ('W23', 23)

    >>> well_path = "another/path/W1"
    >>> extract_well_name_and_number(well_path)
    ('W1', 1)

    """

    split_well_path = well.split(os.sep)
    split_well_path = list(filter(None, split_well_path))
    well_name = split_well_path[-1]
    well_number = int(split_well_path[-1].replace("W", ""))

    return well_name, well_number


def extract_position_name(pos):
    """
    Extract the position name from a given position path.

    This function takes a position path string, splits it by the OS-specific path separator,
    filters out any empty components, and extracts the position name, which is the last
    component of the path.

    Parameters
    ----------
    pos : str
            The position path string, where the position name is the last component.

    Returns
    -------
    pos_name : str
            The name of the position, extracted from the last component of the path.

    Examples
    --------
    >>> pos_path = "path/to/position1"
    >>> extract_position_name(pos_path)
    'position1'

    >>> pos_path = "another/path/positionA"
    >>> extract_position_name(pos_path)
    'positionA'

    """

    split_pos_path = pos.split(os.sep)
    split_pos_path = list(filter(None, split_pos_path))
    pos_name = split_pos_path[-1]

    return pos_name


def extract_experiment_channels(experiment):
    """
    Extracts channel names and their indices from an experiment project.

    Parameters
    ----------
    experiment : str
            The file system path to the directory of the experiment project.

    Returns
    -------
    tuple
            A tuple containing two numpy arrays: `channel_names` and `channel_indices`. `channel_names` includes
            the names of the channels as specified in the configuration, and `channel_indices` includes their
            corresponding indices. Both arrays are ordered according to the channel indices.

    Examples
    --------
    >>> experiment = "path/to/my_experiment"
    >>> channels, indices = extract_experiment_channels(experiment)
    >>> print(channels)
    # array(['brightfield_channel', 'adhesion_channel', 'fitc_channel',
    #    'cy5_channel'], dtype='<U19')
    >>> print(indices)
    # array([0, 1, 2, 3])
    """

    config = get_config(experiment)
    return _extract_channels_from_config(config)


def get_spatial_calibration(experiment):
    """
    Retrieves the spatial calibration factor for an experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the experiment directory.

    Returns
    -------
    float
            The spatial calibration factor (pixels to micrometers conversion), extracted from the experiment's configuration file.

    Raises
    ------
    AssertionError
            If the configuration file (`config.ini`) does not exist in the specified experiment directory.
    KeyError
            If the "pxtoum" key is not found under the "MovieSettings" section in the configuration file.
    ValueError
            If the retrieved "pxtoum" value cannot be converted to a float.

    Notes
    -----
    - The function retrieves the calibration factor by first locating the configuration file for the experiment using `get_config()`.
    - It expects the configuration file to have a section named `MovieSettings` containing the key `pxtoum`.
    - This factor defines the conversion from pixels to micrometers for spatial measurements.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> calibration = get_spatial_calibration(experiment)
    >>> print(calibration)
    0.325  # pixels-to-micrometers conversion factor

    """

    config = get_config(experiment)
    px_to_um = float(config_section_to_dict(config, "MovieSettings")["pxtoum"])

    return px_to_um


def get_temporal_calibration(experiment):
    """
    Retrieves the temporal calibration factor for an experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the experiment directory.

    Returns
    -------
    float
            The temporal calibration factor (frames to minutes conversion), extracted from the experiment's configuration file.

    Raises
    ------
    AssertionError
            If the configuration file (`config.ini`) does not exist in the specified experiment directory.
    KeyError
            If the "frametomin" key is not found under the "MovieSettings" section in the configuration file.
    ValueError
            If the retrieved "frametomin" value cannot be converted to a float.

    Notes
    -----
    - The function retrieves the calibration factor by locating the configuration file for the experiment using `get_config()`.
    - It expects the configuration file to have a section named `MovieSettings` containing the key `frametomin`.
    - This factor defines the conversion from frames to minutes for temporal measurements.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> calibration = get_temporal_calibration(experiment)
    >>> print(calibration)
    0.5  # frames-to-minutes conversion factor

    """

    config = get_config(experiment)
    frame_to_min = float(config_section_to_dict(config, "MovieSettings")["frametomin"])

    return frame_to_min


def get_experiment_metadata(experiment):

    config = get_config(experiment)
    metadata = config_section_to_dict(config, "Metadata")
    return metadata


def get_experiment_labels(experiment):

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    labels = config_section_to_dict(config, "Labels")
    for k in list(labels.keys()):
        values = labels[k].split(",")
        if nbr_of_wells != len(values):
            values = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]
        if np.all(np.array([s.isnumeric() for s in values])):
            values = [float(s) for s in values]
        labels.update({k: values})

    return labels


def get_experiment_concentrations(experiment, dtype=str):
    """
    Retrieves the concentrations associated with each well in an experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the experiment directory.
    dtype : type, optional
            The data type to which the concentrations should be converted (default is `str`).

    Returns
    -------
    numpy.ndarray
            An array of concentrations for each well, converted to the specified data type.

    Raises
    ------
    AssertionError
            If the configuration file (`config.ini`) does not exist in the specified experiment directory.
    KeyError
            If the "concentrations" key is not found under the "Labels" section in the configuration file.
    ValueError
            If the retrieved concentrations cannot be converted to the specified data type.

    Notes
    -----
    - The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing
      a key `concentrations`.
    - The concentrations are assumed to be comma-separated values.
    - If the number of wells does not match the number of concentrations, the function generates a default set
      of values ranging from 0 to the number of wells minus 1.
    - The resulting concentrations are converted to the specified `dtype` before being returned.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> concentrations = get_experiment_concentrations(experiment, dtype=float)
    >>> print(concentrations)
    [0.1, 0.2, 0.5, 1.0]

    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    concentrations = config_section_to_dict(config, "Labels")["concentrations"].split(
        ","
    )
    if nbr_of_wells != len(concentrations):
        concentrations = [
            str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)
        ]

    return np.array([dtype(c) for c in concentrations])


def get_experiment_cell_types(experiment, dtype=str):
    """
    Retrieves the cell types associated with each well in an experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the experiment directory.
    dtype : type, optional
            The data type to which the cell types should be converted (default is `str`).

    Returns
    -------
    numpy.ndarray
            An array of cell types for each well, converted to the specified data type.

    Raises
    ------
    AssertionError
            If the configuration file (`config.ini`) does not exist in the specified experiment directory.
    KeyError
            If the "cell_types" key is not found under the "Labels" section in the configuration file.
    ValueError
            If the retrieved cell types cannot be converted to the specified data type.

    Notes
    -----
    - The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing
      a key `cell_types`.
    - The cell types are assumed to be comma-separated values.
    - If the number of wells does not match the number of cell types, the function generates a default set
      of values ranging from 0 to the number of wells minus 1.
    - The resulting cell types are converted to the specified `dtype` before being returned.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> cell_types = get_experiment_cell_types(experiment, dtype=str)
    >>> print(cell_types)
    ['TypeA', 'TypeB', 'TypeC', 'TypeD']

    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    cell_types = config_section_to_dict(config, "Labels")["cell_types"].split(",")
    if nbr_of_wells != len(cell_types):
        cell_types = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

    return np.array([dtype(c) for c in cell_types])


def get_experiment_antibodies(experiment, dtype=str):
    """
    Retrieve the list of antibodies used in an experiment.

    This function extracts antibody labels for the wells in the given experiment
    based on the configuration file. If the number of wells does not match the
    number of antibody labels provided in the configuration, it generates a
    sequence of default numeric labels.

    Parameters
    ----------
    experiment : str
            The identifier or name of the experiment to retrieve antibodies for.
    dtype : type, optional
            The data type to which the antibody labels should be cast. Default is `str`.

    Returns
    -------
    numpy.ndarray
            An array of antibody labels with the specified data type. If no antibodies
            are specified or there is a mismatch, numeric labels are generated instead.

    Notes
    -----
    - The function assumes the experiment's configuration can be loaded using
      `get_config` and that the antibodies are listed under the "Labels" section
      with the key `"antibodies"`.
    - A mismatch between the number of wells and antibody labels will result in
      numeric labels generated using `numpy.linspace`.

    Examples
    --------
    >>> get_experiment_antibodies("path/to/experiment1")
    array(['A1', 'A2', 'A3'], dtype='<U2')

    >>> get_experiment_antibodies("path/to/experiment2", dtype=int)
    array([0, 1, 2])

    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    antibodies = config_section_to_dict(config, "Labels")["antibodies"].split(",")
    if nbr_of_wells != len(antibodies):
        antibodies = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

    return np.array([dtype(c) for c in antibodies])


def get_experiment_pharmaceutical_agents(experiment, dtype=str):
    """
    Retrieves the antibodies associated with each well in an experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the experiment directory.
    dtype : type, optional
            The data type to which the antibodies should be converted (default is `str`).

    Returns
    -------
    numpy.ndarray
            An array of antibodies for each well, converted to the specified data type.

    Raises
    ------
    AssertionError
            If the configuration file (`config.ini`) does not exist in the specified experiment directory.
    KeyError
            If the "antibodies" key is not found under the "Labels" section in the configuration file.
    ValueError
            If the retrieved antibody values cannot be converted to the specified data type.

    Notes
    -----
    - The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing
      a key `antibodies`.
    - The antibody names are assumed to be comma-separated values.
    - If the number of wells does not match the number of antibodies, the function generates a default set
      of values ranging from 0 to the number of wells minus 1.
    - The resulting antibody names are converted to the specified `dtype` before being returned.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> antibodies = get_experiment_antibodies(experiment, dtype=str)
    >>> print(antibodies)
    ['AntibodyA', 'AntibodyB', 'AntibodyC', 'AntibodyD']

    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    pharmaceutical_agents = config_section_to_dict(config, "Labels")[
        "pharmaceutical_agents"
    ].split(",")
    if nbr_of_wells != len(pharmaceutical_agents):
        pharmaceutical_agents = [
            str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)
        ]

    return np.array([dtype(c) for c in pharmaceutical_agents])


def get_experiment_populations(experiment, dtype=str):

    config = get_config(experiment)
    populations_str = config_section_to_dict(config, "Populations")
    if populations_str is not None:
        populations = populations_str["populations"].split(",")
    else:
        populations = ["effectors", "targets"]
    return list([dtype(c) for c in populations])


def get_config(experiment: Union[str, Path]) -> str:
    """
    Retrieves the path to the configuration file for a given experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the directory of the experiment project.

    Returns
    -------
    str
            The full path to the configuration file (`config.ini`) within the experiment directory.

    Raises
    ------
    AssertionError
            If the `config.ini` file does not exist in the specified experiment directory.

    Notes
    -----
    - The function ensures that the provided experiment path ends with the appropriate file separator (`os.sep`)
      before appending `config.ini` to locate the configuration file.
    - The configuration file is expected to be named `config.ini` and located at the root of the experiment directory.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> config_path = get_config(experiment)
    >>> print(config_path)
    '/path/to/experiment/config.ini'

    """

    if isinstance(experiment, (PosixPath, PurePosixPath, WindowsPath)):
        experiment = str(experiment)

    if not experiment.endswith(os.sep):
        experiment += os.sep

    config = experiment + "config.ini"
    config = rf"{config}"

    assert os.path.exists(
        config
    ), "The experiment configuration could not be located..."
    return config


def extract_experiment_from_well(well_path):
    """
    Extracts the experiment directory path from a given well directory path.

    Parameters
    ----------
    well_path : str
            The file system path to a well directory. The path should end with the well folder,
            but it does not need to include a trailing separator.

    Returns
    -------
    str
            The path to the experiment directory, which is assumed to be two levels above the well directory.

    Notes
    -----
    - This function expects the well directory to be organized such that the experiment directory is
      two levels above it in the file system hierarchy.
    - If the input path does not end with a file separator (`os.sep`), one is appended before processing.

    Example
    -------
    >>> well_path = "/path/to/experiment/plate/well"
    >>> extract_experiment_from_well(well_path)
    '/path/to/experiment'

    """

    if not well_path.endswith(os.sep):
        well_path += os.sep
    exp_path_blocks = well_path.split(os.sep)[:-2]
    experiment = os.sep.join(exp_path_blocks)
    return experiment


def collect_experiment_metadata(pos_path=None, well_path=None):
    """
    Collects and organizes metadata for an experiment based on a given position or well directory path.

    Parameters
    ----------
    pos_path : str, optional
            The file system path to a position directory. If provided, it will be used to extract metadata.
            This parameter takes precedence over `well_path`.
    well_path : str, optional
            The file system path to a well directory. If `pos_path` is not provided, this path will be used to extract metadata.

    Returns
    -------
    dict
            A dictionary containing the following metadata:
            - `"pos_path"`: The path to the position directory (or `None` if not provided).
            - `"position"`: The same as `pos_path`.
            - `"pos_name"`: The name of the position (or `0` if `pos_path` is not provided).
            - `"well_path"`: The path to the well directory.
            - `"well_name"`: The name of the well.
            - `"well_nbr"`: The numerical identifier of the well.
            - `"experiment"`: The path to the experiment directory.
            - `"antibody"`: The antibody associated with the well.
            - `"concentration"`: The concentration associated with the well.
            - `"cell_type"`: The cell type associated with the well.
            - `"pharmaceutical_agent"`: The pharmaceutical agent associated with the well.

    Notes
    -----
    - At least one of `pos_path` or `well_path` must be provided.
    - The function determines the experiment path by navigating the directory structure and extracts metadata for the
      corresponding well and position.
    - The metadata is derived using helper functions like `extract_experiment_from_position`, `extract_well_from_position`,
      and `get_experiment_*` family of functions.

    Example
    -------
    >>> pos_path = "/path/to/experiment/plate/well/position"
    >>> metadata = collect_experiment_metadata(pos_path=pos_path)
    >>> metadata["well_name"]
    'W1'

    >>> well_path = "/path/to/experiment/plate/well"
    >>> metadata = collect_experiment_metadata(well_path=well_path)
    >>> metadata["concentration"]
    10.0

    """

    if pos_path is not None:
        if not pos_path.endswith(os.sep):
            pos_path += os.sep
        experiment = extract_experiment_from_position(pos_path)
        well_path = extract_well_from_position(pos_path)
    elif well_path is not None:
        if not well_path.endswith(os.sep):
            well_path += os.sep
        experiment = extract_experiment_from_well(well_path)
    else:
        print("Please provide a position or well path...")
        return None

    wells = list(get_experiment_wells(experiment))
    idx = wells.index(well_path)
    well_name, well_nbr = extract_well_name_and_number(well_path)
    if pos_path is not None:
        pos_name = extract_position_name(pos_path)
    else:
        pos_name = 0

    dico = {
        "pos_path": pos_path,
        "position": pos_path,
        "pos_name": pos_name,
        "well_path": well_path,
        "well_name": well_name,
        "well_nbr": well_nbr,
        "experiment": experiment,
    }

    meta = get_experiment_metadata(experiment)  # None or dict of metadata
    if meta is not None:
        keys = list(meta.keys())
        for k in keys:
            dico.update({k: meta[k]})

    labels = get_experiment_labels(experiment)
    for k in list(labels.keys()):
        values = labels[k]
        try:
            dico.update({k: values[idx]})
        except Exception as e:
            print(f"{e=}")

    return dico


def interpret_wells_and_positions(
    experiment: str,
    well_option: Union[str, int, List[int]],
    position_option: Union[str, int, List[int]],
) -> Union[Tuple[List[int], List[int]], None]:
    """
    Interpret well and position options for a given experiment.

    This function takes an experiment and well/position options to return the selected
    wells and positions. It supports selection of all wells or specific wells/positions
    as specified. The well numbering starts from 0 (i.e., Well 0 is W1 and so on).

    Parameters
    ----------
    experiment : str
            The experiment path containing well information.
    well_option : str, int, or list of int
            The well selection option:
            - '*' : Select all wells.
            - int : Select a specific well by its index.
            - list of int : Select multiple wells by their indices.
    position_option : str, int, or list of int
            The position selection option:
            - '*' : Select all positions (returns None).
            - int : Select a specific position by its index.
            - list of int : Select multiple positions by their indices.

    Returns
    -------
    well_indices : numpy.ndarray or list of int
            The indices of the selected wells.
    position_indices : numpy.ndarray or list of int or None
            The indices of the selected positions. Returns None if all positions are selected.

    Examples
    --------
    >>> experiment = ...  # Some experiment object
    >>> interpret_wells_and_positions(experiment, '*', '*')
    (array([0, 1, 2, ..., n-1]), None)

    >>> interpret_wells_and_positions(experiment, 2, '*')
    ([2], None)

    >>> interpret_wells_and_positions(experiment, [1, 3, 5], 2)
    ([1, 3, 5], array([2]))

    """

    wells = get_experiment_wells(experiment)
    nbr_of_wells = len(wells)

    if well_option == "*":
        well_indices = np.arange(nbr_of_wells)
    elif isinstance(well_option, int) or isinstance(well_option, np.int_):
        well_indices = [int(well_option)]
    elif isinstance(well_option, list):
        well_indices = well_option
    else:
        print("Well indices could not be interpreted...")
        return None

    if position_option == "*":
        position_indices = None
    elif isinstance(position_option, int):
        position_indices = np.array([position_option], dtype=int)
    elif isinstance(position_option, list):
        position_indices = position_option
    else:
        print("Position indices could not be interpreted...")
        return None

    return well_indices, position_indices


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
