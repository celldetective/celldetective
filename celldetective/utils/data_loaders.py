import os

import numpy as np


from tqdm import tqdm
from celldetective import get_logger
from celldetective.utils.image_loaders import locate_stack_and_labels
from celldetective.utils.experiment import (
    get_config,
    get_experiment_wells,
    get_experiment_labels,
    get_experiment_metadata,
    extract_well_name_and_number,
    extract_position_name,
    interpret_wells_and_positions,
    get_position_movie_path,
    get_positions_in_well,
)
from celldetective.utils.parsing import (
    config_section_to_dict,
    _extract_labels_from_config,
)

logger = get_logger()


def get_position_table(pos, population, return_path=False):
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

    import pandas as pd

    if not pos.endswith(os.sep):
        table = os.sep.join([pos, "output", "tables", f"trajectories_{population}.csv"])
    else:
        table = pos + os.sep.join(
            ["output", "tables", f"trajectories_{population}.csv"]
        )

    if os.path.exists(table):
        try:
            df_pos = pd.read_csv(table, low_memory=False)
        except Exception as e:
            logger.error(e)
            df_pos = None
    else:
        df_pos = None

    if return_path:
        return df_pos, table
    else:
        return df_pos


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


def load_experiment_tables(
    experiment,
    population="targets",
    well_option="*",
    position_option="*",
    return_pos_info=False,
    load_pickle=False,
    progress_callback=None,
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

    import pandas as pd

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

    total_wells = len(well_indices)

    iterator = wells[well_indices]
    if progress_callback is None:
        iterator = tqdm(wells[well_indices])

    for k, well_path in enumerate(iterator):

        if progress_callback is not None:
            well_progress = round(k / total_wells * 100)
            # Signal keep_going logic if needed, but for now just send progress
            # If callback returns False, we could abort, but simpler to just update for now.

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
        total_positions = len(positions)

        for pidx, pos_path in enumerate(positions):

            if progress_callback is not None:
                pos_progress = round(pidx / total_positions * 100)
                should_continue = progress_callback(well_progress, pos_progress)
                if should_continue is False:
                    return None, None if return_pos_info else None

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


def load_tracking_data(position, prefix="Aligned", population="target"):
    """

    Load the tracking data, labels, and stack for a given position and population.

    Parameters
    ----------
    position : str
            The position or directory where the data is located.
    prefix : str, optional
            The prefix used in the filenames of the stack images (default is "Aligned").
    population : str, optional
            The population to load the data for. Options are "target" or "effector" (default is "target").

    Returns
    -------
    trajectories : DataFrame
            The tracking data loaded as a pandas DataFrame.
    labels : ndarray
            The segmentation labels loaded as a numpy ndarray.
    stack : ndarray
            The image stack loaded as a numpy ndarray.

    Notes
    -----
    This function loads the tracking data, labels, and stack for a given position and population.
    It reads the trajectories from the appropriate CSV file based on the specified population.
    The stack and labels are located using the `locate_stack_and_labels` function.
    The resulting tracking data is returned as a pandas DataFrame, and the labels and stack are returned as numpy ndarrays.

    Examples
    --------
    >>> trajectories, labels, stack = load_tracking_data(position, population="target")
    # Load the tracking data, labels, and stack for the specified position and target population.

    """

    import pandas as pd

    position = position.replace("\\", "/")
    if population.lower() == "target" or population.lower() == "targets":
        trajectories = pd.read_csv(
            position + os.sep.join(["output", "tables", "trajectories_targets.csv"])
        )
    elif population.lower() == "effector" or population.lower() == "effectors":
        trajectories = pd.read_csv(
            position + os.sep.join(["output", "tables", "trajectories_effectors.csv"])
        )
    else:
        trajectories = pd.read_csv(
            position
            + os.sep.join(["output", "tables", f"trajectories_{population}.csv"])
        )

    stack, labels = locate_stack_and_labels(
        position, prefix=prefix, population=population
    )

    return trajectories, labels, stack


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
                    # "celldetective",
                    "models",
                    "tracking_configs",
                    os.sep,
                ]
            )
            if os.path.exists(modelpath + config + ".json"):
                return modelpath + config + ".json"
            else:
                from btrack.datasets import cell_config

                config = cell_config()
    elif config is None:
        from btrack.datasets import cell_config

        config = cell_config()

    return config
