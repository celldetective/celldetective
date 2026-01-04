import re
from typing import Optional, List

import numpy as np
import pandas as pd


def _remove_invalid_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes invalid columns from a DataFrame.

    This function identifies and removes columns in the DataFrame whose names
    start with "Unnamed", or that contain only NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame from which invalid columns will be removed.

    Returns
    -------
    pandas.DataFrame
            A new DataFrame with the invalid columns removed. If no invalid
            columns are found, the original DataFrame is returned unchanged.
    """

    invalid_cols = [c for c in list(df.columns) if c.startswith("Unnamed")]
    if len(invalid_cols) > 0:
        df = df.drop(invalid_cols, axis=1)
    df = df.dropna(axis=1, how="all")
    return df


def _extract_coordinates_from_features(
    df: pd.DataFrame, timepoint: int
) -> pd.DataFrame:
    """
    Re-format coordinates from a regionprops table to tracking/measurement table format.

    Parameters
    ----------
    df : pandas.DataFrame
            A DataFrame containing feature data, including columns for centroids
            (`'centroid-1'` and `'centroid-0'`) and feature classes (`'class_id'`).
    timepoint : int
            The timepoint (frame) to assign to all features. This is used to populate
            the `'FRAME'` column in the output.

    Returns
    -------
    pandas.DataFrame
            A DataFrame containing the extracted coordinates and additional metadata,
            with the following columns:
            - `'POSITION_X'`: X-coordinate of the centroid.
            - `'POSITION_Y'`: Y-coordinate of the centroid.
            - `'class_id'`: The label associated to the cell mask.
            - `'ID'`: A unique identifier for each cell (index-based).
            - `'FRAME'`: The timepoint associated with the features.

    Notes
    -----
    - The function assumes that the input DataFrame contains columns `'centroid-1'`,
      `'centroid-0'`, and `'class_id'`. Missing columns will raise a KeyError.
    - The `'ID'` column is created based on the index of the input DataFrame.
    - This function renames `'centroid-1'` to `'POSITION_X'` and `'centroid-0'`
      to `'POSITION_Y'`.
    """

    coords = df[["centroid-1", "centroid-0", "class_id"]].copy()
    coords["ID"] = np.arange(len(coords))
    coords.rename(
        columns={"centroid-1": "POSITION_X", "centroid-0": "POSITION_Y"}, inplace=True
    )
    coords["FRAME"] = int(timepoint)

    return coords


def _mask_intensity_measurements(df: pd.DataFrame, mask_channels: Optional[List[str]]):
    """
    Removes columns from a DataFrame that match specific channel name patterns.

    This function filters out intensity measurement columns in a DataFrame based on
    specified channel names. It identifies columns containing the channel
    names as substrings and drops them from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
            The input DataFrame containing intensity measurement data. Column names should
            include the mask channel names if they are to be filtered.
    mask_channels : list of str or None
            A list of channel names (as substrings) to use for identifying columns
            to remove. If `None`, no filtering is applied, and the original DataFrame is
            returned.

    Returns
    -------
    pandas.DataFrame
            The modified DataFrame with specified columns removed. If no columns match
            the mask channels, the original DataFrame is returned.

    Notes
    -----
    - The function searches for mask channel substrings in column names.
      Partial matches are sufficient to mark a column for removal.
    - If no mask channels are specified (`mask_channels` is `None`), the function
      does not modify the input DataFrame.
    """

    if isinstance(mask_channels, str):
        mask_channels = [mask_channels]

    if mask_channels is not None:

        cols_to_drop = []
        columns = list(df.columns)

        for mc in mask_channels:
            cols_to_remove = [c for c in columns if mc in c]
            cols_to_drop.extend(cols_to_remove)

        if len(cols_to_drop) > 0:
            df = df.drop(cols_to_drop, axis=1)
    return df


def extract_cols_from_table_list(tables, nrows=1):
    """
    Extracts a unique list of column names from a list of CSV tables.

    Parameters
    ----------
    tables : list of str
            A list of file paths to the CSV tables from which to extract column names.
    nrows : int, optional
            The number of rows to read from each table to identify the columns.
            Default is 1.

    Returns
    -------
    numpy.ndarray
            An array of unique column names found across all the tables.

    Notes
    -----
    - This function reads only the first `nrows` rows of each table to improve performance when dealing with large files.
    - The function ensures that column names are unique by consolidating them using `numpy.unique`.

    Examples
    --------
    >>> tables = ["table1.csv", "table2.csv"]
    >>> extract_cols_from_table_list(tables)
    array(['Column1', 'Column2', 'Column3'], dtype='<U8')
    """

    all_columns = []
    for tab in tables:
        cols = pd.read_csv(tab, nrows=1).columns.tolist()
        all_columns.extend(cols)
    all_columns = np.unique(all_columns)
    return all_columns


def extract_identity_col(trajectories):
    """
    Determines the identity column name in a DataFrame of trajectories.

    This function checks the provided DataFrame for the presence of a column
    that can serve as the identity column. It first looks for the column
    'TRACK_ID'. If 'TRACK_ID' exists but contains only null values, it checks
    for the column 'ID' instead. If neither column is found, the function
    returns `None` and prints a message indicating the issue.

    Parameters
    ----------
    trajectories : pandas.DataFrame
            A DataFrame containing trajectory data. The function assumes that
            the identity of each trajectory might be stored in either the
            'TRACK_ID' or 'ID' column.

    Returns
    -------
    str or None
            The name of the identity column ('TRACK_ID' or 'ID') if found;
            otherwise, `None`.
    """

    for col in ["TRACK_ID", "ID"]:
        if col in trajectories.columns and not trajectories[col].isnull().all():
            return col

    print("ID or TRACK_ID column could not be found in the table...")
    return None


def rename_intensity_column(df, channels):
    """

    Rename intensity columns in a DataFrame based on the provided channel names.

    Parameters
    ----------
    df : pandas DataFrame
            The DataFrame containing the intensity columns.
    channels : list
            A list of channel names corresponding to the intensity columns.

    Returns
    -------
    pandas DataFrame
            The DataFrame with renamed intensity columns.

    Notes
    -----
    This function renames the intensity columns in a DataFrame based on the provided channel names.
    It searches for columns containing the substring 'intensity' in their names and replaces it with
    the respective channel name. The renaming is performed according to the order of the channels
    provided in the `channels` list. If multiple channels are provided, the function assumes that the
    intensity columns have a naming pattern that includes a numerical index indicating the channel.
    If only one channel is provided, the function replaces 'intensity' with the single channel name.

    Examples
    --------
    >>> data = {'intensity_0': [1, 2, 3], 'intensity_1': [4, 5, 6]}
    >>> df = pd.DataFrame(data)
    >>> channels = ['channel1', 'channel2']
    >>> renamed_df = rename_intensity_column(df, channels)
    # Rename the intensity columns in the DataFrame based on the provided channel names.

    """

    channel_names = np.array(channels)
    channel_indices = np.arange(len(channel_names), dtype=int)
    intensity_cols = [s for s in list(df.columns) if "intensity" in s]

    to_rename = {}

    for k in range(len(intensity_cols)):

        # identify if digit in section
        sections = np.array(re.split("-|_", intensity_cols[k]))
        test_digit = np.array([False for s in sections])
        for j, s in enumerate(sections):
            if str(s).isdigit():
                if int(s) < len(channel_names):
                    test_digit[j] = True

        if np.any(test_digit):
            index = int(sections[np.where(test_digit)[0]][-1])
        else:
            print(
                f"No valid channel index found for {intensity_cols[k]}... Skipping the renaming for {intensity_cols[k]}..."
            )
            continue

        channel_name = channel_names[np.where(channel_indices == index)[0]][0]
        new_name = np.delete(
            sections, np.where(test_digit)[0]
        )  # np.where(test_digit)[0]
        new_name = "_".join(list(new_name))
        new_name = new_name.replace("intensity", channel_name)
        new_name = new_name.replace("-", "_")
        new_name = new_name.replace("_nanmean", "_mean")

        to_rename.update({intensity_cols[k]: new_name})

        if "centre" in intensity_cols[k]:

            measure = np.array(re.split("-|_", new_name))

            if sections[-2] == "0":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(new_name))
                if "edge" in intensity_cols[k]:
                    new_name = new_name.replace(
                        "center_of_mass_displacement",
                        "edge_center_of_mass_displacement_in_px",
                    )
                else:
                    new_name = new_name.replace(
                        "center_of_mass", "center_of_mass_displacement_in_px"
                    )
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

            elif sections[-2] == "1":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(new_name))
                if "edge" in intensity_cols[k]:
                    new_name = new_name.replace(
                        "center_of_mass_displacement", "edge_center_of_mass_orientation"
                    )
                else:
                    new_name = new_name.replace(
                        "center_of_mass", "center_of_mass_orientation"
                    )
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

            elif sections[-2] == "2":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(new_name))
                if "edge" in intensity_cols[k]:
                    new_name = new_name.replace(
                        "center_of_mass_displacement", "edge_center_of_mass_x"
                    )
                else:
                    new_name = new_name.replace("center_of_mass", "center_of_mass_x")
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

            elif sections[-2] == "3":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(new_name))
                if "edge" in intensity_cols[k]:
                    new_name = new_name.replace(
                        "center_of_mass_displacement", "edge_center_of_mass_y"
                    )
                else:
                    new_name = new_name.replace("center_of_mass", "center_of_mass_y")
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

        if "radial_gradient" in intensity_cols[k]:
            # sections = np.array(re.split('-|_', intensity_columns[k]))
            measure = np.array(re.split("-|_", new_name))

            if sections[-2] == "0":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(measure))
                new_name = new_name.replace("radial_gradient", "radial_gradient")
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

            elif sections[-2] == "1":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(measure))
                new_name = new_name.replace("radial_gradient", "radial_intercept")
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

            elif sections[-2] == "2":
                new_name = np.delete(measure, -1)
                new_name = "_".join(list(measure))
                new_name = new_name.replace(
                    "radial_gradient", "radial_gradient_r2_score"
                )
                to_rename.update({intensity_cols[k]: new_name.replace("-", "_")})

    df = df.rename(columns=to_rename)

    return df


def remove_redundant_features(features, reference_features, channel_names=None):
    """

    Remove redundant features from a list of features based on a reference feature list.

    Parameters
    ----------
    features : list
            The list of features to be filtered.
    reference_features : list
            The reference list of features.
    channel_names : list or None, optional
            The list of channel names. If provided, it is used to identify and remove redundant intensity features.
            Default is None.

    Returns
    -------
    list
            The filtered list of features without redundant entries.

    Notes
    -----
    This function removes redundant features from the input list based on a reference list of features. Features that
    appear in the reference list are removed from the input list. Additionally, if the channel_names parameter is provided,
    it is used to identify and remove redundant intensity features. Intensity features that have the same mode (e.g., 'mean',
    'min', 'max') as any of the channel names in the reference list are also removed.

    Examples
    --------
    >>> features = ['area', 'intensity_mean', 'intensity_max', 'eccentricity']
    >>> reference_features = ['area', 'eccentricity']
    >>> filtered_features = remove_redundant_features(features, reference_features)
    >>> filtered_features
    ['intensity_mean', 'intensity_max']

    >>> channel_names = ['brightfield', 'channel1', 'channel2']
    >>> filtered_features = remove_redundant_features(features, reference_features, channel_names)
    >>> filtered_features
    ['area', 'eccentricity']

    """

    new_features = features[:]

    for f in features:

        if f in reference_features:
            new_features.remove(f)

        if ("intensity" in f) and (channel_names is not None):

            mode = f.split("_")[-1]
            pattern = [a + "_" + mode for a in channel_names]

            for p in pattern:
                if p in reference_features:
                    try:
                        new_features.remove(f)
                    except:
                        pass
    return new_features


def remove_trajectory_measurements(
    trajectories,
    column_labels={
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
):
    """
    Clear a measurement table, while keeping the tracking information.

    Parameters
    ----------
    trajectories : pandas.DataFrame
            The measurement table where each line is a cell at a timepoint and each column a tracking feature or measurement.
    column_labels : dict, optional
            The column labels to use in the output DataFrame. Default is {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.


    Returns
    -------
    pandas.DataFrame
            A filtered DataFrame containing only the tracking columns.

    Examples
    --------
    >>> trajectories_df = pd.DataFrame({
    ...     'TRACK_ID': [1, 1, 2],
    ...     'FRAME': [0, 1, 0],
    ...     'POSITION_X': [100, 105, 200],
    ...     'POSITION_Y': [150, 155, 250],
    ...     'area': [10,100,100],  # Additional column to be removed
    ... })
    >>> filtered_df = remove_trajectory_measurements(trajectories_df)
    >>> print(filtered_df)
    #   pd.DataFrame({
    #    'TRACK_ID': [1, 1, 2],
    #    'FRAME': [0, 1, 0],
    #    'POSITION_X': [100, 105, 200],
    #    'POSITION_Y': [150, 155, 250],
    #    })
    """

    tracks = trajectories.copy()

    columns_to_keep = [
        column_labels["track"],
        column_labels["time"],
        column_labels["x"],
        column_labels["y"],
        column_labels["x"] + "_um",
        column_labels["y"] + "_um",
        "class_id",
        "t",
        "state",
        "generation",
        "root",
        "parent",
        "ID",
        "t0",
        "class",
        "status",
        "class_color",
        "status_color",
        "class_firstdetection",
        "t_firstdetection",
        "status_firstdetection",
        "velocity",
    ]
    cols = list(tracks.columns)
    for c in columns_to_keep:
        if c not in cols:
            columns_to_keep.remove(c)

    keep = [x for x in columns_to_keep if x in cols]
    tracks = tracks[keep]

    return tracks


def collapse_trajectories_by_status(
    df,
    status=None,
    projection="mean",
    population="effectors",
    groupby_columns=["position", "TRACK_ID"],
):

    static_columns = [
        "well_index",
        "well_name",
        "pos_name",
        "position",
        "well",
        "status",
        "t0",
        "class",
        "cell_type",
        "concentration",
        "antibody",
        "pharmaceutical_agent",
        "TRACK_ID",
        "position",
        "neighbor_population",
        "reference_population",
        "NEIGHBOR_ID",
        "REFERENCE_ID",
        "FRAME",
    ]

    if status is None or status not in list(df.columns):
        print("invalid status selection...")
        return None

    df = df.dropna(subset=status, ignore_index=True)
    unique_statuses = np.unique(df[status].to_numpy())

    df_sections = []
    for s in unique_statuses:
        subtab = df.loc[df[status] == s, :]
        op = getattr(subtab.groupby(groupby_columns), projection)
        subtab_projected = op(subtab.groupby(groupby_columns))
        frame_duration = subtab.groupby(groupby_columns).size().to_numpy()
        for c in static_columns:
            try:
                subtab_projected[c] = subtab.groupby(groupby_columns)[c].apply(
                    lambda x: x.unique()[0]
                )
            except Exception as e:
                print(e)
                pass
        subtab_projected["duration_in_state"] = frame_duration
        df_sections.append(subtab_projected)

    group_table = pd.concat(df_sections, axis=0, ignore_index=True)
    if population == "pairs":
        for col in [
            "duration_in_state",
            status,
            "neighbor_population",
            "reference_population",
            "NEIGHBOR_ID",
            "REFERENCE_ID",
        ]:
            first_column = group_table.pop(col)
            group_table.insert(0, col, first_column)
    else:
        for col in ["duration_in_state", status, "TRACK_ID"]:
            first_column = group_table.pop(col)
            group_table.insert(0, col, first_column)

    group_table.pop("FRAME")
    group_table = group_table.sort_values(
        by=groupby_columns + [status], ignore_index=True
    )
    group_table = group_table.reset_index(drop=True)

    return group_table


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
