import numpy as np
import os
import subprocess
import json

# TensorFlow imports are lazy-loaded in functions that need them to avoid
# slow import times for modules that don't require TensorFlow.

from celldetective.utils.model_loaders import locate_signal_model
from celldetective.utils.data_loaders import get_position_table, get_position_pickle
from celldetective.tracking import clean_trajectories, interpolate_nan_properties
import matplotlib.pyplot as plt
from natsort import natsorted
from celldetective.utils.color_mappings import color_from_status, color_from_class
from math import floor
from scipy.optimize import curve_fit
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import median_abs_deviation

abs_path = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
)


def analyze_signals(
    trajectories,
    model,
    interpolate_na=True,
    selected_signals=None,
    model_path=None,
    column_labels={
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
    plot_outcome=False,
    output_dir=None,
):
    """
    Analyzes signals from trajectory data using a specified signal detection model and configuration.

    This function preprocesses trajectory data, selects specified signals, and applies a pretrained signal detection
    model to predict classes and times of interest for each trajectory. It supports custom column labeling, interpolation
    of missing values, and plotting of analysis outcomes.

    Parameters
    ----------
    trajectories : pandas.DataFrame
            DataFrame containing trajectory data with columns for track ID, frame, position, and signals.
    model : str
            The name of the signal detection model to be used for analysis.
    interpolate_na : bool, optional
            Whether to interpolate missing values in the trajectories (default is True).
    selected_signals : list of str, optional
            A list of column names from `trajectories` representing the signals to be analyzed. If None, signals will
            be automatically selected based on the model configuration (default is None).
    column_labels : dict, optional
            A dictionary mapping the default column names ('track', 'time', 'x', 'y') to the corresponding column names
            in `trajectories` (default is {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}).
    plot_outcome : bool, optional
            If True, generates and saves a plot of the signal analysis outcome (default is False).
    output_dir : str, optional
            The directory where the outcome plot will be saved. Required if `plot_outcome` is True (default is None).

    Returns
    -------
    pandas.DataFrame
            The input `trajectories` DataFrame with additional columns for predicted classes, times of interest, and
            corresponding colors based on status and class.

    Raises
    ------
    AssertionError
            If the model or its configuration file cannot be located.

    Notes
    -----
    - The function relies on an external model configuration file (`config_input.json`) located in the model's directory.
    - Signal selection and preprocessing are based on the requirements specified in the model's configuration.

    """
    from celldetective.event_detection_models import SignalDetectionModel

    model_path = locate_signal_model(model, path=model_path)
    complete_path = model_path  # +model
    complete_path = rf"{complete_path}"
    model_config_path = os.sep.join([complete_path, "config_input.json"])
    model_config_path = rf"{model_config_path}"
    assert os.path.exists(
        complete_path
    ), f"Model {model} could not be located in folder {model_path}... Abort."
    assert os.path.exists(
        model_config_path
    ), f"Model configuration could not be located in folder {model_path}... Abort."

    available_signals = list(trajectories.columns)
    # print('The available_signals are : ',available_signals)

    f = open(model_config_path)
    config = json.load(f)
    required_signals = config["channels"]
    if "selected_channels" in config:
        selected_signals = config["selected_channels"]
        if np.any([s == "None" for s in selected_signals]):
            trajectories["None"] = 0.0
    model_signal_length = config["model_signal_length"]

    try:
        label = config["label"]
        if label == "":
            label = None
    except:
        label = None

    if selected_signals is None:
        selected_signals = []
        for s in required_signals:
            priority_cols = [a for a in available_signals if a == s]
            second_priority_cols = [
                a for a in available_signals if a.startswith(s) and a != s
            ]
            third_priority_cols = [
                a for a in available_signals if s in a and not a.startswith(s)
            ]
            candidates = priority_cols + second_priority_cols + third_priority_cols
            assert (
                len(candidates) > 0
            ), f"No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort."
            print(
                f"Selecting the first time series among: {candidates} for input requirement {s}..."
            )
            selected_signals.append(candidates[0])
    else:
        assert len(selected_signals) == len(
            required_signals
        ), f"Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort."

    print(f"The following channels will be passed to the model: {selected_signals}")
    trajectories_clean = clean_trajectories(
        trajectories,
        interpolate_na=interpolate_na,
        interpolate_position_gaps=interpolate_na,
        column_labels=column_labels,
    )

    max_signal_size = int(trajectories_clean[column_labels["time"]].max()) + 2
    assert (
        max_signal_size <= model_signal_length
    ), f"The current signals are longer ({max_signal_size}) than the maximum expected input ({model_signal_length}) for this signal analysis model. Abort..."

    tracks = trajectories_clean[column_labels["track"]].unique()
    signals = np.zeros((len(tracks), max_signal_size, len(selected_signals)))

    for i, (tid, group) in enumerate(
        trajectories_clean.groupby(column_labels["track"])
    ):
        frames = group[column_labels["time"]].to_numpy().astype(int)
        for j, col in enumerate(selected_signals):
            signal = group[col].to_numpy()
            signals[i, frames, j] = signal
            signals[i, max(frames) :, j] = signal[-1]

    model = SignalDetectionModel(pretrained=complete_path)
    if not model.pretrained is None:

        classes = model.predict_class(signals)
        times_recast = model.predict_time_of_interest(signals)

        if label is None:
            class_col = "class"
            time_col = "t0"
            status_col = "status"
        else:
            class_col = "class_" + label
            time_col = "t_" + label
            status_col = "status_" + label

        for i, (tid, group) in enumerate(trajectories.groupby(column_labels["track"])):
            indices = group.index
            trajectories.loc[indices, class_col] = classes[i]
            trajectories.loc[indices, time_col] = times_recast[i]
        print("Done.")

        for tid, group in trajectories.groupby(column_labels["track"]):

            indices = group.index
            t0 = group[time_col].to_numpy()[0]
            cclass = group[class_col].to_numpy()[0]
            timeline = group[column_labels["time"]].to_numpy()
            status = np.zeros_like(timeline)
            if t0 > 0:
                status[timeline >= t0] = 1.0
            if cclass == 2:
                status[:] = 2
            if cclass > 2:
                status[:] = 42
            status_color = [color_from_status(s) for s in status]
            class_color = [color_from_class(cclass) for i in range(len(status))]

            trajectories.loc[indices, status_col] = status
            trajectories.loc[indices, "status_color"] = status_color
            trajectories.loc[indices, "class_color"] = class_color

        if plot_outcome:
            fig, ax = plt.subplots(1, len(selected_signals), figsize=(10, 5))
            for i, s in enumerate(selected_signals):
                for k, (tid, group) in enumerate(
                    trajectories.groupby(column_labels["track"])
                ):
                    cclass = group[class_col].to_numpy()[0]
                    t0 = group[time_col].to_numpy()[0]
                    timeline = group[column_labels["time"]].to_numpy()
                    if cclass == 0:
                        if len(selected_signals) > 1:
                            ax[i].plot(
                                timeline - t0,
                                group[s].to_numpy(),
                                c="tab:blue",
                                alpha=0.1,
                            )
                        else:
                            ax.plot(
                                timeline - t0,
                                group[s].to_numpy(),
                                c="tab:blue",
                                alpha=0.1,
                            )
            if len(selected_signals) > 1:
                for a, s in zip(ax, selected_signals):
                    a.set_title(s)
                    a.set_xlabel(r"time - t$_0$ [frame]")
                    a.spines["top"].set_visible(False)
                    a.spines["right"].set_visible(False)
            else:
                ax.set_title(s)
                ax.set_xlabel(r"time - t$_0$ [frame]")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            plt.tight_layout()
            if output_dir is not None:
                plt.savefig(
                    output_dir + "signal_collapse.png", bbox_inches="tight", dpi=300
                )
            plt.pause(3)
            plt.close()

    if "None" in list(trajectories.columns):
        trajectories = trajectories.drop(columns=["None"])
    return trajectories


def analyze_signals_at_position(pos, model, mode, use_gpu=True, return_table=False):
    """
    Analyzes signals for a given position directory using a specified model and mode, with an option to use GPU acceleration.

    This function executes an external Python script to analyze signals within the specified position directory, applying
    a predefined model in a specified mode. It supports GPU acceleration for faster processing. Optionally, the function
    can return the resulting analysis table as a pandas DataFrame.

    Parameters
    ----------
    pos : str
            The file path to the position directory containing the data to be analyzed. The path must be valid and accessible.
    model : str
            The name of the model to use for signal analysis.
    mode : str
            The operation mode specifying how the analysis should be conducted.
    use_gpu : bool, optional
            Specifies whether to use GPU acceleration for the analysis (default is True).
    return_table : bool, optional
            If True, the function returns a pandas DataFrame containing the analysis results (default is False).

    Returns
    -------
    pandas.DataFrame or None
            If `return_table` is True, returns a DataFrame containing the analysis results. Otherwise, returns None.

    Raises
    ------
    AssertionError
            If the specified position path does not exist.

    Notes
    -----
    - The analysis is performed by an external script (`analyze_signals.py`) located in a specific directory relative
      to this function.
    - The results of the analysis are expected to be saved in the "output/tables" subdirectory within the position
      directory, following a naming convention based on the analysis `mode`.

    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    assert os.path.exists(pos), f"Position {pos} is not a valid path."
    if not pos.endswith("/"):
        pos += "/"

    script_path = os.sep.join([abs_path, "scripts", "analyze_signals.py"])
    cmd = f'python "{script_path}" --pos "{pos}" --model "{model}" --mode "{mode}" --use_gpu "{use_gpu}"'
    subprocess.call(cmd, shell=True)

    table = pos + os.sep.join(["output", "tables", f"trajectories_{mode}.csv"])
    if return_table:
        df = pd.read_csv(table)
        return df
    else:
        return None


def analyze_pair_signals_at_position(
    pos, model, use_gpu=True, populations=["targets", "effectors"]
):

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    assert os.path.exists(pos), f"Position {pos} is not a valid path."
    if not pos.endswith("/"):
        pos += "/"

    dataframes = {}
    for pop in populations:
        dataframes.update({pop: get_position_pickle(pos, population=pop)})
    df_pairs = get_position_table(pos, population="pairs")

    # Need to identify expected reference / neighbor tables
    model_path = locate_signal_model(model, pairs=True)
    print(f"Looking for model in {model_path}...")
    complete_path = model_path
    complete_path = rf"{complete_path}"
    model_config_path = os.sep.join([complete_path, "config_input.json"])
    model_config_path = rf"{model_config_path}"
    f = open(model_config_path)
    model_config_path = json.load(f)

    reference_population = model_config_path["reference_population"]
    neighbor_population = model_config_path["neighbor_population"]

    if dataframes[reference_population] is None:
        print(
            f"No tabulated data can be found for the reference population ({reference_population})... Abort..."
        )
        return None

    if dataframes[neighbor_population] is None:
        print(
            f"No tabulated data can be found for the neighbor population ({neighbor_population})... Abort..."
        )
        return None

    df = analyze_pair_signals(
        df_pairs,
        dataframes[reference_population],
        dataframes[neighbor_population],
        model=model,
    )
    table = pos + os.sep.join(["output", "tables", f"trajectories_pairs.csv"])
    df.to_csv(table, index=False)

    return None


def analyze_pair_signals(
    trajectories_pairs,
    trajectories_reference,
    trajectories_neighbors,
    model,
    interpolate_na=True,
    selected_signals=None,
    model_path=None,
    plot_outcome=False,
    output_dir=None,
    column_labels={
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
):
    from celldetective.event_detection_models import SignalDetectionModel

    model_path = locate_signal_model(model, path=model_path, pairs=True)
    print(f"Looking for model in {model_path}...")
    complete_path = model_path
    complete_path = rf"{complete_path}"
    model_config_path = os.sep.join([complete_path, "config_input.json"])
    model_config_path = rf"{model_config_path}"
    assert os.path.exists(
        complete_path
    ), f"Model {model} could not be located in folder {model_path}... Abort."
    assert os.path.exists(
        model_config_path
    ), f"Model configuration could not be located in folder {model_path}... Abort."

    trajectories_pairs = trajectories_pairs.rename(columns=lambda x: "pair_" + x)
    trajectories_reference = trajectories_reference.rename(
        columns=lambda x: "reference_" + x
    )
    trajectories_neighbors = trajectories_neighbors.rename(
        columns=lambda x: "neighbor_" + x
    )

    if "pair_position" in list(trajectories_pairs.columns):
        pair_groupby_cols = ["pair_position", "pair_REFERENCE_ID", "pair_NEIGHBOR_ID"]
    else:
        pair_groupby_cols = ["pair_REFERENCE_ID", "pair_NEIGHBOR_ID"]

    if "reference_position" in list(trajectories_reference.columns):
        reference_groupby_cols = ["reference_position", "reference_TRACK_ID"]
    else:
        reference_groupby_cols = ["reference_TRACK_ID"]

    if "neighbor_position" in list(trajectories_neighbors.columns):
        neighbor_groupby_cols = ["neighbor_position", "neighbor_TRACK_ID"]
    else:
        neighbor_groupby_cols = ["neighbor_TRACK_ID"]

    available_signals = (
        []
    )  # list(trajectories_pairs.columns) + list(trajectories_reference.columns) + list(trajectories_neighbors.columns)
    for col in list(trajectories_pairs.columns):
        if is_numeric_dtype(trajectories_pairs[col]):
            available_signals.append(col)
    for col in list(trajectories_reference.columns):
        if is_numeric_dtype(trajectories_reference[col]):
            available_signals.append(col)
    for col in list(trajectories_neighbors.columns):
        if is_numeric_dtype(trajectories_neighbors[col]):
            available_signals.append(col)

    print("The available signals are : ", available_signals)

    f = open(model_config_path)
    config = json.load(f)
    required_signals = config["channels"]

    try:
        label = config["label"]
        if label == "":
            label = None
    except:
        label = None

    if selected_signals is None:
        selected_signals = []
        for s in required_signals:
            pattern_test = [s in a or s == a for a in available_signals]
            print(f"Pattern test for signal {s}: ", pattern_test)
            assert np.any(
                pattern_test
            ), f"No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort."
            valid_columns = np.array(available_signals)[np.array(pattern_test)]
            if len(valid_columns) == 1:
                selected_signals.append(valid_columns[0])
            else:
                # print(test_number_of_nan(trajectories, valid_columns))
                print(f"Found several candidate signals: {valid_columns}")
                for vc in natsorted(valid_columns):
                    if "circle" in vc:
                        selected_signals.append(vc)
                        break
                else:
                    selected_signals.append(valid_columns[0])
                # do something more complicated in case of one to many columns
                # pass
    else:
        assert len(selected_signals) == len(
            required_signals
        ), f"Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort."

    print(f"The following channels will be passed to the model: {selected_signals}")
    trajectories_reference_clean = interpolate_nan_properties(
        trajectories_reference, track_label=reference_groupby_cols
    )
    trajectories_neighbors_clean = interpolate_nan_properties(
        trajectories_neighbors, track_label=neighbor_groupby_cols
    )
    trajectories_pairs_clean = interpolate_nan_properties(
        trajectories_pairs, track_label=pair_groupby_cols
    )
    print(f"{trajectories_pairs_clean.columns=}")

    max_signal_size = int(trajectories_pairs_clean["pair_FRAME"].max()) + 2
    pair_tracks = trajectories_pairs_clean.groupby(pair_groupby_cols).size()
    signals = np.zeros((len(pair_tracks), max_signal_size, len(selected_signals)))
    print(f"{max_signal_size=} {len(pair_tracks)=} {signals.shape=}")

    for i, (pair, group) in enumerate(
        trajectories_pairs_clean.groupby(pair_groupby_cols)
    ):

        if "pair_position" not in list(trajectories_pairs_clean.columns):
            pos_mode = False
            reference_cell = pair[0]
            neighbor_cell = pair[1]
        else:
            pos_mode = True
            reference_cell = pair[1]
            neighbor_cell = pair[2]
            pos = pair[0]

        if (
            pos_mode
            and "reference_position" in list(trajectories_reference_clean.columns)
            and "neighbor_position" in list(trajectories_neighbors_clean.columns)
        ):
            reference_filter = (
                trajectories_reference_clean["reference_TRACK_ID"] == reference_cell
            ) & (trajectories_reference_clean["reference_position"] == pos)
            neighbor_filter = (
                trajectories_neighbors_clean["neighbor_TRACK_ID"] == neighbor_cell
            ) & (trajectories_neighbors_clean["neighbor_position"] == pos)
        else:
            reference_filter = (
                trajectories_reference_clean["reference_TRACK_ID"] == reference_cell
            )
            neighbor_filter = (
                trajectories_neighbors_clean["neighbor_TRACK_ID"] == neighbor_cell
            )

        pair_frames = group["pair_FRAME"].to_numpy().astype(int)

        for j, col in enumerate(selected_signals):
            if col.startswith("pair_"):
                signal = group[col].to_numpy()
                signals[i, pair_frames, j] = signal
                signals[i, max(pair_frames) :, j] = signal[-1]
            elif col.startswith("reference_"):
                signal = trajectories_reference_clean.loc[
                    reference_filter, col
                ].to_numpy()
                timeline = trajectories_reference_clean.loc[
                    reference_filter, "reference_FRAME"
                ].to_numpy()
                signals[i, timeline, j] = signal
                signals[i, max(timeline) :, j] = signal[-1]
            elif col.startswith("neighbor_"):
                signal = trajectories_neighbors_clean.loc[
                    neighbor_filter, col
                ].to_numpy()
                timeline = trajectories_neighbors_clean.loc[
                    neighbor_filter, "neighbor_FRAME"
                ].to_numpy()
                signals[i, timeline, j] = signal
                signals[i, max(timeline) :, j] = signal[-1]

    model = SignalDetectionModel(pretrained=complete_path)
    print("signal shape: ", signals.shape)

    classes = model.predict_class(signals)
    times_recast = model.predict_time_of_interest(signals)

    if label is None:
        class_col = "pair_class"
        time_col = "pair_t0"
        status_col = "pair_status"
    else:
        class_col = "pair_class_" + label
        time_col = "pair_t_" + label
        status_col = "pair_status_" + label

    for i, (pair, group) in enumerate(trajectories_pairs.groupby(pair_groupby_cols)):
        indices = group.index
        trajectories_pairs.loc[indices, class_col] = classes[i]
        trajectories_pairs.loc[indices, time_col] = times_recast[i]
    print("Done.")

    # At the end rename cols again
    trajectories_pairs = trajectories_pairs.rename(
        columns=lambda x: x.replace("pair_", "")
    )
    trajectories_reference = trajectories_pairs.rename(
        columns=lambda x: x.replace("reference_", "")
    )
    trajectories_neighbors = trajectories_pairs.rename(
        columns=lambda x: x.replace("neighbor_", "")
    )
    invalid_cols = [
        c for c in list(trajectories_pairs.columns) if c.startswith("Unnamed")
    ]
    trajectories_pairs = trajectories_pairs.drop(columns=invalid_cols)

    return trajectories_pairs


def train_signal_model(config):
    """
    Initiates the training of a signal detection model using a specified configuration file.

    This function triggers an external Python script to train a signal detection model. The training
    configuration, including data paths, model parameters, and training options, are specified in a JSON
    configuration file. The function asserts the existence of the configuration file before proceeding
    with the training process.

    Parameters
    ----------
    config : str
            The file path to the JSON configuration file specifying training parameters. This path must be valid
            and the configuration file must be correctly formatted according to the expectations of the
            'train_signal_model.py' script.

    Raises
    ------
    AssertionError
            If the specified configuration file does not exist at the given path.

    Notes
    -----
    - The external training script 'train_signal_model.py' is expected to be located in a predefined directory
      relative to this function and is responsible for the actual model training process.
    - The configuration file should include details such as data directories, model architecture specifications,
      training hyperparameters, and any preprocessing steps required.

    Examples
    --------
    >>> config_path = '/path/to/training_config.json'
    >>> train_signal_model(config_path)
    # This will execute the 'train_signal_model.py' script using the parameters specified in 'training_config.json'.

    """

    config = config.replace("\\", "/")
    config = rf"{config}"
    assert os.path.exists(config), f"Config {config} is not a valid path."

    script_path = os.sep.join([abs_path, "scripts", "train_signal_model.py"])
    cmd = f'python "{script_path}" --config "{config}"'
    subprocess.call(cmd, shell=True)


def T_MSD(x, y, dt):
    """
    Compute the Time-Averaged Mean Square Displacement (T-MSD) of a 2D trajectory.

    Parameters
    ----------
    x : array_like
            The array of x-coordinates of the trajectory.
    y : array_like
            The array of y-coordinates of the trajectory.
    dt : float
            The time interval between successive data points in the trajectory.

    Returns
    -------
    msd : list
            A list containing the Time-Averaged Mean Square Displacement values for different time lags.
    timelag : ndarray
            The array representing the time lags corresponding to the calculated MSD values.

    Notes
    -----
    - T-MSD is a measure of the average spatial extent explored by a particle over a given time interval.
    - The input trajectories (x, y) are assumed to be in the same unit of length.
    - The time interval (dt) should be consistent with the time unit used in the data.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 11])
    >>> y = np.array([0, 3, 5, 8, 10])
    >>> dt = 1.0  # Time interval between data points
    >>> T_MSD(x, y, dt)
    ([6.0, 9.0, 4.666666666666667, 1.6666666666666667],
     array([1., 2., 3., 4.]))

    """

    msd = []
    N = len(x)
    for n in range(1, N):
        s = 0
        for i in range(0, N - n):
            s += (x[n + i] - x[i]) ** 2 + (y[n + i] - y[i]) ** 2
        msd.append(1 / (N - n) * s)

    timelag = np.linspace(dt, (N - 1) * dt, N - 1)
    return msd, timelag


def linear_msd(t, m):
    """
    Function to compute Mean Square Displacement (MSD) with a linear scaling relationship.

    Parameters
    ----------
    t : array_like
            Time lag values.
    m : float
            Linear scaling factor representing the slope of the MSD curve.

    Returns
    -------
    msd : ndarray
            Computed MSD values based on the linear scaling relationship.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([1, 2, 3, 4])
    >>> m = 2.0
    >>> linear_msd(t, m)
    array([2., 4., 6., 8.])

    """

    return m * t


def alpha_msd(t, m, alpha):
    """
    Function to compute Mean Square Displacement (MSD) with a power-law scaling relationship.

    Parameters
    ----------
    t : array_like
            Time lag values.
    m : float
            Scaling factor.
    alpha : float
            Exponent representing the scaling relationship between MSD and time.

    Returns
    -------
    msd : ndarray
            Computed MSD values based on the power-law scaling relationship.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([1, 2, 3, 4])
    >>> m = 2.0
    >>> alpha = 0.5
    >>> alpha_msd(t, m, alpha)
    array([2.        , 4.        , 6.        , 8.        ])

    """

    return m * t**alpha


def sliding_msd(
    x, y, timeline, window, mode="bi", n_points_migration=7, n_points_transport=7
):
    """
    Compute sliding mean square displacement (sMSD) and anomalous exponent (alpha) for a 2D trajectory using a sliding window approach.

    Parameters
    ----------
    x : array_like
            The array of x-coordinates of the trajectory.
    y : array_like
            The array of y-coordinates of the trajectory.
    timeline : array_like
            The array representing the time points corresponding to the x and y coordinates.
    window : int
            The size of the sliding window used for computing local MSD and alpha values.
    mode : {'bi', 'forward', 'backward'}, optional
            The sliding window mode:
            - 'bi' (default): Bidirectional sliding window.
            - 'forward': Forward sliding window.
            - 'backward': Backward sliding window.
    n_points_migration : int, optional
            The number of points used for fitting the linear function in the MSD calculation.
    n_points_transport : int, optional
            The number of points used for fitting the alpha function in the anomalous exponent calculation.

    Returns
    -------
    s_msd : ndarray
            Sliding Mean Square Displacement values calculated using the sliding window approach.
    s_alpha : ndarray
            Sliding anomalous exponent (alpha) values calculated using the sliding window approach.

    Raises
    ------
    AssertionError
            If the window size is not larger than the number of fit points.

    Notes
    -----
    - The input trajectories (x, y) are assumed to be in the same unit of length.
    - The time unit used in the data should be consistent with the time intervals in the timeline array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 11, 15, 20])
    >>> y = np.array([0, 3, 5, 8, 10, 14, 18])
    >>> timeline = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> window = 3
    >>> s_msd, s_alpha = sliding_msd(x, y, timeline, window, n_points_migration=2, n_points_transport=3)

    """

    assert (
        window > n_points_migration
    ), "Please set a window larger than the number of fit points..."

    # modes = bi, forward, backward
    s_msd = np.zeros(len(x))
    s_msd[:] = np.nan
    s_alpha = np.zeros(len(x))
    s_alpha[:] = np.nan
    dt = timeline[1] - timeline[0]

    if mode == "bi":
        assert window % 2 == 1, "Please set an odd window for the bidirectional mode"
        lower_bound = window // 2
        upper_bound = len(x) - window // 2 - 1
    elif mode == "forward":
        lower_bound = 0
        upper_bound = len(x) - window
    elif mode == "backward":
        lower_bound = window
        upper_bound = len(x)

    for t in range(lower_bound, upper_bound):
        if mode == "bi":
            x_sub = x[t - window // 2 : t + window // 2 + 1]
            y_sub = y[t - window // 2 : t + window // 2 + 1]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t+window//2+1] - x[t-window//2]) / (timeline[t+window//2+1] - timeline[t-window//2])
        elif mode == "forward":
            x_sub = x[t : t + window]
            y_sub = y[t : t + window]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
        elif mode == "backward":
            x_sub = x[t - window : t]
            y_sub = y[t - window : t]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])
        popt, pcov = curve_fit(
            linear_msd, timelag[:n_points_migration], msd[:n_points_migration]
        )
        s_msd[t] = popt[0]
        popt_alpha, pcov_alpha = curve_fit(
            alpha_msd, timelag[:n_points_transport], msd[:n_points_transport]
        )
        s_alpha[t] = popt_alpha[1]

    return s_msd, s_alpha


def drift_msd(t, d, v):
    """
    Calculates the mean squared displacement (MSD) of a particle undergoing diffusion with drift.

    The function computes the MSD for a particle that diffuses in a medium with a constant drift velocity.
    The MSD is given by the formula: MSD = 4Dt + V^2t^2, where D is the diffusion coefficient, V is the drift
    velocity, and t is the time.

    Parameters
    ----------
    t : float or ndarray
            Time or an array of time points at which to calculate the MSD.
    d : float
            Diffusion coefficient of the particle.
    v : float
            Drift velocity of the particle.

    Returns
    -------
    float or ndarray
            The mean squared displacement of the particle at time t. Returns a single float value if t is a float,
            or returns an array of MSD values if t is an ndarray.

    Examples
    --------
    >>> drift_msd(t=5, d=1, v=2)
    40
    >>> drift_msd(t=np.array([1, 2, 3]), d=1, v=2)
    array([ 6, 16, 30])

    Notes
    -----
    - This formula assumes that the particle undergoes normal diffusion with an additional constant drift component.
    - The function can be used to model the behavior of particles in systems where both diffusion and directed motion occur.
    """

    return 4 * d * t + v**2 * t**2


def sliding_msd_drift(
    x,
    y,
    timeline,
    window,
    mode="bi",
    n_points_migration=7,
    n_points_transport=7,
    r2_threshold=0.75,
):
    """
    Computes the sliding mean squared displacement (MSD) with drift for particle trajectories.

    This function calculates the diffusion coefficient and drift velocity of particles based on their
    x and y positions over time. It uses a sliding window approach to estimate the MSD at each point in time,
    fitting the MSD to the equation MSD = 4Dt + V^2t^2 to extract the diffusion coefficient (D) and drift velocity (V).

    Parameters
    ----------
    x : ndarray
            The x positions of the particle over time.
    y : ndarray
            The y positions of the particle over time.
    timeline : ndarray
            The time points corresponding to the x and y positions.
    window : int
            The size of the sliding window used to calculate the MSD at each point in time.
    mode : str, optional
            The mode of sliding window calculation. Options are 'bi' for bidirectional, 'forward', or 'backward'. Default is 'bi'.
    n_points_migration : int, optional
            The number of initial points from the calculated MSD to use for fitting the migration model. Default is 7.
    n_points_transport : int, optional
            The number of initial points from the calculated MSD to use for fitting the transport model. Default is 7.
    r2_threshold : float, optional
            The R-squared threshold used to validate the fit. Default is 0.75.

    Returns
    -------
    tuple
            A tuple containing two ndarrays: the estimated diffusion coefficients and drift velocities for each point in time.

    Raises
    ------
    AssertionError
            If the window size is not larger than the number of fit points or if the window size is even when mode is 'bi'.

    Notes
    -----
    - The function assumes a uniform time step between each point in the timeline.
    - The 'bi' mode requires an odd-sized window to symmetrically calculate the MSD around each point in time.
    - The curve fitting is performed using the `curve_fit` function from `scipy.optimize`, fitting to the `drift_msd` model.

    Examples
    --------
    >>> x = np.random.rand(100)
    >>> y = np.random.rand(100)
    >>> timeline = np.arange(100)
    >>> window = 11
    >>> diffusion, velocity = sliding_msd_drift(x, y, timeline, window, mode='bi')
    # Calculates the diffusion coefficient and drift velocity using a bidirectional sliding window.

    """

    assert (
        window > n_points_migration
    ), "Please set a window larger than the number of fit points..."

    # modes = bi, forward, backward
    s_diffusion = np.zeros(len(x))
    s_diffusion[:] = np.nan
    s_velocity = np.zeros(len(x))
    s_velocity[:] = np.nan
    dt = timeline[1] - timeline[0]

    if mode == "bi":
        assert window % 2 == 1, "Please set an odd window for the bidirectional mode"
        lower_bound = window // 2
        upper_bound = len(x) - window // 2 - 1
    elif mode == "forward":
        lower_bound = 0
        upper_bound = len(x) - window
    elif mode == "backward":
        lower_bound = window
        upper_bound = len(x)

    for t in range(lower_bound, upper_bound):
        if mode == "bi":
            x_sub = x[t - window // 2 : t + window // 2 + 1]
            y_sub = y[t - window // 2 : t + window // 2 + 1]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t+window//2+1] - x[t-window//2]) / (timeline[t+window//2+1] - timeline[t-window//2])
        elif mode == "forward":
            x_sub = x[t : t + window]
            y_sub = y[t : t + window]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
        elif mode == "backward":
            x_sub = x[t - window : t]
            y_sub = y[t - window : t]
            msd, timelag = T_MSD(x_sub, y_sub, dt)
            # dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])

        popt, pcov = curve_fit(
            drift_msd, timelag[:n_points_migration], msd[:n_points_migration]
        )
        # if not np.any([math.isinf(a) for a in pcov.flatten()]):
        s_diffusion[t] = popt[0]
        s_velocity[t] = popt[1]

    return s_diffusion, s_velocity


def columnwise_mean(matrix, min_nbr_values=1, projection="mean"):
    """
    Calculate the column-wise mean and standard deviation of non-NaN elements in the input matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
            The input matrix for which column-wise mean and standard deviation are calculated.
    min_nbr_values : int, optional
            The minimum number of non-NaN values required in a column to calculate mean and standard deviation.
            Default is 8.

    Returns
    -------
    mean_line : numpy.ndarray
            An array containing the column-wise mean of non-NaN elements. Elements with fewer than `min_nbr_values` non-NaN
            values are replaced with NaN.
    mean_line_std : numpy.ndarray
            An array containing the column-wise standard deviation of non-NaN elements. Elements with fewer than `min_nbr_values`
            non-NaN values are replaced with NaN.

    Notes
    -----
    1. This function calculates the mean and standard deviation of non-NaN elements in each column of the input matrix.
    2. Columns with fewer than `min_nbr_values` non-zero elements will have NaN as the mean and standard deviation.
    3. NaN values in the input matrix are ignored during calculation.

    """

    mean_line = np.zeros(matrix.shape[1])
    mean_line[:] = np.nan
    mean_line_std = np.zeros(matrix.shape[1])
    mean_line_std[:] = np.nan

    for k in range(matrix.shape[1]):
        values = matrix[:, k]
        values = values[values == values]
        if len(values[values == values]) > min_nbr_values:
            if projection == "mean":
                mean_line[k] = np.nanmean(values)
                mean_line_std[k] = np.nanstd(values)
            elif projection == "median":
                mean_line[k] = np.nanmedian(values)
                mean_line_std[k] = median_abs_deviation(
                    values, center=np.nanmedian, nan_policy="omit"
                )
    return mean_line, mean_line_std


def mean_signal(
    df,
    signal_name,
    class_col,
    time_col=None,
    class_value=[0],
    return_matrix=False,
    forced_max_duration=None,
    min_nbr_values=2,
    conflict_mode="mean",
    projection="mean",
    pairs=False,
):
    """
    Calculate the mean and standard deviation of a specified signal for tracks of a given class in the input DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
            Input DataFrame containing tracking data.
    signal_name : str
            Name of the signal (column) in the DataFrame for which mean and standard deviation are calculated.
    class_col : str
            Name of the column in the DataFrame containing class labels.
    time_col : str, optional
            Name of the column in the DataFrame containing time information. Default is None.
    class_value : int, optional
            Value representing the class of interest. Default is 0.

    Returns
    -------
    mean_signal : numpy.ndarray
            An array containing the mean signal values for tracks of the specified class. Tracks with class not equal to
            `class_value` are excluded from the calculation.
    std_signal : numpy.ndarray
            An array containing the standard deviation of signal values for tracks of the specified class. Tracks with class
            not equal to `class_value` are excluded from the calculation.
    actual_timeline : numpy.ndarray
            An array representing the time points corresponding to the mean signal values.

    Notes
    -----
    1. This function calculates the mean and standard deviation of the specified signal for tracks of a given class.
    2. Tracks with class not equal to `class_value` are excluded from the calculation.
    3. Tracks with missing or NaN values in the specified signal are ignored during calculation.
    4. Tracks are aligned based on their 'FRAME' values and the specified `time_col` (if provided).

    """

    assert signal_name in list(
        df.columns
    ), "The signal you want to plot is not one of the measured features."
    if isinstance(class_value, int):
        class_value = [class_value]
    elif class_value is None or class_col is None:
        class_col = "class_temp"
        df["class_temp"] = 1
        class_value = [1]

    if forced_max_duration is None:
        max_duration = (
            int(df["FRAME"].max()) + 1
        )  # ceil(np.amax(df.groupby(['position','TRACK_ID']).size().values))
    else:
        max_duration = forced_max_duration

    abs_time = False
    if isinstance(time_col, (int, float)):
        abs_time = True

    if not pairs:
        groupby_cols = ["position", "TRACK_ID"]
    else:
        groupby_cols = ["position", "REFERENCE_ID", "NEIGHBOR_ID"]

    n_tracks = len(df.groupby(groupby_cols))
    signal_matrix = np.zeros((n_tracks, int(max_duration) * 2 + 1))
    signal_matrix[:, :] = np.nan

    df = df.sort_values(by=groupby_cols + ["FRAME"])

    trackid = 0
    for track, track_group in df.loc[df[class_col].isin(class_value)].groupby(
        groupby_cols
    ):
        cclass = track_group[class_col].to_numpy()[0]
        if cclass != 0:
            ref_time = 0
            if abs_time:
                ref_time = time_col
        else:
            if not abs_time:
                try:
                    ref_time = floor(track_group[time_col].to_numpy()[0])
                except:
                    continue
            else:
                ref_time = time_col
        if conflict_mode == "mean":
            signal = track_group.groupby("FRAME")[signal_name].mean().to_numpy()
        elif conflict_mode == "first":
            signal = track_group.groupby("FRAME")[signal_name].first().to_numpy()
        else:
            signal = track_group[signal_name].to_numpy()

        if ref_time <= 0:
            ref_time = 0

        timeline = track_group["FRAME"].unique().astype(int)
        timeline_shifted = timeline - ref_time + max_duration
        signal_matrix[trackid, timeline_shifted.astype(int)] = signal
        trackid += 1

    mean_signal, std_signal = columnwise_mean(
        signal_matrix, min_nbr_values=min_nbr_values, projection=projection
    )
    actual_timeline = np.linspace(-max_duration, max_duration, 2 * max_duration + 1)
    if return_matrix:
        return mean_signal, std_signal, actual_timeline, signal_matrix
    else:
        return mean_signal, std_signal, actual_timeline


# if __name__ == "__main__":

# 	# model = MultiScaleResNetModel(3, n_classes = 3, dropout_rate=0, dense_collection=1024, header="classifier", model_signal_length = 128)
# 	# print(model.summary())
# 	model = ResNetModelCurrent(1, 2, depth=2, use_pooling=True, n_classes = 3, dropout_rate=0.1, dense_collection=512,
# 					   header="classifier", model_signal_length = 128)
# 	print(model.summary())
# 	#plot_model(model, to_file='test.png', show_shapes=True)
