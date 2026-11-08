"""
Signals Module
==============

This module is dedicated to the extraction, processing, and analysis of biological signals from tracked objects.
It includes capabilities for signal measurement, feature computation, and event prediction using machine learning models.

Key Features
------------
-   **Signal Extraction**: Retrieves intensity values and other measurements from image channels based on object masks.
-   **Signal Analysis**: Computes signal derivatives, smoothing, and other temporal features.
-   **Event Prediction**: Uses trained models to detect biological events (e.g., apoptosis, division) from signal time-series.

Main Functions
--------------
-   `extract_signals`: Extracts raw signal data from images for tracked objects.
-   `predict_signals`: Applies pre-trained models to classify signals and detect events.
-   `compute_signal_features`: Calculates statistical and temporal features from raw signals.

Dependencies
------------
-   `tensorflow`/`keras`: Required for running event detection models.
-   `scikit-learn`: Used for metric calculation and some processing steps.
"""

import os
import subprocess
import sys
import json
import numpy as np
from celldetective.utils.model_loaders import locate_signal_model, _resolve_signal_model_paths
from celldetective.utils.data_loaders import get_position_table, get_position_pickle
from celldetective.tracking import clean_trajectories, interpolate_nan_properties
import matplotlib.pyplot as plt
from natsort import natsorted
from celldetective.utils import COLUMN_LABELS
from celldetective.utils.event_status import (
    event_class_of_label,
    event_columns,
    write_resolved_event_status,
)
from celldetective.utils.signal_windowing import predict_events_sliding_window
from math import floor
from scipy.optimize import curve_fit
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import median_abs_deviation
from typing import Callable, List, Optional, Union, Dict, Tuple, Literal
import logging

logger = logging.getLogger("celldetective")

abs_path = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
)



def _extract_config_label(config: dict) -> Optional[str]:
    """Return the ``label`` field from a signal model config, or None if absent or empty.

    Parameters
    ----------
    config : dict
        Loaded ``config_input.json`` dictionary.

    Returns
    -------
    str or None
    """
    try:
        label = config["label"]
        return None if label == "" else label
    except KeyError:
        return None


def _select_model_signals(
    available_signals: List[str],
    required_signals: List[str],
    selected_signals: Optional[List[str]] = None,
) -> List[str]:
    """
    The columns passed to a model, one per required signal. Without an explicit
    choice, the first column that matches the name of the required signal is
    taken: an exact match, then a column starting with it, then one containing it.
    """

    if selected_signals is not None:
        if len(selected_signals) != len(required_signals):
            raise ValueError(f"Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort.")
        return list(selected_signals)

    selected = []
    for s in required_signals:
        priority_cols = [a for a in available_signals if a == s]
        second_priority_cols = [a for a in available_signals if a.startswith(s) and a != s]
        third_priority_cols = [a for a in available_signals if s in a and not a.startswith(s)]
        candidates = priority_cols + second_priority_cols + third_priority_cols
        if len(candidates) == 0:
            raise ValueError(f"No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort.")
        if candidates[0] != s:
            logger.warning(
                f"No column is named {s!r}, which the model reads: {candidates[0]!r} is "
                f"used, the first of {candidates}. Map the channels explicitly if it "
                f"is not the right measurement."
            )
        selected.append(candidates[0])
    return selected


def _log_unobserved(n_unobserved: int, what: str) -> None:
    """Log how many tracks (or pairs) were made class 2 for an unobserved event."""

    if n_unobserved:
        logger.info(
            f"{n_unobserved} {what}(s) with an event at or before their first "
            f"observed frame were made class 2: the change was not observed."
        )


def warn_on_frame_interval_mismatch(config: dict, frame_to_min: Optional[float]) -> None:
    """
    Warn when the experiment's frame interval differs from the one the model was
    trained at (``frame_to_min`` in its config). A model sees frames, not
    minutes, so at another frame rate its event times and detections can be off
    without error. Models trained before the interval was recorded are not checked.
    """

    trained = config.get("frame_to_min")
    if trained is None or frame_to_min is None:
        return
    trained = trained if isinstance(trained, list) else [trained]
    if not any(np.isclose(frame_to_min, t, rtol=0.01) for t in trained):
        logger.warning(
            f"This model was trained on movies with {trained} min between frames, "
            f"and this experiment has {frame_to_min} min. Its event times and "
            f"detections may be unreliable: check them, or retrain the model."
        )


def experiment_frame_interval(pos: str) -> Optional[float]:
    """The time between frames, in minutes, of the experiment of a position, or
    None if it cannot be read."""

    from celldetective.utils.experiment import (
        extract_experiment_from_position,
        get_temporal_calibration,
    )

    try:
        return get_temporal_calibration(extract_experiment_from_position(pos))
    except Exception as e:
        logger.debug(f"No frame interval for {pos}: {e}")
        return None


def event_columns_of_model(config: dict) -> Tuple[str, str, str]:
    """The class, time and status columns a model writes, from its label."""

    class_col = event_class_of_label(_extract_config_label(config))
    return (class_col, *event_columns(class_col))


def _predict_events(model, signals: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Class and time of each signal, scanning with sliding windows when the
    signals are longer than the model window."""

    if signals.shape[1] > window:
        logger.info(
            f"Signals are longer than the model window ({signals.shape[1]} > "
            f"{window} frames): scanning each track with overlapping windows and "
            f"keeping the earliest detected event."
        )
        return predict_events_sliding_window(model, signals, window)
    classes = model.predict_class(signals)
    return classes, model.predict_time_of_interest(signals)


def detect_events_with_model(
    trajectories: pd.DataFrame,
    model,
    selected_signals: Optional[List[str]] = None,
    column_labels: Optional[Dict[str, str]] = None,
    interpolate_na: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
    frame_to_min: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[str], Tuple[str, str, str]]:
    """
    Detect events in single-cell tracks with a loaded signal detection model.

    Shared by `analyze_signals` and the signal analysis process. The signals are
    cleaned, laid out on the frame axis, predicted (with sliding windows when the
    tracks are longer than the model window) and written back by track ID.

    Parameters
    ----------
    trajectories : pandas.DataFrame
        The tracks, with the measurements the model reads.
    model : SignalDetectionModel
        A loaded model, with its ``config``.
    selected_signals : list of str, optional
        The columns to pass to the model, one per required signal. By default,
        the model's ``selected_channels`` if it has some, otherwise the columns
        matching the required signals by name.
    column_labels : dict, optional
        The track, time and position columns, `COLUMN_LABELS` by default.
    interpolate_na : bool, optional
        Whether to interpolate missing values and position gaps before prediction.
    progress : callable, optional
        Called as ``progress(i, n)`` after the signals of track ``i`` of ``n`` are read.
    frame_to_min : float, optional
        The experiment's time between frames, in minutes. When given, a warning
        is logged if the model was trained at another one.

    Returns
    -------
    (pandas.DataFrame, list of str, (str, str, str))
        The tracks with the class, time and status columns, the columns passed to
        the model, and the names of the class, time and status columns. A track
        removed by cleaning gets no class (NaN) and time -1.
    """

    column_labels = column_labels or COLUMN_LABELS
    config = model.config
    warn_on_frame_interval_mismatch(config, frame_to_min)
    required_signals = config["channels"]
    model_signal_length = config["model_signal_length"]

    added_none = False
    if "selected_channels" in config:
        # The model's own choice of columns takes precedence, as before.
        selected_signals = config["selected_channels"]
    if selected_signals is not None and any(s == "None" for s in selected_signals):
        if "None" not in trajectories.columns:
            trajectories["None"] = 0.0
            added_none = True
    selected_signals = _select_model_signals(
        list(trajectories.columns), required_signals, selected_signals
    )
    logger.info(f"The following channels will be passed to the model: {selected_signals}")

    track_col, time_col_name = column_labels["track"], column_labels["time"]
    trajectories_clean = clean_trajectories(
        trajectories,
        interpolate_na=interpolate_na,
        interpolate_position_gaps=interpolate_na,
        column_labels=column_labels,
    )

    max_signal_size = int(trajectories_clean[time_col_name].max()) + 2
    # Unique columns: a model may read the same column twice (several "None"
    # channels), which a duplicated label would widen on selection.
    read_cols = list(dict.fromkeys([track_col, time_col_name, *selected_signals]))
    groups = trajectories_clean[read_cols].groupby(track_col)
    tracks = list(groups.groups)
    signals = np.zeros((len(tracks), max_signal_size, len(selected_signals)))
    for i, (tid, group) in enumerate(groups):
        frames = group[time_col_name].to_numpy().astype(int)
        values = group[selected_signals].to_numpy()
        signals[i, frames, :] = values
        signals[i, frames.max() :, :] = values[-1]
        if progress is not None:
            progress(i + 1, len(tracks))

    classes, times = _predict_events(model, signals, model_signal_length)

    class_col, time_col, status_col = event_columns_of_model(config)
    class_of = dict(zip(tracks, np.asarray(classes, dtype=float)))
    time_of = dict(zip(tracks, np.asarray(times, dtype=float)))
    track_ids = trajectories[track_col]
    trajectories[class_col] = track_ids.map(class_of)
    trajectories[time_col] = track_ids.map(time_of).fillna(-1.0)
    n_missing = trajectories.loc[trajectories[class_col].isna(), track_col].nunique()
    if n_missing:
        logger.warning(f"{n_missing} track(s) removed by cleaning were left without a class.")
    _log_unobserved(
        write_resolved_event_status(
            trajectories,
            class_col,
            time_col,
            status_col,
            keys=track_col,
            frame_name=time_col_name,
        ),
        "track",
    )

    if added_none:
        trajectories = trajectories.drop(columns=["None"])
    return trajectories, selected_signals, (class_col, time_col, status_col)


def analyze_signals(
    trajectories: pd.DataFrame,
    model: str,
    interpolate_na: bool = True,
    selected_signals: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
    plot_outcome: bool = False,
    output_dir: Optional[str] = None,
    frame_to_min: Optional[float] = None,
) -> pd.DataFrame:
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
    frame_to_min : float, optional
            The experiment's time between frames, in minutes. When given, a warning is logged if the model was trained
            at another one (default is None).
    model_path : str, optional
            Path to the model directory. If None, it uses the default model location. Default is None.

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

    complete_path, _ = _resolve_signal_model_paths(model, path=model_path)

    model = SignalDetectionModel(pretrained=complete_path)
    if not model.pretrained is None:

        trajectories, selected_signals, (class_col, time_col, _) = (
            detect_events_with_model(
                trajectories,
                model,
                selected_signals=selected_signals,
                column_labels=column_labels,
                interpolate_na=interpolate_na,
                frame_to_min=frame_to_min,
            )
        )
        logger.info("Signal analysis done.")

        if plot_outcome:
            fig, ax = plt.subplots(1, len(selected_signals), figsize=(10, 5))
            for i, s in enumerate(selected_signals):
                if s not in trajectories.columns:
                    # An empty channel of the model ("None"), dropped once read.
                    continue
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

    return trajectories


def analyze_signals_at_position(
    pos: str,
    model: str,
    mode: str,
    use_gpu: bool = True,
    return_table: bool = False,
) -> Optional[pd.DataFrame]:
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
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"

    script_path = os.sep.join([abs_path, "scripts", "analyze_signals.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--pos", pos, "--model", model, "--mode", mode, "--use_gpu", str(use_gpu)],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Signal analysis script exited with code {result.returncode} for position {pos}.")
        raise RuntimeError(f"Signal analysis failed for position {pos} (exit code {result.returncode}).")

    table = pos + os.sep.join(["output", "tables", f"trajectories_{mode}.csv"])
    if return_table:
        df = pd.read_csv(table)
        return df
    else:
        return None


def analyze_pair_signals_at_position(
    pos: str,
    model: str,
    use_gpu: bool = True,
    populations: List[str] = ["targets", "effectors"],
) -> Optional[str]:
    """
    Analyzes pair signals at a position using a specified model.

    Parameters
    ----------
    pos : str
        Path to the experimental position.
    model : str
        Name of the signal model to use.
    use_gpu : bool, optional
        Whether to use GPU acceleration. Default is True.
    populations : list of str, optional
        List of population names involved in the pair analysis. Default is ["targets", "effectors"].

    Returns
    -------
    str or None
        The path of the pair table written, or None if a table it needs is missing.
    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"

    dataframes = {}
    for pop in populations:
        dataframes.update({pop: get_position_pickle(pos, population=pop)})
    df_pairs = get_position_table(pos, population="pairs")
    
    if df_pairs is None:
        logger.error(f"No tabulated data can be found for the pair population in {pos}... Abort...")
        return None

    # Need to identify expected reference / neighbor tables
    complete_path, model_config_path = _resolve_signal_model_paths(model, pairs=True)
    with open(model_config_path) as f:
        model_config = json.load(f)

    reference_population = model_config["reference_population"]
    neighbor_population = model_config["neighbor_population"]

    if dataframes[reference_population] is None:
        logger.error(
            f"No tabulated data can be found for the reference population ({reference_population})... Abort..."
        )
        return None

    if dataframes[neighbor_population] is None:
        logger.error(
            f"No tabulated data can be found for the neighbor population ({neighbor_population})... Abort..."
        )
        return None

    df = analyze_pair_signals(
        df_pairs,
        dataframes[reference_population],
        dataframes[neighbor_population],
        model=model,
        frame_to_min=experiment_frame_interval(pos),
    )
    table = pos + os.sep.join(["output", "tables", f"trajectories_pairs.csv"])
    df.to_csv(table, index=False)

    return table


def analyze_pair_signals(
    trajectories_pairs: pd.DataFrame,
    trajectories_reference: pd.DataFrame,
    trajectories_neighbors: pd.DataFrame,
    model: str,
    interpolate_na: bool = True,
    selected_signals: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    plot_outcome: bool = False,
    output_dir: Optional[str] = None,
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
    frame_to_min: Optional[float] = None,
) -> pd.DataFrame:
    """
    Analyzes signals for pairs of cells using a specified model.

    Parameters
    ----------
    trajectories_pairs : pandas.DataFrame
        DataFrame containing pair data.
    trajectories_reference : pandas.DataFrame
        DataFrame containing reference population data.
    trajectories_neighbors : pandas.DataFrame
        DataFrame containing neighbor population data.
    model : str
        Name of the signal model to use.
    interpolate_na : bool, optional
        Whether to interpolate NaN values. Default is True.
    selected_signals : list of str, optional
        List of signal columns to analyze. Default is None.
    model_path : str, optional
        Path to the model directory. Default is None.
    plot_outcome : bool, optional
        Whether to interpolate NaN values. Default is True.
    output_dir : str, optional
        Directory to save output plots. Default is None.
    column_labels : dict, optional
        Dictionary mapping column names.

    Returns
    -------
    pandas.DataFrame
        DataFrame with analyzed pair signals and classifications.
    """
    from celldetective.event_detection_models import SignalDetectionModel

    complete_path, model_config_path = _resolve_signal_model_paths(
        model, path=model_path, pairs=True
    )

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

    logger.debug(f"The available signals are: {available_signals}")

    with open(model_config_path) as f:
        config = json.load(f)
    warn_on_frame_interval_mismatch(config, frame_to_min)
    required_signals = config["channels"]

    if selected_signals is None:
        selected_signals = []
        for s in required_signals:
            pattern_test = [s in a or s == a for a in available_signals]
            logger.debug(f"Pattern test for signal {s}: {pattern_test}")
            if not np.any(pattern_test):
                raise ValueError(f"No signal matches with the requirements of the model {required_signals}. Please pass the signals manually with the argument selected_signals or add measurements. Abort.")
            valid_columns = np.array(available_signals)[np.array(pattern_test)]
            if len(valid_columns) == 1:
                selected_signals.append(valid_columns[0])
            else:
                logger.debug(f"Found several candidate signals: {valid_columns}")
                for vc in natsorted(valid_columns):
                    if "circle" in vc:
                        selected_signals.append(vc)
                        break
                else:
                    selected_signals.append(valid_columns[0])
                # do something more complicated in case of one to many columns
                # pass
    else:
        if len(selected_signals) != len(required_signals):
            raise ValueError(f"Mismatch between the number of required signals {required_signals} and the provided signals {selected_signals}... Abort.")

    logger.info(f"The following channels will be passed to the model: {selected_signals}")
    trajectories_reference_clean = interpolate_nan_properties(
        trajectories_reference, track_label=reference_groupby_cols
    )
    trajectories_neighbors_clean = interpolate_nan_properties(
        trajectories_neighbors, track_label=neighbor_groupby_cols
    )
    trajectories_pairs_clean = interpolate_nan_properties(
        trajectories_pairs, track_label=pair_groupby_cols
    )
    logger.debug(f"Pair table columns: {list(trajectories_pairs_clean.columns)}")

    max_pair = int(trajectories_pairs_clean["pair_FRAME"].max()) if len(trajectories_pairs_clean) > 0 else 0
    max_ref = int(trajectories_reference_clean["reference_FRAME"].max()) if len(trajectories_reference_clean) > 0 else 0
    max_neigh = int(trajectories_neighbors_clean["neighbor_FRAME"].max()) if len(trajectories_neighbors_clean) > 0 else 0
    max_signal_size = max(max_pair, max_ref, max_neigh) + 2
    model_signal_length = config.get("model_signal_length", max_signal_size)

    pair_tracks = trajectories_pairs_clean.groupby(pair_groupby_cols).size()
    signals = np.zeros((len(pair_tracks), max_signal_size, len(selected_signals)))
    logger.debug(f"max_signal_size={max_signal_size}, n_pair_tracks={len(pair_tracks)}, signals_shape={signals.shape}")

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
                if len(pair_frames) > 0:
                    signals[i, pair_frames, j] = signal
                    signals[i, int(max(pair_frames)) :, j] = signal[-1]
            elif col.startswith("reference_"):
                signal = trajectories_reference_clean.loc[
                    reference_filter, col
                ].to_numpy()
                timeline = trajectories_reference_clean.loc[
                    reference_filter, "reference_FRAME"
                ].to_numpy().astype(int)
                if len(timeline) > 0:
                    signals[i, timeline, j] = signal
                    signals[i, int(max(timeline)) :, j] = signal[-1]
            elif col.startswith("neighbor_"):
                signal = trajectories_neighbors_clean.loc[
                    neighbor_filter, col
                ].to_numpy()
                timeline = trajectories_neighbors_clean.loc[
                    neighbor_filter, "neighbor_FRAME"
                ].to_numpy().astype(int)
                if len(timeline) > 0:
                    signals[i, timeline, j] = signal
                    signals[i, int(max(timeline)) :, j] = signal[-1]

    model = SignalDetectionModel(pretrained=complete_path)
    logger.debug(f"Signal shape: {signals.shape}")

    classes, times_recast = _predict_events(model, signals, model_signal_length)

    class_col, time_col, status_col = event_columns_of_model(config)

    # Results go to their pair by key, as the signals were built from the
    # cleaned table.
    results = pd.DataFrame(
        {"class": classes, "time": times_recast}, index=pair_tracks.index
    ).reindex(pd.MultiIndex.from_frame(trajectories_pairs[pair_groupby_cols]))
    trajectories_pairs["pair_" + class_col] = results["class"].to_numpy(dtype=float)
    trajectories_pairs["pair_" + time_col] = (
        results["time"].fillna(-1.0).to_numpy(dtype=float)
    )
    logger.info("Pair signal analysis done.")

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

    # The status follows from class and time, as for single cells.
    keys = [c.replace("pair_", "") for c in pair_groupby_cols] + [
        c for c in ("reference_population", "neighbor_population")
        if c in trajectories_pairs.columns
    ]
    _log_unobserved(
        write_resolved_event_status(
            trajectories_pairs, class_col, time_col, status_col, keys=keys
        ),
        "pair",
    )

    return trajectories_pairs


def train_signal_model(config: str) -> None:
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
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config {config} is not a valid path.")

    script_path = os.sep.join([abs_path, "scripts", "train_signal_model.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--config", config],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Signal model training script exited with code {result.returncode}.")
        raise RuntimeError(f"Signal model training failed (exit code {result.returncode}).")


def T_MSD(
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    dt: float,
) -> Tuple[List[float], np.ndarray]:
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


def linear_msd(t: Union[np.ndarray, List[float]], m: float) -> Union[np.ndarray, float]:
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


def alpha_msd(
    t: Union[np.ndarray, List[float]], m: float, alpha: float
) -> Union[np.ndarray, float]:
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
    x: Union[np.ndarray, List[float]],
    y: Union[np.ndarray, List[float]],
    timeline: Union[np.ndarray, List[float]],
    window: int,
    mode: Literal["bi", "forward", "backward"] = "bi",
    n_points_migration: int = 7,
    n_points_transport: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
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

    if window <= n_points_migration:
        raise ValueError("Please set a window larger than the number of fit points...")

    # modes = bi, forward, backward
    s_msd = np.zeros(len(x))
    s_msd[:] = np.nan
    s_alpha = np.zeros(len(x))
    s_alpha[:] = np.nan
    dt = timeline[1] - timeline[0]

    if mode == "bi":
        if window % 2 != 1:
            raise ValueError("Please set an odd window for the bidirectional mode")
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


def drift_msd(
    t: Union[float, np.ndarray], d: float, v: float
) -> Union[float, np.ndarray]:
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
    x: np.ndarray,
    y: np.ndarray,
    timeline: np.ndarray,
    window: int,
    mode: Literal["bi", "forward", "backward"] = "bi",
    n_points_migration: int = 7,
    n_points_transport: int = 7,
    r2_threshold: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
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

    if window <= n_points_migration:
        raise ValueError("Please set a window larger than the number of fit points...")

    # modes = bi, forward, backward
    s_diffusion = np.zeros(len(x))
    s_diffusion[:] = np.nan
    s_velocity = np.zeros(len(x))
    s_velocity[:] = np.nan
    dt = timeline[1] - timeline[0]

    if mode == "bi":
        if window % 2 != 1:
            raise ValueError("Please set an odd window for the bidirectional mode")
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


def columnwise_mean(
    matrix: np.ndarray,
    min_nbr_values: int = 1,
    projection: Literal["mean", "median"] = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the column-wise mean and standard deviation of non-NaN elements in the input matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
            The input matrix for which column-wise mean and standard deviation are calculated.
    min_nbr_values : int, optional
            The minimum number of non-NaN values required in a column to calculate mean and standard deviation.
            Default is 1.
    projection : str, optional
            The method to calculate the central tendency, either 'mean' or 'median' (default is 'mean').

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
    df: pd.DataFrame,
    signal_name: str,
    class_col: str,
    time_col: Optional[Union[str, float, int]] = None,
    class_value: Union[int, List[int]] = [0],
    return_matrix: bool = False,
    forced_max_duration: Optional[int] = None,
    min_nbr_values: int = 2,
    conflict_mode: Literal["mean", "first", "all"] = "mean",
    projection: Literal["mean", "median"] = "mean",
    pairs: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
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
    return_matrix : bool, optional
            Whether to return the signal matrix along with the mean and standard deviation (default is False).
    forced_max_duration : int, optional
            The forced maximum duration of the signal (default is None).
    min_nbr_values : int, optional
            The minimum number of values required to calculate the mean (default is 2).
    conflict_mode : str, optional
            The method to handle conflicts when multiple values exist for the same time point, either 'mean', 'first', or 'all' (default is 'mean').
    projection : str, optional
            The method to calculate the central tendency, either 'mean' or 'median' (default is 'mean').
    pairs : bool, optional
            Whether the input DataFrame contains pair data (default is False).

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

    if signal_name not in df.columns:
        raise KeyError("The signal you want to plot is not one of the measured features.")
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
                except (KeyError, IndexError, ValueError):
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
        
        valid_mask = (timeline_shifted >= 0) & (timeline_shifted < signal_matrix.shape[1])
        signal_matrix[trackid, timeline_shifted[valid_mask].astype(int)] = signal[valid_mask]
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
