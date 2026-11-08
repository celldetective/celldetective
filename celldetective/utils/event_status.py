"""
Status column of an event class, derived from the class and time of each
track. Shared by signal detection and the annotators.
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from celldetective.utils.color_mappings import color_from_class, color_from_status
from celldetective.utils.signal_windowing import DELETED, ELSE, EVENT, NO_EVENT


def event_columns(class_name: str) -> Tuple[str, str]:
    """
    Time and status columns of an event class.

    ``class_<event>`` goes with ``t_<event>`` and ``status_<event>``, ``class``
    with ``t0`` and ``status``.
    """

    suffix = class_name.removeprefix("class").removeprefix("_")
    if not suffix:
        return "t0", "status"
    return f"t_{suffix}", f"status_{suffix}"


def event_class_of_label(label: Optional[str]) -> str:
    """The class column of an event label: ``class_<label>``, or ``class`` without one."""

    return f"class_{label}" if label else "class"


def colors_of(values: np.ndarray, color_of: Callable, recently_modified: bool = False) -> list:
    """The color of each value, computed once per distinct value."""

    # NaN gets the code -1, which picks the last color of the palette.
    codes, uniques = pd.factorize(np.asarray(values))
    palette = [color_of(u, recently_modified=recently_modified) for u in uniques]
    palette.append(color_of(np.nan, recently_modified=recently_modified))
    return list(np.asarray(palette, dtype=object)[codes])


def color_events(df: pd.DataFrame, class_name: str, status_name: str) -> None:
    """Color the cells of the table by their status and class."""
    df["status_color"] = colors_of(df[status_name].to_numpy(), color_from_status)
    df["class_color"] = colors_of(df[class_name].to_numpy(), color_from_class)


def event_status(cclass: np.ndarray, t0: np.ndarray, frames: np.ndarray) -> np.ndarray:
    """
    The status of each cell from the class and time of its track.

    - Class 0 (event): 0 before the event time, 1 from it on.
    - Class 1 (no event): 0 throughout.
    - Class 2 ("else", no usable event time): NaN throughout (time -1), since the
      cell may be left-censored or unusable. Tables written before Decision-0028
      can hold a class 2 cell with a time, which gets the step at that time.
    - 42 for a cell marked for deletion, and NaN for a cell with no class.
    """

    cclass = np.asarray(cclass, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    frames = np.asarray(frames, dtype=float)

    has_time = t0 >= 0
    status = ((cclass != NO_EVENT) & has_time & (frames >= t0)).astype(float)
    status[(cclass == ELSE) & ~has_time] = np.nan
    status[np.isnan(cclass)] = np.nan
    status[cclass > ELSE] = DELETED
    return status


def write_event_status(
    df: pd.DataFrame,
    class_name: str,
    time_name: str,
    status_name: str,
    keys: Union[str, Sequence[str]] = "TRACK_ID",
    rows: Optional[pd.Series] = None,
    recently_modified: bool = False,
    frame_name: str = "FRAME",
) -> None:
    """
    Fill the status column of an event class, and the status and class colors,
    from the class and time of each track (see `event_status`).

    Parameters
    ----------
    df : pandas.DataFrame
        The table, modified in place.
    class_name, time_name, status_name : str
        The columns of the event class.
    keys : str or sequence of str, optional
        The columns identifying a track, or a pair.
    rows : pandas.Series of bool, optional
        The rows to fill, all of them by default.
    recently_modified : bool, optional
        Whether to color the cells as just annotated.
    frame_name : str, optional
        The column of the frame.
    """

    keys = _as_keys(keys)
    subset = _track_columns(df, keys, [frame_name, class_name, time_name], rows)
    groups = subset.groupby(keys, sort=False)
    t0 = groups[time_name].transform("first").to_numpy()
    cclass = groups[class_name].transform("first").to_numpy()
    frames = subset[frame_name].to_numpy()

    status = event_status(cclass, t0, frames)

    df.loc[subset.index, status_name] = status
    df.loc[subset.index, "status_color"] = colors_of(
        status, color_from_status, recently_modified
    )
    df.loc[subset.index, "class_color"] = colors_of(
        cclass, color_from_class, recently_modified
    )


def _as_keys(keys: Union[str, Sequence[str]]) -> list:
    return [keys] if isinstance(keys, str) else list(keys)


def _track_columns(
    df: pd.DataFrame, keys: list, columns: list, rows: Optional[pd.Series] = None
) -> pd.DataFrame:
    """The rows (all by default) with a track key, and only the columns read:
    tables can be wide, and a full copy of them is costly."""

    columns = keys + [c for c in columns if c not in keys]
    subset = df.loc[:, columns] if rows is None else df.loc[rows, columns]
    return subset.dropna(subset=keys)


def first_observed_frames(
    df: pd.DataFrame, keys: Sequence[str], frame_name: str = "FRAME"
) -> np.ndarray:
    """
    The first frame where each row's cell (or pair) is observed: the first frame
    of its track, or the first detection when rows were extrapolated before it
    (``t_firstdetection`` is the last frame without the cell). A cell entering from
    the side has no such time, and ``status_firstdetection`` says from when it is
    present.
    """

    groups = df.groupby(list(keys), sort=False)
    first = groups[frame_name].transform("min").to_numpy(dtype=float)
    if "status_firstdetection" in df.columns:
        present = df[frame_name].where(df["status_firstdetection"] == 1)
        seen = present.groupby([df[k] for k in keys], sort=False).transform("min")
        first = np.fmax(first, seen.to_numpy(dtype=float))
    if "t_firstdetection" in df.columns:
        detected = groups["t_firstdetection"].transform("first").to_numpy(dtype=float)
        after = np.where(detected >= 0, detected + 1, -np.inf)
        first = np.maximum(first, after)
    return first


def resolve_unobserved_events(
    df: pd.DataFrame,
    class_name: str,
    time_name: str,
    keys: Union[str, Sequence[str]] = "TRACK_ID",
    frame_name: str = "FRAME",
    rows: Optional[pd.Series] = None,
) -> int:
    """
    Make class 2 every event placed at or before the first frame where the cell
    is observed, with time -1 (Decision-0023): the change itself was not observed,
    and the time column only holds onsets. Used by event detection with a model and by the annotators,
    through `write_resolved_event_status`. An event at the observed arrival of
    the cell stays an event, with the onset of that arrival. The condition route applies the same
    rule as it places each change (``measure._event_outcome``), from the first
    frame after its pre-event.

    Parameters
    ----------
    df : pandas.DataFrame
        The table, modified in place.
    class_name, time_name : str
        The class and time columns of the event.
    keys : str or sequence of str, optional
        The columns identifying a track, or a pair.
    frame_name : str, optional
        The column of the frame.
    rows : pandas.Series of bool, optional
        The rows to check, all of them by default.

    Returns
    -------
    int
        The number of tracks made class 2.
    """

    keys = _as_keys(keys)
    columns = [frame_name, class_name, time_name]
    columns += [
        c
        for c in ("t_firstdetection", "class_firstdetection", "status_firstdetection")
        if c in df.columns
    ]
    subset = _track_columns(df, keys, columns, rows)
    if subset.empty:
        return 0
    first = first_observed_frames(subset, keys, frame_name)
    unobserved = (subset[class_name].to_numpy() == EVENT) & (
        subset[time_name].to_numpy(dtype=float) <= first
    )
    if {"t_firstdetection", "class_firstdetection"} <= set(subset.columns):
        # The first observed frame is the observed arrival of the cell: the change
        # was faster than a frame, an event with the onset of the arrival.
        detected = subset["t_firstdetection"].to_numpy(dtype=float)
        arrived = (
            unobserved
            & (subset["class_firstdetection"].to_numpy() == EVENT)
            & (detected + 1 == first)
        )
        df.loc[subset.index[arrived], time_name] = detected[arrived]
        unobserved &= ~arrived
    if not unobserved.any():
        return 0
    index = subset.index[unobserved]
    df.loc[index, class_name] = ELSE
    df.loc[index, time_name] = -1.0
    return int(subset.loc[index, keys].drop_duplicates().shape[0])


def write_resolved_event_status(
    df: pd.DataFrame,
    class_name: str,
    time_name: str,
    status_name: str,
    keys: Union[str, Sequence[str]] = "TRACK_ID",
    frame_name: str = "FRAME",
    rows: Optional[pd.Series] = None,
    recently_modified: bool = False,
) -> int:
    """
    Make class 2 the events that were not observed (`resolve_unobserved_events`),
    then fill the status from class and time (`write_event_status`).

    Returns
    -------
    int
        The number of tracks made class 2.
    """

    n_unobserved = resolve_unobserved_events(
        df, class_name, time_name, keys=keys, frame_name=frame_name, rows=rows
    )
    write_event_status(
        df,
        class_name,
        time_name,
        status_name,
        keys=keys,
        rows=rows,
        recently_modified=recently_modified,
        frame_name=frame_name,
    )
    return n_unobserved
