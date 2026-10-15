"""
Status column of an event class, derived from the class and time of each
track. Shared by signal detection and the annotators.
"""

from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from celldetective.utils.color_mappings import color_from_class, color_from_status
from celldetective.utils.signal_windowing import DELETED, ELSE


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
    The status of each cell from the class and time of its track: 0 before the
    event, 1 from it on, 1 throughout for the "else" class (the event happened
    before the acquisition) and 42 for a cell marked for deletion.
    """

    after = ((t0 > 0) & (frames >= t0)) | (cclass == ELSE)
    return np.where(cclass > ELSE, DELETED, after.astype(float))


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

    keys = [keys] if isinstance(keys, str) else list(keys)
    subset = df if rows is None else df.loc[rows]
    subset = subset.dropna(subset=keys)
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
