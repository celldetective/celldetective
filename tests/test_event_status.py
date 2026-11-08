"""Status column derived from the class and time of an event."""

import numpy as np
import pandas as pd

from celldetective.utils.event_status import colors_of, event_status, write_event_status


def test_status_follows_the_class_and_time():
    """0 before the event and 1 from it on; class 2 (no usable event time)
    without a time is NaN throughout, a deleted cell 42."""

    frames = [0, 1, 2, 3]
    df = pd.DataFrame(
        {
            "TRACK_ID": np.repeat([1, 2, 3, 4], 4),
            "FRAME": frames * 4,
            "class": np.repeat([0, 1, 2, 42], 4),
            "t0": np.repeat([2.0, -1.0, -1.0, -1.0], 4),
        }
    )
    write_event_status(df, "class", "t0", "status")

    status = df.groupby("TRACK_ID")["status"].apply(list).to_dict()
    assert status[1] == [0, 0, 1, 1]
    assert status[2] == [0, 0, 0, 0]
    assert np.isnan(status[3]).all()
    assert status[4] == [42, 42, 42, 42]


def test_status_with_custom_columns_and_rows():
    """Signal detection passes its own track and time columns; the annotators
    rewrite only some rows. Other rows are left untouched."""

    df = pd.DataFrame(
        {
            "ID": np.repeat([1, 2], 3),
            "TIME": [0, 1, 2] * 2,
            "class": np.repeat([0, 2], 3),
            "t0": np.repeat([1.0, -1.0], 3),
            "status": -5.0,
        }
    )
    write_event_status(
        df, "class", "t0", "status", keys="ID", frame_name="TIME", rows=df["ID"] == 1
    )

    assert df["status"].tolist() == [0, 1, 1, -5, -5, -5]
    assert df.loc[df["ID"] == 1, "status_color"].tolist() == [
        "tab:blue",
        "tab:red",
        "tab:red",
    ]
    assert df.loc[df["ID"] == 1, "class_color"].tolist() == ["tab:red"] * 3


def test_colors_of_groups_nans():
    """Every NaN gets the color of NaN, computed once."""

    calls = []

    def color_of(value, recently_modified=False):
        calls.append(value)
        return "k" if value != value else str(value)

    colors = colors_of(np.array([1.0, np.nan, 1.0, np.nan, 0.0]), color_of)
    assert colors == ["1.0", "k", "1.0", "k", "0.0"]
    assert len(calls) == 3


def test_event_status_of_each_class():
    """The rule shared by signal detection, the annotators and the viewer.

    Class 2 steps at its time when it has one (a transition at the first
    observed frame, Decision-0023), and is NaN without one (Decision-0020).
    A cell with no class has no status."""

    frames = np.array([0, 1, 2, 3] * 6)
    cclass = np.repeat([0, 1, 2, 2, 42, np.nan], 4)
    t0 = np.repeat([1.5, -1.0, 1.0, -1.0, -1.0, 2.0], 4)
    status = event_status(cclass, t0, frames)
    assert status[:12].tolist() == [0, 0, 1, 1] + [0] * 4 + [0, 1, 1, 1]
    assert np.isnan(status[12:16]).all()
    assert status[16:20].tolist() == [42] * 4
    assert np.isnan(status[20:]).all()
