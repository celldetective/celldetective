"""The first detection: its time is the detection frame minus one, so a cell
first seen at frame 1 appeared at time 0, a valid time."""

import numpy as np
import pandas as pd

from celldetective.events import compute_survival, switch_to_events
from celldetective.tracking import write_first_detection_class


def one_track(first_frame, n_frames=6):
    frames = np.arange(n_frames)
    return pd.DataFrame(
        {
            "TRACK_ID": 1,
            "FRAME": frames,
            "POSITION_X": 50.0,
            "POSITION_Y": 50.0,
            "class_id": np.where(frames >= first_frame, 1.0, np.nan),
        }
    )


def test_cell_first_seen_at_frame_1_appeared_at_time_0():
    df = write_first_detection_class(one_track(1), img_shape=(100, 100))
    assert df["class_firstdetection"].unique().tolist() == [0]
    assert df["t_firstdetection"].unique().tolist() == [0.0]
    # status 1 from the time of the event on, as for any later appearance
    assert df["status_firstdetection"].tolist() == [1.0] * 6


def test_later_appearance_follows_the_same_rule():
    df = write_first_detection_class(one_track(3), img_shape=(100, 100))
    assert df["t_firstdetection"].unique().tolist() == [2.0]
    assert df["status_firstdetection"].tolist() == [0, 0, 1, 1, 1, 1]


def test_cell_present_at_frame_0_is_left_censored():
    df = write_first_detection_class(one_track(0), img_shape=(100, 100))
    assert df["class_firstdetection"].unique().tolist() == [2]
    assert df["t_firstdetection"].unique().tolist() == [-1]


def test_survival_accepts_time_0():
    """An event at time 0 is valid, and so is a survival time of 0. A negative
    time is not an event."""

    events, durations = switch_to_events(
        [0, 0, 0], [0.0, 3.0, -1.0], [10, 10, 10], left_censored=False
    )
    assert events == [1, 1] and durations == [0.0, 3.0]

    events, durations = switch_to_events(
        [0], [2.0], [10], origin_times=[2.0], left_censored=True
    )
    assert events == [1] and durations == [0.0]


def spreading_table():
    """Two cells spreading at frame 30, seen from frame 10: one appeared inside
    the field of view, the other entered it from the side (first detection class 2,
    its time kept)."""

    rows = []
    for tid, first_class in [(1, 0), (2, 2)]:
        for frame in range(40):
            rows.append(
                {
                    "position": "p",
                    "TRACK_ID": tid,
                    "FRAME": frame,
                    "class_firstdetection": first_class,
                    "t_firstdetection": 9.0,
                    "class_spread": 0,
                    "t_spread": 30.0,
                }
            )
    return pd.DataFrame(rows)


def test_survival_from_an_unreliable_first_detection_is_left_out():
    """A cell entering from the side keeps its spreading time, but its first
    detection was not observed, so it has no duration from it."""

    ks = compute_survival(
        spreading_table(), "class_spread", "t_spread", t_reference="t_firstdetection"
    )
    assert ks.durations.tolist() == [21.0]


def test_survival_from_the_start_of_the_movie_keeps_every_cell():
    ks = compute_survival(spreading_table(), "class_spread", "t_spread", t_reference=0)
    assert ks.durations.tolist() == [30.0, 30.0]
