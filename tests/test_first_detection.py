"""The first detection: its time is the detection frame minus one, so a cell
first seen at frame 1 appeared at time 0, a valid time."""

import numpy as np
import pandas as pd

from celldetective.events import switch_to_events
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
