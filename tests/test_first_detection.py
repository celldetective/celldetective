"""The first detection: its time is the detection frame minus one, so a cell
first seen at frame 1 appeared at time 0, a valid time."""

import numpy as np
import pandas as pd

from celldetective.events import compute_survival, switch_to_events
from celldetective.tracking import clean_trajectories, write_first_detection_class


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
    its time kept, as in tables written before Decision-0031)."""

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


def test_survival_from_the_start_of_the_movie_leaves_out_side_entries():
    ks = compute_survival(spreading_table(), "class_spread", "t_spread", t_reference=0)
    assert ks.durations.tolist() == [30.0]


def test_survival_from_the_start_keeps_cells_on_the_first_frame():
    """Present on the first frame, class 2, but the movie is the reference: only
    a cell entering later, from the side, is left out."""

    frames = np.arange(40)
    rows = [
        # On the first frame, then spreading.
        pd.DataFrame({"TRACK_ID": 1, "class_firstdetection": 2, "t_firstdetection": -1.0,
                      "status_firstdetection": 1.0, "FRAME": frames}),
        # Entered from the side at frame 10, rows extrapolated before it.
        pd.DataFrame({"TRACK_ID": 2, "class_firstdetection": 2, "t_firstdetection": -1.0,
                      "status_firstdetection": (frames >= 10).astype(float), "FRAME": frames}),
    ]
    df = pd.concat(rows).assign(position="p", class_spread=0, t_spread=30.0)
    ks = compute_survival(df, "class_spread", "t_spread", t_reference=0)
    assert ks.durations.tolist() == [30.0]
    assert ks.event_observed.tolist() == [1]


def test_cell_entering_from_the_side_has_no_first_detection_time():
    """Class 2, time -1, and a status that says from when it is present, also
    once the rows before it are extrapolated."""

    df = pd.DataFrame(
        {
            "TRACK_ID": 0,
            "FRAME": [5, 6, 7],
            "POSITION_X": 5.0,
            "POSITION_Y": 50.0,
            "class_id": 1.0,
        }
    )
    written = write_first_detection_class(df, img_shape=(100, 100))
    assert written["class_firstdetection"].unique().tolist() == [2]
    assert written["t_firstdetection"].unique().tolist() == [-1]

    res = clean_trajectories(written, extrapolate_tracks_pre=True)
    assert res["FRAME"].tolist() == list(range(8))
    assert res["t_firstdetection"].unique().tolist() == [-1]
    assert res["status_firstdetection"].tolist() == [0] * 5 + [1] * 3


def test_event_at_its_arrival_has_a_duration_of_0():
    """A change faster than a frame after an observed arrival: both events share
    their time, so the survival time is 0."""

    events, durations = switch_to_events(
        [0], [9.0], [39], origin_times=[9.0], left_censored=True
    )
    assert events == [1] and durations == [0.0]


def test_class_2_cells_stay_out_of_survival():
    events, durations = switch_to_events(
        [2, 2, 2], [-1.0, 0.0, 10.0], [39, 39, 39], origin_times=[9.0, -1.0, 9.0],
        left_censored=True,
    )
    assert events == [] and durations == []
