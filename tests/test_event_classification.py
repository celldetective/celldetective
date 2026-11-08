"""Events found from a condition: class, time and status of each track.

Class 0 is an event observed after the first observed frame, 1 no event, and 2
no usable event time (Decision-0020): with the first observed frame as time when
the cell is positive from it (Decision-0023), with time -1 and a NaN status when
the track cannot be resolved. The status is always the step from class and time,
and the per-frame result of the condition is kept as ``inst_status_*``
(Decision-0021).
"""

import numpy as np
import pandas as pd
import pytest

from celldetective.measure import (
    classify_irreversible_events,
    classify_tracks_from_query,
    classify_transient_events,
    classify_unique_states,
    estimate_time,
)

N_FRAMES = 20


def tracks_table(statuses, first_detection=None):
    """A table with one track per status list, all starting at frame 0."""

    rows = []
    for tid, status in statuses.items():
        for frame, value in enumerate(status):
            rows.append({"TRACK_ID": tid, "FRAME": frame, "class_e": value})
    df = pd.DataFrame(rows)
    df["status_e"] = df["class_e"]
    df["t_firstdetection"] = -1.0
    for tid, t_first in (first_detection or {}).items():
        df.loc[df["TRACK_ID"] == tid, "t_firstdetection"] = t_first
    return df


def per_track(df, column):
    return df.groupby("TRACK_ID")[column].first().to_dict()


def status_of(df, tid):
    return df.loc[df["TRACK_ID"] == tid].sort_values("FRAME")["status_e"].tolist()


def step(t0, n=N_FRAMES):
    return [0.0] * t0 + [1.0] * (n - t0)


class TestIrreversible:
    @pytest.fixture
    def classified(self):
        statuses = {
            1: [0.0] * N_FRAMES,  # never positive
            2: step(8),  # positive from frame 8
            3: [1.0] * N_FRAMES,  # positive from frame 0
            4: [0, 1] * (N_FRAMES // 2),  # flickers: no usable time
            # Detected at frame 5 (t_firstdetection 4) and positive from then on.
            # Rows before are extrapolated and read negative.
            5: step(5),
        }
        df = tracks_table(statuses, first_detection={5: 4.0})
        return classify_irreversible_events(df, "class_e", r2_threshold=0.5)

    def test_classes(self, classified):
        assert per_track(classified, "class_e") == {1: 1, 2: 0, 3: 2, 4: 2, 5: 2}

    def test_times(self, classified):
        times = per_track(classified, "t_e")
        assert times[1] == -1
        assert times[2] == pytest.approx(8, abs=1)
        assert times[3] == 0  # first observed frame
        assert times[4] == -1
        assert times[5] == 5  # first detection, not the extrapolated rows

    def test_status_is_the_step(self, classified):
        t2 = per_track(classified, "t_e")[2]
        expected = [float(f >= t2) for f in range(N_FRAMES)]
        assert status_of(classified, 2) == expected
        assert status_of(classified, 1) == [0.0] * N_FRAMES
        assert status_of(classified, 3) == [1.0] * N_FRAMES
        assert status_of(classified, 5) == step(5)
        assert np.isnan(status_of(classified, 4)).all()

    def test_condition_kept_per_frame(self, classified):
        track = classified.loc[classified["TRACK_ID"] == 4].sort_values("FRAME")
        assert track["inst_status_e"].tolist() == [0, 1] * (N_FRAMES // 2)


def test_pre_event_hides_frames_before_it():
    """Before the pre-event the status is NaN; a track whose pre-event never
    happened gets no class."""

    df = tracks_table({1: step(10), 2: step(10)})
    df["class_pre"] = np.where(df["TRACK_ID"] == 1, 0, 1)
    df["t_pre"] = np.where(df["TRACK_ID"] == 1, 4.0, -1.0)

    out = classify_irreversible_events(df, "class_e", pre_event="pre")

    classes = per_track(out, "class_e")
    assert classes[1] == 0 and np.isnan(classes[2])
    status = status_of(out, 1)
    assert np.isnan(status[:5]).all()
    assert status[5:] == step(10)[5:]
    assert np.isnan(status_of(out, 2)).all()


def test_unique_state():
    """A positive state is held since the cell was first observed: class 2,
    status 1 from the first observed frame."""

    df = tracks_table({1: [0.0] * N_FRAMES, 2: [1.0] * 15 + [0.0] * 5})
    out = classify_unique_states(df, "class_e")
    assert per_track(out, "class_e") == {1: 1, 2: 2}
    assert per_track(out, "t_e") == {1: -1, 2: 0}
    assert status_of(out, 2) == [1.0] * N_FRAMES
    assert "inst_status_e" in out.columns


def test_transient():
    """A pulse gives an event at its onset; a pulse from the first observed
    frame cannot be placed."""

    pulse = [0.0] * 6 + [1.0] * 5 + [0.0] * 9
    from_start = [1.0] * 5 + [0.0] * 15
    df = tracks_table({1: pulse, 2: [0.0] * N_FRAMES, 3: from_start})
    out = classify_transient_events(df, "class_e")
    assert per_track(out, "class_e") == {1: 0, 2: 1, 3: 2}
    assert per_track(out, "t_e") == {1: 6, 2: -1, 3: 0}
    assert "inst_status_e" in out.columns


def test_transient_onset_at_second_frame_is_observed():
    """The onset is the first positive frame, not the last negative one."""

    df = tracks_table({1: [0.0, 1.0, 1.0] + [0.0] * (N_FRAMES - 3)})
    out = classify_transient_events(df, "class_e")
    assert per_track(out, "class_e") == {1: 0}
    assert per_track(out, "t_e") == {1: 1}


def test_transient_pulse_on_at_the_end():
    """A pulse still on at the last frame is an event at its onset; a cell
    positive throughout cannot be placed."""

    df = tracks_table({1: [0.0] * 15 + [1.0] * 5, 2: [1.0] * N_FRAMES})
    out = classify_transient_events(df, "class_e")
    assert per_track(out, "class_e") == {1: 0, 2: 2}
    assert per_track(out, "t_e") == {1: 15, 2: 0}


def test_pre_event_also_hides_rows_before_first_detection():
    """With a pre-event, extrapolated rows before the first detection are still
    not read: positive from its first detection, the cell is class 2."""

    df = tracks_table({1: step(5)}, first_detection={1: 4.0})
    df["class_pre"] = 2
    df["t_pre"] = -1.0
    out = classify_irreversible_events(df, "class_e", pre_event="pre")
    assert per_track(out, "class_e") == {1: 2}
    assert per_track(out, "t_e") == {1: 5}


def test_fit_at_first_frame_is_class_2(monkeypatch):
    """A good fit placed at or before the first observed frame is not an
    observed event (Decision-0023)."""

    import scipy.optimize

    def fit_before_start(func, t, y, **kwargs):
        return np.array([-2.0, 0.5]), None

    monkeypatch.setattr(scipy.optimize, "curve_fit", fit_before_start)
    # Mixed, so that it is fitted; R² is not the point here.
    status = [0.0] * 4 + [1.0] * (N_FRAMES - 4)
    df = tracks_table({1: status})
    df.loc[df["FRAME"] < 3, "status_e"] = np.nan  # first observed frame is 3
    df["class_e"] = 2
    out = estimate_time(df, "class_e", r2_threshold=-np.inf)
    assert per_track(out, "class_e") == {1: 2}
    assert per_track(out, "t_e") == {1: 3}


def test_poor_fit_has_no_usable_time():
    df = tracks_table({1: [0, 1] * (N_FRAMES // 2)})
    df["class_e"] = 2
    out = estimate_time(df, "class_e", r2_threshold=0.9)
    assert per_track(out, "class_e") == {1: 2}
    assert per_track(out, "t_e") == {1: -1}


def test_classify_tracks_from_query():
    """From a query to class, time and step status, over two positions."""

    rows = []
    for pos in ["p1", "p2"]:
        for frame in range(N_FRAMES):
            rows.append(
                {"position": pos, "TRACK_ID": 1, "FRAME": frame, "PI": float(frame >= 7) * 100}
            )
    df = pd.DataFrame(rows)
    df["t_firstdetection"] = -1.0
    out = classify_tracks_from_query(df, "death", "PI > 50", irreversible_event=True)
    first = out.groupby(["position", "TRACK_ID"])[["class_death", "t_death"]].first()
    assert (first["class_death"] == 0).all()
    assert np.allclose(first["t_death"], 7, atol=1)
    assert set(out["status_death"].unique()) == {0.0, 1.0}


def arrival_table(first_detection_class=0, t_first=9.0, start=0):
    """One cell positive on every frame from its first detection at frame 10.
    With ``start`` 0, the rows before it are extrapolated."""

    df = pd.DataFrame({"TRACK_ID": 1, "FRAME": np.arange(start, N_FRAMES)})
    df["status_e"] = (df["FRAME"] >= 10).astype(float)
    df["class_e"] = df["status_e"]
    df["class_firstdetection"] = first_detection_class
    df["t_firstdetection"] = t_first
    return df


@pytest.mark.parametrize("start", [0, 10])
def test_already_positive_at_an_observed_arrival_is_an_event_at_it(start):
    """Faster than a frame: the event shares the time of the first detection."""

    df = classify_irreversible_events(arrival_table(start=start), "class_e")
    assert per_track(df, "class_e")[1] == 0
    assert per_track(df, "t_e")[1] == 9
    observed = df.loc[df["FRAME"] >= 10, "status_e"]
    assert observed.eq(1).all()


def test_transient_at_an_observed_arrival_is_an_event_at_it():
    df = arrival_table()
    df.loc[df["FRAME"] >= 15, ["status_e", "class_e"]] = 0.0
    df = classify_transient_events(df, "class_e")
    assert per_track(df, "class_e")[1] == 0
    assert per_track(df, "t_e")[1] == 9


def test_already_positive_after_an_unreliable_arrival_stays_class_2():
    """Entering from the side: the first detection was not observed."""

    df = classify_irreversible_events(arrival_table(first_detection_class=2), "class_e")
    assert per_track(df, "class_e")[1] == 2
    assert per_track(df, "t_e")[1] == 10


def test_already_positive_on_the_first_frame_stays_class_2():
    df = arrival_table(first_detection_class=2, t_first=-1.0)
    df["status_e"] = 1.0
    df["class_e"] = 1.0
    df = classify_irreversible_events(df, "class_e")
    assert per_track(df, "class_e")[1] == 2
    assert per_track(df, "t_e")[1] == 0
