"""Event detection with a model: one implementation for the library and the
process (Decision-0022), checked with a stub model."""

import os

import numpy as np
import pandas as pd
import pytest

from celldetective.signals import detect_events_with_model, event_columns_of_model


class StubModel:
    """Calls an event when the first channel exceeds 0.5, at its first crossing."""

    def __init__(self, config):
        self.config = config
        self.seen = []

    def predict_class(self, signals):
        self.seen.append(signals.copy())
        return np.where((signals[:, :, 0] > 0.5).any(axis=1), 0, 1)

    def predict_time_of_interest(self, signals, class_predictions=None):
        above = signals[:, :, 0] > 0.5
        return np.where(above.any(axis=1), above.argmax(axis=1), -1).astype(float)


def config(**extra):
    return {"channels": ["PI"], "model_signal_length": 64, "label": "death", **extra}


def tracks():
    rows = []
    for tid, t_event in [(7, 5), (3, None), (5, 12)]:
        for frame in range(20):
            pi = 1.0 if t_event is not None and frame >= t_event else 0.0
            rows.append(
                {"TRACK_ID": tid, "FRAME": frame, "POSITION_X": 10.0, "POSITION_Y": 10.0,
                 "PI_intensity_mean": pi}
            )
    return pd.DataFrame(rows)


def per_track(df, column):
    return df.groupby("TRACK_ID")[column].first().to_dict()


def test_columns_follow_the_label():
    assert event_columns_of_model(config()) == ("class_death", "t_death", "status_death")
    assert event_columns_of_model(config(label="")) == ("class", "t0", "status")


def test_results_land_on_their_track():
    df, selected, (cls, t, status) = detect_events_with_model(tracks(), StubModel(config()))
    assert selected == ["PI_intensity_mean"]
    assert per_track(df, cls) == {3: 1, 5: 0, 7: 0}
    assert per_track(df, t) == {3: -1, 5: 12, 7: 5}
    track7 = df.loc[df["TRACK_ID"] == 7].sort_values("FRAME")
    assert track7[status].tolist() == [0.0] * 5 + [1.0] * 15


def test_empty_channel_of_the_model():
    """A model with an empty ("None") channel reads zeros, and the column is
    removed afterwards."""

    model = StubModel(config(channels=["PI", "area"], selected_channels=["PI_intensity_mean", "None"]))
    df, selected, _ = detect_events_with_model(tracks(), model)
    assert selected == ["PI_intensity_mean", "None"]
    assert "None" not in df.columns
    assert np.all(model.seen[0][:, :, 1] == 0)


def test_two_empty_channels_of_the_model():
    """The same column read for two channels fills both."""

    model = StubModel(
        config(channels=["PI", "a", "b"], selected_channels=["PI_intensity_mean", "None", "None"])
    )
    df, selected, (cls, _, _) = detect_events_with_model(tracks(), model)
    assert model.seen[0].shape[2] == 3
    assert per_track(df, cls) == {3: 1, 5: 0, 7: 0}


def test_track_left_out_gets_no_class(monkeypatch):
    """A track that cleaning removes gets no class rather than a false event."""

    import celldetective.signals as signals

    real_clean = signals.clean_trajectories

    def drop_track_3(df, **kwargs):
        return real_clean(df.loc[df["TRACK_ID"] != 3].copy(), **kwargs)

    monkeypatch.setattr(signals, "clean_trajectories", drop_track_3)
    df, _, (cls, t, status) = detect_events_with_model(tracks(), StubModel(config()))
    assert np.isnan(per_track(df, cls)[3])
    assert per_track(df, t)[3] == -1
    assert df.loc[df["TRACK_ID"] == 3, status].isna().all()
    assert per_track(df, cls)[7] == 0


def test_process_matches_the_library(tmp_path):
    from celldetective.processes.detect_events import SignalAnalysisProcess

    position = tmp_path / "W1" / "100"
    (position / "output" / "tables").mkdir(parents=True)
    table = position / "output" / "tables" / "trajectories_targets.csv"
    tracks().to_csv(table, index=False)

    class Queue(list):
        put = list.append

    queue = Queue()
    process = SignalAnalysisProcess(queue=queue, process_args={"mode": "targets", "model_name": "stub"})
    process.setup_for_position(str(position) + os.sep)
    process.process_position(model=StubModel(config()))

    from_process = pd.read_csv(table).sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)
    from_library, _, _ = detect_events_with_model(tracks(), StubModel(config()))
    from_library = from_library.sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)

    for column in ["class_death", "t_death", "status_death"]:
        assert np.allclose(from_process[column], from_library[column], equal_nan=True)
    assert any("frame_progress" in m for m in queue if isinstance(m, dict))


def test_training_event_at_first_frame_is_class_2():
    """An annotated event at or before frame 0 trains as class 2, like a shifted
    one in augmentation (Decision-0023)."""

    pytest.importorskip("tensorflow")
    from types import SimpleNamespace

    from celldetective.event_detection_models import SignalDetectionModel

    dataset = [
        {"FRAME": np.arange(10), "PI": np.ones(10), "class": 0, "time_of_interest": 0.0},
        {"FRAME": np.arange(10), "PI": np.ones(10), "class": 0, "time_of_interest": 4.0},
        {"FRAME": np.arange(10), "PI": np.ones(10), "class": 1, "time_of_interest": 3.0},
    ]
    owner = SimpleNamespace(n_channels=1, model_signal_length=16, channel_option=["PI"])
    _, classes, times = SignalDetectionModel.cast_signals_into_training_data(
        owner, dataset, ["PI"], 10
    )
    assert classes.tolist() == [2, 0, 1]
    assert times.tolist() == [-1, 4, -1]


@pytest.mark.parametrize("outcome", ["done", "missing", "fails"])
def test_pair_process_reports_its_end(tmp_path, monkeypatch, outcome):
    """Pair event detection runs in a process that tells the progress window
    when it ends, skips a position whose tables are missing without logging an
    analysis, and reports a failure."""

    import celldetective.signals as signals
    from celldetective.processes.detect_pair_events import PairSignalAnalysisProcess

    pos = str(tmp_path) + os.sep
    calls = []

    def analyze(pos, model, use_gpu=True, populations=None):
        calls.append((pos, model, populations))
        if outcome == "fails":
            raise ValueError("model failed")
        if outcome == "missing":
            return None
        return pos + "output/tables/trajectories_pairs.csv"

    monkeypatch.setattr(signals, "analyze_pair_signals_at_position", analyze)

    class Queue(list):
        put = list.append

    queue = Queue()
    process = PairSignalAnalysisProcess(
        queue=queue,
        process_args={"pos": pos, "model_name": "m", "populations": ["a", "b"]},
    )
    process.run()

    assert calls == [(pos, "m", ["a", "b"])]
    logged = os.path.exists(os.path.join(pos, "log_pairs.txt"))
    if outcome == "fails":
        assert queue[-1] == {"status": "error", "message": "model failed"}
    else:
        assert queue[-1] == {"status": "finished"}
        assert logged == (outcome == "done")


def test_event_at_first_frame_of_track_is_class_2():
    """A model reading a track that is already past the event when it starts
    sees a step from the padding: the change was not observed (Decision-0023)."""

    df = tracks()
    df = df.loc[(df["TRACK_ID"] != 5) | (df["FRAME"] >= 12)].copy()  # starts at its event
    out, _, (cls, t, status) = detect_events_with_model(df, StubModel(config()))
    assert per_track(out, cls) == {3: 1, 5: 2, 7: 0}
    assert per_track(out, t) == {3: -1, 5: 12, 7: 5}
    assert out.loc[out["TRACK_ID"] == 5, status].eq(1).all()


def test_warning_on_another_frame_rate(monkeypatch):
    import celldetective.signals as signals

    warnings = []
    monkeypatch.setattr(signals.logger, "warning", warnings.append)

    signals.warn_on_frame_interval_mismatch(config(frame_to_min=5.0), 5.0)
    signals.warn_on_frame_interval_mismatch(config(frame_to_min=[2.0, 5.0]), 5.02)
    signals.warn_on_frame_interval_mismatch(config(), 1.0)  # older model, not checked
    signals.warn_on_frame_interval_mismatch(config(frame_to_min=5.0), None)
    assert warnings == []

    signals.warn_on_frame_interval_mismatch(config(frame_to_min=5.0), 1.0)
    assert len(warnings) == 1 and "5.0" in warnings[0] and "1.0" in warnings[0]


def test_warning_on_a_channel_matched_by_part_of_its_name(monkeypatch):
    import celldetective.signals as signals

    warnings = []
    monkeypatch.setattr(signals.logger, "warning", warnings.append)
    assert signals._select_model_signals(["PI_intensity_mean"], ["PI"]) == ["PI_intensity_mean"]
    assert signals._select_model_signals(["PI"], ["PI"]) == ["PI"]
    assert len(warnings) == 1 and "PI_intensity_mean" in warnings[0]


def test_training_records_the_frame_interval():
    pytest.importorskip("tensorflow")
    from celldetective.event_detection_models import _frame_intervals_of

    assert _frame_intervals_of([{"frame_to_min": 5}, {"frame_to_min": 5.0}, {}]) == {5.0}
    assert _frame_intervals_of([{"class": 0}]) == set()


def test_training_event_at_an_observed_arrival_stays_an_event():
    """As at inference: faster than a frame, an event at the arrival."""

    pytest.importorskip("tensorflow")
    from types import SimpleNamespace

    from celldetective.event_detection_models import SignalDetectionModel

    sample = {
        "FRAME": np.arange(10, 20), "PI": np.ones(10), "class": 0, "time_of_interest": 10.0,
        "class_firstdetection": np.zeros(10), "t_firstdetection": np.full(10, 9.0),
    }
    owner = SimpleNamespace(n_channels=1, model_signal_length=32, channel_option=["PI"])
    _, classes, times = SignalDetectionModel.cast_signals_into_training_data(
        owner, [sample], ["PI"], 20
    )
    assert classes.tolist() == [0]
    assert times.tolist() == [9]
