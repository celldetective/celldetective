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

    def analyze(
        pos, model, use_gpu=True, populations=None,
        reference_population=None, neighbor_population=None,
    ):
        calls.append((pos, model, populations))
        signals.logger.warning("a warning of the analysis")
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
    with open(os.path.join(pos, "log_pairs.txt")) as log:
        logged = log.read()
    assert "PAIR SIGNAL ANALYSIS" in logged and "a warning of the analysis" in logged
    if outcome == "fails":
        assert queue[-1] == {"status": "error", "message": "model failed"}
        assert "model failed" in logged
    else:
        assert queue[-1] == {"status": "finished"}
        assert ("a table is missing" in logged) == (outcome == "missing")


def use_stub_model(monkeypatch, tmp_path, model_config):
    """`analyze_signals` and `analyze_pair_signals` load this model by name."""

    import json

    import celldetective.event_detection_models as models
    import celldetective.signals as signals

    path = tmp_path / "config_input.json"
    path.write_text(json.dumps(model_config))

    def load(pretrained=None):
        model = StubModel(model_config)
        model.pretrained = pretrained
        return model

    monkeypatch.setattr(models, "SignalDetectionModel", load)
    monkeypatch.setattr(
        signals, "_resolve_signal_model_paths", lambda *a, **k: (str(tmp_path), str(path))
    )


def test_analyze_signals(monkeypatch, tmp_path):
    from celldetective.signals import analyze_signals

    use_stub_model(monkeypatch, tmp_path, config())
    out = analyze_signals(tracks(), "stub")
    assert per_track(out, "class_death") == {3: 1, 5: 0, 7: 0}
    assert per_track(out, "t_death") == {3: -1, 5: 12, 7: 5}


def pair_rows(reference, neighbor, t_event):
    return [
        {"REFERENCE_ID": 1, "NEIGHBOR_ID": 2, "FRAME": frame,
         "reference_population": reference, "neighbor_population": neighbor,
         "PI": float(frame >= t_event)}
        for frame in range(20)
    ]


def single_cells(track):
    return pd.DataFrame(
        {"TRACK_ID": track, "FRAME": range(20), "POSITION_X": 1.0, "POSITION_Y": 1.0}
    )


def analyze_pairs(pairs):
    from celldetective.signals import analyze_pair_signals

    return analyze_pair_signals(pairs, single_cells(1), single_cells(2), "stub")


def per_combination(df, column):
    groups = df.groupby(["reference_population", "neighbor_population"])
    return groups[column].first().to_dict()


def test_analyze_pair_signals_applies_the_model_to_its_populations(monkeypatch, tmp_path):
    """The same IDs in another combination are other cells: they are not read
    through the tables of this model, and keep what they had."""

    use_stub_model(
        monkeypatch, tmp_path,
        config(reference_population="targets", neighbor_population="effectors"),
    )
    pairs = pd.DataFrame(
        pair_rows("targets", "effectors", 5) + pair_rows("effectors", "targets", 12)
    )
    out = analyze_pairs(pairs)

    times = per_combination(out, "t_death")
    assert times == {("targets", "effectors"): 5, ("effectors", "targets"): -1}
    other = out["reference_population"] == "effectors"
    assert out.loc[other, "class_death"].isna().all()
    assert out.loc[other, "status_death"].isna().all()
    assert out.loc[~other, "class_death"].eq(0).all()


def test_analyze_pair_signals_keeps_the_results_of_other_populations(monkeypatch, tmp_path):
    use_stub_model(
        monkeypatch, tmp_path,
        config(reference_population="targets", neighbor_population="effectors"),
    )
    pairs = pd.DataFrame(
        pair_rows("targets", "effectors", 5) + pair_rows("effectors", "targets", 12)
    )
    earlier = pairs["reference_population"] == "effectors"
    pairs["class_death"] = np.where(earlier, 0.0, np.nan)
    pairs["t_death"] = np.where(earlier, 12.0, -1.0)
    out = analyze_pairs(pairs)
    assert per_combination(out, "t_death") == {
        ("targets", "effectors"): 5, ("effectors", "targets"): 12,
    }
    assert out.loc[earlier].sort_values("FRAME")["status_death"].tolist() == [0.0] * 12 + [1.0] * 8


def test_analyze_pair_signals_without_recorded_populations(monkeypatch, tmp_path):
    """Each combination still gets its own signal and prediction, with a warning."""

    import celldetective.signals as signals

    use_stub_model(monkeypatch, tmp_path, config())
    warnings = []
    monkeypatch.setattr(signals.logger, "warning", warnings.append)
    out = analyze_pairs(
        pd.DataFrame(pair_rows("targets", "effectors", 5) + pair_rows("effectors", "targets", 12))
    )
    assert per_combination(out, "t_death") == {
        ("targets", "effectors"): 5, ("effectors", "targets"): 12,
    }
    assert any("several combinations" in w for w in warnings)


def test_analyze_pair_signals_on_populations_chosen_by_the_user(monkeypatch, tmp_path):
    """A model can be deployed on other populations: those pairs are predicted, the
    pairs it was trained on are left alone, and a warning asks to check the traces."""

    import celldetective.signals as signals

    use_stub_model(
        monkeypatch, tmp_path,
        config(reference_population="targets", neighbor_population="effectors"),
    )
    warnings = []
    monkeypatch.setattr(signals.logger, "warning", warnings.append)
    pairs = pd.DataFrame(
        pair_rows("targets", "effectors", 5) + pair_rows("effectors", "targets", 12)
    )
    out = signals.analyze_pair_signals(
        pairs, single_cells(1), single_cells(2), "stub",
        reference_population="effectors", neighbor_population="targets",
    )
    assert per_combination(out, "t_death") == {
        ("targets", "effectors"): -1, ("effectors", "targets"): 12,
    }
    assert any("look like" in w for w in warnings)


@pytest.mark.parametrize(
    "chosen, expected",
    [
        ({}, ("targets", "effectors")),
        (
            {"reference_population": "macrophages", "neighbor_population": "targets"},
            ("macrophages", "targets"),
        ),
    ],
)
def test_pair_detection_at_a_position_reads_the_tables_of_the_populations(
    monkeypatch, tmp_path, chosen, expected
):
    import celldetective.signals as signals

    use_stub_model(
        monkeypatch, tmp_path,
        config(reference_population="targets", neighbor_population="effectors"),
    )
    (tmp_path / "output" / "tables").mkdir(parents=True)
    monkeypatch.setattr(signals, "get_position_pickle", lambda pos, population: population)
    monkeypatch.setattr(signals, "get_position_table", lambda pos, population: pd.DataFrame())
    monkeypatch.setattr(signals, "experiment_frame_interval", lambda pos: 1.0)
    seen = {}

    def record(pairs, reference, neighbor, **kwargs):
        seen.update(tables=(reference, neighbor), kwargs=kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(signals, "analyze_pair_signals", record)
    signals.analyze_pair_signals_at_position(
        str(tmp_path), "stub", populations=["targets", "effectors", "macrophages"], **chosen
    )
    assert seen["tables"] == expected
    assert (seen["kwargs"]["reference_population"], seen["kwargs"]["neighbor_population"]) == expected


def test_analyze_pair_signals_with_no_pair_of_its_populations(monkeypatch, tmp_path):
    use_stub_model(
        monkeypatch, tmp_path,
        config(reference_population="targets", neighbor_population="effectors"),
    )
    pairs = pd.DataFrame(pair_rows("effectors", "targets", 12))
    out = analyze_pairs(pairs)
    pd.testing.assert_frame_equal(out, pairs)


def test_event_at_first_frame_of_track_is_class_2():
    """A model reading a track that is already past the event when it starts
    sees a step from the padding: the change was not observed (Decision-0023)."""

    df = tracks()
    df = df.loc[(df["TRACK_ID"] != 5) | (df["FRAME"] >= 12)].copy()  # starts at its event
    out, _, (cls, t, status) = detect_events_with_model(df, StubModel(config()))
    assert per_track(out, cls) == {3: 1, 5: 2, 7: 0}
    assert per_track(out, t) == {3: -1, 5: -1, 7: 5}
    assert out.loc[out["TRACK_ID"] == 5, status].isna().all()


def test_training_repeats_the_first_and_last_values_of_a_track():
    from types import SimpleNamespace

    from celldetective.event_detection_models import SignalDetectionModel

    dataset = [
        {"FRAME": np.arange(3, 7), "PI": np.array([5.0, 6.0, 7.0, 8.0]), "class": 1, "time_of_interest": -1.0},
    ]
    owner = SimpleNamespace(n_channels=1, model_signal_length=16, channel_option=["PI"])
    signals, _, _ = SignalDetectionModel.cast_signals_into_training_data(owner, dataset, ["PI"], 10)
    assert signals[0, :, 0].tolist() == [5, 5, 5, 5, 6, 7, 8, 8, 8, 8]


@pytest.mark.parametrize("padding", [None, "edge"])
def test_padding_before_a_track_follows_the_model(padding):
    """A model trained with the edges repeated reads them; older models were
    trained with zeros and keep reading zeros before a track."""

    df = tracks()
    df = df.loc[(df["TRACK_ID"] != 5) | (df["FRAME"] >= 12)].copy()  # starts at frame 12
    extra = {} if padding is None else {"padding": padding}
    model = StubModel(config(**extra))
    detect_events_with_model(df, model)
    track5 = model.seen[0][1, :, 0]
    assert track5[12] != 0
    assert np.all(track5[:12] == (track5[12] if padding else 0))


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


def test_training_sample_entering_from_the_side():
    """No first detection time: the status says where the sample is first seen."""

    from celldetective.event_detection_models import _first_observed_frame

    frames = np.arange(20)
    sample = {
        "FRAME": frames,
        "t_firstdetection": np.full(20, -1.0),
        "status_firstdetection": (frames >= 10).astype(float),
    }
    assert _first_observed_frame(sample) == 10
