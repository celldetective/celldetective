"""The process Submit launches: what it runs, and what it tells the progress window."""

import numpy as np
import pytest
import tifffile

from celldetective.preprocessing import first_step_outputs
from celldetective.processes.background_correction import BackgroundCorrectionProcess
from tests.preprocessing_helpers import FIELD, write_experiment


class Queue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)

    def close(self):
        pass


@pytest.fixture
def experiment(tmp_path):
    exp_dir, (movie_dir,) = write_experiment(tmp_path, [1])
    return exp_dir, movie_dir


def _run(exp_dir, **protocol):
    """Run the process in this one, as it would run in its own."""
    queue = Queue()
    BackgroundCorrectionProcess(
        queue,
        process_args={
            "exp_dir": exp_dir,
            "well_option": "*",
            "position_option": "*",
            "target_channel": "Channel1",
            "export": True,
            "return_stacks": False,
            "activation_protocol": [["gauss", 2], ["std", 4]],
            **protocol,
        },
    ).run()
    return queue.messages


def test_fit_receives_the_offset(experiment, monkeypatch):
    received = {}
    monkeypatch.setattr(
        "celldetective.preprocessing.correct_background_model",
        lambda *args, **kwargs: received.update(kwargs),
    )

    _run(experiment[0], **FIT, offset=50.0)

    assert received["offset"] == 50.0


def _run_steps(exp_dir, *steps):
    """Run a list of corrections, as Submit does."""
    queue = Queue()
    common = dict(
        exp_dir=exp_dir,
        well_option="*",
        position_option="*",
        target_channel="Channel1",
        export=True,
        return_stacks=False,
        activation_protocol=[["gauss", 2], ["std", 4]],
    )
    BackgroundCorrectionProcess(
        queue, process_args={"steps": [{**common, **step} for step in steps]}
    ).run()
    return queue.messages


def _chained(exp_dir, first, second):
    sources = first_step_outputs(exp_dir)
    return (
        first,
        {**second, "movie_prefix": "Corrected", "export_prefix": None, "source_stacks": sources},
    )


def _statuses(messages):
    return [m["status"] for m in messages if isinstance(m, dict) and "status" in m]


FIT = dict(
    correction_type="fit",
    model="paraboloid",
    threshold_on_std=100,
    operation="divide",
    clip=False,
)
MODEL_FREE = dict(
    correction_type="model-free",
    threshold_on_std=100,
    operation="divide",
    clip=False,
)
OFFSET = dict(correction_type="offset", correction_horizontal=1)
REGISTRATION = dict(correction_type="registration")


@pytest.mark.parametrize(
    "protocol",
    [FIT, MODEL_FREE, OFFSET, REGISTRATION],
    ids=["fit", "model-free", "offset", "registration"],
)
def test_each_correction_finishes_and_exports(experiment, protocol):
    exp_dir, movie_dir = experiment

    messages = _run(exp_dir, **protocol)

    assert "finished" in _statuses(messages)
    assert tifffile.imread(movie_dir / "Corrected_sample.tif").shape == (3, FIELD, FIELD)


def test_a_list_runs_in_one_window_that_closes_at_the_end(experiment):
    exp_dir, movie_dir = experiment
    alone = _run(exp_dir, **OFFSET)
    (movie_dir / "Corrected_sample.tif").unlink()

    messages = _run_steps(
        exp_dir, *_chained(exp_dir, OFFSET, {**OFFSET, "correction_horizontal": 2})
    )

    statuses = _statuses(messages)
    assert statuses.count("finished") == 1
    assert statuses[-1] == "finished"
    assert _statuses(alone).count("finished") == 1
    assert sorted(f.name for f in movie_dir.iterdir()) == ["Corrected_sample.tif", "sample.tif"]


def test_overall_progress_runs_once_over_the_list(experiment):
    exp_dir, _ = experiment

    messages = _run_steps(exp_dir, *_chained(exp_dir, OFFSET, OFFSET))

    overall = [m["well_progress"] for m in messages if isinstance(m, dict) and "well_progress" in m]
    assert overall == sorted(overall)
    assert 50 in overall
    assert overall[-1] == 100


def test_status_says_which_step_runs(experiment):
    exp_dir, _ = experiment

    messages = _run_steps(
        exp_dir,
        *_chained(exp_dir, dict(OFFSET, step_title="First"), dict(OFFSET, step_title="Second")),
    )

    statuses = _statuses(messages)
    assert "[Step 1/2] First" in statuses
    assert "[Step 2/2] Second" in statuses


def test_each_step_starts_with_empty_position_and_frame_bars(experiment):
    exp_dir, _ = experiment

    messages = _run_steps(
        exp_dir,
        *_chained(exp_dir, dict(OFFSET, step_title="First"), dict(OFFSET, step_title="Second")),
    )

    (start,) = [
        m for m in messages if isinstance(m, dict) and m.get("status") == "[Step 2/2] Second"
    ]
    assert (start["pos_progress"], start["frame_progress"]) == (0, 0)


def test_options_of_a_step_do_not_reach_the_next(experiment, monkeypatch):
    received = []
    monkeypatch.setattr(
        "celldetective.preprocessing.correct_background_model",
        lambda *args, **kwargs: received.append(kwargs["offset"]),
    )

    _run_steps(experiment[0], {**FIT, "offset": 50.0}, FIT)

    assert received == [50.0, None]


def test_a_failed_step_stops_the_list(experiment):
    exp_dir, movie_dir = experiment

    messages = _run_steps(exp_dir, {**FIT, "threshold_on_std": 0.0}, OFFSET)

    assert _statuses(messages).count("error") == 1
    assert "finished" not in _statuses(messages)
    assert not (movie_dir / "Corrected_sample.tif").exists()
    (error,) = [m for m in messages if isinstance(m, dict) and m.get("status") == "error"]
    assert error["message"].startswith("[Step 1/2] ")


def test_chain_corrects_the_stack_of_the_first_step_in_place(experiment):
    exp_dir, movie_dir = experiment
    _run(exp_dir, **OFFSET)
    first = tifffile.imread(movie_dir / "Corrected_sample.tif")

    messages = _run(
        exp_dir,
        **{**OFFSET, "correction_horizontal": 2},
        movie_prefix="Corrected_sample",
        export_prefix=None,
    )

    assert "finished" in _statuses(messages)
    assert sorted(f.name for f in movie_dir.iterdir()) == ["Corrected_sample.tif", "sample.tif"]
    second = tifffile.imread(movie_dir / "Corrected_sample.tif")
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    "protocol, reason",
    [
        ({**MODEL_FREE, "threshold_on_std": 0.0}, "threshold is too low"),
        ({**FIT, "threshold_on_std": 0.0}, "background pixels"),
    ],
    ids=["model-free", "fit"],
)
def test_impossible_correction_reaches_the_progress_window(experiment, protocol, reason):
    exp_dir, movie_dir = experiment

    messages = _run(exp_dir, **protocol)

    (error,) = [m for m in messages if isinstance(m, dict) and m.get("status") == "error"]
    assert reason in error["message"]
    assert "finished" not in _statuses(messages)
    assert not (movie_dir / "Corrected_sample.tif").exists()
