"""The process Submit launches: what it runs, and what it tells the progress window."""

import numpy as np
import pytest
import tifffile

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
