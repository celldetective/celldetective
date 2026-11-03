"""Submit of the preprocessing panel: scope, the warning on corrected movies, and the chain."""

from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QDialog, QMessageBox

from celldetective.gui.preprocessing_block import PreprocessingPanel
from tests.preprocessing_helpers import write_config, write_experiment

OFFSET = {"correction_type": "offset", "target_channel": "Channel1"}


class Selection:
    def __init__(self, indices):
        self.indices = indices

    def selectAll(self):
        pass

    def getSelectedIndices(self):
        return self.indices


@pytest.fixture
def experiment(tmp_path):
    _, (movie_dir,) = write_experiment(tmp_path, [1], frames=2)
    return tmp_path / "Experiment", movie_dir


def _leave_corrected_stack(movie_dir):
    (movie_dir / "Corrected_sample.tif").write_bytes((movie_dir / "sample.tif").read_bytes())


def _panel(experiment_dir, protocols):
    panel = SimpleNamespace(
        exp_dir=str(experiment_dir) + "/",
        protocol_layout=SimpleNamespace(protocols=protocols),
        parent_window=SimpleNamespace(
            well_list=Selection([0]), position_list=Selection([0])
        ),
    )
    panel.confirm_correcting_corrected_movies = (
        lambda *args: PreprocessingPanel.confirm_correcting_corrected_movies(panel, *args)
    )
    return panel


@pytest.fixture
def launched(monkeypatch):
    """The process arguments of each step, without running them."""
    runs = []

    class Window:
        def __init__(self, process, parent_window=None, title="", position_info=True, process_args=None):
            runs.append(process_args)

        def exec_(self):
            return QDialog.Accepted

    monkeypatch.setattr("celldetective.gui.workers.ProgressWindow", Window)
    return runs


@pytest.fixture
def answers(monkeypatch):
    """Answer the message boxes in turn, and keep what they showed."""
    shown = []
    queue = []

    def fake_exec(box):
        shown.append(box.text() + "\n" + box.informativeText())
        return queue.pop(0)

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    return SimpleNamespace(shown=shown, queue=queue)


def test_rerun_on_corrected_movie_shows_the_new_name(experiment, launched, answers, qtbot):
    experiment_dir, movie_dir = experiment
    write_config(experiment_dir, "")
    _leave_corrected_stack(movie_dir)
    answers.queue += [QMessageBox.Yes, QMessageBox.Cancel]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, [dict(OFFSET)]))

    warning = answers.shown[-1]
    assert "Reads: Corrected_sample.tif" in warning
    assert "Writes: Corrected_Corrected_sample.tif" in warning
    assert launched == []


def test_accepting_the_warning_runs_the_correction(experiment, launched, answers, qtbot):
    experiment_dir, movie_dir = experiment
    write_config(experiment_dir, "")
    _leave_corrected_stack(movie_dir)
    answers.queue += [QMessageBox.Yes, QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, [dict(OFFSET)]))

    assert len(launched) == 1


def test_raw_movie_does_not_warn(experiment, launched, answers, qtbot):
    experiment_dir, movie_dir = experiment
    write_config(experiment_dir, "sample")
    _leave_corrected_stack(movie_dir)
    answers.queue += [QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, [dict(OFFSET)]))

    assert len(answers.shown) == 1
    assert len(launched) == 1


def test_all_positions_are_not_sent_as_the_indices_of_the_first_well(
    experiment, launched, answers, qtbot
):
    experiment_dir, _ = experiment
    write_config(experiment_dir, "sample")
    answers.queue += [QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, [dict(OFFSET)]))

    assert launched[0]["well_option"] == "*"
    assert launched[0]["position_option"] == "*"


def test_selected_positions_are_sent_as_selected(experiment, launched, answers, qtbot):
    experiment_dir, _ = experiment
    write_config(experiment_dir, "sample")
    answers.queue += [QMessageBox.No, QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, [dict(OFFSET)]))

    assert launched[0]["well_option"] == [0]
    assert launched[0]["position_option"] == [0]


def test_empty_selection_warns_and_runs_nothing(experiment, launched, answers, monkeypatch, qtbot):
    experiment_dir, _ = experiment
    write_config(experiment_dir, "sample")
    warned = []
    monkeypatch.setattr(
        "celldetective.gui.preprocessing_block.generic_message",
        lambda *args, **kwargs: warned.append(args),
    )
    answers.queue += [QMessageBox.No, QMessageBox.Yes]
    panel = _panel(experiment_dir, [dict(OFFSET)])
    panel.parent_window.position_list = Selection([])

    PreprocessingPanel.launch_preprocessing(panel)

    assert warned
    assert launched == []


def test_chain_reads_the_stack_of_the_first_step(experiment, launched, answers, qtbot):
    experiment_dir, _ = experiment
    write_config(experiment_dir, "sample")
    answers.queue += [QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(
        _panel(experiment_dir, [dict(OFFSET), dict(OFFSET)])
    )

    first, second = launched
    assert (first["movie_prefix"], first["export_prefix"]) == (None, "Corrected")
    assert (second["movie_prefix"], second["export_prefix"]) == ("Corrected_sample", None)


def test_empty_protocol_list_asks_nothing_about_the_movies(experiment, launched, answers, qtbot):
    experiment_dir, movie_dir = experiment
    write_config(experiment_dir, "")
    _leave_corrected_stack(movie_dir)
    answers.queue += [QMessageBox.Yes]

    PreprocessingPanel.launch_preprocessing(_panel(experiment_dir, []))

    assert len(answers.shown) == 1
    assert launched == []
