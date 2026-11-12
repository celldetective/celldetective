"""
The annotators ask before closing with annotations that were not saved.

Tested on a bare window carrying the shared annotator behaviour: opening a real
annotator needs an experiment on disk.
"""

import logging

import pytest
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from celldetective.gui.annotator_common import AnnotatorMixin


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class _Annotator(AnnotatorMixin, QMainWindow):
    mode = "targets"

    def __init__(self, tmp_path):
        super().__init__()
        self.pos = str(tmp_path) + "/"
        self.annotation_log = []


@pytest.fixture
def answers(monkeypatch):
    """Answer the close question with the values pushed, recording each ask."""
    state = {"answer": QMessageBox.No, "asked": 0}

    def question(*args, **kwargs):
        state["asked"] += 1
        return state["answer"]

    monkeypatch.setattr(QMessageBox, "question", question)
    return state


def _window(qtbot, tmp_path):
    window = _Annotator(tmp_path)
    qtbot.addWidget(window)
    window.show()
    return window


def test_a_window_with_nothing_unsaved_closes_without_asking(qtbot, tmp_path, answers):
    window = _window(qtbot, tmp_path)
    assert window.close()
    assert answers["asked"] == 0


def test_an_unsaved_annotation_is_asked_about_and_declining_keeps_the_window(
    qtbot, tmp_path, answers
):
    window = _window(qtbot, tmp_path)
    window.annotation_log.append("track 3: class 0, time 12")
    assert not window.close()
    assert window.isVisible() and answers["asked"] == 1


def test_accepting_closes_and_is_not_asked_twice(qtbot, tmp_path, answers):
    window = _window(qtbot, tmp_path)
    window.annotation_log.append("track 3: class 0, time 12")
    answers["answer"] = QMessageBox.Yes
    assert window.confirm_discard()
    assert window.close()
    assert answers["asked"] == 1


def test_a_class_added_or_deleted_counts_as_unsaved(qtbot, tmp_path, answers):
    window = _window(qtbot, tmp_path)
    window.mark_table_changed()
    assert window.has_unsaved_annotations()
    assert not window.close()


def test_saving_clears_both(qtbot, tmp_path, answers):
    window = _window(qtbot, tmp_path)
    window.annotation_log.append("track 3: class 0, time 12")
    window.mark_table_changed()
    window.write_annotation_log("EVENT ANNOTATION (manual)", "class_name: class")
    assert not window.has_unsaved_annotations()
    assert window.close()
    assert answers["asked"] == 0
