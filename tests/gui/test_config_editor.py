"""
Unit tests for the experiment configuration editor in celldetective.gui.json_readers.

Covers the well labels table (one row per well, labels added, renamed and
removed), the metadata table, and that saving keeps the format the rest of the
software reads, and the prefixes the movie prefix field suggests.
"""

import configparser
import logging
import os

import pytest

from celldetective.gui import json_readers
from celldetective.gui.json_readers import (
    ConfigEditor,
    REQUIRED_LABELS,
    key_error,
    normalize_key,
    read_well_labels,
)

CONFIG = """
[Populations]
populations = targets,effectors

[MovieSettings]
pxtoum = 0.2
len_movie = 40
movie_prefix = 

[Labels]
cell_types = T,T,NK
antibodies = a,b,c
concentrations = 0,1,10
pharmaceutical_agents = None,None,None

[Metadata]
concentration_units = pM
"""


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def no_message_boxes(monkeypatch):
    """Record the messages instead of opening a blocking box."""
    messages = []
    monkeypatch.setattr(
        json_readers, "generic_message", lambda msg, kind="info": messages.append(msg)
    )
    return messages


class FakeControlPanel:
    """The part of the control panel the editor relies on."""

    def __init__(self, folder):
        self.exp_dir = folder + os.sep
        self.exp_config = os.path.join(folder, "config.ini")
        self.wells = [os.path.join(folder, f"W{i}") + os.sep for i in (1, 2, 3)]
        self.reloads = 0

    def load_configuration(self):
        self.reloads += 1


@pytest.fixture
def editor(qtbot, tmp_path):
    """Return an editor opened on a three well experiment."""
    (tmp_path / "config.ini").write_text(CONFIG)
    panel = FakeControlPanel(str(tmp_path))
    widget = ConfigEditor(panel)
    qtbot.addWidget(widget)
    return widget


def saved(editor):
    """Return the configuration written by the editor."""
    config = configparser.ConfigParser(interpolation=None)
    config.read(editor.config_path)
    return config


def test_normalize_key():
    assert normalize_key("  Culture Date ") == "culture_date"


@pytest.mark.parametrize("key", ["", "a=b", "a,b", "[x]"])
def test_key_error_rejects_bad_names(key):
    assert key_error(key, []) is not None


def test_key_error_rejects_duplicates():
    assert key_error("date", ["date"]) is not None
    assert key_error("date", ["time"]) is None


def test_read_well_labels_fits_values_to_wells():
    config = configparser.ConfigParser(interpolation=None)
    config.read_string("[Labels]\ncell_types = a,b,c,d\nextra = x\n")
    labels = read_well_labels(config, 3)
    assert labels["cell_types"] == ["a", "b", "c"]
    assert labels["extra"] == ["x", "", ""]
    assert labels["antibodies"] == ["0", "1", "2"]


def test_one_row_per_well(editor):
    table = editor.labels_table
    assert table.rowCount() == 3
    assert [table.verticalHeaderItem(r).text() for r in range(3)] == ["W1", "W2", "W3"]
    column = editor.label_keys.index("cell_types")
    assert [table.text(r, column) for r in range(3)] == ["T", "T", "NK"]


def test_add_label_and_save(editor):
    assert editor.add_label("Culture Medium")
    column = editor.label_keys.index("culture_medium")
    for row, value in enumerate(["RPMI", "DMEM", "RPMI"]):
        editor.labels_table.set_text(row, column, value)

    assert editor.save_config()
    config = saved(editor)
    assert config.get("Labels", "culture_medium") == "RPMI,DMEM,RPMI"
    assert config.get("Labels", "cell_types") == "T,T,NK"
    assert editor.parent_window.reloads == 1


def test_required_labels_are_kept(editor, no_message_boxes):
    column = editor.label_keys.index(REQUIRED_LABELS[0])
    assert not editor.remove_label(column)
    assert not editor.rename_label(column, "other")
    assert not editor.add_label("antibodies")
    assert no_message_boxes


def test_rename_and_remove_custom_label(editor):
    editor.add_label("medium")
    column = editor.label_keys.index("medium")
    assert editor.rename_label(column, "buffer")
    assert editor.label_keys[column] == "buffer"
    assert editor.remove_label(column)
    assert "buffer" not in editor.label_keys
    assert editor.labels_table.columnCount() == len(editor.label_keys)


def test_paste_block(editor):
    editor.add_label("medium")
    column = editor.label_keys.index("medium")
    editor.labels_table.setCurrentCell(0, column)
    editor.labels_table.paste("RPMI\nDMEM\nRPMI\nignored\n")
    assert [editor.labels_table.text(r, column) for r in range(3)] == ["RPMI", "DMEM", "RPMI"]


def test_comma_in_label_is_refused(editor, no_message_boxes):
    editor.labels_table.set_text(0, 0, "a,b")
    before = open(editor.config_path).read()
    assert not editor.save_config()
    assert no_message_boxes
    assert open(editor.config_path).read() == before


def test_metadata_entries(editor):
    table = editor.metadata_table
    assert table.text(0, 0) == "concentration_units"

    editor.add_metadata_entry()
    table.set_text(1, 0, "Date")
    table.set_text(1, 1, "today, 10am")
    editor.add_metadata_entry()  # left empty: skipped

    assert editor.save_config()
    config = saved(editor)
    assert dict(config.items("Metadata")) == {
        "concentration_units": "pM",
        "date": "today, 10am",
    }


def test_duplicate_metadata_key_is_refused(editor, no_message_boxes):
    editor.add_metadata_entry()
    editor.metadata_table.set_text(1, 0, "concentration_units")
    editor.metadata_table.set_text(1, 1, "nM")
    assert not editor.save_config()
    assert no_message_boxes


def test_settings_are_saved_in_order(editor):
    editor.fields[("MovieSettings", "pxtoum")].setText("0.5")
    assert editor.save_config()
    config = saved(editor)
    assert config.get("MovieSettings", "pxtoum") == "0.5"
    assert config.sections() == ["Populations", "MovieSettings", "Labels", "Metadata"]


@pytest.fixture
def open_editor(qtbot, tmp_path, write_stacks):
    """Return an opener of the editor over the stacks given, once it read them."""

    def open_(movies=None, prefix=""):
        write_stacks(tmp_path, movies or {})
        (tmp_path / "config.ini").write_text(
            CONFIG.replace("movie_prefix = \n", f"movie_prefix = {prefix}\n")
        )
        widget = ConfigEditor(FakeControlPanel(str(tmp_path)))
        qtbot.addWidget(widget)
        field = widget.prefix_widget
        qtbot.waitUntil(
            lambda: field.movies_per_position is not None or bool(field.scan_error)
        )
        return widget

    return open_


THREE_POSITIONS = {
    ("W1", "101"): ["Alexa488_stack.tif", "BF_stack.tif"],
    ("W2", "201"): ["Alexa488_stack.tif", "BF_stack.tif"],
    ("W3", "301"): ["BF_stack.tif"],
}


def test_prefix_suggestions_come_from_the_stacks(open_editor):
    editor = open_editor(THREE_POSITIONS)
    assert editor.prefix_widget.model.stringList() == ["BF_", "Alexa488_"]


def test_the_button_lists_every_prefix(open_editor):
    prefix = open_editor(THREE_POSITIONS).prefix_widget
    assert prefix.suggest_btn.isEnabled()
    prefix.field.setText("sample")
    prefix.show_suggestions()
    assert prefix.completer.completionCount() == 2


def test_the_stacks_are_read_only_once(open_editor, monkeypatch):
    calls = []
    monkeypatch.setattr(
        json_readers, "list_movies_per_position", lambda folder: calls.append(folder) or {}
    )
    prefix = open_editor().prefix_widget
    prefix.field.setText("stack")
    prefix.field.setText("stacks")
    assert len(calls) == 1


def test_the_stored_prefix_is_checked_on_opening(open_editor):
    prefix = open_editor(THREE_POSITIONS, prefix="sample").prefix_widget
    assert "No stack" in prefix.hint.text()


def test_the_hint_tells_what_the_prefix_matches(open_editor):
    prefix = open_editor(THREE_POSITIONS).prefix_widget
    prefix.field.setText("Alexa488_")
    assert "1 of the 3 positions" in prefix.hint.text()

    prefix.field.setText("BF_")
    assert prefix.hint.text() == "One stack in each of the 3 positions."

    prefix.field.setText("")
    assert "5 stacks over 3 positions" in prefix.hint.text()

    prefix.field.setText("Hoechst")
    assert "No stack" in prefix.hint.text()


def test_the_hint_counts_positions_without_movie_folder(open_editor, tmp_path):
    (tmp_path / "W2" / "201").mkdir(parents=True)
    prefix = open_editor({("W1", "101"): ["BF_stack.tif"]}).prefix_widget
    prefix.field.setText("BF_")
    assert "1 of the 2 positions" in prefix.hint.text()


def test_the_hint_signals_an_experiment_without_positions(open_editor):
    prefix = open_editor().prefix_widget
    assert "No position" in prefix.hint.text()
    # Nothing to suggest: the button stays off.
    assert not prefix.suggest_btn.isEnabled()


def test_the_hint_signals_a_folder_that_cannot_be_read(open_editor, monkeypatch):
    def unreachable(folder):
        raise OSError("unreachable share")

    monkeypatch.setattr(json_readers, "list_movies_per_position", unreachable)
    assert "could not be read" in open_editor().prefix_widget.hint.text()


def test_the_prefix_is_saved(open_editor):
    editor = open_editor({("W1", "101"): ["Alexa488_stack.tif"]})
    editor.prefix_widget.field.setText("Alexa488_")
    assert editor.save_config()
    assert saved(editor).get("MovieSettings", "movie_prefix") == "Alexa488_"
