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


def write_stacks(folder, movies):
    """Write empty stacks in the movie folders of an experiment."""
    for (well, position), names in movies.items():
        movie_folder = folder / well / position / "movie"
        movie_folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (movie_folder / name).write_bytes(b"")


def test_prefix_suggestions_come_from_the_stacks(editor, tmp_path):
    write_stacks(
        tmp_path,
        {
            ("W1", "101"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W2", "201"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W3", "301"): ["Alexa488_stack.tif", "BF_stack.tif"],
        },
    )
    editor.show_prefix_suggestions()
    assert editor.prefix_model.stringList() == ["Alexa488_", "BF_"]


def test_the_stacks_are_read_only_once(editor, tmp_path, monkeypatch):
    write_stacks(tmp_path, {("W1", "101"): ["stack.tif"]})
    calls = []
    monkeypatch.setattr(
        json_readers, "list_movies_per_position", lambda folder: calls.append(folder) or {}
    )
    editor.scan_movies()
    editor.scan_movies()
    editor.prefix_field.setText("stack")
    assert len(calls) == 1


def test_the_hint_tells_what_the_prefix_matches(editor, tmp_path):
    write_stacks(
        tmp_path,
        {
            ("W1", "101"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W2", "201"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W3", "301"): ["BF_stack.tif"],
        },
    )
    editor.prefix_field.setText("Alexa488_")
    assert "1 of the 3 positions" in editor.prefix_hint.text()

    editor.prefix_field.setText("BF_")
    assert editor.prefix_hint.text() == "One stack in each of the 3 positions."

    editor.prefix_field.setText("")
    assert "first one" in editor.prefix_hint.text()

    editor.prefix_field.setText("Hoechst")
    assert "No stack" in editor.prefix_hint.text()


def test_the_hint_signals_an_experiment_without_movies(editor):
    editor.prefix_field.setText("a")
    assert "No movie folder" in editor.prefix_hint.text()


def test_the_prefix_is_saved(editor, tmp_path):
    write_stacks(tmp_path, {("W1", "101"): ["Alexa488_stack.tif"]})
    editor.prefix_field.setText("Alexa488_")
    assert editor.save_config()
    assert saved(editor).get("MovieSettings", "movie_prefix") == "Alexa488_"
