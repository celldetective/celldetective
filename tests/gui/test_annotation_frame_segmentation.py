"""
Unit tests for running a segmentation model from the annotation corrector.

The "Correct a segmentation annotation" plugin opens a standalone image, not a
position, so the panel it embeds cannot read the experiment: the channels and
the calibration have to come from the sidecar written at export time. These
tests cover that hand-off, plus the panel accepting them in place of a position.
"""

import json
import logging
import os

import numpy as np
import pytest

from celldetective.napari import frame_segmentation as fs
from celldetective.napari import utils as napari_utils


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _annotation(tmp_path, info=None, name="exp_A1_0000.tif"):
    """Write an annotation image, with its sidecar when `info` is given."""
    image = tmp_path / name
    image.write_bytes(b"")  # never read by the helpers under test
    if info is not None:
        (tmp_path / name.replace(".tif", ".json")).write_text(json.dumps(info))
    return str(image)


class TestAnnotationMetadata:
    """What the model needs to be told is what the export already recorded."""

    def test_the_sidecar_gives_the_channels_and_the_calibration(self, tmp_path):
        path = _annotation(
            tmp_path,
            {"channels": ["brightfield", "dead_nuclei"], "spatial_calibration": 0.31},
        )
        channels, calibration = napari_utils._annotation_metadata(path)
        assert channels == ["brightfield", "dead_nuclei"]
        assert calibration == pytest.approx(0.31)

    def test_a_missing_sidecar_is_not_an_error(self, tmp_path):
        assert napari_utils._annotation_metadata(_annotation(tmp_path)) == (None, None)

    def test_an_unreadable_sidecar_is_not_an_error(self, tmp_path):
        path = _annotation(tmp_path)
        (tmp_path / "exp_A1_0000.json").write_text("{not json")
        assert napari_utils._annotation_metadata(path) == (None, None)

    def test_an_unusable_calibration_is_dropped(self, tmp_path):
        path = _annotation(
            tmp_path, {"channels": ["brightfield"], "spatial_calibration": "n/a"}
        )
        channels, calibration = napari_utils._annotation_metadata(path)
        assert channels == ["brightfield"]
        assert calibration is None

    def test_an_empty_channel_list_reads_as_unknown(self, tmp_path):
        path = _annotation(tmp_path, {"channels": [], "spatial_calibration": 0.5})
        channels, calibration = napari_utils._annotation_metadata(path)
        assert channels is None
        assert calibration == pytest.approx(0.5)


class TestAnnotationPopulation:
    """The folder is the only record of which model family to offer."""

    def test_the_folder_names_the_population(self, tmp_path):
        folder = tmp_path / "annotations_effectors"
        folder.mkdir()
        assert (
            napari_utils._annotation_population(str(folder / "im.tif")) == "effectors"
        )

    def test_anything_else_falls_back_to_the_targets(self, tmp_path):
        assert napari_utils._annotation_population(str(tmp_path / "im.tif")) == "targets"

    def test_a_bare_prefix_falls_back_too(self, tmp_path):
        folder = tmp_path / "annotations_"
        folder.mkdir()
        assert napari_utils._annotation_population(str(folder / "im.tif")) == "targets"


class TestPanelWithoutAPosition:
    """
    The panel has to build from what the sidecar knows.

    Only the attributes the run actually reads are checked here; building the
    widget itself needs a viewer, so construction is bypassed the same way the
    rest of the panel's unit tests do it.
    """

    def test_given_channels_and_calibration_are_used(self, qtbot, monkeypatch):
        monkeypatch.setattr(fs.FrameSegmentationPanel, "_build", lambda self: None)
        monkeypatch.setattr(
            fs.FrameSegmentationPanel, "_install_close_hook", lambda self: None
        )
        panel = fs.FrameSegmentationPanel(
            viewer=None,
            stack=np.zeros((1, 4, 4, 2)),
            position=None,
            population="targets",
            channels=["brightfield", "dead_nuclei"],
            spatial_calibration=0.31,
        )
        qtbot.addWidget(panel)
        assert panel.experiment is None
        assert panel.exp_channels == ["brightfield", "dead_nuclei"]
        assert panel.spatial_calibration == pytest.approx(0.31)

    def test_without_a_position_nothing_is_read_from_an_experiment(
        self, qtbot, monkeypatch
    ):
        def _fail(*args, **kwargs):
            raise AssertionError("the experiment must not be consulted")

        monkeypatch.setattr(fs, "extract_experiment_from_position", _fail)
        monkeypatch.setattr(fs, "extract_experiment_channels", _fail)
        monkeypatch.setattr(fs, "get_spatial_calibration", _fail)
        monkeypatch.setattr(fs.FrameSegmentationPanel, "_build", lambda self: None)
        monkeypatch.setattr(
            fs.FrameSegmentationPanel, "_install_close_hook", lambda self: None
        )

        panel = fs.FrameSegmentationPanel(
            viewer=None, stack=np.zeros((1, 4, 4, 1)), position=None
        )
        qtbot.addWidget(panel)
        assert panel.exp_channels == []
        assert panel.spatial_calibration is None

    def test_a_position_still_wins_when_nothing_is_passed(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            fs, "extract_experiment_from_position", lambda position: "/exp"
        )
        monkeypatch.setattr(
            fs, "extract_experiment_channels", lambda exp: (["brightfield"], [0])
        )
        monkeypatch.setattr(fs, "get_spatial_calibration", lambda exp: 0.25)
        monkeypatch.setattr(fs.FrameSegmentationPanel, "_build", lambda self: None)
        monkeypatch.setattr(
            fs.FrameSegmentationPanel, "_install_close_hook", lambda self: None
        )

        panel = fs.FrameSegmentationPanel(
            viewer=None,
            stack=np.zeros((1, 4, 4, 1)),
            position=f"/exp{os.sep}W1{os.sep}100{os.sep}",
        )
        qtbot.addWidget(panel)
        assert panel.exp_channels == ["brightfield"]
        assert panel.spatial_calibration == pytest.approx(0.25)
