"""
Unit tests for threshold-segmenting the frame on screen in napari.

Covers remembering the last configurations of an experiment, applying a
configuration to one frame of a stack, turning shapes into a region, the rules
for writing labels within that region, and the panel itself against a stub
viewer -- no napari window is opened.
"""

import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from celldetective.napari import threshold_segmentation as ts
from celldetective.utils import threshold_configs as tc


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _config(**overrides):
    config = {
        "target_channel": "nuclei",
        "thresholds": [100, 1e9],
        "filters": [],
        "marker_min_distance": 5,
        "marker_footprint_size": 5,
        "feature_queries": [""],
        "equalize_reference": [False, 0],
        "do_watershed": False,
        "fill_holes": True,
    }
    config.update(overrides)
    return config


def _write_config(path, **overrides):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_config(**overrides)))
    return str(path)


def _stack(n_frames=2):
    """TYXC stack: two bright squares on the second channel, a dim one elsewhere."""
    stack = np.zeros((n_frames, 40, 40, 2), dtype=np.uint16)
    stack[:, 5:12, 5:12, 1] = 500
    stack[:, 25:32, 25:32, 1] = 500
    stack[:, 15:20, 15:20, 0] = 500
    return stack


class TestRememberedConfigs:
    """The last configurations of an experiment come back the next session."""

    def test_a_config_is_recalled(self, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", path)
        assert tc.recall_threshold_configs(str(tmp_path), "targets") == [
            os.path.normpath(path)
        ]

    def test_paths_in_the_experiment_are_stored_relative(self, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", [path])
        memory = json.loads((tmp_path / "configs" / tc.MEMORY_FILENAME).read_text())
        assert memory["targets"] == ["configs/thr.json"]

    def test_the_record_survives_moving_the_experiment(self, tmp_path):
        old = tmp_path / "old"
        _write_config(old / "configs" / "thr.json")
        tc.remember_threshold_configs(
            str(old), "targets", str(old / "configs" / "thr.json")
        )
        new = tmp_path / "new"
        old.rename(new)
        assert tc.recall_threshold_configs(str(new), "targets") == [
            os.path.normpath(str(new / "configs" / "thr.json"))
        ]

    def test_singular_and_plural_populations_are_the_same(self, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", path)
        assert tc.recall_threshold_configs(str(tmp_path), "target") == [
            os.path.normpath(path)
        ]

    def test_populations_are_kept_apart(self, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", path)
        assert tc.recall_threshold_configs(str(tmp_path), "effectors") == []

    def test_a_deleted_config_is_not_offered(self, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", path)
        os.remove(path)
        assert tc.recall_threshold_configs(str(tmp_path), "targets") == []

    def test_an_unreadable_record_is_not_an_error(self, tmp_path):
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / tc.MEMORY_FILENAME).write_text("{not json")
        assert tc.recall_threshold_configs(str(tmp_path), "targets") == []

    def test_no_experiment_means_nothing_recorded(self, tmp_path):
        tc.remember_threshold_configs(None, "targets", "x.json")
        assert tc.recall_threshold_configs(None, "targets") == []
        # A mocked experiment path must not break the main window either.
        tc.remember_threshold_configs(MagicMock(), "targets", "x.json")
        assert tc.recall_threshold_configs(MagicMock(), "targets") == []

    def test_a_file_that_is_not_a_config_is_refused(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"model_name": "x"}))
        with pytest.raises(ValueError, match="target_channel"):
            tc.load_threshold_config(str(path))


class TestThresholdFrame:
    """A configuration is applied to one frame as the batch pipeline applies it."""

    def test_the_channel_is_resolved_by_name(self):
        labels = ts.threshold_frame(
            _stack(), 0, [_config(target_channel="nuclei")], ["brightfield", "nuclei"]
        )
        assert len(np.unique(labels)) - 1 == 2
        assert labels[8, 8] > 0 and labels[17, 17] == 0

    def test_an_index_from_an_older_config_is_accepted(self):
        labels = ts.threshold_frame(
            _stack(), 0, [_config(target_channel=0)], ["brightfield", "nuclei"]
        )
        assert labels[17, 17] > 0 and labels[8, 8] == 0

    def test_an_unknown_channel_is_reported(self):
        with pytest.raises(ValueError, match="'dapi'"):
            ts.threshold_frame(
                _stack(), 0, [_config(target_channel="dapi")], ["brightfield", "nuclei"]
            )

    def test_the_equalization_reference_is_taken_from_the_stack(self, monkeypatch):
        import celldetective.segmentation as seg

        seen = {}
        real = seg.segment_frame_from_thresholds

        def spy(frame, **kwargs):
            seen.update(kwargs)
            return real(frame, **kwargs)

        monkeypatch.setattr(seg, "segment_frame_from_thresholds", spy)
        stack = _stack(3)
        stack[2, 0, 0, 1] = 7
        ts.threshold_frame(
            stack,
            0,
            [_config(equalize_reference=[True, 2])],
            ["brightfield", "nuclei"],
        )
        assert seen["equalize_reference"].shape == (40, 40)
        assert seen["equalize_reference"][0, 0] == 7

    def test_a_reference_outside_the_stack_is_skipped(self):
        labels = ts.threshold_frame(
            _stack(1),
            0,
            [_config(equalize_reference=[True, 5])],
            ["brightfield", "nuclei"],
        )
        assert labels.max() > 0


def _shapes(data, shape_type):
    return SimpleNamespace(data=[np.asarray(d, dtype=float) for d in data], shape_type=shape_type)


class TestShapesRegion:
    """Only the shapes drawn on the frame, and only those enclosing an area."""

    def test_a_rectangle_on_this_frame(self):
        rect = [[3, 10, 10], [3, 10, 20], [3, 20, 20], [3, 20, 10]]
        region, n, source = ts.shapes_region(_shapes([rect], ["rectangle"]), 3, (40, 40))
        assert n == 1 and source == 3
        assert region[15, 15] and not region[5, 5]

    def test_shapes_on_later_frames_are_ignored(self):
        rect = [[5, 10, 10], [5, 10, 20], [5, 20, 20], [5, 20, 10]]
        region, n, _ = ts.shapes_region(_shapes([rect], ["rectangle"]), 3, (40, 40))
        assert region is None and n == 0

    def test_shapes_carry_on_from_the_closest_earlier_frame(self):
        """Regions drawn once apply to every upcoming frame without redrawing."""
        old = [[0, 0, 0], [0, 0, 5], [0, 5, 5], [0, 5, 0]]
        recent = [[2, 10, 10], [2, 10, 20], [2, 20, 20], [2, 20, 10]]
        region, n, source = ts.shapes_region(
            _shapes([old, recent], ["rectangle", "rectangle"]), 7, (40, 40)
        )
        assert source == 2 and n == 1
        assert region[15, 15] and not region[2, 2]

    def test_shapes_of_this_frame_win_over_earlier_ones(self):
        old = [[0, 0, 0], [0, 0, 5], [0, 5, 5], [0, 5, 0]]
        now = [[4, 10, 10], [4, 10, 20], [4, 20, 20], [4, 20, 10]]
        region, _, source = ts.shapes_region(
            _shapes([old, now], ["rectangle", "rectangle"]), 4, (40, 40)
        )
        assert source == 4 and not region[2, 2]

    def test_a_2d_layer_applies_to_every_frame(self):
        rect = [[10, 10], [10, 20], [20, 20], [20, 10]]
        region, n, source = ts.shapes_region(_shapes([rect], ["rectangle"]), 7, (40, 40))
        assert n == 1 and region[15, 15] and source is None

    def test_lines_and_paths_enclose_nothing(self):
        line = [[0, 1, 1], [0, 30, 30]]
        path = [[0, 1, 1], [0, 30, 1], [0, 30, 30]]
        region, n, _ = ts.shapes_region(_shapes([line, path], ["line", "path"]), 0, (40, 40))
        assert region is None

    def test_an_ellipse_is_not_its_bounding_box(self):
        box = [[0, 10, 10], [0, 10, 30], [0, 30, 30], [0, 30, 10]]
        region, _, _ = ts.shapes_region(_shapes([box], ["ellipse"]), 0, (40, 40))
        assert region[20, 20]
        assert not region[11, 11]

    def test_several_shapes_are_joined(self):
        a = [[0, 0, 0], [0, 0, 5], [0, 5, 5], [0, 5, 0]]
        b = [[0, 30, 30], [0, 30, 35], [0, 35, 35], [0, 35, 30]]
        region, n, _ = ts.shapes_region(_shapes([a, b], ["rectangle", "polygon"]), 0, (40, 40))
        assert n == 2 and region[2, 2] and region[32, 32]


class TestCombineLabels:
    """What is written, and what is kept, in the whole frame or a region."""

    def setup_method(self):
        self.current = np.zeros((20, 20), dtype=np.uint16)
        self.current[1:4, 1:4] = 5  # inside the region
        self.current[15:18, 15:18] = 9  # outside
        self.new = np.zeros((20, 20), dtype=np.int32)
        self.new[2:6, 2:6] = 1  # inside, overlapping label 5
        self.new[12:14, 1:3] = 2  # outside
        self.region = np.zeros((20, 20), dtype=bool)
        self.region[:10, :10] = True

    def test_replacing_the_whole_frame(self):
        out = ts.combine_labels(self.current, self.new, replace=True)
        assert set(np.unique(out)) == {0, 1, 2}

    def test_merging_on_the_whole_frame_keeps_what_is_drawn(self):
        out = ts.combine_labels(self.current, self.new, replace=False)
        assert out[2, 2] == 5 and out[16, 16] == 9
        # New objects fill the background, numbered past what is there.
        assert out[5, 5] > 9 and out[12, 1] > 9

    def test_replacing_in_a_region_leaves_the_rest_alone(self):
        out = ts.combine_labels(self.current, self.new, region=self.region, replace=True)
        assert 5 not in out  # the old object in the region is gone
        assert out[16, 16] == 9  # the one outside is kept
        assert out[2, 2] > 0 and out[12, 1] == 0  # only the new one inside

    def test_merging_in_a_region_only_fills_the_background(self):
        out = ts.combine_labels(self.current, self.new, region=self.region, replace=False)
        assert out[2, 2] == 5 and out[5, 5] not in (0, 5, 9)
        assert out[12, 1] == 0

    def test_an_object_across_the_edge_is_kept_whole(self):
        new = np.zeros((20, 20), dtype=np.int32)
        new[6:13, 6:13] = 1  # centroid (9, 9) inside a region ending at 10
        out = ts.combine_labels(np.zeros((20, 20), np.uint16), new, region=self.region)
        assert out[12, 12] > 0


class _Layers(dict):
    """Just enough of napari's layer list for the panel."""

    def __init__(self, *layers):
        super().__init__((layer.name, layer) for layer in layers)
        self.events = MagicMock()

    def __iter__(self):
        return iter(self.values())


def _viewer(frame=0, n_frames=2):
    labels = SimpleNamespace(
        name="segmentation",
        data=np.zeros((n_frames, 40, 40), dtype=np.uint16),
        ndim=3,
        refresh=MagicMock(),
        _save_history=MagicMock(),
    )
    viewer = SimpleNamespace(
        layers=_Layers(labels),
        dims=SimpleNamespace(current_step=(frame, 0, 0)),
        status="",
    )
    return viewer, labels


class TestPanel:

    def _panel(self, qtbot, viewer, exp_dir=None):
        panel = ts.ThresholdSegmentationPanel(
            viewer=viewer,
            stack=_stack(),
            channels=["brightfield", "nuclei"],
            exp_dir=exp_dir,
        )
        qtbot.addWidget(panel)
        return panel

    def test_nothing_to_run_without_a_config(self, qtbot):
        viewer, _ = _viewer()
        panel = self._panel(qtbot, viewer)
        assert not panel.run_btn.isEnabled()
        # No position, so no movie for the wizard to open on.
        assert not panel.wizard_btn.isEnabled()

    def test_the_last_config_is_there_on_opening(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        tc.remember_threshold_configs(str(tmp_path), "targets", path)
        viewer, _ = _viewer()
        panel = self._panel(qtbot, viewer, exp_dir=str(tmp_path))
        assert panel.config_lbl.text() == "thr.json"
        assert panel.run_btn.isEnabled()

    def test_a_loaded_config_is_remembered(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        viewer, _ = _viewer()
        panel = self._panel(qtbot, viewer, exp_dir=str(tmp_path))
        panel.load_configs([path])
        assert tc.recall_threshold_configs(str(tmp_path), "targets") == [
            os.path.normpath(path)
        ]

    def test_a_config_saved_by_the_wizard_is_picked_up(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "configs" / "thr.json")
        viewer, _ = _viewer()
        panel = self._panel(qtbot, viewer, exp_dir=str(tmp_path))
        panel._on_wizard_saved(path)
        assert panel.config_paths == [path]

    def test_shapes_layers_are_offered_as_regions(self, qtbot):
        viewer, labels = _viewer()
        panel = self._panel(qtbot, viewer)
        roi = type("Shapes", (), {})()
        roi.name = "cells of interest"
        viewer.layers["cells of interest"] = roi
        panel._refresh_regions()
        items = [panel.region_cb.itemText(i) for i in range(panel.region_cb.count())]
        assert items == [ts.WHOLE_FRAME, "cells of interest"]

    def test_the_current_frame_is_thresholded(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer(frame=1)
        panel = self._panel(qtbot, viewer)
        panel.load_configs([path])
        panel.threshold_current_frame()
        qtbot.waitUntil(lambda: panel._worker is None, timeout=30000)
        assert labels.data[0].max() == 0
        assert len(np.unique(labels.data[1])) - 1 == 2
        labels._save_history.assert_called_once()

    def test_an_empty_region_is_reported_before_running(self, qtbot, tmp_path, monkeypatch):
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer(frame=1)
        panel = self._panel(qtbot, viewer)
        panel.load_configs([path])
        shapes = type("Shapes", (), {})()
        shapes.name = "ROIs"
        shapes.data = []
        shapes.shape_type = []
        viewer.layers["ROIs"] = shapes
        panel._refresh_regions()
        panel.region_cb.setCurrentText("ROIs")
        failures = []
        monkeypatch.setattr(panel, "_failed", failures.append)
        panel.threshold_current_frame()
        assert panel._worker is None
        assert failures and "frame 1" in failures[0]

    def test_labels_that_do_not_match_the_movie_are_reported(
        self, qtbot, tmp_path, monkeypatch
    ):
        """Transposed labels used to surface as a numpy broadcasting error."""
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer()
        labels.data = np.zeros((2, 30, 40), dtype=np.uint16)
        panel = self._panel(qtbot, viewer)
        panel.load_configs([path])
        failures = []
        monkeypatch.setattr(panel, "_failed", failures.append)
        panel.threshold_current_frame()
        assert panel._worker is None
        assert failures and "30x40" in failures[0] and "40x40" in failures[0]

    def _run(self, qtbot, panel):
        panel.threshold_current_frame()
        qtbot.waitUntil(lambda: panel._worker is None, timeout=30000)

    def test_the_following_frames_are_thresholded_too(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer(frame=1, n_frames=4)
        panel = ts.ThresholdSegmentationPanel(
            viewer=viewer, stack=_stack(4), channels=["brightfield", "nuclei"]
        )
        qtbot.addWidget(panel)
        panel.load_configs([path])
        panel.following_cb.setChecked(True)
        assert panel.run_btn.text() == "Threshold from this frame on"
        self._run(qtbot, panel)
        assert labels.data[0].max() == 0
        for t in (1, 2, 3):
            assert len(np.unique(labels.data[t])) - 1 == 2
        # The whole run is taken back with one Ctrl+Z.
        labels._save_history.assert_called_once()
        indices = labels._save_history.call_args[0][0][0]
        assert set(np.unique(indices[0])) == {1, 2, 3}

    def test_the_roi_of_this_frame_applies_to_the_following_ones(self, qtbot, tmp_path):
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer(frame=1, n_frames=3)
        panel = ts.ThresholdSegmentationPanel(
            viewer=viewer, stack=_stack(3), channels=["brightfield", "nuclei"]
        )
        qtbot.addWidget(panel)
        panel.load_configs([path])
        shapes = type("Shapes", (), {})()
        shapes.name = "ROIs"
        # Around the top-left square only, drawn on frame 0.
        shapes.data = [np.array([[0, 0, 0], [0, 0, 15], [0, 15, 15], [0, 15, 0]], float)]
        shapes.shape_type = ["rectangle"]
        viewer.layers["ROIs"] = shapes
        panel._refresh_regions()
        panel.region_cb.setCurrentText("ROIs")
        panel.following_cb.setChecked(True)
        self._run(qtbot, panel)
        for t in (1, 2):
            assert labels.data[t][8, 8] > 0
            assert labels.data[t][28, 28] == 0
        assert labels.data[0].max() == 0

    def test_a_cancelled_run_keeps_the_frames_done(self, qtbot, tmp_path, monkeypatch):
        path = _write_config(tmp_path / "thr.json")
        viewer, labels = _viewer(frame=0, n_frames=5)
        panel = ts.ThresholdSegmentationPanel(
            viewer=viewer, stack=_stack(5), channels=["brightfield", "nuclei"]
        )
        qtbot.addWidget(panel)
        panel.load_configs([path])
        panel.following_cb.setChecked(True)

        real = ts.threshold_frame

        def slow_after_first(stack, t, configs, names):
            if t == 1:
                panel._worker.cancel()
            return real(stack, t, configs, names)

        monkeypatch.setattr(ts, "threshold_frame", slow_after_first)
        self._run(qtbot, panel)
        assert labels.data[0].max() > 0 and labels.data[1].max() > 0
        assert labels.data[4].max() == 0
        assert panel.run_btn.text() == "Threshold from this frame on"
