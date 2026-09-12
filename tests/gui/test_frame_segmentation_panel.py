"""
Unit tests for the napari single-frame segmentation panel.

Covers the parts that can be exercised without a napari viewer or a real model:
the label-writing rules (dtype safety, merging, undo), the prepared-model cache,
and the model listing. The panel itself is built against a stub viewer, so these
stay fast and do not touch the GPU or the model repository.
"""

import logging
from collections import OrderedDict
from unittest.mock import MagicMock

import numpy as np
import pytest
from PyQt5.QtWidgets import QVBoxLayout

from celldetective.gui.base.model_channel_selection import ModelChannelSelection
from celldetective.napari import frame_segmentation as fs


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


class TestFitToLayerDtype:
    """
    Labels must never be wrapped round to fit the layer.

    napari keeps the labels in the integer type they were read as, often uint16.
    A silent wrap merges unrelated cells under one identifier, and nothing
    downstream can tell that apart from a genuine segmentation.
    """

    def test_labels_that_fit_are_cast(self):
        labels = np.array([[0, 1], [2, 3]], dtype=np.int64)
        out = fs._fit_to_layer_dtype(labels, np.dtype(np.uint16))
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, labels)

    def test_labels_that_do_not_fit_are_refused(self):
        labels = np.array([[0, 70000]], dtype=np.int64)
        with pytest.raises(ValueError) as excinfo:
            fs._fit_to_layer_dtype(labels, np.dtype(np.uint16))
        assert "70000" in str(excinfo.value)
        assert "65535" in str(excinfo.value)

    def test_the_remedy_is_included_in_the_message(self):
        labels = np.array([[300]], dtype=np.int64)
        with pytest.raises(ValueError) as excinfo:
            fs._fit_to_layer_dtype(labels, np.dtype(np.uint8), "Do the other thing.")
        assert "Do the other thing." in str(excinfo.value)

    def test_an_empty_frame_is_allowed(self):
        labels = np.zeros((0, 0), dtype=np.int64)
        assert fs._fit_to_layer_dtype(labels, np.dtype(np.uint8)).dtype == np.uint8


def _panel():
    """A panel with construction bypassed, for testing its pure logic."""
    panel = fs.FrameSegmentationPanel.__new__(fs.FrameSegmentationPanel)
    panel._prepared = OrderedDict()
    panel._closing = False
    return panel


class TestMergedLabels:
    """Merging must not collide with, or overwrite, what is already drawn."""

    def test_new_labels_only_fill_the_background(self):
        panel = _panel()
        current = np.array([[7, 0], [0, 0]], dtype=np.uint16)
        new = np.array([[1, 1], [2, 2]], dtype=np.uint16)

        merged = panel._merged_labels(current, new)

        # The existing label is untouched...
        assert merged[0, 0] == 7
        # ...and the incoming ones are pushed past it so nothing collides.
        assert merged[0, 1] == 1 + 7
        assert merged[1, 0] == 2 + 7
        assert merged.dtype == np.uint16

    def test_background_stays_background(self):
        panel = _panel()
        current = np.zeros((2, 2), dtype=np.uint16)
        new = np.zeros((2, 2), dtype=np.uint16)
        assert np.count_nonzero(panel._merged_labels(current, new)) == 0

    def test_an_offset_that_would_overflow_is_refused(self):
        """
        Regression: the offset used to be added in the labels' own dtype.

        `uint16 + python int` stays uint16 under both NEP 50 and value-based
        casting, so a sum past 65535 wrapped round to a small number and quietly
        aliased a new cell onto an existing one.
        """

        panel = _panel()
        current = np.array([[65000, 0]], dtype=np.uint16)
        new = np.array([[0, 1000]], dtype=np.uint16)
        with pytest.raises(ValueError):
            panel._merged_labels(current, new)


class TestPreparedModelCache:
    """
    The cache holds whole networks, so it must be bounded and rarely missed.

    The Cellpose thresholds are arguments to the forward pass; keying the cache
    on them reloaded a network every time one was nudged. The diameter is not a
    panel setting at all -- it is the size the network was trained to see -- so a
    cached model always gets the model's own value back.
    """

    def _prepared(self, model_type="cellpose"):
        prepared = MagicMock()
        prepared.model_type = model_type
        prepared.config_defaults = {
            "diameter": 30.0,
            "cellprob_threshold": 0.0,
            "flow_threshold": 0.4,
        }
        return prepared

    def test_a_miss_returns_none(self):
        panel = _panel()
        assert panel._reuse_prepared(("m", None, None), None, None) is None

    def test_thresholds_are_re_applied_to_a_cached_model(self):
        panel = _panel()
        prepared = self._prepared()
        panel._cache_prepared(("m", None, None), prepared)

        reused = panel._reuse_prepared(("m", None, None), 0.25, 0.6)

        assert reused is prepared
        assert prepared.cellprob_threshold == 0.25
        assert prepared.flow_threshold == 0.6
        # Not a panel setting: the frame is rescaled to the trained size instead.
        assert prepared.diameter == 30.0

    def test_a_blank_field_restores_the_models_own_value(self):
        panel = _panel()
        prepared = self._prepared()
        panel._cache_prepared(("m", None, None), prepared)

        panel._reuse_prepared(("m", None, None), 0.25, 0.6)
        panel._reuse_prepared(("m", None, None), None, None)

        assert prepared.diameter == 30.0
        assert prepared.cellprob_threshold == 0.0
        assert prepared.flow_threshold == 0.4

    def test_stardist_models_are_left_alone(self):
        panel = _panel()
        prepared = self._prepared(model_type="stardist")
        panel._cache_prepared(("m", None, None), prepared)
        reused = panel._reuse_prepared(("m", None, None), 0.25, None)
        assert reused is prepared
        assert not isinstance(prepared.diameter, float)

    def test_the_cache_is_bounded(self):
        panel = _panel()
        for i in range(fs.MAX_CACHED_MODELS + 3):
            panel._cache_prepared((f"model-{i}", None, None), self._prepared())
        assert len(panel._prepared) == fs.MAX_CACHED_MODELS

    def test_eviction_drops_the_coldest_entry(self):
        panel = _panel()
        keys = [(f"model-{i}", None, None) for i in range(fs.MAX_CACHED_MODELS + 1)]
        for key in keys[:-1]:
            panel._cache_prepared(key, self._prepared())

        # Touch the oldest so it is no longer the coldest.
        panel._reuse_prepared(keys[0], None, None)
        panel._cache_prepared(keys[-1], self._prepared())

        assert keys[0] in panel._prepared
        assert keys[1] not in panel._prepared


class TestRecordUndo:
    """A bulk write should be undoable, and never fatal if napari has moved on."""

    def test_only_the_changed_pixels_are_recorded(self):
        panel = _panel()
        layer = MagicMock()
        before = np.array([[0, 0], [1, 0]], dtype=np.uint16)
        after = np.array([[0, 5], [1, 0]], dtype=np.uint16)

        panel._record_undo(layer, 3, before, after)

        (value,), _ = layer._save_history.call_args
        indices, old, new = value
        assert len(indices) == 3  # frame, row, column
        np.testing.assert_array_equal(indices[0], [3])
        np.testing.assert_array_equal(old, [0])
        np.testing.assert_array_equal(new, [5])

    def test_an_unchanged_frame_records_nothing(self):
        panel = _panel()
        layer = MagicMock()
        same = np.ones((2, 2), dtype=np.uint16)
        panel._record_undo(layer, 0, same, same.copy())
        layer._save_history.assert_not_called()

    def test_a_napari_without_the_private_history_does_not_raise(self):
        panel = _panel()
        layer = MagicMock()
        layer._save_history.side_effect = AttributeError("moved in a later napari")
        before = np.zeros((2, 2), dtype=np.uint16)
        after = np.ones((2, 2), dtype=np.uint16)
        panel._record_undo(layer, 0, before, after)  # must not raise


class TestAvailableSegmentationModels:
    """The dropdown must always build, whatever the model tree looks like."""

    def test_singular_populations_are_normalised(self, monkeypatch):
        seen = []

        def fake_list(mode, return_path=False, cleanup=True):
            seen.append((mode, cleanup))
            return [f"{mode}-model"]

        monkeypatch.setattr(
            "celldetective.utils.model_getters.get_segmentation_models_list", fake_list
        )
        fs.available_segmentation_models("target")

        # "segmentation_target" is not a real category directory.
        assert [mode for mode, _ in seen] == ["targets", "generic"]

    def test_listing_never_rewrites_the_model_tree(self, monkeypatch):
        seen = []

        def fake_list(mode, return_path=False, cleanup=True):
            seen.append(cleanup)
            return []

        monkeypatch.setattr(
            "celldetective.utils.model_getters.get_segmentation_models_list", fake_list
        )
        fs.available_segmentation_models("targets")
        assert seen == [False, False]

    def test_duplicates_across_families_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            "celldetective.utils.model_getters.get_segmentation_models_list",
            lambda mode, return_path=False, cleanup=True: ["shared", mode],
        )
        models = fs.available_segmentation_models("targets")
        assert models == ["shared", "targets", "generic"]

    def test_a_placeholder_is_offered_when_listing_fails(self, monkeypatch):
        def explode(mode, return_path=False, cleanup=True):
            raise OSError("model repository unreachable")

        monkeypatch.setattr(
            "celldetective.utils.model_getters.get_segmentation_models_list", explode
        )
        assert fs.available_segmentation_models("targets") == [fs.NO_MODEL]


class TestChannelRows:
    """
    The channel rows are the main window's, not a copy of them.

    Mapping a model's inputs onto the experiment's channels is the same job here
    as in the main window's channel dialog, and the two drifting apart is how a
    model ends up fed a different channel depending on where it was run from.
    """

    def _panel_with_rows(self, qtbot, config, exp_channels):
        panel = _panel()
        panel.exp_channels = exp_channels
        panel.config = config
        panel.channel_selection = None
        # A real layout, so `_build_channel_rows` is exercised as it runs in the
        # panel; the panel itself never has to be a live QWidget for that.
        panel.channel_layout = QVBoxLayout()
        panel._build_channel_rows()
        qtbot.addWidget(panel.channel_selection)
        return panel

    def test_the_rows_are_the_shared_widget(self, qtbot):
        panel = self._panel_with_rows(
            qtbot, {"channels": ["live_nuclei_channel"]}, ["live_nuclei_channel"]
        )
        assert isinstance(panel.channel_selection, ModelChannelSelection)
        assert panel._selected_channels() == ["live_nuclei_channel"]

    def test_the_stored_mapping_is_honoured(self, qtbot):
        """
        A model declaring channels no experiment is named after is still usable.

        CP_cyto3 declares ['fluorescenceuv', 'None']; only the mapping saved by
        the main window says which channel that slot should be fed.
        """

        panel = self._panel_with_rows(
            qtbot,
            {
                "channels": ["fluorescenceuv", "None"],
                "selected_channels": ["brightfield_channel", "None"],
            },
            ["brightfield_channel"],
        )
        assert panel._selected_channels() == ["brightfield_channel", "None"]

    def test_a_mapping_of_the_wrong_shape_is_ignored(self, qtbot):
        # `selected_channels` is read off a JSON file that nothing guarantees the
        # shape of; a bad one must fall back rather than break the panel.
        panel = self._panel_with_rows(
            qtbot,
            {"channels": ["live_nuclei_channel"], "selected_channels": "brightfield"},
            ["live_nuclei_channel"],
        )
        assert panel._selected_channels() == ["live_nuclei_channel"]

    def test_a_model_without_inputs_reports_no_mapping(self, qtbot):
        panel = self._panel_with_rows(qtbot, {"channels": []}, ["brightfield_channel"])
        assert panel._selected_channels() is None

    def test_no_rows_at_all_reports_no_mapping(self):
        panel = _panel()
        panel.channel_selection = None
        assert panel._selected_channels() is None


class TestTrainedCellSize:
    """
    The cell size is the only handle on rescaling, and every Cellpose model has
    one whether or not it says so in microns.

    A model is trained to see objects at a fixed pixel size, and the frame is
    rescaled so they arrive at it. Models built through celldetective record the
    physical size as ``cell_size_um``; a generic Cellpose model does not, but
    records the same thing as ``diameter`` px at its own ``spatial_calibration``.
    Without the second reading, a generic model could not be rescaled at all --
    the correction needs the trained size and the target, and one was missing.
    """

    def _panel(self, config):
        panel = _panel()
        panel.config = config
        return panel

    def test_a_declared_cell_size_is_used_as_is(self):
        panel = self._panel({"model_type": "cellpose", "cell_size_um": 9.211})
        assert panel._trained_cell_size_um() == 9.211

    def test_a_generic_cellpose_model_derives_it_from_the_diameter(self):
        panel = self._panel(
            {"model_type": "cellpose", "diameter": 30.0, "spatial_calibration": 0.5}
        )
        assert panel._trained_cell_size_um() == pytest.approx(15.0)

    def test_a_declared_size_wins_over_the_derivation(self):
        panel = self._panel(
            {
                "model_type": "cellpose",
                "cell_size_um": 9.211,
                "diameter": 30.0,
                "spatial_calibration": 0.5,
            }
        )
        assert panel._trained_cell_size_um() == 9.211

    def test_a_generic_stardist_model_has_none(self):
        # StarDist has no diameter to read a size off, so there is nothing to
        # rescale against and the row is not offered.
        panel = self._panel({"model_type": "stardist", "spatial_calibration": 0.31})
        assert panel._trained_cell_size_um() is None

    @pytest.mark.parametrize(
        "config",
        [
            {"model_type": "cellpose", "diameter": 30.0},
            {"model_type": "cellpose", "spatial_calibration": 0.5},
            {"model_type": "cellpose", "diameter": 0, "spatial_calibration": 0.5},
            {"model_type": "cellpose", "diameter": 30.0, "spatial_calibration": 0},
            {"model_type": "cellpose", "cell_size_um": 0},
        ],
    )
    def test_a_size_that_cannot_be_worked_out_is_none(self, config):
        assert self._panel(config)._trained_cell_size_um() is None

    def test_no_config_reports_none(self):
        assert self._panel(None)._trained_cell_size_um() is None


class TestParameterRows:
    """The panel offers only the parameters a run should actually vary."""

    def _rows(self, qtbot, config):
        from PyQt5.QtWidgets import QFormLayout, QWidget

        panel = _panel()
        panel.config = config
        panel.diameter_le = None
        panel.cellprob_le = None
        panel.flow_le = None
        panel.cell_size_le = None
        # Parented to the panel object so the layout and its fields outlive this
        # helper: Qt deletes the C++ widgets as soon as the last holder goes.
        holder = QWidget()
        qtbot.addWidget(holder)
        panel._holder = holder
        panel.param_form = QFormLayout(holder)
        panel._build_parameter_rows()
        labels = [
            panel.param_form.itemAt(i, QFormLayout.LabelRole).widget().text()
            for i in range(panel.param_form.rowCount())
            if panel.param_form.itemAt(i, QFormLayout.LabelRole) is not None
        ]
        return panel, labels

    def test_a_generic_cellpose_model_offers_a_cell_size(self, qtbot):
        panel, labels = self._rows(
            qtbot,
            {"model_type": "cellpose", "diameter": 30.0, "spatial_calibration": 0.5},
        )
        assert any("cell size" in text for text in labels)
        assert panel.cell_size_le is not None
        assert panel.cell_size_le.value() == pytest.approx(15.0)

    def test_the_diameter_is_never_offered(self, qtbot):
        """
        It is the size the network was trained to see, not a setting: a model
        trained on 30 px objects should keep being asked for 30 px ones, and the
        frame rescaled until that is what it gets.
        """
        _, labels = self._rows(
            qtbot,
            {
                "model_type": "cellpose",
                "diameter": 30.0,
                "spatial_calibration": 0.5,
                "cellprob_threshold": 0.0,
                "flow_threshold": 0.4,
            },
        )
        assert not any("diameter" in text.lower() for text in labels)
        assert any("cell probability" in text for text in labels)
        assert any("flow threshold" in text for text in labels)

    def test_a_stored_target_seeds_the_field(self, qtbot):
        panel, _ = self._rows(
            qtbot,
            {
                "model_type": "cellpose",
                "cell_size_um": 10.0,
                "target_cell_size_um": 25.0,
            },
        )
        assert panel.cell_size_le.value() == pytest.approx(25.0)

    def test_a_model_with_nothing_to_tune_says_so(self, qtbot):
        panel, labels = self._rows(qtbot, {"model_type": "stardist"})
        assert panel.cell_size_le is None
        assert panel.param_form.rowCount() == 1
