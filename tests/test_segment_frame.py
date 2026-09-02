"""
Tests for the per-frame segmentation API.

``segment()`` was split into :func:`prepare_segmentation_model` (load the model
once) and :func:`segment_frame` (run it on one image), so that the napari viewer
can segment the frame on screen without rebuilding the model. These tests pin
the property that makes the split safe: composing the two halves must reproduce
exactly what ``segment()`` returns for the same stack.
"""

import json
import os
import unittest

import numpy as np
from tifffile import imread

from celldetective.segmentation import (
    PreparedSegmentationModel,
    prepare_segmentation_model,
    segment,
    segment_frame,
)

TEST_IMAGE_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.tif"])
)
TEST_CONFIG_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.json"])
)

MODEL = "mcf7_nuc_multimodal"


class TestSegmentFrameMatchesSegment(unittest.TestCase):
    """The split halves must reproduce the whole-stack function exactly."""

    @classmethod
    def setUpClass(cls):
        img = imread(TEST_IMAGE_FILENAME)
        # Three identical frames, as in test_segmentation.py.
        cls.stack = np.moveaxis([img, img, img], 1, -1)
        with open(TEST_CONFIG_FILENAME) as config_file:
            config = json.load(config_file)
        cls.channels = config["channels"]
        cls.spatial_calibration = config["spatial_calibration"]

        cls.reference = segment(
            cls.stack,
            MODEL,
            channels=cls.channels,
            spatial_calibration=cls.spatial_calibration,
            view_on_napari=False,
            use_gpu=False,
        )

    def _prepare(self):
        return prepare_segmentation_model(
            MODEL,
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
        )

    def test_prepare_returns_a_usable_model(self):
        prepared = self._prepare()
        self.assertIsInstance(prepared, PreparedSegmentationModel)
        self.assertIsNotNone(prepared.model)
        self.assertIn(prepared.model_type, ("stardist", "cellpose"))
        self.assertEqual(list(prepared.channels), list(self.channels))
        # Every channel the model needs is accounted for, either transferred or
        # explicitly zeroed.
        self.assertEqual(
            len(prepared.channel_intersection) + len(prepared.none_channel_indices),
            len(prepared.required_channels),
        )

    def test_single_frame_matches_the_stack_result(self):
        prepared = self._prepare()
        for t in range(len(self.stack)):
            with self.subTest(frame=t):
                np.testing.assert_array_equal(
                    segment_frame(self.stack[t], prepared), self.reference[t]
                )

    def test_prepared_model_is_reusable_across_calls(self):
        """Reusing one prepared model must not drift between calls."""
        prepared = self._prepare()
        first = segment_frame(self.stack[0], prepared)
        second = segment_frame(self.stack[0], prepared)
        np.testing.assert_array_equal(first, second)

    def test_frame_shape_is_preserved(self):
        prepared = self._prepare()
        labels = segment_frame(self.stack[0], prepared)
        self.assertEqual(labels.shape, self.stack[0].shape[:2])

    def test_input_frame_is_not_mutated(self):
        prepared = self._prepare()
        frame = self.stack[0].copy()
        untouched = frame.copy()
        segment_frame(frame, prepared)
        np.testing.assert_array_equal(frame, untouched)


class TestPrepareSegmentationModelContract(unittest.TestCase):
    """Argument handling that the napari widget relies on."""

    @classmethod
    def setUpClass(cls):
        with open(TEST_CONFIG_FILENAME) as config_file:
            cls.channels = json.load(config_file)["channels"]

    def test_unknown_model_returns_none(self):
        self.assertIsNone(
            prepare_segmentation_model(
                "a-model-that-does-not-exist", channels=self.channels, use_gpu=False
            )
        )

    def test_channels_default_to_the_model_requirements(self):
        """channels=None is documented as valid and must not raise."""
        prepared = prepare_segmentation_model(MODEL, channels=None, use_gpu=False)
        self.assertIsNotNone(prepared)
        self.assertEqual(
            list(prepared.channels), list(prepared.required_channels)
        )

    def test_disjoint_channels_are_rejected(self):
        with self.assertRaises(ValueError):
            prepare_segmentation_model(
                MODEL, channels=["not_a_real_channel"], use_gpu=False
            )


class TestChannelMappingAndParameters(unittest.TestCase):
    """
    The channel mapping and inference parameters the napari panel exposes.

    ``segment()`` used to ignore ``selected_channels`` and ``target_cell_size_um``
    while ``SegmentCellDLProcess`` honoured both, so the library and the pipeline
    produced different masks for the same model. These pin the aligned behaviour.
    """

    @classmethod
    def setUpClass(cls):
        img = imread(TEST_IMAGE_FILENAME)
        cls.stack = np.moveaxis([img, img, img], 1, -1)
        with open(TEST_CONFIG_FILENAME) as config_file:
            config = json.load(config_file)
        cls.channels = config["channels"]
        cls.spatial_calibration = config["spatial_calibration"]

    def _prepare(self, **kwargs):
        return prepare_segmentation_model(
            MODEL,
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
            **kwargs,
        )

    def _remapped_slots(self, base):
        """
        Feed the model's first slot from a different experiment channel.

        Picking any fixed index would risk landing on the channel that already
        occupies slot 0, making the "remapping" a no-op that proves nothing.
        """
        slots = list(base.required_channels)
        replacement = next(ch for ch in self.channels if ch != slots[0])
        slots[0] = replacement
        return slots

    def test_selected_channels_overrides_the_model_slots(self):
        base = self._prepare()
        remapped = self._remapped_slots(base)
        alt = self._prepare(selected_channels=remapped)
        self.assertEqual(list(alt.required_channels), remapped)
        self.assertNotEqual(
            list(alt.required_channels), list(base.required_channels)
        )

    def test_remapping_changes_the_masks(self):
        """A mapping that is not a no-op must actually reach inference."""
        base = self._prepare()
        alt = self._prepare(selected_channels=self._remapped_slots(base))
        self.assertFalse(
            np.array_equal(
                segment_frame(self.stack[0], base),
                segment_frame(self.stack[0], alt),
            )
        )

    def test_target_cell_size_rescales(self):
        """scale = cell_size_um / target_cell_size_um, as SegmentCellDLProcess does."""
        prepared = self._prepare(target_cell_size=6.0)
        cell_size = 13.46  # mcf7_nuc_multimodal, from config_input.json
        self.assertIsNotNone(prepared.scale_model)
        self.assertAlmostEqual(prepared.scale_model, cell_size / 6.0, places=6)

    def test_matching_cell_sizes_leave_the_scale_alone(self):
        """No rescaling when the images already match the training size."""
        self.assertIsNone(self._prepare(target_cell_size=13.46).scale_model)

    def test_stardist_model_reports_no_cellpose_parameters(self):
        prepared = self._prepare()
        self.assertEqual(prepared.model_type, "stardist")
        self.assertIsNone(prepared.diameter)


class TestCellposeModelPreparation(unittest.TestCase):
    """Cellpose model loading, which was broken on Windows."""

    CELLPOSE_MODEL = "CP_cyto3"
    # CP_cyto3 declares ['fluorescenceuv', 'None'], which no real experiment
    # has, so it can only be reached through an explicit mapping.
    EXPERIMENT_CHANNELS = ["brightfield_channel", "live_nuclei_channel"]
    MAPPING = ["live_nuclei_channel", "None"]

    def _prepare(self, **kwargs):
        return prepare_segmentation_model(
            self.CELLPOSE_MODEL,
            channels=self.EXPERIMENT_CHANNELS,
            spatial_calibration=0.3112,
            use_gpu=False,
            selected_channels=self.MAPPING,
            **kwargs,
        )

    def test_cellpose_model_loads(self):
        """
        Regression: the Cellpose branch derived the model name as
        ``model_path.split("/")[-2]``. locate_segmentation_model returns an
        os.sep-joined path, so on Windows the split yields a single element and
        the index raised IndexError -- no Cellpose model could be prepared at all.
        """
        prepared = self._prepare()
        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.model_type, "cellpose")
        self.assertIsNotNone(prepared.model)
        self.assertEqual(list(prepared.required_channels), self.MAPPING)

    def test_cellpose_parameters_come_from_the_config(self):
        prepared = self._prepare()
        self.assertIsNotNone(prepared.diameter)
        self.assertIsNotNone(prepared.cellprob_threshold)
        self.assertIsNotNone(prepared.flow_threshold)

    def test_cellpose_parameters_can_be_overridden(self):
        prepared = self._prepare(
            diameter=12.0, cellprob_threshold=0.25, flow_threshold=0.6
        )
        self.assertEqual(prepared.diameter, 12.0)
        self.assertEqual(prepared.cellprob_threshold, 0.25)
        self.assertEqual(prepared.flow_threshold, 0.6)

    def test_segment_frame_runs_with_a_cellpose_model(self):
        prepared = self._prepare()
        frame = np.random.default_rng(0).random((64, 64, 2)) * 100
        labels = segment_frame(frame, prepared)
        self.assertEqual(labels.shape, (64, 64))


if __name__ == "__main__":
    unittest.main()
