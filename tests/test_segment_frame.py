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


if __name__ == "__main__":
    unittest.main()
