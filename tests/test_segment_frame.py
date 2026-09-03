"""
Tests for the per-frame segmentation API.

``segment()`` was split into :func:`prepare_segmentation_model` (load the model
once) and :func:`segment_frame` (run it on one image), so that the napari viewer
can segment the frame on screen without rebuilding the model. These tests pin
the property that makes the split safe: composing the two halves must reproduce
exactly what ``segment()`` returns for the same stack.

Everything is reached through the ``segmentation`` module rather than bound at
import time. ``tests.test_partial_install`` reloads that module to check it
survives missing extras, which rebinds every class and function in it; a name
captured up here would then belong to the previous incarnation of the module,
and an ``isinstance`` check against it would fail against objects the reloaded
module builds.
"""

import json
import os
import unittest

import numpy as np
from tifffile import imread

import celldetective.segmentation as segmentation

TEST_IMAGE_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.tif"])
)
TEST_CONFIG_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.json"])
)

MODEL = "mcf7_nuc_multimodal"
CELLPOSE_MODEL = "CP_cyto3"


def _model_config(model_name):
    """Read a model's ``config_input.json``, downloading the model if need be."""
    model_path = segmentation.locate_segmentation_model(model_name)
    with open(os.path.join(model_path, "config_input.json")) as config_file:
        return json.load(config_file)


def _requires_cellpose():
    """
    Skip when Cellpose cannot actually run here.

    Loading a real Cellpose model is the only way to cover the Windows path bug,
    but it is also the only test in the suite that genuinely initialises Torch --
    so a broken Torch install (the Windows CI runners currently fail to load
    ``c10.dll``) would turn these into failures about the environment rather than
    about the code.
    """

    try:
        import torch  # noqa: F401
        from cellpose.models import CellposeModel  # noqa: F401
    except Exception as e:
        return unittest.skip(f"Cellpose/Torch unavailable: {e}")
    return lambda cls: cls


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

        cls.reference = segmentation.segment(
            cls.stack,
            MODEL,
            channels=cls.channels,
            spatial_calibration=cls.spatial_calibration,
            view_on_napari=False,
            use_gpu=False,
        )

    def _prepare(self):
        return segmentation.prepare_segmentation_model(
            MODEL,
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
        )

    def test_prepare_returns_a_usable_model(self):
        prepared = self._prepare()
        self.assertIsInstance(prepared, segmentation.PreparedSegmentationModel)
        self.assertIsNotNone(prepared.model)
        self.assertIn(prepared.model_type, ("stardist", "cellpose"))
        self.assertEqual(list(prepared.channels), list(self.channels))
        # One resolution per model input slot, and every slot accounted for:
        # either it has a source channel or it is explicitly zeroed.
        self.assertEqual(
            len(prepared.channel_indices), len(prepared.required_channels)
        )
        self.assertEqual(
            len(prepared.channel_intersection) + len(prepared.none_channel_indices),
            len(prepared.required_channels),
        )

    def test_single_frame_matches_the_stack_result(self):
        prepared = self._prepare()
        for t in range(len(self.stack)):
            with self.subTest(frame=t):
                np.testing.assert_array_equal(
                    segmentation.segment_frame(self.stack[t], prepared), self.reference[t]
                )

    def test_prepared_model_is_reusable_across_calls(self):
        """Reusing one prepared model must not drift between calls."""
        prepared = self._prepare()
        first = segmentation.segment_frame(self.stack[0], prepared)
        second = segmentation.segment_frame(self.stack[0], prepared)
        np.testing.assert_array_equal(first, second)

    def test_frame_shape_is_preserved(self):
        prepared = self._prepare()
        labels = segmentation.segment_frame(self.stack[0], prepared)
        self.assertEqual(labels.shape, self.stack[0].shape[:2])

    def test_input_frame_is_not_mutated(self):
        prepared = self._prepare()
        frame = self.stack[0].copy()
        untouched = frame.copy()
        segmentation.segment_frame(frame, prepared)
        np.testing.assert_array_equal(frame, untouched)


class TestPrepareSegmentationModelContract(unittest.TestCase):
    """Argument handling that the napari widget relies on."""

    @classmethod
    def setUpClass(cls):
        with open(TEST_CONFIG_FILENAME) as config_file:
            cls.channels = json.load(config_file)["channels"]

    def test_unknown_model_returns_none(self):
        self.assertIsNone(
            segmentation.prepare_segmentation_model(
                "a-model-that-does-not-exist", channels=self.channels, use_gpu=False
            )
        )

    def test_channels_default_to_the_model_requirements(self):
        """channels=None is documented as valid and must not raise."""
        prepared = segmentation.prepare_segmentation_model(MODEL, channels=None, use_gpu=False)
        self.assertIsNotNone(prepared)
        self.assertEqual(
            list(prepared.channels), list(prepared.required_channels)
        )

    def test_disjoint_channels_are_rejected(self):
        with self.assertRaises(ValueError):
            segmentation.prepare_segmentation_model(
                MODEL, channels=["not_a_real_channel"], use_gpu=False
            )


class TestChannelMappingAndParameters(unittest.TestCase):
    """
    The channel mapping and inference parameters the napari panel exposes.

    ``segment()`` used to ignore ``selected_channels`` and ``target_cell_size_um``
    while ``SegmentCellDLProcess`` honoured both, so the library and the pipeline
    produced different masks for the same model. These pin the aligned behaviour,
    which the stored configuration reaches only through ``use_stored_mapping``.
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
        return segmentation.prepare_segmentation_model(
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
                segmentation.segment_frame(self.stack[0], base),
                segmentation.segment_frame(self.stack[0], alt),
            )
        )

    def test_slots_sharing_one_channel_are_all_filled(self):
        """
        A mapping may feed one image channel into several of a model's slots.

        The channel-selection dialog allows it, and the pipeline handles it -
        `_get_img_num_per_channel` gives every slot its own row, so the same frame
        is simply loaded twice. Matching names instead of walking slots collapsed
        the duplicates onto the first slot and left the rest black.
        """

        base = self._prepare()
        shared = [self.channels[0]] * len(base.required_channels)
        prepared = self._prepare(selected_channels=shared)

        self.assertEqual(
            prepared.channel_indices, [0] * len(base.required_channels)
        )
        self.assertEqual(len(prepared.none_channel_indices), 0)

    def test_channel_matching_is_case_insensitive_end_to_end(self):
        """
        Index resolution and pixel transfer must agree on case.

        `_extract_channel_indices` lowercases both sides, so a slot named in a
        different case resolves to a real channel; the transfer used to compare
        names case-sensitively and skip it, leaving a slot that reported as found
        but was never filled.
        """

        base = self._prepare()
        upper = [str(ch).upper() for ch in base.required_channels]
        prepared = self._prepare(selected_channels=upper)

        self.assertEqual(prepared.channel_indices, base.channel_indices)
        np.testing.assert_array_equal(
            segmentation.segment_frame(self.stack[0], prepared),
            segmentation.segment_frame(self.stack[0], base),
        )

    def test_target_cell_size_rescales(self):
        """scale = cell_size_um / target_cell_size_um, as SegmentCellDLProcess does."""
        cell_size = _model_config(MODEL)["cell_size_um"]
        prepared = self._prepare(target_cell_size=cell_size / 2)
        self.assertIsNotNone(prepared.scale_model)
        self.assertAlmostEqual(prepared.scale_model, 2.0, places=6)

    def test_matching_cell_sizes_leave_the_scale_alone(self):
        """No rescaling when the images already match the training size."""
        cell_size = _model_config(MODEL)["cell_size_um"]
        self.assertIsNone(self._prepare(target_cell_size=cell_size).scale_model)

    def test_stardist_model_reports_no_cellpose_parameters(self):
        prepared = self._prepare()
        self.assertEqual(prepared.model_type, "stardist")
        self.assertIsNone(prepared.diameter)


@_requires_cellpose()
class TestCellposeModelPreparation(unittest.TestCase):
    """Cellpose model loading, which was broken on Windows."""

    # CP_cyto3 declares ['fluorescenceuv', 'None'], which no real experiment
    # has, so it can only be reached through an explicit mapping.
    EXPERIMENT_CHANNELS = ["brightfield_channel", "live_nuclei_channel"]
    MAPPING = ["live_nuclei_channel", "None"]

    def _prepare(self, **kwargs):
        return segmentation.prepare_segmentation_model(
            CELLPOSE_MODEL,
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

    def test_config_defaults_survive_an_override(self):
        """
        The model's own values stay reachable after a caller overrides them.

        The napari panel reuses one prepared model across parameter edits, and a
        field cleared back to blank has to mean "the model's value" - which it can
        only restore if the prepared model still remembers it.
        """

        overridden = self._prepare(
            diameter=12.0, cellprob_threshold=0.25, flow_threshold=0.6
        )
        stored = _model_config(CELLPOSE_MODEL)
        self.assertEqual(
            overridden.config_defaults["diameter"], stored["diameter"]
        )
        self.assertEqual(
            overridden.config_defaults["cellprob_threshold"],
            stored["cellprob_threshold"],
        )
        self.assertEqual(
            overridden.config_defaults["flow_threshold"], stored["flow_threshold"]
        )

    def test_segment_frame_runs_with_a_cellpose_model(self):
        prepared = self._prepare()
        frame = np.random.default_rng(0).random((64, 64, 2)) * 100
        labels = segmentation.segment_frame(frame, prepared)
        self.assertEqual(labels.shape, (64, 64))


class TestSegmentHonoursTheStoredConfiguration(unittest.TestCase):
    """
    ``segment()`` now reads the same model settings the pipeline reads.

    That is a deliberate behaviour change (see the changelog): the mapping and the
    cell size live in the model directory, which is shared across experiments, so
    these pin both halves of it - that the stored values are picked up, and that an
    explicit argument still wins so a caller can opt out.
    """

    @classmethod
    def setUpClass(cls):
        img = imread(TEST_IMAGE_FILENAME)
        cls.stack = np.moveaxis([img, img], 1, -1)
        with open(TEST_CONFIG_FILENAME) as config_file:
            config = json.load(config_file)
        cls.channels = config["channels"]
        cls.spatial_calibration = config["spatial_calibration"]

    def test_segment_passes_the_overrides_through(self):
        """
        `segment()` must reach the same masks as the two halves it wraps.

        The arguments only exist so a caller can pin the behaviour rather than
        inherit whatever the model directory happens to hold, so they are worth
        nothing unless they actually arrive at `prepare_segmentation_model`.
        """

        config = _model_config(MODEL)
        mapping = list(config["channels"])
        target = config["cell_size_um"] / 2

        labels = segmentation.segment(
            self.stack,
            MODEL,
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
            selected_channels=mapping,
            target_cell_size=target,
        )

        prepared = segmentation.prepare_segmentation_model(
            MODEL,
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
            selected_channels=mapping,
            target_cell_size=target,
        )
        self.assertAlmostEqual(prepared.scale_model, 2.0, places=6)
        np.testing.assert_array_equal(
            labels[0], segmentation.segment_frame(self.stack[0], prepared)
        )

    def test_stored_selected_channels_are_opt_in(self):
        """A mapping written into config_input.json is used only when asked for."""

        model_path = segmentation.locate_segmentation_model(MODEL)
        config_path = os.path.join(model_path, "config_input.json")
        with open(config_path) as config_file:
            original = config_file.read()

        config = json.loads(original)
        mapping = [str(ch).upper() for ch in config["channels"]]
        config["selected_channels"] = mapping
        try:
            with open(config_path, "w") as config_file:
                json.dump(config, config_file)

            # config_input.json is installed once and shared by every experiment,
            # so a mapping saved in the GUI must not silently reach a library call.
            ignored = segmentation.prepare_segmentation_model(
                MODEL,
                channels=self.channels,
                spatial_calibration=self.spatial_calibration,
                use_gpu=False,
            )
            self.assertEqual(
                list(ignored.required_channels), list(config["channels"])
            )

            # Opting in gets the pipeline's behaviour...
            prepared = segmentation.prepare_segmentation_model(
                MODEL,
                channels=self.channels,
                spatial_calibration=self.spatial_calibration,
                use_gpu=False,
                use_stored_mapping=True,
            )
            self.assertEqual(list(prepared.required_channels), mapping)

            # ...and an explicit argument still wins over what is stored.
            pinned = segmentation.prepare_segmentation_model(
                MODEL,
                channels=self.channels,
                spatial_calibration=self.spatial_calibration,
                use_gpu=False,
                use_stored_mapping=True,
                selected_channels=list(config["channels"]),
            )
            self.assertEqual(
                list(pinned.required_channels), list(config["channels"])
            )
        finally:
            with open(config_path, "w") as config_file:
                config_file.write(original)


class TestGpuVisibility(unittest.TestCase):
    """
    ``CUDA_VISIBLE_DEVICES`` must be put back the way it was found.

    Preparing a model used to set it and leave it set, so one CPU-only call from
    the GUI pinned the whole process to the CPU for the rest of the session.
    """

    def setUp(self):
        self.original = os.environ.get("CUDA_VISIBLE_DEVICES")

    def tearDown(self):
        if self.original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.original

    def test_existing_value_is_restored(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        with segmentation._gpu_visibility(False):
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_absent_value_is_left_absent(self):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        with segmentation._gpu_visibility(True):
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_restored_even_when_loading_raises(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        with self.assertRaises(RuntimeError):
            with segmentation._gpu_visibility(False):
                raise RuntimeError("model failed to load")
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_preparing_a_model_does_not_leak_the_setting(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        with open(TEST_CONFIG_FILENAME) as config_file:
            channels = json.load(config_file)["channels"]
        segmentation.prepare_segmentation_model(
            MODEL, channels=channels, use_gpu=False
        )
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")


if __name__ == "__main__":
    unittest.main()
