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


class TestPipelineAgreesOnTheTrainedCellSize(unittest.TestCase):
    """
    A cell size set once must mean the same thing everywhere it is applied.

    The main window writes ``target_cell_size_um`` into the model directory, and
    three places read it back: ``segment()``, the napari single-frame panel, and
    the full-position run. The first two go through ``trained_cell_size_um()``,
    which can work the trained size out for a generalist Cellpose model from its
    ``diameter`` and ``spatial_calibration``. The pipeline used to insist on a
    ``cell_size_um`` key instead, which those models do not carry -- so the same
    setting rescaled the preview and did nothing to the run it was previewing.
    """

    def _process_with(self, config):
        """A bare process object carrying `config`, with no position set up."""

        from celldetective.processes.segment_cells import SegmentCellDLProcess

        process = SegmentCellDLProcess.__new__(SegmentCellDLProcess)
        process.input_config = config
        process.extract_model_input_parameters()
        return process

    def test_a_generalist_cellpose_size_reaches_the_pipeline(self):
        config = dict(_model_config(CELLPOSE_MODEL))
        self.assertNotIn("cell_size_um", config)
        config["target_cell_size_um"] = 25.0

        process = self._process_with(config)

        self.assertAlmostEqual(
            process.cell_size,
            config["diameter"] * config["spatial_calibration"],
            places=6,
        )
        self.assertEqual(process.target_cell_size, 25.0)

    def test_the_pipeline_scale_matches_the_prepared_model(self):
        """The two paths must land on the same number, not merely both rescale."""

        config = dict(_model_config(CELLPOSE_MODEL))
        config["target_cell_size_um"] = 25.0
        calibration = 0.3112

        process = self._process_with(config)
        process.spatial_calibration = calibration
        process.detect_rescaling()

        expected = (
            calibration / config["spatial_calibration"]
        ) * (config["diameter"] * config["spatial_calibration"] / 25.0)
        self.assertAlmostEqual(process.scale, expected, places=6)

    def test_a_model_declaring_no_size_still_rescales_on_calibration_only(self):
        """Nothing to measure against means no cell-size correction, not a crash."""

        # A real StarDist configuration with the one key that carries the
        # trained size taken away: a StarDist model states no diameter, so there
        # is nothing left to work the size out from.
        config = dict(_model_config(MODEL))
        config.pop("cell_size_um", None)
        config["target_cell_size_um"] = 25.0

        process = self._process_with(config)
        self.assertIsNone(process.cell_size)
        self.assertIsNone(process.target_cell_size)

    def test_a_non_positive_target_is_refused_rather_than_divided_by(self):
        """
        A size that cannot be one must not reach the scale.

        `segment()` raises on these, and the channel dialog no longer writes
        one, so only a configuration saved by an earlier build still carries
        them. Zero used to raise ZeroDivisionError once the run had started, and
        a negative flipped the scale; both now fall back to the calibration-only
        scale, which is what the model declaring no size at all does.
        """

        calibration = 0.3112
        for target in (0.0, -12.0):
            with self.subTest(target=target):
                config = dict(_model_config(CELLPOSE_MODEL))
                config["target_cell_size_um"] = target

                process = self._process_with(config)
                self.assertIsNone(process.target_cell_size)

                process.spatial_calibration = calibration
                process.detect_rescaling()
                self.assertAlmostEqual(
                    process.scale,
                    calibration / config["spatial_calibration"],
                    places=6,
                )


@_requires_cellpose()
@_requires_cellpose()
class TestRescalingInvariance(unittest.TestCase):
    """
    Pixel size, trained cell size and target cell size must agree.

    Three numbers decide how a frame is resized before it reaches the network,
    and they are easy to get subtly wrong in a way no unit test on any one of
    them would catch. The property that ties them together is an invariance: one
    physical cell, of one size in microns, must arrive at the network at the
    pixel size it was trained on -- no matter how finely the microscope sampled
    it, and no matter how big the cells in the sample happen to be.

    So these tests do not check a formula. They put a disc of a known physical
    diameter into a small synthetic frame, run the real preprocessing, intercept
    the image on its way into the network, and measure the disc there.
    """

    MODEL = CELLPOSE_MODEL
    EXPERIMENT_CHANNELS = ["fluorescenceuv"]
    MAPPING = ["fluorescenceuv", "None"]

    #: The disc is drawn on a lit background, not on zero. Normalization ignores
    #: the gray value 0, so a disc on a black field leaves it nothing but the
    #: disc's own constant interior to stretch, and the frame comes out blank.
    BACKGROUND = 0.2
    FOREGROUND = 1.0

    def setUp(self):
        self.captured = []
        self._real_inference = segmentation._segment_image_with_cellpose_model

        def spy(img, **kwargs):
            self.captured.append(np.array(img))
            return np.zeros(img.shape[:2], dtype=np.uint16)

        segmentation._segment_image_with_cellpose_model = spy

    def tearDown(self):
        segmentation._segment_image_with_cellpose_model = self._real_inference

    @staticmethod
    def _frame_with_a_disc(diameter_px, size_px, background, foreground):
        """A one-channel frame holding a single disc of the given pixel size."""
        yy, xx = np.mgrid[:size_px, :size_px]
        centre = (size_px - 1) / 2.0
        radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2)
        frame = np.where(radius <= diameter_px / 2.0, foreground, background)
        return frame.astype(float)[:, :, None]

    @staticmethod
    def _diameter_of_the_disc(image):
        """
        The disc's diameter, in pixels, as it stands in `image`.

        Measured from the area above half of the intensity range rather than by
        counting a row: rescaling interpolates, so the edge is a ramp a pixel or
        two wide and the area is the steadier reading.
        """
        channel = image[:, :, 0]
        low, high = float(channel.min()), float(channel.max())
        area = int(np.count_nonzero(channel > (low + high) / 2.0))
        return 2.0 * np.sqrt(area / np.pi)

    def _diameter_reaching_the_network(
        self, spatial_calibration, cell_size_um, background=None
    ):
        """
        Segment one synthetic frame and report the disc size the network saw.

        Parameters
        ----------
        spatial_calibration : float
            Microns per pixel of the imaginary microscope.
        cell_size_um : float
            The physical diameter of the disc, which is also what the user would
            enter as the cell size for these images.
        background : float, optional
            The level the disc is drawn on. Defaults to :attr:`BACKGROUND`.

        Returns
        -------
        tuple of (float, float)
            The disc's diameter in pixels as the network received it, and the
            rescaling factor that was applied to get it there.
        """

        diameter_px = cell_size_um / spatial_calibration
        # Wide enough that the disc never touches the border, so nothing is lost
        # to the edge on the way through.
        size_px = int(max(64, round(diameter_px * 3)))

        prepared = segmentation.prepare_segmentation_model(
            self.MODEL,
            channels=self.EXPERIMENT_CHANNELS,
            spatial_calibration=spatial_calibration,
            selected_channels=self.MAPPING,
            target_cell_size=cell_size_um,
            use_stored_mapping=False,
            use_gpu=False,
        )
        self.assertIsNotNone(prepared)

        frame = self._frame_with_a_disc(
            diameter_px,
            size_px,
            self.BACKGROUND if background is None else background,
            self.FOREGROUND,
        )
        segmentation.segment_frame(frame, prepared)

        self.assertEqual(len(self.captured), 1)
        scale = 1.0 if prepared.scale_model is None else prepared.scale_model
        return self._diameter_of_the_disc(self.captured[0]), scale

    def _trained_diameter_px(self):
        """The pixel size the model expects its objects at."""
        return _model_config(self.MODEL)["diameter"]

    def test_the_pixel_size_does_not_change_what_the_network_sees(self):
        """
        The same 20 um cell, sampled four ways, arrives at one size.

        A finer pixel size makes the cell wider in the raw image and must shrink
        the image by exactly as much on the way in.
        """

        expected = self._trained_diameter_px()
        for calibration in (0.2, 0.4, 0.5789739776951672, 1.0):
            with self.subTest(spatial_calibration=calibration):
                self.setUp()
                measured, _ = self._diameter_reaching_the_network(calibration, 20.0)
                self.assertAlmostEqual(measured / expected, 1.0, delta=0.05)

    def test_the_cell_size_does_not_change_what_the_network_sees(self):
        """
        Cells of 8, 20 and 40 um, at one pixel size, arrive at one size too.

        This is the leg that a generic Cellpose model could not do at all before
        it was given a trained size in microns to be rescaled against.
        """

        expected = self._trained_diameter_px()
        for cell_size in (8.0, 20.0, 40.0):
            with self.subTest(cell_size_um=cell_size):
                self.setUp()
                measured, _ = self._diameter_reaching_the_network(0.4, cell_size)
                self.assertAlmostEqual(measured / expected, 1.0, delta=0.05)

    def test_doubling_the_cell_size_halves_the_image(self):
        """Cells twice as big must be shrunk twice as much, and nothing else."""

        _, scale = self._diameter_reaching_the_network(0.4, 20.0)
        self.setUp()
        _, doubled = self._diameter_reaching_the_network(0.4, 40.0)
        self.assertAlmostEqual(scale / doubled, 2.0, places=6)

    def test_halving_the_pixel_size_halves_the_image(self):
        """A cell sampled twice as finely must be shrunk twice as much."""

        _, scale = self._diameter_reaching_the_network(0.4, 20.0)
        self.setUp()
        _, finer = self._diameter_reaching_the_network(0.2, 20.0)
        self.assertAlmostEqual(scale / finer, 2.0, places=6)

    def test_the_background_level_cannot_be_what_makes_this_come_out(self):
        """
        The rescaling is the same on any background, a black one included.

        Which is the point: the factor is worked out from the pixel size and the
        two cell sizes alone, and no pixel of the image is ever read to arrive at
        it. So the level these tests happen to draw on cannot be what produces
        the invariance above -- it only has to let the disc be measured again
        afterwards.
        """

        scales = []
        for background in (0.0, 0.05, 0.2, 0.5, 0.9):
            with self.subTest(background=background):
                self.setUp()
                _, scale = self._diameter_reaching_the_network(
                    0.4, 20.0, background=background
                )
                scales.append(scale)
        self.assertEqual(len(set(scales)), 1)

    def test_the_measured_size_does_not_depend_on_the_background(self):
        """And the disc measures the same size on any background that is lit."""

        expected = self._trained_diameter_px()
        for background in (0.05, 0.2, 0.5, 0.9):
            with self.subTest(background=background):
                self.setUp()
                measured, _ = self._diameter_reaching_the_network(
                    0.4, 20.0, background=background
                )
                self.assertAlmostEqual(measured / expected, 1.0, delta=0.05)

    def test_a_disc_on_black_is_the_one_background_that_cannot_be_measured(self):
        """
        Why :attr:`BACKGROUND` is not zero, pinned rather than left in a comment.

        Normalization ignores the gray value 0, so on a black field the only
        pixels it has left to stretch are the disc's own constant interior: the
        low and high percentiles come out equal and the frame reaches the network
        carrying no disc at all. The rescaling is untouched -- the test above
        shows the factor is the same -- but there is then nothing to measure, so
        the disc is drawn on a lit field instead.
        """

        expected = self._trained_diameter_px()
        measured, _ = self._diameter_reaching_the_network(0.4, 20.0, background=0.0)
        self.assertLess(measured, 0.5 * expected)

    def test_the_two_effects_cancel(self):
        """
        Cells twice as big, sampled twice as finely, come out where they started.

        The clearest statement that the pixel size and the cell size enter the
        rescaling the same way: doubling both is the same scene, so the frame
        must be resized by the same factor.
        """

        _, scale = self._diameter_reaching_the_network(0.4, 20.0)
        self.setUp()
        _, both = self._diameter_reaching_the_network(0.8, 40.0)
        self.assertAlmostEqual(scale, both, places=6)


class TestCellposeModelPreparation(unittest.TestCase):
    """Cellpose model loading, which was broken on Windows."""

    # CP_cyto3 declares ['fluorescenceuv', 'None'], which no real experiment
    # has, so it can only be reached through an explicit mapping.
    EXPERIMENT_CHANNELS = ["brightfield_channel", "live_nuclei_channel"]
    MAPPING = ["live_nuclei_channel", "None"]

    # Finer than the images CP_cyto3 was trained on, so every scale below
    # carries a calibration correction on top of any cell-size correction.
    SPATIAL_CALIBRATION = 0.3112

    def _prepare(self, **kwargs):
        return segmentation.prepare_segmentation_model(
            CELLPOSE_MODEL,
            channels=self.EXPERIMENT_CHANNELS,
            spatial_calibration=self.SPATIAL_CALIBRATION,
            use_gpu=False,
            selected_channels=self.MAPPING,
            **kwargs,
        )

    def _calibration_scale(self, config):
        """
        The resampling the frame needs on pixel size alone, before any
        correction for how big the cells are.
        """
        return self.SPATIAL_CALIBRATION / config["spatial_calibration"]

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

    def test_generic_cellpose_gets_a_trained_cell_size_from_its_diameter(self):
        """
        A generic Cellpose model declares no ``cell_size_um``, so before this it
        could not be rescaled at all: the correction needs both the trained size
        and the target, and one of them was always missing. The model does say
        the same thing in pixels -- ``diameter`` px at its own
        ``spatial_calibration`` -- so the physical size is that product.
        """
        config = _model_config(CELLPOSE_MODEL)
        self.assertNotIn("cell_size_um", config)
        trained = config["diameter"] * config["spatial_calibration"]
        calibration = self._calibration_scale(config)

        # Cells twice the trained size: halved on top of the calibration
        # correction, so they reach the network at the diameter it was
        # trained on.
        prepared = self._prepare(target_cell_size=2 * trained)
        self.assertAlmostEqual(prepared.scale_model, 0.5 * calibration, places=6)

        # Half the trained size: doubled instead.
        prepared = self._prepare(target_cell_size=trained / 2)
        self.assertAlmostEqual(prepared.scale_model, 2.0 * calibration, places=6)

    def test_matching_cell_size_leaves_only_the_calibration_correction(self):
        """
        Cells already at the trained physical size need no correction of their
        own -- but the frame still carries the model's pixel size, so what is
        left is the calibration correction, not no rescaling at all.
        """
        config = _model_config(CELLPOSE_MODEL)
        trained = config["diameter"] * config["spatial_calibration"]
        prepared = self._prepare(target_cell_size=trained)
        self.assertAlmostEqual(
            prepared.scale_model, self._calibration_scale(config), places=6
        )

    def test_the_trained_diameter_is_never_rescaled_away(self):
        """
        Rescaling the frame is the whole mechanism, so the diameter handed to
        Cellpose stays the one the network was trained on.
        """
        config = _model_config(CELLPOSE_MODEL)
        trained = config["diameter"] * config["spatial_calibration"]
        for target in (None, trained, 2 * trained, trained / 2):
            with self.subTest(target=target):
                prepared = self._prepare(target_cell_size=target)
                self.assertEqual(prepared.diameter, config["diameter"])

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
