"""
Test Image Format Compatibility
================================

Verifies that all 6 documented image formats (XY, CXY, TXY, PXY, TCXY, PCXY)
are correctly handled by the image loading pipeline, including OME-TIFF files.

Documented formats (from first-experiment.rst):
- XY   (2D) : single-timepoint & single-channel
- CXY  (3D) : single-timepoint & multichannel
- PXY  (3D) : multi-position, single-channel & single-timepoint
- TXY  (3D) : time-lapse
- PCXY (4D) : multi-position, multichannel & single-timepoint
- TCXY (4D) : multi-channel time-lapse
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from tifffile import imwrite

from celldetective.utils.io import save_tiff_imagej_compatible
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    load_frames,
    locate_stack,
    _rearrange_multichannel_frame,
    _load_stack_from_series,
)

H, W = 64, 64
N_CHANNELS = 3
N_FRAMES = 5


def _make_position_dir(base, name):
    """Create a position directory with a movie/ subfolder."""
    pos = os.path.join(base, name)
    movie_dir = os.path.join(pos, "movie")
    os.makedirs(movie_dir, exist_ok=True)
    return pos, movie_dir


class TestAutoLoadNumberOfFrames(unittest.TestCase):
    """Test that auto_load_number_of_frames returns the correct frame count
    for every documented image format."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_fmt_test_")

        # XY – (H, W) – 1 frame, 1 channel
        cls.xy_path = os.path.join(cls.tmpdir, "xy.tif")
        save_tiff_imagej_compatible(
            cls.xy_path, np.zeros((H, W), dtype=np.float32), axes="YX"
        )

        # CXY – (C, H, W) – 1 frame, 3 channels
        cls.cxy_path = os.path.join(cls.tmpdir, "cxy.tif")
        save_tiff_imagej_compatible(
            cls.cxy_path, np.zeros((N_CHANNELS, H, W), dtype=np.float32), axes="CYX"
        )

        # TXY – (T, H, W) – 5 frames, 1 channel
        cls.txy_path = os.path.join(cls.tmpdir, "txy.tif")
        save_tiff_imagej_compatible(
            cls.txy_path, np.zeros((N_FRAMES, H, W), dtype=np.float32), axes="TYX"
        )

        # PXY – structurally identical to TXY (positions = time axis in storage)
        cls.pxy_path = os.path.join(cls.tmpdir, "pxy.tif")
        save_tiff_imagej_compatible(
            cls.pxy_path, np.zeros((4, H, W), dtype=np.float32), axes="TYX"
        )

        # TCXY – (T, C, H, W) – 5 frames, 3 channels
        cls.tcxy_path = os.path.join(cls.tmpdir, "tcxy.tif")
        save_tiff_imagej_compatible(
            cls.tcxy_path,
            np.zeros((N_FRAMES, N_CHANNELS, H, W), dtype=np.float32),
            axes="TCYX",
        )

        # PCXY – structurally identical to TCXY
        cls.pcxy_path = os.path.join(cls.tmpdir, "pcxy.tif")
        save_tiff_imagej_compatible(
            cls.pcxy_path,
            np.zeros((4, N_CHANNELS, H, W), dtype=np.float32),
            axes="TCYX",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_xy_returns_1_frame(self):
        self.assertEqual(auto_load_number_of_frames(self.xy_path), 1)

    def test_cxy_returns_1_frame(self):
        self.assertEqual(auto_load_number_of_frames(self.cxy_path), 1)

    def test_txy_returns_correct_frames(self):
        self.assertEqual(auto_load_number_of_frames(self.txy_path), N_FRAMES)

    def test_pxy_returns_correct_frames(self):
        self.assertEqual(auto_load_number_of_frames(self.pxy_path), 4)

    def test_tcxy_returns_correct_frames(self):
        self.assertEqual(auto_load_number_of_frames(self.tcxy_path), N_FRAMES)

    def test_pcxy_returns_correct_frames(self):
        self.assertEqual(auto_load_number_of_frames(self.pcxy_path), 4)


class TestLocateStack(unittest.TestCase):
    """Test that locate_stack reshapes every format to (T, H, W, C)."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_locate_test_")

        # XY
        cls.pos_xy, movie = _make_position_dir(cls.tmpdir, "pos_xy")
        save_tiff_imagej_compatible(
            os.path.join(movie, "Aligned.tif"),
            np.zeros((H, W), dtype=np.float32),
            axes="YX",
        )

        # CXY
        cls.pos_cxy, movie = _make_position_dir(cls.tmpdir, "pos_cxy")
        save_tiff_imagej_compatible(
            os.path.join(movie, "Aligned.tif"),
            np.zeros((N_CHANNELS, H, W), dtype=np.float32),
            axes="CYX",
        )

        # TXY
        cls.pos_txy, movie = _make_position_dir(cls.tmpdir, "pos_txy")
        save_tiff_imagej_compatible(
            os.path.join(movie, "Aligned.tif"),
            np.zeros((N_FRAMES, H, W), dtype=np.float32),
            axes="TYX",
        )

        # TCXY
        cls.pos_tcxy, movie = _make_position_dir(cls.tmpdir, "pos_tcxy")
        save_tiff_imagej_compatible(
            os.path.join(movie, "Aligned.tif"),
            np.zeros((N_FRAMES, N_CHANNELS, H, W), dtype=np.float32),
            axes="TCYX",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_xy_shape(self):
        stack = locate_stack(self.pos_xy)
        self.assertEqual(stack.shape, (1, H, W, 1))

    def test_cxy_shape(self):
        stack = locate_stack(self.pos_cxy)
        self.assertEqual(stack.shape, (1, H, W, N_CHANNELS))

    def test_txy_shape(self):
        stack = locate_stack(self.pos_txy)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, 1))

    def test_tcxy_shape(self):
        stack = locate_stack(self.pos_tcxy)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, N_CHANNELS))


class TestLoadFrames(unittest.TestCase):
    """Test that load_frames successfully loads individual frames for each format."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_loadfr_test_")

        # XY
        cls.xy_path = os.path.join(cls.tmpdir, "xy.tif")
        save_tiff_imagej_compatible(
            cls.xy_path,
            np.random.randint(0, 255, (H, W), dtype=np.uint16),
            axes="YX",
        )

        # TXY
        cls.txy_path = os.path.join(cls.tmpdir, "txy.tif")
        save_tiff_imagej_compatible(
            cls.txy_path,
            np.random.randint(0, 255, (N_FRAMES, H, W), dtype=np.uint16),
            axes="TYX",
        )

        # TCXY
        cls.tcxy_path = os.path.join(cls.tmpdir, "tcxy.tif")
        save_tiff_imagej_compatible(
            cls.tcxy_path,
            np.random.randint(0, 255, (N_FRAMES, N_CHANNELS, H, W), dtype=np.uint16),
            axes="TCYX",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_load_xy_frame(self):
        frames = load_frames(0, self.xy_path, normalize_input=False)
        self.assertIsNotNone(frames)
        # Channel-last: (H, W, 1)
        self.assertEqual(frames.ndim, 3)
        self.assertEqual(frames.shape[:2], (H, W))

    def test_load_txy_frame(self):
        frames = load_frames(0, self.txy_path, normalize_input=False)
        self.assertIsNotNone(frames)
        self.assertEqual(frames.ndim, 3)
        self.assertEqual(frames.shape[:2], (H, W))

    def test_load_tcxy_frame(self):
        frames = load_frames(0, self.tcxy_path, normalize_input=False)
        self.assertIsNotNone(frames)
        self.assertEqual(frames.ndim, 3)
        self.assertEqual(frames.shape[:2], (H, W))


class TestRearrangeMultichannelFrame(unittest.TestCase):
    """Test that _rearrange_multichannel_frame correctly places channels last."""

    def test_2d_gets_singleton_channel(self):
        frame = np.zeros((H, W))
        result = _rearrange_multichannel_frame(frame)
        self.assertEqual(result.shape, (H, W, 1))

    def test_3d_channel_first_moved_to_last(self):
        frame = np.zeros((N_CHANNELS, H, W))
        result = _rearrange_multichannel_frame(frame)
        self.assertEqual(result.shape, (H, W, N_CHANNELS))

    def test_3d_channel_last_unchanged(self):
        frame = np.zeros((H, W, N_CHANNELS))
        result = _rearrange_multichannel_frame(frame)
        self.assertEqual(result.shape, (H, W, N_CHANNELS))


class TestOMETiff(unittest.TestCase):
    """Verify that OME-TIFF files are handled correctly."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_ome_test_")

        # Single-channel OME time-lapse
        cls.ome_txy_path = os.path.join(cls.tmpdir, "txy.ome.tif")
        imwrite(
            cls.ome_txy_path,
            np.zeros((N_FRAMES, H, W), dtype=np.uint16),
            ome=True,
            metadata={"axes": "TYX"},
        )

        # Multi-channel OME time-lapse
        cls.ome_tcxy_path = os.path.join(cls.tmpdir, "tcxy.ome.tif")
        imwrite(
            cls.ome_tcxy_path,
            np.zeros((N_FRAMES, N_CHANNELS, H, W), dtype=np.uint16),
            ome=True,
            metadata={"axes": "TCYX"},
        )

        # locate_stack position with OME file
        cls.pos_ome, movie = _make_position_dir(cls.tmpdir, "pos_ome")
        imwrite(
            os.path.join(movie, "Aligned.ome.tif"),
            np.zeros((N_FRAMES, N_CHANNELS, H, W), dtype=np.uint16),
            ome=True,
            metadata={"axes": "TCYX"},
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_ome_txy_frame_count(self):
        self.assertEqual(auto_load_number_of_frames(self.ome_txy_path), N_FRAMES)

    def test_ome_tcxy_frame_count(self):
        self.assertEqual(auto_load_number_of_frames(self.ome_tcxy_path), N_FRAMES)

    def test_ome_locate_stack(self):
        stack = locate_stack(self.pos_ome)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, N_CHANNELS))


class TestLoadStackFromSeries(unittest.TestCase):
    """Test the _load_stack_from_series helper with various axis orders."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_series_test_")

        # YX (2D)
        cls.yx_path = os.path.join(cls.tmpdir, "yx.tif")
        save_tiff_imagej_compatible(
            cls.yx_path, np.zeros((H, W), dtype=np.float32), axes="YX"
        )

        # CYX (3D multichannel)
        cls.cyx_path = os.path.join(cls.tmpdir, "cyx.tif")
        save_tiff_imagej_compatible(
            cls.cyx_path, np.zeros((N_CHANNELS, H, W), dtype=np.float32), axes="CYX"
        )

        # TYX (3D time-lapse)
        cls.tyx_path = os.path.join(cls.tmpdir, "tyx.tif")
        save_tiff_imagej_compatible(
            cls.tyx_path, np.zeros((N_FRAMES, H, W), dtype=np.float32), axes="TYX"
        )

        # TCYX (4D)
        cls.tcyx_path = os.path.join(cls.tmpdir, "tcyx.tif")
        save_tiff_imagej_compatible(
            cls.tcyx_path,
            np.zeros((N_FRAMES, N_CHANNELS, H, W), dtype=np.float32),
            axes="TCYX",
        )

        # TZCYX with Z=1 (should be squeezed)
        cls.tzcyx_path = os.path.join(cls.tmpdir, "tzcyx.ome.tif")
        imwrite(
            cls.tzcyx_path,
            np.zeros((N_FRAMES, 1, N_CHANNELS, H, W), dtype=np.uint16),
            ome=True,
            metadata={"axes": "TZCYX"},
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_yx_to_tyxc(self):
        stack = _load_stack_from_series(self.yx_path)
        self.assertIsNotNone(stack)
        self.assertEqual(stack.shape, (1, H, W, 1))

    def test_cyx_to_tyxc(self):
        stack = _load_stack_from_series(self.cyx_path)
        self.assertIsNotNone(stack)
        self.assertEqual(stack.shape, (1, H, W, N_CHANNELS))

    def test_tyx_to_tyxc(self):
        stack = _load_stack_from_series(self.tyx_path)
        self.assertIsNotNone(stack)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, 1))

    def test_tcyx_to_tyxc(self):
        stack = _load_stack_from_series(self.tcyx_path)
        self.assertIsNotNone(stack)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, N_CHANNELS))

    def test_tzcyx_z_squeezed(self):
        """Z=1 should be squeezed out, producing (T, Y, X, C)."""
        stack = _load_stack_from_series(self.tzcyx_path)
        self.assertIsNotNone(stack)
        self.assertEqual(stack.shape, (N_FRAMES, H, W, N_CHANNELS))


if __name__ == "__main__":
    unittest.main()
