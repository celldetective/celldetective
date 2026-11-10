"""
Known limitations of image loading
==================================

Each test reproduces one limitation recorded on the wiki page
``Feature-Image-loading`` (IDs IMG-n) and asserts the behaviour the loaders
should have. They fail on purpose until the limitation is fixed; a fix turns
its test green, and the wiki row can then be closed.

Every movie is synthetic and tiny, with pixel values that tell which frame,
slice and channel they come from, so a wrong read shows as a wrong value.
"""

import logging
import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
from tifffile import imwrite

from celldetective.utils.image_loaders import (
    _get_img_num_per_channel,
    _load_frames_to_segment,
    auto_load_number_of_frames,
    load_frames,
    locate_stack,
    locate_stack_lazy,
)

H, W = 8, 8


def _make_position_dir(base, name):
    """Create a position directory with a movie/ subfolder."""
    pos = os.path.join(base, name)
    movie_dir = os.path.join(pos, "movie")
    os.makedirs(movie_dir, exist_ok=True)
    return pos, movie_dir


class _ListHandler(logging.Handler):
    """Keep the log records emitted while it is attached."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _value(t, c):
    """Pixel value of frame ``t``, channel ``c``."""
    return 10 * t + c + 1


class _TmpDirCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_img_limits_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


class TestChannelFirstPageOrder(_TmpDirCase):
    """IMG-1: the analysis loops read pages as if channels were interleaved
    within each time point, so a channel-first movie is read wrongly."""

    def test_load_frames_matches_locate_stack_for_ctyx(self):
        n_t, n_c = 4, 2
        data = np.zeros((n_c, n_t, H, W), dtype=np.uint16)
        for t in range(n_t):
            for c in range(n_c):
                data[c, t] = _value(t, c)
        pos, movie = _make_position_dir(self.tmpdir, "ctyx")
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(path, data, ome=True, metadata={"axes": "CTYX"})

        stack = locate_stack(pos, prefix="Aligned")
        len_movie = auto_load_number_of_frames(path)
        img_nums = _get_img_num_per_channel(list(range(n_c)), len_movie, n_c)

        for t in range(n_t):
            loop_frame = load_frames(
                img_nums[:, t].tolist(), path, normalize_input=False
            )
            np.testing.assert_array_equal(
                loop_frame[0, 0, :],
                stack[t, 0, 0, :],
                err_msg=f"frame {t}: the loops and the viewer disagree",
            )


class TestZAndTime(_TmpDirCase):
    """IMG-2: a movie with both Z and time is flattened into one time axis,
    and the loops then read the slices of the first time point as frames."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        n_t, n_z = 3, 4
        data = np.zeros((n_t, n_z, H, W), dtype=np.uint16)
        for t in range(n_t):
            for z in range(n_z):
                data[t, z] = 100 * t + z + 1
        cls.pos, movie = _make_position_dir(cls.tmpdir, "tzyx")
        cls.path = os.path.join(movie, "Aligned.tif")
        imwrite(cls.path, data, imagej=True, metadata={"axes": "TZYX"})
        cls.n_t = n_t

    def test_frame_count_is_number_of_time_points(self):
        stack = locate_stack(self.pos, prefix="Aligned")
        self.assertEqual(
            stack.shape[0],
            self.n_t,
            "Z slices were merged into the time axis without warning",
        )

    def test_loop_frame_comes_from_its_time_point(self):
        len_movie = auto_load_number_of_frames(self.path)
        img_nums = _get_img_num_per_channel([0], len_movie, 1)
        frame_1 = load_frames(img_nums[:, 1].tolist(), self.path, normalize_input=False)
        # Any slice of time point 1 is in [101, 104]; slice 2 of time point 0 is 2.
        self.assertGreaterEqual(
            int(frame_1[0, 0, 0]), 101, "frame 1 of the loops is a slice of time 0"
        )


class TestManyChannelsWithoutTime(_TmpDirCase):
    """IMG-3: with no time axis, more than 5 channels are taken for time."""

    def test_six_channel_image_keeps_its_channels(self):
        data = np.stack([np.full((H, W), c + 1, dtype=np.uint16) for c in range(6)])
        pos, movie = _make_position_dir(self.tmpdir, "cyx6")
        imwrite(
            os.path.join(movie, "Aligned.tif"),
            data,
            imagej=True,
            metadata={"axes": "CYX"},
        )
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape, (1, H, W, 6))


class TestSeveralMatchingMovies(_TmpDirCase):
    """IMG-4: when several files match the prefix, one is picked silently."""

    def test_ambiguous_prefix_is_reported(self):
        pos, movie = _make_position_dir(self.tmpdir, "two_movies")
        for name, v in (("Aligned_a.tif", 1), ("Aligned_b.tif", 2)):
            imwrite(
                os.path.join(movie, name),
                np.full((2, H, W), v, dtype=np.uint16),
                imagej=True,
                metadata={"axes": "TYX"},
            )
        # Refusing the folder, a warning or a logged warning all count.
        logger = logging.getLogger("celldetective")
        handler = _ListHandler()
        logger.addHandler(handler)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    locate_stack(pos, prefix="Aligned")
                except (ValueError, FileExistsError):
                    return
        finally:
            logger.removeHandler(handler)
        reported = [str(w.message) for w in caught] + [
            r.getMessage() for r in handler.records if r.levelno >= logging.WARNING
        ]
        self.assertTrue(
            reported, "two movies match the prefix and one was picked silently"
        )


class TestCompressedLazy(_TmpDirCase):
    """IMG-5: lazy reading gives up on compressed TIFFs, so napari loads
    the whole movie into memory."""

    def test_compressed_movie_loads_lazily(self):
        pos, movie = _make_position_dir(self.tmpdir, "zlib")
        imwrite(
            os.path.join(movie, "Aligned.tif"),
            np.ones((3, H, W), dtype=np.uint16),
            imagej=True,
            metadata={"axes": "TYX"},
            compression="zlib",
        )
        stack = locate_stack_lazy(pos, prefix="Aligned")
        self.assertIsNotNone(stack, "compressed movie was not opened lazily")
        self.assertEqual(stack.shape, (3, H, W, 1))


class TestZWithManyChannels(_TmpDirCase):
    """IMG-6: Z plus more than 5 channels and no time gives a stack whose
    image size is wrong (slices stacked along Y)."""

    def test_zcyx_shape(self):
        data = np.ones((2, 6, H, W), dtype=np.uint16)
        pos, movie = _make_position_dir(self.tmpdir, "zcyx6")
        imwrite(
            os.path.join(movie, "Aligned.tif"),
            data,
            imagej=True,
            metadata={"axes": "ZCYX"},
        )
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape[1:3], (H, W), f"got {stack.shape}")


class TestRGBTimeLapse(_TmpDirCase):
    """IMG-7: an RGB time-lapse comes out with X and the colour channels
    swapped."""

    def test_rgb_time_lapse(self):
        rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
        for t in range(2):
            for c in range(3):
                rgb[t, :, :, c] = _value(t, c)
        pos, movie = _make_position_dir(self.tmpdir, "rgbt")
        imwrite(os.path.join(movie, "Aligned.tif"), rgb, photometric="rgb")
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape, (2, H, W, 3))
        np.testing.assert_array_equal(stack[1, 0, 0, :], [11, 12, 13])


class TestNoMetadataLongMovie(_TmpDirCase):
    """IMG-8: a TIFF without ImageJ or OME metadata that has more pages than
    its smaller image side is read as one frame, with Y taken for channels."""

    def test_long_plain_movie(self):
        n_t = H + 4
        data = np.stack([np.full((H, W), t + 1, dtype=np.uint16) for t in range(n_t)])
        pos, movie = _make_position_dir(self.tmpdir, "plain_long")
        imwrite(os.path.join(movie, "Aligned.tif"), data, photometric="minisblack")
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape, (n_t, H, W, 1))


class TestUnreadableFrameInSegmentation(_TmpDirCase):
    """Errors: a frame load failure reaches segmentation as ``None`` and
    surfaces as an unrelated ``AttributeError``."""

    def test_clear_error(self):
        _, movie = _make_position_dir(self.tmpdir, "short")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(
            path,
            np.ones((3, H, W), dtype=np.uint16),
            imagej=True,
            metadata={"axes": "TYX"},
        )
        with self.assertRaises(Exception) as ctx:
            _load_frames_to_segment(path, np.array([0, 99]))
        self.assertNotIsInstance(
            ctx.exception,
            AttributeError,
            "load failure surfaced as an AttributeError on None",
        )


if __name__ == "__main__":
    unittest.main()
