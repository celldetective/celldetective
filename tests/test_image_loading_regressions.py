"""
Image loading behaviour that works today
========================================

Companion of ``test_image_loading_limitations.py``. These tests pin down
the cases the loaders already handle correctly that ``test_image_formats.py``
does not cover yet, so that fixing a known limitation (wiki page
``Feature-Image-loading``) cannot break them silently.

Pixel values tell which frame and channel they come from, so a read that
returns the right shape but the wrong image still fails.
"""

import os
import shutil
import struct
import tempfile
import unittest

import numpy as np
from tifffile import TiffFile, imwrite

from celldetective.utils.image_loaders import (
    _MEMMAP_PAGE_SHAPES,
    _get_img_num_per_channel,
    auto_load_number_of_frames,
    load_frames,
    locate_stack,
    locate_stack_lazy,
)
from celldetective.utils.io import save_tiff_imagej_compatible
from tests.bioformats_ome import write_bioformats_ome

H, W = 8, 8
N_T, N_C = 4, 3


def _make_position_dir(base, name):
    """Create a position directory with a movie/ subfolder."""
    pos = os.path.join(base, name)
    movie_dir = os.path.join(pos, "movie")
    os.makedirs(movie_dir, exist_ok=True)
    return pos, movie_dir


def _tcyx():
    """A ``(T, C, Y, X)`` movie whose pixels hold ``10 * t + c + 1``."""
    data = np.zeros((N_T, N_C, H, W), dtype=np.uint16)
    for t in range(N_T):
        for c in range(N_C):
            data[t, c] = 10 * t + c + 1
    return data


def _expected_tyxc():
    return np.moveaxis(_tcyx(), 1, -1)


class _TmpDirCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="cd_img_regr_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


class TestMetadataLayouts(_TmpDirCase):
    """``locate_stack`` and ``locate_stack_lazy`` return the same pixels, at
    the right place, for every layout the metadata describes correctly."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        tcyx = _tcyx()
        cls.cases = {}

        def add(name, data, axes, expected, ome=False, filename="Aligned.tif"):
            pos, movie = _make_position_dir(cls.tmpdir, name)
            path = os.path.join(movie, filename)
            if ome:
                imwrite(path, data, ome=True, metadata={"axes": axes})
            else:
                imwrite(path, data, imagej=True, metadata={"axes": axes})
            cls.cases[name] = (pos, path, expected)

        add("ij_yx", tcyx[0, 0], "YX", tcyx[:1, :1].transpose(0, 2, 3, 1))
        add("ij_cyx", tcyx[0], "CYX", _expected_tyxc()[:1])
        add("ij_tyx", tcyx[:, 0], "TYX", _expected_tyxc()[..., :1])
        add("ij_tcyx", tcyx, "TCYX", _expected_tyxc())
        add(
            "ome_tcyx",
            tcyx,
            "TCYX",
            _expected_tyxc(),
            ome=True,
            filename="Aligned.ome.tif",
        )
        # Channel-first storage: the whole-movie loaders follow the metadata.
        add(
            "ome_ctyx",
            np.ascontiguousarray(np.moveaxis(tcyx, 1, 0)),
            "CTYX",
            _expected_tyxc(),
            ome=True,
            filename="Aligned.ome.tif",
        )

    def test_locate_stack(self):
        for name, (pos, _, expected) in self.cases.items():
            with self.subTest(layout=name):
                np.testing.assert_array_equal(
                    locate_stack(pos, prefix="Aligned"), expected
                )

    def test_locate_stack_lazy_matches_eager(self):
        for name, (pos, _, expected) in self.cases.items():
            with self.subTest(layout=name):
                lazy = locate_stack_lazy(pos, prefix="Aligned")
                self.assertIsNotNone(lazy, "uncompressed movie not opened lazily")
                np.testing.assert_array_equal(np.asarray(lazy), expected)

    def test_frame_count(self):
        expected_len = {
            "ij_yx": 1,
            "ij_cyx": 1,
            "ij_tyx": N_T,
            "ij_tcyx": N_T,
            "ome_tcyx": N_T,
            "ome_ctyx": N_T,
        }
        for name, (_, path, _) in self.cases.items():
            with self.subTest(layout=name):
                self.assertEqual(auto_load_number_of_frames(path), expected_len[name])


class TestLoopsReadInterleavedMovies(_TmpDirCase):
    """The analysis loops (page numbers from ``_get_img_num_per_channel``,
    pixels from ``load_frames``) read every frame and channel of a movie
    stored time then channel, ImageJ or OME, as ``locate_stack`` does."""

    def _check(self, name, ome):
        pos, movie = _make_position_dir(self.tmpdir, name)
        path = os.path.join(movie, "Aligned.ome.tif" if ome else "Aligned.tif")
        if ome:
            imwrite(path, _tcyx(), ome=True, metadata={"axes": "TCYX"})
        else:
            imwrite(path, _tcyx(), imagej=True, metadata={"axes": "TCYX"})
        stack = locate_stack(pos, prefix="Aligned")
        len_movie = auto_load_number_of_frames(path)
        img_nums = _get_img_num_per_channel(list(range(N_C)), len_movie, N_C)
        for t in range(N_T):
            frame = load_frames(img_nums[:, t].tolist(), path, normalize_input=False)
            np.testing.assert_array_equal(frame, stack[t], err_msg=f"frame {t}")
        # One channel at a time, in another order, as measurements ask for it.
        img_nums = _get_img_num_per_channel([2, 0], len_movie, N_C)
        frame = load_frames(img_nums[:, 1].tolist(), path, normalize_input=False)
        np.testing.assert_array_equal(frame, stack[1][..., [2, 0]])

    def test_imagej(self):
        self._check("loops_ij", ome=False)

    def test_ome(self):
        self._check("loops_ome", ome=True)

    def test_missing_channel_gets_placeholder(self):
        img_nums = _get_img_num_per_channel([1, None], N_T, N_C)
        np.testing.assert_array_equal(img_nums[0], [1, 4, 7, 10])
        np.testing.assert_array_equal(img_nums[1], [-1] * N_T)


class TestFilesWithoutMetadata(_TmpDirCase):
    """Plain TIFFs fall back on shape heuristics that are right in the
    common cases."""

    def test_single_rgb_image(self):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(3):
            rgb[..., c] = c + 1
        pos, movie = _make_position_dir(self.tmpdir, "rgb")
        imwrite(os.path.join(movie, "Aligned.tif"), rgb, photometric="rgb")
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape, (1, H, W, 3))
        np.testing.assert_array_equal(stack[0, 0, 0], [1, 2, 3])

    def test_short_plain_movie(self):
        n_t = 3
        data = np.stack([np.full((H, W), t + 1, np.uint16) for t in range(n_t)])
        pos, movie = _make_position_dir(self.tmpdir, "plain_short")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, data, photometric="minisblack")
        stack = locate_stack(pos, prefix="Aligned")
        self.assertEqual(stack.shape, (n_t, H, W, 1))
        np.testing.assert_array_equal(stack[:, 0, 0, 0], [1, 2, 3])
        self.assertEqual(auto_load_number_of_frames(path), n_t)


class TestMovieLookup(_TmpDirCase):
    """Which file a prefix selects."""

    def test_no_movie(self):
        pos, _ = _make_position_dir(self.tmpdir, "empty")
        with self.assertRaises(FileNotFoundError):
            locate_stack(pos, prefix="Aligned")
        self.assertIsNone(locate_stack_lazy(pos, prefix="Aligned"))

    def test_prefix_selects_its_movie_only(self):
        pos, movie = _make_position_dir(self.tmpdir, "two_prefixes")
        for name, v in (("Aligned.ome.tif", 1), ("Raw.tif", 2)):
            imwrite(
                os.path.join(movie, name),
                np.full((2, H, W), v, np.uint16),
                imagej=True,
                metadata={"axes": "TYX"},
            )
        self.assertEqual(locate_stack(pos, prefix="Aligned")[0, 0, 0, 0], 1)
        self.assertEqual(locate_stack(pos, prefix="Raw")[0, 0, 0, 0], 2)


class TestSingleIFDCache(_TmpDirCase):
    """A hyperstack saved by Fiji past 4 GB (one IFD) is read by memory map;
    the page shape is remembered, and forgotten when the file is replaced."""

    def _write(self, path, data):
        save_tiff_imagej_compatible(path, data, axes="TCYX")
        with TiffFile(path) as tif:
            offset = tif.pages.first.offset
        with open(path, "r+b") as f:
            f.seek(offset)
            (n_tags,) = struct.unpack("<H", f.read(2))
            f.seek(offset + 2 + 12 * n_tags)
            f.write(struct.pack("<I", 0))

    def test_replaced_file_is_read_again(self):
        path = os.path.join(self.tmpdir, "Corrected.tif")
        first = _tcyx().astype(np.float32)
        self._write(path, first)
        frame = load_frames(N_C + 1, path, normalize_input=False)
        np.testing.assert_array_equal(frame[..., 0], first[1, 1])
        self.assertIn(os.path.abspath(path), _MEMMAP_PAGE_SHAPES)

        # A different image size: a stale page shape would cut the pages wrongly.
        second = np.concatenate([first, first + 100], axis=2) + 1000
        self._write(path, second)
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        frame = load_frames(N_C + 1, path, normalize_input=False)
        np.testing.assert_array_equal(frame[..., 0], second[1, 1])


class TestBioformatsOME(_TmpDirCase):
    """OME-TIFFs laid out as Bio-Formats writes them (see
    ``tests/bioformats_ome.py``) that store each time point's channels
    together: ``DimensionOrder`` ``XYZCT`` (the order of the OME
    ``bioformats-artificial`` samples) and ``XYCZT``. Both the viewer and
    the loops read them correctly."""

    def _check(self, name, dimension_order):
        pos, movie = _make_position_dir(self.tmpdir, name)
        path = os.path.join(movie, "Aligned.ome.tif")
        tczyx = _tcyx()[:, :, np.newaxis]
        write_bioformats_ome(path, [tczyx], dimension_order)

        stack = locate_stack(pos, prefix="Aligned")
        np.testing.assert_array_equal(stack, _expected_tyxc())
        self.assertEqual(auto_load_number_of_frames(path), N_T)

        img_nums = _get_img_num_per_channel(list(range(N_C)), N_T, N_C)
        for t in range(N_T):
            frame = load_frames(img_nums[:, t].tolist(), path, normalize_input=False)
            np.testing.assert_array_equal(frame, stack[t], err_msg=f"frame {t}")

    def test_xyzct(self):
        self._check("bf_xyzct", "XYZCT")

    def test_xyczt(self):
        self._check("bf_xyczt", "XYCZT")

    def test_first_position_of_a_multi_position_file(self):
        pos, movie = _make_position_dir(self.tmpdir, "bf_multiposition")
        tczyx = _tcyx()[:, :, np.newaxis]
        write_bioformats_ome(
            os.path.join(movie, "Aligned.ome.tif"), [tczyx, tczyx + 1000], "XYZCT"
        )
        np.testing.assert_array_equal(
            locate_stack(pos, prefix="Aligned"), _expected_tyxc()
        )


if __name__ == "__main__":
    unittest.main()
