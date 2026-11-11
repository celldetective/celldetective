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

import logging
import os
import shutil
import struct
import sys
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


def _loops_read(path, n_frames, n_channels):
    """Every frame of ``path`` as the analysis loops read it, ``(T, Y, X, C)``."""
    img_nums = _get_img_num_per_channel(list(range(n_channels)), n_frames, n_channels)
    return np.stack(
        [
            load_frames(img_nums[:, t].tolist(), path, normalize_input=False)
            for t in range(n_frames)
        ]
    )


class TestReadersAgreeAfterFixes(_TmpDirCase):
    """The readers fixed for the limitations of ``Feature-Image-loading``
    (eager, lazy page by page, and the loops) give the same pixels, on the
    layouts the old path read through its fallback too."""

    def _all_agree(self, pos, path, expected):
        stack = locate_stack(pos, prefix="Aligned")
        np.testing.assert_array_equal(stack, expected)
        lazy = locate_stack_lazy(pos, prefix="Aligned")
        self.assertIsNotNone(lazy)
        np.testing.assert_array_equal(np.asarray(lazy), expected)
        n_frames = auto_load_number_of_frames(path)
        self.assertEqual(n_frames, expected.shape[0])
        np.testing.assert_array_equal(
            _loops_read(path, n_frames, expected.shape[-1]), expected
        )

    def test_plain_tcyx_stack(self):
        """A 4D TIFF without metadata (axes ``QQYX``) is time then channels,
        as the old fallback read it."""
        pos, movie = _make_position_dir(self.tmpdir, "plain_tcyx")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, _tcyx(), photometric="minisblack")
        self._all_agree(pos, path, _expected_tyxc())

    def test_compressed_channel_first(self):
        """Compressed and channel-first: lazy read page by page."""
        pos, movie = _make_position_dir(self.tmpdir, "zlib_ctyx")
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(
            path,
            np.moveaxis(_tcyx(), 1, 0),
            ome=True,
            metadata={"axes": "CTYX"},
            compression="zlib",
        )
        self._all_agree(pos, path, _expected_tyxc())

    def test_z_and_time_projected(self):
        """Z and time: each time point is its maximum intensity projection,
        for every reader."""
        data = np.zeros((3, 4, H, W), dtype=np.uint16)
        for t in range(3):
            for z in range(4):
                data[t, z] = 100 * t + (z * 7) % 4 + 1
        pos, movie = _make_position_dir(self.tmpdir, "tzyx_mip")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, data, imagej=True, metadata={"axes": "TZYX"})
        self._all_agree(pos, path, data.max(axis=1)[..., np.newaxis])

    def test_rgb_time_lapse_in_the_loops(self):
        rgb = np.zeros((2, H, W, 3), dtype=np.uint8)
        for t in range(2):
            for c in range(3):
                rgb[t, :, :, c] = 10 * t + c + 1
        pos, movie = _make_position_dir(self.tmpdir, "rgbt_loops")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, rgb, photometric="rgb")
        self._all_agree(pos, path, rgb)

    def test_multi_file_set(self):
        """A multi-file set: the files match the prefix together without
        being reported as several movies, and every reader reads them all."""
        pos, movie = _make_position_dir(self.tmpdir, "bf_set")
        files = write_bioformats_ome(
            os.path.join(movie, "Aligned.ome.tif"),
            [_tcyx()[:, :, np.newaxis]],
            "XYZTC",
            one_file_per_channel=True,
        )
        logger = logging.getLogger("celldetective")
        records = []
        handler = logging.Handler(level=logging.WARNING)
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            self._all_agree(pos, sorted(files)[0], _expected_tyxc())
        finally:
            logger.removeHandler(handler)
        self.assertEqual([r.getMessage() for r in records], [])

    @unittest.skipUnless(sys.platform == "win32", "Windows short names")
    def test_multi_file_set_under_a_short_path(self):
        """The same set reached through a Windows 8.3 short name (as the
        temporary folder of the GitHub runners, ``RUNNER~1``): tifffile opens
        the other files by their long name, and they are still one movie."""
        import ctypes

        from celldetective.utils.experiment import get_position_movie_path

        long_dir = os.path.join(self.tmpdir, "long_folder_name_for_8dot3")
        os.makedirs(long_dir, exist_ok=True)
        buffer = ctypes.create_unicode_buffer(1024)
        ctypes.windll.kernel32.GetShortPathNameW(long_dir, buffer, 1024)
        if not buffer.value or os.path.basename(buffer.value) == os.path.basename(
            long_dir
        ):
            self.skipTest("8.3 short names are disabled on this volume")
        pos, movie = _make_position_dir(buffer.value, "bf_set_short")
        write_bioformats_ome(
            os.path.join(movie, "Aligned.ome.tif"),
            [_tcyx()[:, :, np.newaxis]],
            "XYZTC",
            one_file_per_channel=True,
        )
        logger = logging.getLogger("celldetective")
        records = []
        handler = logging.Handler(level=logging.WARNING)
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            path = get_position_movie_path(pos, "Aligned")
            self._all_agree(pos, path, _expected_tyxc())
        finally:
            logger.removeHandler(handler)
        self.assertEqual([r.getMessage() for r in records], [])

    def _experiment(self, name, channels):
        """An experiment folder declaring ``channels``, with one position."""
        experiment = os.path.join(self.tmpdir, name)
        os.makedirs(experiment, exist_ok=True)
        with open(os.path.join(experiment, "config.ini"), "w") as f:
            f.write(
                "[MovieSettings]\nmovie_prefix = Aligned\n\n[Channels]\n"
                + "".join(f"{c} = {i}\n" for i, c in enumerate(channels))
            )
        return _make_position_dir(os.path.join(experiment, "W1"), "100")

    def test_channels_read_as_frames_in_an_experiment(self):
        """``tifffile`` writes a 3D OME-TIFF without axes as ``CYX``: the
        frames of a single-channel movie (as in the ``demo_ricm`` demo) are
        frames when the experiment declares one channel, however few."""
        frames = np.stack([np.full((H, W), t + 1, np.uint16) for t in range(4)])
        pos, movie = self._experiment("one_channel", ["adhesion_channel"])
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(path, frames, ome=True)
        with TiffFile(path) as tif:
            self.assertEqual(tif.series[0].axes, "CYX")
        self._all_agree(pos, path, frames[..., np.newaxis])

    def test_more_channels_declared_than_the_movie_has(self):
        """``demo_ricm`` in an experiment declaring a second channel: the check
        names the counts and the channels, instead of the viewer failing on an
        image past the end of the movie."""
        from celldetective.exceptions import MovieChannelsMismatchError
        from celldetective.utils.image_loaders import check_movie_channels

        frames = np.stack([np.full((H, W), t + 1, np.uint16) for t in range(40)])
        channels = ["adhesion_channel", "brightfield_channel"]
        _, movie = self._experiment("ricm_two_channels", channels)
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(path, frames, ome=True)

        with self.assertRaises(MovieChannelsMismatchError) as caught:
            check_movie_channels(path, 2, channels)
        message = str(caught.exception)
        self.assertIn("1 channel per frame (40 frames)", message)
        self.assertIn("declares 2 channels (adhesion_channel, brightfield_channel)", message)
        # The control panel passes the names as an array.
        with self.assertRaises(MovieChannelsMismatchError) as caught:
            check_movie_channels(path, 2, np.array(channels))
        self.assertEqual(str(caught.exception), message)
        # Fewer channels than the movie has, or no count, are not refused.
        check_movie_channels(path, 1)
        check_movie_channels(path, None)

    def test_channels_kept_when_the_experiment_declares_them(self):
        data = np.stack([np.full((H, W), c + 1, np.uint16) for c in range(3)])
        pos, movie = self._experiment(
            "three_channels", ["brightfield_channel", "fitc_channel", "cy5_channel"]
        )
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(path, data, ome=True)
        self._all_agree(pos, path, np.moveaxis(data, 0, -1)[np.newaxis])

    def test_many_channels_outside_an_experiment_are_frames(self):
        """Without an experiment, more than 5 channels and no time are frames,
        as before."""
        frames = np.stack([np.full((H, W), t + 1, np.uint16) for t in range(8)])
        pos, movie = _make_position_dir(self.tmpdir, "no_experiment_cyx8")
        path = os.path.join(movie, "Aligned.ome.tif")
        imwrite(path, frames, ome=True)
        self._all_agree(pos, path, frames[..., np.newaxis])

    def test_plain_stack_padded_with_singleton_axes(self):
        """tifffile pads a plain array with generic axes of size 1 (``QQQYXQ``,
        as in a lattice light-sheet export on Zenodo): they take no role, and
        the planes are frames."""
        frames = np.stack([np.full((H, W), t + 1, np.uint16) for t in range(6)])
        pos, movie = _make_position_dir(self.tmpdir, "padded_plain")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, frames.reshape(1, 1, 6, H, W, 1), photometric="minisblack")
        with TiffFile(path) as tif:
            self.assertEqual(tif.series[0].axes, "QQQYXQ")
        self._all_agree(pos, path, frames[..., np.newaxis])

    def test_unreadable_ome_metadata(self):
        """OME metadata tifffile cannot parse (a file of a multi-file set on
        Zenodo, old 2008 schema, raised 'OME series missing NumPlanes or
        PlaneCount'): the pages are read in order, as frames. The failure is
        simulated, the file itself is not redistributable as a fixture."""
        from unittest import mock

        frames = _tcyx()[:, :1, np.newaxis]  # (T, 1, 1, Y, X)
        pos, movie = _make_position_dir(self.tmpdir, "ome_unparsable")
        path = os.path.join(movie, "Aligned.ome.tif")
        write_bioformats_ome(path, [frames], "XYZCT")

        original = TiffFile.__dict__["series"]
        read_series = getattr(original, "fget", None) or original.func

        def series(tif):
            if tif.is_ome:
                raise ValueError("OME series missing NumPlanes or PlaneCount")
            return read_series(tif)

        with mock.patch.object(TiffFile, "series", property(series)):
            self._all_agree(pos, path, frames[:, 0, 0, ..., np.newaxis])

    def test_series_pages_out_of_file_order(self):
        """A series whose planes are not the file's pages in order, with one
        missing (a Micro-Manager NDTiff on Zenodo: tiles stored in acquisition
        order, an aborted acquisition): every reader follows the series, and
        the missing plane is zeros. The NDTiff layout is simulated over a plain
        file, its index file cannot be written by tifffile."""
        from unittest import mock

        from tifffile import TiffPageSeries

        frames = np.stack([np.full((H, W), 10 * (t + 1), np.uint16) for t in range(3)])
        pos, movie = _make_position_dir(self.tmpdir, "ndtiff_order")
        path = os.path.join(movie, "Aligned.tif")
        imwrite(path, frames, photometric="minisblack")

        def series(tif):
            pages = [tif.pages[2], None, tif.pages[0]]
            return [TiffPageSeries(pages, (3, H, W), frames.dtype, "TYX", kind="ndtiff")]

        expected = np.stack([frames[2], np.zeros((H, W), np.uint16), frames[0]])
        with mock.patch.object(TiffFile, "series", property(series)):
            self._all_agree(pos, path, expected[..., np.newaxis])

    def test_ome_tiff_extension_in_the_loops(self):
        """The loops find a ``.tiff`` movie the way ``locate_stack`` does."""
        from celldetective.utils.experiment import get_position_movie_path

        pos, movie = _make_position_dir(self.tmpdir, "tiff_ext")
        path = os.path.join(movie, "Aligned.ome.tiff")
        imwrite(path, _tcyx(), ome=True, metadata={"axes": "TCYX"})
        self.assertEqual(
            os.path.abspath(get_position_movie_path(pos, "Aligned")),
            os.path.abspath(path),
        )
        self._all_agree(pos, path, _expected_tyxc())


if __name__ == "__main__":
    unittest.main()
