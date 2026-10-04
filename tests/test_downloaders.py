"""
Downloads of Zenodo models and datasets are all or nothing.

A model folder is taken for an installed model as soon as it exists, so a
folder left half-written by an interrupted extraction, or a file truncated by a
dropped connection, was never downloaded again and failed at every load. The
archive is now extracted into a staging folder moved into place once complete,
and a file shorter than the size the server announced is refused.
"""

import io
import os
import shutil
import tempfile
import unittest
import zipfile
from unittest import mock

from celldetective.utils import downloaders


def _make_archive(path, entries):
    """Write a zip archive holding `entries` ({name in archive: bytes})."""
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


class _FakeResponse(io.BytesIO):
    """A urlopen response that ends after `data`."""

    def close(self):
        pass


class TestExtractZenodoArchive(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, "models")
        os.makedirs(self.out)
        self.zip = os.path.join(self.dir, "archive.zip")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_model_folder_appears_complete_with_its_weights_renamed(self):
        _make_archive(
            self.zip,
            {
                "my_model/CP_model": b"weights",
                "my_model/config_input.json": b"{}",
            },
        )
        downloaders.extract_zenodo_archive(self.zip, self.out, "my_model")

        self.assertEqual(os.listdir(self.out), ["my_model"])
        self.assertEqual(
            sorted(os.listdir(os.path.join(self.out, "my_model"))),
            ["config_input.json", "my_model"],
        )

    def test_an_interrupted_extraction_leaves_no_model_folder(self):
        _make_archive(self.zip, {"my_model/a": b"a", "my_model/b.json": b"{}"})

        def extract_then_die(zf, path=None, *args, **kwargs):
            zf.extract(zf.namelist()[0], path)  # half-way through
            raise KeyboardInterrupt

        with mock.patch.object(zipfile.ZipFile, "extractall", extract_then_die):
            with self.assertRaises(KeyboardInterrupt):
                downloaders.extract_zenodo_archive(self.zip, self.out, "my_model")

        self.assertEqual(os.listdir(self.out), [])

    def test_a_staging_folder_left_by_a_killed_extraction_is_removed(self):
        stale = os.path.join(self.out, ".extract-killed")
        os.makedirs(os.path.join(stale, "my_model"))
        _make_archive(self.zip, {"my_model/a.json": b"{}"})

        downloaders.extract_zenodo_archive(self.zip, self.out, "my_model")

        self.assertEqual(os.listdir(self.out), ["my_model"])

    def test_extracting_over_a_folder_keeps_what_the_archive_does_not_hold(self):
        demo = os.path.join(self.out, "demo_ricm")
        os.makedirs(os.path.join(demo, "W1", "100", "output"))
        with open(os.path.join(demo, "W1", "100", "output", "mine.csv"), "w") as f:
            f.write("user analysis")
        with open(os.path.join(demo, "config.ini"), "w") as f:
            f.write("old")
        _make_archive(
            self.zip,
            {"demo_ricm/config.ini": b"new", "demo_ricm/W1/100/movie/s.tif": b"t"},
        )

        downloaders.extract_zenodo_archive(self.zip, self.out, "demo_ricm")

        with open(os.path.join(demo, "config.ini")) as f:
            self.assertEqual(f.read(), "new")
        self.assertTrue(
            os.path.exists(os.path.join(demo, "W1", "100", "output", "mine.csv"))
        )
        self.assertTrue(os.path.exists(os.path.join(demo, "W1", "100", "movie", "s.tif")))
        self.assertEqual(os.listdir(self.out), ["demo_ricm"])


class TestDownloadUrlToFile(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.dst = os.path.join(self.dir, "temp.zip")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _download(self, data, announced):
        with mock.patch.object(
            downloaders,
            "open_url_with_retries",
            return_value=(_FakeResponse(data), announced),
        ):
            downloaders.download_url_to_file("https://x/f.zip", self.dst, progress=False)

    def test_a_complete_download_is_kept(self):
        self._download(b"0123456789", 10)
        with open(self.dst, "rb") as f:
            self.assertEqual(f.read(), b"0123456789")

    def test_a_truncated_download_is_refused_and_nothing_is_left(self):
        with self.assertRaises(downloaders.IncompleteDownloadError):
            self._download(b"01234", 10)
        self.assertEqual(os.listdir(self.dir), [])

    def test_without_a_content_length_there_is_nothing_to_check(self):
        self._download(b"01234", None)
        self.assertTrue(os.path.exists(self.dst))


class TestDownloadZenodoFileConsole(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_failed_download_leaves_neither_archive_nor_folder(self):
        name = downloaders.get_zenodo_files()[0][0]

        def truncated(url, dst, progress=True):
            with open(dst, "wb") as f:
                f.write(b"PK\x03\x04 not a whole archive")
            raise downloaders.IncompleteDownloadError("cut")

        with mock.patch.object(downloaders, "download_url_to_file", truncated):
            with self.assertRaises(downloaders.IncompleteDownloadError):
                downloaders.download_zenodo_file(name, self.dir)

        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
