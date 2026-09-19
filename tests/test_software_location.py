import os
import unittest
import importlib.util

from celldetective import get_package_location, get_software_location


def _pyqt_available():
    return importlib.util.find_spec("PyQt5") is not None


class TestSoftwareLocation(unittest.TestCase):
    """
    Guard the path helpers the whole GUI relies on to find its shipped data
    (icons, help pages, models). These must behave identically on Windows,
    Linux and macOS, so every assertion here is separator-agnostic.
    """

    def test_package_location_is_the_package_folder(self):
        pkg = get_package_location()
        self.assertTrue(os.path.isabs(pkg), f"not absolute: {pkg}")
        self.assertTrue(os.path.isdir(pkg), f"not a directory: {pkg}")
        self.assertEqual(os.path.basename(pkg), "celldetective")
        self.assertTrue(os.path.isfile(os.path.join(pkg, "__init__.py")))

    def test_paths_use_native_separators(self):
        """A path built by hand with the wrong separator breaks os.path.isdir
        on POSIX, and mixed separators break string comparisons everywhere."""
        for path in (get_package_location(), get_software_location()):
            self.assertEqual(path, os.path.normpath(path))
            if os.sep == "/":
                self.assertNotIn("\\", path)

    def test_software_location_is_parent_of_package(self):
        """~370 call sites do sep.join([get_software_location(), 'celldetective', ...]),
        so this round-trip has to hold."""
        rebuilt = os.path.join(get_software_location(), "celldetective")
        self.assertEqual(
            os.path.normcase(os.path.realpath(rebuilt)),
            os.path.normcase(os.path.realpath(get_package_location())),
        )

    def test_shipped_data_files_are_reachable(self):
        """Spot-check one asset per package_data glob family."""
        pkg = get_package_location()
        for relative in (
            ("icons", "splash.png"),
            ("gui", "help", "tracking.json"),
            ("scripts", "segment_cells.py"),
        ):
            with self.subTest(asset="/".join(relative)):
                self.assertTrue(
                    os.path.isfile(os.path.join(pkg, *relative)),
                    f"missing shipped asset: {os.path.join(pkg, *relative)}",
                )

    @unittest.skipUnless(_pyqt_available(), "PyQt5 not installed")
    def test_console_script_entry_point_resolves(self):
        """setup.py advertises `celldetective = celldetective.__main__:main`;
        if that attribute goes missing the installed command dies on launch."""
        module = importlib.import_module("celldetective.__main__")
        self.assertTrue(
            callable(getattr(module, "main", None)),
            "celldetective.__main__:main is missing or not callable",
        )


if __name__ == "__main__":
    unittest.main()
