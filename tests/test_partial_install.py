"""
The package must import and fail gracefully when the optional extras are absent.

Every test here reloads a module with its heavy dependency mocked away, which
re-executes it and rebinds every class and function it defines. Any module
reloaded inside a test is therefore reloaded again on the way out, with the real
dependencies back in ``sys.modules``: without that, later tests in the run would
be talking to a module that was built while TensorFlow, Torch, StarDist and
Cellpose all looked missing, and any object they had already imported by name
would belong to the previous incarnation of it.
"""

import unittest
from unittest.mock import patch
import sys
import importlib
from contextlib import contextmanager


@contextmanager
def _reloaded_without(module, missing):
    """
    Reload `module` with `missing` mocked out, then reload it again to restore it.

    Parameters
    ----------
    module : module
        The module to reload.
    missing : dict
        ``sys.modules`` entries to override for the duration, mapped to None to
        make importing them fail.

    Yields
    ------
    module
        The module, reloaded in the degraded environment.
    """

    try:
        with patch.dict(sys.modules, missing):
            yield importlib.reload(module)
    finally:
        importlib.reload(module)


class TestPartialValidation(unittest.TestCase):

    def test_imports_without_extras(self):
        """Test that main modules can be imported even if optional extras are missing."""
        # This test assumes the environment MIGHT have them, so we must mock them as missing
        # to ensure the code handles it.

        import celldetective.segmentation

        missing = {
            "tensorflow": None,
            "torch": None,
            "stardist": None,
            "cellpose": None,
            "cellpose.models": None,
            "stardist.models": None,
        }
        try:
            with _reloaded_without(celldetective.segmentation, missing):
                pass
        except ImportError as e:
            self.fail(
                f"Could not import celldetective.segmentation without extras: {e}"
            )
        except Exception as e:
            self.fail(f"Unexpected error importing celldetective.segmentation: {e}")

    def test_reloading_does_not_leave_the_module_degraded(self):
        """
        The suite must be handed back a module built with the real extras present.

        `tests.test_segment_frame` asserts on classes from
        `celldetective.segmentation`; if the reload above were the last one, those
        classes would be the ones defined while every extra looked missing, and
        would no longer be the ones the module hands out.
        """

        import celldetective.segmentation as seg

        before = seg.PreparedSegmentationModel
        with _reloaded_without(seg, {"tensorflow": None, "stardist": None}) as during:
            self.assertIsNot(during.PreparedSegmentationModel, before)

        prepared = seg.prepare_segmentation_model("a-model-that-does-not-exist")
        self.assertIsNone(prepared)
        self.assertIs(
            seg.PreparedSegmentationModel,
            sys.modules["celldetective.segmentation"].PreparedSegmentationModel,
        )

    def test_graceful_failure_stardist(self):
        """Test that calling stardist functions raises RuntimeError if missing."""
        import celldetective.utils.stardist_utils

        with _reloaded_without(
            celldetective.utils.stardist_utils,
            {"stardist": None, "stardist.models": None},
        ) as stardist_utils:
            with self.assertRaises(RuntimeError) as cm:
                stardist_utils._prep_stardist_model("fake_model", "fake_path")

            self.assertIn("StarDist is not installed", str(cm.exception))

    def test_graceful_failure_cellpose(self):
        """Test that calling cellpose functions raises RuntimeError if missing."""
        import celldetective.utils.cellpose_utils

        with _reloaded_without(
            celldetective.utils.cellpose_utils,
            {"cellpose": None, "cellpose.models": None, "torch": None},
        ) as cellpose_utils:
            with self.assertRaises(RuntimeError) as cm:
                cellpose_utils._prep_cellpose_model("fake_model", "fake_path")

            # Message check might correspond to torch or cellpose depending on which import hits first
            # Our code checks torch first.
            self.assertTrue(
                "Torch is not installed" in str(cm.exception)
                or "Cellpose is not installed" in str(cm.exception)
            )


if __name__ == "__main__":
    unittest.main()
