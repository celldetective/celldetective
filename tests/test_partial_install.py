import unittest
from unittest.mock import MagicMock, patch
import sys
import importlib


class TestPartialValidation(unittest.TestCase):

    def test_imports_without_extras(self):
        """Test that main modules can be imported even if optional extras are missing."""
        # This test assumes the environment MIGHT have them, so we must mock them as missing
        # to ensure the code handles it.

        with patch.dict(
            sys.modules,
            {
                "tensorflow": None,
                "torch": None,
                "stardist": None,
                "cellpose": None,
                "cellpose.models": None,
                "stardist.models": None,
            },
        ):
            # Force reload of celldetective.segmentation to test its imports
            try:
                import celldetective.segmentation

                importlib.reload(celldetective.segmentation)
            except ImportError as e:
                self.fail(
                    f"Could not import celldetective.segmentation without extras: {e}"
                )
            except Exception as e:
                self.fail(f"Unexpected error importing celldetective.segmentation: {e}")

    def test_graceful_failure_stardist(self):
        """Test that calling stardist functions raises RuntimeError if missing."""
        with patch.dict(sys.modules, {"stardist": None, "stardist.models": None}):
            # We need to reload the util module to pick up the missing module
            import celldetective.utils.stardist_utils

            importlib.reload(celldetective.utils.stardist_utils)

            from celldetective.utils.stardist_utils import _prep_stardist_model

            with self.assertRaises(RuntimeError) as cm:
                _prep_stardist_model("fake_model", "fake_path")

            self.assertIn("StarDist is not installed", str(cm.exception))

    def test_graceful_failure_cellpose(self):
        """Test that calling cellpose functions raises RuntimeError if missing."""
        with patch.dict(
            sys.modules, {"cellpose": None, "cellpose.models": None, "torch": None}
        ):
            import celldetective.utils.cellpose_utils

            importlib.reload(celldetective.utils.cellpose_utils)

            from celldetective.utils.cellpose_utils import _prep_cellpose_model

            with self.assertRaises(RuntimeError) as cm:
                _prep_cellpose_model("fake_model", "fake_path")

            # Message check might correspond to torch or cellpose depending on which import hits first
            # Our code checks torch first.
            self.assertTrue(
                "Torch is not installed" in str(cm.exception)
                or "Cellpose is not installed" in str(cm.exception)
            )


if __name__ == "__main__":
    unittest.main()
