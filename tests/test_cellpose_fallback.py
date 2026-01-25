import unittest
from unittest.mock import MagicMock, patch
import sys
import torch


class TestCellposeFallback(unittest.TestCase):

    def setUp(self):
        # Patch modules before importing the function under test
        self.cellpose_patcher = patch.dict(
            sys.modules, {"cellpose": MagicMock(), "cellpose.models": MagicMock()}
        )
        self.cellpose_patcher.start()

        # Define a mock CellposeModel that we can control
        self.MockCellposeModel = MagicMock()
        sys.modules["cellpose.models"].CellposeModel = self.MockCellposeModel

    def tearDown(self):
        self.cellpose_patcher.stop()

    def test_gpu_fallback_on_assertion_error(self):
        """
        Test that _prep_cellpose_model falls back to CPU if GPU init fails with AssertionError.
        """
        # Lazy import to ensure mocks are in place
        from celldetective.utils.cellpose_utils import _prep_cellpose_model

        # Side effect for CellposeModel constructor
        def side_effect(gpu=False, **kwargs):
            if gpu:
                raise AssertionError("Torch not compiled with CUDA enabled")

            # Return a mock model object
            model = MagicMock()
            model.diam_mean = 30.0
            model.diam_labels = 30.0
            return model

        self.MockCellposeModel.side_effect = side_effect

        # Call the function with use_gpu=True
        # We expect it to try with gpu=True, fail, print warning, and retry with gpu=False
        model, scale = _prep_cellpose_model(
            model_name="fake_model", path="fake_path/", use_gpu=True, n_channels=2
        )

        # Check call history
        # Should be called twice.
        # First call: gpu=True
        # Second call: gpu=False
        self.assertEqual(self.MockCellposeModel.call_count, 2)

        args1, kwargs1 = self.MockCellposeModel.call_args_list[0]
        self.assertTrue(kwargs1.get("gpu"), "First call should try gpu=True")

        args2, kwargs2 = self.MockCellposeModel.call_args_list[1]
        self.assertFalse(kwargs2.get("gpu"), "Second call should retry with gpu=False")

        # Ensure we got a valid model back
        self.assertIsNotNone(model)

    def test_gpu_success(self):
        """
        Test that _prep_cellpose_model works normally if GPU init succeeds.
        """
        # Lazy import
        from celldetective.utils.cellpose_utils import _prep_cellpose_model

        # Side effect for success
        def side_effect(gpu=False, **kwargs):
            model = MagicMock()
            model.diam_mean = 30.0
            model.diam_labels = 30.0
            return model

        self.MockCellposeModel.side_effect = side_effect

        model, scale = _prep_cellpose_model(
            model_name="fake_model", path="fake_path/", use_gpu=True, n_channels=2
        )

        # Should be called once with gpu=True
        self.assertEqual(self.MockCellposeModel.call_count, 1)
        args, kwargs = self.MockCellposeModel.call_args
        self.assertTrue(kwargs.get("gpu"))


if __name__ == "__main__":
    unittest.main()
