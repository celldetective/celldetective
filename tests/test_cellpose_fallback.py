import unittest
from unittest.mock import MagicMock, patch
import sys

# Do not import torch here to avoid WinError 1114 if environment is broken.
# We will mock it in setUp.


class TestCellposeFallback(unittest.TestCase):

    def setUp(self):
        # Create a mock for torch
        self.mock_torch = MagicMock()
        self.mock_torch.device = MagicMock(return_value="cpu")
        self.mock_torch.cuda = MagicMock()
        self.mock_torch.cuda.is_available.return_value = (
            False  # Default to CPU environment simulation
        )

        # Patch modules so that 'import torch' and 'import cellpose' work with our mocks
        # We need to patch 'torch' in sys.modules BEFORE importing code that uses it.
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "torch": self.mock_torch,
                "cellpose": MagicMock(),
                "cellpose.models": MagicMock(),
            },
        )
        self.modules_patcher.start()

        # Define a mock CellposeModel that we can control
        self.MockCellposeModel = MagicMock()
        sys.modules["cellpose.models"].CellposeModel = self.MockCellposeModel

    def tearDown(self):
        self.modules_patcher.stop()

    def test_gpu_fallback_on_assertion_error(self):
        """
        Test that _prep_cellpose_model falls back to CPU if GPU init fails with AssertionError.
        """
        # Lazy import inside the test method/patch context
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
        self.assertEqual(self.MockCellposeModel.call_count, 2)

        args1, kwargs1 = self.MockCellposeModel.call_args_list[0]
        self.assertTrue(kwargs1.get("gpu"), "First call should try gpu=True")

        args2, kwargs2 = self.MockCellposeModel.call_args_list[1]
        self.assertFalse(kwargs2.get("gpu"), "Second call should retry with gpu=False")

        self.assertIsNotNone(model)

    def test_gpu_success(self):
        """
        Test that _prep_cellpose_model works normally if GPU init succeeds.
        """
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

        self.assertEqual(self.MockCellposeModel.call_count, 1)
        args, kwargs = self.MockCellposeModel.call_args
        self.assertTrue(kwargs.get("gpu"))


if __name__ == "__main__":
    unittest.main()
