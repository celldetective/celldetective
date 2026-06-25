import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import tempfile
import os
from celldetective.processes.train_segmentation_model import TrainSegModelProcess

class TestStarDistPadding(unittest.TestCase):

    def test_stardist_padding_logic(self):
        # Create a subclass of TrainSegModelProcess to bypass __init__ file operations
        class DummyTrainProcess(TrainSegModelProcess):
            def __init__(self):
                # Bypassing the parent __init__ which reads files
                self.queue = None
                self.X_trn = []
                self.Y_trn = []
                self.X_val = []
                self.Y_val = []
                self.pretrained = None
                self.use_gpu = False
                self.learning_rate = 0.0003
                self.epochs = 1
                self.batch_size = 1
                self.augmentation_factor = 1.0
                self.model_name = "test_model"
                self.files_train = ["train1.tif"]
                self.files_val = ["val1.tif"]
                self.target_channels = ["ch1"]
                self.normalization_percentile = [True]
                self.normalization_clip = True
                self.normalization_values = [(0.0, 99.9)]
                self.spatial_calibration = 1.0

        proc = DummyTrainProcess()
        # Set up small mock images and labels (e.g. 100x120 pixels)
        # Images have channels (e.g. shape (100, 120, 1)), labels are 2D (shape (100, 120))
        proc.X_trn = [np.ones((100, 120, 1), dtype=np.float32)]
        proc.Y_trn = [np.ones((100, 120), dtype=np.int32)]
        proc.X_val = [np.ones((80, 150, 1), dtype=np.float32)]
        proc.Y_val = [np.ones((80, 150), dtype=np.int32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            proc.target_directory = tmpdir
            os.makedirs(os.path.join(tmpdir, proc.model_name), exist_ok=True)

            # Mock the StarDist2D class and other external imports
            mock_model = MagicMock()
            mock_model.config.train_patch_size = (256, 256)
            
            # We patch StarDist2D to return our mock_model
            with patch('stardist.models.StarDist2D', return_value=mock_model), \
                 patch('stardist.calculate_extents', return_value=10.0), \
                 patch('stardist.gputools_available', return_value=False), \
                 patch('csbdeep.utils.save_json'):
                 
                 # Call the train_stardist_model method
                 proc.train_stardist_model()
             
        # Check that X_trn, Y_trn, X_val, Y_val have been padded to (256, 256)
        self.assertEqual(proc.X_trn[0].shape, (256, 256, 1))
        self.assertEqual(proc.Y_trn[0].shape, (256, 256))
        self.assertEqual(proc.X_val[0].shape, (256, 256, 1))
        self.assertEqual(proc.Y_val[0].shape, (256, 256))

        # Check centered padding implementation details:
        # For X_trn: original is 100x120, target is 256x256.
        # Height pad: 256 - 100 = 156 -> top = 78, bottom = 78
        # Width pad: 256 - 120 = 136 -> left = 68, right = 68
        self.assertEqual(proc.X_trn[0][77, 68, 0], 0.0) # top border padded region
        self.assertEqual(proc.X_trn[0][78, 68, 0], 1.0) # start of original image
        self.assertEqual(proc.X_trn[0][78, 67, 0], 0.0) # left border padded region
        self.assertEqual(proc.X_trn[0][78, 68, 0], 1.0) # start of original image

        # Similarly for labels
        self.assertEqual(proc.Y_trn[0][77, 68], 0)
        self.assertEqual(proc.Y_trn[0][78, 68], 1)

    def test_stardist_depth_auto_adjustment(self):
        class DummyTrainProcess(TrainSegModelProcess):
            def __init__(self):
                self.queue = None
                self.X_trn = []
                self.Y_trn = []
                self.X_val = []
                self.Y_val = []
                self.pretrained = None
                self.use_gpu = False
                self.learning_rate = 0.0003
                self.epochs = 1
                self.batch_size = 1
                self.augmentation_factor = 1.0
                self.model_name = "test_model"
                self.files_train = ["train1.tif"]
                self.files_val = ["val1.tif"]
                self.target_channels = ["ch1"]
                self.normalization_percentile = [True]
                self.normalization_clip = True
                self.normalization_values = [(0.0, 99.9)]
                self.spatial_calibration = 1.0

        proc = DummyTrainProcess()
        proc.X_trn = [np.ones((256, 256, 1), dtype=np.float32)]
        proc.Y_trn = [np.ones((256, 256), dtype=np.int32)]
        proc.X_val = [np.ones((256, 256, 1), dtype=np.float32)]
        proc.Y_val = [np.ones((256, 256), dtype=np.int32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            proc.target_directory = tmpdir
            os.makedirs(os.path.join(tmpdir, proc.model_name), exist_ok=True)

            mock_model = MagicMock()
            mock_model.config.train_patch_size = (256, 256)
            mock_model.config.unet_n_depth = 3
            mock_model._axes_tile_overlap.return_value = [94, 94]

            with patch('stardist.models.StarDist2D', return_value=mock_model) as mock_stardist_class, \
                 patch('stardist.calculate_extents', return_value=np.array([120.0, 120.0])), \
                 patch('stardist.gputools_available', return_value=False), \
                 patch('csbdeep.utils.save_json'):
                 
                 proc.train_stardist_model()

            # Verify that StarDist2D was re-instantiated with depth 4
            self.assertEqual(mock_stardist_class.call_count, 2)
            # The second call's first positional argument is the config object
            second_config = mock_stardist_class.call_args_list[1][0][0]
            self.assertEqual(second_config.unet_n_depth, 4)


if __name__ == "__main__":
    unittest.main()
