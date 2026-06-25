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
        proc.X_trn = [np.ones((256, 256, 1), dtype=np.float32)]
        proc.Y_trn = [np.ones((256, 256), dtype=np.int32)]
        proc.X_val = [np.ones((256, 256, 1), dtype=np.float32)]
        proc.Y_val = [np.ones((256, 256), dtype=np.int32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            proc.target_directory = tmpdir
            os.makedirs(os.path.join(tmpdir, proc.model_name), exist_ok=True)

            def create_mock_model(conf, name=None, basedir=None):
                depth = getattr(conf, "unet_n_depth", 3) if conf is not None else 3
                m = MagicMock()
                m.config.train_patch_size = (256, 256)
                m.config.unet_n_depth = depth
                if depth > 3:
                    m._axes_tile_overlap.return_value = [150, 150]
                else:
                    m._axes_tile_overlap.return_value = [94, 94]
                return m

            with patch('stardist.models.StarDist2D', side_effect=create_mock_model) as mock_stardist_class, \
                 patch('stardist.calculate_extents', return_value=np.array([120.0, 120.0])), \
                 patch('stardist.gputools_available', return_value=False), \
                 patch('csbdeep.utils.save_json'):
                 
                 proc.train_stardist_model()

            # Verify that StarDist2D was re-instantiated with depth 4
            self.assertEqual(mock_stardist_class.call_count, 2)
            # The second call's first positional argument is the config object
            second_config = mock_stardist_class.call_args_list[1][0][0]
            self.assertEqual(second_config.unet_n_depth, 4)

    def test_stardist_depth_auto_adjustment_max_limit(self):
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

            def create_mock_model_no_growth(conf, name=None, basedir=None):
                depth = getattr(conf, "unet_n_depth", 3) if conf is not None else 3
                m = MagicMock()
                m.config.train_patch_size = (256, 256)
                m.config.unet_n_depth = depth
                m._axes_tile_overlap.return_value = [94, 94]
                return m

            with patch('stardist.models.StarDist2D', side_effect=create_mock_model_no_growth) as mock_stardist_class, \
                 patch('stardist.calculate_extents', return_value=np.array([120.0, 120.0])), \
                 patch('stardist.gputools_available', return_value=False), \
                 patch('csbdeep.utils.save_json'):
                 
                 proc.train_stardist_model()

            # StarDist2D should be called 4 times: initial (depth 3) + depth 4 + depth 5 + depth 6 (max)
            self.assertEqual(mock_stardist_class.call_count, 4)
            final_config = mock_stardist_class.call_args_list[-1][0][0]
            self.assertEqual(final_config.unet_n_depth, 6)

    def test_stardist_inference_padding(self):
        from celldetective.utils.stardist_utils import _segment_image_with_stardist_model
        
        # Mock model and its config
        mock_model = MagicMock()
        mock_model.config.train_patch_size = (256, 256)
        
        # Mock guess_n_tiles
        mock_model._guess_n_tiles.return_value = (1, 1)
        
        # When predict_instances is called, it should return a mock padded label mask.
        # Original size is 200x960. Padded size is 256x960.
        # Height pad: top = (256 - 200) // 2 = 28, bottom = 28.
        def mock_predict_instances(img, **kwargs):
            self.assertEqual(img.shape[:2], (256, 960))
            # Put a specific label at the padding boundary (e.g. at index 28, 10)
            res = np.zeros(img.shape[:2], dtype=np.uint16)
            res[28, 10] = 5
            return res, {
                "points": np.array([[28, 10], [5, 10]]),
                "prob": np.array([0.9, 0.8]),
                "coord": np.array([[[28, 10]], [[5, 10]]])
            }
            
        mock_model.predict_instances.side_effect = mock_predict_instances
        
        # Input image of size 200x960
        img = np.ones((200, 960, 1), dtype=np.float32)
        
        # Run prediction
        lbl, details = _segment_image_with_stardist_model(img, model=mock_model, return_details=True)
        
        # Check that the returned label mask is cropped back to 200x960
        self.assertEqual(lbl.shape, (200, 960))
        # Label at (28, 10) in padded becomes (0, 10) in cropped
        self.assertEqual(lbl[0, 10], 5)
        
        # Check details points are adjusted/cropped
        # point [28, 10] (padded) -> [0, 10] (cropped, valid)
        # point [5, 10] (padded) -> [-23, 10] (cropped, invalid, should be filtered out)
        self.assertEqual(len(details["points"]), 1)
        self.assertEqual(details["points"][0][0], 0)
        self.assertEqual(details["points"][0][1], 10)
        self.assertEqual(details["coord"][0][0][0], 0)
        self.assertEqual(details["coord"][0][0][1], 10)

    def test_stardist_safe_tiling_single_pass(self):
        # A frame below the single-pass pixel threshold must return all-ones and
        # must NEVER call _axes_tile_overlap (the trigger for the receptive-field
        # probe that hangs StarDist forever on elongated images).
        from celldetective.utils.stardist_utils import _get_safe_n_tiles

        mock_model = MagicMock()
        mock_model._guess_n_tiles.return_value = (1, 4, 1)

        # 256 x 960 = 0.25 MP, well under the 12 MP single-pass threshold.
        img = np.ones((256, 960, 1), dtype=np.float32)
        n_tiles = _get_safe_n_tiles(img, mock_model)

        self.assertEqual(n_tiles, (1, 1, 1))
        # The probe must never be triggered for single-pass frames.
        mock_model._axes_tile_overlap.assert_not_called()
        mock_model._compute_receptive_field.assert_not_called()

    def test_stardist_safe_tiling_large_uses_analytic_overlap(self):
        # A frame above the single-pass threshold must tile, but must derive its
        # overlap analytically (seeding model._tile_overlap) rather than running
        # the _compute_receptive_field probe.
        from celldetective.utils.stardist_utils import (
            _get_safe_n_tiles,
            _MAX_SINGLE_PASS_PIXELS,
        )

        mock_model = MagicMock()
        # Real oocyst model geometry: grid 8, depth 4, kernel 3, pool 2.
        mock_model.config.grid = (8, 8)
        mock_model.config.unet_n_depth = 4
        mock_model.config.unet_kernel_size = (3, 3)
        mock_model.config.unet_pool = (2, 2)
        # div_by = pool**depth * grid = 16 * 8 = 128
        mock_model._axes_div_by.return_value = (128, 128)
        mock_model._guess_n_tiles.return_value = (1, 8, 1)
        # Ensure the cache slot starts empty so _seed_tile_overlap fills it.
        mock_model._tile_overlap = None

        # 4000 x 4000 = 16 MP > 12 MP threshold -> must tile.
        side = 4000
        self.assertGreater(side * side, _MAX_SINGLE_PASS_PIXELS)
        img = np.ones((side, side, 1), dtype=np.float32)
        n_tiles = _get_safe_n_tiles(img, mock_model)

        # Tiling is active and the probe was never invoked.
        self.assertEqual(len(n_tiles), 3)
        mock_model._compute_receptive_field.assert_not_called()
        mock_model._axes_tile_overlap.assert_not_called()
        # The analytic overlap was seeded onto the model so StarDist's own tiling
        # setup will hit the cache instead of probing.
        self.assertIsInstance(mock_model._tile_overlap, list)
        self.assertEqual(len(mock_model._tile_overlap), 2)

    def test_seed_tile_overlap_prevents_probe(self):
        # Seeding must populate model._tile_overlap (so StarDist's _axes_tile_overlap
        # hits the cache) WITHOUT ever running the _compute_receptive_field probe.
        # This is what protects the training transfer-learning path from hanging when
        # fine-tuning a large-grid model (e.g. the oocyst model, grid 8).
        from celldetective.utils.stardist_utils import _seed_tile_overlap

        mock_model = MagicMock()
        mock_model.config.grid = (8, 8)
        mock_model.config.unet_n_depth = 4
        mock_model.config.unet_kernel_size = (3, 3)
        mock_model.config.unet_pool = (2, 2)
        mock_model._axes_div_by.return_value = (128, 128)
        mock_model._tile_overlap = None

        _seed_tile_overlap(mock_model)

        mock_model._compute_receptive_field.assert_not_called()
        self.assertIsInstance(mock_model._tile_overlap, list)
        self.assertEqual(len(mock_model._tile_overlap), 2)
        # (before, after) per spatial axis, strictly positive overlaps.
        for pair in mock_model._tile_overlap:
            self.assertEqual(len(pair), 2)
            self.assertGreater(pair[0], 0)

        # Idempotent: a second call must not overwrite an existing overlap.
        existing = [(7, 7), (7, 7)]
        mock_model._tile_overlap = existing
        _seed_tile_overlap(mock_model)
        self.assertEqual(mock_model._tile_overlap, existing)


if __name__ == "__main__":
    unittest.main()
