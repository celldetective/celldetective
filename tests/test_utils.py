import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
from celldetective.utils.dataset_helpers import split_by_ratio
from celldetective.utils.masks import create_patch_mask
from celldetective.utils.image_transforms import (
    estimate_unreliable_edge,
    unpad,
    mask_edges,
    pad_to_patch_size,
    pad_dataset_to_patch_size,
)
from celldetective.utils.image_loaders import (
    _get_img_num_per_channel,
    _extract_channel_indices,
)
from celldetective.utils.data_cleaning import remove_redundant_features
from celldetective.utils.experiment import extract_experiment_channels
from celldetective.utils.model_loaders import freeze_model_encoder
from celldetective.utils.io import make_json_safe


class TestPatchMask(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.radius = 3

    def test_correct_shape(self):
        self.patch = create_patch_mask(self.radius, self.radius)
        self.assertEqual(self.patch.shape, (3, 3))

    def test_correct_ring(self):
        self.patch = create_patch_mask(5, 5, radius=[1, 2])
        self.assertFalse(self.patch[2, 2])


class TestRemoveRedundantFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.list_a = ["feat1", "feat2", "feat3", "feat4", "intensity_mean"]
        self.list_b = ["feat5", "feat2", "feat1", "feat6", "test_channel_mean"]
        self.expected = ["feat3", "feat4"]

    def test_remove_red_features(self):
        self.assertEqual(
            remove_redundant_features(
                self.list_a, self.list_b, channel_names=["test_channel"]
            ),
            self.expected,
        )


class TestExtractChannelIndices(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.channels = ["ch1", "ch2", "ch3", "ch4"]
        self.required_channels = ["ch4", "ch2"]
        self.expected_indices = [3, 1]
        self.channels_mixed = ["Ch1", "CH2", "ch3", "cH4"]
        self.required_channels_mixed = ["ch4", "Ch2"]

    def test_extracted_channels_are_correct(self):
        self.assertEqual(
            list(_extract_channel_indices(self.channels, self.required_channels)),
            self.expected_indices,
        )

    def test_extracted_channels_case_insensitive(self):
        self.assertEqual(
            list(_extract_channel_indices(self.channels_mixed, self.required_channels_mixed)),
            self.expected_indices,
        )


class TestImgIndexPerChannel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.channels_indices = [1]
        self.len_movie = 5
        self.nbr_channels = 3
        self.expected_indices = [1, 4, 7, 10, 13]

    def test_index_sequence_is_correct(self):
        self.assertEqual(
            list(
                _get_img_num_per_channel(
                    self.channels_indices, self.len_movie, self.nbr_channels
                )[0]
            ),
            self.expected_indices,
        )


class TestSplitArrayByRatio(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.array_length = 100
        self.array = np.ones(self.array_length)

    def test_ratio_split_is_correct(self):
        split_array = split_by_ratio(self.array, 0.5, 0.25, 0.1)
        self.assertTrue(
            np.all(
                [
                    len(split_array[0]) == 50,
                    len(split_array[1]) == 25,
                    len(split_array[2]) == 10,
                ]
            )
        )


class TestUnpad(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.array = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_unpad(self):
        expected_unpad_array = np.array([[1]])
        test_array = unpad(self.array, 1)
        self.assertTrue(np.array_equal(test_array, expected_unpad_array))


class TestMaskEdge(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.binary_mask = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )

    def test_mask_edge_properly(self):
        expected_output = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ]
        )
        actual_output = mask_edges(self.binary_mask, 1)
        self.assertTrue(np.array_equal(actual_output, expected_output))


class TestEstimateFilterEdge(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.protocol1 = [["gauss", 2], ["std", 4]]
        self.expected1 = 6
        self.protocol2 = [["gauss", 4], ["variance", "string_arg"]]
        self.expected2 = 4

    def test_edge_is_estimated_properly_with_only_number_arguments(self):
        self.assertEqual(self.expected1, estimate_unreliable_edge(self.protocol1))

    def test_edge_is_estimated_properly_with_mixed_arguments(self):
        self.assertEqual(self.expected2, estimate_unreliable_edge(self.protocol2))


class TestPadToPatchSize(unittest.TestCase):

    def test_no_padding_needed(self):
        x = np.ones((100, 100, 1), dtype=np.float32)
        y = np.ones((100, 100), dtype=np.int32)
        xp, yp, padded = pad_to_patch_size(x, y, 80, 80)
        self.assertFalse(padded)
        self.assertTrue(np.array_equal(xp, x))
        self.assertTrue(np.array_equal(yp, y))

    def test_padding_needed_centered(self):
        x = np.ones((80, 120, 1), dtype=np.float32)
        y = np.ones((80, 120), dtype=np.int32)
        xp, yp, padded = pad_to_patch_size(x, y, 100, 150)
        self.assertTrue(padded)
        self.assertEqual(xp.shape, (100, 150, 1))
        self.assertEqual(yp.shape, (100, 150))
        
        # Original is 80x120, target is 100x150
        # Height padding: 100 - 80 = 20 -> top = 10, bottom = 10
        # Width padding: 150 - 120 = 30 -> left = 15, right = 15
        self.assertEqual(xp[9, 15, 0], 0.0)
        self.assertEqual(xp[10, 15, 0], 1.0)
        self.assertEqual(xp[10, 14, 0], 0.0)
        self.assertEqual(yp[9, 15], 0)
        self.assertEqual(yp[10, 15], 1)


class TestPadDatasetToPatchSize(unittest.TestCase):

    def test_pad_dataset(self):
        X = [np.ones((100, 100, 1), dtype=np.float32), np.ones((80, 120, 1), dtype=np.float32)]
        Y = [np.ones((100, 100), dtype=np.int32), np.ones((80, 120), dtype=np.int32)]
        Xp, Yp, padded_count = pad_dataset_to_patch_size(X, Y, 100, 150)
        
        self.assertEqual(padded_count, 2)
        self.assertEqual(Xp[0].shape, (100, 150, 1))
        self.assertEqual(Yp[0].shape, (100, 150))
        self.assertEqual(Xp[1].shape, (100, 150, 1))
        self.assertEqual(Yp[1].shape, (100, 150))


class TestFreezeModelEncoder(unittest.TestCase):

    def test_freeze_stardist(self):
        class MockLayer:
            def __init__(self):
                self.trainable = True

        class MockKerasModel:
            def __init__(self):
                self.layers = [MockLayer() for _ in range(4)]

        class MockStarDistModel:
            def __init__(self):
                self.keras_model = MockKerasModel()

        model = MockStarDistModel()
        freeze_model_encoder(model, "stardist")
        
        # Encoder depth is 4 // 2 = 2 layers
        # First 2 layers should be frozen (trainable = False)
        self.assertFalse(model.keras_model.layers[0].trainable)
        self.assertFalse(model.keras_model.layers[1].trainable)
        # Remaining 2 layers should be trainable
        self.assertTrue(model.keras_model.layers[2].trainable)
        self.assertTrue(model.keras_model.layers[3].trainable)

    def test_freeze_cellpose(self):
        class MockParameter:
            def __init__(self):
                self.requires_grad = True

        class MockModule:
            def __init__(self):
                self._params = [MockParameter(), MockParameter()]
            def parameters(self):
                return self._params

        class MockNet:
            def __init__(self):
                self.downsample = MockModule()
                self.make_style = MockModule()
                self.upsample = MockModule()
                self.output = MockModule()
                self.flow = MockModule()
                self.prob = MockModule()

        class MockCellposeModel:
            def __init__(self):
                self.net = MockNet()

        model = MockCellposeModel()
        freeze_model_encoder(model, "cellpose")

        # Downsample and style layers should be frozen (requires_grad = False)
        for p in model.net.downsample.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.net.make_style.parameters():
            self.assertFalse(p.requires_grad)

        # Decoder (upsample), output, and dynamic heads should be trainable (requires_grad = True)
        for p in model.net.upsample.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.net.output.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.net.flow.parameters():
            self.assertTrue(p.requires_grad)
        for p in model.net.prob.parameters():
            self.assertTrue(p.requires_grad)


class TestMakeJsonSafe(unittest.TestCase):

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        self.assertEqual(make_json_safe(arr), [1, 2, 3])

    def test_numpy_scalars(self):
        self.assertEqual(make_json_safe(np.int64(42)), 42)
        self.assertEqual(make_json_safe(np.float64(3.14)), 3.14)

    def test_other_types(self):
        self.assertEqual(make_json_safe("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
