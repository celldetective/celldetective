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
)
from celldetective.utils.image_loaders import (
    _get_img_num_per_channel,
    _extract_channel_indices,
)
from celldetective.utils.data_cleaning import remove_redundant_features
from celldetective.utils.experiment import extract_experiment_channels


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

    def test_extracted_channels_are_correct(self):
        self.assertEqual(
            list(_extract_channel_indices(self.channels, self.required_channels)),
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


if __name__ == "__main__":
    unittest.main()
