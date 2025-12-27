import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tifffile import imread
from celldetective.segmentation import segment, segment_frame_from_thresholds
from tensorflow.keras.metrics import BinaryIoU

TEST_IMAGE_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.tif"])
)
TEST_LABEL_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample_labelled.tif"])
)
TEST_CONFIG_FILENAME = os.path.join(
    os.path.dirname(__file__), os.sep.join(["assets", "sample.json"])
)


class TestDLMCF7Segmentation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.img = imread(TEST_IMAGE_FILENAME)
        self.label_true = imread(TEST_LABEL_FILENAME)
        self.stack = np.moveaxis([self.img, self.img, self.img], 1, -1)
        with open(TEST_CONFIG_FILENAME) as config_file:
            self.config = json.load(config_file)
        self.channels = self.config["channels"]
        print(f"{self.channels=}")
        self.spatial_calibration = self.config["spatial_calibration"]

    def test_correct_segmentation_with_multimodal_model(self):

        labels = segment(
            self.stack,
            "mcf7_nuc_multimodal",
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            view_on_napari=False,
            use_gpu=False,
        )
        np.testing.assert_array_equal(labels[0], labels[1])

        self.binary_label_true = self.label_true.copy().astype(float)
        self.binary_label_true[self.binary_label_true > 0] = 1.0

        label_binary = labels[0].copy().astype(float)
        label_binary[label_binary > 0] = 1.0

        m = BinaryIoU(target_class_ids=[1])
        m.update_state(self.binary_label_true, label_binary)
        score = m.result().numpy()

        self.assertGreater(score, 0.85)

    def test_correct_segmentation_with_transferred_model(self):

        labels = segment(
            self.stack,
            "mcf7_nuc_stardist_transfer",
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            view_on_napari=False,
            use_gpu=True,
        )
        np.testing.assert_array_equal(labels[0], labels[1])

        self.binary_label_true = self.label_true.copy().astype(float)
        self.binary_label_true[self.binary_label_true > 0] = 1.0

        label_binary = labels[0].copy().astype(float)
        label_binary[label_binary > 0] = 1.0

        m = BinaryIoU(target_class_ids=[1])
        m.update_state(self.binary_label_true, label_binary)
        score = m.result().numpy()

        self.assertGreater(score, 0.85)


class TestSegmentFunctionExtensive(unittest.TestCase):
    """
    Extensive unit tests for the segment function covering edge cases,
    input validation, and special value handling.
    """

    @classmethod
    def setUpClass(self):
        self.img = imread(TEST_IMAGE_FILENAME)
        # Create a stack (frames, height, width, channels)
        # The sample image seems to have channels in the first dimension (C, H, W) based on valid tests usage:
        # np.moveaxis([self.img, self.img, self.img], 1, -1) -> (3, H, C, W)? No, wait.
        # Original test: self.stack = np.moveaxis([self.img, self.img, self.img],1,-1)
        # img shape is likely (C, H, W) or (H, W).
        # If sample.tif is 2D + channels?
        # Let's trust the existing setup logic:
        self.stack = np.moveaxis([self.img, self.img, self.img], 1, -1)
        with open(TEST_CONFIG_FILENAME) as config_file:
            self.config = json.load(config_file)
        self.channels = self.config["channels"]
        self.spatial_calibration = self.config["spatial_calibration"]

    def test_segment_invalid_model(self):
        """Test that an invalid model name returns None or raises appropriate error."""
        labels = segment(
            self.stack,
            "non_existent_model_12345",
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
        )
        self.assertIsNone(labels, "Segment should return None for invalid model names.")

    def test_segment_input_validation(self):
        """Test input validation for mismatched channels."""
        # Pass wrong number of channels
        wrong_channels = ["ch1"]  # Expecting more
        with self.assertRaises(AssertionError):
            segment(
                self.stack,
                "mcf7_nuc_multimodal",
                channels=wrong_channels,
                spatial_calibration=self.spatial_calibration,
            )

    def test_segment_output_shape(self):
        """Verify that the output shape matches (frames, height, width)."""
        # We use a valid model for this
        labels = segment(
            self.stack,
            "mcf7_nuc_multimodal",
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
        )
        # self.stack shape: (3, H, W, C) -> we expect (3, H, W)
        expected_shape = self.stack.shape[:-1]
        self.assertEqual(labels.shape, expected_shape)

    def test_segment_inf_nan_handling(self):
        """Verify that infinite values in the input do not crash the segmentation and are handled."""
        # Create a stack with some Inf values
        stack_with_inf = self.stack.copy().astype(float)
        # Set a pixel to infinity
        stack_with_inf[0, 10, 10, 0] = np.inf

        # Should not crash
        try:
            labels = segment(
                stack_with_inf,
                "mcf7_nuc_multimodal",
                channels=self.channels,
                spatial_calibration=self.spatial_calibration,
                use_gpu=False,
            )
        except Exception as e:
            self.fail(f"segment crashed with Inf values: {e}")

        self.assertIsNotNone(labels)
        self.assertEqual(labels.shape, self.stack.shape[:-1])

    def test_segment_gpu_flag_env(self):
        """Verify that use_gpu flag sets the environment variable."""
        # This is a bit tricky as it modifies global state, but we can check if it runs.
        # modifying correct usage:
        # We can check os.environ after the call, but the function might reset it? No, it sets it.
        # The function does: if not use_gpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import os

        # Save current state
        old_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")

        segment(
            self.stack,
            "mcf7_nuc_multimodal",
            channels=self.channels,
            spatial_calibration=self.spatial_calibration,
            use_gpu=False,
        )

        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "-1")

        # Restore
        if old_cuda_env:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_env
        else:
            del os.environ["CUDA_VISIBLE_DEVICES"]


class TestThresholdMCF7Segmentation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.img = imread(TEST_IMAGE_FILENAME)
        self.label_true = imread(TEST_LABEL_FILENAME)
        with open(TEST_CONFIG_FILENAME) as config_file:
            self.config = json.load(config_file)
        self.channels = self.config["channels"]
        self.spatial_calibration = self.config["spatial_calibration"]

    def test_correct_segmentation_with_threshold(self):

        label = segment_frame_from_thresholds(
            np.moveaxis(self.img, 0, -1),
            target_channel=3,
            thresholds=[8000, 1.0e10],
            equalize_reference=None,
            filters=[["variance", 4], ["gauss", 2]],
            marker_min_distance=13,
            marker_footprint_size=34,
            marker_footprint=None,
            feature_queries=["area < 80"],
            channel_names=None,
        )

        self.binary_label_true = self.label_true.copy().astype(float)
        self.binary_label_true[self.binary_label_true > 0] = 1.0

        label_binary = label.copy().astype(float)
        label_binary[label_binary > 0] = 1.0

        m = BinaryIoU(target_class_ids=[1])
        m.update_state(self.binary_label_true, label_binary)
        score = m.result().numpy()

        self.assertGreater(score, 0.7)


if __name__ == "__main__":
    unittest.main()
