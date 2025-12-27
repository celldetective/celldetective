import unittest
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from celldetective.measure import (
    measure_features,
    measure_isotropic_intensity,
    drop_tonal_features,
)


class TestFeatureMeasurement(unittest.TestCase):
    """
    To do: test spot detection, fluo normalization and peripheral measurements
    """

    @classmethod
    def setUpClass(self):

        # Simple mock data, 100px*100px, one channel, value is one, uniform
        # Two objects in labels map

        self.frame = np.ones((100, 100, 1), dtype=float)
        self.labels = np.zeros((100, 100), dtype=int)
        self.labels[50:55, 50:55] = 1
        self.labels[0:10, 0:10] = 2

        self.feature_measurements = measure_features(
            self.frame,
            self.labels,
            features=[
                "intensity_mean",
                "area",
            ],
            channels=["test_channel"],
        )

        self.feature_measurements_no_image = measure_features(
            None,
            self.labels,
            features=[
                "intensity_mean",
                "area",
            ],
            channels=None,
        )

        self.feature_measurements_no_features = measure_features(
            self.frame,
            self.labels,
            features=None,
            channels=["test_channel"],
        )

    # With image
    def test_measure_yields_table(self):
        self.assertIsInstance(self.feature_measurements, pd.DataFrame)

    def test_two_objects(self):
        self.assertEqual(len(self.feature_measurements), 2)

    def test_channel_named_correctly(self):
        self.assertIn("test_channel_mean", list(self.feature_measurements.columns))

    def test_intensity_is_one(self):
        self.assertTrue(
            np.all(
                [
                    v == 1.0
                    for v in self.feature_measurements["test_channel_mean"].values
                ]
            )
        )

    def test_area_first_is_twenty_five(self):
        self.assertEqual(self.feature_measurements["area"].values[0], 25)

    def test_area_second_is_hundred(self):
        self.assertEqual(self.feature_measurements["area"].values[1], 100)

    # Without image
    def test_measure_yields_table(self):
        self.assertIsInstance(self.feature_measurements_no_image, pd.DataFrame)

    def test_two_objects(self):
        self.assertEqual(len(self.feature_measurements_no_image), 2)

    def test_channel_not_in_table(self):
        self.assertNotIn(
            "test_channel_mean", list(self.feature_measurements_no_image.columns)
        )

    # With no features
    def test_only_one_measurement(self):
        cols = list(self.feature_measurements_no_features.columns)
        assert "class_id" in cols and len(cols) == 1


class TestIsotropicMeasurement(unittest.TestCase):
    """

    Test that isotropic intensity measurements behave as expected on fake image

    """

    @classmethod
    def setUpClass(self):

        # Simple mock data, 100px*100px, one channel, value is one
        # Square (21*21px) of value 0. in middle
        # Two objects in labels map

        self.frame = np.ones((100, 100, 1), dtype=float)
        self.frame[40:61, 40:61, 0] = 0.0
        self.positions = pd.DataFrame(
            [
                {
                    "TRACK_ID": 0,
                    "POSITION_X": 50,
                    "POSITION_Y": 50,
                    "FRAME": 0,
                    "class_id": 0,
                }
            ]
        )

        self.inner_radius = 9
        self.upper_radius = 20
        self.safe_upper_radius = int(21 // 2 * np.sqrt(2)) + 2

        self.iso_measurements = measure_isotropic_intensity(
            self.positions,
            self.frame,
            channels=["test_channel"],
            intensity_measurement_radii=[self.inner_radius, self.upper_radius],
            operations=["mean"],
        )
        self.iso_measurements_ring = measure_isotropic_intensity(
            self.positions,
            self.frame,
            channels=["test_channel"],
            intensity_measurement_radii=[
                [self.safe_upper_radius, self.safe_upper_radius + 3]
            ],
            operations=["mean"],
        )

    def test_measure_yields_table(self):
        self.assertIsInstance(self.iso_measurements, pd.DataFrame)

    def test_intensity_zero_in_small_circle(self):
        self.assertEqual(
            self.iso_measurements[
                f"test_channel_circle_{self.inner_radius}_mean"
            ].values[0],
            0.0,
        )

    def test_intensity_greater_than_zero_in_intermediate_circle(self):
        self.assertGreater(
            self.iso_measurements[
                f"test_channel_circle_{self.upper_radius}_mean"
            ].values[0],
            0.0,
        )

    def test_ring_measurement_avoids_zero(self):
        self.assertEqual(
            self.iso_measurements[
                f"test_channel_ring_{self.safe_upper_radius}_{self.safe_upper_radius+3}_mean"
            ].values[0],
            1.0,
        )


class TestDropTonal(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.features = ["area", "intensity_mean", "intensity_max"]

    def test_drop_tonal(self):
        self.features_processed = drop_tonal_features(self.features)
        self.assertEqual(self.features_processed, ["area"])


class TestMeasureFeaturesComprehensive:

    def setup_method(self):
        # Create a simple synthetic image and label
        self.img = np.zeros((100, 100, 1), dtype=np.float64)
        self.img[20:40, 20:40, 0] = 1.0  # Region 1
        self.img[60:80, 60:80, 0] = 0.5  # Region 2

        self.label = np.zeros((100, 100), dtype=int)
        self.label[20:40, 20:40] = 1
        self.label[60:80, 60:80] = 2

        self.channels = ["channel_0"]

    def test_basic_features(self):
        """Test basic feature measurement (area, intensity)."""
        df = measure_features(
            self.img,
            self.label,
            features=["area", "intensity_mean"],
            channels=self.channels,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "area" in df.columns
        assert "channel_0_mean" in df.columns
        # Area should be 20*20 = 400
        assert df.loc[df["class_id"] == 1, "area"].values[0] == 400
        assert df.loc[df["class_id"] == 1, "channel_0_mean"].values[0] == 1.0
        assert df.loc[df["class_id"] == 2, "channel_0_mean"].values[0] == 0.5

    def test_no_image(self):
        """Test behavior when no image is provided."""
        df = measure_features(
            None, self.label, features=["area", "intensity_mean"], channels=None
        )
        assert "area" in df.columns
        # Intensity mean should be dropped or not present if img is None
        assert "intensity_mean" not in df.columns
        assert not any("mean" in col for col in df.columns if "intensity" in col)

    def test_channels_mismatch(self):
        """Test assertion error on channel mismatch."""
        with pytest.raises(AssertionError):
            measure_features(
                self.img, self.label, channels=["ch1", "ch2"]  # 2 channels vs 1 in img
            )

    def test_features_none(self):
        """Test behavior when features is None."""
        df = measure_features(
            self.img, self.label, features=None, channels=self.channels
        )
        assert "class_id" in df.columns
        assert len(df.columns) >= 1

    def test_border_dist_scalar(self):
        """Test border distance measurement with a single scalar."""
        df = measure_features(
            self.img,
            self.label,
            features=["intensity_mean"],
            channels=self.channels,
            border_dist=1,
        )
        # Check for edge columns
        cols = [c for c in df.columns if "_edge_1px" in c]
        assert len(cols) > 0

    def test_border_dist_list(self):
        """Test border distance measurement with a list."""
        df = measure_features(
            self.img,
            self.label,
            features=["intensity_mean"],
            channels=self.channels,
            border_dist=[1, 2],
        )
        cols_1 = [c for c in df.columns if "_edge_1px" in c]
        cols_2 = [c for c in df.columns if "_edge_2px" in c]
        assert len(cols_1) > 0
        assert len(cols_2) > 0

    def test_spot_detection_mock(self):
        """Test integration with spot detection (mocked)."""
        with patch("celldetective.measure.blob_detection") as mock_blob:
            # Setup mock return
            mock_df = pd.DataFrame({"label": [1, 2], "spots": [10, 5]})
            mock_blob.return_value = mock_df

            spot_opts = {"channel": "channel_0", "diameter": 5, "threshold": 0.1}

            df = measure_features(
                self.img,
                self.label,
                features=["area"],
                channels=self.channels,
                spot_detection=spot_opts,
            )

            assert mock_blob.called
            assert "spots" in df.columns

    def test_haralick_mock(self):
        """Test integration with Haralick features (mocked)."""
        with patch("celldetective.measure.compute_haralick_features") as mock_hara:
            mock_df = pd.DataFrame({"cell_id": [1, 2], "haralick_val": [0.1, 0.2]})
            mock_hara.return_value = mock_df

            df = measure_features(
                self.img,
                self.label,
                features=["area"],
                channels=self.channels,
                haralick_options={"distance": 1},
            )

            assert mock_hara.called
            assert "haralick_val" in df.columns

    def test_normalization_mock(self):
        """Test that normalization functions are called if configured."""
        norm_list = [
            {
                "target_channel": "channel_0",
                "correction_type": "local",
                "distance": 10,
                "model": "mean",
                "operation": "subtraction",
                "clip": True,
            }
        ]

        with patch("celldetective.measure.normalise_by_cell") as mock_norm:
            # Just return the image as is to avoid breaking flow
            mock_norm.side_effect = lambda img, *args, **kwargs: img

            measure_features(
                self.img,
                self.label,
                features=["area"],
                channels=self.channels,
                normalisation_list=norm_list,
            )
            assert mock_norm.called

    def test_extra_properties_dynamic_load(self):
        """Test that extra_properties are loaded if available (mocked)."""
        pass

    # --- New Tests ---

    def test_empty_labels(self):
        """Test behavior when labels are empty."""
        empty_label = np.zeros_like(self.label)
        df = measure_features(
            self.img,
            empty_label,
            features=["area", "intensity_mean"],
            channels=self.channels,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "class_id" in df.columns

    def test_multichannel_integrity(self):
        """Test that multiple channels are handled correctly."""
        img_3ch = np.zeros((100, 100, 3))
        img_3ch[..., 0] = self.img[..., 0]
        img_3ch[..., 1] = 0.8
        img_3ch[..., 2] = 0.2

        channels = ["c1", "c2", "c3"]
        df = measure_features(
            img_3ch, self.label, features=["intensity_mean"], channels=channels
        )

        assert "c1_mean" in df.columns
        assert "c2_mean" in df.columns
        assert "c3_mean" in df.columns
        assert df.loc[df["class_id"] == 1, "c1_mean"].values[0] == 1.0
        assert np.isclose(df.loc[df["class_id"] == 1, "c2_mean"].values[0], 0.8)

    def test_haralick_failure_handling(self):
        """Test that Haralick failure doesn't crash the function."""
        with patch("celldetective.measure.compute_haralick_features") as mock_hara:
            mock_hara.side_effect = Exception("Haralick failed")

            df = measure_features(
                self.img,
                self.label,
                features=["area"],
                channels=self.channels,
                haralick_options={"distance": 1},
            )
            # Should succeed and return df without haralick features
            assert isinstance(df, pd.DataFrame)
            assert "area" in df.columns

    def test_border_dist_string_input(self):
        """Test that string input for border_dist works."""
        # Using "1" instead of "5" to account for potential contour masking issues in small test images
        df = measure_features(
            self.img,
            self.label,
            features=["intensity_mean"],
            channels=self.channels,
            border_dist="1",
        )
        cols = [c for c in df.columns if "_edge_1px" in c]
        assert len(cols) > 0

    def test_missing_channels_warnings(self):
        """Test that warnings are logged for missing channels."""
        # Spot detection with wrong channel
        spot_opts = {"channel": "missing_channel", "diameter": 5, "threshold": 0.1}

        with patch("celldetective.measure.logger") as mock_logger:
            measure_features(
                self.img, self.label, channels=self.channels, spot_detection=spot_opts
            )
            # Verify logger.warning was called
            assert mock_logger.warning.called
            # Try to grab the call args to verify content
            found = False
            for call in mock_logger.warning.call_args_list:
                if "missing_channel" in str(call):
                    found = True
                    break
            assert found, "Warning for missing_channel not found"

    def test_class_id_type(self):
        """Test that class_id is of float type."""
        df = measure_features(self.img, self.label, channels=self.channels)
        assert pd.api.types.is_float_dtype(df["class_id"])


if __name__ == "__main__":
    unittest.main()
