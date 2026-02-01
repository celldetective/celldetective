import unittest
import pandas as pd
import numpy as np
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


class TestNumpyArrayHandling(unittest.TestCase):
    """
    Regression test for bug where passing channels as numpy array caused AttributeError.
    Fix: 'numpy.ndarray' object has no attribute 'index'
    """

    @classmethod
    def setUpClass(self):
        self.frame = np.ones((100, 100, 2), dtype=float)
        self.labels = np.zeros((100, 100), dtype=int)
        self.labels[50:60, 50:60] = 1

        # KEY: Pass channels as numpy array to trigger the potential bug
        self.channels = np.array(["channel_1", "channel_2"])

    def test_measure_features_with_numpy_channels(self):
        """
        Test that measure_features works when channels is a numpy array.
        Prevents regression of AttributeError: 'numpy.ndarray' object has no attribute 'index'
        """
        try:
            df = measure_features(
                self.frame,
                self.labels,
                features=["intensity_mean"],
                channels=self.channels,
            )
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn("channel_1_mean", df.columns)
        except AttributeError as e:
            self.fail(f"measure_features failed with numpy array channels: {e}")

    def test_spot_detection_with_numpy_channels_match(self):
        """
        Test spot detection logic with numpy array channels.
        The bug also appeared in spot detection channel matching.
        """
        spot_opts = {
            "channel": "channel_1",  # Matches one of the channels
            "diameter": 5,
            "threshold": 0.1,
        }

        try:
            # Should not raise AttributeError
            df = measure_features(
                self.frame,
                self.labels,
                channels=self.channels,
                spot_detection=spot_opts,
            )
            self.assertIsInstance(df, pd.DataFrame)
        except AttributeError as e:
            self.fail(f"Spot detection failed with numpy array channels: {e}")


if __name__ == "__main__":
    unittest.main()
