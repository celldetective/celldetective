import unittest
import numpy as np
import pandas as pd
from celldetective.tracking import (
    filter_by_endpoints,
    extrapolate_tracks,
    filter_by_tracklength,
    interpolate_time_gaps,
    interpolate_nan_properties,
    compute_instantaneous_velocity,
    compute_instantaneous_diffusion,
    write_first_detection_class,
    clean_trajectories,
)


class TestTrackFilteringByEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tracks = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 30, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 40, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 10, "POSITION_Y": 25},
            ]
        )

    def test_filter_not_in_last(self):
        self.filtered_tracks = filter_by_endpoints(
            self.tracks, remove_not_in_first=False, remove_not_in_last=True
        )
        track_ids = list(self.filtered_tracks["TRACK_ID"].unique())
        self.assertEqual(track_ids, [0.0])

    def test_filter_not_in_first(self):
        self.filtered_tracks = filter_by_endpoints(
            self.tracks, remove_not_in_first=True, remove_not_in_last=False
        )
        track_ids = list(self.filtered_tracks["TRACK_ID"].unique())
        self.assertEqual(track_ids, [0.0, 2.0])

    def test_no_filter_does_nothing(self):
        self.filtered_tracks = filter_by_endpoints(
            self.tracks, remove_not_in_first=False, remove_not_in_last=False
        )
        track_ids = list(self.filtered_tracks["TRACK_ID"].unique())
        self.assertEqual(track_ids, list(self.tracks["TRACK_ID"].unique()))


class TestTrackFilteringByLength(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tracks = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 30, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 40, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 10, "POSITION_Y": 25},
            ]
        )

    def test_filter_by_tracklength_of_zero(self):
        self.filtered_tracks = filter_by_tracklength(self.tracks, minimum_tracklength=0)
        track_ids = list(self.filtered_tracks["TRACK_ID"].unique())
        self.assertEqual(track_ids, [0.0, 1.0, 2.0])

    def test_filter_by_tracklength_of_three(self):
        self.filtered_tracks = filter_by_tracklength(self.tracks, minimum_tracklength=3)
        track_ids = list(self.filtered_tracks["TRACK_ID"].unique())
        self.assertEqual(track_ids, [0.0])


class TestTrackInterpolation(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.tracks = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                # {"TRACK_ID": 0., "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                # {"TRACK_ID": 2., "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )
        self.tracks_real_intep = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )

    def test_interpolate_tracks_as_expected(self):
        self.interpolated_tracks = interpolate_time_gaps(self.tracks)
        # We use allclose because interpolation returns floats and strict equality might fail on some platforms
        self.assertTrue(
            np.allclose(
                self.interpolated_tracks.to_numpy().astype(float),
                self.tracks_real_intep.to_numpy().astype(float),
                equal_nan=True,
            )
        )


class TestTrackExtrapolation(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.tracks = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )
        self.tracks_pre_extrapol = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 0, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )
        self.tracks_post_extrapol = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 1.0, "FRAME": 3, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 3, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )

        self.tracks_full_extrapol = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": 5},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 0, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {"TRACK_ID": 1.0, "FRAME": 2, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 1.0, "FRAME": 3, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 2, "POSITION_X": 0, "POSITION_Y": 25},
                {"TRACK_ID": 2.0, "FRAME": 3, "POSITION_X": 0, "POSITION_Y": 25},
            ]
        )

    def test_pre_extrapolate(self):
        self.extrapolated_tracks = extrapolate_tracks(self.tracks, post=False, pre=True)
        self.assertTrue(
            np.array_equal(
                self.extrapolated_tracks.to_numpy(),
                self.tracks_pre_extrapol.to_numpy(),
                equal_nan=True,
            )
        )

    def test_post_extrapolate(self):
        self.extrapolated_tracks = extrapolate_tracks(self.tracks, post=True, pre=False)
        self.assertTrue(
            np.array_equal(
                self.extrapolated_tracks.to_numpy(),
                self.tracks_post_extrapol.to_numpy(),
                equal_nan=True,
            )
        )

    def test_full_extrapolate(self):
        self.extrapolated_tracks = extrapolate_tracks(self.tracks, post=True, pre=True)
        self.assertTrue(
            np.array_equal(
                self.extrapolated_tracks.to_numpy(),
                self.tracks_full_extrapol.to_numpy(),
                equal_nan=True,
            )
        )


class TestTrackInterpolationNaN(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tracks = pd.DataFrame(
            [
                {"TRACK_ID": 0.0, "FRAME": 0, "POSITION_X": np.nan, "POSITION_Y": 15},
                {"TRACK_ID": 0.0, "FRAME": 1, "POSITION_X": 15, "POSITION_Y": 10},
                {"TRACK_ID": 0.0, "FRAME": 2, "POSITION_X": 20, "POSITION_Y": np.nan},
                {"TRACK_ID": 0.0, "FRAME": 3, "POSITION_X": 25, "POSITION_Y": 0},
                {"TRACK_ID": 1.0, "FRAME": 1, "POSITION_X": 5, "POSITION_Y": 20},
                {
                    "TRACK_ID": 1.0,
                    "FRAME": 2,
                    "POSITION_X": np.nan,
                    "POSITION_Y": np.nan,
                },
                {"TRACK_ID": 1.0, "FRAME": 3, "POSITION_X": 15, "POSITION_Y": 30},
            ]
        )

    def test_interpolate_nan(self):
        interpolated = interpolate_nan_properties(self.tracks.copy())

        # Track 0: Start NaN should be filled by first valid (bfill), End NaN should be filled by last valid (ffill)
        # But `interpolate_per_track` uses limit_direction="both", so it acts as ffill+bfill

        # Track 0, Frame 0, Pos X: Should be 15 (bfill from next)
        self.assertEqual(
            interpolated.loc[
                (interpolated.TRACK_ID == 0) & (interpolated.FRAME == 0), "POSITION_X"
            ].values[0],
            15.0,
        )

        # Track 0, Frame 2, Pos Y: Should be (10 + 0) / 2 = 5 (linear interp)
        self.assertEqual(
            interpolated.loc[
                (interpolated.TRACK_ID == 0) & (interpolated.FRAME == 2), "POSITION_Y"
            ].values[0],
            5.0,
        )

        # Track 1, Frame 2, Pos X: (5 + 15) / 2 = 10
        self.assertEqual(
            interpolated.loc[
                (interpolated.TRACK_ID == 1) & (interpolated.FRAME == 2), "POSITION_X"
            ].values[0],
            10.0,
        )


class TestPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Linear motion: dx=1, dy=0, dt=1 -> v=1
        self.tracks_linear = pd.DataFrame(
            [
                {"TRACK_ID": 0, "FRAME": 0, "POSITION_X": 0, "POSITION_Y": 0},
                {"TRACK_ID": 0, "FRAME": 1, "POSITION_X": 1, "POSITION_Y": 0},
                {"TRACK_ID": 0, "FRAME": 2, "POSITION_X": 2, "POSITION_Y": 0},
            ]
        )

        # Stationary: v=0
        self.tracks_static = pd.DataFrame(
            [
                {"TRACK_ID": 1, "FRAME": 0, "POSITION_X": 10, "POSITION_Y": 10},
                {"TRACK_ID": 1, "FRAME": 1, "POSITION_X": 10, "POSITION_Y": 10},
            ]
        )

    def test_velocity(self):
        v_linear = compute_instantaneous_velocity(self.tracks_linear.copy())
        # First point has NaN velocity ideally or 0 depending on implementation.
        # Looking at code: diff() produces NaN for first element.
        self.assertTrue(np.isnan(v_linear.iloc[0]["velocity"]))
        self.assertTrue(np.allclose(v_linear.iloc[1:]["velocity"], 1.0))

        v_static = compute_instantaneous_velocity(self.tracks_static.copy())
        self.assertTrue(np.allclose(v_static.iloc[1:]["velocity"], 0.0))

    def test_diffusion(self):
        # Simple diffusion test
        # Track 0: Linear motion should have diffusion related to displacement
        d_linear = compute_instantaneous_diffusion(self.tracks_linear.copy())
        # Diffusion computation requires 3 points (t-1, t, t+1)
        # So for 3 points, only the middle one (index 1) might have value

        # We need more points for a meaningful test or check code logic
        # Code: if len(x) > 3: ...

        tracks_long = pd.DataFrame(
            {
                "TRACK_ID": [0] * 5,
                "FRAME": [0, 1, 2, 3, 4],
                "POSITION_X": [0, 1, 2, 3, 4],
                "POSITION_Y": [0, 0, 0, 0, 0],
            }
        )
        d_long = compute_instantaneous_diffusion(tracks_long)
        self.assertIn("diffusion", d_long.columns)
        # Check that we have some non-nan values
        valid_diff = d_long["diffusion"].dropna()
        self.assertGreater(len(valid_diff), 0)


class TestFirstDetection(unittest.TestCase):
    def test_first_detection_start(self):
        df = pd.DataFrame(
            [
                {
                    "TRACK_ID": 0,
                    "FRAME": 0,
                    "POSITION_X": 10,
                    "POSITION_Y": 10,
                    "class_id": 1,
                },
                {
                    "TRACK_ID": 0,
                    "FRAME": 1,
                    "POSITION_X": 11,
                    "POSITION_Y": 11,
                    "class_id": 1,
                },
            ]
        )
        # Track starts at frame 0 -> class 2 (invalid/start)
        res = write_first_detection_class(df.copy())
        self.assertEqual(res["class_firstdetection"].iloc[0], 2)
        self.assertEqual(res["t_firstdetection"].iloc[0], -1)

    def test_first_detection_middle(self):
        df = pd.DataFrame(
            [
                {
                    "TRACK_ID": 1,
                    "FRAME": 5,
                    "POSITION_X": 50,
                    "POSITION_Y": 50,
                    "class_id": 1,
                },
                {
                    "TRACK_ID": 1,
                    "FRAME": 6,
                    "POSITION_X": 51,
                    "POSITION_Y": 51,
                    "class_id": 1,
                },
            ]
        )
        # Track starts at frame 5 -> class 0 (valid)
        res = write_first_detection_class(df.copy())
        self.assertEqual(res["class_firstdetection"].iloc[0], 0)
        # t_first should be float(t_first) - dt (dt=1) => 5 - 1 = 4.0
        self.assertEqual(res["t_firstdetection"].iloc[0], 4.0)

    def test_first_detection_edge(self):
        df = pd.DataFrame(
            [
                {
                    "TRACK_ID": 2,
                    "FRAME": 5,
                    "POSITION_X": 5,
                    "POSITION_Y": 50,
                    "class_id": 1,
                },  # Near edge x=5 < 20
            ]
        )
        # Edge threshold default 20
        res = write_first_detection_class(df.copy(), img_shape=(100, 100))
        self.assertEqual(res["class_firstdetection"].iloc[0], 2)


class TestCleanTrajectories(unittest.TestCase):
    def test_clean_pipeline(self):
        # A mix of short tracks, nan gaps, time gaps
        tracks = pd.DataFrame(
            [
                # Short track (length 2)
                {"TRACK_ID": 0, "FRAME": 0, "POSITION_X": 0, "POSITION_Y": 0},
                {"TRACK_ID": 0, "FRAME": 1, "POSITION_X": 1, "POSITION_Y": 1},
                # Good track with gap
                {"TRACK_ID": 1, "FRAME": 0, "POSITION_X": 0, "POSITION_Y": 0},
                {
                    "TRACK_ID": 1,
                    "FRAME": 2,
                    "POSITION_X": 2,
                    "POSITION_Y": 2,
                },  # Time gap
                # Track with NaN
                {"TRACK_ID": 2, "FRAME": 0, "POSITION_X": 0, "POSITION_Y": 0},
                {"TRACK_ID": 2, "FRAME": 1, "POSITION_X": np.nan, "POSITION_Y": 1},
                {"TRACK_ID": 2, "FRAME": 2, "POSITION_X": 2, "POSITION_Y": 2},
            ]
        )

        # Clean: min length 3, interpolate time, interpolate nan
        cleaned = clean_trajectories(
            tracks.copy(),
            minimum_tracklength=2,
            interpolate_position_gaps=True,
            interpolate_na=True,
            remove_not_in_first=False,
            remove_not_in_last=False,
        )

        # Track 0 (len 2) is NOT > 2, so it should be gone.
        self.assertNotIn(0, cleaned["TRACK_ID"].unique())

        # Track 1 should have frame 1 filled
        self.assertIn(1, cleaned["TRACK_ID"].unique())
        t1 = cleaned[cleaned.TRACK_ID == 1]
        self.assertIn(1.0, t1.FRAME.values)  # interpolated frame

        # Track 2 should have nan filled
        self.assertIn(2, cleaned["TRACK_ID"].unique())
        t2_f1 = cleaned[(cleaned.TRACK_ID == 2) & (cleaned.FRAME == 1)]
        self.assertFalse(np.isnan(t2_f1.POSITION_X.values[0]))


if __name__ == "__main__":
    unittest.main()
