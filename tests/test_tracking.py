import unittest
import numpy as np
import pandas as pd
from celldetective.tracking import (
    filter_by_endpoints,
    extrapolate_tracks,
    filter_by_tracklength,
    interpolate_time_gaps,
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
        # Sort columns to ensure order is consistent
        self.interpolated_tracks = self.interpolated_tracks.sort_index(axis=1)
        self.tracks_real_intep = self.tracks_real_intep.sort_index(axis=1)

        # Allow for some floating point tolerance and ignore int/float differences
        pd.testing.assert_frame_equal(
            self.interpolated_tracks,
            self.tracks_real_intep,
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
            rtol=1e-5,
            atol=1e-8,
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


if __name__ == "__main__":
    unittest.main()
