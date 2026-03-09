"""
Unit tests for track collapse functions in celldetective.utils.data_cleaning.

Covers:
- collapse_trajectories_by_status: core logic, invalid inputs, all projection modes,
  pairs population, ordering of output columns, regression for missing import.
"""

import pytest
import numpy as np
import pandas as pd

from celldetective.utils.data_cleaning import collapse_trajectories_by_status


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_track_df():
    """Two tracks, each with 3 frames, a binary status column."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 6,
            "TRACK_ID": [1, 1, 1, 2, 2, 2],
            "FRAME": [0, 1, 2, 0, 1, 2],
            "POSITION_X": [10.0, 12.0, 14.0, 20.0, 22.0, 24.0],
            "POSITION_Y": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "area": [100.0, 110.0, 120.0, 200.0, 210.0, 220.0],
            "well_name": ["W1"] * 6,
            "pos_name": ["100"] * 6,
            "status": [0, 0, 1, 0, 1, 1],
        }
    )


@pytest.fixture
def multi_status_df():
    """Three tracks with three distinct status values (0, 1, 2)."""
    rows = []
    for tid in range(1, 4):
        for frame in range(4):
            rows.append(
                {
                    "position": "pos1",
                    "TRACK_ID": tid,
                    "FRAME": frame,
                    "POSITION_X": float(tid * 10 + frame),
                    "area": float(tid * 100 + frame),
                    "status": frame % 3,  # 0, 1, 2, 0
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def pairs_df():
    """Minimal pairs-population dataframe."""
    return pd.DataFrame(
        {
            "position": ["pos1"] * 4,
            "REFERENCE_ID": [1, 1, 2, 2],
            "NEIGHBOR_ID": [10, 10, 20, 20],
            "FRAME": [0, 1, 0, 1],
            "signal": [1.0, 3.0, 2.0, 4.0],
            "reference_population": ["eff"] * 4,
            "neighbor_population": ["tar"] * 4,
            "status": [0, 1, 0, 1],
        }
    )


# ---------------------------------------------------------------------------
# Invalid / edge-case inputs
# ---------------------------------------------------------------------------


class TestCollapseInvalidInputs:
    def test_returns_none_when_status_is_none(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status=None)
        assert result is None

    def test_returns_none_when_status_not_in_columns(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="nonexistent_col"
        )
        assert result is None

    def test_handles_all_nan_status(self, basic_track_df):
        df = basic_track_df.copy()
        df["status"] = np.nan
        # dropna(subset='status') will produce empty df, should raise or return
        # gracefully – currently concat on empty list would raise; verify it
        # does not silently swallow the error and that we get a meaningful result.
        # The function drops all rows, leaving no sections → concat([]) raises.
        with pytest.raises(Exception):
            collapse_trajectories_by_status(df, status="status")


# ---------------------------------------------------------------------------
# Return shape / type
# ---------------------------------------------------------------------------


class TestCollapseReturnType:
    def test_returns_dataframe(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert isinstance(result, pd.DataFrame)

    def test_frame_column_removed(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert "FRAME" not in result.columns

    def test_duration_in_state_column_present(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert "duration_in_state" in result.columns

    def test_status_column_present_in_output(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert "status" in result.columns

    def test_row_count_equals_track_status_combinations(self, basic_track_df):
        """
        track 1: statuses 0, 1   → 2 rows
        track 2: statuses 0, 1   → 2 rows
        Total: 4 rows.
        """
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Projection correctness
# ---------------------------------------------------------------------------


class TestCollapseProjectionValues:
    def test_mean_projection(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection="mean"
        )
        # track 1, status 0 → POSITION_X mean of [10, 12] = 11.0
        row = result[(result["TRACK_ID"] == 1) & (result["status"] == 0)]
        assert not row.empty
        assert pytest.approx(row["POSITION_X"].values[0], abs=1e-6) == 11.0

    def test_min_projection(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection="min"
        )
        # track 2, status 1 → area min of [210, 220] = 210.0
        row = result[(result["TRACK_ID"] == 2) & (result["status"] == 1)]
        assert not row.empty
        assert pytest.approx(row["area"].values[0], abs=1e-6) == 210.0

    def test_max_projection(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection="max"
        )
        # track 1, status 1 → POSITION_X max of [14] = 14.0
        row = result[(result["TRACK_ID"] == 1) & (result["status"] == 1)]
        assert not row.empty
        assert pytest.approx(row["POSITION_X"].values[0], abs=1e-6) == 14.0

    def test_sum_projection(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection="sum"
        )
        # track 1, status 0 → POSITION_X sum of [10, 12] = 22.0
        row = result[(result["TRACK_ID"] == 1) & (result["status"] == 0)]
        assert not row.empty
        assert pytest.approx(row["POSITION_X"].values[0], abs=1e-6) == 22.0

    def test_median_projection(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection="median"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "op", ["mean", "median", "min", "max", "sum", "first", "last"]
    )
    def test_all_projection_modes_return_dataframe(self, basic_track_df, op):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", projection=op
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Duration-in-state values
# ---------------------------------------------------------------------------


class TestCollapseDurationInState:
    def test_duration_in_state_correct(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        # track 1: 2 frames in status=0, 1 frame in status=1
        row0 = result[(result["TRACK_ID"] == 1) & (result["status"] == 0)]
        row1 = result[(result["TRACK_ID"] == 1) & (result["status"] == 1)]
        assert row0["duration_in_state"].values[0] == 2
        assert row1["duration_in_state"].values[0] == 1

    def test_duration_sums_to_total_frames(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        tid1 = result[result["TRACK_ID"] == 1]
        assert tid1["duration_in_state"].sum() == 3  # 3 frames per track


# ---------------------------------------------------------------------------
# Column ordering (regression: key columns promoted to front)
# ---------------------------------------------------------------------------


class TestCollapseColumnOrdering:
    def test_key_columns_are_at_front_non_pairs(self, basic_track_df):
        """
        The function does insert(0) in order [duration_in_state, status, TRACK_ID],
        so the actual column order is TRACK_ID, status, duration_in_state at positions 0-2.
        """
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", population="targets"
        )
        first_three = list(result.columns[:3])
        assert "TRACK_ID" in first_three
        assert "status" in first_three
        assert "duration_in_state" in first_three

    def test_frame_not_in_columns(self, basic_track_df):
        result = collapse_trajectories_by_status(
            basic_track_df, status="status", population="targets"
        )
        assert "FRAME" not in result.columns


# ---------------------------------------------------------------------------
# Multi-status (more than 2 distinct values)
# ---------------------------------------------------------------------------


class TestCollapseMultiStatus:
    def test_three_status_values_produce_correct_rows(self, multi_status_df):
        # status values per track: [0,1,2,0] → unique {0,1,2}
        # each track has 3 unique statuses → 3 × 3 = 9 rows
        result = collapse_trajectories_by_status(multi_status_df, status="status")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 9

    def test_all_statuses_present_in_result(self, multi_status_df):
        result = collapse_trajectories_by_status(multi_status_df, status="status")
        assert set(result["status"].unique()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Pairs population
# ---------------------------------------------------------------------------


class TestCollapsePairsPopulation:
    def test_pairs_returns_dataframe(self, pairs_df):
        result = collapse_trajectories_by_status(
            pairs_df,
            status="status",
            population="pairs",
            groupby_columns=["position", "REFERENCE_ID", "NEIGHBOR_ID"],
        )
        assert isinstance(result, pd.DataFrame)

    def test_pairs_duration_in_state_present(self, pairs_df):
        result = collapse_trajectories_by_status(
            pairs_df,
            status="status",
            population="pairs",
            groupby_columns=["position", "REFERENCE_ID", "NEIGHBOR_ID"],
        )
        assert "duration_in_state" in result.columns

    def test_pairs_frame_removed(self, pairs_df):
        result = collapse_trajectories_by_status(
            pairs_df,
            status="status",
            population="pairs",
            groupby_columns=["position", "REFERENCE_ID", "NEIGHBOR_ID"],
        )
        assert "FRAME" not in result.columns


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------


class TestCollapseSorting:
    def test_output_is_sorted_by_groupby_and_status(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        # After sort_values(by=['position', 'TRACK_ID', 'status'])
        sorted_result = result.sort_values(
            by=["position", "TRACK_ID", "status"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, sorted_result)

    def test_output_index_is_reset(self, basic_track_df):
        result = collapse_trajectories_by_status(basic_track_df, status="status")
        assert list(result.index) == list(range(len(result)))
