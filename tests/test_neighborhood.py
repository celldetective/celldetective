"""
Tests for celldetective.neighborhood.

Unit tests build tiny tables by hand (a few cells whose distances are obvious)
and check exact neighbor lists and counts. The public-API tests at the end run
``compute_neighborhood_at_position`` and
``compute_contact_neighborhood_at_position`` on a position folder written to
``tmp_path`` and check the tables they write back.

Contact-site intensity (``_contact_site_mask``, ``_measure_contact_intensity_at_t``,
``mask_contact_neighborhood(intensity_images=...)``) is covered in
``test_contact_site_intensity.py``.
"""

import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from celldetective.neighborhood import (
    compute_attention_weight,
    compute_contact_neighborhood_at_position,
    compute_neighborhood_at_position,
    compute_neighborhood_metrics,
    contact_neighborhood,
    distance_cut_neighborhood,
    extract_neighborhood_in_pair_table,
    find_contact_neighbors,
    mean_neighborhood_after_event,
    mean_neighborhood_before_event,
    set_live_status,
)


def neighborhood_kwargs(**overrides):
    """A fresh kwargs dict (the *_at_position functions mutate the one they get)."""
    kwargs = {
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    }
    kwargs.update(overrides)
    return kwargs


def ids_of(neighbors):
    return [n["id"] for n in neighbors]


# ---------------------------------------------------------------------------
# set_live_status
# ---------------------------------------------------------------------------


class TestSetLiveStatus:

    @pytest.mark.parametrize(
        "status",
        [None, [None, None], ["live_status", "live_status"]],
        ids=["none", "pair-of-none", "explicit-live_status"],
    )
    def test_no_status_marks_every_cell_live(self, status):
        setA = pd.DataFrame({"TRACK_ID": [1, 2]})
        setB = pd.DataFrame({"TRACK_ID": [3]})
        setA, setB, status = set_live_status(setA, setB, status, None)
        assert status == ["live_status", "live_status"]
        assert setA["live_status"].tolist() == [1, 1]
        assert setB["live_status"].tolist() == [1]

    def test_past_events_count_as_events_and_not_option_inverts(self):
        """Status 2 (event already happened) becomes 1; with not_status_option the
        column is inverted into a new `not_` column and that name is returned.
        Values other than 0/1 become NaN in the inverted column."""
        setA = pd.DataFrame({"TRACK_ID": [1, 2, 3, 4], "status_dead": [0, 1, 2, np.nan]})
        setB = pd.DataFrame({"TRACK_ID": [5, 6], "status_act": [0, 2]})

        setA, setB, status = set_live_status(
            setA, setB, ["status_dead", "status_act"], [True, False]
        )

        assert status == ["not_status_dead", "status_act"]
        np.testing.assert_array_equal(setA["status_dead"].to_numpy(), [0, 1, 1, np.nan])
        np.testing.assert_array_equal(
            setA["not_status_dead"].to_numpy(dtype=float), [1, 0, 0, np.nan]
        )
        assert setB["status_act"].tolist() == [0, 1]
        assert "not_status_act" not in setB.columns

    def test_one_sided_status_keeps_the_other_side_live(self):
        setA = pd.DataFrame({"TRACK_ID": [1], "status_x": [0]})
        setB = pd.DataFrame({"TRACK_ID": [2]})
        setA, setB, status = set_live_status(setA, setB, ["status_x", None], [False, False])
        assert status == ["status_x", "live_status"]
        assert setB["live_status"].tolist() == [1]

    def test_more_than_two_status_columns_is_rejected(self):
        with pytest.raises(ValueError, match="only two columns"):
            set_live_status(pd.DataFrame(), pd.DataFrame(), ["a", "b", "c"], None)

    def test_missing_status_column_raises_keyerror(self):
        setA = pd.DataFrame({"TRACK_ID": [1]})
        setB = pd.DataFrame({"TRACK_ID": [2]})
        with pytest.raises(KeyError, match="status_absent"):
            set_live_status(setA, setB, ["status_absent", None], None)


# ---------------------------------------------------------------------------
# compute_attention_weight
# ---------------------------------------------------------------------------


class TestComputeAttentionWeight:
    """Rows are reference cells A (ids 10 alive, 20 dead), columns cells B.
    B0 is within reach of both A cells (A20 closer), B1 of A10 only, B2 of none."""

    dist = np.array([[2.0, 5.0, 50.0], [1.0, 50.0, 60.0]])
    status_A = np.array([1, 0])
    ids_A = np.array([10, 20])

    def test_weight_is_one_over_the_number_of_cells_sharing_b(self):
        weights, closest = compute_attention_weight(
            self.dist, 10, self.status_A, self.ids_A, axis=1, include_dead_weight=True
        )
        np.testing.assert_array_equal(weights, [0.5, 1.0, np.nan])
        np.testing.assert_array_equal(closest, [20, 10, 10])

    def test_dead_cells_neither_share_weight_nor_count_as_closest(self):
        weights, closest = compute_attention_weight(
            self.dist, 10, self.status_A, self.ids_A, axis=1, include_dead_weight=False
        )
        np.testing.assert_array_equal(weights, [1.0, 1.0, np.nan])
        np.testing.assert_array_equal(closest, [10, 10, 10])


# ---------------------------------------------------------------------------
# distance_cut_neighborhood
# ---------------------------------------------------------------------------


def two_population_tables(n_frames=2):
    """Reference cell 1 at the origin; neighbor 7 at distance 5, neighbor 8 at 15."""
    setA = pd.DataFrame(
        {
            "TRACK_ID": [1] * n_frames,
            "FRAME": list(range(n_frames)),
            "POSITION_X": [0.0] * n_frames,
            "POSITION_Y": [0.0] * n_frames,
        }
    )
    setB = pd.DataFrame(
        {
            "TRACK_ID": [7] * n_frames + [8] * n_frames,
            "FRAME": list(range(n_frames)) * 2,
            "POSITION_X": [3.0] * n_frames + [15.0] * n_frames,
            "POSITION_Y": [4.0] * n_frames + [0.0] * n_frames,
        }
    )
    return setA, setB


class TestDistanceCutNeighborhood:

    def test_neighbors_within_the_radius_with_their_distance(self):
        setA, setB = two_population_tables()
        setA, setB = distance_cut_neighborhood(setA, setB, [10, 40], compute_cum_sum=False)

        for neighs in setA["neighborhood_2_circle_10_px"]:
            assert ids_of(neighs) == [7]
            assert neighs[0]["distance"] == 5.0
            assert neighs[0]["weight"] == 1.0
            assert neighs[0]["closest"] is True
        for neighs in setA["neighborhood_2_circle_40_px"]:
            assert sorted(ids_of(neighs)) == [7, 8]
            assert sorted(n["distance"] for n in neighs) == [5.0, 15.0]

    def test_symmetrize_writes_the_reference_cell_into_its_neighbors(self):
        setA, setB = two_population_tables()
        _, setB = distance_cut_neighborhood(setA, setB, 10, compute_cum_sum=False)
        col = setB.set_index(["TRACK_ID", "FRAME"])["neighborhood_2_circle_10_px"]
        for t in range(2):
            assert ids_of(col.loc[(7, t)]) == [1]
            assert col.loc[(7, t)][0]["distance"] == 5.0
            assert not isinstance(col.loc[(8, t)], list)  # out of reach: untouched

    def test_without_symmetrize_set_b_is_left_empty(self):
        setA, setB = two_population_tables()
        _, setB = distance_cut_neighborhood(
            setA, setB, 10, compute_cum_sum=False, symmetrize=False
        )
        assert setB["neighborhood_2_circle_10_px"].isna().all()

    def test_cumulated_presence_counts_frames_spent_as_neighbor(self):
        """Neighbor 7 is near at frames 0, 1 and 3 but away at frame 2: its
        presence count must keep the frames it was already there (1, 2, -, 3)."""
        setA, _ = two_population_tables(n_frames=4)
        setB = pd.DataFrame(
            {
                "TRACK_ID": [7] * 4,
                "FRAME": [0, 1, 2, 3],
                "POSITION_X": [3.0, 3.0, 300.0, 3.0],
                "POSITION_Y": [4.0, 4.0, 0.0, 4.0],
            }
        )
        setA, _ = distance_cut_neighborhood(setA, setB, 10, compute_cum_sum=True)
        neighs = setA.sort_values("FRAME")["neighborhood_2_circle_10_px"].tolist()
        assert [n[0]["cumulated_presence"] if n else None for n in neighs] == [1, 2, None, 3]
        assert [n[0]["cumulated_presence_weighted"] for n in neighs if n] == [1.0, 2.0, 3.0]

    def test_untracked_data_skips_cumulated_presence(self):
        """With only an ID column (no tracks), the cumulated presence is meaningless
        and must not be computed even if requested."""
        setA, setB = two_population_tables()
        setA = setA.rename(columns={"TRACK_ID": "ID"})
        setB = setB.rename(columns={"TRACK_ID": "ID"})
        setA, _ = distance_cut_neighborhood(setA, setB, 10, compute_cum_sum=True)
        for neighs in setA["neighborhood_2_circle_10_px"]:
            assert ids_of(neighs) == [7]
            assert "cumulated_presence" not in neighs[0]

    def test_self_mode_never_lists_a_cell_as_its_own_neighbor(self):
        cells = pd.DataFrame(
            {
                "TRACK_ID": [1, 2, 3],
                "FRAME": [0, 0, 0],
                "POSITION_X": [0.0, 3.0, 100.0],
                "POSITION_Y": [0.0, 4.0, 0.0],
            }
        )
        setA, _ = distance_cut_neighborhood(
            cells.copy(), cells.copy(), 10, mode="self", compute_cum_sum=False,
            symmetrize=False,
        )
        col = setA.set_index("TRACK_ID")["neighborhood_self_circle_10_px"]
        assert ids_of(col.loc[1]) == [2]
        assert ids_of(col.loc[2]) == [1]
        assert col.loc[3] == []

    def test_invalid_mode_or_missing_table_returns_none(self):
        setA, setB = two_population_tables()
        assert distance_cut_neighborhood(setA, setB, 10, mode="bogus") == (None, None)
        assert distance_cut_neighborhood(None, setB, 10) == (None, None)


# ---------------------------------------------------------------------------
# compute_neighborhood_metrics
# ---------------------------------------------------------------------------


class TestComputeNeighborhoodMetrics:

    def test_counts_per_frame_and_per_neighbor_status(self):
        """Frame 0: a live closest neighbor (weight 0.5) and two dead ones (weight 1
        each); frame 1: nobody; frame 2: not computed (NaN, must stay NaN)."""
        col = "neighborhood_x"
        table = pd.DataFrame(
            {
                "TRACK_ID": [1, 1, 1],
                "FRAME": [0, 1, 2],
                col: [
                    [
                        {"id": 7, "status": 1, "weight": 0.5, "closest": True},
                        {"id": 8, "status": 0, "weight": 1.0, "closest": False},
                        {"id": 9, "status": 0, "weight": 1.0, "closest": False},
                    ],
                    [],
                    np.nan,
                ],
            }
        )
        out = compute_neighborhood_metrics(table, col, decompose_by_status=True)

        expected = {
            "inclusive_count_": [3, 0, np.nan],
            "exclusive_count_": [1, 0, np.nan],
            "intermediate_count_": [2.5, 0, np.nan],
            "inclusive_count_s0_": [2, 0, np.nan],
            "inclusive_count_s1_": [1, 0, np.nan],
            "exclusive_count_s0_": [0, 0, np.nan],
            "exclusive_count_s1_": [1, 0, np.nan],
            "intermediate_count_s0_": [2.0, 0, np.nan],
            "intermediate_count_s1_": [0.5, 0, np.nan],
        }
        for prefix, values in expected.items():
            np.testing.assert_array_equal(
                out[prefix + col].to_numpy(dtype=float), values, err_msg=prefix
            )


# ---------------------------------------------------------------------------
# mean_neighborhood_before_event / mean_neighborhood_after_event
# ---------------------------------------------------------------------------


def track_with_counts(event_time):
    """One track over frames 0-3 with live-neighbor counts 1, 2, 3, 4. The plain
    (all-status) count is set to 10x to show that the s1 count is preferred."""
    col = "neighborhood_x"
    return col, pd.DataFrame(
        {
            "TRACK_ID": [1, 1, 1, 1],
            "FRAME": [0, 1, 2, 3],
            "t_event": [event_time] * 4,
            col: [[{}]] * 4,
            "inclusive_count_s1_" + col: [1.0, 2.0, 3.0, 4.0],
            "inclusive_count_" + col: [10.0, 20.0, 30.0, 40.0],
        }
    )


class TestMeanNeighborhoodAroundEvent:

    def test_mean_before_and_after_the_event_split_at_the_event_frame(self):
        col, table = track_with_counts(event_time=1)
        out = mean_neighborhood_before_event(table, col, "t_event", metrics=["inclusive"])
        out = mean_neighborhood_after_event(out, col, "t_event", metrics=["inclusive"])
        assert out[f"mean_count_inclusive_{col}_before_event"].tolist() == [1.5] * 4
        assert out[f"mean_count_inclusive_{col}_after_event"].tolist() == [3.5] * 4

    def test_no_event_averages_the_whole_track_before_and_nothing_after(self):
        """A negative event time means the event never happened."""
        col, table = track_with_counts(event_time=-1)
        out = mean_neighborhood_before_event(table, col, "t_event", metrics=["inclusive"])
        out = mean_neighborhood_after_event(out, col, "t_event", metrics=["inclusive"])
        assert out[f"mean_count_inclusive_{col}_before_event"].tolist() == [2.5] * 4
        assert f"mean_count_inclusive_{col}_after_event" not in out.columns

    def test_without_event_column_the_mean_covers_the_whole_movie(self):
        col, table = track_with_counts(event_time=1)
        out = mean_neighborhood_before_event(table, col, None, metrics=["inclusive"])
        assert out[f"mean_count_inclusive_{col}"].tolist() == [2.5] * 4
        assert "event_time_temp" not in out.columns
        out = mean_neighborhood_after_event(out, col, None, metrics=["inclusive"])
        assert out[f"mean_count_inclusive_{col}"].tolist() == [2.5] * 4
        assert "event_time_temp" not in out.columns
        assert not any(c.endswith(("_before_event", "_after_event")) for c in out.columns)


# ---------------------------------------------------------------------------
# contact_neighborhood / find_contact_neighbors
# ---------------------------------------------------------------------------


class TestContactNeighborhood:

    @staticmethod
    def three_cells():
        """Cells 1 and 2 touch; cell 3 sits 10 px away from both."""
        labels = np.zeros((30, 40), dtype=int)
        labels[5:15, 5:15] = 1
        labels[5:15, 15:25] = 2
        labels[5:15, 35:40] = 3
        return labels

    def test_self_contact_pairs_only_touching_cells(self):
        pairs = contact_neighborhood(self.three_cells(), border=0)
        np.testing.assert_array_equal(pairs, [[1, 2]])

    def test_two_population_pairs_are_signed_a_negative_b_positive(self):
        labels = self.three_cells()
        labelsA = np.where(labels == 1, 1, 0)
        labelsB = np.where(labels >= 2, labels - 1, 0)  # B cells 1 (touching) and 2
        pairs = contact_neighborhood(labelsA, labelsB=labelsB, border=0)
        np.testing.assert_array_equal(pairs, [[-1, 1]])

    def test_find_contact_neighbors_rejects_stacks(self):
        with pytest.raises(ValueError, match="dimension"):
            find_contact_neighbors(np.zeros((2, 5, 5), dtype=int))


# ---------------------------------------------------------------------------
# extract_neighborhood_in_pair_table
# ---------------------------------------------------------------------------


@pytest.fixture
def pair_table():
    status = "status_neighborhood_(targets-effectors)_circle_50_px"
    return pd.DataFrame(
        {
            "reference_population": ["targets", "targets", "targets", "effectors"],
            "neighbor_population": ["effectors", "effectors", "targets", "targets"],
            status: [1, 0, 1, 1],
            "status_neighborhood_self_circle_50_px": [1, 0, 1, 1],
        }
    )


class TestExtractNeighborhoodInPairTable:

    def test_contact_only_keeps_pairs_in_contact(self, pair_table):
        out = extract_neighborhood_in_pair_table(pair_table, distance=50, mode="circle")
        assert out.index.tolist() == [0]

    def test_without_contact_only_keeps_every_pair_of_that_neighborhood(self, pair_table):
        out = extract_neighborhood_in_pair_table(
            pair_table, distance=50, mode="circle", contact_only=False
        )
        assert out.index.tolist() == [0, 1]

    def test_neighbor_population_is_read_from_the_key(self, pair_table):
        out = extract_neighborhood_in_pair_table(
            pair_table,
            reference_population="targets",
            neighbor_population="ignored",
            neighborhood_key="status_neighborhood_(targets-effectors)_circle_50_px",
        )
        assert out.index.tolist() == [0]

    def test_self_key_pairs_a_population_with_itself(self, pair_table):
        out = extract_neighborhood_in_pair_table(
            pair_table,
            reference_population="targets",
            neighborhood_key="neighborhood_self_circle_50_px",
        )
        assert out.index.tolist() == [2]

    def test_invalid_mode_and_unknown_neighborhood_raise(self, pair_table):
        with pytest.raises(ValueError, match="mode"):
            extract_neighborhood_in_pair_table(pair_table, distance=50, mode="square")
        with pytest.raises(KeyError, match="does not appear"):
            extract_neighborhood_in_pair_table(pair_table, distance=99, mode="circle")


# ---------------------------------------------------------------------------
# Public API on a position folder
# ---------------------------------------------------------------------------


@pytest.fixture
def position(tmp_path):
    """A position with two frames. Target 1 (label 1) touches effector 7 (label 1)
    and is 5 px from it; target 2 (label 2) is far from everything."""
    pos = tmp_path / "W1" / "100"
    tables = pos / "output" / "tables"
    tables.mkdir(parents=True)

    pd.DataFrame(
        {
            "TRACK_ID": [1, 1, 2, 2],
            "FRAME": [0, 1, 0, 1],
            "POSITION_X": [10.0, 10.0, 100.0, 100.0],
            "POSITION_Y": [10.0, 10.0, 100.0, 100.0],
            "class_id": [1, 1, 2, 2],
            "t_event": [0, 0, -1, -1],
        }
    ).to_csv(tables / "trajectories_targets.csv", index=False)
    pd.DataFrame(
        {
            "TRACK_ID": [7, 7],
            "FRAME": [0, 1],
            "POSITION_X": [13.0, 13.0],
            "POSITION_Y": [14.0, 14.0],
            "class_id": [1, 1],
        }
    ).to_csv(tables / "trajectories_effectors.csv", index=False)

    (pos / "labels_targets").mkdir()
    (pos / "labels_effectors").mkdir()
    for t in range(2):
        lt = np.zeros((40, 40), np.uint16)
        lt[5:15, 5:15] = 1
        lt[25:35, 25:35] = 2
        le = np.zeros((40, 40), np.uint16)
        le[5:15, 15:25] = 1
        tifffile.imwrite(pos / "labels_targets" / f"{t:04d}.tif", lt)
        tifffile.imwrite(pos / "labels_effectors" / f"{t:04d}.tif", le)
    return pos


def read_tables(pos, population):
    tables = pos / "output" / "tables"
    csv = pd.read_csv(tables / f"trajectories_{population}.csv")
    pkl = pd.read_pickle(tables / f"trajectories_{population}.pkl")
    return csv, pkl


TWO_POP_20 = "neighborhood_(targets-effectors)_circle_20_px"


class TestComputeNeighborhoodAtPosition:

    def test_two_population_run_writes_counts_to_csv_and_lists_to_pickle(self, position):
        compute_neighborhood_at_position(
            str(position), 20, population=["targets", "effectors"],
            event_time_col="t_event", neighborhood_kwargs=neighborhood_kwargs(),
        )
        csv, pkl = read_tables(position, "targets")
        csv = csv.set_index(["TRACK_ID", "FRAME"])
        pkl = pkl.set_index(["TRACK_ID", "FRAME"])

        # the CSV holds the counts, renamed after the populations, but no neighbor lists
        assert not any(c.startswith("neighborhood_") for c in csv.columns)
        assert csv[f"inclusive_count_{TWO_POP_20}"].tolist() == [1, 1, 0, 0]
        assert csv[f"inclusive_count_s1_{TWO_POP_20}"].tolist() == [1, 1, 0, 0]
        assert csv[f"mean_count_inclusive_{TWO_POP_20}_before_event"].tolist() == [1, 1, 0, 0]
        np.testing.assert_array_equal(
            csv[f"mean_count_inclusive_{TWO_POP_20}_after_event"].to_numpy(),
            [1, 1, np.nan, np.nan],
        )
        # the pickle keeps the neighbor lists
        assert ids_of(pkl.loc[(1, 0), TWO_POP_20]) == [7]
        assert pkl.loc[(2, 0), TWO_POP_20] == []

        # the neighbor population's tables carry no neighborhood of their own
        eff_csv, eff_pkl = read_tables(position, "effectors")
        assert not any("neighborhood" in c for c in eff_csv.columns)
        assert not any("neighborhood" in c for c in eff_pkl.columns)

    def test_a_second_run_keeps_the_neighborhoods_of_the_first(self, position):
        """Without clear_neigh, neighborhoods from earlier runs are recovered from
        the pickle and kept next to the new one."""
        compute_neighborhood_at_position(
            str(position), 20, population=["targets", "effectors"],
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        compute_neighborhood_at_position(
            str(position), 40, population=["targets", "effectors"],
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        csv, pkl = read_tables(position, "targets")
        neigh_cols = sorted(c for c in pkl.columns if c.startswith("neighborhood_"))
        assert neigh_cols == [TWO_POP_20, "neighborhood_(targets-effectors)_circle_40_px"]
        assert len(pkl) == 4
        assert f"inclusive_count_{TWO_POP_20}" in csv.columns

    def test_clear_neigh_drops_previous_neighborhoods_and_the_pair_table(self, position):
        compute_neighborhood_at_position(
            str(position), 20, population=["targets", "effectors"],
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        pairs = position / "output" / "tables" / "trajectories_pairs.csv"
        pd.DataFrame({"REFERENCE_ID": [1]}).to_csv(pairs, index=False)

        compute_neighborhood_at_position(
            str(position), 40, population=["targets", "effectors"], clear_neigh=True,
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        csv, pkl = read_tables(position, "targets")
        assert [c for c in pkl.columns if c.startswith("neighborhood_")] == [
            "neighborhood_(targets-effectors)_circle_40_px"
        ]
        assert not any("_20_px" in c for c in csv.columns)
        assert not pairs.exists()

    def test_a_single_population_is_its_own_neighbor_population(self, position):
        tables = compute_neighborhood_at_position(
            str(position), 200, population="targets", return_tables=True,
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        df_A, _ = tables
        col = df_A.set_index(["TRACK_ID", "FRAME"])["neighborhood_self_circle_200_px"]
        assert ids_of(col.loc[(1, 0)]) == [2]
        assert ids_of(col.loc[(2, 1)]) == [1]
        csv, _ = read_tables(position, "targets")
        assert csv["inclusive_count_neighborhood_self_circle_200_px"].tolist() == [1] * 4

    def test_invalid_arguments_raise(self, position, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_neighborhood_at_position(str(tmp_path / "nowhere"), 20)
        with pytest.raises(ValueError, match="edge thresholds"):
            compute_neighborhood_at_position(
                str(position), [20, 40], theta_dist=[5],
                neighborhood_kwargs=neighborhood_kwargs(),
            )

    def test_missing_table_returns_none_without_writing(self, position):
        os.remove(position / "output" / "tables" / "trajectories_effectors.csv")
        assert (
            compute_neighborhood_at_position(
                str(position), 20, population=["targets", "effectors"],
                neighborhood_kwargs=neighborhood_kwargs(),
            )
            is None
        )
        assert not (position / "output" / "tables" / "trajectories_targets.pkl").exists()


TWO_POP_CONTACT = "neighborhood_(targets-effectors)_contact_3_px"


class TestComputeContactNeighborhoodAtPosition:

    def test_touching_masks_are_neighbors_and_counts_reach_the_csv(self, position):
        df_A, df_B = compute_contact_neighborhood_at_position(
            str(position), 3, population=["targets", "effectors"],
            neighborhood_kwargs=neighborhood_kwargs(),
        )
        col = df_A.set_index(["TRACK_ID", "FRAME"])[TWO_POP_CONTACT]
        assert ids_of(col.loc[(1, 0)]) == [7]
        assert ids_of(col.loc[(1, 1)]) == [7]
        assert col.loc[(2, 0)] == []
        assert not any("neighborhood_" in c for c in df_B.columns)

        csv, pkl = read_tables(position, "targets")
        csv = csv.set_index(["TRACK_ID", "FRAME"])
        assert csv[f"inclusive_count_{TWO_POP_CONTACT}"].tolist() == [1, 1, 0, 0]
        assert csv[f"mean_count_inclusive_{TWO_POP_CONTACT}"].tolist() == [1, 1, 0, 0]
        assert not any(c.startswith("neighborhood_") for c in csv.columns)
        assert TWO_POP_CONTACT in pkl.columns

    def test_contact_site_intensity_is_measured_from_the_movie(self, position):
        """With channel_names, the movie is loaded and the contact strip of target 1
        (rows 5-14, columns 12-14 for a 3 px border) is measured. The channel
        holds the column index, so mean = 13, max = 14, std = sqrt(2/3)."""
        (position / "movie").mkdir()
        movie = np.broadcast_to(np.arange(40, dtype=np.float32), (2, 1, 40, 40)).copy()
        tifffile.imwrite(
            position / "movie" / "Aligned_test.tif", movie, imagej=True,
            metadata={"axes": "TCYX"},
        )
        df_A, _ = compute_contact_neighborhood_at_position(
            str(position), 3, population=["targets", "effectors"],
            channel_names=["c0"], neighborhood_kwargs=neighborhood_kwargs(),
        )
        row = df_A.set_index(["TRACK_ID", "FRAME"]).loc[(1, 0)]
        assert row["contact_c0_mean"] == pytest.approx(13.0)
        assert row["contact_c0_max"] == pytest.approx(14.0)
        assert row["contact_c0_std"] == pytest.approx(np.sqrt(2 / 3))
        far = df_A.set_index(["TRACK_ID", "FRAME"]).loc[(2, 0)]
        assert np.isnan(far["contact_c0_mean"])

        csv, _ = read_tables(position, "targets")
        assert "contact_c0_mean" in csv.columns

    def test_missing_movie_skips_contact_intensity_but_keeps_the_neighborhood(self, position):
        df_A, _ = compute_contact_neighborhood_at_position(
            str(position), 3, population=["targets", "effectors"],
            channel_names=["c0"], neighborhood_kwargs=neighborhood_kwargs(),
        )
        assert not any(c.startswith("contact_") for c in df_A.columns)
        assert TWO_POP_CONTACT in df_A.columns
