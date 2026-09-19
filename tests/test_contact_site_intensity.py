"""
Tests for contact-site intensity measurement:
  - _contact_site_mask    (helper)
  - _measure_contact_intensity_at_t (per-frame writer)
  - mask_contact_neighborhood with intensity_images / channel_names
  - _measure_contact_site_intensity (pair-table helper)
"""

import numpy as np
import pandas as pd
import pytest

from celldetective.neighborhood import (
    _contact_site_mask,
    _measure_contact_intensity_at_t,
    mask_contact_neighborhood,
)
from celldetective.relative_measurements import _measure_contact_site_intensity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_adjacent_labels(size=60, gap=0):
    """Two cells side by side, optionally with a gap between them."""
    lA = np.zeros((size, size), dtype=int)
    lB = np.zeros((size, size), dtype=int)
    mid = size // 2
    lA[:, :mid - gap] = 1
    lB[:, mid + gap:] = 2
    return lA, lB


def make_intensity(size=60, n_channels=2, seed=0):
    """Random float32 intensity stack (H, W, C)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (size, size, n_channels)).astype(np.float32)


def make_cell_dfs(ids_A, ids_B, mask_ids_A, mask_ids_B, frame=0):
    """Minimal DataFrames for mask_contact_neighborhood."""
    n = len(ids_A)
    dfA = pd.DataFrame({
        "TRACK_ID": ids_A,
        "FRAME": [frame] * n,
        "POSITION_X": np.linspace(5, 25, n),
        "POSITION_Y": [30] * n,
        "class_id": mask_ids_A,
        "status_Live": [1] * n,
    })
    m = len(ids_B)
    dfB = pd.DataFrame({
        "TRACK_ID": ids_B,
        "FRAME": [frame] * m,
        "POSITION_X": np.linspace(35, 55, m),
        "POSITION_Y": [30] * m,
        "class_id": mask_ids_B,
        "status_Live": [1] * m,
    })
    return dfA, dfB


# ---------------------------------------------------------------------------
# _contact_site_mask
# ---------------------------------------------------------------------------

class TestContactSiteMask:

    def test_returns_pixels_inside_cell_A(self):
        lA, lB = make_adjacent_labels()
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        # every True pixel must belong to cell A
        assert np.all(lA[zone] == 1), "Contact zone contains pixels outside cell A"

    def test_zone_nonempty_for_adjacent_cells(self):
        lA, lB = make_adjacent_labels()
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.sum() > 0, "Contact zone should be non-empty for adjacent cells"

    def test_zone_empty_for_distant_cells(self):
        lA, lB = make_adjacent_labels(gap=20)
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.sum() == 0, "Contact zone should be empty when cells are far apart"

    def test_border_zero_gives_only_overlapping_pixels(self):
        lA, lB = make_adjacent_labels(gap=0)
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=1)
        # border=1 means at least the cells must share a border pixel
        # result is restricted to lA pixels within 1 px of lB
        assert np.all(lA[zone] == 1)

    def test_self_contact_mode(self):
        """Pass labelsB=None: both cells live in labelsA."""
        lAB = np.zeros((60, 60), dtype=int)
        lAB[:, :30] = 1
        lAB[:, 30:] = 3
        zone = _contact_site_mask(lAB, None, mask_id_A=1, mask_id_B=3, border=3)
        assert zone.sum() > 0
        assert np.all(lAB[zone] == 1)

    def test_larger_border_gives_larger_zone(self):
        lA, lB = make_adjacent_labels()
        z3 = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        z8 = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=8)
        assert z8.sum() >= z3.sum(), "Larger border should give equal or larger contact zone"

    def test_output_shape_matches_input(self):
        lA, lB = make_adjacent_labels(size=40)
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.shape == lA.shape

    def test_zone_is_boolean(self):
        lA, lB = make_adjacent_labels()
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.dtype == bool


# ---------------------------------------------------------------------------
# _measure_contact_intensity_at_t
# ---------------------------------------------------------------------------

class TestMeasureContactIntensityAtT:

    def setup_method(self):
        self.size = 60
        self.lA, self.lB = make_adjacent_labels(self.size)
        self.intensity = make_intensity(self.size, n_channels=2)
        # Make channel 0 bright on the left (cell A side)
        self.intensity[:, :self.size // 2, 0] = 0.9
        self.intensity[:, self.size // 2:, 1] = 0.8

        col = {
            "track": "TRACK_ID", "time": "FRAME",
            "x": "POSITION_X", "y": "POSITION_Y", "mask_id": "class_id",
        }
        self.cl = col
        self.dfA, self.dfB = make_cell_dfs([1], [2], [1], [2], frame=0)
        # Initialize contact intensity columns
        for ch in ("ch0", "ch1"):
            for stat in ("mean", "max", "std"):
                self.dfA[f"contact_{ch}_{stat}"] = np.nan

    def _build_dist_map(self):
        """Fake dist_map: cell A (row 0) is close to cell B (col 0)."""
        dist = np.array([[5.0]])  # within contact threshold
        return dist

    def test_columns_written_for_contact_pair(self):
        dist_map = self._build_dist_map()
        _measure_contact_intensity_at_t(
            0, self.dfA, self.dfA, self.dfB,
            self.lA, self.lB,
            self.intensity, ["ch0", "ch1"],
            dist_map, border=3,
            column_labelsA=self.cl, column_labelsB=self.cl,
        )
        for ch in ("ch0", "ch1"):
            for stat in ("mean", "max", "std"):
                val = self.dfA.iloc[0][f"contact_{ch}_{stat}"]
                assert np.isfinite(val), f"contact_{ch}_{stat} should be finite after contact"

    def test_no_contact_leaves_nan(self):
        # dist_map = 1e6 means no contact
        dist_map = np.array([[1.0e6]])
        _measure_contact_intensity_at_t(
            0, self.dfA, self.dfA, self.dfB,
            self.lA, self.lB,
            self.intensity, ["ch0", "ch1"],
            dist_map, border=3,
            column_labelsA=self.cl, column_labelsB=self.cl,
        )
        for ch in ("ch0", "ch1"):
            for stat in ("mean", "max", "std"):
                assert np.isnan(self.dfA.iloc[0][f"contact_{ch}_{stat}"])

    def test_mean_bounded_by_min_max(self):
        dist_map = self._build_dist_map()
        _measure_contact_intensity_at_t(
            0, self.dfA, self.dfA, self.dfB,
            self.lA, self.lB,
            self.intensity, ["ch0"],
            dist_map, border=5,
            column_labelsA=self.cl, column_labelsB=self.cl,
        )
        mean_val = self.dfA.iloc[0]["contact_ch0_mean"]
        max_val  = self.dfA.iloc[0]["contact_ch0_max"]
        min_val  = 0.0  # minimum possible
        assert mean_val <= max_val
        assert mean_val >= min_val

    def test_std_nonnegative(self):
        dist_map = self._build_dist_map()
        _measure_contact_intensity_at_t(
            0, self.dfA, self.dfA, self.dfB,
            self.lA, self.lB,
            self.intensity, ["ch0"],
            dist_map, border=5,
            column_labelsA=self.cl, column_labelsB=self.cl,
        )
        assert self.dfA.iloc[0]["contact_ch0_std"] >= 0


# ---------------------------------------------------------------------------
# mask_contact_neighborhood integration
# ---------------------------------------------------------------------------

class TestMaskContactNeighborhoodWithIntensity:

    def setup_method(self):
        self.size = 60
        self.n_frames = 3
        self.channel_names = ["DAPI", "GFP"]

        lA, lB = make_adjacent_labels(self.size)
        self.labelsA = [lA] * self.n_frames
        self.labelsB = [lB] * self.n_frames

        rng = np.random.default_rng(10)
        frame_img = rng.uniform(0, 1, (self.size, self.size, 2)).astype(np.float32)
        self.intensity_images = [frame_img] * self.n_frames

        # Build multi-frame DataFrames
        rows_A, rows_B = [], []
        for t in range(self.n_frames):
            rows_A.append({"TRACK_ID": 1, "FRAME": t, "POSITION_X": 15.0,
                           "POSITION_Y": 30.0, "class_id": 1,
                           "status_Live": 1})
            rows_B.append({"TRACK_ID": 2, "FRAME": t, "POSITION_X": 45.0,
                           "POSITION_Y": 30.0, "class_id": 2,
                           "status_Live": 1})
        self.dfA = pd.DataFrame(rows_A)
        self.dfB = pd.DataFrame(rows_B)

    def test_contact_intensity_columns_created(self):
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=self.intensity_images,
            channel_names=self.channel_names,
        )
        for ch in self.channel_names:
            for stat in ("mean", "max", "std"):
                col = f"contact_{ch}_{stat}"
                assert col in dfA_out.columns, f"Missing column: {col}"

    def test_contact_intensity_values_finite(self):
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=self.intensity_images,
            channel_names=self.channel_names,
        )
        for ch in self.channel_names:
            col = f"contact_{ch}_mean"
            non_nan = dfA_out[col].dropna()
            assert len(non_nan) > 0, f"Expected at least some finite values in {col}"
            assert np.all(np.isfinite(non_nan)), f"Non-finite values found in {col}"

    def test_no_intensity_images_no_contact_columns(self):
        """Without intensity_images, no contact_* columns should appear."""
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=None,
            channel_names=None,
        )
        contact_cols = [c for c in dfA_out.columns if c.startswith("contact_")]
        assert len(contact_cols) == 0

    def test_neighborhood_column_still_populated(self):
        """Contact neighborhood column must still be populated alongside intensity."""
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=self.intensity_images,
            channel_names=self.channel_names,
        )
        neigh_cols = [c for c in dfA_out.columns if c.startswith("neighborhood")]
        assert len(neigh_cols) > 0, "Neighborhood column missing"

    def test_mean_bounded_by_image_range(self):
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=self.intensity_images,
            channel_names=self.channel_names,
        )
        for ch in self.channel_names:
            vals = dfA_out[f"contact_{ch}_mean"].dropna()
            assert np.all(vals >= 0.0)
            assert np.all(vals <= 1.0)


# ---------------------------------------------------------------------------
# _measure_contact_site_intensity (pair-table helper)
# ---------------------------------------------------------------------------

class TestMeasureContactSiteIntensity:
    """Tests for the pair-table contact intensity helper in relative_measurements."""

    def setup_method(self):
        size = 60
        rng = np.random.default_rng(7)
        lA, lB = make_adjacent_labels(size)
        self.lA = lA
        self.lB = lB
        # Two channels: first channel bright on left half, second on right
        self.img = np.stack(
            [rng.uniform(0.8, 1.0, (size, size)),  # bright everywhere
             rng.uniform(0.0, 0.2, (size, size))],  # dim everywhere
            axis=-1,
        ).astype(np.float32)
        self.channel_names = ["bright", "dim"]

    def test_returns_all_stat_keys(self):
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=3
        )
        for ch in self.channel_names:
            for stat in ("mean", "max", "std"):
                assert f"contact_{ch}_{stat}" in result, f"Missing key contact_{ch}_{stat}"

    def test_adjacent_cells_give_finite_values(self):
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=3
        )
        for ch in self.channel_names:
            assert np.isfinite(result[f"contact_{ch}_mean"])
            assert np.isfinite(result[f"contact_{ch}_max"])
            assert np.isfinite(result[f"contact_{ch}_std"])

    def test_distant_cells_give_nan(self):
        lA, lB = make_adjacent_labels(60, gap=20)  # 20-px gap
        result = _measure_contact_site_intensity(
            lA, lB, 1, 2, self.img, self.channel_names, border=3
        )
        for ch in self.channel_names:
            assert np.isnan(result[f"contact_{ch}_mean"])

    def test_mean_leq_max(self):
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=5
        )
        for ch in self.channel_names:
            assert result[f"contact_{ch}_mean"] <= result[f"contact_{ch}_max"]

    def test_std_nonnegative(self):
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=5
        )
        for ch in self.channel_names:
            assert result[f"contact_{ch}_std"] >= 0.0

    def test_none_labelsB_uses_labelsA(self):
        """Self-contact: labelsB=None should fall back to labelsA without error."""
        result = _measure_contact_site_intensity(
            self.lA, None, 1, 2, self.img, self.channel_names, border=3
        )
        # With the same label image for both, result may be NaN (cell 2 not in lA),
        # but the function must not raise
        assert isinstance(result, dict)
        assert len(result) == len(self.channel_names) * 3

    def test_bright_channel_mean_higher(self):
        """The 'bright' channel contact mean should exceed the 'dim' channel mean."""
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=5
        )
        assert result["contact_bright_mean"] > result["contact_dim_mean"]
