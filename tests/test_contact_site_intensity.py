"""
Tests for contact-site intensity measurement:
  - _contact_site_mask    (helper)
  - _measure_contact_intensity_at_t (per-frame writer)
  - mask_contact_neighborhood with intensity_images / channel_names
  - _measure_contact_site_intensity (pair-table helper)

Geometry used throughout: cell A fills the left half of a 60x60 image
(columns 0-29), cell B the right half (columns 30-59). Dilating B by
``border`` pixels (cross-shaped structuring element) reaches columns
``30 - border`` to 29 of A, so the contact zone is exactly that vertical strip.
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


STATS = ("mean", "max", "std")


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


def expected_strip(size, border):
    """The contact zone of A facing B for two adjacent half-image cells."""
    zone = np.zeros((size, size), dtype=bool)
    zone[:, size // 2 - border:size // 2] = True
    return zone


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


def assert_stats_match_zone(values, image, zone, channel_names):
    """``values[contact_{ch}_{stat}]`` equals the stat of ``image`` over ``zone``."""
    for ch_idx, ch in enumerate(channel_names):
        pixels = image[..., ch_idx][zone].astype(float)
        np.testing.assert_allclose(values[f"contact_{ch}_mean"], pixels.mean(), rtol=1e-6)
        np.testing.assert_allclose(values[f"contact_{ch}_max"], pixels.max(), rtol=1e-6)
        np.testing.assert_allclose(values[f"contact_{ch}_std"], pixels.std(), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# _contact_site_mask
# ---------------------------------------------------------------------------

class TestContactSiteMask:

    @pytest.mark.parametrize("border", [1, 3, 8], ids=["border1", "border3", "border8"])
    def test_zone_is_the_strip_of_A_within_border_of_B(self, border):
        """The zone is boolean, shaped like the input, inside cell A, non-empty
        for adjacent cells, and exactly ``border`` columns wide (so a larger
        border always gives a larger zone; border=1 keeps only the boundary column)."""
        lA, lB = make_adjacent_labels()
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=border)

        assert zone.dtype == bool
        assert zone.shape == lA.shape
        assert np.all(lA[zone] == 1), "Contact zone contains pixels outside cell A"
        assert zone.sum() > 0, "Contact zone should be non-empty for adjacent cells"
        np.testing.assert_array_equal(zone, expected_strip(60, border))

    def test_output_shape_follows_a_non_default_image_size(self):
        lA, lB = make_adjacent_labels(size=40)
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.shape == lA.shape
        np.testing.assert_array_equal(zone, expected_strip(40, 3))

    def test_zone_empty_for_distant_cells(self):
        lA, lB = make_adjacent_labels(gap=20)
        zone = _contact_site_mask(lA, lB, mask_id_A=1, mask_id_B=2, border=3)
        assert zone.sum() == 0, "Contact zone should be empty when cells are far apart"

    def test_self_contact_mode(self):
        """Pass labelsB=None: both cells live in labelsA."""
        lAB = np.zeros((60, 60), dtype=int)
        lAB[:, :30] = 1
        lAB[:, 30:] = 3
        zone = _contact_site_mask(lAB, None, mask_id_A=1, mask_id_B=3, border=3)
        assert zone.sum() > 0
        assert np.all(lAB[zone] == 1)
        np.testing.assert_array_equal(zone, expected_strip(60, 3))


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
            for stat in STATS:
                self.dfA[f"contact_{ch}_{stat}"] = np.nan

    def _measure(self, dist_map, border, channels=("ch0", "ch1")):
        _measure_contact_intensity_at_t(
            0, self.dfA, self.dfA, self.dfB,
            self.lA, self.lB,
            self.intensity, list(channels),
            dist_map, border=border,
            column_labelsA=self.cl, column_labelsB=self.cl,
        )
        return self.dfA.iloc[0]

    @pytest.mark.parametrize("border", [3, 5], ids=["border3", "border5"])
    def test_contact_pair_gets_the_stats_of_the_contact_zone(self, border):
        """A pair in contact (distance 5 < 1e5) gets, for every channel, the
        mean/max/std of the pixels in the contact strip: finite, mean between
        0 and max, std >= 0, and exactly 0.9/0.9/0 on the uniform channel 0."""
        row = self._measure(np.array([[5.0]]), border)

        for ch in ("ch0", "ch1"):
            for stat in STATS:
                assert np.isfinite(row[f"contact_{ch}_{stat}"]), f"contact_{ch}_{stat} should be finite after contact"
            assert 0.0 <= row[f"contact_{ch}_mean"] <= row[f"contact_{ch}_max"]
            assert row[f"contact_{ch}_std"] >= 0

        np.testing.assert_allclose(row["contact_ch0_mean"], 0.9, rtol=1e-6)
        np.testing.assert_allclose(row["contact_ch0_max"], 0.9, rtol=1e-6)
        assert row["contact_ch0_std"] == pytest.approx(0.0, abs=1e-7)
        assert_stats_match_zone(row, self.intensity, expected_strip(self.size, border), ["ch0", "ch1"])

    def test_no_contact_leaves_nan(self):
        # dist_map = 1e6 means no contact
        row = self._measure(np.array([[1.0e6]]), border=3)
        for ch in ("ch0", "ch1"):
            for stat in STATS:
                assert np.isnan(row[f"contact_{ch}_{stat}"])


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

    def test_contact_intensity_written_alongside_the_neighborhood(self):
        """With intensity images, every contact_{ch}_{stat} column is created,
        holds finite values in [0, 1] equal to the stats of the 5-px contact
        strip, and the contact neighborhood column is still populated."""
        dfA_out, _ = mask_contact_neighborhood(
            self.dfA.copy(), self.dfB.copy(),
            self.labelsA, self.labelsB,
            distance=5,
            intensity_images=self.intensity_images,
            channel_names=self.channel_names,
        )
        for ch in self.channel_names:
            for stat in STATS:
                col = f"contact_{ch}_{stat}"
                assert col in dfA_out.columns, f"Missing column: {col}"
            non_nan = dfA_out[f"contact_{ch}_mean"].dropna()
            assert len(non_nan) > 0, f"Expected at least some finite values in contact_{ch}_mean"
            assert np.all(np.isfinite(non_nan)), f"Non-finite values found in contact_{ch}_mean"
            assert np.all(non_nan >= 0.0)
            assert np.all(non_nan <= 1.0)

        zone = expected_strip(self.size, 5)
        for _, row in dfA_out.iterrows():
            assert_stats_match_zone(row, self.intensity_images[0], zone, self.channel_names)

        neigh_cols = [c for c in dfA_out.columns if c.startswith("neighborhood")]
        assert len(neigh_cols) > 0, "Neighborhood column missing"
        for neighs in dfA_out["neighborhood_2_contact_5_px"]:
            assert [n["id"] for n in neighs] == [2]

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
        # Two channels: first bright everywhere, second dim everywhere
        self.img = np.stack(
            [rng.uniform(0.8, 1.0, (size, size)),  # bright everywhere
             rng.uniform(0.0, 0.2, (size, size))],  # dim everywhere
            axis=-1,
        ).astype(np.float32)
        self.channel_names = ["bright", "dim"]

    @pytest.mark.parametrize("border", [3, 5], ids=["border3", "border5"])
    def test_adjacent_cells_give_the_stats_of_the_contact_zone(self, border):
        """Every contact_{ch}_{stat} key is returned with a finite value,
        mean <= max, std >= 0, the bright channel above the dim one, and the
        values equal the stats of the contact strip."""
        result = _measure_contact_site_intensity(
            self.lA, self.lB, 1, 2, self.img, self.channel_names, border=border
        )
        for ch in self.channel_names:
            for stat in STATS:
                key = f"contact_{ch}_{stat}"
                assert key in result, f"Missing key {key}"
                assert np.isfinite(result[key])
            assert result[f"contact_{ch}_mean"] <= result[f"contact_{ch}_max"]
            assert result[f"contact_{ch}_std"] >= 0.0
        assert result["contact_bright_mean"] > result["contact_dim_mean"]
        assert_stats_match_zone(result, self.img, expected_strip(60, border), self.channel_names)

    def test_distant_cells_give_nan(self):
        lA, lB = make_adjacent_labels(60, gap=20)  # 20-px gap
        result = _measure_contact_site_intensity(
            lA, lB, 1, 2, self.img, self.channel_names, border=3
        )
        for ch in self.channel_names:
            assert np.isnan(result[f"contact_{ch}_mean"])

    def test_none_labelsB_uses_labelsA(self):
        """Self-contact: labelsB=None falls back to labelsA. With both cells in
        one image the result equals passing that image twice; with the partner
        absent from labelsA, every stat is NaN and nothing raises."""
        lAB = self.lA + self.lB
        self_result = _measure_contact_site_intensity(
            lAB, None, 1, 2, self.img, self.channel_names, border=3
        )
        explicit = _measure_contact_site_intensity(
            lAB, lAB, 1, 2, self.img, self.channel_names, border=3
        )
        assert self_result == pytest.approx(explicit)
        assert np.isfinite(self_result["contact_bright_mean"])

        result = _measure_contact_site_intensity(
            self.lA, None, 1, 2, self.img, self.channel_names, border=3
        )
        assert isinstance(result, dict)
        assert len(result) == len(self.channel_names) * 3
        assert all(np.isnan(v) for v in result.values())
