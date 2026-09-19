"""
Tests for the new extra_properties functions:
  - circularity
  - aspect_ratio
  - intensity_skewness
  - intensity_kurtosis
  - intensity_membrane_cytoplasm_ratio

Covers unit behaviour and integration through measure_features.
"""

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from celldetective.extra_properties import (
    aspect_ratio,
    circularity,
    intensity_kurtosis,
    intensity_membrane_cytoplasm_ratio,
    intensity_skewness,
)
from celldetective.measure import measure_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circle_mask(radius=20, size=60):
    """Boolean circular mask centred in a square canvas."""
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def rect_mask(h=10, w=30, canvas=60):
    """Boolean rectangular mask — strongly elongated."""
    m = np.zeros((canvas, canvas), dtype=bool)
    r0 = canvas // 2 - h // 2
    c0 = canvas // 2 - w // 2
    m[r0:r0 + h, c0:c0 + w] = True
    return m


# ---------------------------------------------------------------------------
# circularity
# ---------------------------------------------------------------------------

class TestCircularity:

    def test_circle_close_to_one(self):
        mask = circle_mask(radius=20, size=60)
        result = circularity(mask)
        assert 0.85 <= result <= 1.05, f"Expected ~1 for circle, got {result:.4f}"

    def test_rectangle_lower_than_circle(self):
        c_mask = circle_mask(radius=15, size=60)
        r_mask = rect_mask(h=5, w=50, canvas=60)
        c_val = circularity(c_mask)
        r_val = circularity(r_mask)
        assert r_val < c_val, "Rectangle should have lower circularity than circle"

    def test_value_in_valid_range(self):
        mask = circle_mask()
        val = circularity(mask)
        assert 0.0 < val <= 1.05

    def test_empty_mask_returns_nan(self):
        mask = np.zeros((20, 20), dtype=bool)
        val = circularity(mask)
        assert np.isnan(val)

    def test_shape_only_signature_callable(self):
        """circularity takes one positional arg (regionmask only)."""
        import inspect
        sig = inspect.signature(circularity)
        n_required = sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        assert n_required == 1


# ---------------------------------------------------------------------------
# aspect_ratio
# ---------------------------------------------------------------------------

class TestAspectRatio:

    def test_circle_close_to_one(self):
        mask = circle_mask(radius=20, size=60)
        val = aspect_ratio(mask)
        assert 0.95 <= val <= 1.1, f"Expected ~1 for circle, got {val:.4f}"

    def test_elongated_greater_than_one(self):
        mask = rect_mask(h=5, w=40, canvas=60)
        val = aspect_ratio(mask)
        assert val > 3.0, f"Expected aspect ratio >> 1 for elongated rect, got {val:.4f}"

    def test_always_geq_one(self):
        for r in [5, 10, 20]:
            mask = circle_mask(radius=r, size=60)
            val = aspect_ratio(mask)
            assert val >= 1.0

    def test_empty_mask_returns_nan(self):
        mask = np.zeros((20, 20), dtype=bool)
        val = aspect_ratio(mask)
        assert np.isnan(val)

    def test_shape_only_signature_callable(self):
        """aspect_ratio takes one positional arg (regionmask only)."""
        import inspect
        sig = inspect.signature(aspect_ratio)
        n_required = sum(
            1 for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
        assert n_required == 1


# ---------------------------------------------------------------------------
# intensity_skewness
# ---------------------------------------------------------------------------

class TestIntensitySkewness:

    def test_symmetric_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        mask = circle_mask()
        img = np.zeros(mask.shape)
        img[mask] = rng.normal(0.5, 0.1, mask.sum())
        val = intensity_skewness(mask, img)
        assert abs(val) < 0.5, f"Symmetric distribution skewness should be ~0, got {val}"

    def test_positive_skew_for_right_heavy_tail(self):
        mask = circle_mask()
        img = np.zeros(mask.shape)
        # Most pixels near 0, a few very bright ones
        img[mask] = 0.01
        ys, xs = np.where(mask)
        img[ys[:5], xs[:5]] = 10.0
        val = intensity_skewness(mask, img)
        assert val > 1.0, f"Expected positive skew, got {val}"

    def test_too_few_pixels_returns_nan(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        mask[5, 6] = True  # only 2 pixels
        img = np.ones((10, 10))
        val = intensity_skewness(mask, img)
        assert np.isnan(val)

    def test_uniform_image_skewness_near_zero(self):
        mask = circle_mask()
        img = np.ones(mask.shape) * 0.5
        val = intensity_skewness(mask, img)
        # scipy.stats.skew of constant array → 0 (or NaN with bias correction for small n)
        assert np.isnan(val) or np.isclose(val, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# intensity_kurtosis
# ---------------------------------------------------------------------------

class TestIntensityKurtosis:

    def test_uniform_distribution_negative_kurtosis(self):
        rng = np.random.default_rng(1)
        mask = circle_mask(radius=25, size=60)
        img = np.zeros(mask.shape)
        img[mask] = rng.uniform(0, 1, mask.sum())
        val = intensity_kurtosis(mask, img)
        # Uniform distribution excess kurtosis ≈ -1.2
        assert val < 0, f"Uniform distribution should have negative excess kurtosis, got {val}"

    def test_normal_distribution_near_zero(self):
        rng = np.random.default_rng(2)
        mask = circle_mask(radius=25, size=60)
        img = np.zeros(mask.shape)
        img[mask] = rng.normal(0.5, 0.1, mask.sum())
        val = intensity_kurtosis(mask, img)
        assert -1.5 < val < 1.5, f"Normal kurtosis should be near 0 (Fisher), got {val}"

    def test_too_few_pixels_returns_nan(self):
        mask = np.zeros((10, 10), dtype=bool)
        for i in range(3):
            mask[i, 0] = True  # 3 pixels — below the 4-pixel threshold
        img = np.ones((10, 10))
        val = intensity_kurtosis(mask, img)
        assert np.isnan(val)

    def test_leptokurtic_distribution_positive(self):
        rng = np.random.default_rng(3)
        mask = circle_mask(radius=25, size=60)
        img = np.zeros(mask.shape)
        # Laplace distribution has excess kurtosis = 3
        img[mask] = rng.laplace(0, 1, mask.sum())
        val = intensity_kurtosis(mask, img)
        assert val > 0, f"Laplace distribution should have positive excess kurtosis, got {val}"


# ---------------------------------------------------------------------------
# intensity_membrane_cytoplasm_ratio
# ---------------------------------------------------------------------------

class TestIntensityMembraneCytoplasmRatio:

    def test_membrane_enriched_returns_greater_than_one(self):
        mask = circle_mask(radius=20, size=60)
        img = np.zeros(mask.shape)
        core = binary_erosion(mask, iterations=3)
        # membrane ring bright, cytoplasm dark
        img[mask & ~core] = 1.0
        img[core] = 0.1
        val = intensity_membrane_cytoplasm_ratio(mask, img)
        assert val > 1.0, f"Membrane-enriched should give ratio > 1, got {val}"

    def test_cytoplasm_enriched_returns_less_than_one(self):
        mask = circle_mask(radius=20, size=60)
        img = np.zeros(mask.shape)
        core = binary_erosion(mask, iterations=3)
        # cytoplasm bright, membrane dark
        img[core] = 1.0
        img[mask & ~core] = 0.1
        val = intensity_membrane_cytoplasm_ratio(mask, img)
        assert val < 1.0, f"Cytoplasm-enriched should give ratio < 1, got {val}"

    def test_uniform_returns_one(self):
        mask = circle_mask(radius=20, size=60)
        img = np.ones(mask.shape) * 0.5
        val = intensity_membrane_cytoplasm_ratio(mask, img)
        assert np.isclose(val, 1.0, atol=0.01), f"Uniform image should give ratio ~1, got {val}"

    def test_too_small_mask_returns_nan(self):
        # Mask too small to survive erosion by 3 px
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True  # 2×2 pixels, erodes to nothing
        img = np.ones((10, 10))
        val = intensity_membrane_cytoplasm_ratio(mask, img)
        assert np.isnan(val)

    def test_zero_cytoplasm_returns_nan(self):
        mask = circle_mask(radius=20, size=60)
        core = binary_erosion(mask, iterations=3)
        img = np.zeros(mask.shape)
        # cytoplasm mean = 0, membrane = 1
        img[mask & ~core] = 1.0
        val = intensity_membrane_cytoplasm_ratio(mask, img)
        assert np.isnan(val)


# ---------------------------------------------------------------------------
# Integration through measure_features
# ---------------------------------------------------------------------------

class TestMeasureFeaturesIntegration:
    """Verifies column names produced by measure_features for each new function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        size = 80
        mask = circle_mask(radius=25, size=size)
        self.label = mask.astype(np.int32)
        rng = np.random.default_rng(42)
        ch = rng.uniform(0, 1, (size, size)).astype(np.float32)
        self.img = ch[:, :, np.newaxis]  # (H, W, 1)
        self.channel = "mychan"

    def _run(self, feature_name):
        return measure_features(
            self.img, self.label,
            features=[feature_name],
            channels=[self.channel],
        )

    def test_circularity_column(self):
        df = self._run("circularity")
        assert "circularity" in df.columns
        assert df["circularity"].iloc[0] > 0

    def test_aspect_ratio_column(self):
        df = self._run("aspect_ratio")
        assert "aspect_ratio" in df.columns
        assert df["aspect_ratio"].iloc[0] >= 1.0

    def test_intensity_skewness_column(self):
        df = self._run("intensity_skewness")
        col = f"{self.channel}_skewness"
        assert col in df.columns, f"Expected '{col}', got {list(df.columns)}"
        assert np.isfinite(df[col].iloc[0])

    def test_intensity_kurtosis_column(self):
        df = self._run("intensity_kurtosis")
        col = f"{self.channel}_kurtosis"
        assert col in df.columns, f"Expected '{col}', got {list(df.columns)}"
        assert np.isfinite(df[col].iloc[0])

    def test_intensity_membrane_cytoplasm_ratio_column(self):
        df = self._run("intensity_membrane_cytoplasm_ratio")
        col = f"{self.channel}_membrane_cytoplasm_ratio"
        assert col in df.columns, f"Expected '{col}', got {list(df.columns)}"
        assert np.isfinite(df[col].iloc[0])

    def test_multiple_new_features_together(self):
        df = measure_features(
            self.img, self.label,
            features=["circularity", "aspect_ratio",
                      "intensity_skewness", "intensity_kurtosis",
                      "intensity_membrane_cytoplasm_ratio"],
            channels=[self.channel],
        )
        expected = [
            "circularity",
            "aspect_ratio",
            f"{self.channel}_skewness",
            f"{self.channel}_kurtosis",
            f"{self.channel}_membrane_cytoplasm_ratio",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column '{col}'"
        assert len(df) == 1
