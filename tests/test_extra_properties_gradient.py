import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
from celldetective.extra_properties import intensity_radial_gradient
from celldetective.measure import measure_features
import pandas as pd


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def test_intensity_radial_gradient_positive_slope():
    """
    Test that intensity_radial_gradient correctly identifies a positive slope
    (intensity increases towards the edge).
    """
    # Create valid mask
    mask = create_circular_mask(100, 100, radius=40)

    # Distance transform: 0 at edge, max at center
    dist = distance_transform_edt(mask)
    if dist.max() == 0:
        pytest.fail("Mask is empty or invalid")

    max_dist = dist.max()

    # Construct intensity that increases towards the edge
    # dist_from_center ~ (max_dist - dist)
    # We want intensity ~ dist_from_center
    # expected x in polyfit is (max_dist - dist)
    # so if y = (max_dist - dist), slope should be 1.0
    intensity = (max_dist - dist) * mask

    slope, intercept, r2 = intensity_radial_gradient(mask, intensity)

    # We expect positive slope close to 1
    assert slope > 0.9, f"Slope {slope} should be close to 1 (positive gradient)"
    assert r2 > 0.9, f"R2 {r2} should be high for perfect linear gradient"


def test_intensity_radial_gradient_negative_slope():
    """
    Test that intensity_radial_gradient correctly identifies a negative slope
    (intensity decreases towards the edge, i.e., brighter at center).
    """
    mask = create_circular_mask(100, 100, radius=40)
    dist = distance_transform_edt(mask)

    # Construct intensity that is higher at center
    # y = dist
    # x = max_dist - dist
    # y = max_dist - x
    # slope should be -1
    intensity = dist * mask

    slope, intercept, r2 = intensity_radial_gradient(mask, intensity)

    # We expect negative slope close to -1
    assert slope < -0.9, f"Slope {slope} should be close to -1 (negative gradient)"
    assert r2 > 0.9, f"R2 {r2} should be high for perfect linear gradient"


def test_intensity_radial_gradient_flat():
    """
    Test that intensity_radial_gradient identifies a flat intensity profile (slope ~ 0).
    """
    mask = create_circular_mask(100, 100, radius=40)

    # Constant intensity
    intensity = np.ones_like(mask, dtype=float) * 0.5
    intensity = intensity * mask

    slope, intercept, r2 = intensity_radial_gradient(mask, intensity)

    assert np.isclose(
        slope, 0, atol=1e-10
    ), f"Slope {slope} should be approx 0 for flat intensity"


def test_intensity_radial_gradient_nan_handling():
    """
    Test that NaNs in intensity image are handled (interpolated) and don't crash.
    """
    mask = create_circular_mask(50, 50, radius=20)
    dist = distance_transform_edt(mask)
    intensity = dist * mask

    # Introduce some NaNs inside the mask
    intensity_with_nans = intensity.copy()
    intensity_with_nans[25, 25] = np.nan  # Center pixel
    intensity_with_nans[25, 26] = np.nan

    slope, intercept, r2 = intensity_radial_gradient(mask, intensity_with_nans)

    # Should still give a valid result similar to clean version (negative slope)
    assert not np.isnan(slope)
    assert slope < -0.5


def test_intensity_radial_gradient_integration():
    """
    Test that intensity_radial_gradient works when called through measure_features.
    """
    # Create synthetic image and label
    mask = create_circular_mask(100, 100, radius=40)
    label = mask.astype(int)

    # Positive gradient (brighter at edge)
    dist = distance_transform_edt(mask)
    max_dist = dist.max()
    intensity = (max_dist - dist) * mask

    # Create image with 1 channel
    img = intensity[:, :, np.newaxis]

    # measure_features expects a list of feature names
    # "intensity_radial_gradient" should be in extra_properties and picked up dynamically
    features = ["intensity_radial_gradient", "area"]
    channels = ["ch0"]

    df = measure_features(img, label, features=features, channels=channels)

    # Check if the columns exist
    # observed renaming:
    # intensity_radial_gradient-0 -> ch0_radial_gradient (slope)
    # intensity_radial_gradient-1 -> ch0_radial_intercept_1 (intercept)
    # intensity_radial_gradient-2 -> ch0_radial_gradient_r2_score_2 (r2)

    slope_col = "ch0_radial_gradient"
    r2_col = "ch0_radial_gradient_r2_score_2"

    expected_cols = [slope_col, "ch0_radial_intercept_1", r2_col]

    for col in expected_cols:
        assert (
            col in df.columns
        ), f"Column {col} missing from output DataFrame. Columns found: {df.columns.tolist()}"

    # Check values
    slope = df[slope_col].iloc[0]
    r2 = df[r2_col].iloc[0]

    assert (
        slope > 0.9
    ), f"Integration test slope {slope} should be positive and close to 1"
    assert r2 > 0.9, f"Integration test R2 {r2} should be high"
