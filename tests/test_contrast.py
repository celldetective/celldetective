"""
Display contrast leaves out the frame border.

Registration shifts the frames and fills the uncovered border with zeros, which
would otherwise set the lower contrast bound of every viewer.
"""

import numpy as np
import pytest

from celldetective.utils.contrast import (
    contrast_limits,
    image_interior,
    interior_slices,
    value_range,
)
from celldetective.utils.experiment import _get_contrast_limits


def _registered_frame(shape=(100, 100), seed=0):
    frame = np.random.default_rng(seed).normal(100.0, 5.0, shape)
    frame[:4, :] = 0
    frame[:, -3:] = 0
    return frame


class TestImageInterior:

    def test_crops_five_percent_on_each_side(self):
        image = np.zeros((100, 200))
        assert image_interior(image).shape == (90, 180)

    def test_crops_only_the_spatial_axes_of_a_stack(self):
        stack = np.zeros((7, 100, 40))
        assert image_interior(stack).shape == (7, 90, 36)
        assert image_interior(np.zeros((5, 100, 40, 3)), axes=(-3, -2)).shape == (
            5,
            90,
            36,
            3,
        )

    def test_slices_match_the_interior(self):
        image = np.arange(100 * 60).reshape(100, 60)
        np.testing.assert_array_equal(image[interior_slices(image.shape)], image_interior(image))

    def test_leaves_a_small_image_whole(self):
        image = np.arange(81).reshape(9, 9)
        np.testing.assert_array_equal(image_interior(image), image)


class TestContrastLimits:

    def test_the_zero_border_sets_neither_the_range_nor_the_limits(self):
        (vmin, _), (low, high) = contrast_limits(_registered_frame())
        assert vmin > 50 and low > 50 and high > low

    def test_a_stack_is_cropped_frame_by_frame(self):
        stack = np.stack([_registered_frame(seed=s) for s in range(3)])
        (vmin, _), (low, _) = contrast_limits(stack)
        assert vmin > 50 and low > 50

    def test_non_finite_pixels_are_ignored(self):
        frame = _registered_frame()
        frame[50, 50] = np.nan
        frame[60, 60] = np.inf
        (vmin, vmax), (low, high) = contrast_limits(frame)
        assert np.isfinite([vmin, vmax, low, high]).all()

    def test_an_empty_or_flat_image_falls_back(self):
        assert contrast_limits(np.full((20, 20), np.nan)) == ((0.0, 1.0), (0.0, 1.0))
        assert contrast_limits(np.full((20, 20), 3.0)) == ((3.0, 3.0), (3.0, 3.0))

    def test_a_large_image_is_subsampled_border_excluded(self):
        frame = _registered_frame(shape=(2000, 2000))
        (vmin, _), (low, high) = contrast_limits(frame, max_samples=10_000)
        assert vmin > 50 and low > 50 and high > low
        assert value_range(frame, max_samples=10_000)[0] > 50

    def test_value_range_ignores_the_border_and_non_finite_pixels(self):
        frame = _registered_frame()
        frame[50, 50] = np.inf
        vmin, vmax = value_range(frame)
        assert vmin > 50 and np.isfinite(vmax)

    def test_a_bright_cell_on_the_edge_stays_within_the_range(self):
        frame = _registered_frame()
        frame[50, 1] = 1000.0
        (_, vmax), (_, high) = contrast_limits(frame)
        assert vmax == 1000.0 and high < 200
        assert value_range(frame)[1] == 1000.0

    def test_a_long_stack_of_small_frames_is_sampled_over_each_frame(self):
        # Over the budget, each frame keeps a grid rather than shrinking to a single pixel.
        stack = np.stack([_registered_frame(seed=s) for s in range(1000)])
        stack[:, 10:20, 10:20] = 500.0
        (vmin, vmax), (low, high) = contrast_limits(stack, max_samples=10_000)
        assert vmin > 50 and vmax == 500.0
        assert low > 50 and high == pytest.approx(500.0)
        assert value_range(np.full((20, 20), np.nan)) == (0.0, 1.0)

    def test_napari_limits_leave_out_the_border(self):
        stack = np.stack([_registered_frame(seed=s) for s in range(3)])[..., None]
        ((low, high),) = _get_contrast_limits(stack)
        assert low > 50 and high > low
