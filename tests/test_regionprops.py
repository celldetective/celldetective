"""
The intensity crop of custom measurements is not masked, the weighted properties are.

Custom measurements receive the raw crop of the bounding box, so that they can
look at the pixels around a cell. The weighted centroid and moments describe the
cell itself: a bright neighbor in the bounding box must not move them.
"""

import numpy as np
import pytest
from skimage.measure import regionprops_table as skimage_regionprops_table

from celldetective.regionprops import regionprops_table

WEIGHTED_PROPERTIES = [
    "centroid_weighted",
    "centroid_weighted_local",
    "moments_weighted",
    "moments_weighted_central",
    "moments_weighted_normalized",
    "moments_weighted_hu",
]


@pytest.fixture
def labels():
    """An L-shaped cell, whose bounding box holds many pixels outside of it."""
    labels = np.zeros((12, 12), dtype=int)
    labels[2:10, 2] = 1
    labels[9, 2:10] = 1
    return labels


@pytest.fixture
def image():
    """A varying signal, with a bright neighbor inside the bounding box of the cell."""
    rows, cols = np.indices((12, 12))
    image = 10.0 + rows + 2.0 * cols
    image[2:6, 6:10] = 1000.0
    return image


def assert_same_columns(ours, reference):
    assert set(ours) == set(reference)
    for column, values in reference.items():
        np.testing.assert_allclose(
            ours[column], values, equal_nan=True, err_msg=column
        )


@pytest.mark.parametrize("prop", WEIGHTED_PROPERTIES)
def test_weighted_property_matches_scikit_image(labels, image, prop):
    assert_same_columns(
        regionprops_table(labels, image, properties=[prop]),
        skimage_regionprops_table(labels, image, properties=[prop]),
    )


@pytest.mark.parametrize("prop", WEIGHTED_PROPERTIES)
def test_weighted_property_matches_scikit_image_multichannel(labels, image, prop):
    multichannel = np.stack([image, image[::-1]], axis=-1)
    assert_same_columns(
        regionprops_table(labels, multichannel, properties=[prop]),
        skimage_regionprops_table(labels, multichannel, properties=[prop]),
    )


def test_weighted_centroid_ignores_the_neighbor(labels, image):
    without_neighbor = np.where(labels > 0, image, 0.0)
    with_neighbor = regionprops_table(labels, image, properties=["centroid_weighted"])
    alone = regionprops_table(
        labels, without_neighbor, properties=["centroid_weighted"]
    )
    assert_same_columns(with_neighbor, alone)


def test_nan_outside_the_cell_leaves_the_weighted_centroid_defined(labels, image):
    image[3, 7] = np.nan
    props = regionprops_table(labels, image, properties=["centroid_weighted"])
    assert np.isfinite(props["centroid_weighted-0"]).all()
    assert np.isfinite(props["centroid_weighted-1"]).all()


def test_intensity_statistics_still_use_the_cell_only(labels, image):
    properties = ["intensity_mean", "intensity_max", "intensity_min", "intensity_std"]
    assert_same_columns(
        regionprops_table(labels, image, properties=properties),
        skimage_regionprops_table(labels, image, properties=properties),
    )


def test_custom_measurement_receives_the_raw_crop(labels, image):
    def crop_max(regionmask, intensity_image):
        return intensity_image.max()

    def cell_max(regionmask, intensity_image):
        return intensity_image[regionmask].max()

    props = regionprops_table(
        labels, image, properties=["label"], extra_properties=[crop_max, cell_max]
    )
    assert props["crop_max"][0] == 1000.0
    assert props["cell_max"][0] < 1000.0
