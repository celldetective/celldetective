"""
Extra Properties Module
=======================

This module defines custom measurement functions that extend `skimage.measure.regionprops`.

These functions are designed to be dynamically discovered and applied to single-cell regions during the feature extraction phase. They allow for complex, user-defined measurements that go beyond standard morphological or intensity features (e.g., area of dark regions, specific intensity percentiles).

Function Signature Specification
--------------------------------

To be valid, a function in this module must adhere to the following signature:

There are three valid signatures depending on whether the function needs the intensity image and, if so, which channels it should run on:

**Shape-only measurement** (regionmask only — called once per cell, no channel suffix):

.. code-block:: python

    def my_shape_measurement(regionmask):
        # No intensity image needed.  Produces column named exactly 'my_shape_measurement'.
        return scalar_value

**Run on all channels** (no ``target_channel`` — called once per channel):

.. code-block:: python

    def my_custom_measurement(regionmask, intensity_image, **kwargs):
        # intensity_image is a single-channel 2-D crop for each channel in turn
        return scalar_value

**Run on one specific channel** (with ``target_channel`` default):

.. code-block:: python

    def my_custom_measurement(regionmask, intensity_image, target_channel='my_channel', **kwargs):
        # intensity_image is the crop for 'my_channel' only
        return scalar_value

**How channel selection works:**

*   ``target_channel`` is **never passed** as an argument at call time.
*   The framework reads its **default value** via ``inspect.signature`` to identify which channel to use.
*   The function is then called **once**, receiving only the crop for that channel.
*   All other channel output slots are filled with ``NaN`` automatically.
*   If the named channel is not present in the experiment, a warning is logged and all slots remain ``NaN``.

**Arguments:**

*   **regionmask** (*ndarray*): A binary mask of the object (cell) within its bounding box.
*   **intensity_image** (*ndarray*): The intensity image crop. **Note:** Unlike scikit-image's ``regionprops``, this image is *not* masked — the background is not zeroed — allowing threshold-based analysis within the bounding box.
*   **target_channel** (*str, optional*): If present, its **default value** names the channel this function applies to. Do not pass it explicitly — it is read by the framework via ``inspect.signature``.
*   **kwargs**: Additional keyword arguments may be passed by the system.

**Return Value:**

*   Must return a **scalar** (float or int).
*   Returning `NaN` is permitted and handled.

Code Examples
-------------

Here are some simple examples to illustrate the structure:

.. code-block:: python

    def area(regionmask, intensity_image, **kwargs):
        return np.sum(regionmask)

    def mean_intensity(regionmask, intensity_image, **kwargs):
        # We select only the pixels within the cell mask
        masked_pixels = intensity_image[regionmask]
        return np.mean(masked_pixels)


Naming and Indexing Rules
-------------------------

The name of the function determines the column name in the output measurement table.

1.  **Channel Replacement**: If the function name contains the substring ``intensity``, it is automatically replaced by the actual channel name being measured.

    *   *Example:* ``intensity_mean`` -> ``green_channel_mean``

2.  **Multi-channel Indexing**: Since these functions are often run on multiple channels, avoid using simple digits (0-9) in the function name if they could conflict with channel indexing.

    *   *Bad:* ``measure_ch1``
    *   *Good:* ``measure_channel_one``

Integration Details
-------------------

*   **Automatic Discovery**: Any function defined in this module is automatically detected and listed in the GUI settings under "Extra features".
*   **Execution**: These functions are called by `celldetective.measure.measure_features`.

Copyright © 2022 Laboratoire Adhesion et Inflammation
Authored by R. Torro, K. Dervanova, L. Limozin
"""

import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass, binary_erosion
from scipy.stats import skew, kurtosis
from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.utils.image_cleaning import interpolate_nan
import skimage.measure as skm
from celldetective.utils.mask_cleaning import fill_label_holes
from celldetective.segmentation import segment_frame_from_thresholds
from sklearn.metrics import r2_score
from typing import Tuple


# def area_detected_in_ricm(regionmask, intensity_image, target_channel='adhesion_channel'):

# 	instructions = {
# 		"thresholds": [
# 			0.02,
# 			1000
# 		],
# 		"filters": [
# 			[
# 				"subtract",
# 				1
# 			],
# 			[
# 				"abs",
# 				2
# 			],
# 			[
# 				"gauss",
# 				0.8
# 			]
# 		],
# 		#"marker_min_distance": 1,
# 		#"marker_footprint_size": 10,
# 		"feature_queries": [
# 			"eccentricity > 0.99 or area < 60"
# 		],
# 	}

# 	lbl = segment_frame_from_thresholds(intensity_image, fill_holes=True, do_watershed=False, equalize_reference=None, edge_exclusion=False, **instructions)
# 	lbl[lbl>0] = 1 # instance to binary
# 	lbl[~regionmask] = 0 # make sure we don't measure stuff outside cell

# 	return np.sum(lbl)


def fraction_of_area_detected_in_intensity(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the fraction of the region area that is detected in the intensity image.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        Fraction of the area detected in the intensity image.
    """

    instructions = {
        "thresholds": [0.02, 1000],
        "filters": [["subtract", 1], ["abs", 2], ["gauss", 0.8]],
    }

    lbl = segment_frame_from_thresholds(
        intensity_image,
        do_watershed=False,
        fill_holes=True,
        equalize_reference=None,
        edge_exclusion=False,
        **instructions
    )
    lbl[lbl > 0] = 1  # instance to binary
    lbl[~regionmask] = 0  # make sure we don't measure stuff outside cell

    return float(np.sum(lbl)) / float(np.sum(regionmask))


def area_detected_in_intensity(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the detected area within the regionmask based on threshold-based segmentation.

    The function applies a predefined filtering and thresholding pipeline to the intensity image (normalized adhesion channel)
    to detect significant regions. The resulting segmented regions are restricted to the
    `regionmask`, ensuring that only the relevant area is measured.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.

    Returns
    -------
    detected_area : float
            The total area (number of pixels) detected based on intensity-based segmentation.

    Notes
    -----
    - The segmentation is performed using `segment_frame_from_thresholds()` with predefined parameters:

      - Thresholding range: `[0.02, 1000]`
      - Filters applied in sequence:

            - `"subtract"` with value `1` (subtract 1 from intensity values)
            - `"abs"` (take absolute value of intensities)
            - `"gauss"` with sigma `0.8` (apply Gauss filter with sigma `0.8`)

    - The segmentation includes hole filling.
    - The detected regions are converted to a binary mask (`lbl > 0`).
    - Any pixels outside the `regionmask` are excluded from the measurement.

    """

    instructions = {
        "thresholds": [0.02, 1000],
        "filters": [["subtract", 1], ["abs", 2], ["gauss", 0.8]],
    }

    lbl = segment_frame_from_thresholds(
        intensity_image,
        do_watershed=False,
        fill_holes=True,
        equalize_reference=None,
        edge_exclusion=False,
        **instructions
    )
    lbl[lbl > 0] = 1  # instance to binary
    lbl[~regionmask] = 0  # make sure we don't measure stuff outside cell

    return float(np.sum(lbl))


def area_dark_intensity(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
    threshold: float = 0.95,
) -> float:
    """
    Computes the absolute area within the regionmask where the intensity is below a given threshold.

    This function identifies pixels in the region where the intensity is lower than `threshold`.
    If `fill_holes` is `True`, small enclosed holes in the detected dark regions are filled before
    computing the total area.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.
    fill_holes : bool, optional
            If `True`, fills enclosed holes in the detected dark intensity regions before computing
            the area. Defaults to `True`.
    threshold : float, optional
            Intensity threshold below which a pixel is considered part of a dark region.
            Defaults to `0.95`.

    Returns
    -------
    dark_area : float
            The absolute area (number of pixels) where intensity values are below `threshold`, within the regionmask.

    Notes
    -----
    - The default threshold for defining "dark" intensity regions is `0.95`, but it can be adjusted.
    - If `fill_holes` is `True`, the function applies hole-filling to the detected dark regions
      using `skimage.measure.label` and `fill_label_holes()`.
    - The ``target_channel`` default value (``'adhesion_channel'``) tells the framework which
      channel to measure. The parameter is never passed at call time; instead, the framework
      reads the default via ``inspect.signature``, calls the function once with that channel's
      crop, and fills all other channel slots with ``NaN``.

    """

    subregion = (
        intensity_image < threshold
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion))


def fraction_of_area_dark_intensity(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
    threshold: float = 0.95,
) -> float:
    """
    Computes the fraction of the region area where intensity is below a threshold.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.
    fill_holes : bool, optional
        Whether to fill holes in the dark regions. Default is True.
    threshold : float, optional
        Intensity threshold. Default is 0.95.

    Returns
    -------
    float
        Fraction of the area with dark intensity.
    """

    subregion = (
        intensity_image < threshold
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion)) / float(np.sum(regionmask))

# STD


def intensity_std(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Computes the standard deviation of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        Standard deviation of intensity.
    """
    return np.nanstd(intensity_image[regionmask])


def intensity_median(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Computes the median intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        Median intensity.
    """
    return np.nanmedian(intensity_image[regionmask])


def intensity_nanmean(regionmask: np.ndarray, intensity_image: np.ndarray) -> float:
    """
    Computes the mean intensity within the region, ignoring NaNs.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        Mean intensity, or NaN if the image is all zeros.
    """

    if np.all(intensity_image == 0):
        return np.nan
    else:
        return np.nanmean(intensity_image[regionmask])


def intensity_center_of_mass_displacement(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Computes the displacement between the geometric centroid and the
    intensity-weighted center of mass of a region.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values indicate the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.

    Returns
    -------
    distance : float
            Euclidean distance between the geometric centroid and the intensity-weighted center of mass.
    direction_arctan : float
            Angle (in degrees) of displacement from the geometric centroid to the intensity-weighted center of mass,
            computed using `arctan2(delta_y, delta_x)`.
    delta_x : float
            Difference in x-coordinates (intensity-weighted centroid - geometric centroid).
    delta_y : float
            Difference in y-coordinates (intensity-weighted centroid - geometric centroid).

    Notes
    -----
    - If the `intensity_image` contains NaN values, it is first processed using `interpolate_nan()`.
    - Negative intensity values are set to zero to prevent misbehavior in center of mass calculation.
    - If the intensity image is entirely zero, all outputs are `NaN`.

    """

    if np.any(intensity_image != intensity_image):
        intensity_image = interpolate_nan(intensity_image.copy())

    if not np.all(intensity_image.flatten() == 0):

        y, x = np.mgrid[: regionmask.shape[0], : regionmask.shape[1]]
        xtemp = x.copy()
        ytemp = y.copy()

        intensity_image[intensity_image <= 0.0] = (
            0.0  # important to clip as negative intensities misbehave with center of mass
        )
        intensity_weighted_center = center_of_mass(
            intensity_image * regionmask, regionmask, 1
        )
        centroid_x = intensity_weighted_center[1]
        centroid_y = intensity_weighted_center[0]

        geometric_centroid_x = np.sum(xtemp * regionmask) / np.sum(regionmask)
        geometric_centroid_y = np.sum(ytemp * regionmask) / np.sum(regionmask)
        distance = np.sqrt(
            (geometric_centroid_y - centroid_y) ** 2
            + (geometric_centroid_x - centroid_x) ** 2
        )

        delta_x = geometric_centroid_x - centroid_x
        delta_y = geometric_centroid_y - centroid_y
        direction_arctan = np.arctan2(delta_y, delta_x) * 180 / np.pi

        return (
            distance,
            direction_arctan,
            centroid_x - geometric_centroid_x,
            centroid_y - geometric_centroid_y,
        )

    else:
        return np.nan, np.nan, np.nan, np.nan


def intensity_center_of_mass_displacement_edge(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Computes displacement of center of mass relative to the edge of the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    tuple
        (distance, direction_arctan, delta_x, delta_y).
        Returns (NaN, NaN, NaN, NaN) if calculation fails.
    """

    if np.any(intensity_image != intensity_image):
        intensity_image = interpolate_nan(intensity_image.copy())

    edge_mask = contour_of_instance_segmentation(regionmask, 3)

    if not np.all(intensity_image.flatten() == 0) and np.sum(edge_mask) > 0:

        y, x = np.mgrid[: edge_mask.shape[0], : edge_mask.shape[1]]
        xtemp = x.copy()
        ytemp = y.copy()

        intensity_image[intensity_image <= 0.0] = (
            0.0  # important to clip as negative intensities misbehave with center of mass
        )
        intensity_weighted_center = center_of_mass(
            intensity_image * edge_mask, edge_mask, 1
        )
        centroid_x = intensity_weighted_center[1]
        centroid_y = intensity_weighted_center[0]

        # centroid_x = np.sum(xtemp * intensity_image) / np.sum(intensity_image)
        geometric_centroid_x = np.sum(xtemp * regionmask) / np.sum(regionmask)
        geometric_centroid_y = np.sum(ytemp * regionmask) / np.sum(regionmask)

        distance = np.sqrt(
            (geometric_centroid_y - centroid_y) ** 2
            + (geometric_centroid_x - centroid_x) ** 2
        )

        delta_x = geometric_centroid_x - centroid_x
        delta_y = geometric_centroid_y - centroid_y
        direction_arctan = np.arctan2(delta_y, delta_x) * 180 / np.pi

        return (
            distance,
            direction_arctan,
            centroid_x - geometric_centroid_x,
            centroid_y - geometric_centroid_y,
        )
    else:
        return np.nan, np.nan, np.nan, np.nan


def intensity_radial_gradient(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> Tuple[float, float, float]:
    """
    Determines whether the intensity follows a radial gradient from the center to the edge of the cell.

    The function fits a linear model to the intensity values as a function of distance from the center
    (computed via the Euclidean distance transform). The slope of the fitted line indicates whether
    the intensity is higher at the center or at the edges.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.

    Returns
    -------
    slope : float
            Slope of the fitted linear model.

            - If `slope > 0`: Intensity increases towards the edge.
            - If `slope < 0`: Intensity is higher at the center.

    intercept : float
            Intercept of the fitted linear model.
    r2 : float
            Coefficient of determination (R²), indicating how well the linear model fits the intensity profile.

    Notes
    -----
    - If the `intensity_image` contains NaN values, they are interpolated using `interpolate_nan()`.
    - The Euclidean distance transform (`distance_transform_edt`) is used to compute the distance
      of each pixel from the edge.
    - The x-values for the linear fit are reversed so that the origin is at the center.
    - A warning suppression is applied to ignore messages about poorly conditioned polynomial fits.

    """

    if np.any(intensity_image != intensity_image):
        intensity_image = interpolate_nan(intensity_image.copy())

    # try:
    warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

    # intensities
    y = intensity_image[regionmask].flatten()

    # distance to edge
    x = distance_transform_edt(regionmask.copy())
    x = x[regionmask].flatten()
    x = max(x) - x  # origin at center of cells

    params = np.polyfit(x, y, 1)
    line = np.poly1d(params)
    # coef > 0 --> more signal at edge than center, coef < 0 --> more signal at center than edge

    r2 = r2_score(y, line(x))

    return line.coefficients[0], line.coefficients[1], r2

# Variations on intensity (for LAI)


def area_dark_intensity_ninetyfive(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
) -> float:
    """
    Computes the area of the region where intensity is below 0.95.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.
    fill_holes : bool, optional
        Whether to fill holes in the dark regions. Default is True.

    Returns
    -------
    float
        Area with intensity below 0.95.
    """

    subregion = (
        intensity_image < 0.95
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion))


def area_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
) -> float:
    """
    Computes the area of the region where intensity is below 0.90.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.
    fill_holes : bool, optional
        Whether to fill holes in the dark regions. Default is True.

    Returns
    -------
    float
        Area with intensity below 0.90.
    """

    subregion = (
        intensity_image < 0.90
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion))


def mean_dark_intensity_ninetyfive(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
) -> float:
    """
    Calculate the mean intensity in a dark subregion below 95, handling NaN values.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.
    fill_holes : bool, optional
            If `True`, fills enclosed holes in the detected dark intensity regions before computing
            the area. Defaults to `True`.

    Returns
    -------
    float
            Mean intensity value in the dark subregion.
    """
    subregion = (intensity_image < 0.95) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_ninetyfive_fillhole_false(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Calculate the mean intensity in a dark subregion below 95, handling NaN values.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.

    Returns
    -------
    float
            Mean intensity value in the dark subregion.
    """
    subregion = (
        intensity_image < 0.95
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def mean_dark_intensity_ninety_fillhole_false(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Calculate the mean intensity in a dark subregion, handling NaN values.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.

    Returns
    -------
    float
            Mean intensity value in the dark subregion.
    """
    subregion = (
        intensity_image < 0.90
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def mean_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
) -> float:
    """
    Calculate the mean intensity in a dark subregion below 90, handling NaN values.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.
    fill_holes : bool, optional
            If `True`, fills enclosed holes in the detected dark intensity regions before computing
            the area. Defaults to `True`.

    Returns
    -------
    float
            Mean intensity value in the dark subregion.
    """
    subregion = (intensity_image < 0.90) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_eighty_five(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
    fill_holes: bool = True,
) -> float:
    """
    Calculate the mean intensity in a dark subregion below 85, handling NaN values.

    Parameters
    ----------
    regionmask : ndarray
            A binary mask (2D array) where nonzero values define the region of interest.
    intensity_image : ndarray
            A 2D array of the same shape as `regionmask`, representing the intensity
            values associated with the region.
    target_channel : str, optional
            Name of the intensity channel used for measurement. Defaults to `'adhesion_channel'`.
    fill_holes : bool, optional
            If `True`, fills enclosed holes in the detected dark intensity regions before computing
            the area. Defaults to `True`.

    Returns
    -------
    float
            Mean intensity value in the dark subregion.
    """
    subregion = (intensity_image < 0.85) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_eight_five_fillhole_false(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the mean intensity in dark regions (< 0.85), ignoring NaNs and without hole filling.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        Mean intensity in dark regions.
    """

    subregion = (
        intensity_image < 0.85
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def percentile_zero_one_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the 0.1th percentile of intensity in dark regions (< 0.95).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        0.1th percentile of intensity in dark regions.
    """

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 0.1))


def percentile_one_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the 1st percentile of intensity in dark regions (< 0.95).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        1st percentile of intensity in dark regions.
    """

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 1))


def percentile_five_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the 5th percentile of intensity in dark regions (< 0.95).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        5th percentile of intensity in dark regions.
    """

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 5))


def percentile_ten_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the 10th percentile of intensity in dark regions (< 0.95).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        10th percentile of intensity in dark regions.
    """

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 10))


def percentile_ninty_five_dark_intensity_ninety(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
    target_channel: str = "adhesion_channel",
) -> float:
    """
    Computes the 95th percentile of intensity in dark regions (< 0.95).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.
    target_channel : str, optional
        Target channel name. Default is 'adhesion_channel'.

    Returns
    -------
    float
        95th percentile of intensity in dark regions.
    """

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 95))


def intensity_percentile_ninety_nine(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 99th percentile of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        99th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 99)


def intensity_percentile_ninety_five(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 95th percentile of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        95th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 95)


def intensity_percentile_ninety(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 90th percentile of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        90th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 90)


def intensity_percentile_seventy_five(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 75th percentile of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        75th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 75)


def intensity_percentile_fifty(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 50th percentile (median) of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        50th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 50)


def intensity_percentile_twenty_five(
    regionmask: np.ndarray, intensity_image: np.ndarray
) -> float:
    """
    Computes the 25th percentile of intensity within the region.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image.

    Returns
    -------
    float
        25th percentile of intensity.
    """
    return np.nanpercentile(intensity_image[regionmask], 25)


# Shape descriptors


def circularity(regionmask: np.ndarray) -> float:
    """
    Computes the circularity of the cell mask.

    Circularity is defined as 4π × area / perimeter², equal to 1.0 for a
    perfect circle and approaching 0 for elongated or highly irregular shapes.
    Valid for both fluorescence and label-free cells.

    This function uses a **1-argument signature** (regionmask only), so the
    framework calls it once per cell — not once per channel — and produces a
    single scalar column named ``circularity`` with no channel suffix.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.

    Returns
    -------
    float
        Circularity value in [0, 1]. Returns NaN if the perimeter is zero.
    """
    area = float(np.sum(regionmask))
    p = skm.perimeter(regionmask)
    if p == 0:
        return np.nan
    return float((4.0 * np.pi * area) / (p ** 2))


def aspect_ratio(regionmask: np.ndarray) -> float:
    """
    Computes the aspect ratio of the cell mask (major axis / minor axis).

    Values close to 1 indicate round cells; higher values indicate elongated
    cells. Valid for both fluorescence and label-free cells.

    This function uses a **1-argument signature** (regionmask only), so the
    framework calls it once per cell — not once per channel — and produces a
    single scalar column named ``aspect_ratio`` with no channel suffix.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.

    Returns
    -------
    float
        Aspect ratio (≥ 1). Returns NaN if the minor axis has zero length.
    """
    props = skm.regionprops(regionmask.astype(np.uint8))
    if not props:
        return np.nan
    p = props[0]
    if p.axis_minor_length == 0:
        return np.nan
    return float(p.axis_major_length / p.axis_minor_length)


# Per-channel intensity distribution moments


def intensity_skewness(regionmask: np.ndarray, intensity_image: np.ndarray, **kwargs) -> float:
    """
    Computes the skewness of the in-mask pixel intensity distribution.

    Positive skewness indicates a long right tail (e.g., sparse bright puncta
    over a dim background). Negative skewness indicates a long left tail.
    Near-zero values are typical of uniformly distributed cytoplasmic staining.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image (single channel).

    Returns
    -------
    float
        Fisher skewness of in-mask pixels. Returns NaN if fewer than 3 pixels
        are available.
    """
    pixels = intensity_image[regionmask]
    if len(pixels) < 3:
        return np.nan
    return float(skew(pixels))


def intensity_kurtosis(regionmask: np.ndarray, intensity_image: np.ndarray, **kwargs) -> float:
    """
    Computes the excess kurtosis of the in-mask pixel intensity distribution.

    High positive kurtosis indicates a heavy-tailed or peaked distribution
    (e.g., bright puncta concentrated in a small area). Negative kurtosis
    indicates a flatter distribution than a Gaussian. Zero is the Gaussian
    reference (Fisher definition).

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image (single channel).

    Returns
    -------
    float
        Excess kurtosis of in-mask pixels (Fisher definition, Gaussian = 0).
        Returns NaN if fewer than 4 pixels are available.
    """
    pixels = intensity_image[regionmask]
    if len(pixels) < 4:
        return np.nan
    return float(kurtosis(pixels))


# Membrane-to-cytoplasm intensity ratio


def intensity_membrane_cytoplasm_ratio(
    regionmask: np.ndarray, intensity_image: np.ndarray, **kwargs
) -> float:
    """
    Computes the ratio of mean membrane-zone intensity to mean cytoplasm intensity.

    The membrane zone is approximated by eroding the cell mask by 3 pixels; the
    resulting core is the cytoplasm region, and the ring between the full mask
    and the eroded core is the membrane zone.

    A ratio > 1 indicates membrane enrichment; a ratio < 1 indicates cytoplasmic
    enrichment. Useful for assessing receptor polarisation, membrane trafficking,
    or cortical actin accumulation.

    Parameters
    ----------
    regionmask : ndarray
        Binary mask of the region of interest.
    intensity_image : ndarray
        Intensity image (single channel).

    Returns
    -------
    float
        Mean membrane intensity divided by mean cytoplasm intensity. Returns NaN
        if either zone is empty or if the cytoplasm mean is zero.
    """
    cytoplasm_mask = binary_erosion(regionmask, iterations=3)
    membrane_mask = regionmask & ~cytoplasm_mask
    if not np.any(membrane_mask) or not np.any(cytoplasm_mask):
        return np.nan
    mem_mean = np.mean(intensity_image[membrane_mask])
    cyt_mean = np.mean(intensity_image[cytoplasm_mask])
    if cyt_mean == 0:
        return np.nan
    return float(mem_mean / cyt_mean)
