"""
Copyright © 2022 Laboratoire Adhesion et Inflammation
Authored by R. Torro, K. Dervanova, L. Limozin

This module defines additional measurement functions for use with `regionprops` via `measure_features`.

Usage
-----
Each function must follow these conventions:

- **First argument:** `regionmask` (numpy array)
  A binary mask of the cell of interest, as provided by `regionprops`.
- **Optional second argument:** `intensity_image` (numpy array)
  An image crop/bounding box associated with the cell (single-channel at a time).

Unlike the default `regionprops` from `scikit-image`, the cell image is **not** masked with zeros outside its boundaries.
This allows thresholding techniques to be used in measurements.

Naming Conventions & Indexing
------------------------------
- The measurement name is derived from the function name.
- If a function returns multiple values (e.g., for multichannel images), outputs are labeled sequentially:
  `function-0`, `function-1`, etc.
- To rename these outputs, use `rename_intensity_column` from `celldetective.utils`.
- `"intensity"` in function names is automatically replaced with the actual channel name:
  - Example: `"intensity-0"` → `"brightfield_channel"`.
- **Avoid digits smaller than the number of channels in function names** to prevent indexing conflicts.
  Prefer text-based names instead:

  .. code-block:: python

          # Bad practice:
          def intensity2(regionmask, intensity_image):
                  pass

          # Recommended:
          def intensity_two(regionmask, intensity_image):
                  pass

GUI Integration
---------------
New functions are **automatically** added to the list of available measurements in the graphical interface.
"""

import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt, center_of_mass
from scipy.spatial.distance import euclidean
from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.utils.image_cleaning import interpolate_nan
import skimage.measure as skm
from celldetective.utils.mask_cleaning import fill_label_holes
from celldetective.segmentation import segment_frame_from_thresholds
from sklearn.metrics import r2_score


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
    regionmask, intensity_image, target_channel="adhesion_channel"
):

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
    regionmask, intensity_image, target_channel="adhesion_channel"
):
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
    regionmask,
    intensity_image,
    target_channel="adhesion_channel",
    fill_holes=True,
    threshold=0.95,
):  # , target_channel='adhesion_channel'
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
    - The `target_channel` parameter tells regionprops to only measure this channel.

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
    regionmask,
    intensity_image,
    target_channel="adhesion_channel",
    fill_holes=True,
    threshold=0.95,
):  # , target_channel='adhesion_channel'

    subregion = (
        intensity_image < threshold
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion)) / float(np.sum(regionmask))


def area_dark_intensity_nintyfive(
    regionmask, intensity_image, target_channel="adhesion_channel", fill_holes=True
):  # , target_channel='adhesion_channel'

    subregion = (
        intensity_image < 0.95
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion))


def area_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel", fill_holes=True
):  # , target_channel='adhesion_channel'

    subregion = (
        intensity_image < 0.90
    ) * regionmask  # under one, under 0.8, under 0.6, whatever value!
    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    return float(np.sum(subregion))


def mean_dark_intensity_nintyfive(
    regionmask, intensity_image, target_channel="adhesion_channel", fill_holes=True
):
    """
    Calculate the mean intensity in a dark subregion below 95, handling NaN values.

    """
    subregion = (intensity_image < 0.95) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_nintyfive_fillhole_false(
    regionmask, intensity_image, target_channel="adhesion_channel"
):
    """
    Calculate the mean intensity in a dark subregion below 95, handling NaN values.
    """
    subregion = (
        intensity_image < 0.95
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def mean_dark_intensity_ninty_fillhole_false(
    regionmask, intensity_image, target_channel="adhesion_channel"
):
    """
    Calculate the mean intensity in a dark subregion, handling NaN values.
    """
    subregion = (
        intensity_image < 0.90
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def mean_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel", fill_holes=True
):
    """
    Calculate the mean intensity in a dark subregion below 90, handling NaN values.

    """
    subregion = (intensity_image < 0.90) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_eight_five(
    regionmask, intensity_image, target_channel="adhesion_channel", fill_holes=True
):
    """
    Calculate the mean intensity in a dark subregion below 85, handling NaN values.

    """
    subregion = (intensity_image < 0.85) * regionmask

    if fill_holes:
        subregion = skm.label(subregion, connectivity=2, background=0)
        subregion = fill_label_holes(subregion)
        subregion[subregion > 0] = 1

    masked_intensity = intensity_image[subregion == 1]

    return float(np.nanmean(masked_intensity))


def mean_dark_intensity_eight_five_fillhole_false(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (
        intensity_image < 0.85
    ) * regionmask  # Select dark regions within the mask

    masked_intensity = intensity_image[
        subregion == 1
    ]  # Extract pixel values from the selected region

    return float(np.nanmean(masked_intensity))  # Compute mean, ignoring NaNs


def percentile_zero_one_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 0.1))


def percentile_one_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 1))


def percentile_five_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 5))


def percentile_ten_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 10))


def percentile_ninty_five_dark_intensity_ninty(
    regionmask, intensity_image, target_channel="adhesion_channel"
):

    subregion = (intensity_image < 0.95) * regionmask
    return float(np.nanpercentile(intensity_image[subregion], 95))


def intensity_percentile_ninety_nine(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 99)


def intensity_percentile_ninety_five(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 95)


def intensity_percentile_ninety(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 90)


def intensity_percentile_seventy_five(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 75)


def intensity_percentile_fifty(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 50)


def intensity_percentile_twenty_five(regionmask, intensity_image):
    return np.nanpercentile(intensity_image[regionmask], 25)


# STD


def intensity_std(regionmask, intensity_image):
    return np.nanstd(intensity_image[regionmask])


def intensity_median(regionmask, intensity_image):
    return np.nanmedian(intensity_image[regionmask])


def intensity_nanmean(regionmask, intensity_image):

    if np.all(intensity_image == 0):
        return np.nan
    else:
        return np.nanmean(intensity_image[regionmask])


def intensity_center_of_mass_displacement(regionmask, intensity_image):
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


def intensity_center_of_mass_displacement_edge(regionmask, intensity_image):

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


def intensity_radial_gradient(regionmask, intensity_image):
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
