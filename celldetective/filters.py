"""
Image Filters Module
====================

This module provides a collection of image filtering and thresholding functions used throughout Celldetective.

These functions act as building blocks for image preprocessing pipelines, particularly within custom `extra_properties` measurements and configuration-based image segmentation protocols (e.g. `segment_frame_from_thresholds`).

Usage in Configuration
----------------------

Filters are typically specified in configuration dictionaries (e.g., `instructions`) as a list of lists or tuples, where the first element is the filter name and subsequent elements are arguments.

**Example Configuration:**

.. code-block:: python

    "filters": [
        ["gauss", 2.0],       # Applies gauss_filter(img, 2.0)
        ["subtract", 100],    # Applies subtract_filter(img, 100)
        ["abs"]               # Applies abs_filter(img)
    ]

This sequence is executed by `filter_image`, applying each filter to the output of the previous one.

Available Operations
--------------------

*   **Smoothing/Denoising**: `gauss`, `median`
*   **Edge/Texture**: `laplace`, `variance`, `std`, `dog` (Difference of Gaussians), `log` (Laplacian of Gaussian)
*   **Morphological**: `maximum` (dilation), `minimum` (erosion), `tophat` (white top-hat)
*   **Arithmetic**: `subtract`, `abs`, `invert`, `ln` (natural log), `percentile`
*   **Thresholding**: `otsu`, `multiotsu`, `local`, `niblack`, `sauvola`

Function Naming Convention
--------------------------
Each filter function is named ``<filter_name>_filter`` (e.g., ``gauss_filter``). In configuration lists, refer to them by ``<filter_name>`` (e.g., ``"gauss"``).

Copyright © 2022 Laboratoire Adhesion et Inflammation
Authored by R. Torro, K. Dervanova, L. Limozin
"""

import numpy

from celldetective.utils.image_cleaning import interpolate_nan
import numpy as np


def gauss_filter(img, sigma, interpolate=True, *kwargs):
    """
    Applies a Gaussian filter to an image.

    Parameters
    ----------
    img : ndarray
        The input image.
    sigma : float or sequence of scalars
        Standard deviation for Gaussian kernel.
    interpolate : bool, optional
        Whether to interpolate NaN values before filtering. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.gaussian_filter`.

    Returns
    -------
    ndarray
        The filtered image.
    """
    import scipy.ndimage as snd

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    return snd.gaussian_filter(img.astype(float), sigma, *kwargs)


def median_filter(img, size, interpolate=True, *kwargs):
    """
    Applies a median filter to an image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the median filter window.
    interpolate : bool, optional
        Whether to interpolate NaN values before filtering. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.median_filter`.

    Returns
    -------
    ndarray
        The filtered image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    size = int(size)

    import scipy.ndimage as snd

    return snd.median_filter(img, size, *kwargs)


def maximum_filter(img, size, interpolate=True, *kwargs):
    """
    Applies a maximum filter to an image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the maximum filter window.
    interpolate : bool, optional
        Whether to interpolate NaN values before filtering. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.maximum_filter`.

    Returns
    -------
    ndarray
        The filtered image.
    """
    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    import scipy.ndimage as snd

    return snd.maximum_filter(img.astype(float), size, *kwargs)


def minimum_filter(img, size, interpolate=True, *kwargs):
    """
    Applies a minimum filter to an image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the minimum filter window.
    interpolate : bool, optional
        Whether to interpolate NaN values before filtering. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.minimum_filter`.

    Returns
    -------
    ndarray
        The filtered image.
    """
    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    import scipy.ndimage as snd

    return snd.minimum_filter(img.astype(float), size, *kwargs)


def percentile_filter(img, percentile, size, interpolate=True, *kwargs):
    """
    Applies a percentile filter to an image.

    Parameters
    ----------
    img : ndarray
        The input image.
    percentile : float
        The percentile value to calculate.
    size : int
        The size of the percentile filter window.
    interpolate : bool, optional
        Whether to interpolate NaN values before filtering. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.percentile_filter`.

    Returns
    -------
    ndarray
        The filtered image.
    """
    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    import scipy.ndimage as snd

    return snd.percentile_filter(img.astype(float), percentile, size, *kwargs)


def subtract_filter(img, value, *kwargs):
    """
    Subtracts a scalar value from the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    value : float
        The value to subtract.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The image with the value subtracted.
    """
    return img.astype(float) - value


def abs_filter(img, *kwargs):
    """
    Computes the absolute value of the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The absolute value of the image.
    """
    return np.abs(img)


def ln_filter(img, interpolate=True, *kwargs):
    """
    Computes the natural logarithm of the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The natural logarithm of the image.
    """
    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    img[np.where(img > 0.0)] = np.log(img[np.where(img > 0.0)])
    img[np.where(img <= 0.0)] = 0.0

    return img


def variance_filter(img, size, interpolate=True):
    """
    Computes the local variance of the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the window over which to compute the variance.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.

    Returns
    -------
    ndarray
        The local variance image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    size = int(size)
    img = img.astype(float)
    import scipy.ndimage as snd

    win_mean = snd.uniform_filter(img, (size, size), mode="wrap")
    win_sqr_mean = snd.uniform_filter(img**2, (size, size), mode="wrap")
    img = win_sqr_mean - win_mean**2

    return img


def std_filter(img, size, interpolate=True):
    """
    Computes the local standard deviation of the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the window over which to compute the standard deviation.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.

    Returns
    -------
    ndarray
        The local standard deviation image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))

    size = int(size)
    img = img.astype(float)

    import scipy.ndimage as snd

    win_mean = snd.uniform_filter(img, (size, size), mode="wrap")
    win_sqr_mean = snd.uniform_filter(img**2, (size, size), mode="wrap")
    win_sqr_mean[win_sqr_mean <= 0.0] = 0.0  # add this to prevent sqrt from breaking

    sub = np.subtract(win_sqr_mean, win_mean**2)
    sub[sub <= 0.0] = 0.0
    img = np.sqrt(sub)

    return img


def laplace_filter(img, output=float, interpolate=True, *kwargs):
    """
    Applies a Laplace filter to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    output : type, optional
        The data type of the output. Default is float.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.laplace`.

    Returns
    -------
    ndarray
        The filtered image.
    """
    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))
    import scipy.ndimage as snd

    return snd.laplace(img.astype(float), *kwargs)


def dog_filter(
    img, blob_size=None, sigma_low=1, sigma_high=2, interpolate=True, *kwargs
):
    """
    Applies a Difference of Gaussians (DoG) filter to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    blob_size : float, optional
        Expected blob size, used to calculate sigmas if provided. Default is None.
    sigma_low : float, optional
        Standard deviation for the lower Gaussian. Default is 1.
    sigma_high : float, optional
        Standard deviation for the higher Gaussian. Default is 2.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.
    *kwargs
        Additional arguments passed to `skimage.filters.difference_of_gaussians`.

    Returns
    -------
    ndarray
        The filtered image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))
    if blob_size is not None:
        sigma_low = 1.0 / (1.0 + np.sqrt(2)) * blob_size
        sigma_high = np.sqrt(2) * sigma_low
    from skimage.filters import difference_of_gaussians

    return difference_of_gaussians(img.astype(float), sigma_low, sigma_high, *kwargs)


def otsu_filter(img, *kwargs):
    """
    Applies Otsu's thresholding to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The binary image after thresholding (0 or 1, as float).
    """
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(img.astype(float))
    binary = img >= thresh
    return binary.astype(float)


def multiotsu_filter(img, classes=3, *kwargs):
    """
    Applies Multi-Otsu thresholding to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    classes : int, optional
        number of classes to be found. Default is 3.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The segmented image (regions labeled).
    """
    from skimage.filters import threshold_multiotsu

    thresholds = threshold_multiotsu(img, classes=classes)
    regions = np.digitize(img, bins=thresholds)
    return regions.astype(float)


def local_filter(img, *kwargs):
    """
    Applies local thresholding to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    *kwargs
        Additional arguments passed to `skimage.filters.threshold_local`.

    Returns
    -------
    ndarray
        The binary image after thresholding (0 or 1, as float).
    """
    from skimage.filters import threshold_local

    thresh = threshold_local(img.astype(float), *kwargs)
    binary = img >= thresh
    return binary.astype(float)


def niblack_filter(img, *kwargs):
    """
    Applies Niblack thresholding to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    *kwargs
        Additional arguments passed to `skimage.filters.threshold_niblack`.

    Returns
    -------
    ndarray
        The binary image after thresholding (0 or 1, as float).
    """

    from skimage.filters import threshold_niblack

    thresh = threshold_niblack(img, *kwargs)
    binary = img >= thresh
    return binary.astype(float)


def sauvola_filter(img, *kwargs):
    """
    Applies Sauvola thresholding to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    *kwargs
        Additional arguments passed to `skimage.filters.threshold_sauvola`.

    Returns
    -------
    ndarray
        The binary image after thresholding (0 or 1, as float).
    """

    from skimage.filters import threshold_sauvola

    thresh = threshold_sauvola(img, *kwargs)
    binary = img >= thresh
    return binary.astype(float)


def log_filter(img, blob_size=None, sigma=1, interpolate=True, *kwargs):
    """
    Applies a Laplacian of Gaussian (LoG) filter to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    blob_size : float, optional
        Expected blob size, used to calculate sigmas if provided. Default is None.
    sigma : float, optional
        Standard deviation for the Gaussian kernel. Default is 1.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.gaussian_laplace`.

    Returns
    -------
    ndarray
        The filtered image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))
    if blob_size is not None:
        sigma_low = 1.0 / (1.0 + np.sqrt(2)) * blob_size
        sigma_high = np.sqrt(2) * sigma_low

    import scipy.ndimage as snd

    return snd.gaussian_laplace(img.astype(float), sigma, *kwargs)


def tophat_filter(img, size, connectivity=4, interpolate=True, *kwargs):
    """
    Applies a White Top-Hat filter to the image.

    Parameters
    ----------
    img : ndarray
        The input image.
    size : int
        The size of the structuring element.
    connectivity : int, optional
        The connectivity for determining the neighborhood. Default is 4.
    interpolate : bool, optional
        Whether to interpolate NaN values. Default is True.
    *kwargs
        Additional arguments passed to `scipy.ndimage.white_tophat`.

    Returns
    -------
    ndarray
        The filtered image.
    """

    if np.any(img != img) and interpolate:
        img = interpolate_nan(img.astype(float))
    import scipy.ndimage as snd

    structure = snd.generate_binary_structure(rank=2, connectivity=connectivity)
    img = snd.white_tophat(img.astype(float), structure=structure, size=size, *kwargs)
    return img


def invert_filter(img, value=65535, *kwargs):
    """
    Inverts the image by subtracting it from a maximum value.

    Parameters
    ----------
    img : ndarray
        The input image.
    value : float or int, optional
        The maximum value to subtract the image from. Default is 65535.
    *kwargs
        Unused arguments.

    Returns
    -------
    ndarray
        The inverted image.
    """

    img = img.astype(float)

    image_fill = np.zeros_like(img)
    image_fill[:, :] = value

    inverted = np.subtract(image_fill, img, where=img == img)
    return inverted


def filter_image(img, filters=None):
    """

    Apply one or more image filters to the input image.

    Parameters
    ----------
    img : ndarray
            The input image to be filtered.
    filters : list or None, optional
            A list of filters to be applied to the image. Each filter is represented as a tuple or list with the first element being
            the filter function name (minus the '_filter' extension, as listed in software.filters) and the subsequent elements being
            the arguments for that filter function. If None, the original image is returned without any filtering applied. Default is None.

    Returns
    -------
    ndarray
            The filtered image.

    Notes
    -----
    This function applies a series of image filters to the input image. The filters are specified as a list of tuples,
    where each tuple contains the name of the filter function and its corresponding arguments. The filters are applied
    sequentially to the image. If no filters are provided, the original image is returned unchanged.

    Examples
    --------
    >>> image = np.random.rand(256, 256)
    >>> filtered_image = filter_image(image, filters=[('gaussian', 3), ('median', 5)])

    """

    if filters is None:
        return img

    if img.ndim == 3:
        img = np.squeeze(img)

    for f in filters:
        func = eval(f[0] + "_filter")
        img = func(img, *f[1:])
    return img
