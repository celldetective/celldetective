"""Display contrast shared by the stack viewers, the annotators and napari."""

from typing import Sequence, Tuple

import numpy as np

# Fraction of the image left out on each side when computing contrast bounds: a
# registered stack keeps a residual border of zeros that drags the lower bound down.
CONTRAST_EDGE_MARGIN = 0.05

# Percentiles of the interior pixels used as the initial contrast limits.
CONTRAST_PERCENTILES = (0.1, 99.9)

# Above this many pixels, the bounds are estimated on a regular subsample, keeping at
# least MIN_FRAME_SAMPLES of them per frame of a stack.
MAX_CONTRAST_SAMPLES = 1_000_000
MIN_FRAME_SAMPLES = 4096


def interior_slices(
    shape: Sequence[int],
    margin: float = CONTRAST_EDGE_MARGIN,
    axes: Sequence[int] = (-2, -1),
) -> Tuple[slice, ...]:
    """
    Index of the interior of an array of the given shape, ``margin`` of each spatial axis
    cut off on each side.

    Parameters
    ----------
    shape : sequence of int
        Shape of the array.
    margin : float, optional
        Fraction of each spatial dimension cut off on each side (below 0.5).
    axes : sequence of int, optional
        The spatial axes (Y and X), by default the last two.

    Returns
    -------
    tuple of slice
        One slice per axis (unchanged along an axis too short to crop).
    """
    index = [slice(None)] * len(shape)
    for axis in axes:
        n = shape[axis]
        d = int(n * margin)
        index[axis] = slice(d, n - d)
    return tuple(index)


def image_interior(
    image: np.ndarray,
    margin: float = CONTRAST_EDGE_MARGIN,
    axes: Sequence[int] = (-2, -1),
) -> np.ndarray:
    """
    The image without a border of ``margin`` of its size on each side of its spatial axes.

    Parameters
    ----------
    image : numpy.ndarray
        Image or stack of images (a dask array stays lazy).
    margin : float, optional
        Fraction of each spatial dimension cut off on each side (below 0.5).
    axes : sequence of int, optional
        The spatial axes (Y and X), by default the last two.

    Returns
    -------
    numpy.ndarray
        A view of the interior of the image.
    """
    return image[interior_slices(image.shape, margin, axes)]


def _grid_sample(image: np.ndarray, axes: Sequence[int], max_samples: int) -> np.ndarray:
    """
    The image on a regular grid of at most about ``max_samples`` pixels, as a NumPy array.

    Every frame of a stack keeps a spatial grid of the same pitch; frames are skipped
    only when even a coarse grid on each of them would exceed the budget.
    """
    if image.size <= max_samples:
        return np.asarray(image)
    spatial_axes = [axis % image.ndim for axis in axes]
    other_axes = [axis for axis in range(image.ndim) if axis not in spatial_axes]
    n_spatial = int(np.prod([image.shape[axis] for axis in spatial_axes]))
    n_frames = image.size // n_spatial
    per_frame = max(max_samples // n_frames, MIN_FRAME_SAMPLES)
    step = max(1, int(np.ceil(np.sqrt(n_spatial / per_frame))))
    kept_per_frame = int(np.prod([-(-image.shape[axis] // step) for axis in spatial_axes]))
    frame_step = max(1, int(np.ceil(n_frames * kept_per_frame / max_samples)))

    # Strided slicing rather than flattening: no copy of the full image, and a lazy
    # (dask) stack is subsampled before it is realised.
    index = [slice(None)] * image.ndim
    for axis in spatial_axes:
        index[axis] = slice(None, None, step)
    if other_axes:
        index[other_axes[0]] = slice(None, None, frame_step)
    return np.asarray(image[tuple(index)])


def _finite(values: np.ndarray) -> np.ndarray:
    values = values.ravel()
    if np.issubdtype(values.dtype, np.integer):
        return values
    return values[np.isfinite(values)]


def _samples(
    image: np.ndarray, axes: Sequence[int], max_samples: int
) -> Tuple[np.ndarray, float]:
    """
    Finite interior pixels of a grid sample of the image, and the maximum of the sample.

    The border is left out for the lower bounds only: registration fills it with zeros,
    but a bright cell close to the edge must stay within reach of the slider.
    """
    sample = _grid_sample(image, axes, max_samples)
    everywhere = _finite(sample)
    vmax = float(everywhere.max()) if everywhere.size else np.nan
    return _finite(image_interior(sample, axes=axes)), vmax


def value_range(
    image: np.ndarray,
    axes: Sequence[int] = (-2, -1),
    max_samples: int = MAX_CONTRAST_SAMPLES,
) -> Tuple[float, float]:
    """
    Range of the finite pixels of an image, the border left out of its minimum.

    Parameters
    ----------
    image : numpy.ndarray
        Image or stack of images.
    axes : sequence of int, optional
        The spatial axes (Y and X), by default the last two.
    max_samples : int, optional
        Above this many pixels, the range is that of a regular subsample.

    Returns
    -------
    tuple of float
        ``(min, max)``, or ``(0, 1)`` if there is no finite interior pixel.
    """
    values, vmax = _samples(image, axes, max_samples)
    if values.size == 0:
        return 0.0, 1.0
    return float(values.min()), vmax


def contrast_limits(
    image: np.ndarray,
    percentiles: Tuple[float, float] = CONTRAST_PERCENTILES,
    axes: Sequence[int] = (-2, -1),
    max_samples: int = MAX_CONTRAST_SAMPLES,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Range and initial limits of a contrast slider for an image, its border left out.

    Parameters
    ----------
    image : numpy.ndarray
        Image or stack of images.
    percentiles : tuple of float, optional
        Percentiles of the interior pixels used as initial limits.
    axes : sequence of int, optional
        The spatial axes (Y and X), by default the last two.
    max_samples : int, optional
        Above this many pixels, the bounds are those of a regular subsample.

    Returns
    -------
    tuple
        ``((min, max), (low, high))``: the range of the finite pixels (the border left out
        of its minimum, see ``value_range``) and the percentiles of the interior ones.
        ``((0, 1), (0, 1))`` if there are none; the limits fall back on the range when the
        percentiles do not span an interval.
    """
    values, vmax = _samples(image, axes, max_samples)
    if values.size == 0:
        return (0.0, 1.0), (0.0, 1.0)
    # One partition for the interior bounds.
    vmin, low, high = (float(v) for v in np.percentile(values, [0, *percentiles]))
    if not high > low:
        low, high = vmin, vmax
    return (vmin, vmax), (low, high)
