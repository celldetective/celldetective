"""
Stack registration by Fourier phase cross-correlation.

The drift of a time-lapse is estimated on a single registration channel, frame by frame, with
:func:`skimage.registration.phase_cross_correlation`. Before the Fourier transform, each frame is
mean-centred and multiplied by a Tukey window. The taper brings the image smoothly to zero at the
border of the correlation region, which suppresses the spectral leakage (the bright cross in the
power spectrum) that an abrupt image edge would otherwise produce and that biases the peak toward
a null shift.

The correlation region is a disk of a given radius around the image centre, so that the same
setting applies to every position of a batch. Everything outside the disk (vignetting, dust on
the edge of the field, stitching or illumination artefacts) is ignored. Without a radius, the
taper is applied separately along each axis over the full frame.
"""

from typing import Optional, Tuple

import numpy as np

REFERENCE_MODES = ("previous", "first")


def downscale_frame(frame: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Reduce a frame by block averaging, to estimate the drift faster on large images.

    Averaging over ``factor x factor`` blocks low-pass filters the image before subsampling, so
    fine texture does not alias into spurious correlation peaks. The frame is first cropped to a
    multiple of ``factor`` so no padded border enters the correlation. Being a pure change of
    scale, a translation of ``d`` pixels in the reduced frame is ``d * factor`` at full scale.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image.
    factor : int, optional
        Downscaling factor. 1 (default) returns the frame unchanged.

    Returns
    -------
    numpy.ndarray
        Frame of shape ``(Y // factor, X // factor)``.
    """

    factor = int(factor)
    if factor < 1:
        raise ValueError(f"The downscaling factor must be at least 1, got {factor}.")
    if factor == 1:
        return frame
    ny, nx = (s // factor for s in frame.shape)
    blocks = np.asarray(frame, dtype=float)[: ny * factor, : nx * factor]
    return blocks.reshape(ny, factor, nx, factor).mean(axis=(1, 3))


def tukey_window(
    shape: Tuple[int, int],
    alpha: float = 0.25,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Build the 2D Tukey window that weights a frame before the phase correlation.

    Parameters
    ----------
    shape : tuple of int
        Shape ``(Y, X)`` of the frames.
    alpha : float, optional
        Fraction of the window occupied by the cosine taper, between 0 (rectangular window) and
        1 (Hann window). Default is 0.25.
    radius : float, optional
        Radius in pixels of the correlation disk centred on the image. Weights fall to zero at
        ``radius``, the cosine taper covering its outer ``alpha`` fraction. If None (default), a
        separable Tukey window spans the full frame.

    Returns
    -------
    numpy.ndarray
        Float window of the given shape, with values in [0, 1].
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"The Tukey alpha must be within [0, 1], got {alpha}.")

    ny, nx = shape
    if radius is None:
        from scipy.signal.windows import tukey

        return np.outer(tukey(ny, alpha), tukey(nx, alpha))

    if radius <= 0:
        raise ValueError(f"The correlation radius must be positive, got {radius}.")

    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - (ny - 1) / 2.0, xx - (nx - 1) / 2.0)

    window = np.zeros(shape, dtype=float)
    plateau = radius * (1.0 - alpha)
    window[r <= plateau] = 1.0
    taper = (r > plateau) & (r < radius)
    window[taper] = 0.5 * (1.0 + np.cos(np.pi * (r[taper] - plateau) / (radius - plateau)))
    return window


def prepare_for_correlation(frame: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    Mean-centre a frame inside the window and apply the window.

    NaN pixels (e.g. left by a previous background correction) are set to the weighted mean, so
    they contribute nothing once centred.

    Parameters
    ----------
    frame : numpy.ndarray
        2D image.
    window : numpy.ndarray
        Weights from :func:`tukey_window`, same shape as ``frame``.

    Returns
    -------
    numpy.ndarray
        The windowed, zero-mean frame.
    """

    frame = np.asarray(frame, dtype=float)
    valid = np.isfinite(frame) & (window > 0)
    if not np.any(valid):
        return np.zeros_like(frame)
    mean = np.average(frame[valid], weights=window[valid])
    centred = np.where(np.isfinite(frame), frame - mean, 0.0)
    return centred * window


def has_signal(frame: np.ndarray) -> bool:
    """Return True if ``frame`` has finite values that are not all equal."""
    return bool(np.any(np.isfinite(frame))) and np.nanmax(frame) != np.nanmin(frame)


def estimate_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    window: np.ndarray,
    upsample_factor: int = 10,
) -> np.ndarray:
    """
    Estimate the translation that registers ``moving`` onto ``reference``.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference 2D image.
    moving : numpy.ndarray
        2D image to register.
    window : numpy.ndarray
        Weights from :func:`tukey_window`.
    upsample_factor : int, optional
        Sub-pixel precision is ``1 / upsample_factor`` pixel. Default is 10.

    Returns
    -------
    numpy.ndarray
        Shift ``(dy, dx)`` in pixels to apply to ``moving`` (e.g. with
        :func:`scipy.ndimage.shift`) to align it with ``reference``.
    """

    from skimage.registration import phase_cross_correlation

    if not has_signal(reference) or not has_signal(moving):
        # Empty or uniform frame: no signal to correlate.
        return np.zeros(2)

    shift, _, _ = phase_cross_correlation(
        prepare_for_correlation(reference, window),
        prepare_for_correlation(moving, window),
        upsample_factor=upsample_factor,
        normalization=None,
    )
    return np.asarray(shift, dtype=float)


def estimate_drift(
    frames,
    window: np.ndarray,
    reference: str = "previous",
    upsample_factor: int = 10,
    progress_callback=None,
) -> np.ndarray:
    """
    Estimate the shift of every frame of a single-channel sequence.

    Parameters
    ----------
    frames : sequence of numpy.ndarray
        2D frames in temporal order. May be a lazy iterable, so a movie never has to be held in
        memory.
    window : numpy.ndarray
        Weights from :func:`tukey_window`.
    reference : {"previous", "first"}, optional
        ``"previous"`` (default) correlates each frame with the one before and accumulates the
        shifts, which follows a drift that makes the field change a lot over the movie.
        ``"first"`` correlates each frame with the first one, which does not accumulate error
        but needs the first frame to stay similar to the others.
    upsample_factor : int, optional
        Sub-pixel precision factor passed to :func:`estimate_shift`. Default is 10.
    progress_callback : callable, optional
        Called as ``progress_callback(iter=k)`` after each frame.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(T, 2)`` with the ``(dy, dx)`` shift that aligns each frame onto the
        first frame with signal. Frames up to that one get a zero shift.

    Notes
    -----
    Empty or uniform frames (e.g. a dropped acquisition) are never used as reference: they keep
    the shift of the frame before them, and the next frame is correlated with the last frame
    that had signal, so the drift across the gap is not lost.
    """

    if reference not in REFERENCE_MODES:
        raise ValueError(f"reference must be one of {REFERENCE_MODES}, got {reference!r}.")

    shifts = []
    # Last frame with signal (reference="previous") or first one (reference="first").
    anchor = None
    for k, frame in enumerate(frames):
        if not has_signal(frame):
            shifts.append(shifts[-1] if shifts else np.zeros(2))
        elif anchor is None:
            anchor = frame
            shifts.append(shifts[-1] if shifts else np.zeros(2))
        elif reference == "first":
            shifts.append(estimate_shift(anchor, frame, window, upsample_factor))
        else:
            step = estimate_shift(anchor, frame, window, upsample_factor)
            shifts.append(shifts[-1] + step)
            anchor = frame
        if progress_callback:
            progress_callback(iter=k)

    return np.array(shifts).reshape(-1, 2)
