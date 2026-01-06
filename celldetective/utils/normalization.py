import gc

import numpy as np


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    # from csbdeep https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)
    try:
        import numexpr

        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalize(
    frame,
    percentiles=(0.0, 99.99),
    values=None,
    ignore_gray_value=0.0,
    clip=False,
    amplification=None,
    dtype=float,
):
    """

    Normalize the intensity values of a frame.

    Parameters
    ----------
    frame : ndarray
            The input frame to be normalized.
    percentiles : tuple, optional
            The percentiles used to determine the minimum and maximum values for normalization. Default is (0.0, 99.99).
    values : tuple or None, optional
            The specific minimum and maximum values to use for normalization. If None, percentiles are used. Default is None.
    ignore_gray_value : float or None, optional
            The gray value to ignore during normalization. If specified, the pixels with this value will not be normalized. Default is 0.0.

    Returns
    -------
    ndarray
            The normalized frame.

    Notes
    -----
    This function performs intensity normalization on a frame. It computes the minimum and maximum values for normalization either
    using the specified values or by calculating percentiles from the frame. The frame is then normalized between the minimum and
    maximum values using the `normalize_mi_ma` function. If `ignore_gray_value` is specified, the pixels with this value will be
    left unmodified during normalization.

    Examples
    --------
    >>> frame = np.array([[10, 20, 30],
                                              [40, 50, 60],
                                              [70, 80, 90]])
    >>> normalized = normalize(frame)
    >>> normalized

    array([[0. , 0.2, 0.4],
               [0.6, 0.8, 1. ],
               [1.2, 1.4, 1.6]], dtype=float32)

    >>> normalized = normalize(frame, percentiles=(10.0, 90.0))
    >>> normalized

    array([[0.33333334, 0.44444445, 0.5555556 ],
               [0.6666667 , 0.7777778 , 0.8888889 ],
               [1.        , 1.1111112 , 1.2222222 ]], dtype=float32)

    """

    frame = frame.astype(float)

    if ignore_gray_value is not None:
        subframe = frame[frame != ignore_gray_value]
    else:
        subframe = frame.copy()

    if values is not None:
        mi = values[0]
        ma = values[1]
    else:
        mi = np.nanpercentile(subframe.flatten(), percentiles[0], keepdims=True)
        ma = np.nanpercentile(subframe.flatten(), percentiles[1], keepdims=True)

    frame0 = frame.copy()
    frame = normalize_mi_ma(frame0, mi, ma, clip=False, eps=1e-20, dtype=np.float32)
    if amplification is not None:
        frame *= amplification
    if clip:
        if amplification is None:
            amplification = 1.0
        frame[frame >= amplification] = amplification
        frame[frame <= 0.0] = 0.0
    if ignore_gray_value is not None:
        frame[np.where(frame0) == ignore_gray_value] = ignore_gray_value

    return frame.copy().astype(dtype)


def normalize_multichannel(
    multichannel_frame: np.ndarray,
    percentiles=None,
    values=None,
    ignore_gray_value=0.0,
    clip=False,
    amplification=None,
    dtype=float,
):
    """
    Normalizes a multichannel frame by adjusting the intensity values of each channel based on specified percentiles,
    direct value ranges, or amplification factors, with options to ignore a specific gray value and to clip the output.

    Parameters
    ----------
    multichannel_frame : ndarray
            The input multichannel image frame to be normalized, expected to be a 3-dimensional array where the last dimension
            represents the channels.
    percentiles : list of tuples or tuple, optional
            Percentile ranges (low, high) for each channel used to scale the intensity values. If a single tuple is provided,
            it is applied to all channels. If None, the default percentile range of (0., 99.99) is used for each channel.
    values : list of tuples or tuple, optional
            Direct value ranges (min, max) for each channel to scale the intensity values. If a single tuple is provided, it
            is applied to all channels. This parameter overrides `percentiles` if provided.
    ignore_gray_value : float, optional
            A specific gray value to ignore during normalization (default is 0.).
    clip : bool, optional
            If True, clips the output values to the range [0, 1] or the specified `dtype` range if `dtype` is not float
            (default is False).
    amplification : float, optional
            A factor by which to amplify the intensity values after normalization. If None, no amplification is applied.
    dtype : data-type, optional
            The desired data-type for the output normalized frame. The default is float, but other types can be specified
            to change the range of the output values.

    Returns
    -------
    ndarray
            The normalized multichannel frame as a 3-dimensional array of the same shape as `multichannel_frame`.

    Raises
    ------
    AssertionError
            If the input `multichannel_frame` does not have 3 dimensions, or if the length of `values` does not match the
            number of channels in `multichannel_frame`.

    Notes
    -----
    - This function provides flexibility in normalization by allowing the use of percentile ranges, direct value ranges,
      or amplification factors.
    - The function makes a copy of the input frame to avoid altering the original data.
    - When both `percentiles` and `values` are provided, `values` takes precedence for normalization.

    Examples
    --------
    >>> multichannel_frame = np.random.rand(100, 100, 3)  # Example multichannel frame
    >>> normalized_frame = normalize_multichannel(multichannel_frame, percentiles=[(1, 99), (2, 98), (0, 100)])
    # Normalizes each channel of the frame using specified percentile ranges.

    """

    mf = multichannel_frame.copy().astype(float)
    assert mf.ndim == 3, f"Wrong shape for the multichannel frame: {mf.shape}."
    if percentiles is None:
        percentiles = [(0.0, 99.99)] * mf.shape[-1]
    elif isinstance(percentiles, tuple):
        percentiles = [percentiles] * mf.shape[-1]
    if values is not None:
        if isinstance(values, tuple):
            values = [values] * mf.shape[-1]
        assert (
            len(values) == mf.shape[-1]
        ), "Mismatch between the normalization values provided and the number of channels."

    mf_new = []
    for c in range(mf.shape[-1]):
        if values is not None:
            v = values[c]
        else:
            v = None

        if np.all(mf[:, :, c] == 0.0):
            mf_new.append(mf[:, :, c].copy())
        else:
            norm = normalize(
                mf[:, :, c].copy(),
                percentiles=percentiles[c],
                values=v,
                ignore_gray_value=ignore_gray_value,
                clip=clip,
                amplification=amplification,
                dtype=dtype,
            )
            mf_new.append(norm)

    return np.moveaxis(mf_new, 0, -1)


def get_stack_normalization_values(stack, percentiles=None, ignore_gray_value=0.0):
    """
    Computes the normalization value ranges (minimum and maximum) for each channel in a 4D stack based on specified percentiles.

    This function calculates the value ranges for normalizing each channel within a 4-dimensional stack, with dimensions
    expected to be in the order of Time (T), Y (height), X (width), and Channels (C). The normalization values are determined
    by the specified percentiles for each channel. An option to ignore a specific gray value during computation is provided,
    though its effect is not implemented in this snippet.

    Parameters
    ----------
    stack : ndarray
            The input 4D stack with dimensions TYXC from which to calculate normalization values.
    percentiles : tuple, list of tuples, optional
            The percentile values (low, high) used to calculate the normalization ranges for each channel. If a single tuple
            is provided, it is applied to all channels. If a list of tuples is provided, each tuple is applied to the
            corresponding channel. If None, defaults to (0., 99.99) for each channel.
    ignore_gray_value : float, optional
            A gray value to potentially ignore during the calculation. This parameter is provided for interface consistency
            but is not utilized in the current implementation (default is 0.).

    Returns
    -------
    list of tuples
            A list where each tuple contains the (minimum, maximum) values for normalizing each channel based on the specified
            percentiles.

    Raises
    ------
    AssertionError
            If the input stack does not have 4 dimensions, or if the length of the `percentiles` list does not match the number
            of channels in the stack.

    Notes
    -----
    - The function assumes the input stack is in TYXC format, where T is the time dimension, Y and X are spatial dimensions,
      and C is the channel dimension.
    - Memory management via `gc.collect()` is employed after calculating normalization values for each channel to mitigate
      potential memory issues with large datasets.

    Examples
    --------
    >>> stack = np.random.rand(5, 100, 100, 3)  # Example 4D stack with 3 channels
    >>> normalization_values = get_stack_normalization_values(stack, percentiles=((1, 99), (2, 98), (0, 100)))
    # Calculates normalization ranges for each channel using the specified percentiles.

    """

    assert (
        stack.ndim == 4
    ), f"Wrong number of dimensions for the stack, expect TYXC (4) got {stack.ndim}."
    if percentiles is None:
        percentiles = [(0.0, 99.99)] * stack.shape[-1]
    elif isinstance(percentiles, tuple):
        percentiles = [percentiles] * stack.shape[-1]
    elif isinstance(percentiles, list):
        assert (
            len(percentiles) == stack.shape[-1]
        ), f"Mismatch between the provided percentiles and the number of channels {stack.shape[-1]}. If you meant to apply the same percentiles to all channels, please provide a single tuple."

    values = []
    for c in range(stack.shape[-1]):
        perc = percentiles[c]
        mi = np.nanpercentile(stack[:, :, :, c].flatten(), perc[0], keepdims=True)[0]
        ma = np.nanpercentile(stack[:, :, :, c].flatten(), perc[1], keepdims=True)[0]
        values.append(tuple((mi, ma)))
        gc.collect()

    return values


def normalize_per_channel(
    X,
    normalization_percentile_mode=True,
    normalization_values=[0.1, 99.99],
    normalization_clipping=False,
):
    """
    Applies per-channel normalization to a list of multi-channel images.

    This function normalizes each channel of every image in the list `X` based on either percentile values
    or fixed min-max values. Optionally, it can also clip the normalized values to stay within the [0, 1] range.
    The normalization can be applied in a percentile mode, where the lower and upper bounds for normalization
    are determined based on the specified percentiles of the non-zero values in each channel.

    Parameters
    ----------
    X : list of ndarray
            A list of 3D numpy arrays, where each array represents a multi-channel image with dimensions
            (height, width, channels).
    normalization_percentile_mode : bool or list of bool, optional
            If True (or a list of True values), normalization bounds are determined by percentiles specified
            in `normalization_values` for each channel. If False, fixed `normalization_values` are used directly.
            Default is True.
    normalization_values : list of two floats or list of lists of two floats, optional
            The percentile values [lower, upper] used for normalization in percentile mode, or the fixed
            min-max values [min, max] for direct normalization. Default is [0.1, 99.99].
    normalization_clipping : bool or list of bool, optional
            Determines whether to clip the normalized values to the [0, 1] range for each channel. Default is False.

    Returns
    -------
    list of ndarray
            The list of normalized multi-channel images.

    Raises
    ------
    AssertionError
            If the input images do not have a channel dimension, or if the lengths of `normalization_values`,
            `normalization_clipping`, and `normalization_percentile_mode` do not match the number of channels.

    Notes
    -----
    - The normalization is applied in-place, modifying the input list `X`.
    - This function is designed to handle multi-channel images commonly used in image processing and
      computer vision tasks, particularly when different channels require separate normalization strategies.

    Examples
    --------
    >>> X = [np.random.rand(100, 100, 3) for _ in range(5)]  # Example list of 5 RGB images
    >>> normalized_X = normalize_per_channel(X)
    # Normalizes each channel of each image based on the default percentile values [0.1, 99.99].
    """

    assert X[0].ndim == 3, "Channel axis does not exist. Abort."
    n_channels = X[0].shape[-1]
    if isinstance(normalization_percentile_mode, bool):
        normalization_percentile_mode = [normalization_percentile_mode] * n_channels
    if isinstance(normalization_clipping, bool):
        normalization_clipping = [normalization_clipping] * n_channels
    if len(normalization_values) == 2 and not isinstance(normalization_values[0], list):
        normalization_values = [normalization_values] * n_channels

    assert len(normalization_values) == n_channels
    assert len(normalization_clipping) == n_channels
    assert len(normalization_percentile_mode) == n_channels

    X_normalized = []
    for i in range(len(X)):
        x = X[i].copy()
        loc_i, loc_j, loc_c = np.where(x == 0.0)
        norm_x = np.zeros_like(x, dtype=np.float32)
        for k in range(x.shape[-1]):
            chan = x[:, :, k].copy()
            if not np.all(chan.flatten() == 0):
                if normalization_percentile_mode[k]:
                    min_val = np.nanpercentile(
                        chan[chan != 0.0].flatten(), normalization_values[k][0]
                    )
                    max_val = np.nanpercentile(
                        chan[chan != 0.0].flatten(), normalization_values[k][1]
                    )
                else:
                    min_val = normalization_values[k][0]
                    max_val = normalization_values[k][1]

                clip_option = normalization_clipping[k]
                norm_x[:, :, k] = normalize_mi_ma(
                    chan.astype(np.float32).copy(),
                    min_val,
                    max_val,
                    clip=clip_option,
                    eps=1e-20,
                    dtype=np.float32,
                )
            else:
                norm_x[:, :, k] = 0.0
        norm_x[loc_i, loc_j, loc_c] = 0.0
        X_normalized.append(norm_x.copy())

    return X_normalized
