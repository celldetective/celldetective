import collections

import numpy as np


def consume(iterator):
    """
    adapted from https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py
    """
    collections.deque(iterator, maxlen=0)


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    adapted from https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = "STCZYX"
    assert axes is not None,ValueError("axis cannot be None.")
    axes = str(axes).upper()
    consume(a in allowed for a in axes)
    disallowed is None or consume(a not in disallowed for a in axes)
    consume(axes.count(a) == 1 for a in axes)
    length is None or len(axes) == length
    return (axes, allowed) if return_allowed else axes


def axes_dict(axes):
    """
    adapted from https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes, return_allowed=True)
    return {a: None if axes.find(a) == -1 else axes.find(a) for a in allowed}
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])


def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    adapted from https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i, a in enumerate(fr):
            if (a not in to) and (x.shape[i] == 1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a, "")
        x = x[tuple(slices)]
        # add dummy axes present in 'to'
        for i, a in enumerate(to):
            if a not in fr:
                # add singleton axis
                x = np.expand_dims(x, -1)
                fr += a

    if set(fr) != set(to):
        _adjusted = (
            "(adjusted to %s and %s) " % (x.shape, fr) if adjust_singletons else ""
        )
        raise ValueError(
            "image with shape %s and axes %s %snot compatible with target axes %s."
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])


def estimate_unreliable_edge(activation_protocol=[["gauss", 2], ["std", 4]]):
    """
    Safely estimate the distance to the edge of an image in which the filtered image values can be artefactual.

    Parameters
    ----------
    activation_protocol : list of list, optional
            A list of lists, where each sublist contains a string naming the filter function, followed by its arguments (usually a kernel size).
            Default is [['gauss', 2], ['std', 4]].

    Returns
    -------
    int or None
            The sum of the kernel sizes in the activation protocol if the protocol
            is not empty. Returns None if the activation protocol is empty.

    Notes
    -----
    This function assumes that the second element of each sublist in the
    activation protocol is a kernel size.

    Examples
    --------
    >>> estimate_unreliable_edge([['gauss', 2], ['std', 4]])
    6
    >>> estimate_unreliable_edge([])
    None
    """

    if activation_protocol == []:
        return None
    else:
        edge = 0
        for fct in activation_protocol:
            if isinstance(fct[1], (int, np.int_)) and not fct[0] == "invert":
                edge += fct[1]
        return edge


def unpad(img, pad):
    """
    Remove padding from an image.

    This function removes the specified amount of padding from the borders
    of an image. The padding is assumed to be the same on all sides.

    Parameters
    ----------
    img : ndarray
            The input image from which the padding will be removed.
    pad : int
            The amount of padding to remove from each side of the image.

    Returns
    -------
    ndarray
            The image with the padding removed.

    Raises
    ------
    ValueError
            If `pad` is greater than or equal to half of the smallest dimension
            of `img`.

    See Also
    --------
    numpy.pad : Pads an array.

    Notes
    -----
    This function assumes that the input image is a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.array([[0, 0, 0, 0, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 1, 1, 1, 0],
    ...                 [0, 0, 0, 0, 0]])
    >>> unpad(img, 1)
    array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]])
    """

    return img[pad:-pad, pad:-pad]


def mask_edges(binary_mask, border_size):
    """
    Mask the edges of a binary mask.

    This function sets the edges of a binary mask to False, effectively
    masking out a border of the specified size.

    Parameters
    ----------
    binary_mask : ndarray
            A 2D binary mask array where the edges will be masked.
    border_size : int
            The size of the border to mask (set to False) on all sides.

    Returns
    -------
    ndarray
            The binary mask with the edges masked out.

    Raises
    ------
    ValueError
            If `border_size` is greater than or equal to half of the smallest
            dimension of `binary_mask`.

    Notes
    -----
    This function assumes that the input `binary_mask` is a 2D array. The
    input mask is converted to a boolean array before masking the edges.

    Examples
    --------
    >>> import numpy as np
    >>> binary_mask = np.array([[1, 1, 1, 1, 1],
    ...                         [1, 1, 1, 1, 1],
    ...                         [1, 1, 1, 1, 1],
    ...                         [1, 1, 1, 1, 1],
    ...                         [1, 1, 1, 1, 1]])
    >>> mask_edges(binary_mask, 1)
    array([[False, False, False, False, False],
               [False,  True,  True,  True, False],
               [False,  True,  True,  True, False],
               [False,  True,  True,  True, False],
               [False, False, False, False, False]])
    """

    binary_mask = binary_mask.astype(bool)
    binary_mask[:border_size, :] = False
    binary_mask[(binary_mask.shape[0] - border_size) :, :] = False
    binary_mask[:, :border_size] = False
    binary_mask[:, (binary_mask.shape[1] - border_size) :] = False

    return binary_mask


def _estimate_scale_factor(spatial_calibration, required_spatial_calibration):
    """
    Estimates the scale factor needed to adjust spatial calibration to a required value.

    This function calculates the scale factor by which spatial dimensions (e.g., in microscopy images)
    should be adjusted to align with a specified calibration standard. This is particularly useful when
    preparing data for analysis with models trained on data of a specific spatial calibration.

    Parameters
    ----------
    spatial_calibration : float or None
            The current spatial calibration factor of the data, expressed as units per pixel (e.g., micrometers per pixel).
            If None, indicates that the current spatial calibration is unknown or unspecified.
    required_spatial_calibration : float or None
            The spatial calibration factor required for compatibility with the model or analysis standard, expressed
            in the same units as `spatial_calibration`. If None, indicates no adjustment is required.

    Returns
    -------
    float or None
            The scale factor by which the current data should be rescaled to match the required spatial calibration,
            or None if no scaling is necessary or if insufficient information is provided.

    Notes
    -----
    - A scale factor close to 1 (within a tolerance defined by `epsilon`) indicates that no significant rescaling
      is needed, and the function returns None.
    - The function issues a warning if a significant rescaling is necessary, indicating the scale factor to be applied.

    Examples
    --------
    >>> scale_factor = _estimate_scale_factor(spatial_calibration=0.5, required_spatial_calibration=0.25)
    # Each frame will be rescaled by a factor 2.0 to match with the model training data...

    >>> scale_factor = _estimate_scale_factor(spatial_calibration=None, required_spatial_calibration=0.25)
    # Returns None due to insufficient information about current spatial calibration.
    """

    if (required_spatial_calibration is not None) * (spatial_calibration is not None):
        scale = spatial_calibration / required_spatial_calibration
    else:
        scale = None

    epsilon = 0.05
    if scale is not None:
        if not np.all([scale >= (1 - epsilon), scale <= (1 + epsilon)]):
            print(
                f"Each frame will be rescaled by a factor {scale} to match with the model training data..."
            )
        else:
            scale = None
    return scale


def threshold_image(
    img,
    min_threshold,
    max_threshold,
    foreground_value=255.0,
    fill_holes=True,
    edge_exclusion=None,
):
    """

    Threshold the input image to create a binary mask.

    Parameters
    ----------
    img : ndarray
            The input image to be thresholded.
    min_threshold : float
            The minimum threshold value.
    max_threshold : float
            The maximum threshold value.
    foreground_value : float, optional
            The value assigned to foreground pixels in the binary mask. Default is 255.
    fill_holes : bool, optional
            Whether to fill holes in the binary mask. If True, the binary mask will be processed to fill any holes.
            If False, the binary mask will not be modified. Default is True.

    Returns
    -------
    ndarray
            The binary mask after thresholding.

    Notes
    -----
    This function applies a threshold to the input image to create a binary mask. Pixels with values within the specified
    threshold range are considered as foreground and assigned the `foreground_value`, while pixels outside the range are
    considered as background and assigned 0. If `fill_holes` is True, the binary mask will be processed to fill any holes
    using morphological operations.

    Examples
    --------
    >>> image = np.random.rand(256, 256)
    >>> binary_mask = threshold_image(image, 0.2, 0.8, foreground_value=1., fill_holes=True)

    """
    from scipy import ndimage as ndi

    binary = np.zeros_like(img).astype(bool)
    binary[img == img] = (
        (img[img == img] >= min_threshold)
        * (img[img == img] <= max_threshold)
        * foreground_value
    )
    if isinstance(edge_exclusion, (int, np.int_)):
        binary = mask_edges(binary, edge_exclusion)
    if fill_holes:
        binary = ndi.binary_fill_holes(binary.astype(int))
    return binary
