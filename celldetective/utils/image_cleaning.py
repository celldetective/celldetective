from typing import Union

import numpy as np


def _fix_no_contrast(frames: np.ndarray, value: Union[float, int] = 1):
    """
    Ensures that frames with no contrast (i.e., containing only a single unique value) are adjusted.

    This function modifies frames that lack contrast by adding a small value to the first pixel in
    the affected frame. This prevents downstream issues in image processing pipelines that require
    a minimum level of contrast.

    Parameters
    ----------
    frames : ndarray
            A 3D array of shape `(H, W, N)`, where:
            - `H` is the height of the frame,
            - `W` is the width of the frame,
            - `N` is the number of frames or channels.
            Each frame (or channel) is independently checked for contrast.
    value : int or float, optional
            The value to add to the first pixel (`frames[0, 0, k]`) of any frame that lacks contrast.
            Default is `1`.

    Returns
    -------
    ndarray
            The modified `frames` array, where frames with no contrast have been adjusted.

    Notes
    -----
    - A frame is determined to have "no contrast" if all its pixel values are identical.
    - Only the first pixel (`[0, 0, k]`) of a no-contrast frame is modified, leaving the rest
      of the frame unchanged.
    """

    for k in range(frames.shape[2]):
        unique_values = np.unique(frames[:, :, k])
        if len(unique_values) == 1:
            frames[0, 0, k] += value
    return frames


def interpolate_nan_multichannel(frames):
    frames = np.moveaxis(
        [interpolate_nan(frames[:, :, c].copy()) for c in range(frames.shape[-1])],
        0,
        -1,
    )
    return frames


def interpolate_nan(img, method="nearest"):
    """
    Interpolate NaN on single channel array 2D
    """
    from scipy.interpolate import griddata

    if np.all(img == 0):
        return img

    if np.any(img.flatten() != img.flatten()):
        # then need to interpolate
        x_grid, y_grid = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        mask = [~np.isnan(img)][0]
        x = x_grid[mask].reshape(-1)
        y = y_grid[mask].reshape(-1)
        points = np.array([x, y]).T
        values = img[mask].reshape(-1)
        interp_grid = griddata(points, values, (x_grid, y_grid), method=method)
        return interp_grid
    else:
        return img
