import os
from pathlib import Path
from typing import Union
import numpy as np

from celldetective.utils.image_transforms import axes_check_and_normalize, move_image_axes

try:
    from tifffile import imwrite as imsave
except ImportError:
    from tifffile import imsave
import warnings


def remove_file_if_exists(file: Union[str, Path]):
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as e:
            print(e)


def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.
    adapted from https://github.com/CSBDeep/CSBDeep/blob/main/csbdeep/utils/utils.py

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    axes = axes_check_and_normalize(axes, img.ndim, disallowed="S")

    # convert to imagej-compatible data type
    t = img.dtype
    if "float" in t.name:
        t_new = np.float32
    elif "uint" in t.name:
        t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif "int" in t.name:
        t_new = np.int16
    else:
        t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn(
            "Converting data type from '%s' to ImageJ-compatible '%s'."
            % (t, np.dtype(t_new))
        )

    # move axes to correct positions for imagej
    img = move_image_axes(img, axes, "TZCYX", True)

    imsave_kwargs["imagej"] = True
    imsave(file, img, **imsave_kwargs)
