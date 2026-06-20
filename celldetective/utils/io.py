import os
import logging
from pathlib import Path
from typing import Union, Any
import numpy as np

logger = logging.getLogger("celldetective")

from celldetective.utils.image_transforms import (
    axes_check_and_normalize,
    move_image_axes,
)

try:
    from tifffile import imwrite as imsave
except ImportError:
    from tifffile import imsave
import warnings


def remove_file_if_exists(file: Union[str, Path]):
    """
    Remove a file if it exists.

    Parameters
    ----------
    file : str or Path
        Path to the file to remove.
    """
    if os.path.exists(file):
        try:
            os.remove(file)
        except Exception as e:
            logger.warning(f"Failed to remove file {file}: {e}")


def save_tiff_imagej_compatible(
    file: Union[str, Path], img: np.ndarray, axes: str, **imsave_kwargs: Any
) -> None:
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

    # Write atomically: cancelling a job hard-kills the worker process, which
    # must never leave a half-written (truncated/corrupt) TIFF on disk. Write to
    # a temp file in the same directory, then os.replace() it into place — atomic
    # on the same filesystem on both POSIX and Windows. If the worker is killed
    # before the replace, the final file is simply left untouched (a missing
    # frame, which fix_missing_labels can repair) rather than corrupted. The temp
    # name does not end in .tif, so the label loaders' "*.tif" globs ignore any
    # leftover.
    file = os.fspath(file)
    tmp_file = f"{file}.{os.getpid()}.tmp"
    try:
        imsave(tmp_file, img, **imsave_kwargs)
        os.replace(tmp_file, file)
    except BaseException:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass
        raise
