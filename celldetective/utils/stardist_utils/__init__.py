from pathlib import Path
from typing import Union
import os

os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np


def _prep_stardist_model(
    model_name: str, path: Union[str, Path], use_gpu: bool = False, scale: float = 1
):
    """
    Prepares and loads a StarDist2D model for segmentation tasks.

    This function initializes a StarDist2D model with the specified parameters, sets GPU usage if desired,
    and allows scaling to adapt the model for specific applications.

    Parameters
    ----------
    model_name : str
            The name of the StarDist2D model to load. This name should match the model saved in the specified path.
    path : str
            The directory where the model is stored.
    use_gpu : bool, optional
            If `True`, the model will be configured to use GPU acceleration for computations. Default is `False`.
    scale : int or float, optional
            A scaling factor for the model. This can be used to adapt the model for specific image resolutions.
            Default is `1`.

    Returns
    -------
    tuple
            - model : StarDist2D
                    The loaded StarDist2D model configured with the specified parameters.
            - scale_model : int or float
                    The scaling factor passed to the function.

    Notes
    -----
    - Ensure the StarDist2D package is installed and the model files are correctly stored in the provided path.
    - GPU support depends on the availability of compatible hardware and software setup.
    """

    try:
        from stardist.models import StarDist2D
    except ImportError as e:
        raise RuntimeError(
            "StarDist is not installed. Please install it to use this feature.\n"
            "You can install the full package with: pip install celldetective[all]"
        ) from e

    model = StarDist2D(None, name=model_name, basedir=path)
    model.config.use_gpu = use_gpu
    model.use_gpu = use_gpu

    scale_model = scale

    print(f"StarDist model {model_name} successfully loaded...")
    return model, scale_model


def _segment_image_with_stardist_model(
    img, model=None, return_details=False, channel_axis=-1
):
    """
    Segments an input image using a StarDist model.

    This function applies a preloaded StarDist model to segment an input image and returns the resulting labeled mask.
    Optionally, additional details about the segmentation can also be returned.

    Parameters
    ----------
    img : ndarray
            The input image to be segmented. It is expected to have a channel axis specified by `channel_axis`.
    model : StarDist2D, optional
            A preloaded StarDist model instance used for segmentation.
    return_details : bool, optional
            Whether to return additional details from the model alongside the labeled mask. Default is `False`.
    channel_axis : int, optional
            The axis of the input image that represents the channels. Default is `-1` (channel-last format).

    Returns
    -------
    ndarray
            A labeled mask of the same spatial dimensions as the input image, with segmented regions assigned unique
            integer labels. The dtype of the mask is `uint16`.
    tuple of (ndarray, dict), optional
            If `return_details` is `True`, returns a tuple where the first element is the labeled mask and the second
            element is a dictionary containing additional details about the segmentation.

    Notes
    -----
    - The `img` array is internally rearranged to move the specified `channel_axis` to the last dimension to comply
      with the StarDist model's input requirements.
    - Ensure the provided `model` is a properly initialized StarDist model instance.
    - The model automatically determines the number of tiles (`n_tiles`) required for processing large images.
    """

    if channel_axis != -1:
        img = np.moveaxis(img, channel_axis, -1)

    lbl, details = model.predict_instances(
        img, n_tiles=model._guess_n_tiles(img), show_tile_progress=False, verbose=False
    )
    if not return_details:
        return lbl.astype(np.uint16)
    else:
        return lbl.astype(np.uint16), details
