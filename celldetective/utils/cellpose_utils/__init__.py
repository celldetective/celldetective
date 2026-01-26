from pathlib import Path
from typing import Union, Optional

import numpy as np


def _segment_image_with_cellpose_model(
    img,
    model=None,
    diameter=None,
    cellprob_threshold=None,
    flow_threshold=None,
    channel_axis=-1,
):
    """
    Segments an input image using a Cellpose model.

    This function applies a preloaded Cellpose model to segment an input image and returns the resulting labeled mask.
    The image is rearranged into the format expected by the Cellpose model, with the specified channel axis moved to the first dimension.

    Parameters
    ----------
    img : ndarray
            The input image to be segmented. It is expected to have a channel axis specified by `channel_axis`.
    model : CellposeModel, optional
            A preloaded Cellpose model instance used for segmentation.
    diameter : float, optional
            The diameter of objects to segment. If `None`, the model's default diameter is used.
    cellprob_threshold : float, optional
            The threshold for the probability of cells used during segmentation. If `None`, the default threshold is used.
    flow_threshold : float, optional
            The threshold for flow error during segmentation. If `None`, the default threshold is used.
    channel_axis : int, optional
            The axis of the input image that represents the channels. Default is `-1` (channel-last format).

    Returns
    -------
    ndarray
            A labeled mask of the same spatial dimensions as the input image, with segmented regions assigned unique
            integer labels. The dtype of the mask is `uint16`.

    Notes
    -----
    - The `img` array is internally rearranged to move the specified `channel_axis` to the first dimension to comply
      with the Cellpose model's input requirements.
    - Ensure the provided `model` is a properly initialized Cellpose model instance.
    - Parameters `diameter`, `cellprob_threshold`, and `flow_threshold` allow fine-tuning of the segmentation process.
    """

    img = np.moveaxis(img, channel_axis, 0)
    lbl, _, _ = model.eval(
        img,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        channels=None,
        normalize=False,
        model_loaded=True,
    )

    return lbl.astype(np.uint16)


def _prep_cellpose_model(
    model_name: str,
    path: Union[str, Path],
    use_gpu: bool = False,
    n_channels: int = 2,
    scale: Optional[float] = None,
):
    """
    Prepares and loads a Cellpose model for segmentation tasks.

    This function initializes a Cellpose model with the specified parameters, configures GPU usage if available,
    and calculates or applies a scaling factor for the model based on image resolution.

    Parameters
    ----------
    model_name : str
            The name of the pretrained Cellpose model to load.
    path : str
            The directory where the model is stored.
    use_gpu : bool, optional
            If `True`, the model will use GPU acceleration for computations. Default is `False`.
    n_channels : int, optional
            The number of input channels expected by the model. Default is `2`.
    scale : float, optional
            A scaling factor to adjust the model's output to match the image resolution. If not provided, the scale is
            automatically calculated based on the model's diameter parameters.

    Returns
    -------
    tuple
            - model : CellposeModel
                    The loaded Cellpose model configured with the specified parameters.
            - scale_model : float
                    The scaling factor applied to the model, calculated or provided.

    Notes
    -----
    - Ensure the Cellpose package is installed and the model files are correctly stored in the provided path.
    - GPU support depends on the availability of compatible hardware and software setup.
    - The scale is calculated as `(diam_mean / diam_labels)` if `scale` is not provided, where `diam_mean` and
      `diam_labels` are attributes of the model.
    """

    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Torch is not installed. Please install it to use this feature.\n"
            "You can install the full package with: pip install celldetective[all]\n"
        ) from e

    if not use_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    try:
        from cellpose.models import CellposeModel
    except ImportError as e:
        raise RuntimeError(
            "Cellpose is not installed. Please install it to use this feature.\n"
            "You can install the full package with: pip install celldetective[all]\n"
            "Or specifically: pip install celldetective[cellpose]"
        ) from e

    try:
        model = CellposeModel(
            gpu=use_gpu,
            device=device,
            pretrained_model=path + model_name,
            model_type=None,
            nchan=n_channels,
        )  # diam_mean=30.0,
    except AssertionError as e:
        if use_gpu:
            print(
                f"[WARNING] Could not load Cellpose model with GPU ({e}). Retrying with CPU..."
            )
            device = torch.device("cpu")
            model = CellposeModel(
                gpu=False,
                device=device,
                pretrained_model=path + model_name,
                model_type=None,
                nchan=n_channels,
            )
        else:
            raise e
    if scale is None:
        scale_model = model.diam_mean / model.diam_labels
    else:
        scale_model = scale * model.diam_mean / model.diam_labels

    print(f"Cell size in model: {model.diam_mean} pixels...")
    print(f"Cell size in training set: {model.diam_labels} pixels...")
    print(f"Rescaling factor to apply: {scale_model}...")

    print(f"Cellpose model {model_name} successfully loaded...")
    return model, scale_model
