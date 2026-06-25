from pathlib import Path
from typing import Union, Optional, Any, Tuple, Dict
import logging
import os

logger = logging.getLogger("celldetective")

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

    logger.info(f"StarDist model {model_name} successfully loaded...")
    return model, scale_model


# Largest frame (in pixels) we push through StarDist in a single forward pass.
# Below this we never tile, so StarDist's receptive-field probe is never invoked.
_MAX_SINGLE_PASS_PIXELS = 12_000_000  # ~3500 x 3500


def _analytic_tile_overlap(model: Any) -> Tuple[int, int]:
    """
    Estimate the StarDist tile overlap (receptive field, in px) for the Y/X axes
    WITHOUT running ``model._compute_receptive_field``.

    StarDist measures the overlap by running two forward passes on a ``grid*128``
    probe image and, if the impulse response is empty on any axis, rebuilding an
    *untrained* model and recursing (stardist/models/base.py:_compute_receptive_field).
    For models with a large grid/depth (e.g. grid=8, depth=4 -> effective RF depth 7)
    this probe pins the CPU at 100% and can recurse without terminating, which
    surfaces as segmentation "stuck at 0% forever" on the first frame the moment
    tiling is enabled.

    We instead use csbdeep's closed-form U-Net overlap table (extrapolated past
    depth 5) and round up to the network block size. Over-estimating is always safe:
    it only makes tiles overlap more.
    """
    from csbdeep.internals.predict import tile_overlap

    cfg = model.config
    grid = tuple(cfg.grid)
    kernel = tuple(cfg.unet_kernel_size)
    pool = tuple(cfg.unet_pool)
    div_by = model._axes_div_by("YX")  # analytic, no forward pass

    overlaps = []
    for ax in range(2):
        # the output grid acts like extra pooling stages on top of the U-Net depth
        eff_depth = int(cfg.unet_n_depth + np.log2(grid[ax]))
        k, p = int(kernel[ax]), int(pool[ax])
        try:
            rf = tile_overlap(eff_depth, k, p)
        except ValueError:
            # csbdeep's table maxes out at depth 5; extrapolate rf(n) ~= 2*rf(n-1)+2
            rf = tile_overlap(5, k, p)
            for _ in range(eff_depth - 5):
                rf = 2 * rf + 2
        block = int(div_by[ax])
        rf = int(np.ceil(rf / block) * block) + block  # round up to a block, +1 block margin
        overlaps.append(rf)
    return overlaps[0], overlaps[1]


def _seed_tile_overlap(model: Any) -> None:
    """
    Pre-fill ``model._tile_overlap`` with an analytic estimate so StarDist never runs
    its slow / possibly non-terminating ``_compute_receptive_field`` probe.
    """
    if getattr(model, "_tile_overlap", None) is None:
        oy, ox = _analytic_tile_overlap(model)
        # same structure _compute_receptive_field returns: (before, after) per spatial axis
        model._tile_overlap = [(oy, oy), (ox, ox)]


def _get_safe_n_tiles(img: np.ndarray, model: Any) -> Tuple[int, ...]:
    """
    Choose an n_tiles that never triggers StarDist's pathological receptive-field
    probe. If the frame fits in one forward pass we return all-ones (no tiling, so
    the probe is never reached). Otherwise we seed an analytic tile overlap first,
    then derive a tile count that keeps each tile strictly larger than 2 * overlap.
    """
    h, w = img.shape[:2]

    # Small/medium frames: single pass. With prod(n_tiles) == 1, StarDist never calls
    # _axes_tile_overlap -> _compute_receptive_field, so it cannot hang here.
    if h * w <= _MAX_SINGLE_PASS_PIXELS:
        return tuple([1] * img.ndim)

    # Large frames: must tile. Seed the overlap analytically so the probe is skipped
    # both here and inside StarDist's own tiling setup.
    _seed_tile_overlap(model)
    overlap_y, overlap_x = _analytic_tile_overlap(model)

    try:
        n_tiles = list(model._guess_n_tiles(img))
    except Exception:
        n_tiles = [1] * img.ndim

    # Tile size must be strictly larger than 2 * overlap; keep a 10 px cushion.
    max_n_tiles_y = max(1, h // (2 * overlap_y + 10))
    n_tiles[0] = max(1, min(n_tiles[0], max_n_tiles_y))

    max_n_tiles_x = max(1, w // (2 * overlap_x + 10))
    n_tiles[1] = max(1, min(n_tiles[1], max_n_tiles_x))

    return tuple(n_tiles)


def _segment_image_with_stardist_model(
    img: np.ndarray,
    model: Any = None,
    return_details: bool = False,
    channel_axis: int = -1,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
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

    # Pad image if smaller than train_patch_size (centered constant padding)
    h, w = img.shape[:2]
    train_patch_size = getattr(model.config, "train_patch_size", (256, 256))
    patch_h, patch_w = train_patch_size[0], train_patch_size[1]
    
    from celldetective.utils.image_transforms import pad_to_patch_size
    img, _, padded = pad_to_patch_size(img, None, patch_h, patch_w)

    n_tiles = _get_safe_n_tiles(img, model)
    logger.debug(f"Predicting instances with StarDist. Input shape: {img.shape}, range: [{img.min()}, {img.max()}]. n_tiles: {n_tiles}")
    lbl, details = model.predict_instances(
        img, n_tiles=n_tiles, show_tile_progress=False, verbose=False
    )

    if padded:
        pad_h = max(0, patch_h - h)
        pad_w = max(0, patch_w - w)
        pad_h_top = pad_h // 2
        pad_w_left = pad_w // 2

        # Crop back the predicted label mask to original size (H, W)
        lbl = lbl[pad_h_top:pad_h_top+h, pad_w_left:pad_w_left+w]
        
        # Adjust details coordinate values if return_details is True
        if return_details and details is not None:
            if 'points' in details:
                points = details['points'] - np.array([pad_h_top, pad_w_left])
                valid = (points[:, 0] >= 0) & (points[:, 0] < h) & (points[:, 1] >= 0) & (points[:, 1] < w)
                details['points'] = points[valid]
                if 'prob' in details:
                    details['prob'] = details['prob'][valid]
                if 'coord' in details:
                    coord = details['coord'] - np.array([pad_h_top, pad_w_left])
                    details['coord'] = coord[valid]

    if not return_details:
        return lbl.astype(np.uint16)
    else:
        return lbl.astype(np.uint16), details
