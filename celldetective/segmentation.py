"""
Segmentation Module
===================

This module handles the segmentation of cells and nuclei in microscopy images.
It provides a unified interface for various segmentation algorithms, including deep learning-based methods and classical image processing techniques.

Key Features
------------
-   **Deep Learning Integration**: Supports `StarDist` and `Cellpose` for robust instance segmentation.
-   **Classical Methods**: Includes thresholding and watershed-based segmentation.
-   **Preprocessing Integration**: Often works in tandem with the `preprocessing` module to prepare images for segmentation.

Main Functions
--------------
-   `process_image` / `segment_image`: Main wrapper functions to execute segmentation based on provided configuration.
-   `stardist_segmentation`: Wrapper for StarDist segmentation.
-   `cp_segmentation`: Wrapper for Cellpose segmentation.
-   `threshold_segmentation`: Implements threshold-based segmentation logic.

Configuration
-------------
Segmentation parameters are typically passed via a dictionary or configuration object, specifying the method (e.g., 'stardist', 'cellpose') and its specific parameters (e.g., probability threshold, diameter).
"""

import json
import os
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from celldetective.utils.model_loaders import (
    locate_segmentation_model,
    trained_cell_size_um,
)
from celldetective.utils.normalization import normalize_multichannel
from pathlib import Path
from tqdm import tqdm
from celldetective.utils.image_loaders import (
    locate_stack,
    locate_labels,
    _rearrange_multichannel_frame,
    zoom_multiframes,
    _extract_channel_indices,
)
from celldetective.utils.mask_cleaning import _check_label_dims, auto_correct_masks
from celldetective.utils.image_cleaning import (
    _fix_no_contrast,
    interpolate_nan_multichannel,
)

from celldetective.filters import filter_image
from celldetective.utils.stardist_utils import (
    _prep_stardist_model,
    _segment_image_with_stardist_model,
)
from celldetective.utils.cellpose_utils import (
    _segment_image_with_cellpose_model,
    _prep_cellpose_model,
)
from celldetective.utils.mask_transforms import _rescale_labels
from celldetective.utils.image_transforms import (
    estimate_unreliable_edge,
    _combined_scale_factor,
    threshold_image,
)
from celldetective.utils.data_cleaning import rename_intensity_column
from celldetective.utils.parsing import _get_normalize_kwargs_from_config

import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.exposure import match_histograms

import subprocess
import sys
from celldetective.log_manager import get_logger

logger = get_logger(__name__)

abs_path = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
)


class PreparedSegmentationModel:
    """
    A segmentation model loaded and configured for a given set of image channels.

    Built by :func:`prepare_segmentation_model` and consumed by
    :func:`segment_frame`. Holding the loaded model and its resolved channel
    mapping in one object means the expensive setup (reading
    ``config_input.json``, estimating the rescaling factor, instantiating the
    StarDist / Cellpose model) happens once and can then be reused across as many
    frames as needed - a whole stack, or a single frame picked in napari.

    Attributes
    ----------
    model : object
            The instantiated StarDist or Cellpose model.
    model_type : str
            Either ``"stardist"`` or ``"cellpose"``.
    scale_model : float or None
            Zoom factor applied to a frame before inference, or None when the
            image calibration already matches the model's.
    channels : list of str
            Channel names of the images that will be passed to :func:`segment_frame`.
    required_channels : list of str
            Channel names the model expects, in the order it expects them.
    channel_indices : list of int or None
            One entry per model input slot: the index into `channels` of the image
            channel feeding it, or None when the slot has no source. Matching is
            case-insensitive, and the same image channel may feed several slots.
            This is what :func:`segment_frame` transfers with.
    channel_intersection : list of str
            The channel names actually transferred, one per filled slot.
    none_channel_indices : numpy.ndarray
            Indices of required channels missing from the image; zeroed before inference.
    normalize_kwargs : dict
            Normalization settings read from the model configuration.
    diameter : float or None
            Cellpose object diameter. None for StarDist models.
    cellprob_threshold : float or None
            Cellpose cell probability threshold. None for StarDist models.
    flow_threshold : float or None
            Cellpose flow threshold. None for StarDist models.
    config_defaults : dict
            The inference parameters as the model configuration declares them, before
            any caller override. Lets a caller restore a model's own values without
            re-reading ``config_input.json``.
    use_gpu : bool
            The device choice this model was built under. :func:`segment_frame`
            re-applies it around inference, since a framework that defers creating
            its device context until the first prediction would otherwise see the
            caller's environment rather than this one.
    """

    __slots__ = (
        "model",
        "model_type",
        "scale_model",
        "channels",
        "required_channels",
        "channel_indices",
        "channel_intersection",
        "none_channel_indices",
        "normalize_kwargs",
        "diameter",
        "cellprob_threshold",
        "flow_threshold",
        "config_defaults",
        "use_gpu",
    )

    def __init__(
        self,
        model: Any,
        model_type: str,
        scale_model: Optional[float],
        channels: List[str],
        required_channels: List[str],
        channel_indices: List[Optional[int]],
        channel_intersection: List[str],
        none_channel_indices: np.ndarray,
        normalize_kwargs: Dict[str, Any],
        diameter: Optional[float] = None,
        cellprob_threshold: Optional[float] = None,
        flow_threshold: Optional[float] = None,
        config_defaults: Optional[Dict[str, Any]] = None,
        use_gpu: bool = True,
    ) -> None:
        self.model = model
        self.model_type = model_type
        self.scale_model = scale_model
        self.channels = channels
        self.required_channels = required_channels
        self.channel_indices = channel_indices
        self.channel_intersection = channel_intersection
        self.none_channel_indices = none_channel_indices
        self.normalize_kwargs = normalize_kwargs
        self.diameter = diameter
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.config_defaults = config_defaults if config_defaults is not None else {}
        self.use_gpu = use_gpu


@contextmanager
def _gpu_visibility(use_gpu: bool):
    """
    Expose or hide the GPU to the DL frameworks, then put the environment back.

    ``CUDA_VISIBLE_DEVICES`` is process-global, and both TensorFlow and Torch read
    it once, when they first initialise a device context. Leaving it set - as this
    module used to - meant that a single CPU-only call from the GUI silently pinned
    the whole process to the CPU for good. Entering this context around both model
    construction and inference gets the intended device wherever that context
    happens to be created, and leaves the caller's environment exactly as it was
    found.

    Parameters
    ----------
    use_gpu : bool
            Whether the GPU should be visible inside the block.
    """

    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def prepare_segmentation_model(
    model_name: str,
    channels: Optional[List[str]] = None,
    spatial_calibration: Optional[float] = None,
    use_gpu: bool = True,
    cellprob_threshold: Optional[float] = None,
    flow_threshold: Optional[float] = None,
    selected_channels: Optional[List[str]] = None,
    target_cell_size: Optional[float] = None,
    diameter: Optional[float] = None,
    use_stored_mapping: bool = False,
) -> Optional[PreparedSegmentationModel]:
    """

    Load a segmentation model and resolve it against a set of image channels.

    This is the setup half of :func:`segment`, split out so that the loaded model
    can be reused frame by frame instead of being rebuilt on every call.

    Parameters
    ----------
    model_name : str
            The name of the pre-trained segmentation model to use.
    channels : list or None, optional
            The names of the channels in the images to segment. When None, the
            model's own required channels are assumed, in order. Default is None.
    spatial_calibration : float or None, optional
            The spatial calibration factor of the images. If None, the calibration
            factor from the model configuration will be used. Default is None.
    use_gpu : bool, optional
            Whether to use GPU acceleration if available. Default is True.
    cellprob_threshold : float, optional
            Cell probability threshold for Cellpose mask computation. Default is None.
    flow_threshold : float, optional
            Flow threshold for Cellpose mask computation. Default is None.
    selected_channels : list or None, optional
            The experiment channels feeding the model's input slots, one per slot,
            in the model's own order; ``"None"`` leaves a slot blank. Overrides the
            ``selected_channels`` mapping stored in the model configuration by the
            channel-selection dialog. Default is None: the stored mapping when
            ``use_stored_mapping`` is set, the model's own ``channels`` list
            otherwise.
    target_cell_size : float or None, optional
            Typical object size in the images, in µm. Combined with the model's
            ``cell_size_um`` it rescales the images so objects match the size the
            model was trained on. Overrides the stored ``target_cell_size_um``.
            Default is None: the stored value when ``use_stored_mapping`` is set,
            no cell-size rescaling otherwise.
    diameter : float or None, optional
            Cellpose object diameter, in pixels. Overrides the value stored in the
            model configuration. Ignored for StarDist models. Default is None.
    use_stored_mapping : bool, optional
            Whether to fall back on the ``selected_channels`` and
            ``target_cell_size_um`` entries that the channel-selection dialog writes
            into the model's ``config_input.json``. That file lives in the installed
            package, shared by every experiment, so honouring it silently would let a
            mapping saved in the GUI change what this function returns for an
            unrelated call. The GUI opts in to get pipeline parity; library callers
            get the model's own channel list unless they ask. Default is False.

    Returns
    -------
    PreparedSegmentationModel or None
            The prepared model, or None if the model or its input configuration
            could not be located.

    Raises
    ------
    ValueError
            If none of the channels required by the model are present in `channels`.

    Examples
    --------
    >>> prepared = prepare_segmentation_model('model_name', channels=['brightfield_channel'])
    >>> labels = segment_frame(frame, prepared)

    """

    model_path = locate_segmentation_model(model_name)
    if model_path is None:
        logger.error(f"Could not locate model {model_name}. Aborting segmentation.")
        return None

    input_config = model_path + "config_input.json"
    if os.path.exists(input_config):
        with open(input_config) as config:
            logger.info("Loading input configuration from 'config_input.json'.")
            input_config = json.load(config)
    else:
        logger.error("Model input configuration could not be located...")
        return None

    # The channel-selection dialog stores the experiment channel feeding each of
    # the model's input slots as `selected_channels`, same length and order as
    # `channels`. Honour it exactly as SegmentCellDLProcess does, so calling this
    # function directly gives the same masks as running the pipeline.
    required_channels = input_config["channels"]
    if selected_channels is not None:
        required_channels = list(selected_channels)
    elif use_stored_mapping and "selected_channels" in input_config:
        required_channels = input_config["selected_channels"]

    # The docstring has always allowed channels=None; without this the channel
    # intersection below raises a TypeError instead of falling back.
    if channels is None:
        channels = list(required_channels)
    else:
        channels = list(channels)

    # One entry per model input slot: the index into `channels` of the image
    # channel feeding it, or None. Resolving per slot rather than intersecting the
    # two name lists means the same image channel can feed several slots - which
    # the channel-selection dialog allows and the pipeline handles, since
    # `_get_img_num_per_channel` gives every slot its own row - and it keeps the
    # case-insensitive matching of `_extract_channel_indices` instead of pairing
    # it with a case-sensitive membership test.
    channel_indices = _extract_channel_indices(channels, required_channels)
    channel_intersection = [channels[i] for i in channel_indices if i is not None]
    if len(channel_intersection) == 0:
        raise ValueError("None of the channels required by the model can be found in the images to segment... Abort.")

    required_spatial_calibration = input_config["spatial_calibration"]
    model_type = input_config["model_type"]

    normalize_kwargs = _get_normalize_kwargs_from_config(input_config)

    # Kept alongside the resolved values so a caller can go back to the model's
    # own settings without re-reading the configuration from disk.
    config_defaults: Dict[str, Any] = {}

    if model_type == "cellpose":
        # `.get`, not indexing: a hand-written or older configuration may omit
        # these, and a caller that passes them explicitly should not be stopped by
        # a default it never uses.
        config_defaults = {
            "diameter": input_config.get("diameter"),
            "cellprob_threshold": input_config.get("cellprob_threshold"),
            "flow_threshold": input_config.get("flow_threshold"),
        }
        if diameter is None:
            diameter = config_defaults["diameter"]
        if cellprob_threshold is None:
            cellprob_threshold = config_defaults["cellprob_threshold"]
        if flow_threshold is None:
            flow_threshold = config_defaults["flow_threshold"]
    else:
        diameter = None

    # Both corrections in one factor, mirroring SegmentCellDLProcess.
    # `cell_size_um` is what the model saw, `target_cell_size_um` what these
    # images hold. Read from the model configuration, never from a caller's
    # override: the override says what to ask Cellpose for, not what the network
    # was trained on.
    cell_size = trained_cell_size_um(input_config)

    if target_cell_size is None and use_stored_mapping:
        target_cell_size = input_config.get("target_cell_size_um")
    if target_cell_size is not None and target_cell_size <= 0:
        raise ValueError(
            f"The target cell size must be strictly positive, got {target_cell_size}."
        )

    scale = _combined_scale_factor(
        spatial_calibration,
        required_spatial_calibration,
        cell_size,
        target_cell_size,
    )

    logger.info(
        f"{spatial_calibration=} {required_spatial_calibration=} "
        f"{cell_size=} {target_cell_size=} Scale = {scale}..."
    )

    model = None
    scale_model = None
    with _gpu_visibility(use_gpu):
        if model_type == "stardist":
            model, scale_model = _prep_stardist_model(
                model_name, Path(model_path).parent, use_gpu=use_gpu, scale=scale
            )
        elif model_type == "cellpose":
            # `model_name` directly, as SegmentCellDLProcess does. Deriving it from
            # the path as `model_path.split("/")[-2]` raised IndexError on Windows,
            # where locate_segmentation_model returns os.sep-joined backslashes and
            # the split yields a single element.
            model, scale_model = _prep_cellpose_model(
                model_name,
                model_path,
                use_gpu=use_gpu,
                n_channels=len(required_channels),
                scale=scale,
            )

    if model is None:
        logger.error(f"Could not load model {model_name}. Aborting segmentation.")
        return None

    # Resolve the missing required channels once, rather than on every frame.
    none_channel_indices = np.array(
        [i for i, v in enumerate(channel_indices) if v is None], dtype=int
    )

    return PreparedSegmentationModel(
        model=model,
        model_type=model_type,
        scale_model=scale_model,
        channels=channels,
        required_channels=required_channels,
        channel_indices=channel_indices,
        channel_intersection=channel_intersection,
        none_channel_indices=none_channel_indices,
        normalize_kwargs=normalize_kwargs,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        config_defaults=config_defaults,
        use_gpu=use_gpu,
    )


def segment_frame(
    frame: np.ndarray,
    prepared: PreparedSegmentationModel,
) -> np.ndarray:
    """

    Segment a single frame with an already-prepared model.

    This is the per-frame half of :func:`segment`. It performs the channel
    transfer, normalization, rescaling and inference for one image, then rescales
    the resulting labels back to that frame's original dimensions.

    Parameters
    ----------
    frame : ndarray
            A single multichannel image, with shape (height, width, channels) -
            the channels being those named in ``prepared.channels``.
    prepared : PreparedSegmentationModel
            A model prepared by :func:`prepare_segmentation_model`.

    Returns
    -------
    ndarray
            The segmented labels, with shape (height, width).

    Examples
    --------
    >>> prepared = prepare_segmentation_model('model_name', channels=['brightfield_channel'])
    >>> labels = segment_frame(stack[0], prepared)

    """

    required_channels = prepared.required_channels

    frame = _rearrange_multichannel_frame(frame).astype(float)

    # Same contract as segment(): the frame's channels are the ones named in
    # prepared.channels. Checked here too, because the mapping is resolved
    # against those names and not against this frame -- without the check a
    # too-narrow frame raises deep inside the channel transfer, and a wider one
    # silently feeds arbitrary planes to the model.
    if frame.shape[-1] != len(prepared.channels):
        raise ValueError(
            f"The frame has {frame.shape[-1]} channel(s) but the model was "
            f"prepared for {len(prepared.channels)}: {prepared.channels}."
        )

    # Walk the slots, not the matched names: a name lookup collapses a mapping
    # that feeds one image channel into several slots down to the first of them,
    # leaving the rest silently black.
    frame_to_segment = np.zeros(
        (frame.shape[0], frame.shape[1], len(required_channels))
    ).astype(float)
    for slot, source in enumerate(prepared.channel_indices):
        if source is not None:
            frame_to_segment[:, :, slot] = frame[:, :, source]
    frame = frame_to_segment
    template = frame.copy()

    frame = normalize_multichannel(frame, **prepared.normalize_kwargs)

    if prepared.scale_model is not None:
        frame = zoom_multiframes(frame, prepared.scale_model)

    frame = _fix_no_contrast(frame)
    frame = interpolate_nan_multichannel(frame)
    frame[:, :, prepared.none_channel_indices] = 0.0

    # Same device visibility as when the model was built: TensorFlow reads
    # CUDA_VISIBLE_DEVICES when it creates its device context, and whether that
    # happens in the StarDist constructor or on the first `predict` is an
    # implementation detail we should not be relying on.
    with _gpu_visibility(prepared.use_gpu):
        if prepared.model_type == "stardist":
            Y_pred = _segment_image_with_stardist_model(
                frame, model=prepared.model, return_details=False
            )
        elif prepared.model_type == "cellpose":
            Y_pred = _segment_image_with_cellpose_model(
                frame,
                model=prepared.model,
                diameter=prepared.diameter,
                cellprob_threshold=prepared.cellprob_threshold,
                flow_threshold=prepared.flow_threshold,
            )
        else:
            raise ValueError(f"Unknown model type {prepared.model_type}...")

    # `template` carries this frame's pre-rescaling dimensions, so the check is
    # made per frame rather than assuming every frame matches the first.
    if Y_pred.shape != template.shape[:2]:
        Y_pred = _rescale_labels(Y_pred, prepared.scale_model)

    return _check_label_dims(Y_pred, template=template)


def segment(
    stack: Union[np.ndarray, List[np.ndarray]],
    model_name: str,
    channels: Optional[List[str]] = None,
    spatial_calibration: Optional[float] = None,
    view_on_napari: bool = False,
    use_gpu: bool = True,
    channel_axis: int = -1,
    cellprob_threshold: Optional[float] = None,
    flow_threshold: Optional[float] = None,
    selected_channels: Optional[List[str]] = None,
    target_cell_size: Optional[float] = None,
    diameter: Optional[float] = None,
    use_stored_mapping: bool = False,
) -> Optional[np.ndarray]:
    """

    Segment objects in a stack using a pre-trained segmentation model.

    Parameters
    ----------
    stack : ndarray
            The input stack to be segmented, with shape (frames, height, width, channels).
    model_name : str
            The name of the pre-trained segmentation model to use.
    channels : list or None, optional
            The names of the channels in the stack. If None, assumes the channels are indexed from 0 to `stack.shape[-1] - 1`.
            Default is None.
    spatial_calibration : float or None, optional
            The spatial calibration factor of the stack. If None, the calibration factor from the model configuration will be used.
            Default is None.
    view_on_napari : bool, optional
            Whether to visualize the segmentation results using Napari. Default is False.
    use_gpu : bool, optional
            Whether to use GPU acceleration if available. Default is True.
    channel_axis : int, optional
                Channel axis in the input array. Default is the last (-1).
    cellprob_threshold : float, optional
                Cell probability threshold for Cellpose mask computation. Default is None.
    flow_threshold : float, optional
                Flow threshold for Cellpose mask computation. Default is None.
    selected_channels : list or None, optional
            The experiment channels feeding the model's input slots, one per slot, in
            the model's own order; ``"None"`` leaves a slot blank. Overrides the
            ``selected_channels`` mapping stored in the model configuration. Pass the
            model's own ``channels`` list to ignore the stored mapping entirely.
            Default is None: the stored mapping when ``use_stored_mapping`` is set,
            the model's own ``channels`` list otherwise.
    target_cell_size : float or None, optional
            Typical object size in the images, in µm, used together with the model's
            ``cell_size_um`` to rescale the images. Overrides the stored
            ``target_cell_size_um``. Default is None: the stored value when
            ``use_stored_mapping`` is set, no cell-size rescaling otherwise.
    diameter : float or None, optional
            Cellpose object diameter, in pixels, overriding the model configuration.
            Ignored for StarDist models. Default is None.
    use_stored_mapping : bool, optional
            Whether to fall back on the ``selected_channels`` and
            ``target_cell_size_um`` entries stored in the model's
            ``config_input.json`` by the channel-selection dialog. Set it to True to
            reproduce exactly what the pipeline does for the same model; leave it
            False to keep a call reproducible whatever was last set in the GUI.
            Default is False.

    Returns
    -------
    ndarray or None
            The segmented labels with shape (frames, height, width), or None if the
            model or its input configuration could not be located.

    Notes
    -----
    This function applies object segmentation to a stack of images using a pre-trained segmentation model. The stack is first
    preprocessed by normalizing the intensity values, rescaling the spatial dimensions, and applying the segmentation model.
    The resulting labels are returned as an ndarray with the same number of frames as the input stack.

    This is a thin wrapper over :func:`prepare_segmentation_model` (called once) and
    :func:`segment_frame` (called per frame). Reach for those two directly when the
    same model has to run on frames that do not arrive as one stack.

    .. versionchanged:: 1.5.4
            Two settings that ``SegmentCellDLProcess`` had always applied, and that
            this function had always ignored, can now be honoured here too, so the
            library and the pipeline return the same masks for the same model:

            - the ``selected_channels`` mapping stored in the model's
              ``config_input.json`` by the channel-selection dialog, which then takes
              precedence over the model's own ``channels`` list;
            - the ``cell_size_um`` / ``target_cell_size_um`` rescaling.

            Both come from the model directory, which is installed once and shared by
            every experiment, so applying them by default would let a mapping saved
            while working on one experiment change what this function returns for
            another. They are therefore opt-in: pass `use_stored_mapping=True` for
            pipeline parity, or `selected_channels` / `target_cell_size` to set them
            explicitly. Without either, the model's own ``channels`` list is used, as
            before.

    Examples
    --------
    >>> stack = np.random.rand(10, 256, 256, 3)
    >>> labels = segment(stack, 'model_name', channels=['channel_1', 'channel_2', 'channel_3'], spatial_calibration=0.5)

    """

    if channel_axis != -1:
        stack = np.moveaxis(stack, channel_axis, -1)

    if channels is not None:
        if len(channels) != stack.shape[-1]:
            raise ValueError(f"The channel names provided do not match with the expected number of channels in the stack: {stack.shape[-1]}.")

    prepared = prepare_segmentation_model(
        model_name,
        channels=channels,
        spatial_calibration=spatial_calibration,
        use_gpu=use_gpu,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        selected_channels=selected_channels,
        target_cell_size=target_cell_size,
        diameter=diameter,
        use_stored_mapping=use_stored_mapping,
    )
    if prepared is None:
        return None

    labels = np.array(
        [
            segment_frame(stack[t], prepared)
            for t in tqdm(range(len(stack)), desc="frame")
        ],
        dtype=int,
    )

    if view_on_napari:
        from celldetective.napari.utils import _view_on_napari

        _view_on_napari(tracks=None, stack=stack, labels=labels)

    return labels


def segment_from_thresholds(
    stack: np.ndarray,
    target_channel: int = 0,
    thresholds: Optional[List[Tuple[float, float]]] = None,
    view_on_napari: bool = False,
    equalize_reference: Optional[int] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    marker_min_distance: int = 30,
    marker_footprint_size: int = 20,
    marker_footprint: Optional[np.ndarray] = None,
    feature_queries: Optional[List[str]] = None,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Segments objects from a stack of images based on provided thresholds and optional image processing steps.

    This function applies instance segmentation to each frame in a stack of images. Segmentation is based on intensity
    thresholds, optionally preceded by image equalization and filtering. Identified objects can
    be distinguished by applying distance-based marker detection. The segmentation results can be optionally viewed in Napari.

    Parameters
    ----------
    stack : ndarray
            A 4D numpy array representing the image stack with dimensions (T, Y, X, C) where T is the
            time dimension and C the channel dimension.
    target_channel : int, optional
            The channel index to be used for segmentation (default is 0).
    thresholds : list of tuples, optional
            A list of tuples specifying intensity thresholds for segmentation. Each tuple corresponds to a frame in the stack,
            with values (lower_threshold, upper_threshold). If None, global thresholds are determined automatically (default is None).
    view_on_napari : bool, optional
            If True, displays the original stack and segmentation results in Napari (default is False).
    equalize_reference : int or None, optional
            The index of a reference frame used for histogram equalization. If None, equalization is not performed (default is None).
    filters : list of dict, optional
            A list of dictionaries specifying filters to be applied pre-segmentation. Each dictionary should
            contain filter parameters (default is None).
    marker_min_distance : int, optional
            The minimum distance between markers used for distinguishing separate objects (default is 30).
    marker_footprint_size : int, optional
            The size of the footprint used for local maxima detection when generating markers (default is 20).
    marker_footprint : ndarray or None, optional
            An array specifying the footprint used for local maxima detection. Overrides `marker_footprint_size` if provided
            (default is None).
    feature_queries : list of str or None, optional
            A list of query strings used to select features of interest from the segmented objects (default is None).
    fill_holes : bool, optional
            Whether to fill holes in the binary mask. If True, the binary mask will be processed to fill any holes.
            If False, the binary mask will not be modified. Default is True.

    Returns
    -------
    ndarray
            A 3D numpy array (T, Y, X) of type int16, where each element represents the segmented object label at each pixel.

    Notes
    -----
    - The segmentation process can be customized extensively via the parameters, allowing for complex segmentation tasks.

    """

    masks = []
    for t in tqdm(range(len(stack))):
        instance_seg = segment_frame_from_thresholds(
            stack[t],
            target_channel=target_channel,
            thresholds=thresholds,
            equalize_reference=equalize_reference,
            filters=filters,
            marker_min_distance=marker_min_distance,
            marker_footprint_size=marker_footprint_size,
            marker_footprint=marker_footprint,
            feature_queries=feature_queries,
            fill_holes=fill_holes,
        )
        masks.append(instance_seg)

    masks = np.array(masks, dtype=np.int16)
    if view_on_napari:
        from celldetective.napari.utils import _view_on_napari

        _view_on_napari(tracks=None, stack=stack, labels=masks)
    return masks


def segment_frame_from_thresholds(
    frame: np.ndarray,
    target_channel: int = 0,
    thresholds: Optional[Tuple[float, float]] = None,
    equalize_reference: Optional[np.ndarray] = None,
    filters: Optional[List[Dict[str, Any]]] = None,
    marker_min_distance: int = 30,
    marker_footprint_size: int = 20,
    marker_footprint: Optional[np.ndarray] = None,
    feature_queries: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None,
    do_watershed: bool = True,
    edge_exclusion: bool = True,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Segments objects within a single frame based on intensity thresholds and optional image processing steps.

    This function performs instance segmentation on a single frame using intensity thresholds, with optional steps
    including histogram equalization, filtering, and marker-based watershed segmentation. The segmented
    objects can be further filtered based on specified features.

    Parameters
    ----------
    frame : ndarray
            A 3D numpy array representing a single frame with dimensions (Y, X, C).
    target_channel : int, optional
            The channel index to be used for segmentation (default is 0).
    thresholds : tuple of int, optional
            A tuple specifying the intensity thresholds for segmentation, in the form (lower_threshold, upper_threshold).
    equalize_reference : ndarray or None, optional
            A 2D numpy array used as a reference for histogram equalization. If None, equalization is not performed (default is None).
    filters : list of dict, optional
            A list of dictionaries specifying filters to be applied to the image before segmentation. Each dictionary
            should contain filter parameters (default is None).
    marker_min_distance : int, optional
            The minimum distance between markers used for distinguishing separate objects during watershed segmentation (default is 30).
    marker_footprint_size : int, optional
            The size of the footprint used for local maxima detection when generating markers for watershed segmentation (default is 20).
    marker_footprint : ndarray or None, optional
            An array specifying the footprint used for local maxima detection. Overrides `marker_footprint_size` if provided (default is None).
    feature_queries : list of str or None, optional
            A list of query strings used to select features of interest from the segmented objects for further filtering (default is None).
    channel_names : list of str or None, optional
            A list of channel names corresponding to the dimensions in `frame`, used in conjunction with `feature_queries` for feature selection (default is None).
    do_watershed : bool, optional
            Whether to perform watershed segmentation. Default is True.
    edge_exclusion : bool, optional
            Whether to exclude unreliable edges based on filtering. Default is True.
    fill_holes : bool, optional
            Whether to fill holes in the binary mask. Default is True.

    Returns
    -------
    ndarray
            A 2D numpy array of type int, where each element represents the segmented object label at each pixel.

    """

    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]
    img = frame[:, :, target_channel]

    if np.any(img != img):
        img = interpolate_nan(img)

    if equalize_reference is not None:
        img = match_histograms(img, equalize_reference)

    img_mc = frame.copy()
    img = filter_image(img, filters=filters)
    if edge_exclusion:
        edge = estimate_unreliable_edge(filters)
    else:
        edge = None

    binary_image = threshold_image(
        img, thresholds[0], thresholds[1], fill_holes=fill_holes, edge_exclusion=edge
    )

    if do_watershed:
        coords, distance = identify_markers_from_binary(
            binary_image,
            marker_min_distance,
            footprint_size=marker_footprint_size,
            footprint=marker_footprint,
            return_edt=True,
        )
        instance_seg = apply_watershed(
            binary_image, coords, distance, fill_holes=fill_holes
        )
    else:
        instance_seg, _ = ndi.label(binary_image.astype(int).copy())

    instance_seg = filter_on_property(
        instance_seg,
        intensity_image=img_mc,
        queries=feature_queries,
        channel_names=channel_names,
    )

    return instance_seg


def filter_on_property(
    labels: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    queries: Optional[Union[str, List[str]]] = None,
    channel_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Filters segmented objects in a label image based on specified properties and queries.

    This function evaluates each segmented object (label) in the input label image against a set of queries related to its
    morphological and intensity properties. Objects not meeting the criteria defined in the queries are removed from the label
    image. This allows for the exclusion of objects based on size, shape, intensity, or custom-defined properties.

    Parameters
    ----------
    labels : ndarray
            A 2D numpy array where each unique non-zero integer represents a segmented object (label).
    intensity_image : ndarray, optional
            A 2D numpy array of the same shape as `labels`, providing intensity values for each pixel. This is used to calculate
            intensity-related properties of the segmented objects if provided (default is None).
    queries : str or list of str, optional
            One or more query strings used to filter the segmented objects based on their properties. Each query should be a
            valid pandas query string (default is None).
    channel_names : list of str or None, optional
            A list of channel names corresponding to the dimensions in the `intensity_image`. This is used to rename intensity
            property columns appropriately (default is None).

    Returns
    -------
    ndarray
            A 2D numpy array of the same shape as `labels`, with objects not meeting the query criteria removed.

    Notes
    -----
    - The function computes a set of predefined morphological properties and, if `intensity_image` is provided, intensity properties.
    - Queries should be structured according to pandas DataFrame query syntax and can reference any of the computed properties.
    - If `channel_names` is provided, intensity property column names are renamed to reflect the corresponding channel.

    """

    if queries is None:
        return labels
    else:
        if isinstance(queries, str):
            queries = [queries]

    props = [
        "label",
        "area",
        "area_bbox",
        "area_convex",
        "area_filled",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "equivalent_diameter_area",
        "euler_number",
        "feret_diameter_max",
        "orientation",
        "perimeter",
        "perimeter_crofton",
        "solidity",
        "centroid",
    ]

    intensity_props = ["intensity_mean", "intensity_max", "intensity_min"]

    if intensity_image is not None:
        props.extend(intensity_props)

    import pandas as pd

    properties = pd.DataFrame(
        regionprops_table(labels, intensity_image=intensity_image, properties=props)
    )

    if channel_names is not None:
        properties = rename_intensity_column(properties, channel_names)
    properties["radial_distance"] = np.sqrt(
        (properties["centroid-1"] - labels.shape[0] / 2) ** 2
        + (properties["centroid-0"] - labels.shape[1] / 2) ** 2
    )

    for query in queries:
        if query != "":
            try:
                properties = properties.query(f"not ({query})")
            except Exception as e:
                logger.error(
                    f"Query {query} could not be applied. Ensure that the feature exists. {e}"
                )

    cell_ids = list(np.unique(labels)[1:])
    leftover_cells = list(properties["label"].unique())
    to_remove = [value for value in cell_ids if value not in leftover_cells]

    for c in to_remove:
        labels[np.where(labels == c)] = 0.0

    return labels


def apply_watershed(
    binary_image: np.ndarray,
    coords: np.ndarray,
    distance: np.ndarray,
    fill_holes: bool = True,
) -> np.ndarray:
    """
    Applies the watershed algorithm to segment objects in a binary image using given markers and distance map.

    This function uses the watershed segmentation algorithm to delineate objects in a binary image. Markers for watershed
    are determined by the coordinates of local maxima, and the segmentation is guided by a distance map to separate objects
    that are close to each other.

    Parameters
    ----------
    binary_image : ndarray
            A 2D numpy array of type bool, where True represents the foreground objects to be segmented and False represents the background.
    coords : ndarray
            An array of shape (N, 2) containing the (row, column) coordinates of local maxima points that will be used as markers for the
            watershed algorithm. N is the number of local maxima.
    distance : ndarray
            A 2D numpy array of the same shape as `binary_image`, containing the distance transform of the binary image. This map is used
            to guide the watershed segmentation.
    fill_holes : bool, optional
            Whether to fill holes in the binary mask after watershed segmentation. Default is True.

    Returns
    -------
    ndarray
            A 2D numpy array of type int, where each unique non-zero integer represents a segmented object (label).

    Notes
    -----
    - The function assumes that `coords` are derived from the distance map of `binary_image`, typically obtained using
      peak local max detection on the distance transform.
    - The watershed algorithm treats each local maximum as a separate object and segments the image by "flooding" from these points.
    - This implementation uses the `skimage.morphology.watershed` function under the hood.

    Examples
    --------
    >>> from skimage import measure, morphology
    >>> binary_image = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
    >>> distance = morphology.distance_transform_edt(binary_image)
    >>> coords = measure.peak_local_max(distance, indices=True)
    >>> labels = apply_watershed(binary_image, coords, distance)
    # Segments the objects in `binary_image` using the watershed algorithm.

    """

    mask = np.zeros(binary_image.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_image)
    if fill_holes:
        from celldetective.utils.mask_cleaning import fill_label_holes

        try:
            labels = fill_label_holes(labels)
        except ImportError as ie:
            logger.warning(f"Stardist not found, cannot fill holes... {ie}")
    return labels


def identify_markers_from_binary(
    binary_image: np.ndarray,
    min_distance: int,
    footprint_size: int = 20,
    footprint: Optional[np.ndarray] = None,
    return_edt: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """

    Identify markers from a binary image using distance transform and peak detection.

    Parameters
    ----------
    binary_image : ndarray
            The binary image from which to identify markers.
    min_distance : int
            The minimum distance between markers. Only the markers with a minimum distance greater than or equal to
            `min_distance` will be identified.
    footprint_size : int, optional
            The size of the footprint or structuring element used for peak detection. Default is 20.
    footprint : ndarray, optional
            The footprint or structuring element used for peak detection. If None, a square footprint of size
            `footprint_size` will be used. Default is None.
    return_edt : bool, optional
            Whether to return the Euclidean distance transform image along with the identified marker coordinates.
            If True, the function will return the marker coordinates and the distance transform image as a tuple.
            If False, only the marker coordinates will be returned. Default is False.

    Returns
    -------
    ndarray or tuple
            If `return_edt` is False, returns the identified marker coordinates as an ndarray of shape (N, 2), where N is
            the number of identified markers. If `return_edt` is True, returns a tuple containing the marker coordinates
            and the distance transform image.

    Notes
    -----
    This function uses the distance transform of the binary image to identify markers by detecting local maxima. The
    distance transform assigns each pixel a value representing the Euclidean distance to the nearest background pixel.
    By finding peaks in the distance transform, we can identify the markers in the original binary image. The `min_distance`
    parameter controls the minimum distance between markers to avoid clustering.

    """

    distance = ndi.distance_transform_edt(binary_image.astype(float))
    if footprint is None:
        footprint = np.ones((footprint_size, footprint_size))
    coords = peak_local_max(
        distance,
        footprint=footprint,
        labels=binary_image.astype(int),
        min_distance=min_distance,
    )
    if return_edt:
        return coords, distance
    else:
        return coords


def segment_at_position(
    pos: str,
    mode: str,
    model_name: str,
    stack_prefix: Optional[str] = None,
    use_gpu: bool = True,
    return_labels: bool = False,
    view_on_napari: bool = False,
    threads: int = 1,
) -> Optional[np.ndarray]:
    """
    Perform image segmentation at the specified position using a pre-trained model.

    Parameters
    ----------
    pos : str
            The path to the position directory containing the input images to be segmented.
    mode : str
            The segmentation mode. This determines the type of objects to be segmented ('target' or 'effector').
    model_name : str
            The name of the pre-trained segmentation model to be used.
    stack_prefix : str or None, optional
            The prefix of the stack file name. Defaults to None.
    use_gpu : bool, optional
            Whether to use the GPU for segmentation if available. Defaults to True.
    return_labels : bool, optional
            If True, the function returns the segmentation labels as an output. Defaults to False.
    view_on_napari : bool, optional
            If True, the segmented labels are displayed in a Napari viewer. Defaults to False.
    threads : int, optional
            Number of threads to use for segmentation. Defaults to 1.

    Returns
    -------
    numpy.ndarray or None
            If `return_labels` is True, the function returns the segmentation labels as a NumPy array. Otherwise, it returns None. The subprocess writes the
            segmentation labels in the position directory.

    Examples
    --------
    >>> labels = segment_at_position('ExperimentFolder/W1/100/', 'effector', 'mice_t_cell_RICM', return_labels=True)

    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"

    name_path = locate_segmentation_model(model_name)

    script_path = os.sep.join([abs_path, "scripts", "segment_cells.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--pos", pos, "--model", model_name, "--mode", mode, "--use_gpu", str(use_gpu), "--threads", str(threads)],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Segmentation script exited with code {result.returncode} for position {pos}.")
        raise RuntimeError(f"Segmentation failed for position {pos} (exit code {result.returncode}).")

    if return_labels or view_on_napari:
        labels = locate_labels(pos, population=mode)
    if view_on_napari:
        from celldetective.napari.utils import _view_on_napari

        if stack_prefix is None:
            stack_prefix = ""
        stack = locate_stack(pos, prefix=stack_prefix)
        _view_on_napari(tracks=None, stack=stack, labels=labels)
    if return_labels:
        return labels
    else:
        return None


def segment_from_threshold_at_position(
    pos: str, mode: str, config: str, threads: int = 1
) -> None:
    """
    Executes a segmentation script on a specified position directory using a given configuration and mode.

    This function calls an external Python script designed to segment images at a specified position directory.
    The segmentation is configured through a JSON file and can operate in different modes specified by the user.
    The function can leverage multiple threads to potentially speed up the processing.

    Parameters
    ----------
    pos : str
            The file path to the position directory where images to be segmented are stored. The path must be valid.
    mode : str
            The operation mode for the segmentation script. The mode determines how the segmentation is performed and
            which algorithm or parameters are used.
    config : str
            The file path to the JSON configuration file that specifies parameters for the segmentation process. The
            path must be valid.
    threads : int, optional
            The number of threads to use for processing. Using more than one thread can speed up segmentation on
            systems with multiple CPU cores (default is 1).

    Raises
    ------
    AssertionError
            If either the `pos` or `config` paths do not exist.

    Notes
    -----
    - The external segmentation script (`segment_cells_thresholds.py`) is expected to be located in a specific
      directory relative to this function.
    - The segmentation process and its parameters, including modes and thread usage, are defined by the external
      script and the configuration file.

    Examples
    --------
    >>> pos = '/path/to/position'
    >>> mode = 'default'
    >>> config = '/path/to/config.json'
    >>> segment_from_threshold_at_position(pos, mode, config, threads=2)
    # This will execute the segmentation script on the specified position directory with the given mode and
    # configuration, utilizing 2 threads.

    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"

    config = config.replace("\\", "/")
    config = rf"{config}"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config {config} is not a valid path.")

    script_path = os.sep.join([abs_path, "scripts", "segment_cells_thresholds.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--pos", pos, "--config", config, "--mode", mode, "--threads", str(threads)],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Threshold segmentation script exited with code {result.returncode} for position {pos}.")
        raise RuntimeError(f"Threshold segmentation failed for position {pos} (exit code {result.returncode}).")


def train_segmentation_model(config: str, use_gpu: bool = True) -> None:
    """
    Trains a segmentation model based on a specified configuration file.

    This function initiates the training of a segmentation model by calling an external Python script,
    which reads the training parameters and dataset information from a given JSON configuration file.
    The training process, including model architecture, training data, and hyperparameters, is defined
    by the contents of the configuration file.

    Parameters
    ----------
    config : str
            The file path to the JSON configuration file that specifies training parameters and dataset
            information for the segmentation model. The path must be valid.
    use_gpu : bool, optional
            Whether to use GPU acceleration for training. Defaults to True.

    Raises
    ------
    AssertionError
            If the `config` path does not exist.

    Notes
    -----
    - The external training script (`train_segmentation_model.py`) is assumed to be located in a specific
      directory relative to this function.
    - The segmentation model and training process are highly dependent on the details specified in the
      configuration file, including the model architecture, loss functions, optimizer settings, and
      training/validation data paths.

    Examples
    --------
    >>> config = '/path/to/training_config.json'
    >>> train_segmentation_model(config)
    # Initiates the training of a segmentation model using the parameters specified in the given configuration file.

    """

    config = config.replace("\\", "/")
    config = rf"{config}"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config {config} is not a valid path.")

    script_path = os.sep.join([abs_path, "scripts", "train_segmentation_model.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--config", config, "--use_gpu", str(use_gpu)],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Segmentation model training script exited with code {result.returncode}.")
        raise RuntimeError(f"Segmentation model training failed (exit code {result.returncode}).")


def merge_instance_segmentation(
    labels: List[np.ndarray], iou_matching_threshold: float = 0.05, mode: str = "OR"
) -> np.ndarray:
    """
    Merges multiple instance segmentation masks into a single mask.

    Parameters
    ----------
    labels : list of ndarray
        List of label images to merging.
    iou_matching_threshold : float, optional
        IoU threshold for matching instances. Default is 0.05.
    mode : str, optional
        Merging mode for overlapping instances ('OR', 'AND', 'XOR'). Default is 'OR'.

    Returns
    -------
    ndarray
        Merged label image.
    """

    label_reference = labels[0]
    try:
        from stardist.matching import matching
    except ImportError:
        logger.warning(
            "StarDist not installed. Cannot perform instance matching/merging..."
        )
        return label_reference

    for i in range(1, len(labels)):

        label_to_merge = labels[i]
        pairs = matching(
            label_reference,
            label_to_merge,
            thresh=0.5,
            criterion="iou",
            report_matches=True,
        ).matched_pairs
        scores = matching(
            label_reference,
            label_to_merge,
            thresh=0.5,
            criterion="iou",
            report_matches=True,
        ).matched_scores

        accepted_pairs = []
        for k, p in enumerate(pairs):
            s = scores[k]
            if s > iou_matching_threshold:
                accepted_pairs.append(p)

        merge = np.copy(label_reference)

        for p in accepted_pairs:
            merge[np.where(merge == p[0])] = 0.0
            cdt1 = label_reference == p[0]
            cdt2 = label_to_merge == p[1]
            if mode == "OR":
                cdt = np.logical_or(cdt1, cdt2)
            elif mode == "AND":
                cdt = np.logical_and(cdt1, cdt2)
            elif mode == "XOR":
                cdt = np.logical_xor(cdt1, cdt2)
            loc_i, loc_j = np.where(cdt)
            merge[loc_i, loc_j] = p[0]

        cells_to_ignore = [p[1] for p in accepted_pairs]
        for c in cells_to_ignore:
            label_to_merge[label_to_merge == c] = 0

        label_to_merge[label_to_merge != 0] = label_to_merge[label_to_merge != 0] + int(
            np.amax(label_reference)
        )
        merge[label_to_merge != 0] = label_to_merge[label_to_merge != 0]

        label_reference = merge

    merge = auto_correct_masks(merge)

    return merge


if __name__ == "__main__":
    print(segment(None, "test"))
