import gc
import json
import os
from glob import glob
from typing import Optional

import numpy as np
from celldetective.utils.io import save_tiff_imagej_compatible
from imageio import v2 as imageio
from natsort import natsorted
from tifffile import imread, TiffFile

from celldetective.utils.image_cleaning import (
    _fix_no_contrast,
    interpolate_nan_multichannel,
)
from celldetective.utils.normalization import normalize_multichannel
from celldetective import get_logger

import logging
import warnings

logger = get_logger(__name__)

# Suppress tifffile warnings about missing files in MMStack
logging.getLogger("tifffile").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*MMStack series is missing files.*")


def locate_stack(position, prefix="Aligned"):
    """

    Locate and load a stack of images.

    Parameters
    ----------
    position : str
            The position folder within the well where the stack is located.
    prefix : str, optional
            The prefix used to identify the stack. The default is 'Aligned'.

    Returns
    -------
    stack : ndarray
            The loaded stack as a NumPy array.

    Raises
    ------
    AssertionError
            If no stack with the specified prefix is found.

    Notes
    -----
    This function locates and loads a stack of images based on the specified position and prefix.
    It assumes that the stack is stored in a directory named 'movie' within the specified position.
    The function loads the stack as a NumPy array and performs shape manipulation to have the channels
    at the end.

    Examples
    --------
    >>> stack = locate_stack(position, prefix='Aligned')
    # Locate and load a stack of images for further processing.

    """

    if not position.endswith(os.sep):
        position += os.sep

    stack_path = glob(position + os.sep.join(["movie", f"{prefix}*.tif"]))
    if not stack_path:
        raise FileNotFoundError(f"No movie with prefix {prefix} found...")

    stack = imread(stack_path[0].replace("\\", "/"))
    stack_length = auto_load_number_of_frames(stack_path[0])

    if stack.ndim == 4:
        stack = np.moveaxis(stack, 1, -1)
    elif stack.ndim == 3:
        if min(stack.shape) != stack_length:
            channel_axis = np.argmin(stack.shape)
            if channel_axis != (stack.ndim - 1):
                stack = np.moveaxis(stack, channel_axis, -1)
            stack = stack[np.newaxis, :, :, :]
        else:
            stack = stack[:, :, :, np.newaxis]
    elif stack.ndim == 2:
        stack = stack[np.newaxis, :, :, np.newaxis]

    return stack


def locate_labels(position, population="target", frames=None):
    """
    Locate and load label images for a given position and population in an experiment.

    This function retrieves and optionally loads labeled images (e.g., targets or effectors)
    for a specified position in an experiment. It supports loading all frames, a specific
    frame, or a list of frames.

    Parameters
    ----------
    position : str
            Path to the position directory containing label images.
    population : str, optional
            The population to load labels for. Options are `'target'` (or `'targets'`) and
            `'effector'` (or `'effectors'`). Default is `'target'`.
    frames : int, list of int, numpy.ndarray, or None, optional
            Specifies which frames to load:
            - `None`: Load all frames (default).
            - `int`: Load a single frame, identified by its index.
            - `list` or `numpy.ndarray`: Load multiple specific frames.

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
            If `frames` is `None` or a single integer, returns a NumPy array of the corresponding
            labels. If `frames` is a list or array, returns a list of NumPy arrays for each frame.
            If a frame is not found, `None` is returned for that frame.

    Notes
    -----
    - The function assumes label images are stored in subdirectories named `"labels_targets"`
      or `"labels_effectors"`, with filenames formatted as `####.tif` (e.g., `0001.tif`).
    - Frame indices are zero-padded to four digits for matching.
    - If `frames` is invalid or a frame is not found, `None` is returned for that frame.

    Examples
    --------
    Load all label images for a position:

    >>> labels = locate_labels("/path/to/position", population="target")

    Load a single frame (frame index 3):

    >>> label = locate_labels("/path/to/position", population="effector", frames=3)

    Load multiple specific frames:

    >>> labels = locate_labels("/path/to/position", population="target", frames=[0, 1, 2])

    """

    if not position.endswith(os.sep):
        position += os.sep

    if population.lower() == "target" or population.lower() == "targets":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_targets", "*.tif"]))
        )
    elif population.lower() == "effector" or population.lower() == "effectors":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_effectors", "*.tif"]))
        )
    else:
        label_path = natsorted(
            glob(position + os.sep.join([f"labels_{population}", "*.tif"]))
        )

    label_names = [os.path.split(lbl)[-1] for lbl in label_path]

    if frames is None:

        labels = np.array([imread(i.replace("\\", "/")) for i in label_path])

    elif isinstance(frames, (int, float, np.int_)):

        tzfill = str(int(frames)).zfill(4)
        try:
            idx = label_names.index(f"{tzfill}.tif")
        except:
            idx = -1

        if idx == -1:
            labels = None
        else:
            labels = np.array(imread(label_path[idx].replace("\\", "/")))

    elif isinstance(frames, (list, np.ndarray)):
        labels = []
        for f in frames:
            tzfill = str(int(f)).zfill(4)
            try:
                idx = label_names.index(f"{tzfill}.tif")
            except:
                idx = -1

            if idx == -1:
                labels.append(None)
            else:
                labels.append(np.array(imread(label_path[idx].replace("\\", "/"))))
    else:
        print("Frames argument must be None, int or list...")

    return labels


def locate_stack_and_labels(position, prefix="Aligned", population="target"):
    """

    Locate and load the stack and corresponding segmentation labels.

    Parameters
    ----------
    position : str
            The position or directory path where the stack and labels are located.
    prefix : str, optional
            The prefix used to identify the stack. The default is 'Aligned'.
    population : str, optional
            The population for which the segmentation must be located. The default is 'target'.

    Returns
    -------
    stack : ndarray
            The loaded stack as a NumPy array.
    labels : ndarray
            The loaded segmentation labels as a NumPy array.

    Raises
    ------
    AssertionError
            If no stack with the specified prefix is found or if the shape of the stack and labels do not match.

    Notes
    -----
    This function locates the stack and corresponding segmentation labels based on the specified position and population.
    It assumes that the stack and labels are stored in separate directories: 'movie' for the stack and 'labels' or 'labels_effectors' for the labels.
    The function loads the stack and labels as NumPy arrays and performs shape validation.

    Examples
    --------
    >>> stack, labels = locate_stack_and_labels(position, prefix='Aligned', population="target")
    # Locate and load the stack and segmentation labels for further processing.

    """

    position = position.replace("\\", "/")
    labels = locate_labels(position, population=population)
    stack = locate_stack(position, prefix=prefix)
    if len(labels) < len(stack):
        fix_missing_labels(position, population=population, prefix=prefix)
        labels = locate_labels(position, population=population)
    assert len(stack) == len(
        labels
    ), f"The shape of the stack {stack.shape} does not match with the shape of the labels {labels.shape}"

    return stack, labels


def auto_load_number_of_frames(stack_path):
    """
    Automatically determine the number of frames in a TIFF image stack.

    This function extracts the number of frames (time slices) from the metadata of a TIFF file
    or infers it from the stack dimensions when metadata is unavailable. It is robust to
    variations in metadata structure and handles multi-channel images.

    Parameters
    ----------
    stack_path : str
            Path to the TIFF image stack file.

    Returns
    -------
    int or None
            The number of frames in the image stack. Returns `None` if the path is `None`
            or the frame count cannot be determined.

    Notes
    -----
    - The function attempts to extract the `frames` or `slices` attributes from the
      TIFF metadata, specifically the `ImageDescription` tag.
    - If metadata extraction fails, the function reads the image stack and infers
      the number of frames based on the stack dimensions.
    - Multi-channel stacks are handled by assuming the number of channels is specified
      in the metadata under the `channels` attribute.

    Examples
    --------
    Automatically detect the number of frames in a TIFF stack:

    >>> frames = auto_load_number_of_frames("experiment_stack.tif")
    Automatically detected stack length: 120...

    Handle a single-frame TIFF:

    >>> frames = auto_load_number_of_frames("single_frame_stack.tif")
    Automatically detected stack length: 1...

    Handle invalid or missing paths gracefully:

    >>> frames = auto_load_number_of_frames("stack.tif")
    >>> print(frames)
    None

    """

    if stack_path is None:
        return None

    stack_path = stack_path.replace("\\", "/")
    n_channels = 1

    with TiffFile(stack_path) as tif:
        try:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            img_desc = tif_tags["ImageDescription"]
            attr = img_desc.split("\n")
            n_channels = int(
                attr[np.argmax([s.startswith("channels") for s in attr])].split("=")[-1]
            )
        except Exception as e:
            pass
        try:
            # Try nframes
            nslices = int(
                attr[np.argmax([s.startswith("frames") for s in attr])].split("=")[-1]
            )
            if nslices > 1:
                len_movie = nslices
            else:
                break_the_code()
        except:
            try:
                # try nslices
                frames = int(
                    attr[np.argmax([s.startswith("slices") for s in attr])].split("=")[
                        -1
                    ]
                )
                len_movie = frames
            except:
                pass

    try:
        del tif
        del tif_tags
        del img_desc
    except:
        pass

    if "len_movie" not in locals():
        stack = imread(stack_path)
        len_movie = len(stack)
        if len_movie == n_channels and stack.ndim == 3:
            len_movie = 1
        if stack.ndim == 2:
            len_movie = 1
        del stack
    gc.collect()

    logger.info(f"Automatically detected stack length: {len_movie}...")

    return len_movie if "len_movie" in locals() else None


def _load_frames_to_segment(file, indices, scale_model=None, normalize_kwargs=None):

    frames = load_frames(
        indices,
        file,
        scale=scale_model,
        normalize_input=True,
        normalize_kwargs=normalize_kwargs,
    )
    frames = interpolate_nan_multichannel(frames)

    if np.any(indices == -1):
        frames[:, :, np.where(indices == -1)[0]] = 0.0

    return frames


def _load_frames_to_measure(file, indices):
    return load_frames(indices, file, scale=None, normalize_input=False)


def load_frames(
    img_nums,
    stack_path,
    scale=None,
    normalize_input=True,
    dtype=np.float64,
    normalize_kwargs={"percentiles": (0.0, 99.99)},
):
    """
    Loads and optionally normalizes and rescales specified frames from a stack located at a given path.

    This function reads specified frames from a stack file, applying systematic adjustments to ensure
    the channel axis is last. It supports optional normalization of the input frames and rescaling. An
    artificial pixel modification is applied to frames with uniform values to prevent errors during
    normalization.

    Parameters
    ----------
    img_nums : int or list of int
            The index (or indices) of the image frame(s) to load from the stack.
    stack_path : str
            The file path to the stack from which frames are to be loaded.
    scale : float, optional
            The scaling factor to apply to the frames. If None, no scaling is applied (default is None).
    normalize_input : bool, optional
            Whether to normalize the loaded frames. If True, normalization is applied according to
            `normalize_kwargs` (default is True).
    dtype : data-type, optional
            The desired data-type for the output frames (default is float).
    normalize_kwargs : dict, optional
            Keyword arguments to pass to the normalization function (default is {"percentiles": (0., 99.99)}).

    Returns
    -------
    ndarray or None
            The loaded, and possibly normalized and rescaled, frames as a NumPy array. Returns None if there
            is an error in loading the frames.

    Raises
    ------
    Exception
            Prints an error message if the specified frames cannot be loaded or if there is a mismatch between
            the provided experiment channel information and the stack format.

    Notes
    -----
    - The function uses scikit-image for reading frames and supports multi-frame TIFF stacks.
    - Normalization and scaling are optional and can be customized through function parameters.
    - A workaround is implemented for frames with uniform pixel values to prevent normalization errors by
      adding a 'fake' pixel.

    Examples
    --------
    >>> frames = load_frames([0, 1, 2], '/path/to/stack.tif', scale=0.5, normalize_input=True, dtype=np.uint8)
    # Loads the first three frames from '/path/to/stack.tif', normalizes them, rescales by a factor of 0.5,
    # and converts them to uint8 data type.

    """

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*MMStack series is missing files.*"
            )
            frames = imageio.imread(stack_path, key=img_nums)
    except Exception as e:
        print(
            f"Error in loading the frame {img_nums} {e}. Please check that the experiment channel information is consistent with the movie being read."
        )
        return None
    try:
        if np.any(np.isinf(frames)):
            frames = frames.astype(float)
            frames[np.isinf(frames)] = np.nan
    except Exception as e:
        print(e)

    frames = _rearrange_multichannel_frame(frames)

    if normalize_input:
        frames = normalize_multichannel(frames.astype(float), **normalize_kwargs)

    if scale is not None:
        frames = zoom_multiframes(frames.astype(float), scale)

    # add a fake pixel to prevent auto normalization errors on images that are uniform
    frames = _fix_no_contrast(frames)

    return frames  # .astype(dtype)


def _rearrange_multichannel_frame(
    frame: np.ndarray, n_channels: Optional[int] = None
) -> np.ndarray:
    """
    Rearranges the axes of a multi-channel frame to ensure the channel axis is at the end.

    This function standardizes the input frame to ensure that the channel axis (if present)
    is moved to the last position. For 2D frames, it adds a singleton channel axis at the end.

    Parameters
    ----------
    frame : ndarray
            The input frame to be rearranged. Can be 2D or 3D.
            - If 3D, the function identifies the channel axis (assumed to be the axis with the smallest size)
              and moves it to the last position.
            - If 2D, the function adds a singleton channel axis to make it compatible with 3D processing.

    Returns
    -------
    ndarray
            The rearranged frame with the channel axis at the end.
            - For 3D frames, the output shape will have the channel axis as the last dimension.
            - For 2D frames, the output will have shape `(H, W, 1)` where `H` and `W` are the height and width of the frame.

    Notes
    -----
    - This function assumes that in a 3D input, the channel axis is the one with the smallest size.
    - For 2D frames, this function ensures compatibility with multi-channel processing pipelines by
      adding a singleton dimension for the channel axis.

    Examples
    --------
    Rearranging a 3D multi-channel frame:
    >>> frame = np.zeros((10, 10, 3))  # Already channel-last
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 3)

    Rearranging a 3D frame with channel axis not at the end:
    >>> frame = np.zeros((3, 10, 10))  # Channel-first
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 3)

    Converting a 2D frame to have a channel axis:
    >>> frame = np.zeros((10, 10))  # Grayscale image
    >>> _rearrange_multichannel_frame(frame).shape
    (10, 10, 1)
    """

    if frame.ndim == 3:
        # Systematically move channel axis to the end
        if n_channels is not None and n_channels in list(frame.shape):
            channel_axis = list(frame.shape).index(n_channels)
        else:
            channel_axis = np.argmin(frame.shape)
        frame = np.moveaxis(frame, channel_axis, -1)

    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]

    return frame


def zoom_multiframes(frames: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Applies zooming to each frame (channel) in a multi-frame image.

    This function resizes each channel of a multi-frame image independently using a specified zoom factor.
    The zoom is applied using spline interpolation of the specified order, and the channels are combined
    back into the original format.

    Parameters
    ----------
    frames : ndarray
            A multi-frame image with dimensions `(height, width, channels)`. The last axis represents different
            channels.
    zoom_factor : float
            The zoom factor to apply to each channel. Values greater than 1 increase the size, and values
            between 0 and 1 decrease the size.

    Returns
    -------
    ndarray
            A new multi-frame image with the same number of channels as the input, but with the height and width
            scaled by the zoom factor.

    Notes
    -----
    - The function uses spline interpolation (order 3) for resizing, which provides smooth results.
    - `prefilter=False` is used to prevent additional filtering during the zoom operation.
    - The function assumes that the input is in `height x width x channels` format, with channels along the
      last axis.
    """

    from scipy.ndimage import zoom

    frames = [
        zoom(
            frames[:, :, c].copy(), [zoom_factor, zoom_factor], order=3, prefilter=False
        )
        for c in range(frames.shape[-1])
    ]
    frames = np.moveaxis(frames, 0, -1)
    return frames


def fix_missing_labels(position, population="target", prefix="Aligned"):
    """
    Fix missing label files by creating empty label images for frames that do not have corresponding label files.

    This function locates missing label files in a sequence of frames and creates empty labels (filled with zeros)
    for the frames that are missing. The function works for two types of populations: 'target' or 'effector'.

    Parameters
    ----------
    position : str
        The file path to the folder containing the images/label files. This is the root directory where
        the label files are expected to be found.
    population : str, optional
        Specifies whether to look for 'target' or 'effector' labels. Accepts 'target' or 'effector'
        as valid values. Default is 'target'.
    prefix : str, optional
        The prefix used to locate the image stack (default is 'Aligned').

    Returns
    -------
    None
        The function creates new label files in the corresponding folder for any frames missing label files.

    """

    if not position.endswith(os.sep):
        position += os.sep

    stack = locate_stack(position, prefix=prefix)
    template = np.zeros((stack[0].shape[0], stack[0].shape[1]), dtype=int)
    all_frames = np.arange(len(stack))

    if population.lower() == "target" or population.lower() == "targets":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_targets", "*.tif"]))
        )
        path = position + os.sep + "labels_targets"
    elif population.lower() == "effector" or population.lower() == "effectors":
        label_path = natsorted(
            glob(position + os.sep.join(["labels_effectors", "*.tif"]))
        )
        path = position + os.sep + "labels_effectors"
    else:
        label_path = natsorted(
            glob(position + os.sep.join([f"labels_{population}", "*.tif"]))
        )
        path = position + os.sep + f"labels_{population}"

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if label_path != []:
        # path = os.path.split(label_path[0])[0]
        int_valid = [int(lbl.split(os.sep)[-1].split(".")[0]) for lbl in label_path]
        to_create = [x for x in all_frames if x not in int_valid]
    else:
        to_create = all_frames
    to_create = [str(x).zfill(4) + ".tif" for x in to_create]
    for file in to_create:
        save_tiff_imagej_compatible(
            os.sep.join([path, file]), template.astype(np.int16), axes="YX"
        )
    # imwrite(os.sep.join([path, file]), template.astype(int))


def _get_img_num_per_channel(channels_indices, len_movie, nbr_channels):
    """
    Calculates the image frame numbers for each specified channel in a multi-channel movie.

    Given the indices of channels of interest, the total length of the movie, and the number of channels,
    this function computes the frame numbers corresponding to each channel throughout the movie. If a
    channel index is specified as None, it assigns a placeholder value to indicate no frames for that channel.

    Parameters
    ----------
    channels_indices : list of int or None
            A list containing the indices of channels for which to calculate frame numbers. If an index is None,
            it is interpreted as a channel with no frames to be processed.
    len_movie : int
            The total number of frames in the movie across all channels.
    nbr_channels : int
            The total number of channels in the movie.

    Returns
    -------
    ndarray
            A 2D numpy array where each row corresponds to a channel specified in `channels_indices` and contains
            the frame numbers for that channel throughout the movie. If a channel index is None, the corresponding
            row contains placeholder values (-1).

    Notes
    -----
    - The function assumes that frames in the movie are interleaved by channel, with frames for each channel
      appearing in a regular sequence throughout the movie.
    - This utility is particularly useful for multi-channel time-lapse movies where analysis or processing
      needs to be performed on a per-channel basis.

    Examples
    --------
    >>> channels_indices = [0]  # Indices for channels 1, 3, and a non-existing channel
    >>> len_movie = 10  # Total frames for each channel
    >>> nbr_channels = 3  # Total channels in the movie
    >>> img_num_per_channel = _get_img_num_per_channel(channels_indices, len_movie, nbr_channels)
    >>> print(img_num_per_channel)
    # array([[ 0,  3,  6,  9, 12, 15, 18, 21, 24, 27]])

    >>> channels_indices = [1,2]  # Indices for channels 1, 3, and a non-existing channel
    >>> len_movie = 10  # Total frames for each channel
    >>> nbr_channels = 3  # Total channels in the movie
    >>> img_num_per_channel = _get_img_num_per_channel(channels_indices, len_movie, nbr_channels)
    >>> print(img_num_per_channel)
    # array([[ 1,  4,  7, 10, 13, 16, 19, 22, 25, 28],
    #   [ 2,  5,  8, 11, 14, 17, 20, 23, 26, 29]])

    """

    if isinstance(channels_indices, (int, np.int_)):
        channels_indices = [channels_indices]

    len_movie = int(len_movie)
    nbr_channels = int(nbr_channels)

    img_num_all_channels = []
    for c in channels_indices:
        if c is not None:
            indices = np.arange(len_movie * nbr_channels)[c::nbr_channels]
        else:
            indices = [-1] * len_movie
        img_num_all_channels.append(indices)
    img_num_all_channels = np.array(img_num_all_channels, dtype=int)

    return img_num_all_channels


def _extract_channel_indices(channels, required_channels):
    """
    Extracts the indices of required channels from a list of available channels.

    This function is designed to match the channels required by a model or analysis process with the channels
    present in the dataset. It returns the indices of the required channels within the list of available channels.
    If the required channels are not found among the available channels, the function prints an error message and
    returns None.

    Parameters
    ----------
    channels : list of str or None
            A list containing the names of the channels available in the dataset. If None, it is assumed that the
            dataset channels are in the same order as the required channels.
    required_channels : list of str
            A list containing the names of the channels required by the model or analysis process.

    Returns
    -------
    ndarray or None
            An array of indices indicating the positions of the required channels within the list of available
            channels. Returns None if there is a mismatch between required and available channels.

    Notes
    -----
    - The function is useful for preprocessing steps where specific channels of multi-channel data are needed
      for further analysis or model input.
    - In cases where `channels` is None, indicating that the dataset does not specify channel names, the function
      assumes that the dataset's channel order matches the order of `required_channels` and returns an array of
      indices based on this assumption.

    Examples
    --------
    >>> available_channels = ['DAPI', 'GFP', 'RFP']
    >>> required_channels = ['GFP', 'RFP']
    >>> indices = _extract_channel_indices(available_channels, required_channels)
    >>> print(indices)
    # [1, 2]

    >>> indices = _extract_channel_indices(None, required_channels)
    >>> print(indices)
    # [0, 1]
    """

    channel_indices = []
    for c in required_channels:
        if c != "None" and c is not None:
            try:
                ch_idx = channels.index(c)
                channel_indices.append(ch_idx)
            except Exception as e:
                channel_indices.append(None)
        else:
            channel_indices.append(None)

    return channel_indices


def load_image_dataset(
    datasets, channels, train_spatial_calibration=None, mask_suffix="labelled"
):
    """
    Loads image and corresponding mask datasets, optionally applying spatial calibration adjustments.

    This function iterates over specified datasets, loading image and mask pairs based on provided channels
    and adjusting images according to a specified spatial calibration factor. It supports loading images with
    multiple channels and applies necessary transformations to match the training spatial calibration.

    Parameters
    ----------
    datasets : list of str
            A list of paths to the datasets containing the images and masks.
    channels : str or list of str
            The channel(s) to be loaded from the images. If a string is provided, it is converted into a list.
    train_spatial_calibration : float, optional
            The spatial calibration (e.g., micrometers per pixel) used during model training. If provided, images
            will be rescaled to match this calibration. Default is None, indicating no rescaling is applied.
    mask_suffix : str, optional
            The suffix used to identify mask files corresponding to the images. Default is 'labelled'.

    Returns
    -------
    tuple of lists
            A tuple containing two lists: `X` for images and `Y` for corresponding masks. Both lists contain
            numpy arrays of loaded and optionally transformed images and masks.

    Raises
    ------
    AssertionError
            If the provided `channels` argument is not a list or if the number of loaded images does not match
            the number of loaded masks.

    Notes
    -----
    - The function assumes that mask filenames are derived from image filenames by appending a `mask_suffix`
      before the file extension.
    - Spatial calibration adjustment involves rescaling the images and masks to match the `train_spatial_calibration`.
    - Only images with a corresponding mask and a valid configuration file specifying channel indices and
      spatial calibration are loaded.
    - The image samples must have at least one channel in common with the required channels to be accepted. The missing
      channels are passed as black frames.

    Examples
    --------
    >>> datasets = ['/path/to/dataset1', '/path/to/dataset2']
    >>> channels = ['DAPI', 'GFP']
    >>> X, Y = load_image_dataset(datasets, channels, train_spatial_calibration=0.65)
    # Loads DAPI and GFP channels from specified datasets, rescaling images to match a spatial calibration of 0.65.
    """

    from scipy.ndimage import zoom

    if isinstance(channels, str):
        channels = [channels]

    assert isinstance(channels, list), "Please provide a list of channels. Abort."

    X = []
    Y = []
    files = []

    for ds in datasets:
        print(f"Loading data from dataset {ds}...")
        if not ds.endswith(os.sep):
            ds += os.sep
        img_paths = list(
            set(glob(ds + "*.tif")) - set(glob(ds + f"*_{mask_suffix}.tif"))
        )
        for im in img_paths:
            print(f"{im=}")
            mask_path = os.sep.join(
                [
                    os.path.split(im)[0],
                    os.path.split(im)[-1].replace(".tif", f"_{mask_suffix}.tif"),
                ]
            )
            if os.path.exists(mask_path):
                # load image and mask
                image = imread(im)
                if image.ndim == 2:
                    image = image[np.newaxis]
                if image.ndim > 3:
                    print("Invalid image shape, skipping")
                    continue
                mask = imread(mask_path)
                config_path = im.replace(".tif", ".json")
                if os.path.exists(config_path):
                    # Load config
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    existing_channels = config["channels"]
                    intersection = list(
                        set(list(channels)) & set(list(existing_channels))
                    )
                    print(f"{existing_channels=} {intersection=}")
                    if len(intersection) == 0:
                        print(
                            "Channels could not be found in the config... Skipping image."
                        )
                        continue
                    else:
                        ch_idx = []
                        for c in channels:
                            if c in existing_channels:
                                idx = existing_channels.index(c)
                                ch_idx.append(idx)
                            else:
                                # For None or missing channel pass black frame
                                ch_idx.append(np.nan)
                        im_calib = config["spatial_calibration"]

                ch_idx = np.array(ch_idx)
                ch_idx_safe = np.copy(ch_idx)
                ch_idx_safe[ch_idx_safe != ch_idx_safe] = 0
                ch_idx_safe = ch_idx_safe.astype(int)

                image = image[ch_idx_safe]
                image[np.where(ch_idx != ch_idx)[0], :, :] = 0

                image = np.moveaxis(image, 0, -1)
                assert (
                    image.ndim == 3
                ), "The image has a wrong number of dimensions. Abort."

                if im_calib != train_spatial_calibration:
                    factor = im_calib / train_spatial_calibration
                    image = np.moveaxis(
                        [
                            zoom(
                                image[:, :, c].astype(float).copy(),
                                [factor, factor],
                                order=3,
                                prefilter=False,
                            )
                            for c in range(image.shape[-1])
                        ],
                        0,
                        -1,
                    )  # zoom(image, [factor,factor,1], order=3)
                    mask = zoom(mask, [factor, factor], order=0)

            X.append(image)
            Y.append(mask)

            # fig,ax = plt.subplots(1,image.shape[-1]+1)
            # for k in range(image.shape[-1]):
            # 	ax[k].imshow(image[:,:,k],cmap='gray')
            # ax[image.shape[-1]].imshow(mask)
            # plt.pause(1)
            # plt.close()

            files.append(im)

    assert len(X) == len(
        Y
    ), "The number of images does not match with the number of masks... Abort."
    return X, Y, files
