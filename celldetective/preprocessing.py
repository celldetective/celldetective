"""
Preprocessing Module
====================

This module provides tools for image preprocessing, focusing on background estimation and correction.
It enables the creation of flat-field corrections and background subtraction models to improve image quality for downstream analysis.

Key Features
------------
-   **Background Estimation**: Methods to estimate background from time-series or tiled acquisitions.
-   **Background Correction**: Applies estimated backgrounds to image stacks (subtraction or division).
-   **Surface Fitting**: Functions to fit 2D surfaces (planes, paraboloids) to image data, useful for uneven illumination correction.

Main Functions
--------------
-   `estimate_background_per_condition`: Generates background images for experimental conditions.
-   `correct_background_model_free`: Master function to apply background correction to an experiment.
-   `apply_background_to_stack`: Applies a specific background image to a single image stack.
-   `fit_plane`: Fits a plane model to an image, optionally excluding specific regions.

Notes
-----
The module relies heavily on the directory structure and configuration files of the experiment to locate and process images.
"""

from typing import List, Optional, Union, Callable, Any, Dict, Literal
import numpy as np
import os
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    load_frames,
    _get_img_num_per_channel,
)
from celldetective.utils.image_cleaning import interpolate_nan
from celldetective.utils.experiment import (
    get_experiment_wells,
    extract_well_name_and_number,
    extract_position_name,
    get_config,
    interpret_wells_and_positions,
    get_position_movie_path,
    get_positions_in_well,
)
from celldetective.utils.image_transforms import (
    estimate_unreliable_edge,
    unpad,
    threshold_image,
)
from celldetective.utils.parsing import (
    config_section_to_dict,
    _extract_channel_indices_from_config,
    _extract_nbr_channels_from_config,
)
from contextlib import nullcontext
from gc import collect
from itertools import chain
from tqdm import tqdm
from celldetective import get_logger
from celldetective.log_manager import positionlogger

logger = get_logger(__name__)


def _log_preprocessing_step(pos_path: str, correction_type: str, params: Dict[str, Any]) -> None:
    """
    Record the preprocessing parameters in a position-level log.

    Preprocessing (background correction) is common to every population analysis, so its
    parameters are recorded in a dedicated position-level log (``log_preprocessing.txt``)
    rather than in the population-specific ``log_{mode}.txt`` files. The parameters are
    emitted through the integrated ``celldetective`` logger; the :func:`positionlogger`
    context manager routes those records into the position folder for the duration of the
    call. A logging failure never aborts the correction itself.

    Parameters
    ----------
    pos_path : str
        Path to the position folder.
    correction_type : str
        The kind of correction applied (e.g. "model-free", "model", "offset").
    params : dict
        The parameters that define the correction.
    """

    try:
        with positionlogger(pos_path, filename="log_preprocessing.txt"):
            logger.info(f"PREPROCESS - correction_type: {correction_type}")
            for key, value in params.items():
                logger.info(f"PREPROCESS - {key}: {value}")
    except OSError as e:
        logger.warning(f"Could not write preprocessing log for {pos_path}: {e}")


def _correction_inputs(
    experiment: str, target_channel: str, movie_prefix: Optional[str]
):
    """
    Read the experiment settings shared by the per-position corrections.

    Returns
    -------
    tuple
        ``(len_movie, movie_prefix, channel_index, nbr_channels)``: the configured movie length,
        the movie prefix (``movie_prefix`` if given, else the configured one), the index of
        ``target_channel`` and the number of channels.
    """
    config = get_config(experiment)
    movie_settings = config_section_to_dict(config, "MovieSettings")
    if movie_prefix is None:
        movie_prefix = movie_settings["movie_prefix"]
    channel_index = _extract_channel_indices_from_config(config, [target_channel])[0]
    return (
        float(movie_settings["len_movie"]),
        movie_prefix,
        channel_index,
        _extract_nbr_channels_from_config(config),
    )


def _iter_wells(
    experiment: str,
    well_option,
    position_option,
    progress_callback: Optional[Callable] = None,
    show_progress: bool = True,
):
    """
    Yield ``(well_index, well_path, positions)`` for each selected well of an experiment.

    ``positions`` are the selected positions of the well. The well progress is reported before
    and after each well, including when the caller skips the rest of a well with ``continue``.
    """
    wells = get_experiment_wells(experiment)
    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )
    total = len(well_indices)
    for k, well_path in enumerate(tqdm(wells[well_indices], disable=not show_progress)):
        if progress_callback:
            progress_callback(level="well", iter=k, total=total)
        positions = get_positions_in_well(well_path)[position_indices]
        if isinstance(positions[0], np.ndarray):
            positions = positions[0]
        yield well_indices[k], well_path, positions
        if progress_callback:
            progress_callback(level="well", iter=k + 1, total=total)


def _iter_movies(
    positions,
    movie_prefix: Optional[str],
    len_movie: float,
    progress_callback: Optional[Callable] = None,
    show_progress: bool = True,
):
    """
    Yield ``(pos_path, stack_path, stack_length)`` for each position that holds a movie.

    ``stack_length`` is read from the movie, or ``len_movie`` if it cannot be. Positions without
    a movie are skipped with a warning. The position progress is reported once each position is
    done.
    """
    total = len(positions)
    for pidx, pos_path in enumerate(tqdm(positions, disable=not show_progress)):
        stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
        if stack_path is None:
            logger.warning(f"No stack could be found in {pos_path}... Skip...")
        else:
            stack_length = auto_load_number_of_frames(stack_path)
            yield pos_path, stack_path, (
                len_movie if stack_length is None else stack_length
            )
        if progress_callback:
            progress_callback(
                level="position", iter=pidx, total=total, stage="correcting"
            )


def _write_frames(
    frames,
    n_frames: int,
    stack_path: str,
    prefix: Optional[str],
    export: bool,
    return_stacks: bool,
) -> Optional[np.ndarray]:
    """
    Export and/or collect the corrected frames of a stack, one frame in memory at a time.

    Parameters
    ----------
    frames : iterable of numpy.ndarray
        Corrected ``(Y, X, C)`` frames, in temporal order.
    n_frames : int
        Number of frames in ``frames``.
    stack_path : str
        Source stack. The frames are written next to it as ``<prefix>_<file>``, or over it
        through a temporary file if ``prefix`` is None.
    prefix : str or None
        Prefix of the exported file.
    export : bool
        Whether to write the frames to disk.
    return_stacks : bool
        Whether to collect and return the frames.

    Returns
    -------
    numpy.ndarray or None
        The ``(T, Y, X, C)`` float32 stack if ``return_stacks`` is True, otherwise None.
    """
    import tifffile.tifffile as tiff

    path, file = os.path.split(stack_path)
    output_path = os.sep.join(
        [path, file if prefix is None else "_".join([prefix, file])]
    )
    write_path = os.sep.join([path, "temp_" + file]) if prefix is None else output_path
    writer = (
        tiff.TiffWriter(write_path, bigtiff=True, imagej=True)
        if export
        else nullcontext()
    )
    stack = None
    with writer as tif:
        for t, frame in enumerate(frames):
            if return_stacks:
                if stack is None:
                    stack = np.empty((n_frames,) + frame.shape, dtype=np.float32)
                stack[t] = frame
            if export:
                tif.write(
                    np.moveaxis(frame, -1, 0).astype(np.float32, copy=False),
                    contiguous=True,
                )
    if export and prefix is None:
        os.replace(write_path, output_path)
    return stack


def estimate_background_per_condition(
    experiment: str,
    threshold_on_std: float = 1,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    target_channel: str = "channel_name",
    frame_range: List[int] = [0, 5],
    mode: Literal["timeseries", "tiles"] = "timeseries",
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    show_progress_per_pos: bool = False,
    show_progress_per_well: bool = True,
    offset: Optional[float] = None,
    fix_nan: bool = False,
    progress_callback: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """
    Estimate the background for each condition in an experiment.

    This function calculates the background for each well within
    a given experiment by processing image frames using a specified activation
    protocol. It supports time-series and tile-based modes for background
    estimation.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    threshold_on_std : float, optional
            The threshold value on the standard deviation for masking (default is 1).
    well_option : str, optional
            The option to select specific wells (default is '*').
    target_channel : str, optional
            The name of the target channel for background estimation (default is "channel_name").
    frame_range : list of int, optional
            The range of frames to consider for background estimation (default is [0, 5]).
    mode : str, optional
            The mode of background estimation, either "timeseries" or "tiles" (default is "timeseries").
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss', 2], ['std', 4]]).
    show_progress_per_pos : bool, optional
            Whether to show progress for each position (default is False).
    show_progress_per_well : bool, optional
            Whether to show progress for each well (default is True).
    offset : float or None, optional
            A constant value to subtract from the background. Default is None.
    fix_nan : bool, optional
            Whether to interpolate NaN values in the background. Default is False.
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

    Returns
    -------
    list of dict
            A list of dictionaries, each containing the background image (`bg`) and the corresponding well path (`well`).

    See Also
    --------
    estimate_unreliable_edge : Estimates the unreliable edge value from the activation protocol.
    threshold_image : Thresholds an image based on the specified criteria.

    Notes
    -----
    This function assumes that the experiment directory structure and the configuration
    files follow a specific format expected by the helper functions used within.

    Examples
    --------
    >>> experiment_path = "path/to/experiment"
    >>> backgrounds = estimate_background_per_condition(experiment_path, threshold_on_std=1.5, target_channel="GFP", frame_range=[0, 10], mode="tiles")
    >>> for bg in backgrounds:
    ...     print(bg["well"], bg["bg"].shape)
    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
    movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, "*"
    )

    channel_indices = _extract_channel_indices_from_config(config, [target_channel])
    nbr_channels = _extract_nbr_channels_from_config(config)
    img_num_channels = _get_img_num_per_channel(
        channel_indices, int(len_movie), nbr_channels
    )

    backgrounds = []

    for k, well_path in enumerate(
        tqdm(wells[well_indices], disable=not show_progress_per_well)
    ):

        well_name, _ = extract_well_name_and_number(well_path)
        well_idx = well_indices[k]

        positions = get_positions_in_well(well_path)
        logger.info(
            f"Reconstruct a background in well {well_name} from positions: {[extract_position_name(p) for p in positions]}..."
        )

        frame_mean_per_position = []

        for l, pos_path in enumerate(
            tqdm(positions, disable=not show_progress_per_pos)
        ):
            if progress_callback is not None:
                should_continue = progress_callback(
                    level="position", iter=l, total=len(positions)
                )
                if should_continue is False:
                    logger.info("Background estimation cancelled by user.")
                    return None

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
            if stack_path is not None:
                len_movie_auto = auto_load_number_of_frames(stack_path)
                if len_movie_auto is not None:
                    len_movie = len_movie_auto
                    img_num_channels = _get_img_num_per_channel(
                        channel_indices, int(len_movie), nbr_channels
                    )

                from celldetective.filters import filter_image

                if mode == "timeseries":

                    frames = load_frames(
                        img_num_channels[0, frame_range[0] : frame_range[1]],
                        stack_path,
                        normalize_input=False,
                    )
                    frames = np.moveaxis(frames, -1, 0).astype(float)

                    for i in range(len(frames)):
                        if np.all(frames[i].flatten() == 0):
                            frames[i, :, :] = np.nan

                    frame_mean = np.nanmean(frames, axis=0)

                    frame = frame_mean.copy().astype(float)

                    std_frame = filter_image(frame.copy(), filters=activation_protocol)
                    edge = estimate_unreliable_edge(activation_protocol)
                    mask = threshold_image(
                        std_frame,
                        threshold_on_std,
                        np.inf,
                        foreground_value=1,
                        edge_exclusion=edge,
                    )
                    frame[np.where(mask.astype(int) == 1)] = np.nan

                elif mode == "tiles":

                    frames = load_frames(
                        img_num_channels[0, :], stack_path, normalize_input=False
                    ).astype(float)
                    frames = np.moveaxis(frames, -1, 0).astype(float)

                    new_frames = []
                    for i in range(len(frames)):

                        if np.all(frames[i].flatten() == 0):
                            empty_frame = np.zeros_like(frames[i])
                            empty_frame[:, :] = np.nan
                            new_frames.append(empty_frame)
                            continue

                        f = frames[i].copy()
                        std_frame = filter_image(f.copy(), filters=activation_protocol)
                        edge = estimate_unreliable_edge(activation_protocol)
                        mask = threshold_image(
                            std_frame,
                            threshold_on_std,
                            np.inf,
                            foreground_value=1,
                            edge_exclusion=edge,
                        )
                        f[np.where(mask.astype(int) == 1)] = np.nan
                        new_frames.append(f.copy())

                    frame = np.nanmedian(new_frames, axis=0)
                frame_mean_per_position.append(frame)
            else:
                # Left out of the median: an empty entry would make it fail for the whole well.
                logger.warning(f"Stack not found for position {pos_path}...")

            if progress_callback:
                progress_callback(
                    level="position", iter=l, total=len(positions), stage="estimating"
                )

        try:
            background = np.nanmedian(frame_mean_per_position, axis=0)
            if progress_callback:
                progress_callback(image_preview=background)

            if offset is not None:
                background -= offset
            if fix_nan:
                background = interpolate_nan(background.copy().astype(float))
            backgrounds.append({"bg": background, "well": well_path})
            logger.info(f"Background successfully computed for well {well_name}...")
        except Exception as e:
            logger.error(f"{e}")
            backgrounds.append(None)

    return backgrounds


def correct_background_model_free(
    experiment: str,
    mode: Literal["timeseries", "tiles"] = "timeseries",
    threshold_on_std: float = 1,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    position_option: Union[str, int, List[Union[str, int]]] = "*",
    target_channel: str = "channel_name",
    frame_range: List[int] = [0, 5],
    optimize_option: bool = False,
    opt_coef_range: Union[List[float], tuple[float, float]] = [0.95, 1.05],
    opt_coef_nbr: int = 100,
    operation: Literal["divide", "subtract"] = "divide",
    clip: bool = False,
    offset: Optional[float] = None,
    show_progress_per_well: bool = True,
    show_progress_per_pos: bool = False,
    export: bool = False,
    return_stacks: bool = False,
    movie_prefix: Optional[str] = None,
    fix_nan: bool = False,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    export_prefix: str = "Corrected",
    progress_callback: Optional[Callable] = None,
    **kwargs: Any,
) -> Optional[List[np.ndarray]]:
    """
    Correct the background of image stacks for a given experiment.

    This function processes image stacks by estimating and correcting the background
    for each well and position in the experiment. It supports different modes, such
    as timeseries or tiles, and offers options for optimization and exporting the results.

    Parameters
    ----------
    experiment : str
            Path to the experiment configuration.
    mode : {'timeseries', 'tiles'}, optional
            The mode of processing. Defaults to "timeseries".
    threshold_on_std : float, optional
            The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
    well_option : str, int, or list of int, optional
            Selection of wells to process. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            Selection of positions to process within each well. '*' indicates all positions. Defaults to '*'.
    target_channel : str, optional
            The name of the target channel to be corrected. Defaults to "channel_name".
    frame_range : list of int, optional
            The range of frames to consider for background estimation. Defaults to [0, 5].
    optimize_option : bool, optional
            If True, optimize the correction coefficient. Defaults to False.
    opt_coef_range : list of float or tuple of float, optional
            The range of coefficients to try for optimization. Defaults to [0.95, 1.05].
    opt_coef_nbr : int, optional
            The number of coefficients to test within the optimization range. Defaults to 100.
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction. Defaults to 'divide'.
    clip : bool, optional
            If True, clip the corrected values to be non-negative when using subtraction. Defaults to False.
    offset : float, optional
            A constant value to subtract from the background. Defaults to None.
    show_progress_per_well : bool, optional
            If True, show progress bar for each well. Defaults to True.
    show_progress_per_pos : bool, optional
            If True, show progress bar for each position. Defaults to False.
    export : bool, optional
            If True, export the corrected stacks to files. Defaults to False.
    return_stacks : bool, optional
            If True, return the corrected stacks as a list of numpy arrays. Defaults to False.
    movie_prefix : str, optional
            The prefix of the movie files. Defaults to None.
    fix_nan : bool, optional
            Whether to interpolate NaN values in the background. Defaults to False.
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss', 2], ['std', 4]]).
    export_prefix : str, optional
            The prefix for the exported file name. Defaults to "Corrected".
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).
    **kwargs : Any
            Additional keyword arguments.

    Returns
    -------
    list of numpy.ndarray, optional
            A list of corrected image stacks if `return_stacks` is True.

    Notes
    -----
    The function uses several helper functions, including `interpret_wells_and_positions`,
    `estimate_background_per_condition`, and `apply_background_to_stack`.

    Examples
    --------
    >>> experiment = "path/to/experiment/config"
    >>> corrected_stacks = correct_background_model_free(experiment, well_option=[0, 1], position_option='*', target_channel="DAPI", mode="timeseries", threshold_on_std=2, frame_range=[0, 10], optimize_option=True, operation='subtract', clip=True, return_stacks=True)
    >>> print(len(corrected_stacks))
    2

    """

    len_movie, movie_prefix, channel_index, nbr_channels = _correction_inputs(
        experiment, target_channel, movie_prefix
    )
    stacks = []

    for well_index, well_path, positions in _iter_wells(
        experiment,
        well_option,
        position_option,
        progress_callback=progress_callback,
        show_progress=show_progress_per_well,
    ):
        well_name, _ = extract_well_name_and_number(well_path)

        if progress_callback:
            progress_callback(status="Reconstructing background...")

        try:
            # Estimate background
            background = estimate_background_per_condition(
                experiment,
                threshold_on_std=threshold_on_std,
                well_option=int(well_index),
                target_channel=target_channel,
                frame_range=frame_range,
                mode=mode,
                show_progress_per_pos=True,
                show_progress_per_well=False,
                activation_protocol=activation_protocol,
                offset=offset,
                fix_nan=fix_nan,
                progress_callback=progress_callback,
            )
            background = background[0]["bg"]
        except Exception as e:
            logger.error(
                f'Background could not be estimated due to error "{e}"... Skipping well {well_name}...'
            )
            continue

        if progress_callback:
            progress_callback(
                level="position", iter=-1, total=1, status="Applying background..."
            )

        for pos_path, stack_path, stack_length in _iter_movies(
            positions,
            movie_prefix,
            len_movie,
            progress_callback=progress_callback,
            show_progress=show_progress_per_pos,
        ):
            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            corrected_stack = apply_background_to_stack(
                stack_path,
                background,
                target_channel_index=channel_index,
                nbr_channels=nbr_channels,
                stack_length=stack_length,
                threshold_on_std=threshold_on_std,
                optimize_option=optimize_option,
                opt_coef_range=opt_coef_range,
                opt_coef_nbr=opt_coef_nbr,
                operation=operation,
                clip=clip,
                offset=offset,
                export=export,
                fix_nan=fix_nan,
                activation_protocol=activation_protocol,
                prefix=export_prefix,
                progress_callback=progress_callback,
            )
            logger.info("Correction successful.")
            _log_preprocessing_step(
                pos_path,
                "model-free",
                {
                    "target_channel": target_channel,
                    "mode": mode,
                    "operation": operation,
                    "clip": clip,
                    "threshold_on_std": threshold_on_std,
                    "offset": offset,
                    "frame_range": frame_range,
                    "optimize_option": optimize_option,
                    "opt_coef_range": opt_coef_range,
                    "opt_coef_nbr": opt_coef_nbr,
                    "fix_nan": fix_nan,
                    "activation_protocol": activation_protocol,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                },
            )
            if return_stacks:
                stacks.append(corrected_stack)
            else:
                del corrected_stack
            collect()

    if return_stacks:
        return stacks


def _best_l1_coefficient(
    target: np.ndarray, background: np.ndarray, coefficients: np.ndarray
) -> float:
    """
    Coefficient of the grid minimizing ``sum(|target - c * background|)``.

    The loss is convex in ``c``, so along the sorted grid it decreases, then increases. A binary
    search for the first grid value after which the loss stops decreasing finds the minimum
    with a few loss evaluations instead of one per coefficient.

    Parameters
    ----------
    target : numpy.ndarray
        Finite background pixels of the frame, 1D.
    background : numpy.ndarray
        Finite background model values at the same pixels, 1D.
    coefficients : numpy.ndarray
        Grid of coefficients to choose from.

    Returns
    -------
    float
        The coefficient of the grid with the lowest loss, the smallest one on a tie.
    """

    grid = np.unique(coefficients)
    residual = np.empty_like(target, dtype=float)

    def loss(c):
        # One buffer for the whole evaluation instead of three full-size temporaries.
        np.multiply(background, c, out=residual)
        np.subtract(target, residual, out=residual)
        return np.abs(residual, out=residual).sum()

    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if loss(grid[mid]) <= loss(grid[mid + 1]):
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo])


def apply_background_to_stack(
    stack_path: str,
    background: np.ndarray,
    target_channel_index: int = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = 45,
    offset: Optional[float] = None,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    threshold_on_std: float = 1,
    optimize_option: bool = True,
    opt_coef_range: Union[List[float], tuple[float, float]] = (0.95, 1.05),
    opt_coef_nbr: int = 100,
    operation: Literal["divide", "subtract"] = "divide",
    clip: bool = False,
    export: bool = False,
    prefix: str = "Corrected",
    fix_nan: bool = False,
    progress_callback: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    Apply background correction to an image stack.

    This function corrects the background of an image stack by applying a specified operation
    (either division or subtraction) between the image stack and the background. It also supports
    optimization of the correction coefficient through brute-force regression.

    Parameters
    ----------
    stack_path : str
            The path to the image stack file.
    background : numpy.ndarray
            The background image to be applied for correction.
    target_channel_index : int, optional
            The index of the target channel to be corrected. Defaults to 0.
    nbr_channels : int, optional
            The number of channels in the image stack. Defaults to 1.
    stack_length : int, optional
            The length of the image stack (number of frames). If None, the length is auto-detected. Defaults to 45.
    offset : float or None, optional
            A constant value to subtract from the image. Default is None.
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss', 2], ['std', 4]]).
    fix_nan : bool, optional
            Whether to interpolate NaN values: those of the background once before it is
            applied, then those of a corrected frame only if some are left. Default is False.
    threshold_on_std : float, optional
            The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
    optimize_option : bool, optional
            If True, optimize the correction coefficient using a range of values. Defaults to True.
    opt_coef_range : list of float or tuple of float, optional
            The range of coefficients to try for optimization. Defaults to (0.95, 1.05).
    opt_coef_nbr : int, optional
            The number of coefficients to test within the optimization range. Defaults to 100.
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction. Defaults to 'divide'.
    clip : bool, optional
            If True, clip the corrected values to be non-negative when using subtraction. Defaults to False.
    export : bool, optional
            If True, export the corrected stack to a file. Defaults to False.
    prefix : str, optional
            The prefix for the exported file name. Defaults to "Corrected".
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

    Returns
    -------
    corrected_stack : numpy.ndarray, optional
            The background-corrected image stack.

    Examples
    --------
    >>> stack_path = "path/to/stack.tif"
    >>> background = np.zeros((512, 512))  # Example background
    >>> corrected_stack = apply_background_to_stack(stack_path, background, target_channel_index=0, nbr_channels=3, stack_length=45, optimize_option=False, operation='subtract', clip=True)
    >>> print(corrected_stack.shape)
    (44, 512, 512, 3)

    """
    import os
    import numpy as np

    if stack_length is None:
        stack_length = auto_load_number_of_frames(stack_path)
        if stack_length is None:
            logger.error("stack length not provided")
            return None

    if operation not in ("divide", "subtract"):
        logger.error("Operation not supported... Abort.")
        return None

    background = np.asarray(background, dtype=float)
    if fix_nan:
        background = interpolate_nan(background)
    bg_valid = np.isfinite(background)

    if optimize_option:
        from celldetective.filters import filter_image

        coefficients = np.linspace(
            opt_coef_range[0], opt_coef_range[1], int(opt_coef_nbr)
        )
        coefficients = np.append(coefficients, [1.0])
        edge = estimate_unreliable_edge(activation_protocol)
        bg_crop = unpad(background, edge)
        fit_region_crop = unpad(bg_valid, edge)
    if export:
        path, file = os.path.split(stack_path)
        if prefix is None:
            newfile = file
        else:
            newfile = "_".join([prefix, file])

    corrected_stack = None

    for t, i in enumerate(range(0, int(stack_length * nbr_channels), nbr_channels)):

        frames = load_frames(
            list(np.arange(i, (i + nbr_channels))), stack_path, normalize_input=False
        ).astype(float)
        target_img = frames[:, :, target_channel_index].copy()
        if offset is not None:
            target_img -= offset

        if optimize_option:

            std_frame = filter_image(target_img, filters=activation_protocol)
            mask = threshold_image(
                std_frame,
                threshold_on_std,
                np.inf,
                foreground_value=1,
                edge_exclusion=edge,
            )

            target_crop = unpad(target_img, edge)
            valid = fit_region_crop & ~unpad(mask, edge) & np.isfinite(target_crop)

            if not np.any(valid):
                logger.warning(
                    f"IFD {i}; no background pixel left to optimize the coefficient, "
                    "check the threshold... Using 1."
                )
                c = 1
            else:
                c = _best_l1_coefficient(target_crop[valid], bg_crop[valid], coefficients)
                logger.info(f"IFD {i}; optimal coefficient: {c}...")
        else:
            c = 1

        # NaN where the background is unknown; a NaN of the frame carries through.
        correction = np.full(target_img.shape, np.nan)
        apply = np.divide if operation == "divide" else np.subtract
        with np.errstate(divide="ignore", invalid="ignore"):
            apply(target_img, background * c, out=correction, where=bg_valid)
        if operation == "subtract" and clip:
            correction[correction <= 0.0] = 0.0

        invalid = ~np.isfinite(correction)
        if np.any(invalid):
            correction[invalid] = np.nan
            # The background is already interpolated: only a frame left with NaNs
            # (e.g. 0 / 0 where the background is 0) still needs it, and it is costly.
            if fix_nan:
                correction = interpolate_nan(correction)
        frames[:, :, target_channel_index] = correction
        if corrected_stack is None:
            # Filled in place, in the float32 of the exported file, rather than
            # stacked from a list of float64 frames at the end.
            corrected_stack = np.empty(
                (int(stack_length),) + frames.shape, dtype=np.float32
            )
        corrected_stack[t] = frames

        if progress_callback:
            progress_callback(
                level="frame",
                iter=i,
                total=int(stack_length * nbr_channels),
                stage="correcting",
            )

    if export:
        from celldetective.utils.io import save_tiff_imagej_compatible

        save_tiff_imagej_compatible(
            os.sep.join([path, newfile]), corrected_stack, axes="TYXC"
        )

    return corrected_stack


def paraboloid(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    g: float,
) -> Union[float, np.ndarray]:
    """
    Compute the value of a 2D paraboloid function.

    This function evaluates a paraboloid defined by the equation:
    `a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + g`.

    Parameters
    ----------
    x : float or numpy.ndarray
            The x-coordinate(s) at which to evaluate the paraboloid.
    y : float or numpy.ndarray
            The y-coordinate(s) at which to evaluate the paraboloid.
    a : float
            The coefficient of the x^2 term.
    b : float
            The coefficient of the y^2 term.
    c : float
            The coefficient of the x*y term.
    d : float
            The coefficient of the x term.
    e : float
            The coefficient of the y term.
    g : float
            The constant term.

    Returns
    -------
    float or numpy.ndarray
            The value of the paraboloid at the given (x, y) coordinates. If `x` and
            `y` are arrays, the result is an array of the same shape.

    Examples
    --------
    >>> paraboloid(1, 2, 1, 1, 0, 0, 0, 0)
    5
    >>> paraboloid(np.array([1, 2]), np.array([3, 4]), 1, 1, 0, 0, 0, 0)
    array([10, 20])

    Notes
    -----
    The paraboloid function is a quadratic function in two variables, commonly used
    to model surfaces in three-dimensional space.
    """

    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + g


def plane(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    a: float,
    b: float,
    c: float,
) -> Union[float, np.ndarray]:
    """
    Compute the value of a plane function.

    This function evaluates a plane defined by the equation:
    `a * x + b * y + c`.

    Parameters
    ----------
    x : float or numpy.ndarray
            The x-coordinate(s) at which to evaluate the plane.
    y : float or numpy.ndarray
            The y-coordinate(s) at which to evaluate the plane.
    a : float
            The coefficient of the x term.
    b : float
            The coefficient of the y term.
    c : float
            The constant term.

    Returns
    -------
    float or numpy.ndarray
            The value of the plane at the given (x, y) coordinates. If `x` and
            `y` are arrays, the result is an array of the same shape.

    Examples
    --------
    >>> plane(1, 2, 3, 4, 5)
    16
    >>> plane(np.array([1, 2]), np.array([3, 4]), 3, 4, 5)
    array([20, 27])

    Notes
    -----
    The plane function is a linear function in two variables, commonly used
    to model flat surfaces in three-dimensional space.
    """

    return a * x + b * y + c


def fit_plane(
    image: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    edge_exclusion: Optional[int] = None,
) -> np.ndarray:
    """
    Fit a plane to the given image data.

    This function fits a plane to the provided image data using least squares
    regression. It constructs a mesh grid based on the dimensions of the image
    and fits a plane model to the data points. If cell masks are provided,
    areas covered by cell masks will be excluded from the fitting process.

    Parameters
    ----------
    image : numpy.ndarray
            The input image data.
    cell_masks : numpy.ndarray, optional
            An array specifying cell masks. If provided, areas covered by cell masks
            will be excluded from the fitting process (default is None).
    edge_exclusion : int, optional
            The size of the edge to exclude from the fitting process (default is None).

    Returns
    -------
    numpy.ndarray
            The fitted plane.

    Notes
    -----
    - The `cell_masks` parameter allows excluding areas covered by cell masks from
      the fitting process.
    - The `edge_exclusion` parameter allows excluding edges of the specified size
      from the fitting process to avoid boundary effects.

    See Also
    --------
    plane : The plane function used for fitting.
    """

    data = np.empty(image.shape)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)

    from lmfit import Parameters, Model

    params = Parameters()
    params.add("a", value=1)
    params.add("b", value=1)
    params.add("c", value=1)

    model = Model(plane, independent_vars=["x", "y"])

    weights = np.ones_like(xx, dtype=float)
    if cell_masks is not None:
        weights[np.where(cell_masks > 0)] = 0.0

    if edge_exclusion is not None:
        xx = unpad(xx, edge_exclusion)
        yy = unpad(yy, edge_exclusion)
        weights = unpad(weights, edge_exclusion)
        image = unpad(image, edge_exclusion)

    result = model.fit(image, x=xx, y=yy, weights=weights, params=params, max_nfev=3000)
    del model
    collect()

    xx, yy = np.meshgrid(x, y)

    return plane(xx, yy, **result.params)


def fit_paraboloid(
    image: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    edge_exclusion: Optional[int] = None,
    downsample: int = 10,
) -> np.ndarray:
    """
    Fit a paraboloid to the given image data.

    This function fits a paraboloid to the provided image data using least squares
    regression. It constructs a mesh grid based on the dimensions of the image
    and fits a paraboloid model to the data points. If cell masks are provided,
    areas covered by cell masks will be excluded from the fitting process.

    Parameters
    ----------
    image : numpy.ndarray
            The input image data.
    cell_masks : numpy.ndarray, optional
            An array specifying cell masks. If provided, areas covered by cell masks
            will be excluded from the fitting process (default is None).
    edge_exclusion : int, optional
            The size of the edge to exclude from the fitting process (default is None).
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting.
            Default is 10.

    Returns
    -------
    numpy.ndarray
            The fitted paraboloid.

    Notes
    -----
    - The `cell_masks` parameter allows excluding areas covered by cell masks from
      the fitting process.
    - The `edge_exclusion` parameter allows excluding edges of the specified size
      from the fitting process to avoid boundary effects.
    - Downsampling significantly speeds up the fitting process for large images
      without compromising the accuracy of the low-frequency background estimate.

    See Also
    --------
    paraboloid : The paraboloid function used for fitting.
    """

    data = np.empty(image.shape)
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)

    from lmfit import Parameters, Model

    params = Parameters()
    params.add("a", value=1.0e-05)
    params.add("b", value=1.0e-05)
    params.add("c", value=1.0e-06)
    params.add("d", value=0.01)
    params.add("e", value=0.01)
    params.add("g", value=100)

    model = Model(paraboloid, independent_vars=["x", "y"])

    weights = np.ones_like(xx, dtype=float)
    if cell_masks is not None:
        weights[np.where(cell_masks > 0)] = 0.0

    if edge_exclusion is not None:
        xx = unpad(xx, edge_exclusion)
        yy = unpad(yy, edge_exclusion)
        weights = unpad(weights, edge_exclusion)
        image = unpad(image, edge_exclusion)

    # Downsample for faster fitting
    if downsample > 1:
        image_fit = image[::downsample, ::downsample]
        xx_fit = xx[::downsample, ::downsample]
        yy_fit = yy[::downsample, ::downsample]
        weights_fit = weights[::downsample, ::downsample]
    else:
        image_fit = image
        xx_fit = xx
        yy_fit = yy
        weights_fit = weights

    result = model.fit(
        image_fit, x=xx_fit, y=yy_fit, weights=weights_fit, params=params, max_nfev=3000
    )

    del model
    collect()

    xx, yy = np.meshgrid(x, y)

    return paraboloid(xx, yy, **result.params)


def correct_background_model(
    experiment: str,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    position_option: Union[str, int, List[Union[str, int]]] = "*",
    target_channel: str = "channel_name",
    threshold_on_std: float = 1,
    model: Literal["paraboloid", "plane"] = "paraboloid",
    operation: Literal["divide", "subtract"] = "divide",
    clip: bool = False,
    show_progress_per_well: bool = True,
    show_progress_per_pos: bool = False,
    export: bool = False,
    return_stacks: bool = False,
    movie_prefix: Optional[str] = None,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    export_prefix: str = "Corrected",
    progress_callback: Optional[Callable] = None,
    downsample: int = 10,
    **kwargs: Any,
) -> Optional[List[np.ndarray]]:
    """
    Correct background in image stacks using a specified model.

    This function corrects the background in image stacks obtained from an experiment
    using a specified background correction model. It supports various options for
    specifying wells, positions, target channel, and background correction parameters.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    well_option : str, int, or list of int, optional
            The option to select specific wells. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            The option to select specific positions. '*' indicates all positions. Defaults to '*'.
    target_channel : str, optional
            The name of the target channel for background correction (default is "channel_name").
    threshold_on_std : float, optional
            The threshold value on the standard deviation for masking (default is 1).
    model : {'paraboloid', 'plane'}, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction, either 'divide' or 'subtract' (default is 'divide').
    clip : bool, optional
            Whether to clip the corrected image to ensure non-negative values (default is False).
    show_progress_per_well : bool, optional
            Whether to show progress for each well (default is True).
    show_progress_per_pos : bool, optional
            Whether to show progress for each position (default is False).
    export : bool, optional
            Whether to export the corrected stacks (default is False).
    return_stacks : bool, optional
            Whether to return the corrected stacks (default is False).
    movie_prefix : str, optional
            The prefix for the movie files (default is None).
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss',2],['std',4]]).
    export_prefix : str, optional
            The prefix for exported corrected stacks (default is 'Corrected').
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting (default is 10).
    **kwargs : Any
            Additional keyword arguments to be passed to the underlying correction function.

    Returns
    -------
    list of numpy.ndarray, optional
            A list of corrected image stacks if `return_stacks` is True, otherwise None.

    Notes
    -----
    - This function assumes that the experiment directory structure and the configuration
      files follow a specific format expected by the helper functions used within.
    - Supported background correction models are 'paraboloid' and 'plane'.
    - Supported background correction operations are 'divide' and 'subtract'.

    See Also
    --------
    fit_and_apply_model_background_to_stack : Function to fit and apply background correction to an image stack.
    """

    len_movie, movie_prefix, channel_index, nbr_channels = _correction_inputs(
        experiment, target_channel, movie_prefix
    )
    stacks = []

    for _, _, positions in _iter_wells(
        experiment,
        well_option,
        position_option,
        progress_callback=progress_callback,
        show_progress=show_progress_per_well,
    ):
        for pos_path, stack_path, stack_length in _iter_movies(
            positions,
            movie_prefix,
            len_movie,
            progress_callback=progress_callback,
            show_progress=show_progress_per_pos,
        ):
            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            corrected_stack = fit_and_apply_model_background_to_stack(
                stack_path,
                target_channel_index=channel_index,
                model=model,
                nbr_channels=nbr_channels,
                stack_length=stack_length,
                threshold_on_std=threshold_on_std,
                operation=operation,
                clip=clip,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
                downsample=downsample,
                subset_indices=kwargs.get("subset_indices", None),
            )
            logger.info("Correction successful.")
            _log_preprocessing_step(
                pos_path,
                "model",
                {
                    "target_channel": target_channel,
                    "model": model,
                    "operation": operation,
                    "clip": clip,
                    "threshold_on_std": threshold_on_std,
                    "activation_protocol": activation_protocol,
                    "downsample": downsample,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                },
            )
            if return_stacks:
                stacks.append(corrected_stack)
            else:
                del corrected_stack
            collect()

    if return_stacks:
        return stacks


def fit_and_apply_model_background_to_stack(
    stack_path: str,
    target_channel_index: int = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = 45,
    threshold_on_std: float = 1,
    operation: Literal["divide", "subtract"] = "divide",
    model: Literal["paraboloid", "plane"] = "paraboloid",
    clip: bool = False,
    export: bool = False,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    prefix: str = "Corrected",
    return_stacks: bool = True,
    progress_callback: Optional[Callable] = None,
    downsample: int = 10,
    subset_indices: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """
    Fit and apply a background correction model to an image stack.

    This function fits a background correction model to each frame of the image stack
    and applies the correction accordingly. It supports various options for specifying
    the target channel, number of channels, stack length, threshold on standard deviation,
    correction operation, correction model, clipping, and export.

    Parameters
    ----------
    stack_path : str
            The path to the image stack.
    target_channel_index : int, optional
            The index of the target channel for background correction (default is 0).
    nbr_channels : int, optional
            The number of channels in the image stack (default is 1).
    stack_length : int, optional
            The length of the stack (default is 45).
    threshold_on_std : float, optional
            The threshold value on the standard deviation for masking (default is 1).
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction, either 'divide' or 'subtract' (default is 'divide').
    model : {'paraboloid', 'plane'}, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    clip : bool, optional
            Whether to clip the corrected image to ensure non-negative values (default is False).
    export : bool, optional
            Whether to export the corrected image stack (default is False).
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss',2],['std',4]]).
    prefix : str, optional
            The prefix for exported corrected stacks (default is 'Corrected').
    return_stacks : bool, optional
            Whether to return the corrected stacks (default is True).
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting (default is 10).
    subset_indices : list of int, optional
            List of absolute frame indices to process (default is None).

    Returns
    -------
    numpy.ndarray, optional
            The corrected image stack if `return_stacks` is True, otherwise None.

    Notes
    -----
    - The function loads frames from the image stack, applies background correction to each frame,
      and stores the corrected frames in a new stack.
    - Supported background correction models are 'paraboloid' and 'plane'.
    - Supported background correction operations are 'divide' and 'subtract'.

    See Also
    --------
    field_correction : Function to apply background correction to an image.
    """

    from tqdm import tqdm

    stack_length_auto = auto_load_number_of_frames(stack_path)
    if stack_length_auto is None and stack_length is None:
        logger.error("Stack length not provided...")
        return None
    if stack_length_auto is not None:
        stack_length = stack_length_auto

    # A subset of frames is only corrected for a preview, never exported.
    if subset_indices is None or export:
        frame_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
    else:
        frame_indices = subset_indices

    def corrected_frames():
        for i in tqdm(frame_indices):
            frames = load_frames(
                list(np.arange(i, (i + nbr_channels))),
                stack_path,
                normalize_input=False,
            ).astype(float)
            frames[:, :, target_channel_index] = field_correction(
                frames[:, :, target_channel_index].copy(),
                threshold=threshold_on_std,
                operation=operation,
                model=model,
                clip=clip,
                activation_protocol=activation_protocol,
                downsample=downsample,
            )
            yield frames

            if progress_callback:
                progress_callback(
                    level="frame",
                    iter=int(i // nbr_channels),
                    total=stack_length,
                    stage="correcting",
                )

    return _write_frames(
        corrected_frames(),
        len(frame_indices),
        stack_path,
        prefix,
        export,
        return_stacks,
    )


def field_correction(
    img: np.ndarray,
    threshold: float = 1,
    operation: Literal["divide", "subtract"] = "divide",
    model: Literal["paraboloid", "plane"] = "paraboloid",
    clip: bool = False,
    return_bg: bool = False,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    downsample: int = 10,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Apply field correction to an image.

    This function applies field correction to the given image based on the specified parameters
    including the threshold on standard deviation, operation, background correction model, clipping,
    and activation protocol.

    Parameters
    ----------
    img : numpy.ndarray
            The input image to be corrected.
    threshold : float, optional
            The threshold value on the image, post activation protocol for masking out cells (default is 1).
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction, either 'divide' or 'subtract' (default is 'divide').
    model : {'paraboloid', 'plane'}, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    clip : bool, optional
            Whether to clip the corrected image to ensure non-negative values (default is False).
    return_bg : bool, optional
            Whether to return the background along with the corrected image (default is False).
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss',2],['std',4]]).
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting (default is 10).

    Returns
    -------
    numpy.ndarray or tuple of (numpy.ndarray, numpy.ndarray)
            The corrected image or a tuple containing the corrected image and the background, depending on the value of `return_bg`.

    Notes
    -----
    - This function first estimates the unreliable edge based on the activation protocol.
    - It then applies thresholding to obtain a mask for the background.
    - Next, it fits a background model to the image using the specified model.
    - Depending on the operation specified, it either divides or subtracts the background from the image.
    - If `clip` is True and operation is 'subtract', negative values in the corrected image are clipped to 0.
    - If `return_bg` is True, the function returns a tuple containing the corrected image and the background.

    See Also
    --------
    fit_background_model : Function to fit a background model to an image.
    threshold_image : Function to apply thresholding to an image.
    """

    target_copy = img.copy().astype(float)
    if np.percentile(target_copy.flatten(), 99.9) == 0.0:
        return target_copy

    from celldetective.filters import filter_image

    std_frame = filter_image(target_copy, filters=activation_protocol)
    edge = estimate_unreliable_edge(activation_protocol)
    mask = threshold_image(
        std_frame, threshold, np.inf, foreground_value=1, edge_exclusion=edge
    ).astype(int)
    background = fit_background_model(
        img, cell_masks=mask, model=model, edge_exclusion=edge, downsample=downsample
    )

    if operation == "divide":
        correction = np.divide(img, background, where=background == background)
        correction[background != background] = np.nan
        correction[img != img] = np.nan
        fill_val = 1.0

    elif operation == "subtract":
        correction = np.subtract(img, background, where=background == background)
        correction[background != background] = np.nan
        correction[img != img] = np.nan
        fill_val = 0.0
        if clip:
            correction[correction <= 0.0] = 0.0

    if return_bg:
        return correction.copy(), background
    else:
        return correction.copy()


def fit_background_model(
    img: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    model: Literal["paraboloid", "plane"] = "paraboloid",
    edge_exclusion: Optional[int] = None,
    downsample: int = 10,
) -> Optional[np.ndarray]:
    """
    Fit a background model to the given image.

    This function fits a background model to the given image using either a paraboloid or plane model.
    It supports optional cell masks and edge exclusion for fitting.

    Parameters
    ----------
    img : numpy.ndarray
            The input image data.
    cell_masks : numpy.ndarray, optional
            An array specifying cell masks. If provided, areas covered by cell masks will be excluded from the fitting process.
    model : {'paraboloid', 'plane'}, optional
            The background model to fit, either 'paraboloid' or 'plane' (default is 'paraboloid').
    edge_exclusion : int or None, optional
            The size of the border to exclude from fitting (default is None).
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting (default is 10).

    Returns
    -------
    numpy.ndarray or None
            The fitted background model as a numpy array if successful, otherwise None.

    Notes
    -----
    - This function fits a background model to the image using either a paraboloid or plane model based on the specified `model`.
    - If `cell_masks` are provided, areas covered by cell masks will be excluded from the fitting process.
    - If `edge_exclusion` is provided, a border of the specified size will be excluded from fitting.

    See Also
    --------
    fit_paraboloid : Function to fit a paraboloid model to an image.
    fit_plane : Function to fit a plane model to an image.
    """

    bg: Optional[np.ndarray] = None
    if model == "paraboloid":
        bg = fit_paraboloid(
            img.astype(float),
            cell_masks=cell_masks,
            edge_exclusion=edge_exclusion,
            downsample=downsample,
        ).astype(float)
    elif model == "plane":
        bg = fit_plane(
            img.astype(float), cell_masks=cell_masks, edge_exclusion=edge_exclusion
        ).astype(float)

    if bg is not None:
        bg = np.array(bg)

    return bg


def correct_channel_offset(
    experiment: str,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    position_option: Union[str, int, List[Union[str, int]]] = "*",
    target_channel: str = "channel_name",
    correction_horizontal: int = 0,
    correction_vertical: int = 0,
    show_progress_per_well: bool = True,
    show_progress_per_pos: bool = True,
    export: bool = False,
    return_stacks: bool = False,
    movie_prefix: Optional[str] = None,
    export_prefix: str = "Corrected",
    progress_callback: Optional[Callable] = None,
    **kwargs: Any,
) -> Optional[List[np.ndarray]]:
    """
    Correct the channel shift (chromatic aberration) for an entire experiment.

    This function iterates through all selected wells and positions, correcting the channel offset
    for each specified target channel.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    well_option : str, int, or list of int, optional
            The option to select specific wells. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            The option to select specific positions. '*' indicates all positions. Defaults to '*'.
    target_channel : str, optional
            The name of the target channel for correction (default is "channel_name").
    correction_horizontal : int, optional
            The horizontal shift to apply (default is 0).
    correction_vertical : int, optional
            The vertical shift to apply (default is 0).
    show_progress_per_well : bool, optional
            Whether to show progress for each well (default is True).
    show_progress_per_pos : bool, optional
            Whether to show progress for each position (default is True).
    export : bool, optional
            Whether to export the corrected stacks (default is False).
    return_stacks : bool, optional
            Whether to return the corrected stacks (default is False).
    movie_prefix : str, optional
            The prefix for the movie files (default is None).
    export_prefix : str, optional
            The prefix for exported corrected stacks (default is 'Corrected').
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).
    **kwargs : Any
            Additional keyword arguments.

    Returns
    -------
    list of numpy.ndarray or None
            A list of corrected stacks if `return_stacks` is True, otherwise None.
    """

    len_movie, movie_prefix, channel_index, nbr_channels = _correction_inputs(
        experiment, target_channel, movie_prefix
    )
    stacks = []

    for _, _, positions in _iter_wells(
        experiment,
        well_option,
        position_option,
        progress_callback=progress_callback,
        show_progress=show_progress_per_well,
    ):
        for pos_path, stack_path, stack_length in _iter_movies(
            positions,
            movie_prefix,
            len_movie,
            progress_callback=progress_callback,
            show_progress=show_progress_per_pos,
        ):
            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            corrected_stack = correct_channel_offset_single_stack(
                stack_path,
                target_channel_index=channel_index,
                nbr_channels=nbr_channels,
                stack_length=stack_length,
                correction_vertical=correction_vertical,
                correction_horizontal=correction_horizontal,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
            )

            logger.info("Correction successful.")
            _log_preprocessing_step(
                pos_path,
                "offset",
                {
                    "target_channel": target_channel,
                    "correction_horizontal": correction_horizontal,
                    "correction_vertical": correction_vertical,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                },
            )
            if return_stacks:
                stacks.append(corrected_stack)
            else:
                del corrected_stack
            collect()

    if return_stacks:
        return stacks


def correct_channel_offset_single_stack(
    stack_path: str,
    target_channel_index: int = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = 45,
    correction_vertical: int = 0,
    correction_horizontal: int = 0,
    export: bool = False,
    prefix: str = "Corrected",
    return_stacks: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    Correct the channel shift for a single image stack.

    Parameters
    ----------
    stack_path : str
            The path to the image stack.
    target_channel_index : int, optional
            The index of the target channel to be corrected (default is 0).
    nbr_channels : int, optional
            The number of channels in the image stack (default is 1).
    stack_length : int, optional
            The length of the image stack (default is 45).
    correction_vertical : int, optional
            The vertical shift to apply (default is 0).
    correction_horizontal : int, optional
            The horizontal shift to apply (default is 0).
    export : bool, optional
            Whether to export the corrected stack (default is False).
    prefix : str, optional
            The prefix for the exported file name (default is 'Corrected').
    return_stacks : bool, optional
            Whether to return the corrected stack (default is True).
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

    Returns
    -------
    numpy.ndarray or None
            The corrected stack if `return_stacks` is True, otherwise None.
    """

    if not os.path.exists(stack_path):
        raise FileNotFoundError(f"The stack {stack_path} does not exist... Abort.")

    from tqdm import tqdm
    from scipy.ndimage import shift

    stack_length_auto = auto_load_number_of_frames(stack_path)
    if stack_length_auto is None and stack_length is None:
        logger.error("Stack length not provided...")
        return None
    if stack_length_auto is not None:
        stack_length = stack_length_auto

    frame_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
    shift_yx = [correction_vertical, correction_horizontal]

    def corrected_frames():
        for k, i in enumerate(tqdm(frame_indices)):
            if progress_callback:
                progress_callback(level="frame", iter=k, total=len(frame_indices))

            frames = load_frames(
                list(np.arange(i, (i + nbr_channels))),
                stack_path,
                normalize_input=False,
            ).astype(float)
            target_img = frames[:, :, target_channel_index].copy()

            if np.percentile(target_img.flatten(), 99.9) == 0.0:
                correction = target_img
            elif np.any(target_img.flatten() != target_img.flatten()):
                # Routine to interpolate NaN for the spline filter then mask it again
                correction = shift(interpolate_nan(target_img), shift_yx)
                correction_nan = shift(target_img, shift_yx, prefilter=False)
                correction[correction_nan != correction_nan] = np.nan
            else:
                correction = shift(target_img, shift_yx)

            frames[:, :, target_channel_index] = correction
            yield frames

    return _write_frames(
        corrected_frames(),
        len(frame_indices),
        stack_path,
        prefix,
        export,
        return_stacks,
    )


def register_stacks(
    experiment: str,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    position_option: Union[str, int, List[Union[str, int]]] = "*",
    target_channel: str = "channel_name",
    radius: Optional[float] = None,
    tukey_alpha: float = 0.25,
    upsample_factor: int = 10,
    reference: Literal["previous", "first"] = "previous",
    downscale: int = 1,
    show_progress_per_well: bool = True,
    show_progress_per_pos: bool = True,
    export: bool = False,
    return_stacks: bool = False,
    movie_prefix: Optional[str] = None,
    export_prefix: str = "Corrected",
    progress_callback: Optional[Callable] = None,
) -> Optional[List[np.ndarray]]:
    """
    Register the stacks of an experiment against their own drift.

    For each selected position, the drift is estimated on ``target_channel`` by Fourier phase
    cross-correlation of Tukey-windowed frames (see :mod:`celldetective.utils.registration`),
    then the same shift is applied to every channel of the frame. The shifts are written next to
    the stack as ``<export_prefix>_<movie>_registration_shifts.csv`` (``<movie>_registration_shifts.csv``
    if the source stack is overwritten).

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    well_option : str, int, or list of int, optional
            The option to select specific wells. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            The option to select specific positions. '*' indicates all positions. Defaults to '*'.
    target_channel : str, optional
            The registration channel, on which the drift is estimated.
    radius : float, optional
            Radius in pixels of the disk centred on the image inside which the correlation is
            computed. Structures outside are ignored. If None (default), the full frame is used.
    tukey_alpha : float, optional
            Fraction of the correlation region covered by the Tukey cosine taper (default 0.25).
    upsample_factor : int, optional
            Sub-pixel precision factor of the phase correlation (default 10).
    reference : {"previous", "first"}, optional
            Correlate each frame with the previous frame and accumulate the shifts (default), or
            with the first frame.
    downscale : int, optional
            Estimate the drift on the registration channel reduced by this factor, then apply
            the rescaled shifts at full resolution. 1 (default) disables downscaling.
    show_progress_per_well : bool, optional
            Whether to show progress for each well (default is True).
    show_progress_per_pos : bool, optional
            Whether to show progress for each position (default is True).
    export : bool, optional
            Whether to export the registered stacks (default is False).
    return_stacks : bool, optional
            Whether to return the registered stacks (default is False).
    movie_prefix : str, optional
            The prefix for the movie files (default is None).
    export_prefix : str, optional
            The prefix for exported stacks (default is 'Corrected'). If None, the source stack is
            overwritten.
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

    Returns
    -------
    list of numpy.ndarray or None
            A list of registered stacks if `return_stacks` is True, otherwise None.
    """

    len_movie, movie_prefix, channel_index, nbr_channels = _correction_inputs(
        experiment, target_channel, movie_prefix
    )
    stacks = []

    for _, _, positions in _iter_wells(
        experiment,
        well_option,
        position_option,
        progress_callback=progress_callback,
        show_progress=show_progress_per_well,
    ):
        for pos_path, stack_path, stack_length in _iter_movies(
            positions,
            movie_prefix,
            len_movie,
            progress_callback=progress_callback,
            show_progress=show_progress_per_pos,
        ):
            logger.info(
                f"Registering position {extract_position_name(pos_path)} on channel {target_channel}..."
            )
            registered_stack = register_single_stack(
                stack_path,
                registration_channel_index=channel_index,
                nbr_channels=nbr_channels,
                stack_length=stack_length,
                radius=radius,
                tukey_alpha=tukey_alpha,
                upsample_factor=upsample_factor,
                reference=reference,
                downscale=downscale,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
            )

            logger.info("Registration successful.")
            _log_preprocessing_step(
                pos_path,
                "registration",
                {
                    "target_channel": target_channel,
                    "radius": radius,
                    "tukey_alpha": tukey_alpha,
                    "upsample_factor": upsample_factor,
                    "reference": reference,
                    "downscale": downscale,
                    "movie_prefix": movie_prefix,
                    "export_prefix": export_prefix,
                },
            )
            if return_stacks:
                stacks.append(registered_stack)
            else:
                del registered_stack
            collect()

    if return_stacks:
        return stacks


def _shift_frame(img: np.ndarray, shift_yx: np.ndarray) -> np.ndarray:
    """
    Translate a 2D frame by a sub-pixel shift, preserving NaN regions.

    Pixels brought in from outside the field are set to 0, like the channel offset correction.
    """

    from scipy.ndimage import shift

    nan_mask = ~np.isfinite(img)
    if np.allclose(shift_yx, 0.0) or not np.any(img, where=~nan_mask):
        return img
    if not np.any(nan_mask):
        return shift(img, shift_yx, order=1, mode="constant", cval=0.0)

    shifted = shift(interpolate_nan(img), shift_yx, order=1, mode="constant", cval=0.0)
    shifted_mask = shift(nan_mask.astype(float), shift_yx, order=1, mode="constant", cval=0.0)
    shifted[shifted_mask > 0.5] = np.nan
    return shifted


def register_single_stack(
    stack_path: str,
    registration_channel_index: int = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = None,
    radius: Optional[float] = None,
    tukey_alpha: float = 0.25,
    upsample_factor: int = 10,
    reference: Literal["previous", "first"] = "previous",
    downscale: int = 1,
    export: bool = False,
    prefix: Optional[str] = "Corrected",
    return_stacks: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Optional[np.ndarray]:
    """
    Register a single multichannel stack by phase cross-correlation on one channel.

    The drift is estimated in a first pass that only loads the registration channel. A second
    pass loads each frame, shifts all of its channels and writes or collects the result, so a
    single frame is held in memory at a time when exporting.

    Parameters
    ----------
    stack_path : str
            The path to the image stack.
    registration_channel_index : int, optional
            Index of the channel on which the drift is estimated (default is 0).
    nbr_channels : int, optional
            The number of channels in the image stack (default is 1).
    stack_length : int, optional
            The number of frames, used if it cannot be read from the file.
    radius : float, optional
            Radius in full-scale pixels of the centred correlation disk. None uses the full frame.
    tukey_alpha : float, optional
            Fraction of the correlation region covered by the Tukey taper (default 0.25).
    upsample_factor : int, optional
            Sub-pixel precision factor, in full-scale pixels (default 10).
    reference : {"previous", "first"}, optional
            Reference frame for the correlation (default "previous").
    downscale : int, optional
            Block-averaging factor applied to the registration channel before the correlation.
            The shifts are multiplied back by it and applied to the full-resolution stack.
            1 (default) disables downscaling.
    export : bool, optional
            Whether to export the registered stack (default is False).
    prefix : str, optional
            Prefix for the exported file name (default 'Corrected'). If None, the source stack is
            overwritten.
    return_stacks : bool, optional
            Whether to return the registered stack (default is True).
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

    Returns
    -------
    numpy.ndarray or None
            The registered stack of shape (T, Y, X, C) if `return_stacks` is True, otherwise None.
    """

    if not os.path.exists(stack_path):
        raise FileNotFoundError(f"The stack {stack_path} does not exist... Abort.")

    from celldetective.utils.registration import (
        downscale_frame,
        estimate_drift,
        tukey_window,
    )

    downscale = int(downscale)
    if downscale < 1:
        raise ValueError(f"The downscaling factor must be at least 1, got {downscale}.")
    if upsample_factor < 1:
        raise ValueError(
            f"The upsampling factor must be at least 1, got {upsample_factor}."
        )

    stack_length_auto = auto_load_number_of_frames(stack_path)
    if stack_length_auto is None and stack_length is None:
        logger.error("Stack length not provided...")
        return None
    if stack_length_auto is not None:
        stack_length = stack_length_auto
    stack_length = int(stack_length)

    def registration_frames():
        for t in range(stack_length):
            frame = load_frames(
                [t * nbr_channels + registration_channel_index],
                stack_path,
                normalize_input=False,
            )[:, :, 0].astype(float)
            yield downscale_frame(frame, downscale)

    # The first frame gives the window shape, then goes back in front of the drift pass.
    frames_iter = registration_frames()
    first_frame = next(frames_iter)
    window = tukey_window(
        first_frame.shape,
        alpha=tukey_alpha,
        radius=None if radius is None else radius / downscale,
    )

    def drift_progress(iter):
        if progress_callback:
            progress_callback(level="frame", iter=iter, total=2 * stack_length)

    shifts = estimate_drift(
        chain([first_frame], frames_iter),
        window,
        reference=reference,
        # keep the requested precision in full-scale pixels
        upsample_factor=upsample_factor * downscale,
        progress_callback=drift_progress,
    ) * downscale
    logger.info(
        f"Estimated drift: max |dy|={np.abs(shifts[:, 0]).max():.2f} px, max |dx|={np.abs(shifts[:, 1]).max():.2f} px"
    )

    path, file = os.path.split(stack_path)
    # Named after the output movie, so registrations with different prefixes keep their shifts.
    output_file = file if prefix is None else "_".join([prefix, file])
    output_path = os.sep.join([path, output_file])
    shifts_path = os.path.splitext(output_path)[0] + "_registration_shifts.csv"
    np.savetxt(
        shifts_path,
        np.column_stack([np.arange(stack_length), shifts]),
        delimiter=",",
        header="FRAME,SHIFT_Y,SHIFT_X",
        comments="",
        fmt=["%d", "%.4f", "%.4f"],
    )

    def registered_frames():
        for t in range(stack_length):
            if progress_callback:
                progress_callback(
                    level="frame", iter=stack_length + t, total=2 * stack_length
                )
            i = t * nbr_channels
            frames = load_frames(
                list(np.arange(i, i + nbr_channels)),
                stack_path,
                normalize_input=False,
            ).astype(np.float32)
            for c in range(frames.shape[-1]):
                frames[:, :, c] = _shift_frame(frames[:, :, c], shifts[t])
            yield frames

    return _write_frames(
        registered_frames(), stack_length, stack_path, prefix, export, return_stacks
    )
