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
from gc import collect
from tqdm import tqdm
from celldetective import get_logger

logger = get_logger(__name__)


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
            else:
                logger.warning(f"Stack not found for position {pos_path}...")
                frame = []

            # store
            frame_mean_per_position.append(frame)

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

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
    if movie_prefix is None:
        movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )
    channel_indices = _extract_channel_indices_from_config(config, [target_channel])
    nbr_channels = _extract_nbr_channels_from_config(config)
    img_num_channels = _get_img_num_per_channel(
        channel_indices, int(len_movie), nbr_channels
    )

    stacks = []

    total_wells = len(wells[well_indices])

    for k, well_path in enumerate(
        tqdm(wells[well_indices], disable=not show_progress_per_well)
    ):
        if progress_callback:
            progress_callback(level="well", iter=k, total=total_wells)

        well_name, _ = extract_well_name_and_number(well_path)

        if progress_callback:
            progress_callback(status="Reconstructing background...")

        try:
            # Estimate background
            background = estimate_background_per_condition(
                experiment,
                threshold_on_std=threshold_on_std,
                well_option=int(well_indices[k]),
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
            background = background[0]
            background = background["bg"]
        except Exception as e:
            logger.error(
                f'Background could not be estimated due to error "{e}"... Skipping well {well_name}...'
            )
            if progress_callback:
                progress_callback(level="well", iter=k + 1, total=total_wells)
            if progress_callback:
                progress_callback(level="well", iter=k + 1, total=total_wells)
            continue

        if progress_callback:
            progress_callback(
                level="position", iter=-1, total=1, status="Applying background..."
            )

        positions = get_positions_in_well(well_path)
        selection = positions[position_indices]
        if isinstance(selection[0], np.ndarray):
            selection = selection[0]

        total_pos_in_well = len(selection)

        for pidx, pos_path in enumerate(
            tqdm(selection, disable=not show_progress_per_pos)
        ):

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            if stack_path is not None:
                len_movie_auto = auto_load_number_of_frames(stack_path)
                if len_movie_auto is not None:
                    len_movie = len_movie_auto
                img_num_channels = _get_img_num_per_channel(
                    channel_indices, int(len_movie), nbr_channels
                )

                corrected_stack = apply_background_to_stack(
                    stack_path,
                    background,
                    target_channel_index=channel_indices[0],
                    nbr_channels=nbr_channels,
                    stack_length=len_movie,
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
                if return_stacks:
                    stacks.append(corrected_stack)
                else:
                    del corrected_stack
                collect()
            else:
                stacks.append(None)

            if progress_callback:
                progress_callback(
                    level="position",
                    iter=pidx,
                    total=total_pos_in_well,
                    stage="correcting",
                )

        if progress_callback:
            progress_callback(level="well", iter=k + 1, total=total_wells)

    if return_stacks:
        return stacks


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
            Whether to interpolate NaN values in the corrected image. Default is False.
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

    if optimize_option:
        coefficients = np.linspace(
            opt_coef_range[0], opt_coef_range[1], int(opt_coef_nbr)
        )
        coefficients = np.append(coefficients, [1.0])
    if export:
        path, file = os.path.split(stack_path)
        if prefix is None:
            newfile = file
        else:
            newfile = "_".join([prefix, file])

    corrected_stack = []

    for i in range(0, int(stack_length * nbr_channels), nbr_channels):

        frames = load_frames(
            list(np.arange(i, (i + nbr_channels))), stack_path, normalize_input=False
        ).astype(float)
        target_img = frames[:, :, target_channel_index].copy()
        if offset is not None:
            target_img -= offset

        if optimize_option:

            target_copy = target_img.copy()

            from celldetective.segmentation import threshold_image
            from celldetective.filters import filter_image

            std_frame = filter_image(target_copy.copy(), filters=activation_protocol)
            edge = estimate_unreliable_edge(activation_protocol)
            mask = threshold_image(
                std_frame,
                threshold_on_std,
                np.inf,
                foreground_value=1,
                edge_exclusion=edge,
            )
            target_copy[np.where(mask.astype(int) == 1)] = np.nan

            loss = []

            # brute-force regression, could do gradient descent instead
            for c in coefficients:

                target_crop = unpad(target_copy, edge)
                bg_crop = unpad(background, edge)

                roi = np.zeros_like(target_crop).astype(int)
                roi[target_crop != target_crop] = 1
                roi[bg_crop != bg_crop] = 1

                diff = np.subtract(target_crop, c * bg_crop, where=roi == 0)
                s = np.sum(np.abs(diff, where=roi == 0), where=roi == 0)
                loss.append(s)

            c = coefficients[np.argmin(loss)]
            logger.info(f"IFD {i}; optimal coefficient: {c}...")
            # if c==min(coefficients) or c==max(coefficients):
            # 	print('Warning... The optimal coefficient is beyond the range provided... Please adjust your coefficient range...')
        else:
            c = 1

        if operation == "divide":
            correction = np.divide(
                target_img, background * c, where=background == background
            )
            correction[background != background] = np.nan
            correction[target_img != target_img] = np.nan

        elif operation == "subtract":
            correction = np.subtract(
                target_img, background * c, where=background == background
            )
            correction[background != background] = np.nan
            correction[target_img != target_img] = np.nan
            if clip:
                correction[correction <= 0.0] = 0.0
        else:
            logger.error("Operation not supported... Abort.")
            return

        correction[~np.isfinite(correction)] = np.nan
        if fix_nan:
            correction = interpolate_nan(correction.copy())
        frames[:, :, target_channel_index] = correction
        corrected_stack.append(frames)

        if progress_callback:
            progress_callback(
                level="frame",
                iter=i,
                total=int(stack_length * nbr_channels),
                stage="correcting",
            )

    corrected_stack = np.array(corrected_stack)

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


def gaussian_2d(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> Union[float, np.ndarray]:
    """
    Compute the value of a 2D Gaussian function.

    Parameters
    ----------
    x : float or numpy.ndarray
            The x-coordinate(s).
    y : float or numpy.ndarray
            The y-coordinate(s).
    amplitude : float
            Peak amplitude above offset.
    x0 : float
            Center x position.
    y0 : float
            Center y position.
    sigma_x : float
            Standard deviation along x.
    sigma_y : float
            Standard deviation along y.
    offset : float
            Constant background offset.

    Returns
    -------
    float or numpy.ndarray
            Gaussian evaluated at (x, y).
    """
    return amplitude * np.exp(
        -(((x - x0) ** 2) / (2 * sigma_x ** 2) + ((y - y0) ** 2) / (2 * sigma_y ** 2))
    ) + offset


def fit_plane(
    image: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    edge_exclusion: Optional[int] = None,
    downsample: int = 10,
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
    downsample : int, optional
            The downsampling factor to reduce the number of points used for fitting.
            Default is 10.

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
    - Downsampling significantly speeds up the fitting process for large images
      without compromising the accuracy of the low-frequency background estimate.

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

    result = model.fit(image_fit, x=xx_fit, y=yy_fit, weights=weights_fit, params=params, max_nfev=3000)
    del model
    collect()

    xx, yy = np.meshgrid(x, y)

    return plane(xx, yy, **result.params)


def fit_gaussian(
    image: np.ndarray,
    cell_masks: Optional[np.ndarray] = None,
    edge_exclusion: Optional[int] = None,
    downsample: int = 10,
) -> np.ndarray:
    """
    Fit a 2D Gaussian to the given image data.

    This is the physically accurate illumination model for Köhler-illuminated
    wide-field microscopy. Unlike the paraboloid approximation, the Gaussian
    rolls off correctly at the edges.

    Parameters
    ----------
    image : numpy.ndarray
            The input image data.
    cell_masks : numpy.ndarray, optional
            Areas covered by cell masks will be excluded from fitting (default None).
    edge_exclusion : int, optional
            Border width to exclude from fitting (default None).
    downsample : int, optional
            Downsampling factor for faster fitting (default 10).

    Returns
    -------
    numpy.ndarray
            The fitted Gaussian background.
    """
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)

    from lmfit import Parameters, Model

    params = Parameters()
    params.add("amplitude", value=float(np.percentile(image, 95) - np.percentile(image, 5)), min=0)
    params.add("x0", value=float(image.shape[1] / 2))
    params.add("y0", value=float(image.shape[0] / 2))
    params.add("sigma_x", value=float(image.shape[1] / 4), min=1)
    params.add("sigma_y", value=float(image.shape[0] / 4), min=1)
    params.add("offset", value=float(np.percentile(image, 5)))

    model = Model(gaussian_2d, independent_vars=["x", "y"])

    weights = np.ones_like(xx, dtype=float)
    if cell_masks is not None:
        weights[np.where(cell_masks > 0)] = 0.0

    if edge_exclusion is not None:
        xx = unpad(xx, edge_exclusion)
        yy = unpad(yy, edge_exclusion)
        weights = unpad(weights, edge_exclusion)
        image = unpad(image, edge_exclusion)

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
    return gaussian_2d(xx, yy, **result.params)


def fit_rolling_ball(
    image: np.ndarray,
    radius: float = 100,
    light_background: bool = False,
    smooth: bool = True,
) -> np.ndarray:
    """
    Estimate background using the rolling ball algorithm (ImageJ-style).

    Matches the behaviour of the ImageJ Rolling Ball Background plugin:
    adaptive downsampling based on radius (shrink factors 1/2/4/8), optional
    3×3 pre-smoothing of the shrunken image, bilinear upsampling back to full
    resolution, and a light-background mode for transmitted-light microscopy.

    Parameters
    ----------
    image : numpy.ndarray
            The input image data (2-D, any numeric dtype).
    radius : float, optional
            Rolling ball radius in pixels. Should be at least as large as the
            largest cell diameter. Default is 100.
    light_background : bool, optional
            Set True for transmitted-light images (phase contrast, brightfield)
            where the background is brighter than the foreground. The image is
            inverted before rolling and the result is inverted back. Default False.
    smooth : bool, optional
            Apply a 3×3 uniform pre-smoothing pass to the downsampled image
            before rolling, matching ImageJ's smoothing step. Reduces noise
            sensitivity at the cost of very slight blurring. Default True.

    Returns
    -------
    numpy.ndarray
            The estimated background at full resolution.

    Notes
    -----
    Adaptive shrink factors (identical to the ImageJ plugin):

    ============  =============
    radius (px)   shrink factor
    ============  =============
    ≤ 10          1 (no shrink)
    ≤ 50          2×
    ≤ 100         4×
    > 100         8×
    ============  =============

    The ball is rolled on the downsampled image (radius scaled accordingly),
    then the background is restored to full resolution via bilinear interpolation.
    """
    from skimage.restoration import rolling_ball as _rolling_ball

    img = image.astype(float)
    img_max = img.max()

    if light_background:
        img = img_max - img

    # Adaptive shrink factor matching ImageJ
    if radius <= 10:
        shrink = 1
    elif radius <= 50:
        shrink = 2
    elif radius <= 100:
        shrink = 4
    else:
        shrink = 8

    if shrink > 1:
        from skimage.measure import block_reduce
        from skimage.transform import resize
        from scipy.ndimage import uniform_filter

        # Downsample: minimum value in each block (ImageJ uses minimum)
        img_small = block_reduce(img, block_size=(shrink, shrink), func=np.min)

        if smooth:
            img_small = uniform_filter(img_small, size=3)

        # Roll on the shrunken image; radius scales with pixel size
        bg_small = _rolling_ball(img_small, radius=radius / shrink)

        # Bilinear interpolation back to full resolution
        bg = resize(bg_small, image.shape, order=1, mode="edge", anti_aliasing=False)
    else:
        if smooth:
            from scipy.ndimage import uniform_filter
            img = uniform_filter(img, size=3)
        bg = _rolling_ball(img, radius=radius)

    if light_background:
        bg = img_max - bg

    return bg


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
    model: Literal["paraboloid", "plane", "gaussian", "rolling_ball"] = "paraboloid",
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
    radius: float = 100,
    light_background: bool = False,
    smooth: bool = True,
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

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
    if movie_prefix is None:
        movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )
    channel_indices = _extract_channel_indices_from_config(config, [target_channel])
    nbr_channels = _extract_nbr_channels_from_config(config)
    img_num_channels = _get_img_num_per_channel(
        channel_indices, int(len_movie), nbr_channels
    )

    stacks = []

    total_wells = len(wells[well_indices])
    for k, well_path in enumerate(
        tqdm(wells[well_indices], disable=not show_progress_per_well)
    ):
        if progress_callback:
            progress_callback(level="well", iter=k, total=total_wells)

        well_name, _ = extract_well_name_and_number(well_path)
        positions = get_positions_in_well(well_path)
        selection = positions[position_indices]
        if isinstance(selection[0], np.ndarray):
            selection = selection[0]

        total_pos_in_well = len(selection)

        for pidx, pos_path in enumerate(
            tqdm(selection, disable=not show_progress_per_pos)
        ):

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
            if stack_path is None:
                logger.warning(f"No stack could be found in {pos_path}... Skip...")
                continue

            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            len_movie_auto = auto_load_number_of_frames(stack_path)
            if len_movie_auto is not None:
                len_movie = len_movie_auto
                img_num_channels = _get_img_num_per_channel(
                    channel_indices, int(len_movie), nbr_channels
                )

            corrected_stack = fit_and_apply_model_background_to_stack(
                stack_path,
                target_channel_index=channel_indices[0],
                model=model,
                nbr_channels=nbr_channels,
                stack_length=len_movie,
                threshold_on_std=threshold_on_std,
                operation=operation,
                clip=clip,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
                downsample=downsample,
                radius=radius,
                light_background=light_background,
                smooth=smooth,
                subset_indices=kwargs.get("subset_indices", None),
            )
            logger.info("Correction successful.")
            if return_stacks:
                stacks.append(corrected_stack)
            else:
                del corrected_stack
            collect()

            if progress_callback:
                progress_callback(
                    level="position",
                    iter=pidx,
                    total=total_pos_in_well,
                    stage="correcting",
                )

        if progress_callback:
            progress_callback(level="well", iter=k + 1, total=total_wells)

    if return_stacks:
        return stacks


def fit_and_apply_model_background_to_stack(
    stack_path: str,
    target_channel_index: int = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = 45,
    threshold_on_std: float = 1,
    operation: Literal["divide", "subtract"] = "divide",
    model: Literal["paraboloid", "plane", "gaussian", "rolling_ball"] = "paraboloid",
    clip: bool = False,
    export: bool = False,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    prefix: str = "Corrected",
    return_stacks: bool = True,
    progress_callback: Optional[Callable] = None,
    downsample: int = 10,
    radius: float = 100,
    light_background: bool = False,
    smooth: bool = True,
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

    corrected_stack = []

    if export:
        path, file = os.path.split(stack_path)
        if prefix is None:
            newfile = "temp_" + file
        else:
            newfile = "_".join([prefix, file])

        import tifffile.tifffile as tiff

        with tiff.TiffWriter(
            os.sep.join([path, newfile]), imagej=True, bigtiff=True
        ) as tif:

            for i in tqdm(range(0, int(stack_length * nbr_channels), nbr_channels)):

                frames = load_frames(
                    list(np.arange(i, (i + nbr_channels))),
                    stack_path,
                    normalize_input=False,
                ).astype(float)
                target_img = frames[:, :, target_channel_index].copy()

                correction = field_correction(
                    target_img,
                    threshold=threshold_on_std,
                    operation=operation,
                    model=model,
                    clip=clip,
                    activation_protocol=activation_protocol,
                    downsample=downsample,
                    radius=radius,
                    light_background=light_background,
                    smooth=smooth,
                )
                frames[:, :, target_channel_index] = correction.copy()

                if return_stacks:
                    corrected_stack.append(frames)

                if export:
                    tif.write(
                        np.moveaxis(frames, -1, 0).astype(np.dtype("f")),
                        contiguous=True,
                    )
                del frames
                del target_img
                del correction
                collect()

                if progress_callback:
                    progress_callback(
                        level="frame",
                        iter=int(i // nbr_channels),
                        total=stack_length,
                        stage="correcting",
                    )

        if prefix is None:
            os.replace(os.sep.join([path, newfile]), os.sep.join([path, file]))
    else:

        if subset_indices is None:
            iterator = range(0, int(stack_length * nbr_channels), nbr_channels)
        else:
            iterator = subset_indices

        for i in tqdm(iterator):

            frames = load_frames(
                list(np.arange(i, (i + nbr_channels))),
                stack_path,
                normalize_input=False,
            ).astype(float)
            target_img = frames[:, :, target_channel_index].copy()

            correction = field_correction(
                target_img,
                threshold=threshold_on_std,
                operation=operation,
                model=model,
                clip=clip,
                activation_protocol=activation_protocol,
                downsample=downsample,
                radius=radius,
            )
            frames[:, :, target_channel_index] = correction.copy()

            corrected_stack.append(frames)

            del frames
            del target_img
            del correction
            collect()

            if progress_callback:
                progress_callback(
                    level="frame",
                    iter=int(i // nbr_channels),
                    total=stack_length,
                    stage="correcting",
                )

    if return_stacks:
        return np.array(corrected_stack)
    else:
        return None


def field_correction(
    img: np.ndarray,
    threshold: float = 1,
    operation: Literal["divide", "subtract"] = "divide",
    model: Literal["paraboloid", "plane", "gaussian", "rolling_ball"] = "paraboloid",
    clip: bool = False,
    return_bg: bool = False,
    activation_protocol: List[List[Any]] = [["gauss", 2], ["std", 4]],
    downsample: int = 10,
    radius: float = 100,
    light_background: bool = False,
    smooth: bool = True,
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

    if model == "rolling_ball":
        # Rolling ball estimates background locally — no cell masking needed
        background = fit_background_model(
            img, model=model, radius=radius, light_background=light_background, smooth=smooth
        )
    else:
        from celldetective.filters import filter_image

        std_frame = filter_image(target_copy, filters=activation_protocol)
        edge = estimate_unreliable_edge(activation_protocol)
        mask = threshold_image(
            std_frame, threshold, np.inf, foreground_value=1, edge_exclusion=edge
        ).astype(int)
        background = fit_background_model(
            img, cell_masks=mask, model=model, edge_exclusion=edge, downsample=downsample, radius=radius
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
    model: Literal["paraboloid", "plane", "gaussian", "rolling_ball"] = "paraboloid",
    edge_exclusion: Optional[int] = None,
    downsample: int = 10,
    radius: float = 100,
    light_background: bool = False,
    smooth: bool = True,
) -> Optional[np.ndarray]:
    """
    Fit a background model to the given image.

    Parameters
    ----------
    img : numpy.ndarray
            The input image data.
    cell_masks : numpy.ndarray, optional
            Areas covered by cell masks will be excluded from fitting (not used for rolling_ball).
    model : {'paraboloid', 'plane', 'gaussian', 'rolling_ball'}, optional
            The background model to fit (default is 'paraboloid').
    edge_exclusion : int or None, optional
            Border width to exclude from fitting (default None).
    downsample : int, optional
            Downsampling factor for parametric fits (default 10).
    radius : float, optional
            Rolling ball radius in pixels — only used when model='rolling_ball' (default 100).
    light_background : bool, optional
            Invert before rolling for transmitted-light images (default False).
    smooth : bool, optional
            Apply 3×3 pre-smoothing on the downsampled image before rolling (default True).

    Returns
    -------
    numpy.ndarray or None
            The fitted background, or None on failure.

    See Also
    --------
    fit_paraboloid, fit_plane, fit_gaussian, fit_rolling_ball
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
            img.astype(float), cell_masks=cell_masks, edge_exclusion=edge_exclusion, downsample=downsample
        ).astype(float)
    elif model == "gaussian":
        bg = fit_gaussian(
            img.astype(float),
            cell_masks=cell_masks,
            edge_exclusion=edge_exclusion,
            downsample=downsample,
        ).astype(float)
    elif model == "rolling_ball":
        bg = fit_rolling_ball(
            img.astype(float),
            radius=radius,
            light_background=light_background,
            smooth=smooth,
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

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
    if movie_prefix is None:
        movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )
    channel_indices = _extract_channel_indices_from_config(config, [target_channel])
    nbr_channels = _extract_nbr_channels_from_config(config)
    img_num_channels = _get_img_num_per_channel(
        channel_indices, int(len_movie), nbr_channels
    )

    stacks = []

    # Well loop with progress reporting
    total_wells = len(well_indices)
    for k, well_path in enumerate(wells[well_indices]):
        if progress_callback:
            progress_callback(level="well", iter=k, total=total_wells)
        elif show_progress_per_well:
            logger.info(f"Processing well {k+1}/{total_wells}...")

        well_name, _ = extract_well_name_and_number(well_path)
        positions = get_positions_in_well(well_path)
        selection = positions[position_indices]
        if isinstance(selection[0], np.ndarray):
            selection = selection[0]

        total_pos = len(selection)
        for pidx, pos_path in enumerate(selection):
            if progress_callback:
                progress_callback(
                    level="position",
                    iter=pidx,
                    total=total_pos,
                    stage=f"Pos {extract_position_name(pos_path)}",
                )
            elif show_progress_per_pos:
                logger.info(f"  Processing position {pidx+1}/{total_pos}...")

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
            logger.info(
                f"Applying the correction to position {extract_position_name(pos_path)}..."
            )
            len_movie_auto = auto_load_number_of_frames(stack_path)
            if len_movie_auto is not None:
                len_movie = len_movie_auto
                img_num_channels = _get_img_num_per_channel(
                    channel_indices, int(len_movie), nbr_channels
                )

            corrected_stack = correct_channel_offset_single_stack(
                stack_path,
                target_channel_index=channel_indices[0],
                nbr_channels=nbr_channels,
                stack_length=len_movie,
                correction_vertical=correction_vertical,
                correction_horizontal=correction_horizontal,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
            )

            logger.info("Correction successful.")
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
    import tifffile.tifffile as tiff
    from scipy.ndimage import shift

    stack_length_auto = auto_load_number_of_frames(stack_path)
    if stack_length_auto is None and stack_length is None:
        logger.error("Stack length not provided...")
        return None
    if stack_length_auto is not None:
        stack_length = stack_length_auto

    corrected_stack = []

    if export:
        path, file = os.path.split(stack_path)
        if prefix is None:
            newfile = "temp_" + file
        else:
            newfile = "_".join([prefix, file])

        with tiff.TiffWriter(
            os.sep.join([path, newfile]), bigtiff=True, imagej=True
        ) as tif:
            frames_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
            total_frames = len(frames_indices)
            for k, i in enumerate(tqdm(frames_indices)):
                if progress_callback:
                    progress_callback(level="frame", iter=k, total=total_frames)

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
                    target_interp = interpolate_nan(target_img)
                    from scipy.ndimage import shift

                    correction = shift(
                        target_interp, [correction_vertical, correction_horizontal]
                    )
                    correction_nan = shift(
                        target_img,
                        [correction_vertical, correction_horizontal],
                        prefilter=False,
                    )
                    nan_i, nan_j = np.where(correction_nan != correction_nan)
                    correction[nan_i, nan_j] = np.nan
                else:
                    correction = shift(
                        target_img, [correction_vertical, correction_horizontal]
                    )

                frames[:, :, target_channel_index] = correction.copy()

                if return_stacks:
                    corrected_stack.append(frames)

                if export:
                    tif.write(
                        np.moveaxis(frames, -1, 0).astype(np.dtype("f")),
                        contiguous=True,
                    )
                del frames
                del target_img
                del correction
                collect()

        if prefix is None:
            os.replace(os.sep.join([path, newfile]), os.sep.join([path, file]))
    else:
        frames_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
        total_frames = len(frames_indices)
        for k, i in enumerate(tqdm(frames_indices)):
            if progress_callback:
                progress_callback(level="frame", iter=k, total=total_frames)

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
                target_interp = interpolate_nan(target_img)
                correction = shift(
                    target_interp, [correction_vertical, correction_horizontal]
                )
                correction_nan = shift(
                    target_img,
                    [correction_vertical, correction_horizontal],
                    prefilter=False,
                )
                nan_i, nan_j = np.where(correction_nan != correction_nan)
                correction[nan_i, nan_j] = np.nan
            else:
                correction = shift(
                    target_img, [correction_vertical, correction_horizontal]
                )

            frames[:, :, target_channel_index] = correction.copy()

            corrected_stack.append(frames)

            del frames
            del target_img
            del correction
            collect()

    if return_stacks:
        return np.array(corrected_stack)
    else:
        return None


def _estimate_spt_shifts(
    tif,
    consensus_channels: List[int],
    ref_indices_by_chan: Dict[int, np.ndarray],
    reference_frame_idx: int,
    total_frames: int,
    sigma: float = 1.0,
    min_distance: float = 15.0,
    detection_threshold: float = 0.1,
    search_range: float = 5.0,
    memory: int = 1,
    sliding: bool = False,
    max_shift: float = 0.0,
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Estimate shifts using Single-Particle Tracking (SPT) on the first consensus channel.
    """
    import pandas as pd
    import trackpy as tp
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    from tqdm import tqdm

    # 1. Spot detection on the first consensus channel
    tracking_chan = consensus_channels[0]
    ref_frames = ref_indices_by_chan[tracking_chan]

    detections = []

    logger.info(f"Running spot detection for SPT on channel {tracking_chan}...")
    for k in tqdm(range(total_frames), desc="SPT Spot Detection"):
        if progress_callback:
            progress_callback(level="frame", iter=k, total=total_frames, stage="SPT Spot Detection")

        frame_num = ref_frames[k]
        img = tif.pages[int(frame_num)].asarray().astype(float)

        # Safe interpolation of NaNs
        if np.any(img != img):
            img = interpolate_nan(img)

        # Min-max normalization for relative thresholding
        img_min, img_max = np.nanmin(img), np.nanmax(img)
        if img_max > img_min:
            img_norm = (img - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img)

        # Gaussian smoothing
        if sigma > 0:
            img_smooth = gaussian_filter(img_norm, sigma=sigma)
        else:
            img_smooth = img_norm

        # Find peaks
        coords = peak_local_max(
            img_smooth,
            min_distance=int(min_distance),
            threshold_rel=detection_threshold,
            exclude_border=False
        )

        for y, x in coords:
            detections.append({"frame": k, "y": float(y), "x": float(x)})

    if len(detections) == 0:
        logger.warning("No spots detected across the entire stack for SPT registration.")
        shifts = [np.array([0.0, 0.0]) for _ in range(total_frames)]
        return shifts, list(range(total_frames)), [0] * total_frames

    df_spots = pd.DataFrame(detections)

    # 2. Particle linking via Trackpy
    logger.info(f"Linking spots via trackpy (search_range={search_range}, memory={memory})...")
    # Silence trackpy output to keep progress clean
    import logging
    tp_logger = logging.getLogger("trackpy")
    tp_logger.setLevel(logging.WARNING)

    try:
        df_tracks = tp.link_df(df_spots, search_range=search_range, memory=memory)
    except Exception as e:
        logger.error(f"Trackpy linking failed: {e}")
        shifts = [np.array([0.0, 0.0]) for _ in range(total_frames)]
        return shifts, list(range(total_frames)), [0] * total_frames

    if "particle" not in df_tracks.columns or df_tracks["particle"].nunique() == 0:
        logger.warning("Trackpy linking found no active trajectories.")
        shifts = [np.array([0.0, 0.0]) for _ in range(total_frames)]
        return shifts, list(range(total_frames)), [0] * total_frames

    # 3. Calculate frame shifts
    shifts = [np.array([0.0, 0.0]) for _ in range(total_frames)]
    fallbacks = []
    inliers_list = [0] * total_frames

    last_valid_shift = np.array([0.0, 0.0])

    if sliding:
        step_shifts = {0: np.array([0.0, 0.0])}
        inliers_list[0] = len(df_tracks[df_tracks["frame"] == 0])
        for k in range(1, total_frames):
            df_prev = df_tracks[df_tracks["frame"] == k - 1].set_index("particle")
            df_curr = df_tracks[df_tracks["frame"] == k].set_index("particle")
            common = df_prev.index.intersection(df_curr.index)
            inliers_list[k] = len(common)

            if len(common) > 0:
                diffs = df_prev.loc[common, ["y", "x"]].values - df_curr.loc[common, ["y", "x"]].values
                step_shift = np.nanmedian(diffs, axis=0)

                # Verify max shift constraint
                if max_shift > 0 and np.linalg.norm(step_shift) > max_shift:
                    logger.warning(f"Frame {k}: Stepwise SPT shift {step_shift} exceeds max_shift ({max_shift}). Fallback applied.")
                    step_shift = np.array([0.0, 0.0])
                    fallbacks.append(k)
            else:
                step_shift = np.array([0.0, 0.0])
                fallbacks.append(k)

            step_shifts[k] = step_shift

        # Accumulate shifts relative to reference_frame_idx
        for k in range(reference_frame_idx + 1, total_frames):
            shifts[k] = shifts[k-1] + step_shifts[k]
        for k in range(reference_frame_idx - 1, -1, -1):
            shifts[k] = shifts[k+1] - step_shifts[k+1]

    else:
        df_ref = df_tracks[df_tracks["frame"] == reference_frame_idx].set_index("particle")
        for k in range(total_frames):
            if k == reference_frame_idx:
                shifts[k] = np.array([0.0, 0.0])
                inliers_list[k] = len(df_ref)
                continue

            df_curr = df_tracks[df_tracks["frame"] == k].set_index("particle")
            common = df_ref.index.intersection(df_curr.index)
            inliers_list[k] = len(common)

            shift_vector = None
            if len(common) > 0:
                diffs = df_ref.loc[common, ["y", "x"]].values - df_curr.loc[common, ["y", "x"]].values
                shift_vector = np.nanmedian(diffs, axis=0)

                if max_shift > 0 and np.linalg.norm(shift_vector) > max_shift:
                    logger.warning(f"Frame {k}: SPT shift {shift_vector} exceeds max_shift ({max_shift}). Fallback applied.")
                    shift_vector = last_valid_shift
                    fallbacks.append(k)
                else:
                    last_valid_shift = shift_vector
            else:
                shift_vector = last_valid_shift
                fallbacks.append(k)

            shifts[k] = shift_vector

    return shifts, fallbacks, inliers_list


def register_experiment_fourier(
    experiment: str,
    well_option: Union[str, int, List[Union[str, int]]] = "*",
    position_option: Union[str, int, List[Union[str, int]]] = "*",
    reference_channel: str = "channel_name",
    reference_frame_idx: int = 0,
    upsample_factor: int = 1,
    sliding: bool = False,
    order: int = 1,
    show_progress_per_well: bool = True,
    show_progress_per_pos: bool = True,
    export: bool = False,
    return_stacks: bool = False,
    movie_prefix: Optional[str] = None,
    export_prefix: str = "Aligned",
    progress_callback: Optional[Callable] = None,
    sigma: float = 1.0,
    max_shift: float = 0.0,
    filter_outliers: bool = False,
    method: str = "fourier",
    shift_method: str = "spatial",
    **kwargs: Any,
) -> Optional[List[np.ndarray]]:
    """
    Register the image stacks for an entire experiment based on Fourier cross-correlation.

    This function iterates through all selected wells and positions, calculating the translation shifts
    from the selected reference channel and applying them to align all channels.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    well_option : str, int, or list of int, optional
            The option to select specific wells. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            The option to select specific positions. '*' indicates all positions. Defaults to '*'.
    reference_channel : str, optional
            The name of the reference channel for calculating shifts (default is "channel_name").
    reference_frame_idx : int, optional
            The index of the frame to use as reference template (default is 0).
    upsample_factor : int, optional
            Upsampling factor for subpixel accuracy (default is 1).
    sliding : bool, optional
            Whether to register each frame relative to the previous frame (sliding) rather than to a fixed anchor frame (default is False).
    order : int, optional
            Interpolation order for shifting (default is 1, bilinear).
    show_progress_per_well : bool, optional
            Whether to show progress for each well (default is True).
    show_progress_per_pos : bool, optional
            Whether to show progress for each position (default is True).
    export : bool, optional
            Whether to export the aligned stacks (default is False).
    return_stacks : bool, optional
            Whether to return the aligned stacks (default is False).
    movie_prefix : str, optional
            The prefix for the movie files (default is None).
    export_prefix : str, optional
            The prefix for exported registered stacks (default is 'Aligned').
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).
    **kwargs : Any
            Additional keyword arguments.

    Returns
    -------
    list of numpy.ndarray or None
            A list of registered stacks if `return_stacks` is True, otherwise None.
    """

    config = get_config(experiment)
    wells = get_experiment_wells(experiment)
    len_movie = float(config_section_to_dict(config, "MovieSettings")["len_movie"])
    if movie_prefix is None:
        movie_prefix = config_section_to_dict(config, "MovieSettings")["movie_prefix"]

    well_indices, position_indices = interpret_wells_and_positions(
        experiment, well_option, position_option
    )
    
    if isinstance(reference_channel, str):
        if "," in reference_channel:
            ref_channels = [c.strip() for c in reference_channel.split(",")]
        else:
            ref_channels = [reference_channel]
    elif isinstance(reference_channel, list):
        ref_channels = reference_channel
    else:
        ref_channels = [reference_channel]

    channel_indices = _extract_channel_indices_from_config(config, ref_channels)
    nbr_channels = _extract_nbr_channels_from_config(config)

    stacks = []

    total_wells = len(well_indices)
    for k, well_path in enumerate(wells[well_indices]):
        if progress_callback:
            progress_callback(level="well", iter=k, total=total_wells)
        elif show_progress_per_well:
            logger.info(f"Processing well {k+1}/{total_wells}...")

        well_name, _ = extract_well_name_and_number(well_path)
        positions = get_positions_in_well(well_path)
        selection = positions[position_indices]
        if isinstance(selection[0], np.ndarray):
            selection = selection[0]

        total_pos = len(selection)
        for pidx, pos_path in enumerate(selection):
            if progress_callback:
                progress_callback(
                    level="position",
                    iter=pidx,
                    total=total_pos,
                    stage=f"Pos {extract_position_name(pos_path)}",
                )
            elif show_progress_per_pos:
                logger.info(f"  Processing position {pidx+1}/{total_pos}...")

            stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)
            logger.info(
                f"Applying registration to position {extract_position_name(pos_path)}..."
            )
            len_movie_auto = auto_load_number_of_frames(stack_path)
            if len_movie_auto is not None:
                len_movie = len_movie_auto

            aligned_stack = register_stack_fourier_single_stack(
                stack_path,
                target_channel_index=channel_indices[0] if len(channel_indices) == 1 else channel_indices,
                nbr_channels=nbr_channels,
                stack_length=len_movie,
                reference_frame_idx=reference_frame_idx,
                upsample_factor=upsample_factor,
                sliding=sliding,
                order=order,
                export=export,
                prefix=export_prefix,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
                sigma=sigma,
                max_shift=max_shift,
                filter_outliers=filter_outliers,
                method=method,
                shift_method=shift_method,
            )

            logger.info("Registration successful.")
            if return_stacks:
                stacks.append(aligned_stack)
            else:
                del aligned_stack
            collect()

    if return_stacks:
        return stacks


def _estimate_sift_translation(
    ref_img: np.ndarray,
    curr_img: np.ndarray,
    ratio_threshold: float = 0.8,
    ransac_threshold: float = 2.0
):
    """
    Estimate relative translation between two images using SIFT feature matching and robust RANSAC.
    """
    import numpy as np
    
    try:
        import cv2
        
        # 1. Normalize images to uint8
        ref_uint8 = cv2.normalize(ref_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        curr_uint8 = cv2.normalize(curr_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. Detect keypoints and descriptors
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(ref_uint8, None)
        kp_curr, des_curr = sift.detectAndCompute(curr_uint8, None)
        
        if des_ref is None or des_curr is None or len(des_ref) < 2 or len(des_curr) < 2:
            return None, 0, 0
            
        # 3. Match descriptors using KNN (k=2)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(des_ref, des_curr, k=2)
        
        # 4. Filter matches using Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
                
        if len(good_matches) < 3:
            return None, len(good_matches), 0
            
        # 5. Extract coordinate points
        src_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]) # (N, 2) in (x, y)
        dst_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])  # (N, 2) in (x, y)
        
        # 6. Robust translation consensus via RANSAC
        shifts = dst_pts - src_pts # (N, 2) [dx, dy]
        best_inliers = []
        best_shift_xy = np.array([0.0, 0.0])
        
        for i in range(len(shifts)):
            hyp = shifts[i]
            errors = np.linalg.norm(shifts - hyp, axis=1)
            inliers = np.where(errors < ransac_threshold)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_shift_xy = hyp
                
        if len(best_inliers) > 0:
            # Recompute shift as mean of inliers
            best_shift_xy = np.mean(shifts[best_inliers], axis=0)
            
        # Reverse coordinate order from (x, y) to (row, col) i.e. (dy, dx) for scipy.ndimage.shift
        shift_vector = np.array([best_shift_xy[1], best_shift_xy[0]])
        return shift_vector, len(good_matches), len(best_inliers)
        
    except ImportError:
        # Fallback to scikit-image SIFT
        try:
            from skimage.feature import SIFT, match_descriptors
            
            sift = SIFT()
            sift.detect_and_extract(ref_img)
            kp_ref = sift.keypoints
            des_ref = sift.descriptors
            
            sift.detect_and_extract(curr_img)
            kp_curr = sift.keypoints
            des_curr = sift.descriptors
            
            if des_ref is None or des_curr is None or len(des_ref) < 2 or len(des_curr) < 2:
                return None, 0, 0
                
            matches = match_descriptors(des_ref, des_curr, cross_check=True)
            if len(matches) < 3:
                return None, len(matches), 0
                
            # keypoints in skimage are [row, col] (i.e. [y, x])
            src_pts = kp_curr[matches[:, 1]] # (N, 2)
            dst_pts = kp_ref[matches[:, 0]]  # (N, 2)
            
            shifts = dst_pts - src_pts # (N, 2) [dy, dx]
            best_inliers = []
            best_shift = np.array([0.0, 0.0])
            for i in range(len(shifts)):
                hyp = shifts[i]
                errors = np.linalg.norm(shifts - hyp, axis=1)
                inliers = np.where(errors < ransac_threshold)[0]
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_shift = hyp
            if len(best_inliers) > 0:
                best_shift = np.mean(shifts[best_inliers], axis=0)
            return best_shift, len(matches), len(best_inliers)
        except ImportError:
            raise ImportError("Either opencv-python (cv2) or scikit-image is required for SIFT registration.")


def _shift_fourier(img: np.ndarray, shift_vector: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Perform sub-pixel translation in the Fourier domain to avoid spatial spline interpolation artifacts.
    To match spatial shifting behavior (padding boundaries), it computes out-of-bounds coordinates
    and replaces wrapped pixels with `fill_value`.
    """
    import scipy.fft as fft
    import numpy as np

    ny, nx = img.shape
    
    # 1. Fourier Shift
    v = fft.fftfreq(ny)
    u = fft.fftfreq(nx)
    V, U = np.meshgrid(v, u, indexing="ij")
    phase = np.exp(-2j * np.pi * (V * shift_vector[0] + U * shift_vector[1]))
    
    img_fft = fft.fft2(img)
    shifted_fft = img_fft * phase
    shifted_img = np.real(fft.ifft2(shifted_fft))
    
    # 2. Boundary Masking to prevent wrap-around (periodic) artifacts
    dy, dx = shift_vector[0], shift_vector[1]
    y_coords = np.arange(ny)
    x_coords = np.arange(nx)
    
    # source_y = y - dy
    # out of bounds if source_y < 0 or source_y > ny - 1
    mask_y = (y_coords < dy) | (y_coords > ny - 1 + dy)
    # source_x = x - dx
    # out of bounds if source_x < 0 or source_x > nx - 1
    mask_x = (x_coords < dx) | (x_coords > nx - 1 + dx)
    
    mask = mask_y[:, np.newaxis] | mask_x[np.newaxis, :]
    
    if np.any(mask):
        shifted_img[mask] = fill_value
        
    return shifted_img


def _apply_shift(channel_img: np.ndarray, shift_vector: np.ndarray, shift_method: str = "spatial", order: int = 1) -> np.ndarray:
    """
    Apply translation shift using either spatial interpolation or Fourier domain shifting.
    """
    from scipy.ndimage import shift
    import numpy as np
    
    if np.all(shift_vector == 0.0):
        return channel_img
    if np.all(channel_img == 0):
        return channel_img
        
    has_nans = np.any(channel_img != channel_img)
    
    if shift_method == "fourier":
        if has_nans:
            # Interpolate NaNs
            channel_interp = interpolate_nan(channel_img)
            # Shift in Fourier domain (padding with NaN)
            correction = _shift_fourier(channel_interp, shift_vector, fill_value=np.nan)
            # Find shifted NaN positions using nearest spatial shift on mask
            nan_mask = np.isnan(channel_img)
            shifted_nan_mask = shift(nan_mask.astype(float), shift_vector, order=0, cval=1.0) > 0.5
            correction[shifted_nan_mask] = np.nan
            return correction
        else:
            return _shift_fourier(channel_img, shift_vector, fill_value=0.0)
    else:  # spatial
        if has_nans:
            channel_interp = interpolate_nan(channel_img)
            correction = shift(channel_interp, shift_vector, order=order)
            correction_nan = shift(channel_img, shift_vector, order=order, prefilter=False)
            nan_i, nan_j = np.where(correction_nan != correction_nan)
            correction[nan_i, nan_j] = np.nan
            return correction
        else:
            return shift(channel_img, shift_vector, order=order)


def register_stack_fourier_single_stack(
    stack_path: str,
    target_channel_index: Union[int, List[int]] = 0,
    nbr_channels: int = 1,
    stack_length: Optional[int] = 45,
    reference_frame_idx: int = 0,
    upsample_factor: int = 1,
    sliding: bool = False,
    order: int = 1,
    export: bool = False,
    prefix: str = "Aligned",
    return_stacks: bool = True,
    progress_callback: Optional[Callable] = None,
    sigma: float = 1.0,
    max_shift: float = 0.0,
    filter_outliers: bool = False,
    method: str = "fourier",
    shift_method: str = "spatial",
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """
    Register a single image stack using Fourier cross-correlation or SIFT.

    Parameters
    ----------
    stack_path : str
            The path to the image stack.
    target_channel_index : int or list of int, optional
            The index of the channel(s) to use for shift calculations.
    nbr_channels : int, optional
            The number of channels in the image stack.
    stack_length : int, optional
            The length of the image stack.
    reference_frame_idx : int, optional
            The index of the frame to use as reference template.
    upsample_factor : int, optional
            Upsampling factor for subpixel accuracy.
    sliding : bool, optional
            Whether to register each frame relative to the previous frame.
    order : int, optional
            Interpolation order for shifting.
    export : bool, optional
            Whether to export the aligned stack.
    prefix : str, optional
            The prefix for the exported file name.
    return_stacks : bool, optional
            Whether to return the registered stack.
    progress_callback : callable, optional
            A callback function to be called at each step of the process.

    Returns
    -------
    numpy.ndarray or None
            The registered stack if `return_stacks` is True, otherwise None.
    """

    if not os.path.exists(stack_path):
        raise FileNotFoundError(f"The stack {stack_path} does not exist... Abort.")

    from tqdm import tqdm
    import tifffile.tifffile as tiff
    from scipy.ndimage import shift, gaussian_filter
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        try:
            from skimage.feature import register_translation as phase_cross_correlation
        except ImportError:
            raise ImportError("scikit-image is required for Fourier registration.")

    stack_length_auto = auto_load_number_of_frames(stack_path)
    if stack_length_auto is not None:
        stack_length = stack_length_auto

    if isinstance(target_channel_index, (int, np.integer)):
        consensus_channels = [int(target_channel_index)]
    else:
        consensus_channels = [int(c) for c in target_channel_index]

    # Use tifffile.TiffFile to open the file once for fast reading
    with tiff.TiffFile(stack_path) as tif:
        total_pages = len(tif.pages)
        if stack_length is None:
            stack_length = total_pages // nbr_channels
        else:
            stack_length = min(int(stack_length), total_pages // nbr_channels)

        # Load frame indices for all consensus channels
        ref_indices_list = _get_img_num_per_channel(consensus_channels, stack_length, nbr_channels)
        ref_indices_by_chan = {c: ref_indices_list[idx] for idx, c in enumerate(consensus_channels)}
        total_frames = len(ref_indices_list[0])
        
        # We will keep track of fallbacks, sift inliers, and raw shifts
        fallbacks_list = []
        sift_inliers_list_all = []
        shifts = []

        logger.info(f"Computing translation shifts via {method} cross-correlation/SIFT for channels {consensus_channels}...")
        if sliding:
            step_shifts = {}
            prev_img = {}
            prev_img_reg = {}
            for k in tqdm(range(total_frames), desc="Computing shifts (sliding)"):
                if progress_callback:
                    progress_callback(level="frame", iter=k, total=total_frames, stage="Calculating shifts")
                
                curr_img = {}
                curr_img_reg = {}
                for c in consensus_channels:
                    curr_frame_num = ref_indices_by_chan[c][k]
                    img_c = tif.pages[int(curr_frame_num)].asarray().astype(float)
                    if np.any(img_c != img_c):
                        img_c = interpolate_nan(img_c)
                    curr_img[c] = img_c
                    if sigma > 0:
                        curr_img_reg[c] = gaussian_filter(img_c, sigma=sigma)
                    else:
                        curr_img_reg[c] = img_c

                if k > 0:
                    shift_vector = None
                    is_fallback = False
                    avg_inliers = 0
                    
                    sift_shifts = []
                    sift_weights = []
                    sift_inliers_frame = []
                    
                    if method in ["sift", "hybrid"]:
                        for c in consensus_channels:
                            try:
                                sift_shift, matches, inliers = _estimate_sift_translation(prev_img[c], curr_img[c])
                                sift_inliers_frame.append(inliers)
                                if sift_shift is not None and inliers >= 3:
                                    if max_shift == 0 or np.linalg.norm(sift_shift) <= max_shift:
                                        sift_shifts.append(sift_shift)
                                        sift_weights.append(float(inliers))
                                    else:
                                        logger.warning(f"Frame {k}, Chan {c}: SIFT shift {sift_shift} exceeds max_shift {max_shift}.")
                            except Exception as e:
                                logger.error(f"Error in SIFT at frame {k}, channel {c}: {e}")
                        
                        if len(sift_shifts) > 0:
                            sift_shifts = np.array(sift_shifts)
                            sift_weights = np.array(sift_weights)
                            shift_vector = np.sum(sift_shifts * sift_weights[:, np.newaxis], axis=0) / np.sum(sift_weights)
                            avg_inliers = int(np.mean(sift_inliers_frame))
                            logger.info(f"Frame {k}: Joint SIFT succeeded. Shift: {shift_vector}")
                    
                    if shift_vector is None:
                        if method == "sift":
                            logger.warning(f"Frame {k}: Joint SIFT failed. Falling back to [0.0, 0.0].")
                            shift_vector = np.array([0.0, 0.0])
                            is_fallback = True
                            avg_inliers = 0
                        else:  # method == "fourier" or "hybrid" (fallback)
                            if method == "hybrid":
                                logger.info(f"Frame {k}: Joint SIFT failed. Hybrid falling back to Fourier.")
                                is_fallback = True
                            
                            fourier_shifts = []
                            fourier_weights = []
                            for c in consensus_channels:
                                try:
                                    fourier_shift, error, phasediff = phase_cross_correlation(
                                        prev_img_reg[c], curr_img_reg[c], upsample_factor=upsample_factor
                                    )
                                    fourier_shift = np.array(fourier_shift)
                                    if max_shift > 0 and np.linalg.norm(fourier_shift) > max_shift:
                                        logger.warning(f"Frame {k}, Chan {c}: Stepwise Fourier shift {fourier_shift} exceeds max_shift ({max_shift}).")
                                    else:
                                        fourier_shifts.append(fourier_shift)
                                        fourier_weights.append(max(0.001, 1.0 - error))
                                except Exception as e:
                                    logger.error(f"Error in Fourier at frame {k}, channel {c}: {e}")
                            
                            if len(fourier_shifts) > 0:
                                fourier_shifts = np.array(fourier_shifts)
                                fourier_weights = np.array(fourier_weights)
                                shift_vector = np.sum(fourier_shifts * fourier_weights[:, np.newaxis], axis=0) / np.sum(fourier_weights)
                            else:
                                logger.warning(f"Frame {k}: All Fourier channels failed. Falling back to [0.0, 0.0].")
                                shift_vector = np.array([0.0, 0.0])
                                is_fallback = True
                            avg_inliers = 0
                    
                    step_shifts[k] = shift_vector
                    if is_fallback:
                        fallbacks_list.append(k)
                    sift_inliers_list_all.append(avg_inliers)
                else:
                    sift_inliers_list_all.append(0)
                
                prev_img = curr_img.copy()
                prev_img_reg = curr_img_reg.copy()

            # Accumulate shifts relative to reference_frame_idx
            shifts = [np.array([0.0, 0.0]) for _ in range(total_frames)]
            for k in range(reference_frame_idx + 1, total_frames):
                shifts[k] = shifts[k-1] + step_shifts[k]
            for k in range(reference_frame_idx - 1, -1, -1):
                shifts[k] = shifts[k+1] - step_shifts[k+1]
        else:
            ref_img = {}
            ref_img_reg = {}
            for c in consensus_channels:
                ref_frame_num = ref_indices_by_chan[c][reference_frame_idx]
                img_c = tif.pages[int(ref_frame_num)].asarray().astype(float)
                if np.any(img_c != img_c):
                    img_c = interpolate_nan(img_c)
                ref_img[c] = img_c
                if sigma > 0:
                    ref_img_reg[c] = gaussian_filter(img_c, sigma=sigma)
                else:
                    ref_img_reg[c] = img_c

            last_valid_shift = np.array([0.0, 0.0])

            for k in tqdm(range(total_frames), desc="Computing shifts"):
                if progress_callback:
                    progress_callback(level="frame", iter=k, total=total_frames, stage="Calculating shifts")
                if k == reference_frame_idx:
                    shifts.append(np.array([0.0, 0.0]))
                    sift_inliers_list_all.append(0)
                else:
                    curr_img = {}
                    curr_img_reg = {}
                    for c in consensus_channels:
                        curr_frame_num = ref_indices_by_chan[c][k]
                        img_c = tif.pages[int(curr_frame_num)].asarray().astype(float)
                        if np.any(img_c != img_c):
                            img_c = interpolate_nan(img_c)
                        curr_img[c] = img_c
                        if sigma > 0:
                            curr_img_reg[c] = gaussian_filter(img_c, sigma=sigma)
                        else:
                            curr_img_reg[c] = img_c

                    shift_vector = None
                    is_fallback = False
                    avg_inliers = 0
                    
                    sift_shifts = []
                    sift_weights = []
                    sift_inliers_frame = []
                    
                    if method in ["sift", "hybrid"]:
                        for c in consensus_channels:
                            try:
                                sift_shift, matches, inliers = _estimate_sift_translation(ref_img[c], curr_img[c])
                                sift_inliers_frame.append(inliers)
                                if sift_shift is not None and inliers >= 3:
                                    if max_shift == 0 or np.linalg.norm(sift_shift) <= max_shift:
                                        sift_shifts.append(sift_shift)
                                        sift_weights.append(float(inliers))
                                    else:
                                        logger.warning(f"Frame {k}, Chan {c}: SIFT shift {sift_shift} exceeds max_shift {max_shift}.")
                            except Exception as e:
                                logger.error(f"Error in SIFT at frame {k}, channel {c}: {e}")
                        
                        if len(sift_shifts) > 0:
                            sift_shifts = np.array(sift_shifts)
                            sift_weights = np.array(sift_weights)
                            shift_vector = np.sum(sift_shifts * sift_weights[:, np.newaxis], axis=0) / np.sum(sift_weights)
                            avg_inliers = int(np.mean(sift_inliers_frame))
                            logger.info(f"Frame {k}: Joint SIFT succeeded. Shift: {shift_vector}")
                    
                    if shift_vector is None:
                        if method == "sift":
                            logger.warning(f"Frame {k}: Joint SIFT failed. Falling back to last_valid_shift {last_valid_shift}.")
                            shift_vector = last_valid_shift
                            is_fallback = True
                            avg_inliers = 0
                        else:  # method == "fourier" or "hybrid" (fallback)
                            if method == "hybrid":
                                logger.info(f"Frame {k}: Joint SIFT failed. Hybrid falling back to Fourier.")
                                is_fallback = True
                            
                            fourier_shifts = []
                            fourier_weights = []
                            for c in consensus_channels:
                                try:
                                    fourier_shift, error, phasediff = phase_cross_correlation(
                                        ref_img_reg[c], curr_img_reg[c], upsample_factor=upsample_factor
                                    )
                                    fourier_shift = np.array(fourier_shift)
                                    if max_shift > 0 and np.linalg.norm(fourier_shift) > max_shift:
                                        logger.warning(f"Frame {k}, Chan {c}: Fourier shift {fourier_shift} exceeds max_shift ({max_shift}).")
                                    else:
                                        fourier_shifts.append(fourier_shift)
                                        fourier_weights.append(max(0.001, 1.0 - error))
                                except Exception as e:
                                    logger.error(f"Error in Fourier at frame {k}, channel {c}: {e}")
                            
                            if len(fourier_shifts) > 0:
                                fourier_shifts = np.array(fourier_shifts)
                                fourier_weights = np.array(fourier_weights)
                                shift_vector = np.sum(fourier_shifts * fourier_weights[:, np.newaxis], axis=0) / np.sum(fourier_weights)
                                last_valid_shift = shift_vector
                            else:
                                logger.warning(f"Frame {k}: All Fourier channels failed. Falling back to last_valid_shift {last_valid_shift}.")
                                shift_vector = last_valid_shift
                                is_fallback = True
                            avg_inliers = 0
                    else:
                        last_valid_shift = shift_vector
                    
                    shifts.append(shift_vector)
                    if is_fallback:
                        fallbacks_list.append(k)
                    sift_inliers_list_all.append(avg_inliers)

        # Record raw shifts before median outlier filtering
        raw_shifts = [s.copy() for s in shifts]

        # Apply median filtering on the accumulated shift trajectories if requested
        if filter_outliers and len(shifts) > 2:
            from scipy.signal import medfilt
            shifts_x = np.array([s[0] for s in shifts])
            shifts_y = np.array([s[1] for s in shifts])
            filtered_x = medfilt(shifts_x, kernel_size=3)
            filtered_y = medfilt(shifts_y, kernel_size=3)
            shifts = [np.array([x, y]) for x, y in zip(filtered_x, filtered_y)]

        # Emit drift metadata via progress callback
        if progress_callback:
            progress_callback(
                level="plot_data",
                plot_data={
                    "stack_path": stack_path,
                    "shifts": [list(s) for s in shifts],
                    "raw_shifts": [list(s) for s in raw_shifts],
                    "fallbacks": list(fallbacks_list),
                    "sift_inliers": list(sift_inliers_list_all),
                    "max_shift_limit": max_shift,
                }
            )

        # 3. Shift all channels of all frames and save/return
        corrected_stack = []

        if export:
            path, file = os.path.split(stack_path)
            if prefix is None:
                newfile = "temp_" + file
            else:
                newfile = "_".join([prefix, file])

            with tiff.TiffWriter(
                os.sep.join([path, newfile]), bigtiff=True, imagej=True
            ) as tif_writer:
                frames_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
                for k, i in enumerate(tqdm(frames_indices, desc="Applying registration")):
                    if progress_callback:
                        progress_callback(level="frame", iter=k, total=total_frames, stage="Applying shifts")

                    # Load all channels for timepoint k
                    frames = []
                    for c in range(nbr_channels):
                        page_idx = i + c
                        frame_c = tif.pages[int(page_idx)].asarray().astype(float)
                        frames.append(frame_c)
                    frames = np.stack(frames, axis=-1)

                    shift_vector = shifts[k]

                    for c in range(nbr_channels):
                        channel_img = frames[:, :, c].copy()
                        correction = _apply_shift(channel_img, shift_vector, shift_method=shift_method, order=order)
                        frames[:, :, c] = correction.copy()

                    if return_stacks:
                        corrected_stack.append(frames)

                    tif_writer.write(
                        np.moveaxis(frames, -1, 0).astype(np.dtype("f")),
                        contiguous=True,
                    )
                    del frames
                    collect()

            if prefix is None:
                os.replace(os.sep.join([path, newfile]), os.sep.join([path, file]))
        else:
            frames_indices = range(0, int(stack_length * nbr_channels), nbr_channels)
            for k, i in enumerate(tqdm(frames_indices)):
                if progress_callback:
                    progress_callback(level="frame", iter=k, total=total_frames)

                # Load all channels for timepoint k
                frames = []
                for c in range(nbr_channels):
                    page_idx = i + c
                    frame_c = tif.pages[int(page_idx)].asarray().astype(float)
                    frames.append(frame_c)
                frames = np.stack(frames, axis=-1)

                shift_vector = shifts[k]

                for c in range(nbr_channels):
                    channel_img = frames[:, :, c].copy()
                    correction = _apply_shift(channel_img, shift_vector, shift_method=shift_method, order=order)
                    frames[:, :, c] = correction.copy()

                corrected_stack.append(frames)
                del frames
                collect()

    if return_stacks:
        return np.array(corrected_stack)
    else:
        return None
