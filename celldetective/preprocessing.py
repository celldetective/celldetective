"""
Copright © 2024 Laboratoire Adhesion et Inflammation, Authored by Remy Torro & Ksenija Dervanova.
"""

from typing import List
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
    experiment,
    threshold_on_std=1,
    well_option="*",
    target_channel="channel_name",
    frame_range=[0, 5],
    mode="timeseries",
    activation_protocol=[["gauss", 2], ["std", 4]],
    show_progress_per_pos=False,
    show_progress_per_well=True,
    offset=None,
    fix_nan: bool = False,
    progress_callback=None,
):
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
                print(f"Stack not found for position {pos_path}...")
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
                # print("The offset is applied to background...")
                background -= offset
            if fix_nan:
                background = interpolate_nan(background.copy().astype(float))
            backgrounds.append({"bg": background, "well": well_path})
            logger.info(f"Background successfully computed for well {well_name}...")
        except Exception as e:
            logger.error(e)
            backgrounds.append(None)

    return backgrounds


def correct_background_model_free(
    experiment,
    well_option="*",
    position_option="*",
    target_channel="channel_name",
    mode="timeseries",
    threshold_on_std=1,
    frame_range=[0, 5],
    optimize_option=False,
    opt_coef_range=[0.95, 1.05],
    opt_coef_nbr=100,
    operation="divide",
    clip=False,
    offset=None,
    show_progress_per_well=True,
    show_progress_per_pos=False,
    export=False,
    return_stacks=False,
    movie_prefix=None,
    fix_nan=False,
    activation_protocol=[["gauss", 2], ["std", 4]],
    export_prefix="Corrected",
    progress_callback=None,
    **kwargs,
):
    """
    Correct the background of image stacks for a given experiment.

    This function processes image stacks by estimating and correcting the background
    for each well and position in the experiment. It supports different modes, such
    as timeseries or tiles, and offers options for optimization and exporting the results.

    Parameters
    ----------
    experiment : str
            Path to the experiment configuration.
    well_option : str, int, or list of int, optional
            Selection of wells to process. '*' indicates all wells. Defaults to '*'.
    position_option : str, int, or list of int, optional
            Selection of positions to process within each well. '*' indicates all positions. Defaults to '*'.
    target_channel : str, optional
            The name of the target channel to be corrected. Defaults to "channel_name".
    mode : {'timeseries', 'tiles'}, optional
            The mode of processing. Defaults to "timeseries".
    threshold_on_std : float, optional
            The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
    frame_range : list of int, optional
            The range of frames to consider for background estimation. Defaults to [0, 5].
    optimize_option : bool, optional
            If True, optimize the correction coefficient. Defaults to False.
    opt_coef_range : list of float, optional
            The range of coefficients to try for optimization. Defaults to [0.95, 1.05].
    opt_coef_nbr : int, optional
            The number of coefficients to test within the optimization range. Defaults to 100.
    operation : {'divide', 'subtract'}, optional
            The operation to apply for background correction. Defaults to 'divide'.
    clip : bool, optional
            If True, clip the corrected values to be non-negative when using subtraction. Defaults to False.
    show_progress_per_well : bool, optional
            If True, show progress bar for each well. Defaults to True.
    show_progress_per_pos : bool, optional
            If True, show progress bar for each position. Defaults to False.
    export : bool, optional
            If True, export the corrected stacks to files. Defaults to False.
    return_stacks : bool, optional
            If True, return the corrected stacks as a list of numpy arrays. Defaults to False.
    progress_callback : callable, optional
            A callback function to be called at each step of the process (default is None).

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
    >>> corrected_stacks = correct_background(experiment, well_option=[0, 1], position_option='*', target_channel="DAPI", mode="timeseries", threshold_on_std=2, frame_range=[0, 10], optimize_option=True, operation='subtract', clip=True, return_stacks=True)
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
    stack_path,
    background,
    target_channel_index=0,
    nbr_channels=1,
    stack_length=45,
    offset=None,
    activation_protocol=[["gauss", 2], ["std", 4]],
    threshold_on_std=1,
    optimize_option=True,
    opt_coef_range=(0.95, 1.05),
    opt_coef_nbr=100,
    operation="divide",
    clip=False,
    export=False,
    prefix="Corrected",
    fix_nan=False,
    progress_callback=None,
):
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
    threshold_on_std : float, optional
            The threshold for the standard deviation filter to identify high-variance areas. Defaults to 1.
    optimize_option : bool, optional
            If True, optimize the correction coefficient using a range of values. Defaults to True.
    opt_coef_range : tuple of float, optional
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
    corrected_stack : numpy.ndarray
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
            # print(f"The offset is applied to image...")
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


def paraboloid(x, y, a, b, c, d, e, g):
    """
    Compute the value of a 2D paraboloid function.

    This function evaluates a paraboloid defined by the equation:
    `a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y + g`.

    Parameters
    ----------
    x : float or ndarray
            The x-coordinate(s) at which to evaluate the paraboloid.
    y : float or ndarray
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
    float or ndarray
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


def plane(x, y, a, b, c):
    """
    Compute the value of a plane function.

    This function evaluates a plane defined by the equation:
    `a * x + b * y + c`.

    Parameters
    ----------
    x : float or ndarray
            The x-coordinate(s) at which to evaluate the plane.
    y : float or ndarray
            The y-coordinate(s) at which to evaluate the plane.
    a : float
            The coefficient of the x term.
    b : float
            The coefficient of the y term.
    c : float
            The constant term.

    Returns
    -------
    float or ndarray
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


def fit_plane(image, cell_masks=None, edge_exclusion=None):
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


def fit_paraboloid(image, cell_masks=None, edge_exclusion=None, downsample=10):
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
    experiment,
    well_option="*",
    position_option="*",
    target_channel="channel_name",
    threshold_on_std=1,
    model="paraboloid",
    operation="divide",
    clip=False,
    show_progress_per_well=True,
    show_progress_per_pos=False,
    export=False,
    return_stacks=False,
    movie_prefix=None,
    activation_protocol=[["gauss", 2], ["std", 4]],
    export_prefix="Corrected",
    return_stack=True,
    progress_callback=None,
    downsample=10,
    **kwargs,
):
    """
    Correct background in image stacks using a specified model.

    This function corrects the background in image stacks obtained from an experiment
    using a specified background correction model. It supports various options for
    specifying wells, positions, target channel, and background correction parameters.

    Parameters
    ----------
    experiment : str
            The path to the experiment directory.
    well_option : str, optional
            The option to select specific wells (default is '*').
    position_option : str, optional
            The option to select specific positions (default is '*').
    target_channel : str, optional
            The name of the target channel for background correction (default is "channel_name").
    threshold_on_std : float, optional
            The threshold value on the standard deviation for masking (default is 1).
    model : str, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    operation : str, optional
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
    **kwargs : dict
            Additional keyword arguments to be passed to the underlying correction function.

    Returns
    -------
    list of numpy.ndarray
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
                activation_protocol=activation_protocol,
                return_stacks=return_stacks,
                progress_callback=progress_callback,
                downsample=downsample,
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
    stack_path,
    target_channel_index=0,
    nbr_channels=1,
    stack_length=45,
    threshold_on_std=1,
    operation="divide",
    model="paraboloid",
    clip=False,
    export=False,
    activation_protocol=[["gauss", 2], ["std", 4]],
    prefix="Corrected",
    return_stacks=True,
    progress_callback=None,
    downsample=10,
    subset_indices=None,
):
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
    subset_indices : list of int, optional
            List of absolute frame indices to process (default is None).
    stack_length : int, optional
            The length of the stack (default is 45).
    threshold_on_std : float, optional
            The threshold value on the standard deviation for masking (default is 1).
    operation : str, optional
            The operation to apply for background correction, either 'divide' or 'subtract' (default is 'divide').
    model : str, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    clip : bool, optional
            Whether to clip the corrected image to ensure non-negative values (default is False).
    export : bool, optional
            Whether to export the corrected image stack (default is False).
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss',2],['std',4]]).
    prefix : str, optional
            The prefix for exported corrected stacks (default is 'Corrected').
    subset_indices : list of int, optional
            List of absolute frame indices to process (default is None).

    Returns
    -------
    numpy.ndarray
            The corrected image stack.

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
    operation: str = "divide",
    model: str = "paraboloid",
    clip: bool = False,
    return_bg: bool = False,
    activation_protocol: List[List] = [["gauss", 2], ["std", 4]],
    downsample: int = 10,
):
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
    operation : str, optional
            The operation to apply for background correction, either 'divide' or 'subtract' (default is 'divide').
    model : str, optional
            The background correction model to use, either 'paraboloid' or 'plane' (default is 'paraboloid').
    clip : bool, optional
            Whether to clip the corrected image to ensure non-negative values (default is False).
    return_bg : bool, optional
            Whether to return the background along with the corrected image (default is False).
    activation_protocol : list of list, optional
            The activation protocol consisting of filters and their respective parameters (default is [['gauss',2],['std',4]]).

    Returns
    -------
    numpy.ndarray or tuple
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

    return correction.copy()


def fit_background_model(
    img, cell_masks=None, model="paraboloid", edge_exclusion=None, downsample=10
):
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
    model : str, optional
            The background model to fit, either 'paraboloid' or 'plane' (default is 'paraboloid').
    edge_exclusion : int or None, optional
            The size of the border to exclude from fitting (default is None).

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
    experiment,
    well_option="*",
    position_option="*",
    target_channel="channel_name",
    correction_horizontal=0,
    correction_vertical=0,
    show_progress_per_well=True,
    show_progress_per_pos=True,
    export=False,
    return_stacks=False,
    movie_prefix=None,
    export_prefix="Corrected",
    progress_callback=None,
    **kwargs,
):

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
            print(f"Processing well {k+1}/{total_wells}...")

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
                print(f"  Processing position {pidx+1}/{total_pos}...")

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
    stack_path,
    target_channel_index=0,
    nbr_channels=1,
    stack_length=45,
    correction_vertical=0,
    correction_horizontal=0,
    export=False,
    prefix="Corrected",
    return_stacks=True,
    progress_callback=None,
):

    assert os.path.exists(
        stack_path
    ), f"The stack {stack_path} does not exist... Abort."

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
