import os
import logging
from glob import glob
from typing import Optional, Tuple

from celldetective.utils.downloaders import get_zenodo_files, download_zenodo_file

logger = logging.getLogger("celldetective")


def trained_cell_size_um(input_config: Optional[dict]) -> Optional[float]:
    """
    The physical object size a segmentation model was trained on, in microns.

    This is the size the rescaling is measured against: a frame is resized until
    its objects reach the size the network was trained to see, so both readings
    have to come from the same place wherever that is worked out -- the library,
    the main window's channel dialog, or the napari single-frame panel.

    A model built through celldetective records the size directly as
    ``cell_size_um``. A generic Cellpose model does not, but it states the same
    thing in pixels: it was trained to see objects ``diameter`` px across, at its
    own ``spatial_calibration``, so the product is that size in microns. Without
    the second reading a generic model cannot be rescaled at all -- the
    correction needs the trained size and the target, and one of them is missing.

    Parameters
    ----------
    input_config : dict or None
            A model's ``config_input.json``, already parsed.

    Returns
    -------
    float or None
            The trained object size in microns, or None when the configuration
            gives no way to work it out -- in which case there is nothing to
            rescale against.

    Examples
    --------
    >>> trained_cell_size_um({'cell_size_um': 9.211})
    9.211
    >>> trained_cell_size_um(
    ...     {'model_type': 'cellpose', 'diameter': 30.0, 'spatial_calibration': 0.5}
    ... )
    15.0
    """

    if not input_config:
        return None

    cell_size = input_config.get("cell_size_um")
    if cell_size is not None:
        # Comes from the model's own configuration, so a nonsensical value is
        # worth saying out loud rather than quietly scaling everything to nothing.
        if cell_size > 0:
            return cell_size
        logger.warning(
            f"Ignoring cell_size_um={cell_size}: it must be strictly positive."
        )
        return None

    if input_config.get("model_type") != "cellpose":
        return None

    diameter = input_config.get("diameter")
    calibration = input_config.get("spatial_calibration")
    if (
        diameter is not None
        and calibration is not None
        and diameter > 0
        and calibration > 0
    ):
        return diameter * calibration
    return None


def locate_signal_model(
    name: str, path: Optional[str] = None, pairs: bool = False
) -> Optional[str]:
    """
    Locate a signal detection model by name, either locally or from Zenodo.

    This function searches for a signal detection model with the specified name in the local
    `celldetective` directory. If the model is not found locally, it attempts to download
    the model from Zenodo.

    Parameters
    ----------
    name : str
            The name of the signal detection model to locate.
    path : str, optional
            An additional directory path to search for the model. If provided, this directory
            is also scanned for matching models. Default is `None`.
    pairs : bool, optional
            If `True`, searches for paired signal detection models in the `pair_signal_detection`
            subdirectory. If `False`, searches in the `signal_detection` subdirectory. Default is `False`.

    Returns
    -------
    str or None
            The full path to the located model directory if found, or `None` if the model is not available
            locally or on Zenodo.

    Notes
    -----
    - The function first searches in the `celldetective/models/signal_detection` or
      `celldetective/models/pair_signal_detection` directory based on the `pairs` argument.
    - If a `path` is specified, it is searched in addition to the default directories.
    - If the model is not found locally, the function queries Zenodo for the model. If available,
      the model is downloaded to the appropriate `celldetective` subdirectory.

    Examples
    --------
    Search for a signal detection model locally:

    >>> locate_signal_model("example_model")
    'path/to/celldetective/models/signal_detection/example_model/'

    Search for a paired signal detection model:

    >>> locate_signal_model("paired_model", pairs=True)
    'path/to/celldetective/models/pair_signal_detection/paired_model/'

    Include an additional search path:

    >>> locate_signal_model("custom_model", path="/additional/models/")
    '/additional/models/custom_model/'

    Handle a model available only on Zenodo:

    >>> locate_signal_model("remote_model")
    'path/to/celldetective/models/signal_detection/remote_model/'

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]]
    )
    modelpath = os.sep.join([main_dir, "models", "signal_detection", os.sep])
    if pairs:
        modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
    logger.debug(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")
    if path is not None:
        if not path.endswith(os.sep):
            path += os.sep
        models += glob(path + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def locate_pair_signal_model(name: str, path: Optional[str] = None) -> Optional[str]:
    """
    Locate a pair signal detection model by name.

    This function searches for a pair signal detection model in the default
    `celldetective` directory and optionally in an additional user-specified path.

    Parameters
    ----------
    name : str
            The name of the pair signal detection model to locate.
    path : str, optional
            An additional directory path to search for the model. If provided, this directory
            is also scanned for matching models. Default is `None`.

    Returns
    -------
    str or None
            The full path to the located model directory if found, or `None` if no matching
            model is located.

    Notes
    -----
    - The function first searches in the default `celldetective/models/pair_signal_detection`
      directory.
    - If a `path` is specified, it is searched in addition to the default directory.
    - The function prints the search path and model name during execution.

    Examples
    --------
    Locate a model in the default directory:

    >>> locate_pair_signal_model("example_model")
    'path/to/celldetective/models/pair_signal_detection/example_model/'

    Include an additional search directory:

    >>> locate_pair_signal_model("custom_model", path="/additional/models/")
    '/additional/models/custom_model/'

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]]
    )
    modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
    logger.debug(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")
    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def locate_segmentation_model(name: str, download: bool = True) -> Optional[str]:
    """
    Locates a specified segmentation model within the local 'celldetective' directory or
    downloads it from Zenodo if not found locally.

    This function attempts to find a segmentation model by name within a predefined directory
    structure starting from the 'celldetective/models/segmentation*' path. If the model is not
    found locally, it then tries to locate and download the model from Zenodo, placing it into
    the appropriate category directory within 'celldetective'. The function prints the search
    directory path and returns the path to the found or downloaded model.

    Parameters
    ----------
    name : str
            The name of the segmentation model to locate.
    download : bool, optional
            Whether to download the model from Zenodo if not found locally. Default is True.

    Returns
    -------
    str or None
            The full path to the located or downloaded segmentation model directory, or None if the
            model could not be found or downloaded.

    Raises
    ------
    FileNotFoundError
            If the model cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]]
    )
    modelpath = os.sep.join([main_dir, "models", "segmentation*"]) + os.sep
    # print(f'Looking for {name} in {modelpath}')
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            if not os.path.exists(os.sep.join([m.rstrip(os.sep), "config_input.json"])):
                # An interrupted download leaves the directory behind without its
                # input configuration. Matching on the name alone would hand back
                # a model that can never be loaded, and would shadow the copy on
                # Zenodo forever; treat it as absent so the download below can
                # overwrite it.
                logger.warning(
                    f"Ignoring incomplete local model {name} in {m}: "
                    "no 'config_input.json'."
                )
                continue
            match = m
            return match
    if download:
        # else no match, try zenodo
        files, categories = get_zenodo_files()
        if name in files:
            index = files.index(name)
            cat = categories[index]
            download_zenodo_file(name, os.sep.join([main_dir, cat]))
            match = os.sep.join([main_dir, cat, name]) + os.sep

    return match


def locate_segmentation_dataset(name: str) -> Optional[str]:
    """
    Locates a specified segmentation dataset within the local 'celldetective/datasets/segmentation_annotations' directory
    or downloads it from Zenodo if not found locally.

    This function attempts to find a segmentation dataset by name within a predefined directory structure. If the dataset
    is not found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate
    category directory within 'celldetective'. The function prints the search directory path and returns the path to the
    found or downloaded dataset.

    Parameters
    ----------
    name : str
            The name of the segmentation dataset to locate.

    Returns
    -------
    str or None
            The full path to the located or downloaded segmentation dataset directory, or None if the dataset could not be
            found or downloaded.

    Raises
    ------
    FileNotFoundError
            If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]]
    )
    modelpath = os.sep.join([main_dir, "datasets", "segmentation_annotations", os.sep])
    logger.debug(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def locate_signal_dataset(name: str) -> Optional[str]:
    """
    Locates a specified signal dataset within the local 'celldetective/datasets/signal_annotations' directory or downloads
    it from Zenodo if not found locally.

    This function attempts to find a signal dataset by name within a predefined directory structure. If the dataset is not
    found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate category
    directory within 'celldetective'. The function prints the search directory path and returns the path to the found or
    downloaded dataset.

    Parameters
    ----------
    name : str
            The name of the signal dataset to locate.

    Returns
    -------
    str or None
            The full path to the located or downloaded signal dataset directory, or None if the dataset could not be found or
            downloaded.

    Raises
    ------
    FileNotFoundError
            If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

    """

    main_dir = os.sep.join(
        [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]]
    )
    modelpath = os.sep.join([main_dir, "datasets", "signal_annotations", os.sep])
    logger.debug(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")

    match = None
    for m in models:
        if name == m.replace("\\", os.sep).split(os.sep)[-2]:
            match = m
            return match
    # else no match, try zenodo
    files, categories = get_zenodo_files()
    if name in files:
        index = files.index(name)
        cat = categories[index]
        download_zenodo_file(name, os.sep.join([main_dir, cat]))
        match = os.sep.join([main_dir, cat, name]) + os.sep
    return match


def _resolve_signal_model_paths(
    model: str,
    path: Optional[str] = None,
    pairs: bool = False,
) -> Tuple[str, str]:
    """Locate a signal model directory and config file, raising FileNotFoundError if either is missing.

    Parameters
    ----------
    model : str
        Model name to locate.
    path : str or None, optional
        Override path for model search.
    pairs : bool, optional
        Whether to search for a pair-interaction model. Default False.

    Returns
    -------
    complete_path : str
        Absolute path to the model directory.
    model_config_path : str
        Absolute path to ``config_input.json`` inside the model directory.
    """
    model_dir = locate_signal_model(model, path=path, pairs=pairs)
    logger.info(f"Looking for model in {model_dir}...")
    complete_path = rf"{model_dir}"
    model_config_path = rf"{os.sep.join([complete_path, 'config_input.json'])}"
    if not os.path.exists(complete_path):
        raise FileNotFoundError(
            f"Model {model} could not be located in folder {model_dir}... Abort."
        )
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(
            f"Model configuration could not be located in folder {model_dir}... Abort."
        )
    return complete_path, model_config_path


def freeze_model_encoder(model, model_type: str) -> None:
    """
    Freeze the encoder layers of a model for transfer learning.
    Supports 'stardist' (TensorFlow/Keras) and 'cellpose' (PyTorch) models.

    Parameters
    ----------
    model : Any
        The model object to freeze layers for.
    model_type : str
        The type of model ('stardist' or 'cellpose').
    """
    if model_type == "stardist":
        logger.info("Freezing encoder layers for StarDist model...")
        mod = model.keras_model
        encoder_depth = len(mod.layers) // 2

        for layer in mod.layers[:encoder_depth]:
            layer.trainable = False

        # Keep decoder trainable
        for layer in mod.layers[encoder_depth:]:
            layer.trainable = True

    elif model_type == "cellpose":
        logger.info("Freezing encoder layers for Cellpose model...")
        for param in model.net.downsample.parameters():
            param.requires_grad = False

        # Optional: freeze style branch
        for param in model.net.make_style.parameters():
            param.requires_grad = False

        # Keep decoder trainable
        for param in model.net.upsample.parameters():
            param.requires_grad = True

        # Keep output head trainable
        for param in model.net.output.parameters():
            param.requires_grad = True

        # Unfreeze all output heads (version-safe)
        output_heads = ["output", "output_conv", "flow", "prob"]
        for head_name in output_heads:
            if hasattr(model.net, head_name):
                for param in getattr(model.net, head_name).parameters():
                    param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model_type for encoder freezing: {model_type}")
