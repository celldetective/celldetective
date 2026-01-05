import os
from glob import glob

from celldetective.utils.downloaders import get_zenodo_files, download_zenodo_file


def locate_signal_model(name, path=None, pairs=False):
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
    print(f"Looking for {name} in {modelpath}")
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


def locate_pair_signal_model(name, path=None):
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
    print(f"Looking for {name} in {modelpath}")
    models = glob(modelpath + f"*{os.sep}")
    if path is not None:
        if not path.endswith(os.sep):
            path += os.sep
        models += glob(path + f"*{os.sep}")


def locate_segmentation_model(name, download=True):
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


def locate_segmentation_dataset(name):
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
    print(f"Looking for {name} in {modelpath}")
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


def locate_signal_dataset(name):
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
    print(f"Looking for {name} in {modelpath}")
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
