import os
from glob import glob
from shutil import rmtree

from natsort import natsorted

from celldetective.utils.downloaders import get_zenodo_files


def get_tracking_configs_list(return_path=False):
    """

    Retrieve a list of available tracking configurations.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available tracking configurations.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available tracking configurations by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_tracking_configs_list()
    # Retrieve a list of available tracking configurations.

    >>> models, path = get_tracking_configs_list(return_path=True)
    # Retrieve a list of available tracking configurations.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "models",
            "tracking_configs",
            os.sep,
        ]
    )
    available_models = glob(modelpath + "*.json")
    available_models = [m.replace("\\", "/").split("/")[-1] for m in available_models]
    available_models = [m.replace("\\", "/").split(".")[0] for m in available_models]

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def get_signal_models_list(return_path=False):
    """

    Retrieve a list of available signal detection models.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available signal detection models.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available signal detection models by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_signal_models_list()
    # Retrieve a list of available signal detection models.

    >>> models, path = get_signal_models_list(return_path=True)
    # Retrieve a list of available signal detection models and the path to the models.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "models",
            "signal_detection",
            os.sep,
        ]
    )
    repository_models = get_zenodo_files(
        cat=os.sep.join(["models", "signal_detection"])
    )

    available_models = glob(modelpath + f"*{os.sep}")
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]
    available_models = [
        m
        for m in available_models
        if os.path.exists(os.path.join(modelpath, m, "config_input.json"))
    ]
    for rm in repository_models:
        if rm not in available_models:
            available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def get_pair_signal_models_list(return_path=False):
    """

    Retrieve a list of available signal detection models.

    Parameters
    ----------
    return_path : bool, optional
            If True, also returns the path to the models. Default is False.

    Returns
    -------
    list or tuple
            If return_path is False, returns a list of available signal detection models.
            If return_path is True, returns a tuple containing the list of models and the path to the models.

    Notes
    -----
    This function retrieves the list of available signal detection models by searching for model directories
    in the predefined model path. The model path is derived from the parent directory of the current script
    location and the path to the model directory. By default, it returns only the names of the models.
    If return_path is set to True, it also returns the path to the models.

    Examples
    --------
    >>> models = get_signal_models_list()
    # Retrieve a list of available signal detection models.

    >>> models, path = get_signal_models_list(return_path=True)
    # Retrieve a list of available signal detection models and the path to the models.

    """

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "models",
            "pair_signal_detection",
            os.sep,
        ]
    )
    # repository_models = get_zenodo_files(cat=os.sep.join(["models", "pair_signal_detection"]))

    available_models = glob(modelpath + f"*{os.sep}")
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]
    # for rm in repository_models:
    #   if rm not in available_models:
    #       available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def get_segmentation_models_list(mode="targets", return_path=False):

    modelpath = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "models",
            f"segmentation_{mode}",
            os.sep,
        ]
    )
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
        repository_models = []
    else:
        repository_models = get_zenodo_files(
            cat=os.sep.join(["models", f"segmentation_{mode}"])
        )

    available_models = natsorted(glob(modelpath + "*/"))
    available_models = [m.replace("\\", "/").split("/")[-2] for m in available_models]

    # Auto model cleanup
    to_remove = []
    for model in available_models:
        path = modelpath + model
        files = glob(path + os.sep + "*")
        if path + os.sep + "config_input.json" not in files:
            rmtree(path)
            to_remove.append(model)
    for m in to_remove:
        available_models.remove(m)

    for rm in repository_models:
        if rm not in available_models:
            available_models.append(rm)

    if not return_path:
        return available_models
    else:
        return available_models, modelpath


def get_segmentation_datasets_list(return_path=False):
    """
    Retrieves a list of available segmentation datasets from both the local 'celldetective/datasets/segmentation_annotations'
    directory and a Zenodo repository, optionally returning the path to the local datasets directory.

    This function compiles a list of available segmentation datasets by first identifying datasets stored locally
    within a specified path related to the script's directory. It then extends this list with datasets available
    in a Zenodo repository, ensuring no duplicates are added. The function can return just the list of dataset
    names or, if specified, also return the path to the local datasets directory.

    Parameters
    ----------
    return_path : bool, optional
            If True, the function returns a tuple containing the list of available dataset names and the path to the
            local datasets directory. If False, only the list of dataset names is returned (default is False).

    Returns
    -------
    list or (list, str)
            If return_path is False, returns a list of strings, each string being the name of an available dataset.
            If return_path is True, returns a tuple where the first element is this list and the second element is a
            string representing the path to the local datasets directory.

    """

    datasets_path = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "datasets",
            "segmentation_annotations",
            os.sep,
        ]
    )
    repository_datasets = get_zenodo_files(
        cat=os.sep.join(["datasets", "segmentation_annotations"])
    )

    available_datasets = natsorted(glob(datasets_path + "*/"))
    available_datasets = [
        m.replace("\\", "/").split("/")[-2] for m in available_datasets
    ]
    for rm in repository_datasets:
        if rm not in available_datasets:
            available_datasets.append(rm)

    if not return_path:
        return available_datasets
    else:
        return available_datasets, datasets_path


def get_signal_datasets_list(return_path=False):
    """
    Retrieves a list of available signal datasets from both the local 'celldetective/datasets/signal_annotations' directory
    and a Zenodo repository, optionally returning the path to the local datasets directory.

    This function compiles a list of available signal datasets by first identifying datasets stored locally within a specified
    path related to the script's directory. It then extends this list with datasets available in a Zenodo repository, ensuring
    no duplicates are added. The function can return just the list of dataset names or, if specified, also return the path to
    the local datasets directory.

    Parameters
    ----------
    return_path : bool, optional
            If True, the function returns a tuple containing the list of available dataset names and the path to the local datasets
            directory. If False, only the list of dataset names is returned (default is False).

    Returns
    -------
    list or (list, str)
            If return_path is False, returns a list of strings, each string being the name of an available dataset. If return_path
            is True, returns a tuple where the first element is this list and the second element is a string representing the path
            to the local datasets directory.

    """

    datasets_path = os.sep.join(
        [
            os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],
            # "celldetective",
            "datasets",
            "signal_annotations",
            os.sep,
        ]
    )
    repository_datasets = get_zenodo_files(
        cat=os.sep.join(["datasets", "signal_annotations"])
    )

    available_datasets = natsorted(glob(datasets_path + "*/"))
    available_datasets = [
        m.replace("\\", "/").split("/")[-2] for m in available_datasets
    ]
    for rm in repository_datasets:
        if rm not in available_datasets:
            available_datasets.append(rm)

    if not return_path:
        return available_datasets
    else:
        return available_datasets, datasets_path
