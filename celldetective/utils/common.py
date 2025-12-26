import os
import configparser
from pathlib import PosixPath, PurePosixPath, WindowsPath, Path, PurePath
from typing import Union, Dict, List
import numpy as np
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


def get_software_location() -> str:
    """
    Get the installation folder of celldetective.

    Returns
    -------
    str
        Path to the celldetective installation folder.
    """
    return str(Path(__file__).parent.parent.parent)


def get_config(experiment: Union[str, Path]) -> str:
    """
    Retrieves the path to the configuration file for a given experiment.

    Parameters
    ----------
    experiment : str
            The file system path to the directory of the experiment project.

    Returns
    -------
    str
            The full path to the configuration file (`config.ini`) within the experiment directory.

    Raises
    ------
    AssertionError
            If the `config.ini` file does not exist in the specified experiment directory.

    Notes
    -----
    - The function ensures that the provided experiment path ends with the appropriate file separator (`os.sep`)
      before appending `config.ini` to locate the configuration file.
    - The configuration file is expected to be named `config.ini` and located at the root of the experiment directory.

    Example
    -------
    >>> experiment = "/path/to/experiment"
    >>> config_path = get_config(experiment)
    >>> print(config_path)
    '/path/to/experiment/config.ini'

    """

    if isinstance(experiment, (PosixPath, PurePosixPath, WindowsPath)):
        experiment = str(experiment)

    if not experiment.endswith(os.sep):
        experiment += os.sep

    config = experiment + "config.ini"
    config = rf"{config}"

    assert os.path.exists(
        config
    ), "The experiment configuration could not be located..."
    return config


def config_section_to_dict(
    path: Union[str, PurePath, Path], section: str
) -> Union[Dict, None]:
    """
    Parse the config file to extract experiment parameters
    following https://wiki.python.org/moin/ConfigParserExamples

    Parameters
    ----------

    path: str
                    path to the config.ini file

    section: str
                    name of the section that contains the parameter

    Returns
    -------

    dict1: dictionary

    Examples
    --------
    >>> config = "path/to/config_file.ini"
    >>> section = "Channels"
    >>> channel_dictionary = config_section_to_dict(config,section)
    >>> print(channel_dictionary)
    # {'brightfield_channel': '0',
    #  'live_nuclei_channel': 'nan',
    #  'dead_nuclei_channel': 'nan',
    #  'effector_fluo_channel': 'nan',
    #  'adhesion_channel': '1',
    #  'fluo_channel_1': 'nan',
    #  'fluo_channel_2': 'nan',
    #  'fitc_channel': '2',
    #  'cy5_channel': '3'}
    """

    Config = configparser.ConfigParser(interpolation=None)
    Config.read(path)
    dict1 = {}
    try:
        options = Config.options(section)
    except:
        return None
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                logger.debug("skip: %s" % option)
        except:
            logger.error("exception on %s!" % option)
            dict1[option] = None
    return dict1


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


def _extract_channel_indices_from_config(config, channels_to_extract):
    """
    Extracts the indices of specified channels from a configuration object.

    This function attempts to map required channel names to their respective indices as specified in a
    configuration file. It supports two versions of configuration parsing: a primary method (V2) and a
    fallback legacy method. If the required channels are not found using the primary method, the function
    attempts to find them using the legacy configuration settings.

    Parameters
    ----------
    config : ConfigParser object
            The configuration object parsed from a .ini or similar configuration file that includes channel settings.
    channels_to_extract : list of str
            A list of channel names for which indices are to be extracted from the configuration settings.

    Returns
    -------
    list of int or None
            A list containing the indices of the specified channels as found in the configuration settings.
            If a channel cannot be found, None is appended in its place. If an error occurs during the extraction
            process, the function returns None.

    Notes
    -----
    - This function is designed to be flexible, accommodating changes in configuration file structure by
      checking multiple sections for the required information.
    - The configuration file is expected to contain either "Channels" or "MovieSettings" sections with mappings
      from channel names to indices.
    - An error message is printed if a required channel cannot be found, advising the user to check the
      configuration file.

    Examples
    --------
    >>> config = "path/to/config_file.ini"
    >>> channels_to_extract = ['adhesion_channel', 'brightfield_channel']
    >>> channel_indices = _extract_channel_indices_from_config(config, channels_to_extract)
    >>> print(channel_indices)
    # [1, 0] or None if an error occurs or the channels are not found.
    """

    if isinstance(channels_to_extract, str):
        channels_to_extract = [channels_to_extract]

    channels = []
    for c in channels_to_extract:
        try:
            c1 = int(config_section_to_dict(config, "Channels")[c])
            channels.append(c1)
        except Exception as e:
            logger.warning(
                f"Warning: The channel {c} required by the model is not available in your data..."
            )
            channels.append(None)
    if np.all([c is None for c in channels]):
        channels = None

    return channels


def _extract_channels_from_config(config):
    """
    Extracts channel names and their indices from an experiment configuration.

    Parameters
    ----------
    config : path to config file (.ini)
            The configuration object parsed from an experiment's .ini or similar configuration file.

    Returns
    -------
    tuple
            A tuple containing two numpy arrays: `channel_names` and `channel_indices`. `channel_names` includes
            the names of the channels as specified in the configuration, and `channel_indices` includes their
            corresponding indices. Both arrays are ordered according to the channel indices.

    Examples
    --------
    >>> config = "path/to/config_file.ini"
    >>> channels, indices = _extract_channels_from_config(config)
    >>> print(channels)
    # array(['brightfield_channel', 'adhesion_channel', 'fitc_channel',
    #    'cy5_channel'], dtype='<U19')
    >>> print(indices)
    # array([0, 1, 2, 3])
    """

    channel_names = []
    channel_indices = []
    try:
        fields = config_section_to_dict(config, "Channels")
        for c in fields:
            try:
                idx = int(config_section_to_dict(config, "Channels")[c])
                channel_names.append(c)
                channel_indices.append(idx)
            except:
                pass
    except:
        pass

    channel_indices = np.array(channel_indices)
    channel_names = np.array(channel_names)
    reorder = np.argsort(channel_indices)
    channel_indices = channel_indices[reorder]
    channel_names = channel_names[reorder]

    return channel_names, channel_indices


def extract_experiment_channels(experiment):
    """
    Extracts channel names and their indices from an experiment project.

    Parameters
    ----------
    experiment : str
            The file system path to the directory of the experiment project.

    Returns
    -------
    tuple
            A tuple containing two numpy arrays: `channel_names` and `channel_indices`. `channel_names` includes
            the names of the channels as specified in the configuration, and `channel_indices` includes their
            corresponding indices. Both arrays are ordered according to the channel indices.

    Examples
    --------
    >>> experiment = "path/to/my_experiment"
    >>> channels, indices = extract_experiment_channels(experiment)
    >>> print(channels)
    # array(['brightfield_channel', 'adhesion_channel', 'fitc_channel',
    #    'cy5_channel'], dtype='<U19')
    >>> print(indices)
    # array([0, 1, 2, 3])
    """

    config = get_config(experiment)
    return _extract_channels_from_config(config)
