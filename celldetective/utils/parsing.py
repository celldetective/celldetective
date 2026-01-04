import configparser
import json
import os
import re
from pathlib import PurePath, Path
from typing import Union, Dict

import numpy as np


def _get_normalize_kwargs_from_config(config):

    if isinstance(config, str):
        if os.path.exists(config):
            with open(config) as cfg:
                config = json.load(cfg)
        else:
            print("Configuration could not be loaded...")
            os.abort()

    normalization_percentile = config["normalization_percentile"]
    normalization_clip = config["normalization_clip"]
    normalization_values = config["normalization_values"]
    normalize_kwargs = _get_normalize_kwargs(
        normalization_percentile, normalization_values, normalization_clip
    )

    return normalize_kwargs


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
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


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
            print(
                f"Warning: The channel {c} required by the model is not available in your data..."
            )
            channels.append(None)
    if np.all([c is None for c in channels]):
        channels = None

    return channels


def _extract_nbr_channels_from_config(config, return_names=False):
    """

    Examples
    --------
    >>> config = "path/to/config_file.ini"
    >>> nbr_channels = _extract_channel_indices_from_config(config)
    >>> print(nbr_channels)
    # 4
    """

    # V2
    nbr_channels = 0
    channels = []
    try:
        fields = config_section_to_dict(config, "Channels")
        for c in fields:
            try:
                channel = int(config_section_to_dict(config, "Channels")[c])
                nbr_channels += 1
                channels.append(c)
            except:
                pass
    except:
        pass

    if nbr_channels == 0:

        # Read channels LEGACY
        nbr_channels = 0
        channels = []
        try:
            brightfield_channel = int(
                config_section_to_dict(config, "MovieSettings")["brightfield_channel"]
            )
            nbr_channels += 1
            channels.append("brightfield_channel")
        except:
            brightfield_channel = None

        try:
            live_nuclei_channel = int(
                config_section_to_dict(config, "MovieSettings")["live_nuclei_channel"]
            )
            nbr_channels += 1
            channels.append("live_nuclei_channel")
        except:
            live_nuclei_channel = None

        try:
            dead_nuclei_channel = int(
                config_section_to_dict(config, "MovieSettings")["dead_nuclei_channel"]
            )
            nbr_channels += 1
            channels.append("dead_nuclei_channel")
        except:
            dead_nuclei_channel = None

        try:
            effector_fluo_channel = int(
                config_section_to_dict(config, "MovieSettings")["effector_fluo_channel"]
            )
            nbr_channels += 1
            channels.append("effector_fluo_channel")
        except:
            effector_fluo_channel = None

        try:
            adhesion_channel = int(
                config_section_to_dict(config, "MovieSettings")["adhesion_channel"]
            )
            nbr_channels += 1
            channels.append("adhesion_channel")
        except:
            adhesion_channel = None

        try:
            fluo_channel_1 = int(
                config_section_to_dict(config, "MovieSettings")["fluo_channel_1"]
            )
            nbr_channels += 1
            channels.append("fluo_channel_1")
        except:
            fluo_channel_1 = None

        try:
            fluo_channel_2 = int(
                config_section_to_dict(config, "MovieSettings")["fluo_channel_2"]
            )
            nbr_channels += 1
            channels.append("fluo_channel_2")
        except:
            fluo_channel_2 = None

    if return_names:
        return nbr_channels, channels
    else:
        return nbr_channels


def _extract_labels_from_config(config, number_of_wells):
    """

    Extract each well's biological condition from the configuration file

    Parameters
    ----------

    config: str,
                    path to the configuration file

    number_of_wells: int,
                    total number of wells in the experiment

    Returns
    -------

    labels: string of the biological condition for each well

    """

    # Deprecated, need to read metadata to extract concentration units and discard non essential fields

    try:
        concentrations = config_section_to_dict(config, "Labels")[
            "concentrations"
        ].split(",")
        cell_types = config_section_to_dict(config, "Labels")["cell_types"].split(",")
        antibodies = config_section_to_dict(config, "Labels")["antibodies"].split(",")
        pharmaceutical_agents = config_section_to_dict(config, "Labels")[
            "pharmaceutical_agents"
        ].split(",")
        index = np.arange(len(concentrations)).astype(int) + 1
        if not np.all(pharmaceutical_agents == "None"):
            labels = [
                f"W{idx}: [CT] " + a + "; [Ab] " + b + " @ " + c + " pM " + d
                for idx, a, b, c, d in zip(
                    index, cell_types, antibodies, concentrations, pharmaceutical_agents
                )
            ]
        else:
            labels = [
                f"W{idx}: [CT] " + a + "; [Ab] " + b + " @ " + c + " pM "
                for idx, a, b, c in zip(index, cell_types, antibodies, concentrations)
            ]

    except Exception as e:
        print(
            f"{e}: the well labels cannot be read from the concentration and cell_type fields"
        )
        labels = np.linspace(0, number_of_wells - 1, number_of_wells, dtype=str)

    return labels


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


def _get_normalize_kwargs(
    normalization_percentile, normalization_values, normalization_clip
):

    values = []
    percentiles = []
    for k in range(len(normalization_percentile)):
        if normalization_percentile[k]:
            percentiles.append(normalization_values[k])
            values.append(None)
        else:
            percentiles.append(None)
            values.append(normalization_values[k])

    return {"percentiles": percentiles, "values": values, "clip": normalization_clip}


def demangle_column_name(name):
    if name.startswith("BACKTICK_QUOTED_STRING_"):
        # Unquote backtick-quoted string.
        return (
            name[len("BACKTICK_QUOTED_STRING_") :]
            .replace("_DOT_", ".")
            .replace("_SLASH_", "/")
            .replace("_MINUS_", "-")
            .replace("_PLUS_", "+")
            .replace("_PERCENT_", "%")
            .replace("_STAR_", "*")
            .replace("_LPAR_", "(")
            .replace("_RPAR_", ")")
            .replace("_AMPER_", "&")
        )
    return name


def extract_cols_from_query(query: str):

    backtick_pattern = r"`([^`]+)`"
    backticked = set(re.findall(backtick_pattern, query))

    # 2. Remove backtick sections so they don't get double-counted
    cleaned_query = re.sub(backtick_pattern, "", query)

    # 3. Extract bare identifiers from the remaining string
    identifier_pattern = r"\b([A-Za-z_]\w*)\b"
    bare = set(re.findall(identifier_pattern, cleaned_query))

    # 4. Remove Python keywords, operators, and pandas builtins
    import pandas as pd

    blacklist = (
        set(dir(pd))
        | set(dir(__builtins__))
        | {"and", "or", "not", "in", "True", "False"}
    )
    bare = {c for c in bare if c not in blacklist}
    cols = backticked | bare

    return list([demangle_column_name(c) for c in cols])


def parse_isotropic_radii(string):
    """
    Parse a string representing isotropic radii into a structured list.

    This function extracts integer values and ranges (denoted by square brackets)
    from a string input and returns them as a list. Single values are stored as integers,
    while ranges are represented as lists of two integers.

    Parameters
    ----------
    string : str
            The input string containing radii and ranges, separated by commas or spaces.
            Ranges should be enclosed in square brackets, e.g., `[1 2]`.

    Returns
    -------
    list
            A list of parsed radii where:
            - Single integers are included as `int`.
            - Ranges are included as two-element lists `[start, end]`.

    Examples
    --------
    Parse a string with single radii and ranges:

    >>> parse_isotropic_radii("1, [2 3], 4")
    [1, [2, 3], 4]

    Handle inputs with mixed delimiters:

    >>> parse_isotropic_radii("5 [6 7], 8")
    [5, [6, 7], 8]

    Notes
    -----
    - The function splits the input string by commas or spaces.
    - It identifies ranges using square brackets and assumes that ranges are always
      two consecutive values.
    - Non-integer sections of the string are ignored.

    """

    sections = re.split(r"[ ,]", string)
    radii = []
    for k, s in enumerate(sections):
        if s.isdigit():
            radii.append(int(s))
        if "[" in s:
            ring = [int(s.replace("[", "")), int(sections[k + 1].replace("]", ""))]
            radii.append(ring)
        else:
            pass
    return radii
