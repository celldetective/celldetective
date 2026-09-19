"""
Read threshold configurations and remember the last ones used in an experiment.

A threshold pipeline is written by the threshold configuration wizard as a JSON
file, usually under ``<experiment>/configs``. Picking one again every session is
tedious, so the last configuration(s) used for each population are recorded in
``<experiment>/configs/last_threshold_configs.json`` and offered again the next
time. Paths inside the experiment are stored relative to it, so the record still
holds once the experiment folder has been moved or opened from another machine.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from celldetective import get_logger

logger = get_logger(__name__)

#: Name of the record, kept next to the configurations it points to.
MEMORY_FILENAME = "last_threshold_configs.json"

#: Keys a threshold configuration cannot do without.
REQUIRED_KEYS = ("target_channel", "thresholds")


def _memory_path(exp_dir: str) -> str:
    """Where the record of the last configurations lives for an experiment."""
    return os.path.join(exp_dir, "configs", MEMORY_FILENAME)


def _population_keys(population: str) -> List[str]:
    """
    The population as given, then its singular / plural twin.

    ``"target"`` and ``"targets"`` are used interchangeably across the software,
    so a configuration remembered under one is found under the other.
    """
    if population.endswith("s"):
        return [population, population[:-1]]
    return [population, f"{population}s"]


def _read_memory(exp_dir: str) -> Dict[str, Any]:
    """Read the record, or an empty one when it is missing or unreadable."""
    path = _memory_path(exp_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            memory = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read the last threshold configurations: {e}")
        return {}
    return memory if isinstance(memory, dict) else {}


def remember_threshold_configs(
    exp_dir: Optional[str], population: str, paths: Union[str, Sequence[str]]
) -> None:
    """
    Record the threshold configuration(s) just used for a population.

    Best-effort: failing to write the record never gets in the way of the
    segmentation it is about.

    Parameters
    ----------
    exp_dir : str or None
        The experiment directory. Nothing is recorded when None.
    population : str
        The population the configurations segment.
    paths : str or list of str
        The configuration file(s), in the order they are applied.
    """

    if not isinstance(exp_dir, str) or not exp_dir or not paths:
        return
    if isinstance(paths, str):
        paths = [paths]

    exp_root = os.path.abspath(exp_dir)
    stored = []
    for path in paths:
        path = os.path.abspath(path)
        try:
            inside = os.path.commonpath([exp_root, path]) == exp_root
        except ValueError:
            # Different drives on Windows.
            inside = False
        stored.append(
            os.path.relpath(path, exp_root).replace(os.sep, "/") if inside else path
        )

    memory = _read_memory(exp_dir)
    for key in _population_keys(population)[1:]:
        memory.pop(key, None)
    memory[population] = stored

    try:
        os.makedirs(os.path.dirname(_memory_path(exp_dir)), exist_ok=True)
        with open(_memory_path(exp_dir), "w") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        logger.warning(f"Could not remember the threshold configurations: {e}")


def recall_threshold_configs(exp_dir: Optional[str], population: str) -> List[str]:
    """
    The threshold configuration(s) last used for a population, if still on disk.

    Parameters
    ----------
    exp_dir : str or None
        The experiment directory.
    population : str
        The population to look up.

    Returns
    -------
    list of str
        Absolute paths, in the order they were applied. Empty when nothing was
        recorded or none of the files exist any more.
    """

    if not isinstance(exp_dir, str) or not exp_dir:
        return []
    memory = _read_memory(exp_dir)
    for key in _population_keys(population):
        stored = memory.get(key)
        if not stored:
            continue
        if isinstance(stored, str):
            stored = [stored]
        paths = [
            p if os.path.isabs(p) else os.path.join(exp_dir, *p.split("/"))
            for p in stored
            if isinstance(p, str)
        ]
        return [os.path.normpath(p) for p in paths if os.path.exists(p)]
    return []


def load_threshold_config(path: str) -> Dict[str, Any]:
    """
    Read a threshold configuration written by the wizard.

    Parameters
    ----------
    path : str
        The JSON file.

    Returns
    -------
    dict
        The configuration.

    Raises
    ------
    ValueError
        If the file cannot be read or lacks a key the segmentation needs.
    """

    try:
        with open(path) as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not read {os.path.basename(path)}: {e}") from e
    if not isinstance(config, dict):
        raise ValueError(f"{os.path.basename(path)} is not a threshold configuration.")
    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is not a threshold configuration "
            f"(missing {', '.join(missing)})."
        )
    return config
