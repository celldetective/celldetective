"""
Schema / conventions
====================

Single source of truth for cross-cutting naming conventions in Celldetective:
the **population modes**, the **per-position trajectory table location**, and
(re-exported) the **event-detection encoding**. Centralizing these prevents the
conventions from being re-derived — and silently drifting — across the dozens of
modules that touch them.

Before this module the ``trajectories_{mode}.csv`` pattern was hardcoded ~70
times across ~25 files, and the mode-normalization logic (``"target" ->
"targets"`` etc.) was re-implemented ~27 times. Route those through here.

Depends only on the standard library and the dependency-free ``event_schema``.
"""

import os
from typing import List

# Re-exported so callers have a single conventions import point.
from celldetective.utils import COLUMN_LABELS  # noqa: F401
from celldetective.utils.event_schema import (  # noqa: F401
    EVENT,
    NO_EVENT,
    ELSE,
    STATUS_BEFORE,
    STATUS_AFTER,
    STATUS_ELSE,
    STATUS_INVALID,
    event_column_names,
    status_from_event,
)

# --- Population modes -------------------------------------------------------
TARGETS = "targets"
EFFECTORS = "effectors"
PAIRS = "pairs"
POPULATIONS: List[str] = [TARGETS, EFFECTORS, PAIRS]


def normalize_population(mode: str) -> str:
    """Canonicalize a population/mode string.

    Maps the singular aliases (``"target"``/``"effector"``) to their canonical
    plural forms; any other value is returned unchanged, so custom population
    names pass through.

    Parameters
    ----------
    mode : str
        A population/mode string.

    Returns
    -------
    str
        The canonical population name.
    """
    m = str(mode).lower()
    if m in ("target", "targets"):
        return TARGETS
    if m in ("effector", "effectors"):
        return EFFECTORS
    return mode


# --- Trajectory table location ---------------------------------------------
def trajectory_table_name(mode: str, extension: str = "csv") -> str:
    """Return the per-position trajectory table filename for a population.

    Parameters
    ----------
    mode : str
        Population/mode (normalized via :func:`normalize_population`).
    extension : str, optional
        File extension without the dot (``"csv"`` default, ``"pkl"`` for the
        pickled table).
    """
    return f"trajectories_{normalize_population(mode)}.{extension}"


def trajectory_table_relpath(mode: str, extension: str = "csv") -> str:
    """Return the table path relative to a position dir (``output/tables/...``)."""
    return os.path.join("output", "tables", trajectory_table_name(mode, extension))


def trajectory_table_path(pos: str, mode: str, extension: str = "csv") -> str:
    """Return the full per-position trajectory table path for a population.

    Equivalent to ``<pos>/output/tables/trajectories_<population>.<extension>``.
    """
    return os.path.join(pos, trajectory_table_relpath(mode, extension))


def label_folder_name(mode: str) -> str:
    """Return the labels folder name for a population.

    Parameters
    ----------
    mode : str
        Population/mode (normalized via :func:`normalize_population`).

    Returns
    -------
    str
        The labels folder name (e.g. ``"labels_targets"``).
    """
    return f"labels_{normalize_population(mode)}"


def backup_label_folder_name(mode: str) -> str:
    """Return the backup labels folder name for a population.

    Parameters
    ----------
    mode : str
        Population/mode (normalized via :func:`normalize_population`).

    Returns
    -------
    str
        The backup labels folder name (e.g. ``"labels_targets.bak"``).
    """
    return f"labels_{normalize_population(mode)}.bak"


def tracking_instructions_name(mode: str) -> str:
    """Return the tracking instructions filename for a population.

    Equivalent to ``tracking_instructions_<normalized_population>.json``.
    """
    return f"tracking_instructions_{normalize_population(mode)}.json"


def tracking_instructions_relpath(mode: str) -> str:
    """Return the relative path of the tracking instructions JSON file.

    Equivalent to ``configs/tracking_instructions_<normalized_population>.json``.
    """
    return os.path.join("configs", tracking_instructions_name(mode))


def tracking_instructions_path(exp_dir: str, mode: str) -> str:
    """Return the full path of the tracking instructions JSON file.

    Equivalent to ``<exp_dir>/configs/tracking_instructions_<normalized_population>.json``.
    """
    return os.path.join(exp_dir, tracking_instructions_relpath(mode))


def measurement_instructions_name(mode: str) -> str:
    """Return the measurement instructions filename for a population.

    Equivalent to ``measurement_instructions_<normalized_population>.json``.
    """
    return f"measurement_instructions_{normalize_population(mode)}.json"


def measurement_instructions_relpath(mode: str) -> str:
    """Return the relative path of the measurement instructions JSON file.

    Equivalent to ``configs/measurement_instructions_<normalized_population>.json``.
    """
    return os.path.join("configs", measurement_instructions_name(mode))


def measurement_instructions_path(exp_dir: str, mode: str) -> str:
    """Return the full path of the measurement instructions JSON file.

    Equivalent to ``<exp_dir>/configs/measurement_instructions_<normalized_population>.json``.
    """
    return os.path.join(exp_dir, measurement_instructions_relpath(mode))


def segmentation_instructions_name(mode: str) -> str:
    """Return the segmentation instructions filename for a population.

    Equivalent to ``segmentation_instructions_<normalized_population>.json``.
    """
    return f"segmentation_instructions_{normalize_population(mode)}.json"


def segmentation_instructions_relpath(mode: str) -> str:
    """Return the relative path of the segmentation instructions JSON file.

    Equivalent to ``configs/segmentation_instructions_<normalized_population>.json``.
    """
    return os.path.join("configs", segmentation_instructions_name(mode))


def segmentation_instructions_path(exp_dir: str, mode: str) -> str:
    """Return the full path of the segmentation instructions JSON file.

    Equivalent to ``<exp_dir>/configs/segmentation_instructions_<normalized_population>.json``.
    """
    return os.path.join(exp_dir, segmentation_instructions_relpath(mode))


def napari_trajectories_name(mode: str) -> str:
    """Return the napari trajectories filename for a population.

    Uses ``"napari_target_trajectories.npy"``, ``"napari_effector_trajectories.npy"``,
    or ``"napari_<mode>_trajectories.npy"`` for custom modes.
    """
    m = str(mode).lower()
    if m in ("target", "targets"):
        return "napari_target_trajectories.npy"
    if m in ("effector", "effectors"):
        return "napari_effector_trajectories.npy"
    return f"napari_{mode}_trajectories.npy"


def normalize_path(path: str) -> str:
    """Normalize a path by stripping trailing separators and standardizing slashes.

    Parameters
    ----------
    path : str
        The input path.

    Returns
    -------
    str
        The normalized path.
    """
    if not path:
        return path
    return os.path.normpath(path)


def ensure_trailing_sep(path: str) -> str:
    """Ensure the path ends with a directory separator.

    Parameters
    ----------
    path : str
        The input path.

    Returns
    -------
    str
        The path with a trailing separator.
    """
    if not path:
        return path
    normalized = os.path.normpath(path)
    return normalized + os.sep



