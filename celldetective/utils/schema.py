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
