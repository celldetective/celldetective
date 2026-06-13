"""
Event schema
============

Single source of truth for the **event-detection output contract** shared by the
two detectors in Celldetective:

- the deep-learning detector (``event_detection_models`` →
  ``signals.analyze_signals`` / ``processes.detect_events``), and
- the threshold/query detector (``gui.ClassifierWidget`` /
  ``measure.classify_*`` / ``measure.classify_from_threshold_config``).

Both engines must emit the same columns with the same encoding so that
everything downstream (status coloring, survival analysis, plotting, collapsing
by status) treats their results identically. Keeping the encoding, the column
names, and the per-frame status assembly here — rather than re-deriving them in
each engine — prevents the two paths from silently drifting apart.

This module deliberately depends only on NumPy so it can be imported from any
layer without pulling in heavy dependencies.
"""

from typing import Optional, Tuple

import numpy as np

# --- Track-level class encoding (one value per track) -----------------------
EVENT = 0  # event observed, with a valid time in the t_ column
NO_EVENT = 1  # event not observed within the track
ELSE = 2  # left-censored / miscellaneous (e.g. event already happened, or n/a)

# --- Per-frame status values ------------------------------------------------
STATUS_BEFORE = 0  # before the event (or query condition off)
STATUS_AFTER = 1  # at/after the event (or query condition on)
STATUS_ELSE = 2  # whole track flagged as the ELSE class
STATUS_INVALID = 42  # legacy sentinel for class values above ELSE


def event_column_names(label: Optional[str]) -> Tuple[str, str, str]:
    """Return the ``(class, time, status)`` column names for an event label.

    A ``None`` or empty label maps to the default unsuffixed names used by the
    primary event channel; otherwise the ``<kind>_<label>`` convention is used.

    Parameters
    ----------
    label : str or None
        The event label (e.g. ``"lysis"``), or ``None`` for the default channel.

    Returns
    -------
    (str, str, str)
        ``(class_col, time_col, status_col)``.
    """
    if label is None or label == "":
        return "class", "t0", "status"
    return f"class_{label}", f"t_{label}", f"status_{label}"


def status_from_event(
    timeline: np.ndarray, cclass: float, t0: float
) -> np.ndarray:
    """Build the per-frame status array for one track from its class and time.

    Canonical assembly used by the deep-learning inference paths:

    - frames at or after ``t0`` become :data:`STATUS_AFTER` once a valid
      (positive, non-NaN) event time exists;
    - an :data:`ELSE`-class track is flagged :data:`STATUS_ELSE` throughout;
    - class values above :data:`ELSE` use the legacy :data:`STATUS_INVALID`
      sentinel.

    Parameters
    ----------
    timeline : ndarray
        The track's frame indices.
    cclass : float
        The track's class (see :data:`EVENT` / :data:`NO_EVENT` / :data:`ELSE`).
    t0 : float
        The track's event time (``<= 0`` or NaN means "no event time").

    Returns
    -------
    ndarray
        Per-frame status array, same length as ``timeline``.
    """
    timeline = np.asarray(timeline)
    status = np.zeros_like(timeline)

    if t0 == t0 and t0 > 0:  # t0 == t0 rejects NaN
        status[timeline >= t0] = STATUS_AFTER
    if cclass == ELSE:
        status[:] = STATUS_ELSE
    elif cclass == cclass and cclass > ELSE:
        status[:] = STATUS_INVALID

    return status
