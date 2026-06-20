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


def reclassify_out_of_window(
    classes: np.ndarray, times: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Demote events whose time falls at/after the analyzed window to NO_EVENT.

    When a signal longer than the model input is truncated to ``window`` frames
    (``= model_signal_length``) before inference, an event can only be trusted if
    it was observed *inside* that window: a positive call with a time at or
    beyond ``window`` was never actually seen by the model and is, by the
    window's definition, "not observed". Such tracks are relabelled
    :data:`NO_EVENT` (with time ``-1``) so the deep-learning and threshold
    detectors honour the same truncation contract.

    Parameters
    ----------
    classes : ndarray
        Per-track class predictions.
    times : ndarray
        Per-track event times, aligned with ``classes``.
    window : int
        The analyzed window length in frames (the model signal length).

    Returns
    -------
    (ndarray, ndarray, int)
        The corrected ``(classes, times)`` and the number of tracks demoted.
    """
    classes = np.asarray(classes).copy()
    times = np.asarray(times, dtype=float).copy()
    mask = (classes == EVENT) & (times >= window)
    classes[mask] = NO_EVENT
    times[mask] = -1.0
    return classes, times, int(np.count_nonzero(mask))


def truncate_training_signals(
    signals: np.ndarray,
    classes: np.ndarray,
    times: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Truncate over-long *training* signals to the model window and relabel.

    Training counterpart of :func:`reclassify_out_of_window`, applied to the
    ground-truth annotations so a model is taught the same window contract it
    will be held to at inference: a signal longer than ``window`` frames is cut
    to its first ``window`` frames, and any example whose annotated event time is
    at/after the window is relabelled :data:`NO_EVENT` (time ``-1``) — that event
    is not observable inside the analyzed window. Class labels are accepted as
    1-D integer labels or 2-D one-hot rows (the form is preserved).

    Parameters
    ----------
    signals : ndarray
        3-D training signals ``(samples, time, channels)``.
    classes : ndarray
        Per-sample class labels (1-D) or one-hot rows (2-D).
    times : ndarray
        Per-sample annotated event time, in frames.
    window : int
        The model window length in frames (``model_signal_length``).

    Returns
    -------
    (ndarray, ndarray, ndarray, int)
        ``(signals, classes, times, n_relabelled)`` with ``signals`` truncated
        and ``classes``/``times`` relabelled in place on copies.
    """
    signals = np.asarray(signals)
    if signals.ndim == 3 and signals.shape[1] > window:
        signals = signals[:, :window, :]
    classes = np.asarray(classes).copy()
    times = np.asarray(times, dtype=float).copy()
    one_hot = classes.ndim == 2
    labels = classes.argmax(axis=1) if one_hot else classes
    mask = (labels == EVENT) & (times >= window)
    if one_hot:
        classes[mask] = 0.0
        classes[mask, NO_EVENT] = 1.0
    else:
        classes[mask] = NO_EVENT
    times[mask] = -1.0
    return signals, classes, times, int(np.count_nonzero(mask))
