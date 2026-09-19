"""
Signal-window helpers
=====================

Small helpers that reconcile a **fixed-length** event-detection model with
signals (or annotations) that are longer than its window, shared by the
inference path (:mod:`celldetective.signals`,
:mod:`celldetective.processes.detect_events`) and the training path
(:mod:`celldetective.event_detection_models`).

A signal-detection model only ever sees ``model_signal_length`` frames:

- At **inference**, a track longer than the window is scanned with overlapping
  windows and the per-window ``(class, time)`` predictions are reduced to a
  single event call by *earliest in-window detection*
  (:func:`predict_events_sliding_window`), so an event anywhere along a long
  track can still be found.
- At **training**, an over-long annotated signal is cut to the first window and
  any event past it is relabelled "no event"
  (:func:`truncate_training_signals`), since the fixed-length model is trained
  on window-length, frame-0-anchored examples.

The constants and :func:`truncate_training_signals` depend only on NumPy.
:func:`predict_events_sliding_window` additionally takes a model object but does
not import it, so this module stays import-light.
"""

from typing import Optional, Tuple

import numpy as np

# Track-level class encoding (one value per track), matching the encoding used
# throughout Celldetective's event detection: 0 = event observed (with a valid
# time), 1 = no event observed, 2 = else / left-censored.
EVENT = 0
NO_EVENT = 1
ELSE = 2  # left-censored / miscellaneous (e.g. event already happened, or n/a)


def predict_events_sliding_window(
    model,
    signals: np.ndarray,
    window: int,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict one ``(class, time)`` per track for signals longer than ``window``.

    A fixed-length model only sees ``window`` frames, so a longer track is
    scanned with overlapping windows of length ``window`` and the per-window
    predictions are reduced to a single event call per track by **earliest
    in-window detection**:

    - a window detects the event for a track when it predicts :data:`EVENT` with
      a valid in-window time ``t0`` (``>= 0``); the absolute event time is
      ``window_offset + t0``;
    - across windows, the track's event time is the *earliest* such absolute
      time (events in Celldetective are first-occurrence transitions, and with
      overlapping windows the same event is seen by several windows at
      consistent absolute times);
    - a track no window flags as :data:`EVENT` is :data:`ELSE` if any window
      called it ``ELSE`` (e.g. the event had already happened), otherwise
      :data:`NO_EVENT`.

    Parameters
    ----------
    model : SignalDetectionModel
        Trained model exposing ``predict_class`` and ``predict_time_of_interest``
        (the latter returning window-local frame times, ``-1`` for non-events).
    signals : ndarray
        Signals ``(n_tracks, length, n_channels)`` with ``length > window``.
    window : int
        The model window length in frames (``model_signal_length``).
    stride : int, optional
        Step between successive window starts. Defaults to ``window // 2``
        (50 % overlap), so an event straddling a window boundary is still
        captured whole by a neighbouring window.

    Returns
    -------
    (ndarray, ndarray)
        ``(classes, times)`` of length ``n_tracks``: integer class labels and
        absolute event times in frames (``-1`` where there is no event).
    """
    signals = np.asarray(signals)
    n_tracks, length, _ = signals.shape
    if stride is None:
        stride = max(1, window // 2)

    # Window start offsets, always full windows; append a final flush-right
    # window so the tail of the track is covered even when (length - window) is
    # not a multiple of the stride.
    offsets = list(range(0, length - window + 1, stride))
    if not offsets:
        offsets = [0]
    if offsets[-1] != length - window:
        offsets.append(length - window)

    final_class = np.full(n_tracks, NO_EVENT, dtype=int)
    final_time = np.full(n_tracks, -1.0, dtype=float)
    seen_else = np.zeros(n_tracks, dtype=bool)

    for offset in offsets:
        sub = signals[:, offset : offset + window, :]
        classes = np.asarray(model.predict_class(sub))
        times = np.asarray(
            model.predict_time_of_interest(sub, class_predictions=classes),
            dtype=float,
        )

        is_event = (classes == EVENT) & (times >= 0)
        abs_time = offset + times
        # Keep the earliest absolute detection: take a window's event if the
        # track has no event yet, or this detection is earlier than the one held.
        update = is_event & ((final_class != EVENT) | (abs_time < final_time))
        final_class[update] = EVENT
        final_time[update] = abs_time[update]

        seen_else |= classes == ELSE

    else_mask = (final_class != EVENT) & seen_else
    final_class[else_mask] = ELSE
    final_time[else_mask] = -1.0

    return final_class, final_time


def truncate_training_signals(
    signals: np.ndarray,
    classes: np.ndarray,
    times: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Truncate over-long *training* signals to the model window and relabel.

    Applied to the ground-truth annotations so a fixed-length model is trained on
    window-length, frame-0-anchored examples: a signal longer than ``window``
    frames is cut to its first ``window`` frames, and any example whose annotated
    event time is at/after the window is relabelled :data:`NO_EVENT` (time
    ``-1``) — that event is not observable inside the analyzed window. Class
    labels are accepted as 1-D integer labels or 2-D one-hot rows (the form is
    preserved).

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
