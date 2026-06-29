"""Unit tests for the model-window helpers in
:mod:`celldetective.utils.signal_windowing`.

A fixed-length event-detection model must still handle signals longer than its
window. At inference, long tracks are scanned with overlapping windows and
reduced to one event call by earliest in-window detection
(:func:`predict_events_sliding_window`); at training, over-long annotations are
truncated to the window and relabelled (:func:`truncate_training_signals`).
"""

import numpy as np
import pytest

from celldetective.utils.signal_windowing import (
    predict_events_sliding_window,
    truncate_training_signals,
    EVENT,
    NO_EVENT,
    ELSE,
)


class _FakeModel:
    """Minimal stand-in for ``SignalDetectionModel``.

    The test signals encode each frame's *absolute* index as the channel value
    (``signals[:, t, 0] == t``), so from a window slice the model can recover the
    window's absolute offset. It reports :data:`EVENT` for a track when the
    configured ``event_time`` falls inside the slice, with the correct
    window-local time; tracks listed in ``else_tracks`` always report
    :data:`ELSE`; everything else is :data:`NO_EVENT`.
    """

    def __init__(self, event_time, else_tracks=()):
        self.event_time = event_time
        self.else_tracks = set(else_tracks)
        self._last_classes = None

    def predict_class(self, sub):
        n = sub.shape[0]
        starts = sub[:, 0, 0]
        ends = sub[:, -1, 0]
        classes = np.full(n, NO_EVENT, dtype=int)
        for k in range(n):
            if k in self.else_tracks:
                classes[k] = ELSE
            elif (
                self.event_time is not None
                and starts[k] <= self.event_time <= ends[k]
            ):
                classes[k] = EVENT
        self._last_classes = classes
        return classes

    def predict_time_of_interest(self, sub, class_predictions=None):
        cls = class_predictions if class_predictions is not None else self._last_classes
        starts = sub[:, 0, 0]
        times = np.full(sub.shape[0], -1.0)
        for k in range(sub.shape[0]):
            if cls[k] == EVENT:
                times[k] = self.event_time - starts[k]  # window-local frame
        return times


def _ramp_signals(n_tracks, length, n_channels=1):
    """Signals whose first channel equals the absolute frame index."""
    sig = np.zeros((n_tracks, length, n_channels))
    sig[:, :, 0] = np.arange(length)[None, :]
    return sig


class TestPredictEventsSlidingWindow:
    def test_event_past_first_window_is_found_at_absolute_time(self):
        window = 128
        signals = _ramp_signals(1, 300)
        model = _FakeModel(event_time=200)
        classes, times = predict_events_sliding_window(model, signals, window)
        assert classes[0] == EVENT
        # absolute time recovered exactly, despite being well past frame 128
        assert times[0] == pytest.approx(200.0)

    def test_overlapping_windows_agree_on_earliest_time(self):
        # An event near a window boundary is seen by several overlapping windows;
        # the reduced absolute time must be the single true time, not doubled.
        window = 100
        signals = _ramp_signals(1, 400)
        model = _FakeModel(event_time=150)  # covered by windows at 100 and 150
        classes, times = predict_events_sliding_window(model, signals, window)
        assert classes[0] == EVENT and times[0] == pytest.approx(150.0)

    def test_no_event_track(self):
        window = 64
        signals = _ramp_signals(2, 200)
        model = _FakeModel(event_time=None)
        classes, times = predict_events_sliding_window(model, signals, window)
        np.testing.assert_array_equal(classes, [NO_EVENT, NO_EVENT])
        np.testing.assert_allclose(times, [-1.0, -1.0])

    def test_else_track_when_never_event(self):
        window = 64
        signals = _ramp_signals(2, 200)
        # track 0 sees an event; track 1 is ELSE in every window
        model = _FakeModel(event_time=30, else_tracks=(1,))
        classes, times = predict_events_sliding_window(model, signals, window)
        assert classes[0] == EVENT and times[0] == pytest.approx(30.0)
        assert classes[1] == ELSE and times[1] == pytest.approx(-1.0)

    def test_length_not_multiple_of_stride_covers_tail(self):
        # length - window = 173 is not a multiple of the half-window stride (50);
        # the flush-right window must still catch an event in the tail.
        window = 100
        signals = _ramp_signals(1, 273)
        model = _FakeModel(event_time=265)
        classes, times = predict_events_sliding_window(model, signals, window)
        assert classes[0] == EVENT and times[0] == pytest.approx(265.0)


class TestTruncateTrainingSignals:
    def test_truncates_and_relabels_out_of_window_event(self):
        signals = np.ones((3, 200, 2))
        classes = np.array([EVENT, EVENT, NO_EVENT])
        times = np.array([50.0, 150.0, -1.0])
        s, c, t, n = truncate_training_signals(signals, classes, times, window=128)
        assert s.shape == (3, 128, 2)
        assert n == 1
        # in-window event kept
        assert c[0] == EVENT and t[0] == pytest.approx(50.0)
        # event past the window relabelled
        assert c[1] == NO_EVENT and t[1] == pytest.approx(-1.0)
        # non-event untouched
        assert c[2] == NO_EVENT and t[2] == pytest.approx(-1.0)

    def test_preserves_one_hot_labels(self):
        signals = np.ones((2, 200, 1))
        classes = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)  # both EVENT
        times = np.array([10.0, 150.0])
        s, c, t, n = truncate_training_signals(signals, classes, times, window=128)
        assert c.shape == (2, 3)  # one-hot form preserved
        assert n == 1
        assert c[0].argmax() == EVENT
        assert c[1].argmax() == NO_EVENT and t[1] == pytest.approx(-1.0)

    def test_no_truncation_when_within_window(self):
        signals = np.ones((1, 64, 1))
        classes = np.array([EVENT])
        times = np.array([30.0])
        s, c, t, n = truncate_training_signals(signals, classes, times, window=128)
        assert s.shape == (1, 64, 1)  # unchanged
        assert n == 0 and c[0] == EVENT and t[0] == pytest.approx(30.0)
