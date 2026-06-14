"""Unit tests for event-detection internals refactored during the audit:

- shared channel resolver (train/inference parity)
- normalization fit/apply split (per-batch pooling support)
- pad-length guard
"""

import numpy as np
import pytest

from celldetective.utils.dataset_helpers import resolve_signal_channels


class TestResolveSignalChannels:
    def test_exact_match_preferred_over_substring(self):
        available = ["RFP", "RFP_circle", "nk_RFP"]
        assert resolve_signal_channels(["RFP"], available) == ["RFP"]

    def test_prefix_preferred_over_substring(self):
        available = ["nk_RFP", "RFP_circle"]
        # No exact "RFP"; prefix match ("RFP_circle") beats substring ("nk_RFP").
        assert resolve_signal_channels(["RFP"], available) == ["RFP_circle"]

    def test_substring_fallback(self):
        available = ["nk_RFP_mean"]
        assert resolve_signal_channels(["RFP"], available) == ["nk_RFP_mean"]

    def test_multiple_required_in_order(self):
        available = ["GFP", "RFP", "RFP_circle"]
        assert resolve_signal_channels(["RFP", "GFP"], available) == ["RFP", "GFP"]

    def test_missing_returns_none(self):
        assert resolve_signal_channels(["YFP"], ["RFP", "GFP"]) is None

    def test_no_longer_prefers_circle(self):
        # Regression guard: training used to prefer a "circle" column; it must now
        # match inference (exact name wins) so train and inference agree.
        available = ["RFP_circle", "RFP"]
        assert resolve_signal_channels(["RFP"], available) == ["RFP"]


class TestNormalizationFitApply:
    def _make_set(self):
        # 2 samples, 5 timepoints, 1 channel; no zeros so nothing is masked out.
        return np.array(
            [
                [[1.0], [2.0], [3.0], [4.0], [5.0]],
                [[2.0], [4.0], [6.0], [8.0], [10.0]],
            ],
            dtype=np.float32,
        )

    def test_compute_stats_fixed_values(self):
        from celldetective.event_detection_models import compute_normalization_stats

        x = self._make_set()
        stats = compute_normalization_stats(
            x,
            ["chan"],
            normalization_percentile=[False],
            normalization_values=[[0.0, 10.0]],
        )
        assert stats == [[0.0, 10.0]]

    def test_apply_with_fitted_values_is_deterministic(self):
        from celldetective.event_detection_models import normalize_signal_set

        x1 = self._make_set()
        x2 = self._make_set()

        # Applying an explicit range must ignore the per-call data distribution,
        # so a subset normalizes identically to the full set.
        fitted = [[0.0, 10.0]]
        out_full = normalize_signal_set(
            x1, ["chan"], normalization_percentile=[False], fitted_values=fitted
        )
        out_subset = normalize_signal_set(
            x2[:1], ["chan"], normalization_percentile=[False], fitted_values=fitted
        )
        np.testing.assert_allclose(out_full[0], out_subset[0])
        # value 10 -> 1.0 with range [0,10]
        assert out_full[1, -1, 0] == pytest.approx(1.0)

    def test_fit_then_apply_matches_inline(self):
        from celldetective.event_detection_models import (
            compute_normalization_stats,
            normalize_signal_set,
        )

        inline = normalize_signal_set(
            self._make_set(), ["chan"], normalization_percentile=[True]
        )
        x = self._make_set()
        stats = compute_normalization_stats(
            x, ["chan"], normalization_percentile=[True]
        )
        applied = normalize_signal_set(
            self._make_set(),
            ["chan"],
            normalization_percentile=[True],
            fitted_values=stats,
        )
        np.testing.assert_allclose(inline, applied)


class TestPadGuard:
    def test_pad_raises_when_signal_longer_than_model(self):
        from celldetective.event_detection_models import pad_to_model_length

        x = np.zeros((2, 50, 1))
        with pytest.raises(ValueError):
            pad_to_model_length(x, 32)

    def test_pad_extends_with_edge(self):
        from celldetective.event_detection_models import pad_to_model_length

        x = np.array([[[1.0], [2.0]]])  # 1 sample, len 2
        out = pad_to_model_length(x, 4)
        assert out.shape == (1, 4, 1)
        # edge padding repeats the last value
        assert out[0, -1, 0] == pytest.approx(2.0)


class TestReclassifyOutOfWindow:
    def test_event_beyond_window_demoted_to_no_event(self):
        from celldetective.utils.event_schema import (
            reclassify_out_of_window,
            EVENT,
            NO_EVENT,
        )

        classes = np.array([EVENT, EVENT])
        times = np.array([10.0, 200.0])
        cc, tt, n = reclassify_out_of_window(classes, times, window=128)
        assert n == 1
        # in-window event kept
        assert cc[0] == EVENT and tt[0] == pytest.approx(10.0)
        # out-of-window event demoted with time reset to -1
        assert cc[1] == NO_EVENT and tt[1] == pytest.approx(-1.0)

    def test_event_exactly_at_window_is_demoted(self):
        from celldetective.utils.event_schema import (
            reclassify_out_of_window,
            EVENT,
            NO_EVENT,
        )

        cc, tt, n = reclassify_out_of_window(
            np.array([EVENT]), np.array([128.0]), window=128
        )
        assert n == 1 and cc[0] == NO_EVENT and tt[0] == pytest.approx(-1.0)

    def test_non_event_classes_untouched(self):
        from celldetective.utils.event_schema import (
            reclassify_out_of_window,
            NO_EVENT,
            ELSE,
        )

        # NO_EVENT/ELSE tracks are never touched, even with a large time value.
        classes = np.array([NO_EVENT, ELSE])
        times = np.array([500.0, 500.0])
        cc, tt, n = reclassify_out_of_window(classes, times, window=128)
        assert n == 0
        assert cc[0] == NO_EVENT and cc[1] == ELSE
        np.testing.assert_allclose(tt, [500.0, 500.0])

    def test_no_change_leaves_inputs_intact(self):
        from celldetective.utils.event_schema import reclassify_out_of_window, EVENT

        classes = np.array([EVENT, EVENT])
        times = np.array([5.0, 20.0])
        cc, tt, n = reclassify_out_of_window(classes, times, window=128)
        assert n == 0
        np.testing.assert_array_equal(cc, classes)
        np.testing.assert_allclose(tt, times)


class TestTruncateTrainingSignals:
    def test_truncates_and_relabels_out_of_window_event(self):
        from celldetective.utils.event_schema import (
            truncate_training_signals,
            EVENT,
            NO_EVENT,
        )

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
        from celldetective.utils.event_schema import (
            truncate_training_signals,
            EVENT,
            NO_EVENT,
        )

        signals = np.ones((2, 200, 1))
        classes = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)  # both EVENT
        times = np.array([10.0, 150.0])
        s, c, t, n = truncate_training_signals(signals, classes, times, window=128)
        assert c.shape == (2, 3)  # one-hot form preserved
        assert n == 1
        assert c[0].argmax() == EVENT
        assert c[1].argmax() == NO_EVENT and t[1] == pytest.approx(-1.0)

    def test_no_truncation_when_within_window(self):
        from celldetective.utils.event_schema import truncate_training_signals, EVENT

        signals = np.ones((1, 64, 1))
        classes = np.array([EVENT])
        times = np.array([30.0])
        s, c, t, n = truncate_training_signals(signals, classes, times, window=128)
        assert s.shape == (1, 64, 1)  # unchanged
        assert n == 0 and c[0] == EVENT and t[0] == pytest.approx(30.0)
