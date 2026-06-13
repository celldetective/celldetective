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
