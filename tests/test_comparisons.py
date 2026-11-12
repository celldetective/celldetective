"""Tests of celldetective.comparisons: tests between survival curves and responses of
synchronised signals."""

import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from celldetective.comparisons import (
    holm,
    pairwise_logrank,
    window_responses,
)

FOLLOW_UP = 100.0


def _km(rng, rate, n=60, end=FOLLOW_UP):
    t = rng.exponential(1 / rate, n)
    events = (t <= end).astype(int)
    return KaplanMeierFitter().fit(np.minimum(t, end), event_observed=events)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_holm_matches_definition():
    p = [0.01, 0.04, np.nan, 0.03]
    # sorted 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, 1*0.04 then running max
    assert np.allclose(holm(p)[[0, 3, 1]], [0.03, 0.06, 0.06])
    assert np.isnan(holm(p)[2])


def test_holm_caps_at_one():
    assert holm([0.6, 0.7]).max() == 1.0


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


def _well_fits(rng, rates, n=200):
    return {w: _km(rng, rate, n) for w, rate in rates.items()}


def test_pairwise_logrank_is_lifelines_log_rank():
    rng = np.random.default_rng(4)
    fits = _well_fits(rng, {"A": 0.01, "B": 0.02})
    pivot = pairwise_logrank(fits, adjust=False)
    a, b = fits["A"], fits["B"]
    expected = logrank_test(a.durations, b.durations, a.event_observed, b.event_observed).p_value
    assert pivot.loc["A", "B"] == pytest.approx(expected)
    assert pivot.loc["B", "A"] == pivot.loc["A", "B"]
    assert np.isnan(pivot.loc["A", "A"])


def test_pairwise_logrank_adjusts_for_the_pairs():
    rng = np.random.default_rng(5)
    fits = _well_fits(rng, {"A": 0.01, "B": 0.011, "C": 0.012})
    raw, adjusted = pairwise_logrank(fits, adjust=False), pairwise_logrank(fits)
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    assert np.allclose([adjusted.loc[p] for p in pairs], holm([raw.loc[p] for p in pairs]))
    assert (adjusted.to_numpy() == adjusted.to_numpy().T)[~np.eye(3, dtype=bool)].all()


def test_pairwise_logrank_is_calibrated_and_has_power():
    """No effect: about 5% of runs claim one. HR 2: almost all runs do."""
    rng = np.random.default_rng(11)
    null = [pairwise_logrank(_well_fits(rng, {"A": 0.01, "B": 0.01}, n=100)).loc["A", "B"] for _ in range(200)]
    assert np.mean(np.array(null) < 0.05) <= 0.10
    effect = [pairwise_logrank(_well_fits(rng, {"A": 0.01, "B": 0.02}, n=100)).loc["A", "B"] for _ in range(30)]
    assert np.mean(np.array(effect) < 0.05) >= 0.8


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

TIMELINE = np.arange(-20, 21, dtype=float)


def test_window_response_of_a_step():
    m = np.zeros((2, len(TIMELINE)))
    m[:, TIMELINE >= 0] = 2.0
    r = window_responses(m, TIMELINE, baseline=(-10, -1), response=(0, 9))
    assert np.allclose(r, 2.0)


def test_window_response_needs_enough_frames():
    m = np.zeros((2, len(TIMELINE)))
    m[1, (TIMELINE >= 0) & (TIMELINE <= 7)] = np.nan  # 8 of 10 response frames missing
    r = window_responses(m, TIMELINE, baseline=(-10, -1), response=(0, 9), min_fraction=0.5)
    assert r[0] == 0 and np.isnan(r[1])


def test_window_outside_the_timeline_is_refused():
    with pytest.raises(ValueError):
        window_responses(np.zeros((1, len(TIMELINE))), TIMELINE, (-100, -90), (0, 5))
