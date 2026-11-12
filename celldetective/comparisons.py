"""Statistical comparison of survival curves and synchronised signals.

Groups (usually wells) are compared on their cells pooled. Within one
experiment that answers "do these wells differ?": with one well per
condition, a difference may come from the well rather than its condition,
which only repeating the experiment can tell.

- Survival curves: pairwise log-rank tests.
- Synchronised signals: the response of each cell, the mean over a window
  after the synchronising time minus the mean over a baseline before it. The
  responses are values per cell, compared like any measurement with
  :func:`celldetective.utils.stats.test_2samp_generic`.

Free of Qt; works on the fitted ``KaplanMeierFitter`` of
:func:`celldetective.events.compute_survival` and on the aligned matrices of
:func:`celldetective.signals.mean_signal`.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "holm",
    "pairwise_logrank",
    "window_responses",
]


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values, leaving NaN where no test was made.

    Controls the probability of any false claim across the comparisons, as
    Bonferroni does, while losing less power.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    out[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    return out


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------


def _weights(fit) -> np.ndarray:
    weights = getattr(fit, "weights", None)
    return np.ones(len(fit.durations)) if weights is None else np.asarray(weights, dtype=float)


def pairwise_logrank(fits: Dict[str, object], adjust: bool = True) -> pd.DataFrame:
    """Log-rank test between every pair of groups, on their cells pooled.

    Returns a symmetric matrix of p-values, groups in rows and columns as in
    :func:`celldetective.utils.stats.test_2samp_generic`, NaN on the
    diagonal. With ``adjust``, the p-values are Holm-adjusted for the number
    of pairs.
    """
    from lifelines.statistics import logrank_test

    labels = [str(g) for g in fits]
    pivot = pd.DataFrame(np.nan, index=labels, columns=labels)
    pairs, p_values = [], []
    for (a, fa), (b, fb) in combinations(fits.items(), 2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # lifelines warns on fractional weights
            result = logrank_test(fa.durations, fb.durations, fa.event_observed, fb.event_observed,
                                  weights_A=_weights(fa), weights_B=_weights(fb))
        pairs.append((str(a), str(b)))
        p_values.append(result.p_value)
    if adjust:
        p_values = holm(p_values)
    for (a, b), p in zip(pairs, p_values):
        pivot.loc[a, b] = pivot.loc[b, a] = p
    return pivot


# ---------------------------------------------------------------------------
# Synchronised signals
# ---------------------------------------------------------------------------


def window_responses(
    matrix: np.ndarray,
    timeline: np.ndarray,
    baseline: Tuple[float, float],
    response: Tuple[float, float],
    min_fraction: float = 0.5,
) -> np.ndarray:
    """The response of each row of a matrix of aligned signals.

    Parameters
    ----------
    matrix : ndarray
        One row per cell, one column per time of ``timeline``, NaN where the
        cell is not observed (as returned by ``mean_signal(...,
        return_matrix=True)``).
    timeline : ndarray
        The time of each column, relative to the synchronising time.
    baseline, response : (float, float)
        The windows, both ends included, in the units of ``timeline``.
    min_fraction : float
        The least fraction of the frames of each window a cell must be
        observed in; otherwise its response is NaN.

    Returns
    -------
    ndarray
        Mean over the response window minus mean over the baseline window,
        one value per row.
    """
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    timeline = np.asarray(timeline, dtype=float)
    means = []
    for lo, hi in (baseline, response):
        cols = (timeline >= lo) & (timeline <= hi)
        if not cols.any():
            raise ValueError(f"The window [{lo}, {hi}] holds no time of the timeline.")
        values = matrix[:, cols]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(values, axis=1)
        mean[np.isfinite(values).sum(axis=1) < min_fraction * cols.sum()] = np.nan
        means.append(mean)
    return means[1] - means[0]
