"""Statistical comparison of survival curves and synchronised signals.

Groups (usually wells) are compared on their cells pooled. Within one
experiment that answers "do these wells differ?": with one well per
condition, a difference may come from the well rather than its condition,
which only repeating the experiment can tell.

- Survival curves: pairwise log-rank tests.
- Synchronised signals: the response of each cell, the mean over a window
  after the synchronising time minus the mean over a baseline before it
  (:func:`signal_responses`). The responses are values per cell, compared like
  any measurement with :func:`celldetective.utils.stats.test_2samp_generic`
  (:func:`compare_responses`).

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
    "compare_responses",
    "holm",
    "pairwise_logrank",
    "signal_responses",
    "window_responses",
]

# The tests between groups of responses, as in the table explorer.
RESPONSE_TESTS = ("ks_2samp", "cliffs_delta")


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


def pairwise_logrank(fits: Dict[str, object], adjust: bool = True) -> pd.DataFrame:
    """Log-rank test between every pair of groups, on their cells pooled.

    Returns a symmetric matrix of p-values, groups in rows and columns as in
    :func:`celldetective.utils.stats.test_2samp_generic`, NaN on the
    diagonal. With ``adjust``, the p-values are Holm-adjusted for the number
    of pairs.
    """
    from lifelines.statistics import logrank_test

    fitted = list(fits.values())
    pairs = list(combinations(range(len(fitted)), 2))
    p_values = [
        logrank_test(fitted[i].durations, fitted[j].durations,
                     fitted[i].event_observed, fitted[j].event_observed).p_value
        for i, j in pairs
    ]
    if adjust:
        p_values = holm(p_values)
    matrix = np.full((len(fitted), len(fitted)), np.nan)
    for (i, j), p in zip(pairs, p_values):
        matrix[i, j] = matrix[j, i] = p
    labels = [str(g) for g in fits]
    return pd.DataFrame(matrix, index=labels, columns=labels)


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


def signal_responses(
    positions: pd.DataFrame,
    key: str,
    baseline: Tuple[float, float],
    response: Tuple[float, float],
    min_fraction: float = 0.5,
) -> pd.DataFrame:
    """The response of each cell of some positions.

    Parameters
    ----------
    positions : DataFrame
        One row per position, with ``well_name``, ``pos_name`` and ``signal``:
        the dictionary of synchronised signals of the position, holding
        ``timeline`` and the matrix ``key``. Rows without one are skipped.
    key : str
        The matrix to read, e.g. ``"matrix_event"``.
    baseline, response : (float, float)
        The windows, in frames relative to the synchronising time, both ends
        included.
    min_fraction : float
        See :func:`window_responses`.

    Returns
    -------
    DataFrame
        One row per cell observed enough over both windows: ``well_name``,
        ``pos_name`` and ``response``.

    Notes
    -----
    The windows only mean something if the matrix is aligned on an event:
    ``mean_signal`` aligns cells without the event on the start of the movie,
    unless all cells share an absolute time.
    """
    tables = []
    for _, row in positions.iterrows():
        signal = row["signal"]
        if not isinstance(signal, dict):
            continue
        values = window_responses(signal[key], signal["timeline"], baseline, response, min_fraction)
        values = values[np.isfinite(values)]  # cells not observed enough, and empty rows
        tables.append(pd.DataFrame(dict(well_name=row["well_name"], pos_name=row["pos_name"], response=values)))
    if not tables:
        return pd.DataFrame(columns=["well_name", "pos_name", "response"])
    return pd.concat(tables, ignore_index=True)


def compare_responses(cells: pd.DataFrame, by: str = "well_name") -> Dict[str, pd.DataFrame]:
    """Test the responses between every pair of groups, cells pooled.

    Parameters
    ----------
    cells : DataFrame
        One row per cell, as :func:`signal_responses` returns.
    by : str
        The column defining the groups.

    Returns
    -------
    dict
        ``"ks_2samp"``: the one-sided KS p-values, ``"cliffs_delta"``: the
        effect sizes, each a matrix with the groups in rows and columns, read
        as in :func:`celldetective.utils.stats.test_2samp_generic`. The
        p-values are not adjusted for the number of pairs.
    """
    from celldetective.utils.stats import test_2samp_generic

    return {
        method: test_2samp_generic(cells, feature="response", groupby_cols=[by], method=method)
        for method in RESPONSE_TESTS
    }
