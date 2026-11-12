from itertools import combinations
from typing import Optional, Union, List, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta of ``a`` against ``b``: P(a > b) - P(a < b), ties counting as neither.

    The same value as ``cliffs_delta.cliffs_delta``, from a sorted search instead of a
    Python loop.
    """
    a, b = np.asarray(a, dtype=float), np.sort(np.asarray(b, dtype=float))
    smaller = np.searchsorted(b, a, side="left").sum()  # pairs with b < a
    larger = (len(b) - np.searchsorted(b, a, side="right")).sum()  # pairs with b > a
    return float((smaller - larger) / (len(a) * len(b)))


def test_2samp_generic(
    data: pd.DataFrame,
    feature: Optional[str] = None,
    groupby_cols: Optional[Union[str, List[str]]] = None,
    method: str = "ks_2samp",
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Performs pairwise statistical tests between groups of data, comparing a specified feature using a chosen method.

    The function applies two-sample statistical tests, such as the Kolmogorov-Smirnov (KS) test or Cliff's Delta,
    to compare distributions of a given feature across groups defined by `groupby_cols`. It returns the test results
    in a pivot table format with each group's pairwise comparison.

    Parameters
    ----------
    data : pandas.DataFrame
            The input dataset containing the feature to be tested.
    feature : str
            The name of the column representing the feature to compare between groups.
    groupby_cols : list or str
            The column(s) used to group the data. These columns define the groups that will be compared pairwise.
    method : str, optional, default="ks_2samp"
            The statistical test to use. Options:
            - "ks_2samp": Two-sample Kolmogorov-Smirnov test (default).
            - "cliffs_delta": Cliff's Delta for effect size between two distributions.
    *args, **kwargs :
            Additional arguments and keyword arguments for the selected test method.

    Returns
    -------
    pivot : pandas.DataFrame
            A pivot table containing the pairwise test results (p-values or effect sizes).
            The rows and columns represent the unique groups defined by `groupby_cols`,
            and the values represent the test result (e.g., p-values or effect sizes) between each group.

    Notes
    -----
    - The function compares all unique pairwise combinations of the groups based on `groupby_cols`.
    - For the "ks_2samp" method, the test compares the distributions using the Kolmogorov-Smirnov test.
    - For the "cliffs_delta" method, the function calculates the effect size between two distributions.
    - The results are returned in a pivot table where each cell compares the row group (``cdt1``) to the
      column group (``cdt2``). The table is not symmetric: the direction matters.
    - "ks_2samp" is one-sided (``alternative="less"``): a small p-value means the row group's CDF lies below
      the column group's, i.e. the row group has significantly larger values.
    - "cliffs_delta" gives P(row > column) - P(row < column): -1 when every row value is smaller than every
      column value, +1 when every row value is larger. Mirror cells have opposite signs.

    """

    if groupby_cols is None:
        raise ValueError("Please set a valid groupby_cols...")
    if feature is None:
        raise ValueError("Please set a feature to test...")

    if method not in ("ks_2samp", "cliffs_delta"):
        raise ValueError(f"Unknown method {method!r}.")

    cols = list(np.atleast_1d(groupby_cols))
    # Grouped by one column, a group is labelled by its value; by several, by a tuple.
    grouped = data.dropna(subset=[feature]).groupby(cols[0] if len(cols) == 1 else cols)[feature]
    labels = [str(lbl) for lbl, _ in grouped]
    samples = [group.to_numpy() for _, group in grouped]

    # The diagonal compares a group with itself: KS p-value 1, Cliff's delta 0.
    matrix = np.eye(len(samples)) if method == "ks_2samp" else np.zeros((len(samples), len(samples)))
    for i, j in combinations(range(len(samples)), 2):
        if method == "ks_2samp":
            # One-sided: the order of the pair matters, both cells are tests.
            for row, col in ((i, j), (j, i)):
                matrix[row, col] = ks_2samp(samples[row], samples[col], *args, alternative="less",
                                            method="auto", **kwargs).pvalue
        else:
            # Antisymmetric: the mirror cell is the opposite.
            matrix[i, j] = cliffs_delta(samples[i], samples[j])
            matrix[j, i] = -matrix[i, j]
    return pd.DataFrame(matrix, index=labels, columns=labels)
