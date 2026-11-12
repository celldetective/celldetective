"""Tests of celldetective.utils.stats: the pairwise tests of the table explorer."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from celldetective.utils import stats
from celldetective.utils.stats import cliffs_delta

# Imported through the module: pytest would collect a function named test_*.
pairwise = stats.test_2samp_generic


def _table(rng):
    return pd.DataFrame(
        dict(
            well=np.repeat(["W1", "W2", "W3"], [30, 40, 25]),
            dose=np.repeat([10, 2, 2], [30, 40, 25]),
            value=np.concatenate([rng.integers(0, 6, 30), rng.integers(1, 8, 40), rng.normal(3, 1, 25)]).astype(float),
        )
    )


def test_cliffs_delta_matches_the_package_with_ties():
    package = pytest.importorskip("cliffs_delta")
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = rng.integers(0, 5, rng.integers(1, 30)), rng.integers(0, 5, rng.integers(1, 30))
        assert cliffs_delta(a, b) == pytest.approx(package.cliffs_delta(list(a), list(b))[0])


def test_cliffs_delta_extremes():
    assert cliffs_delta([5, 6], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [5, 6]) == -1.0
    assert cliffs_delta([1, 1], [1, 1]) == 0.0


def test_ks_matrix_is_one_sided_ks_for_every_ordered_pair():
    data = _table(np.random.default_rng(1))
    pivot = pairwise(data, feature="value", groupby_cols=["well"], method="ks_2samp")
    assert list(pivot.index) == ["W1", "W2", "W3"]  # values, not 1-tuples
    groups = {w: g["value"].to_numpy() for w, g in data.groupby("well")}
    for a in groups:
        for b in groups:
            expected = ks_2samp(groups[a], groups[b], alternative="less").pvalue
            assert pivot.loc[a, b] == pytest.approx(expected)


def test_cliffs_matrix_is_antisymmetric():
    pivot = pairwise(_table(np.random.default_rng(2)), feature="value", groupby_cols="well",
                               method="cliffs_delta")
    m = pivot.to_numpy()
    assert np.allclose(m, -m.T)
    assert np.allclose(np.diag(m), 0)


def test_missing_values_are_left_out():
    data = _table(np.random.default_rng(3))
    data.loc[:9, "value"] = np.nan
    full = data.dropna()
    pivot = pairwise(data, feature="value", groupby_cols=["well"], method="cliffs_delta")
    expected = cliffs_delta(full.loc[full.well == "W1", "value"], full.loc[full.well == "W2", "value"])
    assert pivot.loc["W1", "W2"] == pytest.approx(expected)


def test_several_grouping_columns_label_by_tuple():
    pivot = pairwise(_table(np.random.default_rng(4)), feature="value", groupby_cols=["dose", "well"],
                               method="cliffs_delta")
    assert len(pivot) == 3 and pivot.index[0].startswith("(")


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        pairwise(_table(np.random.default_rng(5)), feature="value", groupby_cols="well", method="t")
