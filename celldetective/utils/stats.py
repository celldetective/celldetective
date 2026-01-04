from typing import Optional, Union, List

import pandas as pd
from cliffs_delta import cliffs_delta
from scipy.stats import ks_2samp


def test_2samp_generic(
    data: pd.DataFrame,
    feature: Optional[str] = None,
    groupby_cols: Optional[Union[str, List[str]]] = None,
    method="ks_2samp",
    *args,
    **kwargs,
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
    - The results are returned in a symmetric pivot table where each cell represents the test result for the corresponding group pair.

    """

    assert groupby_cols is not None, "Please set a valid groupby_cols..."
    assert feature is not None, "Please set a feature to test..."

    results = []

    for lbl1, group1 in data.dropna(subset=feature).groupby(groupby_cols):
        for lbl2, group2 in data.dropna(subset=feature).groupby(groupby_cols):

            dist1 = group1[feature].values
            dist2 = group2[feature].values
            if method == "ks_2samp":
                test = ks_2samp(
                    list(dist1),
                    list(dist2),
                    alternative="less",
                    mode="auto",
                    *args,
                    **kwargs,
                )
                val = test.pvalue
            elif method == "cliffs_delta":
                test = cliffs_delta(list(dist1), list(dist2), *args, **kwargs)
                val = test[0]

            results.append({"cdt1": lbl1, "cdt2": lbl2, "value": val})

    results = pd.DataFrame(results)
    results["cdt1"] = results["cdt1"].astype(str)
    results["cdt2"] = results["cdt2"].astype(str)

    pivot = results.pivot(index="cdt1", columns="cdt2", values="value")
    pivot.reset_index(inplace=True)
    pivot.columns.name = None
    pivot.set_index("cdt1", drop=True, inplace=True)
    pivot.index.name = None

    return pivot
