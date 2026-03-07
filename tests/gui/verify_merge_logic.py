import pandas as pd
import numpy as np


def compute_merged_classification(
    df: pd.DataFrame, cols_to_merge: list[str]
) -> pd.Series:
    """Simulate EXACT logic of MergeGroupWidget.compute() from celldetective/gui/table_ops/_merge_groups.py"""

    # Logic extracted from codebase:
    # 1. Compute bases
    bases = [int(df[c].max()) + 1 for c in cols_to_merge]

    # 2. Compute multipliers
    multipliers = np.concatenate(([1], np.cumprod(bases[:-1])))
    print(f"Bases: {bases}, Multipliers: {multipliers}")

    # 3. Compute merged sum
    merged_col = (df[cols_to_merge] * multipliers).sum(axis=1)

    # 4. Propagate NaNs
    merged_col.loc[df[cols_to_merge].isna().any(axis=1)] = np.nan

    return merged_col


def run_verification():
    print("=== Verifying Merge Classification Logic ===")

    # Case 1: Binary Merge (Spread + Dead)
    # spread: base 2 (0,1) -> multiplier 1
    # dead: base 2 (0,1) -> multiplier 2
    df1 = pd.DataFrame({"spread": [0, 1, 0, 1], "dead": [0, 0, 1, 1]})
    res1 = compute_merged_classification(df1, ["spread", "dead"])
    assert res1.tolist() == [0.0, 1.0, 2.0, 3.0]
    print("[OK] Binary merge correctness verified")

    # Case 2: Multi-label Merge (Type + Size)
    # type: base 3 (0,1,2) -> multiplier 1
    # size: base 2 (0,1)   -> multiplier 3
    df2 = pd.DataFrame({"type": [0, 1, 2, 0, 1, 2], "size": [0, 0, 0, 1, 1, 1]})
    res2 = compute_merged_classification(df2, ["type", "size"])
    # 0*1+0*3=0, 1*1+0*3=1, 2*1+0*3=2, 0*1+1*3=3, 1*1+1*3=4, 2*1+1*3=5
    assert res2.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print("[OK] Multi-label merge correctness verified")

    # Case 3: NaN Handling
    df3 = pd.DataFrame({"A": [1, np.nan, 1], "B": [1, 1, np.nan]})
    res3 = compute_merged_classification(df3, ["A", "B"])
    assert res3.iloc[0] == 3.0  # 1*1 + 1*2 = 3
    assert np.isnan(res3.iloc[1])
    assert np.isnan(res3.iloc[2])
    print("[OK] NaN propagation verified")

    print("\nSUCCESS: All logic tests passed.")


if __name__ == "__main__":
    run_verification()
