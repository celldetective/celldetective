"""Tests for celldetective.relative_measurements."""

import numpy as np
import pandas as pd


def test_neighbor_timeline_accepts_arrays_and_nan():
    """Neighbor lists come back as arrays from a pickle; NaN marks no neighborhood."""
    from celldetective.relative_measurements import _build_neighbor_timeline

    group = pd.DataFrame(
        {
            "FRAME": [0, 1, 2],
            "neigh": [
                np.nan,
                np.array([{"id": 1, "intersection": 4}, {"id": 2}], dtype=object),
                [{"id": 1, "intersection": 0}],
            ],
        }
    )
    ids, ids_per_t, intersections, first = _build_neighbor_timeline(group, "neigh")
    assert ids_per_t == [[], [1, 2], [1]]
    assert first == {1: 1, 2: 1}
    assert len(intersections) == 3
