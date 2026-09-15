import math

import numpy as np
import pytest

from celldetective.gui.base.utils import safe_slider_range


@pytest.mark.parametrize(
    "lower, upper, expected",
    [
        (0.0, 10.0, (0.0, 10.0)),
        (5.0, 5.0, (5.0, 6.0)),
        (0, 0, (0.0, 1.0)),
        (8.0, 2.0, (8.0, 9.0)),
        (np.nan, np.nan, (0.0, 1.0)),
        (3.0, np.inf, (3.0, 4.0)),
        (np.float32(1.5), np.uint16(4), (1.5, 4.0)),
    ],
)
def test_safe_slider_range(lower, upper, expected):
    result = safe_slider_range(lower, upper)
    assert result == pytest.approx(expected)
    assert all(isinstance(v, float) and math.isfinite(v) for v in result)
    assert result[1] > result[0]
