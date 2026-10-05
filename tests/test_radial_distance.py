"""
The radial distance is measured to the centre of an image whose shape is (Y, X):
on a non-square movie, X was centred on the height and Y on the width.
"""

import pandas as pd
import pytest

from celldetective.measure import measure_radial_distance_to_center


def test_radial_distance_centres_a_wide_image():
    # A wide image (Y = 512, X = 2048): its centre is at x = 1024, y = 256.
    df = pd.DataFrame({"POSITION_X": [1024.0, 1024.0], "POSITION_Y": [256.0, 0.0]})

    df = measure_radial_distance_to_center(
        df,
        volume=(512, 2048),
        column_labels={"x": "POSITION_X", "y": "POSITION_Y"},
    )

    assert df["radial_distance"].tolist() == pytest.approx([0.0, 256.0])
