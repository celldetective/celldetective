"""
The tracking volume is given as (X, Y), the way bTrack takes it, while the
first-detection class reads an image shape as (Y, X): on a non-square movie,
cells were tested against the wrong edges.
"""

import pandas as pd

from celldetective import tracking


def test_first_detection_class_reads_the_volume_as_width_height():
    # A wide movie (X = 2048, Y = 512): one cell in the middle, one at the
    # right edge, both appearing after the first frame.
    objects = pd.DataFrame(
        {
            "t": [1, 2, 1, 2],
            "x": [1000.0, 1001.0, 2040.0, 2041.0],
            "y": [100.0, 101.0, 300.0, 301.0],
            "class_id": [1, 1, 2, 2],
        }
    )

    df = tracking.track(
        None,
        objects=objects,
        btrack_option=False,
        search_range=10,
        memory=0,
        volume=(2048, 512),
    )

    first = df.groupby("TRACK_ID").first()
    middle = first[first["POSITION_X"] < 1500]
    edge = first[first["POSITION_X"] > 1500]
    assert (middle["class_firstdetection"] == 0).all()
    assert (edge["class_firstdetection"] == 2).all()
