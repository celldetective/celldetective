"""
Isotropic (position-based) measurements do not need tracks.

They were only computed on a tracked table: on a position measured before any
tracking, or on a table without ``TRACK_ID``, the radii were dropped with a log
line and the run reported success with none of their columns (issue #19). They
are now centred on the centroids of the masks when there is no table, and on the
positions of a static table otherwise.
"""

import json
import os
from queue import Queue

import numpy as np
import pandas as pd
import pytest
import tifffile

from celldetective.processes.measure_cells import MeasurementProcess

SIZE = 64
FRAMES = 2
# Two cells, each a disk of uniform intensity on a dark background.
CELLS = [((20, 20), 300.0), ((44, 40), 700.0)]
RADIUS = 6


def _disk(center):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= RADIUS**2


def _write_experiment(tmp_path, table=None):
    exp_dir = tmp_path / "exp"
    pos = exp_dir / "W1" / "100"
    for sub in ("movie", "labels_targets", "output/tables"):
        (pos / sub).mkdir(parents=True)
    (exp_dir / "configs").mkdir()

    frame = np.zeros((SIZE, SIZE), dtype=np.float32)
    labels = np.zeros((SIZE, SIZE), dtype=np.uint16)
    for i, (center, value) in enumerate(CELLS, start=1):
        frame[_disk(center)] = value
        labels[_disk(center)] = i
    stack = np.stack([frame] * FRAMES)[:, np.newaxis]
    tifffile.imwrite(
        pos / "movie" / "sample.tif", stack, imagej=True, metadata={"axes": "TCYX"}
    )
    for t in range(FRAMES):
        tifffile.imwrite(pos / "labels_targets" / f"{t:04d}.tif", labels)

    (exp_dir / "config.ini").write_text(
        f"[MovieSettings]\nmovie_prefix = sample\nlen_movie = {FRAMES}\n"
        f"shape_x = {SIZE}\nshape_y = {SIZE}\npxtoum = 1.0\nframetomin = 1.0\n"
        "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\n"
        "pharmaceutical_agents = none\n[Channels]\nbrightfield_channel = 0\n"
    )
    instructions = {
        "features": ["area", "intensity_mean"],
        "intensity_measurement_radii": [3, [8, 12]],
        "isotropic_operations": ["mean"],
        "clear_previous": False,
    }
    (exp_dir / "configs" / "measurement_instructions_targets.json").write_text(
        json.dumps(instructions)
    )
    if table is not None:
        table.to_csv(pos / "output" / "tables" / "trajectories_targets.csv", index=False)
    return str(pos) + os.sep


def _measure(pos):
    worker = MeasurementProcess(
        queue=Queue(), process_args={"mode": "targets", "n_threads": 1}
    )
    worker.setup_for_position(pos)
    worker.process_position()
    return pd.read_csv(pos + os.sep.join(["output", "tables", "trajectories_targets.csv"]))


def _check_isotropic_columns(df):
    assert len(df) == len(CELLS) * FRAMES
    inside = "brightfield_channel_circle_3_mean"
    ring = "brightfield_channel_ring_8_12_mean"
    assert inside in df.columns and ring in df.columns
    for (center, value) in CELLS:
        rows = df[
            (df["POSITION_Y"].round() == center[0])
            & (df["POSITION_X"].round() == center[1])
        ]
        assert len(rows) == FRAMES
        # Well inside the disk: the cell's own intensity; the ring lies outside
        # every disk: the background.
        np.testing.assert_allclose(rows[inside], value)
        np.testing.assert_allclose(rows[ring], 0.0)


def test_measured_before_tracking(tmp_path):
    _check_isotropic_columns(_measure(_write_experiment(tmp_path)))


def test_static_table_without_track_id(tmp_path):
    table = pd.DataFrame(
        [
            {"ID": i + t * len(CELLS), "FRAME": t, "class_id": i + 1,
             "POSITION_X": float(c[1]), "POSITION_Y": float(c[0])}
            for t in range(FRAMES)
            for i, (c, _) in enumerate(CELLS)
        ]
    )
    _check_isotropic_columns(_measure(_write_experiment(tmp_path, table=table)))
