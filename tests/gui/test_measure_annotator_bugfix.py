import pytest
import os
import pandas as pd
import numpy as np
import logging
from PyQt5 import QtCore
from celldetective.gui.InitWindow import AppInitWindow
from celldetective.gui.measure_annotator import MeasureAnnotator
from celldetective import get_software_location
from unittest.mock import patch
import shutil
import tifffile

software_location = get_software_location()


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logger = logging.getLogger()
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def app(qtbot):
    test_app = AppInitWindow(software_location=software_location)
    qtbot.addWidget(test_app)
    return test_app


def create_dummy_movie(exp_dir, well="W1", pos="100", prefix="sample", frames=5):
    movie_dir = os.path.join(exp_dir, well, pos, "movie")
    os.makedirs(movie_dir, exist_ok=True)
    # Use a single multi-page TIF as expected by locate_stack
    movie_path = os.path.join(movie_dir, f"{prefix}.tif")
    img = np.zeros((frames, 100, 100), dtype=np.uint16)
    tifffile.imwrite(movie_path, img)


def test_measure_annotator_colors_writable(app, qtbot, tmp_path):
    """
    Test that self.colors in MeasureAnnotator contains writable arrays.
    This verifies the fix for 'ValueError: assignment destination is read-only'.
    """
    exp_dir = str(tmp_path / "ExperimentColors")
    os.makedirs(os.path.join(exp_dir, "W1", "100", "output", "tables"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)

    with open(os.path.join(exp_dir, "config.ini"), "w") as f:
        f.write(
            "[MovieSettings]\nmovie_prefix = sample\nlen_movie = 10\nshape_x = 100\nshape_y = 100\npxtoum = 1.0\nframetomin = 1.0\n"
        )
        f.write(
            "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\npharmaceutical_agents = none\n[Channels]\nChannel1 = 0\n"
        )

    create_dummy_movie(exp_dir, well="W1", pos="100", prefix="sample", frames=10)

    # DataFrame with tracks
    df = pd.DataFrame(
        {
            "TRACK_ID": [1, 1],
            "FRAME": [0, 1],
            "group_experimental": ["A", "A"],
            "area": [100.0, 110.0],
            "POSITION_X": [10, 12],
            "POSITION_Y": [10, 12],
            "status": [0, 0],  # Ensure status column exists
        }
    )
    # The 'group_color' column is usually generated inside MeasureAnnotator,
    # but let's see if we need to let it generate it.
    # MeasureAnnotator calls 'color_from_state', then assigns 'group_color'.

    traj_path = os.path.join(
        exp_dir, "W1", "100", "output", "tables", "trajectories_effectors.csv"
    )
    df.to_csv(traj_path, index=False)

    app.experiment_path_selection.setText(exp_dir)
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: hasattr(app, "control_panel"), timeout=30000)

    cp = app.control_panel
    p0 = cp.ProcessPopulations[0]

    qtbot.waitUntil(lambda: cp.well_list.count() > 0, timeout=30000)

    with patch.object(cp.well_list, "getSelectedIndices", return_value=[0]):
        with patch.object(cp.position_list, "getSelectedIndices", return_value=[0]):

            cp.update_position_options()
            qtbot.wait(500)

            qtbot.mouseClick(p0.check_measurements_btn, QtCore.Qt.LeftButton)

            try:
                qtbot.waitUntil(lambda: hasattr(p0, "measure_annotator"), timeout=15000)
            except Exception:
                print("DEBUG: measure_annotator not found on p0.")
                raise

            annotator = p0.measure_annotator
            qtbot.wait(1000)
            assert annotator is not None

            # Verify that self.colors arrays are writable
            # extract_scatter_from_trajectories should have been called during init
            assert hasattr(annotator, "colors")
            assert len(annotator.colors) > 0

            # Check the first frame's colors
            colors_frame_0 = annotator.colors[0]

            # Check flags
            assert colors_frame_0.flags[
                "WRITEABLE"
            ], "self.colors arrays must be writable"

            # Try to modify (should not raise ValueError)
            try:
                colors_frame_0[0] = "lime"
            except ValueError as e:
                pytest.fail(f"Could not modify colors array: {e}")

            annotator.close()
