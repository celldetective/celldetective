import pytest
import os
import pandas as pd
import numpy as np
import logging
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from celldetective.gui.InitWindow import AppInitWindow
from celldetective.gui.measure_annotator import MeasureAnnotator
from celldetective import get_software_location
from unittest.mock import patch
import shutil
import json
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


def test_measure_annotator_enhancements(app, qtbot, tmp_path):
    """
    Test that MeasureAnnotator correctly discovers group_* and status_* columns.
    Uses patching to bypass QCheckableComboBox headless interaction issues.
    """
    exp_dir = str(tmp_path / "Experiment")
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

    # Use numeric class/status to avoid color_from_state failure
    df = pd.DataFrame(
        {
            "TRACK_ID": [1, 1, 2, 2],
            "FRAME": [0, 1, 0, 1],
            "group_experimental": ["A", "A", "B", "B"],
            "class_firstdetection": [0, 0, 1, 1],
            "area": [100.0, 110.0, 105.0, 115.0],  # Needed for MinMaxScaler
            "POSITION_X": [10, 12, 10, 12],
            "POSITION_Y": [10, 12, 10, 12],
        }
    )
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
            try:
                cols = annotator.class_cols
            except RuntimeError:
                pytest.fail("MeasureAnnotator closed unexpectedly (RuntimeError).")

            assert "group_experimental" in cols
            assert "status_firstdetection" in cols

            annotator.close()


def test_event_mapping_anticipation(app, qtbot, tmp_path):
    """
    Test that SignalModelParamsWidget anticipates signals based on metadata.
    """
    exp_dir = str(tmp_path / "ExperimentAnticipation")
    os.makedirs(os.path.join(exp_dir, "W1", "100"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)

    with open(os.path.join(exp_dir, "config.ini"), "w") as f:
        f.write(
            "[MovieSettings]\nmovie_prefix = sample\nlen_movie = 10\nshape_x = 100\nshape_y = 100\npxtoum = 1.0\nframetomin = 1.0\n"
        )
        f.write(
            "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\npharmaceutical_agents = none\n"
        )
        f.write("[Channels]\nDAPI = 0\nGFP = 1\n")

    # Create measurement instructions for anticipation
    instructions = {
        "features": [],
        "intensity_measurement_radii": [5],
        "isotropic_operations": ["mean"],
        "border_distances": [10],
    }
    # Create for both targets and effectors to be safe
    for pop in ["targets", "effectors"]:
        with open(
            os.path.join(exp_dir, "configs", f"measurement_instructions_{pop}.json"),
            "w",
        ) as f:
            json.dump(instructions, f)

    create_dummy_movie(exp_dir, well="W1", pos="100", prefix="sample", frames=10)

    dummy_model_name = "DummyModelTest"
    models_dir = os.path.join(
        software_location, "celldetective", "models", "signal_detection"
    )
    model_path = os.path.join(models_dir, dummy_model_name)
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, "config_input.json"), "w") as f:
        json.dump({"channels": ["DAPI", "GFP"]}, f)

    try:
        app.experiment_path_selection.setText(exp_dir)
        qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
        qtbot.waitUntil(lambda: hasattr(app, "control_panel"), timeout=30000)

        p0 = app.control_panel.ProcessPopulations[0]

        qtbot.waitUntil(lambda: app.control_panel.well_list.count() > 0, timeout=30000)

        with patch.object(
            app.control_panel.well_list, "getSelectedIndices", return_value=[0]
        ):
            with patch.object(
                app.control_panel.position_list, "getSelectedIndices", return_value=[0]
            ):
                app.control_panel.update_position_options()
                qtbot.wait(500)

                p0.signal_analysis_action.setChecked(True)
                p0.refresh_signal_models()
                idx = p0.signal_models_list.findText(dummy_model_name)
                if idx >= 0:
                    p0.signal_models_list.setCurrentIndex(idx)

                qtbot.mouseClick(p0.submit_btn, QtCore.Qt.LeftButton)

                qtbot.waitUntil(
                    lambda: hasattr(p0, "signalChannelWidget"), timeout=30000
                )
                widget = p0.signalChannelWidget
                assert widget is not None

                qtbot.wait(1000)
                items = [
                    widget.channel_cbs[0].itemText(i)
                    for i in range(widget.channel_cbs[0].count())
                ]

                assert "area" in items
                assert "dapi_mean" in items or "DAPI_mean" in items
                assert "gfp_mean" in items or "GFP_mean" in items

                # Check for anticipated measurements from instructions
                # DAPI_circle_5_mean, DAPI_mean_edge_10px
                anticipated = [
                    "DAPI_circle_5_mean",
                    "GFP_circle_5_mean",
                    "DAPI_mean_edge_10px",
                    "GFP_mean_edge_10px",
                ]
                for ant in anticipated:
                    found = any(ant.lower() == item.lower() for item in items)
                    assert found, f"Anticipated measurement {ant} not found in {items}"

                widget.close()

    finally:
        if os.path.exists(model_path):
            try:
                shutil.rmtree(model_path)
            except:
                pass


def test_fix_missing_labels(tmp_path):
    """
    Test that fix_missing_labels creates empty label files when they don't exist.
    """
    from celldetective.utils.image_loaders import fix_missing_labels

    exp_dir = str(tmp_path / "ExperimentFixLabels")
    # Don't create labels_effectors here, let fix_missing_labels do it or ensure it handles missing dirs
    # Actually fix_missing_labels expects the directory to exist if it writes to it?
    # Let's check logic: path = position + os.sep + f"labels_{population}" -> save_tiff...
    # It does NOT appear to create the directory involved in 'path'.
    # But usually creating a dummy movie creates 'movie' dir.
    # We should create the parent 'labels_effectors' dir to be safe, or see if it fails.
    # My previous fix for test_measure_annotator_enhancements added `os.makedirs(..., "labels_effectors")`
    # Check fix_missing_labels logic again?
    # It constructs path... save_tiff_imagej_compatible(os.sep.join([path, file]))
    # save_tiff_imagej_compatible calls imsave -> tifffile.imwrite.
    # If dir doesn't exist, it might fail.
    # But let's follow the standard pattern: "labels" usually pre-exist if segmentation started.
    # Here we simulate "missing files" inside that directory.

    well, pos = "W1", "100"
    os.makedirs(os.path.join(exp_dir, well, pos, "labels_effectors"), exist_ok=True)
    create_dummy_movie(exp_dir, well=well, pos=pos, prefix="sample", frames=5)

    # Verify no labels yet
    lbl_dir = os.path.join(exp_dir, well, pos, "labels_effectors")
    movie_dir = os.path.join(exp_dir, well, pos, "movie")
    assert len(os.listdir(lbl_dir)) == 0

    from celldetective.utils.image_loaders import locate_stack

    stack = locate_stack(os.path.join(exp_dir, well, pos), prefix="sample")

    # Call fix
    fix_missing_labels(
        os.path.join(exp_dir, well, pos), population="effectors", prefix="sample"
    )

    # Verify 5 label files created
    files = os.listdir(lbl_dir)
    assert len(files) == 5
    assert "0000.tif" in files
    assert "0004.tif" in files

    # Verify content is empty (zeros)
    img = tifffile.imread(os.path.join(lbl_dir, "0000.tif"))
    assert np.all(img == 0)
    assert img.shape == (100, 100)


def test_table_exploration_logic(app, qtbot, tmp_path):
    """
    Test the logic for single vs multi-position exploration.
    """
    exp_dir = str(tmp_path / "ExperimentMulti")
    for pos in ["100", "101"]:
        os.makedirs(os.path.join(exp_dir, "W1", pos, "output", "tables"), exist_ok=True)
        create_dummy_movie(exp_dir, well="W1", pos=pos, prefix="sample", frames=10)

        df = pd.DataFrame(
            {
                "TRACK_ID": [1],
                "FRAME": [0],
                "area": [100.0],  # Needed for MinMaxScaler
                "POSITION_X": [10],
                "POSITION_Y": [10],
            }
        )
        traj_path = os.path.join(
            exp_dir, "W1", pos, "output", "tables", "trajectories_effectors.csv"
        )
        df.to_csv(traj_path, index=False)

    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)
    with open(os.path.join(exp_dir, "config.ini"), "w") as f:
        f.write(
            "[MovieSettings]\nmovie_prefix = sample\nlen_movie = 10\nshape_x = 100\nshape_y = 100\npxtoum = 1.0\nframetomin = 1.0\n"
        )
        f.write(
            "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\npharmaceutical_agents = none\n[Channels]\nChannel1 = 0\n"
        )

    app.experiment_path_selection.setText(exp_dir)
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: hasattr(app, "control_panel"), timeout=30000)

    cp = app.control_panel
    p0 = cp.ProcessPopulations[0]

    qtbot.waitUntil(lambda: cp.well_list.count() > 0, timeout=30000)

    # 2. Test Single Selection behavior (default)
    with patch.object(cp.well_list, "getSelectedIndices", return_value=[0]):
        with patch.object(cp.position_list, "getSelectedIndices", return_value=[0]):
            cp.update_position_options()
            qtbot.wait(500)

            with patch.object(MeasureAnnotator, "show") as mock_show:
                qtbot.mouseClick(p0.check_measurements_btn, QtCore.Qt.LeftButton)
                try:
                    qtbot.waitUntil(
                        lambda: hasattr(p0, "measure_annotator"), timeout=15000
                    )
                except:
                    print("DEBUG: Single selection check failed.")
                    raise
                assert p0.measure_annotator is not None
                p0.measure_annotator.close()
                del p0.measure_annotator

    # 3. Test Multi Selection behavior
    with patch.object(cp.well_list, "getSelectedIndices", return_value=[0]):
        with patch.object(cp.position_list, "getSelectedIndices", return_value=[0, 1]):
            cp.update_position_options()
            qtbot.wait(500)

            assert p0.view_tab_btn.isEnabled()
            assert p0.check_measurements_btn.isEnabled()

            with patch.object(p0, "view_table_ui") as mock_view_tab:
                qtbot.mouseClick(p0.check_measurements_btn, QtCore.Qt.LeftButton)
                qtbot.wait(200)
                mock_view_tab.assert_called_once()
