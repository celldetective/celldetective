import pytest
from PyQt5 import QtCore

import celldetective.gui.preprocessing_block
from celldetective.gui.InitWindow import AppInitWindow
from celldetective import get_software_location
import os

software_location = get_software_location()


@pytest.fixture
def app(qtbot):
    test_app = AppInitWindow(software_location=software_location)
    qtbot.addWidget(test_app)
    return test_app


def test_open_project(app, qtbot):
    app.experiment_path_selection.setText(software_location + os.sep + "examples/demo")
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
    qtbot.wait(10000)


def test_launch_demo(app, qtbot):
    app.experiment_path_selection.setText(software_location + os.sep + "examples/demo")
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)


def test_preprocessing_panel(app, qtbot):

    app.experiment_path_selection.setText(software_location + os.sep + "examples/demo")
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)

    qtbot.mouseClick(
        app.control_panel.PreprocessingPanel.collapse_btn,
        QtCore.Qt.LeftButton,
    )
    qtbot.mouseClick(
        app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn,
        QtCore.Qt.LeftButton,
    )
    qtbot.mouseClick(
        app.control_panel.PreprocessingPanel.collapse_btn,
        QtCore.Qt.LeftButton,
    )


def test_app(app, qtbot):

    # Set an experiment folder and open
    app.experiment_path_selection.setText(
        os.sep.join([software_location, "examples", "demo"])
    )
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)

    # Set a position
    # app.control_panel.position_list.setCurrentIndex(0)
    # app.control_panel.update_position_options()

    # View stacl
    qtbot.mouseClick(app.control_panel.view_stack_btn, QtCore.Qt.LeftButton)
    # qtbot.wait(1000)
    app.control_panel.viewer.close()

    # Expand process block
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].collapse_btn, QtCore.Qt.LeftButton
    )

    # Use Threshold Config Wizard
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].upload_model_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[
            0
        ].seg_model_loader.threshold_config_button,
        QtCore.Qt.LeftButton,
    )
    app.control_panel.ProcessPopulations[0].seg_model_loader.thresh_wizard.close()
    app.control_panel.ProcessPopulations[0].seg_model_loader.close()

    # Check segmentation with napari
    # qtbot.mouseClick(app.control_panel.ProcessEffectors.check_seg_btn, QtCore.Qt.LeftButton)
    # close napari?

    # Train model
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].train_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].settings_segmentation_training.close()

    # Config tracking
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].track_config_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].settings_tracking.close()

    # Config measurements
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].measurements_config_btn,
        QtCore.Qt.LeftButton,
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].settings_measurements.close()

    # Classifier widget
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].classify_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].classifier_widget.close()

    # Config signal annotator
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].config_signal_annotator_btn,
        QtCore.Qt.LeftButton,
    )
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].settings_signal_annotator.rgb_btn,
        QtCore.Qt.LeftButton,
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].settings_signal_annotator.close()

    # Signal annotator widget
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].check_signals_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].event_annotator.close()

    # Table widget
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].view_tab_btn, QtCore.Qt.LeftButton
    )
    qtbot.wait(1000)
    app.control_panel.ProcessPopulations[0].tab_ui.close()

    # qtbot.mouseClick(app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn, QtCore.Qt.LeftButton)
    qtbot.mouseClick(
        app.control_panel.ProcessPopulations[0].collapse_btn, QtCore.Qt.LeftButton
    )
