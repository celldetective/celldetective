import pytest
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from celldetective.gui.InitWindow import AppInitWindow
from celldetective import get_software_location
import os
from unittest.mock import patch
import shutil
from pathlib import Path

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]])
print(abs_path)


@pytest.fixture
def app(qtbot):
    software_location = get_software_location()
    test_app = AppInitWindow(software_location=software_location)
    qtbot.addWidget(test_app)
    return test_app


def test_new_project(app, qtbot):
    # app.newExpAction.trigger()
    # qtbot.wait(1000)
    interaction_time = 500
    test_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = str(Path(test_directory).parent)

    # Patch QFileDialog.getExistingDirectory to return test_directory
    with patch(
        "PyQt5.QtWidgets.QFileDialog.getExistingDirectory",
        return_value=parent_directory,
    ):

        if os.path.exists(os.sep.join([parent_directory, "ExperimentTest"])):
            shutil.rmtree(os.sep.join([parent_directory, "ExperimentTest"]))

        app.new_exp_action.trigger()
        qtbot.wait(interaction_time * 3)

        app.new_exp_window.expName.setText("ExperimentTest")
        qtbot.wait(interaction_time)

        app.new_exp_window.SliderWells.setValue(10)
        qtbot.wait(interaction_time)

        app.new_exp_window.SliderPos.setValue(3)
        qtbot.wait(interaction_time)

        app.new_exp_window.MovieLengthSlider.setValue(1)
        qtbot.wait(interaction_time)

        app.new_exp_window.PxToUm_field.setText("0,3112")
        qtbot.wait(interaction_time)

        app.new_exp_window.shape_x_field.setText("660")
        qtbot.wait(interaction_time)

        app.new_exp_window.shape_y_field.setText("682")
        qtbot.wait(interaction_time)

        # set first channel
        app.new_exp_window.checkBoxes[0].setChecked(True)
        qtbot.wait(interaction_time)

        app.new_exp_window.sliders[0].setValue(0)
        qtbot.wait(interaction_time)

        # set second with channel index of 1
        app.new_exp_window.checkBoxes[1].setChecked(True)
        qtbot.wait(interaction_time)

        app.new_exp_window.sliders[1].setValue(3)
        qtbot.wait(interaction_time)

        app.new_exp_window.checkBoxes[2].setChecked(True)
        qtbot.wait(interaction_time)

        app.new_exp_window.sliders[2].setValue(1)
        qtbot.wait(interaction_time)

        # add extra custom channel
        qtbot.mouseClick(app.new_exp_window.addChannelBtn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        app.new_exp_window.name_le.setText("empty_channel")
        qtbot.wait(interaction_time)

        qtbot.mouseClick(app.new_exp_window.createBtn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        app.new_exp_window.checkBoxes[-1].setChecked(True)
        qtbot.wait(interaction_time)

        app.new_exp_window.sliders[-1].setValue(2)
        qtbot.wait(interaction_time)

        # Untick populations and create new one
        for box in app.new_exp_window.population_checkboxes:
            if box.text() == "targets":
                box.setChecked(True)
            else:
                box.setChecked(False)

        qtbot.mouseClick(app.new_exp_window.addPopBtn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        app.new_exp_window.name_le.setText("more_cells")
        qtbot.wait(interaction_time)

        qtbot.mouseClick(app.new_exp_window.addPopBtn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        qtbot.mouseClick(app.new_exp_window.validate_button, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        qtbot.mouseClick(app.new_exp_window.w.submit_btn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        shutil.copy(
            os.sep.join([parent_directory, "assets", "sample.tif"]),
            os.sep.join(
                [parent_directory, "ExperimentTest", "W1", "100", "movie", "sample.tif"]
            ),
        )
        qtbot.wait(interaction_time)

        qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
        qtbot.mouseClick(app.control_panel.view_stack_btn, QtCore.Qt.LeftButton)
        qtbot.wait(interaction_time)

        app.control_panel.viewer.channel_cb.setCurrentIndex(1)
        qtbot.wait(interaction_time * 2)

        app.control_panel.viewer.contrast_slider.setValue([200, 300])
        qtbot.wait(interaction_time * 2)

    # QApplication.closeAllWindows()
    # try:
    # 	shutil.rmtree(os.sep.join([parent_directory, "ExperimentTest"]))
    # except:
    # 	pass


# def test_lauch_app(app, qtbot):
# 	app.show()
# 	qtbot.wait(1000)
#
# def test_open_project(app, qtbot):
# 	app.experiment_path_selection.setText(abs_path + os.sep + 'examples/demo')
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
#
# def test_launch_demo(app, qtbot):
# 	app.experiment_path_selection.setText(abs_path + os.sep + 'examples/demo')
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
#
# def test_preprocessing_panel(app, qtbot):
#
# 	app.experiment_path_selection.setText(abs_path + os.sep + 'examples/demo')
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
#
# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.collapse_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.PreprocessingPanel.collapse_btn, QtCore.Qt.LeftButton)
#
# def test_app(app, qtbot):
#
# 	# Set an experiment folder and open
# 	app.experiment_path_selection.setText(os.sep.join([abs_path,'examples','demo']))
# 	qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
#
# 	# Set a position
# 	#app.control_panel.position_list.setCurrentIndex(0)
# 	#app.control_panel.update_position_options()
#
# 	# View stacl
# 	qtbot.mouseClick(app.control_panel.view_stack_btn, QtCore.Qt.LeftButton)
# 	#qtbot.wait(1000)
# 	app.control_panel.viewer.close()
#
# 	# Expand process block
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].collapse_btn, QtCore.Qt.LeftButton)
#
# 	# Use Threshold Config Wizard
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].upload_model_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].SegModelLoader.threshold_config_button, QtCore.Qt.LeftButton)
# 	app.control_panel.ProcessPopulations[0].SegModelLoader.ThreshWizard.close()
# 	app.control_panel.ProcessPopulations[0].SegModelLoader.close()
#
# 	# Check segmentation with napari
# 	#qtbot.mouseClick(app.control_panel.ProcessEffectors.check_seg_btn, QtCore.Qt.LeftButton)
# 	# close napari?
#
# 	# Train model
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].train_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].ConfigSegmentationTrain.close()
#
# 	# Config tracking
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].track_config_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].ConfigTracking.close()
#
# 	# Config measurements
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].measurements_config_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].ConfigMeasurements.close()
#
# 	# Classifier widget
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].classify_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].ClassifierWidget.close()
#
# 	# Config signal annotator
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].config_signal_annotator_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].ConfigSignalAnnotator.rgb_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].ConfigSignalAnnotator.close()
#
# 	# Signal annotator widget
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].check_signals_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].SignalAnnotator.close()
#
# 	# Table widget
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].view_tab_btn, QtCore.Qt.LeftButton)
# 	qtbot.wait(1000)
# 	app.control_panel.ProcessPopulations[0].tab_ui.close()
#
# 	#qtbot.mouseClick(app.control_panel.PreprocessingPanel.fit_correction_layout.add_correction_btn, QtCore.Qt.LeftButton)
# 	qtbot.mouseClick(app.control_panel.ProcessPopulations[0].collapse_btn, QtCore.Qt.LeftButton)
#
#
#
# def test_click(app, qtbot):
# 	qtbot.mouseClick(app.new_exp_button, QtCore.Qt.LeftButton)
# 	qtbot.wait(10000)
