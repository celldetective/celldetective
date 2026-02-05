"""
Unit tests for SettingsTracking GUI.

Tests the configuration of tracking parameters, ensuring proper UI interaction
and valid configuration generation for bTrack and trackpy.
"""

import pytest
import os
import json
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication

from celldetective.gui.settings._settings_tracking import SettingsTracking
from celldetective import get_software_location

# Test configuration
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = str(Path(TEST_DIR).parent)
EXPERIMENT_TEST_DIR = os.path.join(PARENT_DIR, "ExperimentTest")
INTERACTION_TIME = 100  # milliseconds


@pytest.fixture
def mock_process_panel(tmp_path):
    """
    Create a mock ProcessPanel and parent structure.
    SettingsTracking expects:
    - parent_window (ProcessPanel)
        - mode (str)
        - exp_dir (str)
        - parent_window (CelldetectiveMainWindow/App)
            - len_movie (int)
    """
    # Create valid temp experiment dir structure
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()
    (exp_dir / "configs").mkdir()

    # Create dummy movie file to avoid glob errors if necessary,
    # though SettingsTracking mainly uses it for Haralick checks which we might mock/skip

    # Parent Window (App)
    mock_app = MagicMock()
    mock_app.len_movie = 10

    # Process Panel
    mock_panel = MagicMock()
    mock_panel.parent_window = mock_app
    mock_panel.mode = "cells"  # or "targets"
    mock_panel.exp_dir = str(exp_dir) + os.sep

    # Create dummy config file that load_cell_config can read
    dummy_config = tmp_path / "dummy_config.json"
    with open(dummy_config, "w") as f:
        json.dump({"dummy": "config"}, f)

    with patch(
        "celldetective.gui.settings._settings_tracking.extract_experiment_channels",
        return_value=(["Channel 1", "Channel 2", "Channel 3"], [0, 1, 2]),
    ):
        with patch(
            "celldetective.gui.settings._settings_tracking.interpret_tracking_configuration",
            return_value=str(dummy_config),
        ):
            yield mock_panel


@pytest.fixture
def settings_tracking(qtbot, mock_process_panel, tmp_path):
    """Initialize SettingsTracking widget."""
    # Patch get_software_location to avoid loading real configs from system
    with patch(
        "celldetective.gui.settings._settings_tracking.get_software_location",
        return_value=str(tmp_path),
    ):
        # We also need to ensure the specific config paths in init use tmp_path
        # The widget constructs paths using self.parent_window.exp_dir
        # mock_process_panel already points exp_dir to tmp_path/experiment

        # However, _load_previous_instructions is called in __init__
        # If it finds a file (maybe from a previous run if not cleaned?), it might try to load it.
        # But tmp_path is unique per test function execution in pytest (mostly).

        widget = SettingsTracking(parent_window=mock_process_panel)
        qtbot.addWidget(widget)
        with qtbot.waitExposed(widget):
            widget.show()
        yield widget
        widget.close()


class TestSettingsTrackingInit:
    """Test initialization and basic UI state."""

    def test_init(self, settings_tracking):
        """Test widget initializes without errors."""
        assert settings_tracking.isVisible()
        assert settings_tracking.windowTitle() == "Configure tracking"
        assert settings_tracking.mode == "cells"

    def test_default_options(self, settings_tracking):
        """Test default selection (bTrack)."""
        assert settings_tracking.btrack_option.isChecked()
        assert not settings_tracking.trackpy_option.isChecked()
        assert settings_tracking.config_frame.isVisible()
        assert settings_tracking.features_frame.isVisible()
        assert settings_tracking.config_trackpy_frame.isHidden()


class TestTrackerSelection:
    """Test switching between bTrack and trackpy."""

    def test_switch_to_trackpy(self, settings_tracking, qtbot):
        """Test switching to trackpy hides bTrack specific frames."""
        # Switch to trackpy
        qtbot.mouseClick(settings_tracking.trackpy_option, QtCore.Qt.LeftButton)
        qtbot.wait(INTERACTION_TIME)

        assert settings_tracking.trackpy_option.isChecked()
        assert settings_tracking.config_frame.isHidden()
        assert settings_tracking.features_frame.isHidden()
        assert settings_tracking.config_trackpy_frame.isVisible()

    def test_switch_back_to_btrack(self, settings_tracking, qtbot):
        """Test switching back to bTrack restores UI."""
        # Switch to trackpy first
        qtbot.mouseClick(settings_tracking.trackpy_option, QtCore.Qt.LeftButton)
        qtbot.wait(INTERACTION_TIME)

        # Switch back to bTrack
        qtbot.mouseClick(settings_tracking.btrack_option, QtCore.Qt.LeftButton)
        qtbot.wait(INTERACTION_TIME)

        assert settings_tracking.btrack_option.isChecked()
        assert settings_tracking.config_frame.isVisible()
        assert settings_tracking.features_frame.isVisible()
        assert settings_tracking.config_trackpy_frame.isHidden()


class TestFeaturesFrame:
    """Test feature selection and Haralick options."""

    def test_add_remove_features(self, settings_tracking, qtbot):
        """Test adding and removing features from the list widget."""
        initial_count = settings_tracking.features_list.list_widget.count()

        # Add a feature
        qtbot.mouseClick(settings_tracking.add_feature_btn, QtCore.Qt.LeftButton)
        # Note: FeatureChoice dialog interaction is hard to test without mocking its execution
        # But we can simulate the list modification directly if needed, or check if the button works
        # For this test, we might just verify the button exists and is enabled
        assert settings_tracking.add_feature_btn.isEnabled()

        # Remove a feature (select first item then click remove)
        settings_tracking.features_list.list_widget.setCurrentRow(0)
        qtbot.mouseClick(settings_tracking.del_feature_btn, QtCore.Qt.LeftButton)

        assert settings_tracking.features_list.list_widget.count() == initial_count - 1

    def test_haralick_options_toggle(self, settings_tracking, qtbot):
        """Test toggling Haralick options visibility."""
        # Initial state: unchecked and hidden
        # Force uncheck first to be sure
        settings_tracking.activate_haralick_btn.setChecked(False)
        settings_tracking.show_haralick_options()  # Manual call to ensure state sync

        assert not settings_tracking.activate_haralick_btn.isChecked()
        for i, widget in enumerate(settings_tracking.haralick_to_hide):
            assert (
                widget.isHidden()
            ), f"Widget {i} ({type(widget)}) is not hidden initially"

        # Toggle on
        with qtbot.waitSignal(
            settings_tracking.activate_haralick_btn.toggled, timeout=1000
        ):
            settings_tracking.activate_haralick_btn.setChecked(True)

        assert settings_tracking.activate_haralick_btn.isChecked()

        # Check visibility
        for i, widget in enumerate(settings_tracking.haralick_to_hide):
            assert (
                widget.isVisible()
            ), f"Widget {i} ({type(widget)}) is not visible after toggle"

    def test_haralick_normalization_mode(self, settings_tracking, qtbot):
        """Test switching normalization mode (percentile vs absolute)."""
        # Ensure haralick options are active
        settings_tracking.activate_haralick_btn.setChecked(False)
        settings_tracking.show_haralick_options()

        with qtbot.waitSignal(
            settings_tracking.activate_haralick_btn.toggled, timeout=1000
        ):
            settings_tracking.activate_haralick_btn.setChecked(True)

        # Make sure controls are visible/active
        assert settings_tracking.haralick_normalization_mode_btn.isVisible()

        # Default is percentile mode
        assert settings_tracking.percentile_mode is True
        assert (
            settings_tracking.haralick_percentile_min_lbl.text() == "Min percentile: "
        )

        # Switch to absolute
        assert settings_tracking.haralick_normalization_mode_btn.isEnabled()
        qtbot.mouseClick(
            settings_tracking.haralick_normalization_mode_btn,
            QtCore.Qt.LeftButton,
            delay=100,
        )

        qtbot.wait(100)  # Wait for event loop processing

        assert settings_tracking.percentile_mode is False
        assert settings_tracking.haralick_percentile_min_lbl.text() == "Min value: "


class TestPostProcessing:
    """Test post-processing options."""

    def test_post_processing_toggle(self, settings_tracking, qtbot):
        """Test enabling/disabling post-processing options."""
        # Initially disabled/unchecked
        assert not settings_tracking.post_proc_ticked
        assert not settings_tracking.ContentsPostProc.isVisible()

        # Click the "select all" button (which acts as toggle for this frame)
        # Note: select_post_proc_btn connects to activate_post_proc_options
        qtbot.mouseClick(settings_tracking.select_post_proc_btn, QtCore.Qt.LeftButton)

        assert settings_tracking.post_proc_ticked
        # It enables elements, but doesn't necessarily show the frame if it was collapsed?
        # populate_post_proc_frame hides ContentsPostProc by default
        # activate_post_proc_options enables the widgets INSIDE, but doesn't change visibility of the container?
        # Let's check enabled state of inside widgets
        assert settings_tracking.min_tracklength_slider.isEnabled()

        # Toggle back off
        qtbot.mouseClick(settings_tracking.select_post_proc_btn, QtCore.Qt.LeftButton)
        assert not settings_tracking.post_proc_ticked
        assert not settings_tracking.min_tracklength_slider.isEnabled()


class TestSaveLoadConfig:
    """Test saving and loading configuration."""

    def test_write_instructions_btrack(self, settings_tracking, qtbot, tmp_path):
        """Test writing bTrack instructions."""
        # Setup specific values
        settings_tracking.config_le.setText('{"test": "config"}')
        settings_tracking.btrack_option.setChecked(True)

        # Haralick
        settings_tracking.activate_haralick_btn.setChecked(True)
        settings_tracking.haralick_distance_le.setText("5")

        # Post-proc
        settings_tracking.select_post_proc_btn.click()  # Enable
        settings_tracking.min_tracklength_slider.setValue(4)

        # Trigger write
        settings_tracking._write_instructions()

        # Verify files created
        instr_path = settings_tracking.track_instructions_write_path
        config_path = settings_tracking.config_path

        assert os.path.exists(instr_path)
        assert os.path.exists(config_path)

        # Verify content
        with open(instr_path, "r") as f:
            instr = json.load(f)

        assert instr["btrack_option"] is True
        assert instr["haralick_options"]["distance"] == 5
        assert instr["post_processing_options"]["minimum_tracklength"] == 4

        with open(config_path, "r") as f:
            config_content = f.read()
            assert '{"test": "config"}' in config_content

    def test_write_instructions_trackpy(self, settings_tracking, qtbot):
        """Test writing trackpy instructions."""
        # Switch to trackpy
        settings_tracking.trackpy_option.setChecked(True)
        settings_tracking.search_range_le.setText("15.5")
        settings_tracking.memory_slider.setValue(3)

        # Trigger write
        settings_tracking._write_instructions()

        # Verify content
        instr_path = settings_tracking.track_instructions_write_path
        with open(instr_path, "r") as f:
            instr = json.load(f)

        assert instr["btrack_option"] is False
        assert instr["search_range"] == 15.5
        assert instr["memory"] == 3

    def test_load_instructions(self, settings_tracking, qtbot):
        """Test loading instructions from file updates UI."""
        # Create dummy instruction file
        instr = {
            "btrack_option": False,
            "search_range": 20,
            "memory": 5,
            "features": ["area"],
            "mask_channels": ["Channel 1"],
            "haralick_options": {"distance": 2, "target_channel": 1},
            "post_processing_options": {"minimum_tracklength": 8},
        }

        with open(settings_tracking.track_instructions_write_path, "w") as f:
            json.dump(instr, f)

        # Call load
        settings_tracking._load_previous_instructions()

        # Verify UI updates
        assert settings_tracking.trackpy_option.isChecked()
        assert settings_tracking.search_range_le.text() == "20"
        assert settings_tracking.memory_slider.value() == 5

        # Verify features
        # (Assuming 'area' was added. list widget checking is complex, but we can check if not empty)
        assert settings_tracking.features_list.list_widget.count() > 0

        # Verify Haralick
        assert settings_tracking.activate_haralick_btn.isChecked()
        assert settings_tracking.haralick_distance_le.text() == "2"

        # Verify Post-proc
        assert settings_tracking.post_proc_ticked
        assert settings_tracking.min_tracklength_slider.value() == 8


class TestGeneratedConfigValidity:
    """Test that generated configs are valid for tracking execution."""

    def test_config_validity(self, settings_tracking):
        """Verify that the generated config dict has all keys required by tracking.py."""
        # Generate a standard config
        settings_tracking.btrack_option.setChecked(True)
        settings_tracking.features_ticked = True

        # Mocking items in feature list if empty
        if settings_tracking.features_list.list_widget.count() == 0:
            settings_tracking.features_list.addItem("area")

        settings_tracking.select_post_proc_btn.click()  # Enable post proc

        settings_tracking._write_instructions()

        instr_path = settings_tracking.track_instructions_write_path
        with open(instr_path, "r") as f:
            config = json.load(f)

        # Check keys against what 'track' function expects/uses
        # Based on celldetective.tracking.track signature and usage

        assert "btrack_option" in config
        assert "features" in config
        assert "mask_channels" in config
        assert "haralick_options" in config
        assert "post_processing_options" in config

        # Check post-processing options keys
        pp_opts = config["post_processing_options"]
        expected_pp_keys = [
            "minimum_tracklength",
            "remove_not_in_first",
            "remove_not_in_last",
            "interpolate_position_gaps",
            "extrapolate_tracks_pre",
            "extrapolate_tracks_post",
        ]
        for key in expected_pp_keys:
            assert key in pp_opts
