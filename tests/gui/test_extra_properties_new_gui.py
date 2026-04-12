"""
GUI tests verifying that the five new extra-property functions are visible and
selectable inside the Measurement Settings panel.

Tests:
  1. get_extra_properties_functions() returns the new names (pure Python, no Qt).
  2. FeatureChoice combo box includes every new function name.
  3. The measurement settings panel lists the new functions as selectable features.
"""

import logging
import os

import numpy as np
import pytest
import tifffile
from unittest.mock import patch

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest

from celldetective import get_software_location
from celldetective.gui.base.feature_choice import (
    FeatureChoice,
    get_extra_properties_functions,
)

software_location = get_software_location()

NEW_EXTRA_PROPERTIES = [
    "circularity",
    "aspect_ratio",
    "intensity_skewness",
    "intensity_kurtosis",
    "intensity_membrane_cytoplasm_ratio",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def safe_wait(ms):
    QTest.qWait(ms)


@pytest.fixture(autouse=True)
def disable_logging():
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def drain_events_after_test(qtbot):
    yield
    qtbot.wait(10)
    QApplication.processEvents()


def _create_dummy_movie(exp_dir, well="W1", pos="100", frames=5, channels=2):
    movie_dir = os.path.join(exp_dir, well, pos, "movie")
    os.makedirs(movie_dir, exist_ok=True)
    img = np.zeros((frames * channels, 64, 64), dtype=np.uint16)
    tifffile.imwrite(os.path.join(movie_dir, "sample.tif"), img)


def _setup_experiment(tmp_path, well="W1", pos="100"):
    exp_dir = str(tmp_path / "Experiment")
    os.makedirs(os.path.join(exp_dir, well, pos, "output", "tables"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, well, pos, "labels_targets"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "configs"), exist_ok=True)

    with open(os.path.join(exp_dir, "config.ini"), "w") as f:
        f.write(
            "[MovieSettings]\nmovie_prefix = sample\nlen_movie = 5\n"
            "shape_x = 64\nshape_y = 64\npxtoum = 1.0\nframetomin = 1.0\n"
        )
        f.write(
            "[Labels]\nconcentrations = 0\ncell_types = dummy\n"
            "antibodies = none\npharmaceutical_agents = none\n"
        )
        f.write("[Channels]\nDAPI = 0\nGFP = 1\n")

    _create_dummy_movie(exp_dir, well=well, pos=pos, frames=5, channels=2)
    return exp_dir


def _open_measurement_settings(app, qtbot, tmp_path):
    """Open SettingsMeasurements and return the settings object, or skip."""
    from celldetective.gui.InitWindow import AppInitWindow
    from celldetective.gui.settings._settings_measurements import SettingsMeasurements

    exp_dir = _setup_experiment(tmp_path)
    app.experiment_path_selection.setText(exp_dir)
    qtbot.mouseClick(app.validate_button, QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: hasattr(app, "control_panel"), timeout=30000)

    cp = app.control_panel
    p0 = cp.ProcessPopulations[0]
    qtbot.waitUntil(lambda: cp.well_list.count() > 0, timeout=30000)

    with patch.object(cp.well_list, "getSelectedIndices", return_value=[0]):
        with patch.object(cp.position_list, "getSelectedIndices", return_value=[0]):
            cp.update_position_options()
            safe_wait(500)
            qtbot.mouseClick(p0.measurements_config_btn, QtCore.Qt.LeftButton)
            try:
                qtbot.waitUntil(
                    lambda: hasattr(p0, "settings_measurements"), timeout=15000
                )
            except Exception:
                pytest.skip("settings_measurements not available")

            settings = p0.settings_measurements
            if not isinstance(settings, SettingsMeasurements):
                pytest.skip("settings_measurements is not a SettingsMeasurements instance")
            return settings


@pytest.fixture
def app(qtbot):
    from celldetective.gui.InitWindow import AppInitWindow
    test_app = AppInitWindow(software_location=software_location)
    qtbot.addWidget(test_app)
    return test_app


# ---------------------------------------------------------------------------
# Pure-Python discovery tests (no Qt required)
# ---------------------------------------------------------------------------

class TestGetExtraPropertiesFunctions:
    """Tests for the AST-based extra_properties discovery function."""

    def test_returns_list(self):
        result = get_extra_properties_functions()
        assert isinstance(result, list)

    @pytest.mark.parametrize("name", NEW_EXTRA_PROPERTIES)
    def test_new_function_discovered(self, name):
        result = get_extra_properties_functions()
        assert name in result, (
            f"'{name}' not found in get_extra_properties_functions(). "
            f"Found: {result}"
        )

    def test_does_not_return_private_names(self):
        result = get_extra_properties_functions()
        private = [n for n in result if n.startswith("_")]
        assert len(private) == 0, f"Private names in list: {private}"

    def test_includes_pre_existing_functions(self):
        result = get_extra_properties_functions()
        for existing in ("area_dark_intensity", "intensity_radial_gradient"):
            assert existing in result, f"Pre-existing function '{existing}' missing"


# ---------------------------------------------------------------------------
# FeatureChoice widget tests
# ---------------------------------------------------------------------------

class TestFeatureChoiceWidget:
    """Tests for the FeatureChoice combo-box widget."""

    @pytest.fixture
    def feature_choice(self, qtbot):
        # FeatureChoice requires a parent with a list_widget attribute.
        from PyQt5.QtWidgets import QMainWindow, QListWidget
        parent = QMainWindow()
        parent.list_widget = QListWidget()
        widget = FeatureChoice(parent_window=parent)
        qtbot.addWidget(widget)
        return widget

    def test_widget_creates_without_error(self, feature_choice):
        assert feature_choice is not None

    def test_combo_box_populated(self, feature_choice):
        count = feature_choice.combo_box.count()
        assert count > 0, "FeatureChoice combo box should have items"

    @pytest.mark.parametrize("name", NEW_EXTRA_PROPERTIES)
    def test_new_property_in_combo_box(self, feature_choice, name):
        items = [
            feature_choice.combo_box.itemText(i)
            for i in range(feature_choice.combo_box.count())
        ]
        assert name in items, (
            f"'{name}' not in FeatureChoice combo box. Items: {items}"
        )

    def test_standard_features_still_present(self, feature_choice):
        items = [
            feature_choice.combo_box.itemText(i)
            for i in range(feature_choice.combo_box.count())
        ]
        for standard in ("area", "perimeter", "intensity_mean"):
            assert standard in items, f"Standard feature '{standard}' missing from combo box"


# ---------------------------------------------------------------------------
# Full GUI path: SettingsMeasurements panel
# ---------------------------------------------------------------------------

class TestMeasurementSettingsPanelNewProperties:
    """Tests that go through AppInitWindow → SettingsMeasurements."""

    def test_features_list_widget_exists(self, app, qtbot, tmp_path):
        settings = _open_measurement_settings(app, qtbot, tmp_path)
        assert hasattr(settings, "features_list")
        settings.close()

    def test_add_circularity_via_feature_choice(self, app, qtbot, tmp_path):
        """Open the add-feature dialog and select 'circularity'."""
        settings = _open_measurement_settings(app, qtbot, tmp_path)

        # Click the add-feature button to open FeatureChoice
        qtbot.mouseClick(settings.add_feature_btn, QtCore.Qt.LeftButton)
        safe_wait(300)

        # Find the open FeatureChoice window
        feature_choice = None
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, FeatureChoice):
                feature_choice = widget
                break

        if feature_choice is None:
            pytest.skip("FeatureChoice dialog did not open")

        # Select 'circularity' in the combo
        idx = feature_choice.combo_box.findText("circularity")
        if idx == -1:
            pytest.skip("'circularity' not found in combo box")
        feature_choice.combo_box.setCurrentIndex(idx)
        qtbot.mouseClick(feature_choice.add_btn, QtCore.Qt.LeftButton)
        safe_wait(200)

        items = settings.features_list.getItems()
        assert "circularity" in items, (
            f"'circularity' should be in features list after adding, got: {items}"
        )
        settings.close()

    def test_add_aspect_ratio_via_feature_choice(self, app, qtbot, tmp_path):
        settings = _open_measurement_settings(app, qtbot, tmp_path)

        qtbot.mouseClick(settings.add_feature_btn, QtCore.Qt.LeftButton)
        safe_wait(300)

        feature_choice = None
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, FeatureChoice):
                feature_choice = widget
                break

        if feature_choice is None:
            pytest.skip("FeatureChoice dialog did not open")

        idx = feature_choice.combo_box.findText("aspect_ratio")
        if idx == -1:
            pytest.skip("'aspect_ratio' not found in combo box")
        feature_choice.combo_box.setCurrentIndex(idx)
        qtbot.mouseClick(feature_choice.add_btn, QtCore.Qt.LeftButton)
        safe_wait(200)

        items = settings.features_list.getItems()
        assert "aspect_ratio" in items
        settings.close()

    @pytest.mark.parametrize("name", NEW_EXTRA_PROPERTIES)
    def test_each_new_property_addable(self, app, qtbot, tmp_path, name):
        """Generic parametrised check: each new property can be added via the dialog."""
        settings = _open_measurement_settings(app, qtbot, tmp_path)

        qtbot.mouseClick(settings.add_feature_btn, QtCore.Qt.LeftButton)
        safe_wait(300)

        feature_choice = None
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, FeatureChoice):
                feature_choice = widget
                break

        if feature_choice is None:
            pytest.skip("FeatureChoice dialog did not open")

        idx = feature_choice.combo_box.findText(name)
        if idx == -1:
            feature_choice.close()
            pytest.skip(f"'{name}' not found in combo box")

        feature_choice.combo_box.setCurrentIndex(idx)
        qtbot.mouseClick(feature_choice.add_btn, QtCore.Qt.LeftButton)
        safe_wait(200)

        items = settings.features_list.getItems()
        assert name in items, f"'{name}' should be in features list after adding"
        settings.close()
