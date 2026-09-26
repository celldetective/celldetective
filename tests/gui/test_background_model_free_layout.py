from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QListWidget

from celldetective.gui.layouts import BackgroundModelFreeCorrectionLayout


@pytest.fixture
def layout(qtbot):
    experiment = SimpleNamespace(
        exp_config=None, exp_channels=["brightfield", "nuclei"], len_movie=10, wells=["W1"]
    )
    parent = SimpleNamespace(
        parent_window=experiment, protocols=[], protocol_list=QListWidget()
    )
    return BackgroundModelFreeCorrectionLayout(parent)


@pytest.fixture
def warnings(monkeypatch):
    shown = []
    monkeypatch.setattr(
        "celldetective.gui.layouts.background_model_free_layout.generic_message",
        lambda *args, **kwargs: shown.append(args),
    )
    return shown


def test_protocol_and_preview_share_the_camera_offset(layout):
    layout.camera_offset_le.setText("100,5")
    layout.add_correction_btn.click()

    assert layout.parent_window.protocols[-1]["offset"] == pytest.approx(100.5)
    # The preview reads the same parameters, offset included.
    assert layout.correction_parameters()["offset"] == pytest.approx(100.5)


def test_invalid_offset_or_coefficient_number_is_rejected(layout, warnings):
    layout.camera_offset_le.setText("-")
    layout.add_correction_btn.click()
    assert layout.parent_window.protocols == []

    layout.camera_offset_le.setText("0")
    layout.regress_cb.setChecked(True)
    layout.nbr_coef_le.setText("")
    layout.add_correction_btn.click()
    assert layout.parent_window.protocols == []
    assert len(warnings) == 2


def test_fit_radius_goes_into_the_protocol(layout):
    assert not layout.radius_le.isEnabled()
    layout.regress_cb.setChecked(True)
    assert layout.radius_le.isEnabled()

    layout.radius_le.setText("250,5")
    layout.add_correction_btn.click()

    protocol = layout.parent_window.protocols[-1]
    assert protocol["optimize_option"]
    assert protocol["opt_radius"] == pytest.approx(250.5)


def test_empty_fit_radius_is_full_frame_and_zero_is_rejected(layout, monkeypatch):
    radius_warnings = []
    monkeypatch.setattr(
        "celldetective.gui.gui_utils.generic_message",
        lambda *args, **kwargs: radius_warnings.append(args),
    )
    layout.regress_cb.setChecked(True)
    assert layout.correction_parameters()["opt_radius"] is None

    layout.radius_le.setText("0")
    layout.add_correction_btn.click()
    assert layout.parent_window.protocols == []
    assert radius_warnings
