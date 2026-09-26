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


def test_invalid_offset_is_rejected(layout, warnings):
    layout.camera_offset_le.setText("-")
    layout.add_correction_btn.click()
    assert layout.parent_window.protocols == []
    assert len(warnings) == 1


def test_background_qc_shows_the_background_as_applied(layout, monkeypatch):
    received = {}
    monkeypatch.setattr(
        "celldetective.preprocessing.estimate_background_per_condition",
        lambda experiment, **kwargs: received.update(kwargs) or [],
    )
    layout.camera_offset_le.setText("100,5")
    layout.interpolate_check.setChecked(True)
    monkeypatch.setattr(
        "celldetective.gui.layouts.background_model_free_layout.start_tracked",
        lambda worker: worker.run(),
    )
    layout.attr_parent.exp_dir = "experiment"

    layout.estimate_bg()

    assert received["offset"] == pytest.approx(100.5)
    assert received["fix_nan"] is True


def test_preview_corrects_a_few_frames_spread_over_the_movie(layout, monkeypatch):
    monkeypatch.setattr(
        "celldetective.utils.image_loaders.auto_load_number_of_frames", lambda path: 10
    )
    layout.attr_parent.current_stack = "movie.tif"

    indices = layout.preview_frame_indices()

    # IFDs of frames 0 to 9 of a two-channel movie.
    assert len(indices) == 5
    assert indices[0] == 0 and indices[-1] == 9 * 2
    assert all(i % 2 == 0 for i in indices)


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
