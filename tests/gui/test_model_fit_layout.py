from types import SimpleNamespace

import pytest

from celldetective.gui.layouts.local_correction_layout import LocalCorrectionLayout
from celldetective.gui.layouts.model_fit_layout import BackgroundFitCorrectionLayout


@pytest.fixture
def layout(qtbot):
    experiment = SimpleNamespace(
        locate_image=lambda: None, exp_channels=["brightfield", "nuclei"]
    )
    return BackgroundFitCorrectionLayout(SimpleNamespace(parent_window=experiment))


def test_offset_goes_into_the_protocol(layout):
    assert layout.generate_instructions()
    assert layout.instructions["offset"] == 0

    layout.camera_offset_le.setText("100,5")
    assert layout.generate_instructions()
    assert layout.instructions["offset"] == pytest.approx(100.5)


def test_empty_offset_means_none(layout):
    layout.camera_offset_le.setText("")
    assert layout.generate_instructions()
    assert layout.instructions["offset"] is None


def test_invalid_offset_is_rejected(layout, warnings):
    layout.camera_offset_le.setText("-")

    assert not layout.generate_instructions()
    assert len(warnings) == 1


def test_empty_downsample_is_rejected(layout, warnings):
    assert layout.generate_instructions()
    assert layout.instructions["downsample"] == 10

    layout.downsample_le.setText("")

    assert not layout.generate_instructions()
    assert len(warnings) == 1


@pytest.fixture
def local_layout(qtbot):
    experiment = SimpleNamespace(
        locate_image=lambda: None, exp_channels=["brightfield", "nuclei"]
    )
    parent = SimpleNamespace(
        parent_window=experiment,
        protocols=[],
        protocol_list=SimpleNamespace(addItem=lambda item: None),
    )
    return LocalCorrectionLayout(parent)


def test_local_correction_can_be_added(local_layout):
    local_layout.add_instructions_to_parent_list()

    (instructions,) = local_layout.parent_window.protocols
    assert instructions["correction_type"] == "local"
    assert instructions["distance"] == 5


def test_local_correction_rejects_an_empty_distance(local_layout, warnings):
    local_layout.threshold_le.setText("")

    local_layout.add_instructions_to_parent_list()

    assert not local_layout.parent_window.protocols
    assert len(warnings) == 1


def test_local_correction_takes_the_offset(local_layout):
    local_layout.camera_offset_le.setText("100,5")

    assert local_layout.generate_instructions()
    assert local_layout.instructions["offset"] == pytest.approx(100.5)
