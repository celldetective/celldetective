from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PyQt5.QtWidgets import QListWidget

from celldetective.gui.layouts import RegistrationOptionsLayout
from celldetective.gui.viewers.registration_roi_viewer import RegistrationROIViewer
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.utils.registration import tukey_window


@pytest.fixture
def stack_path(tmp_path):
    rng = np.random.default_rng(0)
    stack = rng.random((3, 2, 60, 80)).astype(np.float32)
    path = tmp_path / "sample.tif"
    tifffile.imwrite(path, stack, imagej=True, metadata={"axes": "TCYX"})
    return str(path)


@pytest.fixture
def options_layout(qtbot):
    parent = SimpleNamespace(
        exp_channels=["brightfield", "nuclei"],
        protocol_layout=SimpleNamespace(protocols=[], protocol_list=QListWidget()),
    )
    return RegistrationOptionsLayout(parent)


def test_size_viewer_radius_measure_and_fixed_center(qtbot):
    viewer = CellSizeViewer(
        stack=np.zeros((1, 60, 80), dtype=np.uint8),
        initial_diameter=20,
        measure="radius",
        follow_view_center=False,
    )
    qtbot.addWidget(viewer)

    assert viewer.diameter_slider.value() == pytest.approx(10.0)
    viewer.diameter_slider.setValue(15.0)
    assert viewer.diameter == pytest.approx(30.0)
    assert viewer.circ.get_radius() == pytest.approx(15.0)

    center = viewer.circ.center
    viewer.ax.set_xlim(0, 20)
    assert viewer.circ.center == center


def test_roi_viewer_matches_registration_window_and_sets_parent(
    qtbot, stack_path, options_layout
):
    viewer = RegistrationROIViewer(
        options_layout,
        stack_path=stack_path,
        channel_names=["brightfield", "nuclei"],
        n_channels=2,
        target_channel=1,
        initial_radius=20,
        tukey_alpha=0.3,
    )
    qtbot.addWidget(viewer)

    assert viewer.circ.center == pytest.approx((39.5, 29.5))
    assert viewer.circ.get_radius() == pytest.approx(20.0)
    np.testing.assert_allclose(
        viewer.im_weight.get_alpha(),
        0.6 * (1 - tukey_window((60, 80), alpha=0.3, radius=20)),
        atol=1e-6,
    )

    viewer.diameter_slider.setValue(25.0)
    viewer.alpha_slider.setValue(0.5)
    assert viewer.circ.get_radius() == pytest.approx(25.0)
    assert viewer.circ_taper.get_radius() == pytest.approx(12.5)

    # The disk stays on the image centre when zooming.
    viewer.ax.set_xlim(0, 20)
    assert viewer.circ.center == pytest.approx((39.5, 29.5))

    viewer.apply_threshold_btn.click()
    assert options_layout.radius_le.text() == "25.0"
    assert options_layout.alpha_le.get_threshold() == pytest.approx(0.5)

    options_layout.add_correction_btn.click()
    protocol = options_layout.parent_window.protocol_layout.protocols[-1]
    assert protocol["radius"] == pytest.approx(25.0)
    assert protocol["tukey_alpha"] == pytest.approx(0.5)
