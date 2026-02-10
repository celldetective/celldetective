"""
GUI tests for StackVisualizer with all documented image formats.

Verifies that the StackVisualizer widget can open and display stacks in
every format documented in first-experiment.rst:
- XY   (2D) : single-timepoint & single-channel
- CXY  (3D) : single-timepoint & multichannel
- TXY  (3D) : time-lapse
- TCXY (4D) : multi-channel time-lapse

Two modes are tested:
1. Direct mode: passing a pre-loaded numpy array.
2. Virtual mode: passing a file path (StackVisualizer loads on demand).
"""

import os
import shutil
import tempfile
import logging

import pytest
import numpy as np
from tifffile import imwrite

from celldetective.utils.io import save_tiff_imagej_compatible
from celldetective.gui.viewers.base_viewer import StackVisualizer

H, W = 64, 64
N_CHANNELS = 3
N_FRAMES = 5


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


# Monkeypatch superqt bug (same as test_viewers.py)
try:
    from superqt.sliders._generic_slider import _GenericSlider

    if not hasattr(_GenericSlider, "_original_to_qinteger_space"):
        _GenericSlider._original_to_qinteger_space = _GenericSlider._to_qinteger_space

        def patched_to_qinteger_space(self, value):
            return int(_GenericSlider._original_to_qinteger_space(self, value))

        _GenericSlider._to_qinteger_space = patched_to_qinteger_space
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Fixtures: synthetic TIF files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tif_dir():
    """Create a temporary directory with synthetic TIFs for all formats."""
    tmpdir = tempfile.mkdtemp(prefix="cd_gui_fmt_")

    paths = {}

    # XY – (H, W) – 1 frame, 1 channel
    paths["xy"] = os.path.join(tmpdir, "xy.tif")
    save_tiff_imagej_compatible(
        paths["xy"],
        np.random.randint(0, 255, (H, W), dtype=np.uint16),
        axes="YX",
    )

    # CXY – (C, H, W) – 1 frame, 3 channels
    paths["cxy"] = os.path.join(tmpdir, "cxy.tif")
    save_tiff_imagej_compatible(
        paths["cxy"],
        np.random.randint(0, 255, (N_CHANNELS, H, W), dtype=np.uint16),
        axes="CYX",
    )

    # TXY – (T, H, W) – 5 frames, 1 channel
    paths["txy"] = os.path.join(tmpdir, "txy.tif")
    save_tiff_imagej_compatible(
        paths["txy"],
        np.random.randint(0, 255, (N_FRAMES, H, W), dtype=np.uint16),
        axes="TYX",
    )

    # TCXY – (T, C, H, W) – 5 frames, 3 channels
    paths["tcxy"] = os.path.join(tmpdir, "tcxy.tif")
    save_tiff_imagej_compatible(
        paths["tcxy"],
        np.random.randint(0, 255, (N_FRAMES, N_CHANNELS, H, W), dtype=np.uint16),
        axes="TCYX",
    )

    # OME-TIFF TCXY
    paths["ome_tcxy"] = os.path.join(tmpdir, "tcxy.ome.tif")
    imwrite(
        paths["ome_tcxy"],
        np.random.randint(0, 255, (N_FRAMES, N_CHANNELS, H, W), dtype=np.uint16),
        ome=True,
        metadata={"axes": "TCYX"},
    )

    # OME-TIFF TXY
    paths["ome_txy"] = os.path.join(tmpdir, "txy.ome.tif")
    imwrite(
        paths["ome_txy"],
        np.random.randint(0, 255, (N_FRAMES, H, W), dtype=np.uint16),
        ome=True,
        metadata={"axes": "TYX"},
    )

    yield paths

    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Direct mode: StackVisualizer receives a pre-loaded numpy array
# ---------------------------------------------------------------------------


class TestStackVisualizerDirect:
    """Open the StackVisualizer with pre-loaded arrays for each format."""

    def test_xy_direct(self, qtbot):
        """XY (2D → expanded to (1, H, W, 1)) opens without error."""
        # locate_stack would produce (1, H, W, 1)
        stack = np.random.randint(0, 255, (1, H, W, 1), dtype=np.uint16)
        viewer = StackVisualizer(stack=stack, n_channels=1)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "direct"
        assert viewer.stack_length == 1
        assert viewer.init_frame.shape == (H, W)

    def test_cxy_direct(self, qtbot):
        """CXY (multichannel, single frame) opens with correct channel count."""
        stack = np.random.randint(0, 255, (1, H, W, N_CHANNELS), dtype=np.uint16)
        viewer = StackVisualizer(stack=stack, n_channels=N_CHANNELS)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "direct"
        assert viewer.stack_length == 1
        assert viewer.n_channels == N_CHANNELS
        assert viewer.init_frame.shape == (H, W)

    def test_txy_direct(self, qtbot):
        """TXY (time-lapse, single channel) opens with correct frame count."""
        # 3D input: (T, Y, X) → load_stack adds channel axis → (T, Y, X, 1)
        stack = np.random.randint(0, 255, (N_FRAMES, H, W), dtype=np.uint16)
        viewer = StackVisualizer(stack=stack, n_channels=1)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "direct"
        assert viewer.stack_length == N_FRAMES
        assert viewer.init_frame.shape == (H, W)

    def test_tcxy_direct(self, qtbot):
        """TCXY (multichannel time-lapse) opens with all channels and frames."""
        stack = np.random.randint(0, 255, (N_FRAMES, H, W, N_CHANNELS), dtype=np.uint16)
        viewer = StackVisualizer(stack=stack, n_channels=N_CHANNELS)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "direct"
        assert viewer.stack_length == N_FRAMES
        assert viewer.n_channels == N_CHANNELS
        assert viewer.init_frame.shape == (H, W)


# ---------------------------------------------------------------------------
# Virtual mode: StackVisualizer loads from a file path
# ---------------------------------------------------------------------------


class TestStackVisualizerVirtual:
    """Open the StackVisualizer in virtual mode (stack_path) for each format."""

    def test_xy_virtual(self, qtbot, tif_dir):
        """XY file opens in virtual mode."""
        viewer = StackVisualizer(stack_path=tif_dir["xy"], n_channels=1)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == 1
        assert viewer.init_frame.shape == (H, W)

    def test_cxy_virtual(self, qtbot, tif_dir):
        """CXY file opens in virtual mode with correct channels."""
        viewer = StackVisualizer(stack_path=tif_dir["cxy"], n_channels=N_CHANNELS)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == 1
        assert viewer.n_channels == N_CHANNELS

    def test_txy_virtual(self, qtbot, tif_dir):
        """TXY file opens in virtual mode with correct frames."""
        viewer = StackVisualizer(stack_path=tif_dir["txy"], n_channels=1)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == N_FRAMES

    def test_tcxy_virtual(self, qtbot, tif_dir):
        """TCXY file opens in virtual mode with correct channels and frames."""
        viewer = StackVisualizer(stack_path=tif_dir["tcxy"], n_channels=N_CHANNELS)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == N_FRAMES
        assert viewer.n_channels == N_CHANNELS


# ---------------------------------------------------------------------------
# OME-TIFF virtual mode
# ---------------------------------------------------------------------------


class TestStackVisualizerOME:
    """Open the StackVisualizer in virtual mode with OME-TIFF files."""

    def test_ome_txy_virtual(self, qtbot, tif_dir):
        """OME-TIFF TXY file opens in virtual mode."""
        viewer = StackVisualizer(stack_path=tif_dir["ome_txy"], n_channels=1)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == N_FRAMES

    def test_ome_tcxy_virtual(self, qtbot, tif_dir):
        """OME-TIFF TCXY file opens in virtual mode."""
        viewer = StackVisualizer(stack_path=tif_dir["ome_tcxy"], n_channels=N_CHANNELS)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.mode == "virtual"
        assert viewer.stack_length == N_FRAMES
        assert viewer.n_channels == N_CHANNELS
