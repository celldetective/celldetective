"""
Unit tests for specialized viewers in celldetective.gui.viewers.

Covers:
- ChannelOffsetViewer
- CellEdgeVisualizer (ContourViewer)
- CellSizeViewer
- ThresholdedStackVisualizer
"""

import pytest
import numpy as np
import logging
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QLineEdit, QListWidget, QMainWindow
from PyQt5.QtCore import Qt

from celldetective.gui.viewers.channel_offset_viewer import ChannelOffsetViewer
from celldetective.gui.viewers.contour_viewer import CellEdgeVisualizer
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.gui.viewers.threshold_viewer import ThresholdedStackVisualizer


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logger = logging.getLogger()
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


# Monkeypatch superqt bug
try:
    from superqt.sliders._generic_slider import _GenericSlider

    # Check if we need to patch
    if not hasattr(_GenericSlider, "_original_to_qinteger_space"):
        _GenericSlider._original_to_qinteger_space = _GenericSlider._to_qinteger_space

        def patched_to_qinteger_space(self, value):
            return int(_GenericSlider._original_to_qinteger_space(self, value))

        _GenericSlider._to_qinteger_space = patched_to_qinteger_space
except ImportError:
    pass


@pytest.fixture
def dummy_stack_4d():
    """Create a 4D dummy stack (T, Y, X, C)."""
    # 5 frames, 100x100, 2 channels
    stack = np.zeros((5, 100, 100, 2), dtype=np.uint8)
    # Add some "features"
    stack[:, 20:40, 20:40, 0] = 200  # Channel 0 feature
    stack[:, 30:50, 30:50, 1] = 150  # Channel 1 feature (offset)
    return stack


@pytest.fixture
def dummy_labels_3d():
    """Create 3D dummy labels (T, Y, X)."""
    # 5 frames, 100x100
    labels = np.zeros((5, 100, 100), dtype=np.uint8)
    # Create a simple square object
    labels[:, 40:60, 40:60] = 1
    return labels


@pytest.fixture
def mock_parent_window():
    """Mock parent window with channels combo string."""
    window = MagicMock(spec=QMainWindow)
    window.channels_cb = MagicMock()
    window.channels_cb.currentIndex.return_value = 0
    window.vertical_shift_le = MagicMock()
    window.horizontal_shift_le = MagicMock()
    return window


@pytest.fixture
def mock_line_edit():
    """Mock QLineEdit."""
    le = MagicMock(spec=QLineEdit)
    le.setText = MagicMock()
    le.set_threshold = MagicMock()
    return le


@pytest.fixture
def mock_list_widget():
    """Mock QListWidget."""
    lw = MagicMock(spec=QListWidget)
    lw.addItems = MagicMock()
    return lw


# =============================================================================
# ChannelOffsetViewer Tests
# =============================================================================


class TestChannelOffsetViewer:
    """Tests for ChannelOffsetViewer."""

    def test_initialization(self, qtbot, dummy_stack_4d, mock_parent_window):
        """Test initialization with 4D stack."""
        viewer = ChannelOffsetViewer(
            stack=dummy_stack_4d,
            n_channels=2,
            channel_names=["Ch1", "Ch2"],
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.n_channels == 2
        assert viewer.channel_names == ["Ch1", "Ch2"]
        assert hasattr(viewer, "im_overlay")

    def test_overlay_channel_selection(self, qtbot, dummy_stack_4d, mock_parent_window):
        """Test changing overlay channel."""
        viewer = ChannelOffsetViewer(
            stack=dummy_stack_4d,
            n_channels=2,
            channel_names=["Ch1", "Ch2"],
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(viewer)

        # Default overlay set in init
        viewer.channels_overlay_cb.setCurrentIndex(1)
        assert viewer.overlay_target_channel == 1

    def test_shift_controls(self, qtbot, dummy_stack_4d, mock_parent_window):
        """Test shift controls."""
        viewer = ChannelOffsetViewer(
            stack=dummy_stack_4d,
            n_channels=2,
            channel_names=["Ch1", "Ch2"],
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(viewer)

        # Test shift buttons/shortcuts logic
        viewer.shift_overlay_right()
        assert viewer.shift_horizontal == 2
        assert viewer.horizontal_shift_le.get_threshold() == 2

        viewer.shift_overlay_down()
        assert viewer.shift_vertical == 2
        assert viewer.vertical_shift_le.get_threshold() == 2

        # Test Apply button (invokes shift_generic)
        with patch.object(viewer, "update_profile") as mock_update:
            viewer.apply_shift_btn.click()
            assert mock_update.called

    def test_set_parent_attributes(self, qtbot, dummy_stack_4d, mock_parent_window):
        """Test setting attributes on parent window."""
        viewer = ChannelOffsetViewer(
            stack=dummy_stack_4d,
            n_channels=2,
            channel_names=["Ch1", "Ch2"],
            parent_window=mock_parent_window,
        )
        qtbot.addWidget(viewer)

        viewer.shift_horizontal = 10
        viewer.horizontal_shift_le.set_threshold(10)

        viewer.set_parent_attributes()

        mock_parent_window.horizontal_shift_le.set_threshold.assert_called_with(10.0)


# =============================================================================
# CellEdgeVisualizer (ContourViewer) Tests
# =============================================================================


class TestContourViewer:
    """Tests for CellEdgeVisualizer."""

    def test_initialization(self, qtbot, dummy_labels_3d):
        """Test initialization with labels."""
        viewer = CellEdgeVisualizer(
            stack=dummy_labels_3d, labels=dummy_labels_3d, initial_edge=5
        )
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.edge_size == 5
        assert hasattr(viewer, "im_mask")
        assert viewer.labels.shape == dummy_labels_3d.shape

    def test_edge_size_update(self, qtbot, dummy_labels_3d):
        """Test that changing edge size updates labels mask."""
        viewer = CellEdgeVisualizer(
            stack=dummy_labels_3d, labels=dummy_labels_3d, initial_edge=5
        )
        qtbot.addWidget(viewer)

        # Mock drawing to speed up
        viewer.canvas.canvas.draw_idle = MagicMock()

        # Change edge size via slider logic or directly
        # The slider is a RangeSlider, value is tuple.
        viewer.change_edge_size((2, 5))

        assert viewer.edge_size == (2, 5)
        viewer.canvas.canvas.draw_idle.assert_called()

    def test_add_measurement(self, qtbot, dummy_labels_3d, mock_list_widget):
        """Test adding measurement to parent list widget."""
        viewer = CellEdgeVisualizer(
            stack=dummy_labels_3d,
            labels=dummy_labels_3d,
            parent_list_widget=mock_list_widget,
            single_value_mode=True,
        )
        qtbot.addWidget(viewer)

        # Set slider value
        viewer.edge_slider.setValue((-5, 5))

        viewer.set_measurement_in_parent_list()

        # single_value_mode=True returns max abs value -> 5
        mock_list_widget.addItems.assert_called_with(["5"])


# =============================================================================
# CellSizeViewer Tests
# =============================================================================


class TestSizeViewer:
    """Tests for CellSizeViewer."""

    def test_initialization(self, qtbot):
        """Test initialization (no stack needed strictly, but base needs it)."""
        # Base viewer needs at least something.
        # StackVisualizer __init__ calls load_stack which handles None?
        # Actually StackVisualizer expects data.
        dummy_stack = np.zeros((1, 100, 100), dtype=np.uint8)

        viewer = CellSizeViewer(stack=dummy_stack, initial_diameter=20)
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.diameter == 20
        assert hasattr(viewer, "circ")

    def test_diameter_change(self, qtbot):
        """Test changing diameter slider."""
        dummy_stack = np.zeros((1, 100, 100), dtype=np.uint8)
        viewer = CellSizeViewer(stack=dummy_stack, initial_diameter=20)
        qtbot.addWidget(viewer)

        viewer.diameter_slider.setValue(40)
        assert viewer.diameter == 40
        assert (
            viewer.circ.radius == 20.0
        )  # Radius is diameter/2 * (1/PxToUm which defaults to 1?)
        # PxToUm defaults to 1.0 if not set in StackVisualizer?
        # Actually PxToUm is set to 1.0 in StackVisualizer init if not passed.

    def test_add_to_parent_le(self, qtbot, mock_line_edit):
        """Test setting value in parent LineEdit."""
        dummy_stack = np.zeros((1, 100, 100), dtype=np.uint8)
        viewer = CellSizeViewer(
            stack=dummy_stack, parent_le=mock_line_edit, initial_diameter=30
        )
        qtbot.addWidget(viewer)

        viewer.set_threshold_in_parent_le()

        mock_line_edit.set_threshold.assert_called_with(30.0)


# =============================================================================
# ThresholdedStackVisualizer Tests
# =============================================================================


class TestThresholdViewer:
    """Tests for ThresholdedStackVisualizer."""

    def test_initialization(self, qtbot, dummy_stack_4d):
        """Test initialization."""
        # Use single channel for thresholding usually,
        # but the viewer handles stacks. It takes frame slices.
        # dummy_stack_4d is (5, 100, 100, 2)
        # StackVisualizer defaults to target_channel=0

        viewer = ThresholdedStackVisualizer(
            stack=dummy_stack_4d, initial_threshold=10.0
        )
        qtbot.addWidget(viewer)
        viewer.show()

        assert viewer.thresh == 10.0
        assert hasattr(viewer, "mask")

    def test_threshold_update(self, qtbot, dummy_stack_4d):
        """Test threshold update recalculates mask."""
        viewer = ThresholdedStackVisualizer(
            stack=dummy_stack_4d, initial_threshold=255.0  # High threshold = empty mask
        )
        qtbot.addWidget(viewer)

        # Initially mask should be all 0s for high thresh
        assert np.all(viewer.mask == 0)

        # Set low threshold
        viewer.change_threshold(50.0)

        # Should now have some True values (dummy stack has 200 in region)
        # Viewer uses target_channel which defaults to 0.
        # Dummy stack ch0 has 200s.
        assert np.any(viewer.mask == 1)
        assert viewer.thresh == 50.0

    def test_set_parent_le(self, qtbot, dummy_stack_4d, mock_line_edit):
        """Test apply button updates parent LE."""
        viewer = ThresholdedStackVisualizer(
            stack=dummy_stack_4d, parent_le=mock_line_edit, initial_threshold=100.0
        )
        qtbot.addWidget(viewer)

        viewer.set_threshold_in_parent_le()

        mock_line_edit.set_threshold.assert_called_with(100.0)
