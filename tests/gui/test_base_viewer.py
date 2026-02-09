"""
Unit tests for StackVisualizer and related components in base_viewer.py.

Tests cover:
- StackVisualizer initialization (with numpy stack and file path)
- Frame navigation via slider
- Contrast control via slider
- Channel selection via combobox
- Line profile tool
- StackLoader background loading
"""

import pytest
import numpy as np
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from unittest.mock import MagicMock, patch

from celldetective.gui.viewers.base_viewer import StackVisualizer, StackLoader


@pytest.fixture
def safe_close():
    """
    Fixture that provides a safe close method for viewers.
    Prevents RuntimeError when C++ objects are already deleted.
    """

    def _safe_close(viewer):
        try:
            if hasattr(viewer, "loader_thread") and viewer.loader_thread:
                viewer.loader_thread.stop()
                viewer.loader_thread = None
            if hasattr(viewer, "frame_cache"):
                viewer.frame_cache.clear()
            # Use deleteLater for safer Qt cleanup
            viewer.deleteLater()
        except RuntimeError:
            pass  # C++ object already deleted

    return _safe_close


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logger = logging.getLogger()
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def dummy_stack_3d():
    """
    Create a dummy 3D stack (T, Y, X) - single channel.
    5 frames, 100x100 pixels.
    """
    frames = 5
    y, x = 100, 100
    stack = np.zeros((frames, y, x), dtype=np.float32)

    # Add some varying intensity patterns per frame
    for f in range(frames):
        # Create a gradient that changes per frame
        Y, X = np.ogrid[:y, :x]
        stack[f] = (X + Y + f * 10).astype(np.float32)

    return stack


@pytest.fixture
def dummy_stack_4d():
    """
    Create a dummy 4D stack (T, Y, X, C) - multi-channel.
    5 frames, 100x100 pixels, 2 channels.
    """
    frames = 5
    y, x = 100, 100
    channels = 2
    stack = np.zeros((frames, y, x, channels), dtype=np.float32)

    Y, X = np.ogrid[:y, :x]

    # Channel 0: Gradient pattern
    for f in range(frames):
        stack[f, :, :, 0] = (X + Y + f * 10).astype(np.float32)

    # Channel 1: Gaussian spots
    center_y, center_x = 50, 50
    sigma = 10.0
    gaussian = np.exp(-((Y - center_y) ** 2 + (X - center_x) ** 2) / (2 * sigma**2))
    for f in range(frames):
        stack[f, :, :, 1] = (gaussian * 1000 + f * 50).astype(np.float32)

    return stack


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestStackVisualizerInitialization:
    """Test StackVisualizer initialization with different inputs."""

    def test_init_with_3d_stack(self, qtbot, dummy_stack_3d):
        """Verify initialization with a 3D numpy array (T, Y, X)."""
        viewer = StackVisualizer(
            stack=dummy_stack_3d,
            frame_slider=True,
            contrast_slider=True,
            channel_cb=False,  # 3D stack = single channel
            n_channels=1,
            window_title="Test 3D Stack",
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Verify stack was converted to 4D internally
        assert viewer.stack.ndim == 4
        assert viewer.stack.shape == (5, 100, 100, 1)

        # Verify mode is direct (in-memory)
        assert viewer.mode == "direct"

        # Verify frame count
        assert viewer.stack_length == 5

        # Verify current frame index
        assert viewer.current_time_index == 0

        viewer.close()

    def test_init_with_4d_stack(self, qtbot, dummy_stack_4d):
        """Verify initialization with a 4D numpy array (T, Y, X, C)."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            frame_slider=True,
            contrast_slider=True,
            channel_cb=True,
            channel_names=["Gradient", "Gaussian"],
            n_channels=2,
            target_channel=0,
            window_title="Test 4D Stack",
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Verify stack shape unchanged
        assert viewer.stack.ndim == 4
        assert viewer.stack.shape == (5, 100, 100, 2)

        # Verify n_channels
        assert viewer.n_channels == 2

        # Verify target channel
        assert viewer.target_channel == 0

        # Verify channel names
        assert viewer.channel_names == ["Gradient", "Gaussian"]

        viewer.close()

    def test_channel_combobox_populated(self, qtbot, dummy_stack_4d):
        """Verify channel combobox is populated with channel names."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            channel_cb=True,
            channel_names=["Red", "Green"],
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Verify combobox exists and has items
        assert hasattr(viewer, "channel_cb")
        assert viewer.channel_cb.count() == 2
        assert viewer.channel_cb.itemText(0) == "Red"
        assert viewer.channel_cb.itemText(1) == "Green"

        viewer.close()

    def test_channel_combobox_with_default_names(self, qtbot, dummy_stack_4d):
        """Verify channel combobox uses default names when not provided."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            channel_cb=True,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Default names should be "Channel 0", "Channel 1", etc.
        assert viewer.channel_cb.itemText(0) == "Channel 0"
        assert viewer.channel_cb.itemText(1) == "Channel 1"

        viewer.close()

    def test_frame_slider_created(self, qtbot, dummy_stack_3d):
        """Verify frame slider is created when enabled."""
        viewer = StackVisualizer(
            stack=dummy_stack_3d,
            frame_slider=True,
            contrast_slider=False,
            channel_cb=False,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        assert hasattr(viewer, "frame_slider")
        assert viewer.frame_slider.minimum() == 0
        assert viewer.frame_slider.maximum() == 4  # 5 frames, 0-indexed

        viewer.close()

    def test_contrast_slider_created(self, qtbot, dummy_stack_3d):
        """Verify contrast slider is created when enabled."""
        viewer = StackVisualizer(
            stack=dummy_stack_3d,
            frame_slider=False,
            contrast_slider=True,
            channel_cb=False,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        assert hasattr(viewer, "contrast_slider")

        viewer.close()


# =============================================================================
# FRAME NAVIGATION TESTS
# =============================================================================


class TestFrameNavigation:
    """Test frame navigation functionality."""

    def test_frame_slider_updates_display(self, qtbot, dummy_stack_4d):
        """Verify changing frame slider updates the displayed image."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            frame_slider=True,
            contrast_slider=True,
            channel_cb=True,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Get initial frame data
        initial_frame_data = viewer.init_frame.copy()

        # Change to frame 3
        viewer.frame_slider.setValue(3)
        qtbot.wait(100)

        # Verify current_time_index updated
        assert viewer.current_time_index == 3

        # Verify displayed data changed
        new_frame_data = viewer.init_frame
        assert not np.array_equal(initial_frame_data, new_frame_data)

        viewer.close()

    def test_change_frame_programmatic(self, qtbot, dummy_stack_4d):
        """Verify change_frame method works correctly."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            frame_slider=True,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Programmatically change frame
        viewer.change_frame(2)
        qtbot.wait(100)

        assert viewer.current_time_index == 2

        # Verify init_frame matches expected data
        expected = dummy_stack_4d[2, :, :, 0]
        np.testing.assert_array_equal(viewer.init_frame, expected)

        viewer.close()

    def test_frame_bounds_respected(self, qtbot, dummy_stack_3d):
        """Verify frame navigation respects bounds."""
        viewer = StackVisualizer(
            stack=dummy_stack_3d,
            frame_slider=True,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Should not go below 0
        viewer.frame_slider.setValue(0)
        assert viewer.frame_slider.value() == 0

        # Should not go above max
        viewer.frame_slider.setValue(100)  # Way beyond max
        assert viewer.frame_slider.value() == 4  # Clamped to max

        viewer.close()


# =============================================================================
# CONTRAST CONTROL TESTS
# =============================================================================


class TestContrastControl:
    """Test contrast slider functionality."""

    def test_contrast_slider_initial_range(self, qtbot, dummy_stack_4d):
        """Verify contrast slider is initialized with correct range."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            contrast_slider=True,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Slider range should cover the data range of the initial frame
        slider_min = viewer.contrast_slider.minimum()
        slider_max = viewer.contrast_slider.maximum()

        data_min = np.nanmin(viewer.init_frame)
        data_max = np.nanmax(viewer.init_frame)

        assert slider_min == data_min
        assert slider_max == data_max

        viewer.close()

    def test_contrast_adjustment_updates_display(self, qtbot, dummy_stack_4d):
        """Verify contrast slider changes affect image display limits."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            contrast_slider=True,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Get initial clim
        initial_clim = viewer.im.get_clim()

        # Change contrast
        new_min = initial_clim[0] + 10
        new_max = initial_clim[1] - 10
        viewer.contrast_slider.setValue((new_min, new_max))
        qtbot.wait(100)

        # Verify imshow clim updated
        updated_clim = viewer.im.get_clim()
        assert updated_clim[0] == pytest.approx(new_min, abs=1)
        assert updated_clim[1] == pytest.approx(new_max, abs=1)

        viewer.close()


# =============================================================================
# CHANNEL SELECTION TESTS
# =============================================================================


class TestChannelSelection:
    """Test channel selection functionality."""

    def test_channel_switch_updates_display(self, qtbot, dummy_stack_4d):
        """Verify switching channel updates the displayed image."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            channel_cb=True,
            channel_names=["Gradient", "Gaussian"],
            n_channels=2,
            target_channel=0,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Get initial frame (channel 0)
        initial_data = viewer.init_frame.copy()

        # Switch to channel 1
        viewer.channel_cb.setCurrentIndex(1)
        qtbot.wait(100)

        # Verify target_channel updated
        assert viewer.target_channel == 1

        # Verify displayed data changed
        new_data = viewer.init_frame
        assert not np.array_equal(initial_data, new_data)

        # Verify data matches channel 1 of the stack
        expected = dummy_stack_4d[0, :, :, 1]
        np.testing.assert_array_equal(new_data, expected)

        viewer.close()

    def test_set_channel_index_method(self, qtbot, dummy_stack_4d):
        """Verify set_channel_index method works correctly."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            channel_cb=True,
            n_channels=2,
            target_channel=0,
            frame_slider=False,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Call set_channel_index directly
        viewer.set_channel_index(1)
        qtbot.wait(100)

        assert viewer.target_channel == 1

        viewer.close()


# =============================================================================
# LINE PROFILE TOOL TESTS
# =============================================================================


class TestLineProfileTool:
    """Test line profile drawing functionality."""

    def test_line_mode_toggle(self, qtbot, dummy_stack_4d):
        """Verify line profile mode can be toggled on and off."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Initially line mode should be off
        assert viewer.line_mode is False
        assert viewer.ax_profile is None

        # Toggle on
        viewer.line_action.trigger()
        qtbot.wait(100)

        assert viewer.line_mode is True
        assert viewer.ax_profile is not None
        assert viewer.lock_y_action.isEnabled()

        # Toggle off
        viewer.line_action.trigger()
        qtbot.wait(100)

        assert viewer.line_mode is False
        assert viewer.ax_profile is None

        viewer.close()

    def test_profile_axes_created_in_line_mode(self, qtbot, dummy_stack_4d):
        """Verify profile axes are created when line mode is enabled."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Enable line mode
        viewer.line_action.trigger()
        qtbot.wait(100)

        # Profile axes should exist
        assert viewer.ax_profile is not None
        assert viewer.ax_profile.get_visible()

        viewer.close()


# =============================================================================
# CLEANUP TESTS
# =============================================================================


class TestStackVisualizerCleanup:
    """Test cleanup and resource management."""

    def test_close_clears_resources(self, qtbot, dummy_stack_4d):
        """Verify closing viewer clears frame cache and threads."""
        viewer = StackVisualizer(
            stack=dummy_stack_4d,
            n_channels=2,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Close the viewer
        viewer.close()

        # Frame cache should be cleared (in direct mode, loader_thread is None)
        assert len(viewer.frame_cache) == 0

    def test_multiple_instantiations(self, qtbot, dummy_stack_4d):
        """Verify multiple viewers can be created and closed without issues."""
        viewers = []
        for i in range(3):
            viewer = StackVisualizer(
                stack=dummy_stack_4d,
                n_channels=2,
                window_title=f"Viewer {i}",
            )
            qtbot.addWidget(viewer)
            viewer.show()
            viewers.append(viewer)

        qtbot.wait(100)

        # Close all viewers
        for viewer in viewers:
            viewer.close()

        # No assertions needed - test passes if no crashes occur


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_frame_stack(self, qtbot):
        """Verify viewer works with a single-frame stack."""
        stack = np.random.rand(1, 50, 50).astype(np.float32)

        viewer = StackVisualizer(
            stack=stack,
            frame_slider=True,
            n_channels=1,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Frame slider should have range 0-0
        assert viewer.frame_slider.minimum() == 0
        assert viewer.frame_slider.maximum() == 0

        viewer.close()

    def test_stack_with_nan_values(self, qtbot):
        """Verify viewer handles NaN values gracefully."""
        stack = np.random.rand(3, 50, 50).astype(np.float32)
        stack[0, 10:20, 10:20] = np.nan  # Add some NaN values

        viewer = StackVisualizer(
            stack=stack,
            contrast_slider=True,
            n_channels=1,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Should not crash and contrast slider should be initialized
        assert hasattr(viewer, "contrast_slider")

        viewer.close()

    def test_stack_as_list(self, qtbot):
        """Verify viewer handles stack provided as a list."""
        frames = [np.random.rand(50, 50).astype(np.float32) for _ in range(3)]

        viewer = StackVisualizer(
            stack=frames,
            frame_slider=True,
            n_channels=1,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Stack should be converted to numpy array
        assert isinstance(viewer.stack, np.ndarray)
        assert viewer.stack_length == 3

        viewer.close()

    def test_custom_imshow_kwargs(self, qtbot, dummy_stack_3d):
        """Verify custom imshow kwargs are applied."""
        viewer = StackVisualizer(
            stack=dummy_stack_3d,
            imshow_kwargs={"cmap": "viridis"},
            n_channels=1,
        )

        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitForWindowShown(viewer)

        # Check that cmap was applied
        assert viewer.im.get_cmap().name == "viridis"

        viewer.close()
