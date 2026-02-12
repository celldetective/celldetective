import pytest
import numpy as np
import logging
from PyQt5.QtWidgets import QApplication
from celldetective.gui.viewers.spot_detection_viewer import SpotDetectionVisualizer
from celldetective.gui.gui_utils import PreprocessingLayout2
from unittest.mock import MagicMock, patch


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
def dummy_data():
    """
    Create a dummy stack: 5 frames, 100x100 pixels, 2 channels.
    Channel 0: Clean background (zeros)
    Channel 1: Two Gaussian spots with high intensity
    """
    frames = 5
    y, x = 100, 100
    channels = 2

    stack = np.zeros((frames, y, x, channels), dtype=np.float32)

    # Create coordinate grids
    Y, X = np.ogrid[:y, :x]

    # Gaussian spot 1 at (22, 22) - center of mask 1
    center_y1, center_x1 = 22, 22
    sigma = 2.0
    gaussian1 = np.exp(-((Y - center_y1) ** 2 + (X - center_x1) ** 2) / (2 * sigma**2))

    # Gaussian spot 2 at (62, 62) - center of mask 2
    center_y2, center_x2 = 62, 62
    gaussian2 = np.exp(-((Y - center_y2) ** 2 + (X - center_x2) ** 2) / (2 * sigma**2))

    # Add to stack with high intensity (1000 for spot 1, 800 for spot 2)
    spots_frame = (gaussian1 * 1000 + gaussian2 * 800).astype(np.float32)
    for f in range(frames):
        stack[f, :, :, 1] = spots_frame

    # Channel 0 stays at zero (clean background)

    # Create dummy masks (labels) - each spot is inside its own cell
    masks = np.zeros((frames, y, x), dtype=np.uint16)
    masks[:, 15:30, 15:30] = 1  # Mask around Spot 1 (22, 22)
    masks[:, 55:70, 55:70] = 2  # Mask around Spot 2 (62, 62)

    return stack, masks


def test_spot_detection_visualizer_interactions(qtbot, dummy_data):
    """
    Test interactions with SpotDetectionVisualizer.
    """
    stack, labels = dummy_data
    channel_names = ["Background", "Spots"]

    # Mock parent widgets that might be updated by the visualizer
    parent_channel_cb = MagicMock()
    parent_diameter_le = MagicMock()
    parent_threshold_le = MagicMock()
    parent_preprocessing_list = MagicMock()

    viewer = SpotDetectionVisualizer(
        stack=stack,
        labels=labels,
        channel_names=channel_names,
        n_channels=2,
        parent_channel_cb=parent_channel_cb,
        parent_diameter_le=parent_diameter_le,
        parent_threshold_le=parent_threshold_le,
        parent_preprocessing_list=parent_preprocessing_list,
        window_title="Test Spot Detective",
        channel_cb=True,
        contrast_slider=False,
        frame_slider=False,
    )

    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitForWindowShown(viewer)

    # 1. Test Channel Selection
    # Default is target_channel=0 (Background)
    assert viewer.detection_channel == 0

    # Switch to Spots channel (Index 1)
    viewer.detection_channel_cb.setCurrentIndex(1)
    assert viewer.detection_channel == 1

    # Force frame update to ensure target_img is correct for channel 1
    viewer.change_frame(0)

    # Verify image updated to Channel 1, Frame 0
    current_img = viewer.target_img
    expected_img = stack[0, :, :, 1]
    np.testing.assert_array_equal(current_img, expected_img)

    # 2. Test Spot Detection Parameters
    # Set Diameter (LoG works best with diameter ~ 2*sqrt(2)*sigma ~ 5.6 for sigma=2)
    viewer.spot_diam_le.clear()
    qtbot.keyClicks(viewer.spot_diam_le, "4")
    assert viewer.spot_diam_le.text() == "4"

    # Set Threshold (low threshold to ensure detection)
    viewer.spot_thresh_le.clear()
    qtbot.keyClicks(viewer.spot_thresh_le, "0.01")
    assert viewer.spot_thresh_le.text() == "0.01"

    # Manually trigger control_valid_parameters to update self.diameter and self.thresh
    viewer.control_valid_parameters()

    # Verify parameters were set
    assert viewer.diameter == 4.0
    assert viewer.thresh == 0.01

    # Trigger detection by clicking apply button
    qtbot.mouseClick(viewer.apply_diam_btn, 1)  # Qt.LeftButton = 1
    qtbot.wait(200)  # Wait for detection to complete

    # Check that we recovered 2 spots
    # In dummy_data: Spot 1 at (22, 22), Spot 2 at (62, 62)
    n_spots = (
        len(viewer.spot_positions)
        if hasattr(viewer, "spot_positions") and viewer.spot_positions is not None
        else 0
    )
    assert (
        n_spots == 2
    ), f"Expected 2 spots, found {n_spots}. Positions: {viewer.spot_positions if hasattr(viewer, 'spot_positions') else 'N/A'}"

    # Verify positions roughly match (22, 22) and (62, 62)
    # spot_positions are (x, y) pairs
    pos = viewer.spot_positions
    has_spot_1 = np.any(np.all(np.abs(pos - [22, 22]) < 5, axis=1))
    has_spot_2 = np.any(np.all(np.abs(pos - [62, 62]) < 5, axis=1))
    assert has_spot_1, f"Spot 1 not found near (22, 22). Positions: {pos}"
    assert has_spot_2, f"Spot 2 not found near (62, 62). Positions: {pos}"

    # 3. Test Preprocessing and Preview
    # Ensure preview is unchecked initially
    assert not viewer.preview_cb.isChecked()

    # Check "Preview" - should show original image if filter list is empty
    viewer.preview_cb.setChecked(True)
    assert viewer.preview_cb.isChecked()
    # Image should still match original target since no filters
    np.testing.assert_array_equal(viewer.im.get_array(), expected_img)

    # Add a filter: "gauss" with sigma=2
    # Directly manipulate the list since dialog interaction is complex
    viewer.preprocessing.list.items.append(["gauss", 2])
    viewer.preprocessing.list.list_widget.addItems(["gauss_filter"])

    # Force preview update
    viewer.update_preview_if_active()
    qtbot.wait(200)

    preview_img = viewer.im.get_array()
    assert not np.array_equal(
        preview_img, expected_img
    ), "Preview image should differ after adding gaussian filter"

    # 4. Remove Filter
    # Select item 0
    viewer.preprocessing.list.list_widget.setCurrentRow(0)
    # Click remove button
    viewer.preprocessing.delete_filter_btn.click()

    qtbot.wait(200)

    # Internal items should be empty
    assert len(viewer.preprocessing.list.items) == 0

    # Preview should revert to original
    reverted_img = viewer.im.get_array()
    np.testing.assert_array_equal(reverted_img, expected_img)


@pytest.fixture
def dark_spots_data():
    """
    Create a dummy stack with DARK spots on a BRIGHT background.
    This simulates scenarios like:
    - Absorbing particles on a bright field
    - Phase contrast imaging with dark nuclei
    The invert preprocessing filter should be used to detect these spots.
    """
    frames = 5
    y, x = 100, 100
    channels = 2

    # Bright background (value 1000)
    stack = np.ones((frames, y, x, channels), dtype=np.float32) * 1000

    # Create coordinate grids
    Y, X = np.ogrid[:y, :x]

    # Dark Gaussian spot 1 at (22, 22)
    center_y1, center_x1 = 22, 22
    sigma = 2.0
    gaussian1 = np.exp(-((Y - center_y1) ** 2 + (X - center_x1) ** 2) / (2 * sigma**2))

    # Dark Gaussian spot 2 at (62, 62)
    center_y2, center_x2 = 62, 62
    gaussian2 = np.exp(-((Y - center_y2) ** 2 + (X - center_x2) ** 2) / (2 * sigma**2))

    # Subtract from bright background to create dark spots
    # Spot 1: drops to ~0 at center, Spot 2: drops to ~200 at center
    dark_spots_frame = 1000 - (gaussian1 * 1000 + gaussian2 * 800)
    dark_spots_frame = dark_spots_frame.astype(np.float32)

    for f in range(frames):
        stack[f, :, :, 1] = dark_spots_frame

    # Channel 0 stays uniform bright (no spots)

    # Create dummy masks
    masks = np.zeros((frames, y, x), dtype=np.uint16)
    masks[:, 15:30, 15:30] = 1  # Mask around Spot 1
    masks[:, 55:70, 55:70] = 2  # Mask around Spot 2

    return stack, masks


def test_dark_spot_detection_with_invert(qtbot, dark_spots_data):
    """
    Test detection of dark spots on a bright background using the invert filter.

    Bug Prevention:
    - Ensures the preprocessing pipeline (invert) correctly transforms images
      before spot detection.
    - Verifies that the detection uses the preprocessed image, not the raw image.

    Steps:
    1. Load stack with dark spots on bright background.
    2. Switch to the dark spots channel.
    3. Add an "invert" preprocessing filter with max value 1000.
    4. Set detection parameters (diameter, threshold).
    5. Trigger detection.
    6. Verify that both dark spots are detected.

    Expected Outcome:
    - 2 spots detected at positions (22, 22) and (62, 62).
    """
    stack, labels = dark_spots_data
    channel_names = ["Uniform", "DarkSpots"]

    parent_channel_cb = MagicMock()
    parent_diameter_le = MagicMock()
    parent_threshold_le = MagicMock()
    parent_preprocessing_list = MagicMock()

    viewer = SpotDetectionVisualizer(
        stack=stack,
        labels=labels,
        channel_names=channel_names,
        n_channels=2,
        parent_channel_cb=parent_channel_cb,
        parent_diameter_le=parent_diameter_le,
        parent_threshold_le=parent_threshold_le,
        parent_preprocessing_list=parent_preprocessing_list,
        window_title="Test Dark Spot Detection",
        channel_cb=True,
        contrast_slider=False,
        frame_slider=False,
    )

    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitForWindowShown(viewer)

    # 1. Switch to DarkSpots channel (Index 1)
    viewer.detection_channel_cb.setCurrentIndex(1)
    assert viewer.detection_channel == 1
    viewer.change_frame(0)

    # Verify image is bright with dark spots
    current_img = viewer.target_img
    # The center of spot 1 (22, 22) should be dark (near 0)
    assert (
        current_img[22, 22] < 100
    ), f"Expected dark spot at (22,22), got {current_img[22, 22]}"
    # The background should be bright (near 1000)
    assert (
        current_img[0, 0] > 900
    ), f"Expected bright background, got {current_img[0, 0]}"

    # 2. Add invert preprocessing filter
    # invert filter inverts the image: output = max_value - input
    # With max_value=1000, dark spots become bright spots
    viewer.preprocessing.list.items.append(["invert", 1000])
    viewer.preprocessing.list.list_widget.addItems(["invert_1000"])

    # 3. Set detection parameters
    viewer.spot_diam_le.clear()
    qtbot.keyClicks(viewer.spot_diam_le, "4")
    viewer.spot_thresh_le.clear()
    qtbot.keyClicks(viewer.spot_thresh_le, "0.01")
    viewer.control_valid_parameters()

    assert viewer.diameter == 4.0
    assert viewer.thresh == 0.01

    # Enable preview to verify inversion works
    viewer.preview_cb.setChecked(True)
    viewer.update_preview_if_active()
    qtbot.wait(200)

    # After inversion, the former dark spot at (22, 22) should now be bright
    preview_img = viewer.im.get_array()
    assert (
        preview_img[22, 22] > 900
    ), f"After invert, spot should be bright. Got {preview_img[22, 22]}"

    # 4. Trigger detection
    qtbot.mouseClick(viewer.apply_diam_btn, 1)
    qtbot.wait(200)

    # 5. Verify spots detected
    n_spots = (
        len(viewer.spot_positions)
        if hasattr(viewer, "spot_positions") and viewer.spot_positions is not None
        else 0
    )
    assert (
        n_spots == 2
    ), f"Expected 2 dark spots detected, found {n_spots}. Positions: {getattr(viewer, 'spot_positions', 'N/A')}"

    # Verify positions
    pos = viewer.spot_positions
    has_spot_1 = np.any(np.all(np.abs(pos - [22, 22]) < 5, axis=1))
    has_spot_2 = np.any(np.all(np.abs(pos - [62, 62]) < 5, axis=1))
    assert has_spot_1, f"Dark Spot 1 not found near (22, 22). Positions: {pos}"
    assert has_spot_2, f"Dark Spot 2 not found near (62, 62). Positions: {pos}"


def test_viewer_initializes_from_parent_values(qtbot, dummy_data):
    """
    Test that the viewer initializes its widgets from initial_* arguments.
    """
    stack, labels = dummy_data
    channel_names = ["Background", "Spots"]

    # Mock parent widgets
    parent_channel_cb = MagicMock()
    parent_diameter_le = MagicMock()
    parent_threshold_le = MagicMock()
    parent_preprocessing_list = MagicMock()

    initial_params = [["gauss", 1.5], ["invert", 500]]

    viewer = SpotDetectionVisualizer(
        stack=stack,
        labels=labels,
        channel_names=channel_names,
        n_channels=2,
        parent_channel_cb=parent_channel_cb,
        parent_diameter_le=parent_diameter_le,
        parent_threshold_le=parent_threshold_le,
        parent_preprocessing_list=parent_preprocessing_list,
        window_title="Test Init Values",
        channel_cb=True,
        contrast_slider=False,
        frame_slider=False,
        # Pass initial values
        initial_diameter="8.5",
        initial_threshold="1.2",
        initial_preprocessing=initial_params,
    )

    qtbot.addWidget(viewer)
    viewer.show()
    qtbot.waitForWindowShown(viewer)

    # Verify widgets were initialized with passed values
    assert viewer.spot_diam_le.text() == "8.5"
    assert viewer.spot_thresh_le.text() == "1.2"

    # Verify preprocessing list was populated
    current_filters = viewer.preprocessing.list.items
    assert len(current_filters) == 2
    assert current_filters[0] == ["gauss", 1.5]
    assert current_filters[1] == ["invert", 500]
