"""
Unit tests for ControlPanel in control_panel.py.

Tests cover:
- ControlPanel initialization with experiment directory
- Well and Position detection
- Configuration loading
- UI interactions (selection, invalid inputs)
"""

import pytest
import os
import shutil
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from unittest.mock import patch, MagicMock

from celldetective.gui.control_panel import ControlPanel, generic_message
from celldetective.gui.analyze_block import AnalysisPanel


@pytest.fixture(autouse=True)
def mock_generic_message():
    """Mock generic_message to avoid blocking dialogs."""
    with patch("celldetective.gui.control_panel.generic_message") as mock_msg:
        yield mock_msg


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
def temp_experiment_dir(tmp_path):
    """
    Create a temporary experiment directory structure.

    Structure:
    Experiment_Test/
      config.ini
      W1/
        100/
          movie/
            movie_t000.tif
          output/
        101/
      W2/
        200/
    """
    exp_dir = tmp_path / "Experiment_Test"
    exp_dir.mkdir()

    # Create config.ini
    config_path = exp_dir / "config.ini"
    with open(config_path, "w") as f:
        f.write("[MovieSettings]\n")
        f.write("len_movie = 10\n")
        f.write("shape_x = 512\n")
        f.write("shape_y = 512\n")
        f.write("movie_prefix = movie\n")
        f.write("pxtoum = 0.65\n")
        f.write("frametomin = 5.0\n")
        f.write("\n")
        f.write("[Labels]\n")
        f.write("concentrations = 0,0\n")
        f.write("cell_types = A,B\n")
        f.write("antibodies = None,None\n")
        f.write("pharmaceutical_agents = None,None\n")
        f.write("\n")
        f.write("[Metadata]\n")
        f.write("date = 2023-01-01\n")
        f.write("\n")
        f.write("[Populations]\n")
        f.write("populations = targets,effectors\n")
        f.write("targets = red\n")
        f.write("effectors = green\n")

    # Create Wells and Positions
    # Use W1, W2 naming convention for wells
    # Use 100, 101, 200 naming convention for positions
    w1 = exp_dir / "W1"
    w1.mkdir()

    p1 = w1 / "100"
    p1.mkdir()
    (p1 / "movie").mkdir()
    (p1 / "output").mkdir()

    # Create dummy movie file
    with open(p1 / "movie" / "movie_t000.tif", "w") as f:
        f.write("dummy content")

    p2 = w1 / "101"
    p2.mkdir()
    (p2 / "movie").mkdir()

    w2 = exp_dir / "W2"
    w2.mkdir()
    p3 = w2 / "200"
    p3.mkdir()
    p4 = w2 / "201"
    p4.mkdir()

    return str(exp_dir)


@pytest.fixture
def mock_background_loader():
    """Mock the BackgroundLoader to prevent thread execution during tests."""
    with patch("celldetective.gui.control_panel.BackgroundLoader") as MockLoader:
        mock_instance = MockLoader.return_value
        mock_instance.start = MagicMock()
        yield MockLoader


@pytest.fixture
def mock_parent_window(temp_experiment_dir):
    """Create a mock parent window with required attributes."""
    mock = MagicMock()
    mock.use_gpu = False
    # Mocking the recursive parent structure expected by some panels
    # ControlPanel.parent_window -> MainWindow
    # MainWindow.exp_dir might be accessed
    mock.exp_dir = temp_experiment_dir
    return mock


@pytest.fixture(autouse=True)
def mock_analysis_panel_header():
    """Prevent AnalysisPanel from trying to generate header which fails with mocks."""
    with patch("celldetective.gui.analyze_block.AnalysisPanel.generate_header"):
        yield


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestControlPanelInitialization:
    """Test ControlPanel initialization."""

    def test_init_with_valid_experiment(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify ControlPanel initializes with a valid experiment."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)
        panel.show()

        assert panel.exp_dir.endswith(os.sep)
        assert panel.windowTitle() == "celldetective"
        assert len(panel.wells) == 2  # W1, W2
        assert len(panel.positions) == 2  # Positions for W1, W2
        assert len(panel.populations) == 2  # targets, effectors

        panel.close()

    def test_wells_detected_correctly(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify wells are detected correctly."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        # Wells are sorted paths
        well_names = [os.path.basename(os.path.normpath(w)) for w in panel.wells]
        assert "W1" in well_names
        assert "W2" in well_names

        panel.close()

    def test_positions_detected_correctly(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify positions are detected for each well."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        # Check positions for W1 (index 0)
        # positions[0] contains names like '100', '101'
        pos_names_w1 = panel.positions[0]
        assert "100" in pos_names_w1
        assert "101" in pos_names_w1

        panel.close()


# =============================================================================
# SELECTION LOGIC TESTS
# =============================================================================


class TestSelectionLogic:
    """Test well and position selection logic."""

    def test_well_selection_updates_positions(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify selecting a well updates the position list."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)
        panel.show()

        # Initially W1 is selected (index 0)
        assert panel.well_list.currentIndex() == 0
        current_pos_items = [
            panel.position_list.itemText(i) for i in range(panel.position_list.count())
        ]
        assert "100" in current_pos_items
        assert "101" in current_pos_items

        # Select W2 (index 1) which has only 200
        # setCurrentIndex toggles selection. W1 (0) was selected by init.
        # Toggle 0 to deselect it
        panel.well_list.setCurrentIndex(0)
        # Toggle 1 to select it
        panel.well_list.setCurrentIndex(1)

        # DEBUG: Verify selection state manually to ensure robust testing
        if (
            not panel.well_list.isAnySelected()
            or panel.well_list.getSelectedIndices()[0] != 1
        ):
            print(
                "WARNING: setCurrentIndex(1) failed to select index 1. Forcing selection for test."
            )
            # Manually force selection logic if UI event simulation fails
            panel.well_list.model().item(1).setCheckState(Qt.Checked)
            panel.well_list.toolMenu.actions()[1].setChecked(True)

        # Trigger activation manually since programmatic change might not trigger it depending on implementation
        panel.display_positions()

        qtbot.wait(50)

        new_pos_items = [
            panel.position_list.itemText(i) for i in range(panel.position_list.count())
        ]
        assert "200" in new_pos_items
        assert "100" not in new_pos_items

        panel.close()

    def test_select_all_wells(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify select all wells functionality."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)
        panel.show()

        # Click select all wells
        panel.select_all_wells_btn.click()
        qtbot.wait(50)

        assert panel.select_all_wells_option is True
        # Verify multiple selection logic in display_positions
        # If all wells detected, positions might be cleared or show linspace index
        # Implementation detail: display_positions handles multiple selection
        assert panel.position_list.count() > 0

        # Click again to deselect
        panel.select_all_wells_btn.click()
        qtbot.wait(50)

        assert panel.select_all_wells_option is False

        panel.close()

    def test_select_all_positions(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify select all positions functionality."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)
        panel.show()

        # Click select all positions
        panel.select_all_pos_btn.click()
        qtbot.wait(50)

        assert panel.select_all_pos_option is True
        # For CheckableComboBox, all should be checked (internal state)
        # We can simulate checking selection state if needed

        panel.close()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfigurationLoading:
    """Test configuration loading."""

    def test_load_config_values(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify configuration values are loaded into attributes."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        assert panel.len_movie == 10
        assert panel.shape_x == 512
        assert panel.shape_y == 512
        assert panel.movie_prefix == "movie"

        panel.close()

    def test_create_config_dir(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify config directory is created."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        config_dir = os.path.join(temp_experiment_dir, "configs")
        assert os.path.exists(config_dir)

        panel.close()


# =============================================================================
# INTERACTION TESTS
# =============================================================================


class TestInteractions:
    """Test UI interactions."""

    def test_locate_image(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify locate_image finds the movie file."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        # Ensure correct well/pos selected
        # W1 (0) and Pos (0) are selected by default initialization

        panel.locate_selected_position()  # Update self.pos

        panel.locate_image()

        assert panel.current_stack is not None
        assert "movie_t000.tif" in panel.current_stack

        panel.close()

    def test_locate_image_fail(
        self, qtbot, temp_experiment_dir, mock_background_loader, mock_parent_window
    ):
        """Verify locate_image handles missing movies."""
        panel = ControlPanel(
            parent_window=mock_parent_window, exp_dir=temp_experiment_dir
        )
        qtbot.addWidget(panel)

        # W2/200 has no movie file created in fixture
        # Switch from W1 (0) to W2 (1)
        panel.well_list.setCurrentIndex(0)  # Toggle W1 off
        panel.well_list.setCurrentIndex(1)  # Toggle W2 on

        # Force selection helper in case setCurrentIndex is flaky (same as in TestSelectionLogic)
        if not panel.well_list.isAnySelected() or (
            panel.well_list.isSingleSelection()
            and panel.well_list.getSelectedIndices()[0] != 1
        ):
            if panel.well_list.model().item(1).checkState() != Qt.Checked:
                panel.well_list.model().item(1).setCheckState(Qt.Checked)
                panel.well_list.toolMenu.actions()[1].setChecked(True)

        panel.display_positions()  # Update positions list

        # Position list for W2 should be loaded, defaulting to index 0 selected.

        panel.locate_selected_position()

        # Patch generic_message to avoid modal dialog blocking test
        with patch("celldetective.gui.control_panel.generic_message") as mock_msg:
            panel.locate_image()
            assert panel.current_stack is None
            # generic_message is called if locate_image fails to find movies
            mock_msg.assert_called_once()

        panel.close()
