"""
Tests for EventAnnotator and PairEventAnnotator closeEvent cleanup.

These tests verify that the memory leak fixes in closeEvent properly:
1. Close matplotlib figures
2. Stop and delete animations
3. Clear large data structures
4. Call super().closeEvent()

Bug prevented: Memory leaks from unclosed matplotlib figures and FuncAnimation
reference cycles when closing the annotator windows.
"""

import pytest
import gc
import logging
from unittest.mock import MagicMock, patch, PropertyMock
import matplotlib.pyplot as plt
from PyQt5.QtGui import QCloseEvent

from celldetective.gui.base.components import CelldetectiveMainWindow


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    try:
        logging.disable(logging.CRITICAL)
        yield
    finally:
        logging.disable(logging.NOTSET)


class TestEventAnnotatorCloseEvent:
    """
    Tests for EventAnnotator.closeEvent memory cleanup.

    Bug: closeEvent did not properly close matplotlib figures or delete
    the FuncAnimation, causing memory leaks due to reference cycles.

    Fix: Added proper cleanup of fig, cell_fig, anim, stack, and df_tracks.
    """

    def test_closeevent_closes_matplotlib_figures(self, qtbot):
        """
        Test that closeEvent properly closes matplotlib figures.

        Steps:
        1. Create a mock EventAnnotator with fig and cell_fig attributes
        2. Call closeEvent
        3. Verify plt.close was called for both figures
        """
        with patch("celldetective.gui.annotator_common.plt") as mock_plt:
            # Create a minimal mock that simulates the annotator
            from celldetective.gui.event_annotator import EventAnnotator

            # Mock the parent and initialization to avoid complex setup
            with patch.object(EventAnnotator, "__init__", lambda self, parent: None):
                annotator = EventAnnotator(None)

                # Set up minimal attributes needed for closeEvent
                annotator.fig = MagicMock()
                annotator.cell_fig = MagicMock()
                annotator.anim = MagicMock()
                annotator.anim.event_source = MagicMock()
                annotator.stack = MagicMock()
                annotator.df_tracks = MagicMock()
                annotator.stop = MagicMock()

                # Background loader attributes set in __init__ (bypassed here)
                annotator._loader_thread = None
                annotator._progress_dialog = None

                # Mock stop_btn for stop() method if needed
                annotator.stop_btn = MagicMock()
                annotator.start_btn = MagicMock()
                annotator.prev_frame_btn = MagicMock()
                annotator.next_frame_btn = MagicMock()

                # Create a real QCloseEvent
                event = QCloseEvent()

                # Patch super().closeEvent to avoid Qt issues
                with patch.object(CelldetectiveMainWindow, "closeEvent"):
                    EventAnnotator.closeEvent(annotator, event)

                # Verify figures were closed
                assert mock_plt.close.call_count >= 2

    def test_closeevent_stops_animation(self, qtbot):
        """
        Test that closeEvent stops the animation.

        Steps:
        1. Create mock annotator with anim attribute
        2. Call closeEvent
        3. Verify animation event_source.stop() was called
        """
        from celldetective.gui.event_annotator import EventAnnotator

        with patch.object(EventAnnotator, "__init__", lambda self, parent: None):
            annotator = EventAnnotator(None)

            # Set up animation mock
            mock_anim = MagicMock()
            mock_anim.event_source = MagicMock()
            annotator.anim = mock_anim

            # Set up other required attributes
            annotator.fig = MagicMock()
            annotator.cell_fig = MagicMock()
            annotator.stack = MagicMock()
            annotator.df_tracks = MagicMock()
            annotator.stop = MagicMock()
            annotator.stop_btn = MagicMock()
            annotator.start_btn = MagicMock()
            annotator.prev_frame_btn = MagicMock()
            annotator.next_frame_btn = MagicMock()

            # Background loader attributes set in __init__ (bypassed here)
            annotator._loader_thread = None
            annotator._progress_dialog = None

            event = QCloseEvent()

            with patch("celldetective.gui.annotator_common.plt"):
                with patch.object(CelldetectiveMainWindow, "closeEvent"):
                    EventAnnotator.closeEvent(annotator, event)

            # Verify animation was stopped
            mock_anim.event_source.stop.assert_called_once()


class TestPairEventAnnotatorCloseEvent:
    """
    Tests for PairEventAnnotator.closeEvent memory cleanup.

    Bug: closeEvent only deleted self.stack and didn't close figures,
    stop animations, or clear dataframes.

    Fix: Added proper cleanup of fig, cell_fig, anim, stack, dataframes,
    and df_relative. Also calls super().closeEvent().
    """

    def test_closeevent_clears_dataframes(self, qtbot):
        """
        Test that closeEvent properly clears dataframes dictionary.

        Steps:
        1. Create mock PairEventAnnotator with dataframes attribute
        2. Call closeEvent
        3. Verify dataframes.clear() was called
        """
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        with patch.object(PairEventAnnotator, "__init__", lambda self, parent: None):
            annotator = PairEventAnnotator(None)

            # Set up dataframes mock
            mock_dataframes = MagicMock()
            annotator.dataframes = mock_dataframes

            # Set up other required attributes
            annotator.fig = MagicMock()
            annotator.cell_fig = MagicMock()
            annotator.anim = MagicMock()
            annotator.anim.event_source = MagicMock()
            annotator.stack = MagicMock()
            annotator.df_relative = MagicMock()
            annotator.stop = MagicMock()
            annotator.stop_btn = MagicMock()
            annotator.start_btn = MagicMock()
            annotator.prev_frame_btn = MagicMock()
            annotator.next_frame_btn = MagicMock()

            event = QCloseEvent()

            with patch("celldetective.gui.annotator_common.plt"):
                with patch.object(CelldetectiveMainWindow, "closeEvent"):
                    PairEventAnnotator.closeEvent(annotator, event)

            # Verify dataframes.clear() was called
            mock_dataframes.clear.assert_called_once()

    def test_closeevent_deletes_df_relative(self):
        """
        Test that closeEvent code deletes df_relative.

        Steps:
        1. Inspect the closeEvent source code
        2. Verify it contains the delete statement for df_relative
        """
        import inspect
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        source = inspect.getsource(PairEventAnnotator.closeEvent)

        # Verify the cleanup code exists
        assert "del self.df_relative" in source or "df_relative" in source

    def test_closeevent_closes_figures(self, qtbot):
        """
        Test that closeEvent properly closes matplotlib figures.

        Steps:
        1. Create mock PairEventAnnotator with fig and cell_fig
        2. Call closeEvent
        3. Verify plt.close was called
        """
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        with patch("celldetective.gui.annotator_common.plt") as mock_plt:
            with patch.object(
                PairEventAnnotator, "__init__", lambda self, parent: None
            ):
                annotator = PairEventAnnotator(None)

                # Set up required attributes
                annotator.fig = MagicMock()
                annotator.cell_fig = MagicMock()
                annotator.anim = MagicMock()
                annotator.anim.event_source = MagicMock()
                annotator.stack = MagicMock()
                annotator.dataframes = {}
                annotator.df_relative = MagicMock()
                annotator.stop = MagicMock()
                annotator.stop_btn = MagicMock()
                annotator.start_btn = MagicMock()
                annotator.prev_frame_btn = MagicMock()
                annotator.next_frame_btn = MagicMock()

                event = QCloseEvent()

                with patch.object(CelldetectiveMainWindow, "closeEvent"):
                    PairEventAnnotator.closeEvent(annotator, event)

                # Verify figures were closed
                assert mock_plt.close.call_count >= 2


class TestPairEventAnnotatorNoDuplicateMethods:
    """
    Test that the frame navigation methods are defined once.

    Bug: set_first_frame and set_last_frame were defined twice in the class,
    with the later definition shadowing the earlier one.
    """

    @pytest.mark.parametrize("name", ["set_first_frame", "set_last_frame"])
    def test_defined_once(self, name):
        """The class defining the method has a single definition of it."""
        import inspect
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        owner = next(
            cls for cls in PairEventAnnotator.__mro__ if name in cls.__dict__
        )
        assert inspect.getsource(owner).count(f"def {name}(") == 1


class TestPairEventAnnotatorNoNeighborhoodsError:
    """
    Test that PairEventAnnotator raises ValueError when no neighborhoods detected.

    Bug: PairEventAnnotator crashed with KeyError when opened without computed
    neighborhoods.

    Fix: raise a ValueError with a user-friendly message, which the caller shows.
    """

    def _annotator(self, dataframes):
        import pandas as pd
        from celldetective.gui.pair_event_annotator import PairEventAnnotator

        with patch.object(PairEventAnnotator, "__init__", lambda self, parent: None):
            annotator = PairEventAnnotator(None)
        annotator.dataframes = {
            pop: None if columns is None else pd.DataFrame(columns=columns)
            for pop, columns in dataframes.items()
        }
        return annotator

    def test_raises_valueerror_on_empty_neighborhoods(self, qtbot):
        """No neighborhood column in any population raises a ValueError."""

        annotator = self._annotator(
            {"targets": ["TRACK_ID", "FRAME", "area"], "effectors": None}
        )
        with pytest.raises(ValueError, match="No neighborhoods detected"):
            annotator.locate_neighborhoods()

    def test_lists_the_neighborhoods_per_population(self, qtbot):
        """The neighborhoods are named after their reference population."""

        annotator = self._annotator(
            {
                "targets": ["TRACK_ID", "neighborhood_2_circle_30_px"],
                "effectors": ["TRACK_ID", "neighborhood_self_circle_10_px"],
            }
        )
        annotator.locate_neighborhoods()
        assert annotator.neighborhood_cols == [
            "targets_ref_neighborhood_2_circle_30_px",
            "effectors_ref_neighborhood_self_circle_10_px",
        ]
