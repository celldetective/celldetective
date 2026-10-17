"""
The collapsible blocks of the control panel.

:class:`ControlPanelBlock` adds to :class:`CollapsibleFrame` the only window
management the control panel needs: once a block has finished opening or
closing, the window is given the height its content asks for, capped at a
fraction of the screen so that it never grows past it -- past that cap the
content scrolls instead.
"""

import logging
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import QMainWindow, QWidget

from celldetective.gui.base.collapsible import CollapsibleFrame
from celldetective.gui.base.utils import fit_window_to_content

logger = logging.getLogger("celldetective")


class ControlPanelBlock(CollapsibleFrame):
    """
    A collapsible block of the control panel.

    Attributes
    ----------
    screen_fraction : float
        The largest share of the height of the screen the control panel may
        take once a block is open.
    """

    screen_fraction = 0.9

    def __init__(
        self,
        title: str,
        parent_window: QMainWindow,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the block.

        Parameters
        ----------
        title : str
            The title shown in the band.
        parent_window : QMainWindow
            The control panel the block belongs to.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(title, parent=parent)

        self.parent_window = parent_window
        self.animation_finished.connect(self.fit_parent_window)

    def collect_batch_structure(self) -> Dict[int, Dict[str, Any]]:
        """
        List the positions selected in the control panel, grouped by well,
        for the wells of ``self.well_index``.

        Returns
        -------
        dict
            Maps the well index to its name and the paths of the selected positions.
        """

        batch_structure = {}

        pos_indices = self.parent_window.position_list.getSelectedIndices()

        for w_idx in self.well_index:
            well = self.parent_window.wells[w_idx]

            batch_structure[w_idx] = {"well_name": well, "positions": []}

            all_well_positions = self.parent_window.position_paths[w_idx]
            for pos_idx in pos_indices:
                if pos_idx < len(all_well_positions):
                    pos = all_well_positions[pos_idx]
                else:
                    logger.warning(
                        f"Position index {pos_idx} out of range for well {well}"
                    )
                    continue

                batch_structure[w_idx]["positions"].append(pos)

        return batch_structure

    def fit_parent_window(self, expanded: bool) -> None:
        """
        Give the window the height its content now asks for.

        Parameters
        ----------
        expanded : bool
            Whether the block was opened. Both states are handled the same
            way: the window follows its content, up to the screen.
        """

        window = self.window()
        area = getattr(self.parent_window, "scroll", None)

        if area is None:
            return

        try:
            fit_window_to_content(window, area, screen_fraction=self.screen_fraction)
        except RuntimeError as e:
            logger.debug(f"Window resizing failed: {e}")
