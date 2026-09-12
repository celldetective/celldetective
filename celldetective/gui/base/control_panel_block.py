"""
The collapsible blocks of the control panel.

:class:`ControlPanelBlock` adds to :class:`CollapsibleFrame` the only window
management the control panel needs: once a block has finished opening or
closing, the window is given the height its content asks for, capped at a
fraction of the screen so that it never grows past it -- past that cap the
content scrolls instead.
"""

import logging
from typing import Optional

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
