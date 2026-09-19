from abc import abstractmethod
from PyQt5.QtWidgets import (
    QApplication,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QPushButton,
)
from PyQt5.QtCore import Qt
from celldetective import get_software_location
from celldetective.gui.base.utils import (
    center_window,
    fit_window_to_content,
    flush_layout_events,
)
from celldetective.gui.base.collapsible import CollapsibleFrame
from celldetective.gui.base.components import (
    CelldetectiveMainWindow,
    CelldetectiveWidget,
)
from celldetective import get_logger
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from typing import Optional


logger = get_logger()


class CelldetectiveSettingsPanel(CelldetectiveMainWindow):

    # The largest share of the height of the screen a settings window may take
    # once its blocks are open; past that the content scrolls instead.
    screen_fraction = 0.8

    def __init__(self, title: Optional[str] = None) -> None:
        """
        Initialize the CelldetectiveSettingsPanel.

        Parameters
        ----------
        title : str, optional
            The title of the settings panel.
        """

        super().__init__()
        self.setWindowTitle(title)

        self._get_screen_height()
        # self.setMinimumWidth(500)
        self.setMaximumHeight(int(0.8 * self._screen_height))
        self._scroll_area = QScrollArea(self)
        self._floatValidator = QDoubleValidator()
        self._intValidator = QIntValidator()
        self._software_path = get_software_location()

        self._create_widgets()
        self._build_layouts()
        self.center_window()

    def _create_widgets(self):
        """Create the widgets."""
        self.submit_btn: QPushButton = QPushButton("Save")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self._write_instructions)

    def center_window(self):
        """Center the window on the screen."""
        return center_window(self)

    def _get_screen_height(self):
        """Get the available screen height for the monitor where the cursor is."""
        from celldetective.gui.base.utils import get_current_screen_geometry
        geometry = get_current_screen_geometry()
        self._screen_width, self._screen_height = geometry.width(), geometry.height()

    def _adjust_size(self):
        """Adjust the size of the widget."""
        self._widget.adjustSize()
        self._scroll_area.adjustSize()
        self.adjustSize()

    def fit_to_content(self):
        """
        Give the window the height its content now asks for, within the screen.

        The counterpart, for a settings window, of what the control panel does
        when one of its blocks is opened or closed.
        """

        try:
            fit_window_to_content(
                self, self._scroll_area, screen_fraction=self.screen_fraction
            )
        except RuntimeError as e:
            logger.debug(f"Window resizing failed: {e}")
            return

        # The height is all `fit_window_to_content` follows. A window narrower
        # than its content is widened to it, chrome of the scroll area included,
        # so that nothing has to be scrolled sideways; a window the user has
        # widened by hand is left alone.
        wanted = (
            max(self._widget.sizeHint().width(), self._widget.minimumWidth())
            + self._scroll_area_chrome()
        )
        if self.width() < wanted:
            self.resize(wanted, self.height())

    def _scroll_area_chrome(self) -> int:
        """
        Return the width the window needs on top of that of its content.

        Measured from the scroll area rather than from the width the viewport
        currently has: a viewport is only given its new width once the window
        has been laid out again, and reading it back right after a resize makes
        the window ratchet up in width, one call adding the chrome of the last.
        """

        margins = self.contentsMargins()
        chrome = margins.left() + margins.right()
        chrome += 2 * self._scroll_area.frameWidth()

        if self._scroll_area.verticalScrollBarPolicy() != Qt.ScrollBarAlwaysOff:
            # Room for the scroll bar whether it shows or not: the blocks are
            # opened and closed, and with them it comes and goes.
            chrome += self._scroll_area.verticalScrollBar().sizeHint().width()

        return chrome

    def _follow_block_states(self):
        """
        Have the window follow its collapsible blocks as they open and close.

        Called once the blocks are in the layout, since it is then that they are
        children of the window.
        """

        for frame in self.findChildren(CollapsibleFrame):
            frame.animation_finished.connect(lambda _expanded: self.fit_to_content())

    def _build_layouts(self):
        """Build the layouts."""

        self._layout: QVBoxLayout = QVBoxLayout()
        self._widget: CelldetectiveWidget = CelldetectiveWidget()
        self._widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create button widget and layout
        self._widget.setLayout(self._layout)
        self._layout.setContentsMargins(30, 30, 30, 30)

        self._scroll_area.setAlignment(Qt.AlignCenter)
        self._scroll_area.setWidget(self._widget)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll_area.setWidgetResizable(True)
        self.setCentralWidget(self._scroll_area)

        flush_layout_events(self)

    @abstractmethod
    def _load_previous_instructions(self):
        """Load previous instructions."""
        pass

    @abstractmethod
    def _write_instructions(self):
        """Write instructions."""
        pass
