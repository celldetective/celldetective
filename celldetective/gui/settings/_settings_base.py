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
from celldetective.gui.base.utils import center_window
from celldetective.gui.base.components import (
    CelldetectiveMainWindow,
    CelldetectiveWidget,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from typing import Optional


class CelldetectiveSettingsPanel(CelldetectiveMainWindow):

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
        """Get the screen height."""
        app = QApplication.instance()
        screen = app.primaryScreen()
        geometry = screen.availableGeometry()
        self._screen_width, self._screen_height = geometry.getRect()[-2:]

    def _adjust_size(self):
        """Adjust the size of the widget."""
        self._widget.adjustSize()
        self._scroll_area.adjustSize()
        self.adjustSize()

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

        QApplication.processEvents()

    @abstractmethod
    def _load_previous_instructions(self):
        """Load previous instructions."""
        pass

    @abstractmethod
    def _write_instructions(self):
        """Write instructions."""
        pass
