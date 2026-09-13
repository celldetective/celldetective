import logging

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
)

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.styles import DANGER_COLOR, MUTED_INK
from celldetective.gui.base.utils import center_window

logger = logging.getLogger("celldetective")


class QueryWidget(CelldetectiveWidget):

    def __init__(self, parent_window: QMainWindow) -> None:
        """
        Initialize the QueryWidget.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        """

        super().__init__()
        self.parent_window = parent_window

        self.setWindowTitle("Filter table")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        hint = QLabel(
            "Keep the rows matching a pandas query. Wrap column names holding "
            "spaces or symbols in backticks."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {MUTED_INK};")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.query_le = QLineEdit()
        self.query_le.setPlaceholderText("e.g. area > 100 and TRACK_ID == 3")
        self.query_le.returnPressed.connect(self.filter_table)
        row.addWidget(self.query_le, 1)

        self.submit_btn = QPushButton("Filter")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.filter_table)
        row.addWidget(self.submit_btn)
        layout.addLayout(row)

        # Said here rather than only in the log: a mistyped query used to do
        # nothing at all as far as the window showed.
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"font-size: 11px; color: {DANGER_COLOR};")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        center_window(self)

    def filter_table(self):
        """Filter the table based on the query."""
        from celldetective.gui.tableUI import TableUI

        try:
            query_text = self.query_le.text()  # .replace('class', '`class`')
            tab = self.parent_window.data.query(query_text)
            self.subtable = TableUI(
                tab,
                query_text,
                plot_mode="static",
                population=self.parent_window.population,
            )
            self.subtable.show()
            self.close()
        except Exception as e:
            logger.error(f"{e}")
            self.error_label.setText(f"The query could not be run: {e}")
            self.error_label.show()
            return None
