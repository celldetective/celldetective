from typing import Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
)

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.styles import MUTED_INK
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import PandasModel


class RenameColWidget(CelldetectiveWidget):

    def __init__(
        self, parent_window: QMainWindow, column: Optional[str] = None
    ) -> None:
        """
        Initialize the RenameColWidget.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window.
        column : str, optional
            The column to rename.
        """

        super().__init__()
        self.parent_window = parent_window
        self.column = column
        if self.column is None:
            self.column = ""

        self.setWindowTitle("Rename column")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        hint = QLabel(f"New name for the column '{self.column}':")
        hint.setStyleSheet(f"font-size: 11px; color: {MUTED_INK};")
        layout.addWidget(hint)

        row = QHBoxLayout()
        self.new_col_name = QLineEdit()
        self.new_col_name.setText(self.column)
        self.new_col_name.selectAll()
        self.new_col_name.returnPressed.connect(self.rename_col)
        row.addWidget(self.new_col_name, 1)

        self.submit_btn = QPushButton("Rename")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.rename_col)
        row.addWidget(self.submit_btn)
        layout.addLayout(row)

        self.new_col_name.textChanged.connect(
            lambda text: self.submit_btn.setEnabled(bool(text.strip()))
        )

        center_window(self)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def rename_col(self):
        """Rename the column."""

        old_name = self.column
        new_name = self.new_col_name.text()
        if not new_name.strip():
            return

        self.parent_window.data = self.parent_window.data.rename(
            columns={old_name: new_name}
        )

        self.parent_window.model = PandasModel(self.parent_window.data)
        self.parent_window.table_view.setModel(self.parent_window.model)
        self.close()
