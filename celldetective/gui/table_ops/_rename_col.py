from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import PandasModel


class RenameColWidget(CelldetectiveWidget):

    def __init__(self, parent_window, column=None):

        super().__init__()
        self.parent_window = parent_window
        self.column = column
        if self.column is None:
            self.column = ""

        self.setWindowTitle("Rename column")
        # Create the QComboBox and add some items
        center_window(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        self.new_col_name = QLineEdit()
        self.new_col_name.setText(self.column)
        layout.addWidget(self.new_col_name, 70)

        self.submit_btn = QPushButton("rename")
        self.submit_btn.clicked.connect(self.rename_col)
        layout.addWidget(self.submit_btn, 30)
        self.setAttribute(Qt.WA_DeleteOnClose)

    def rename_col(self):

        old_name = self.column
        new_name = self.new_col_name.text()
        self.parent_window.data = self.parent_window.data.rename(
            columns={old_name: new_name}
        )

        self.parent_window.model = PandasModel(self.parent_window.data)
        self.parent_window.table_view.setModel(self.parent_window.model)
        self.close()
