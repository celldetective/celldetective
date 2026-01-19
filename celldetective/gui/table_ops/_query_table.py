from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QPushButton

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window


class QueryWidget(CelldetectiveWidget):

    def __init__(self, parent_window):

        super().__init__()
        self.parent_window = parent_window

        self.setWindowTitle("Filter table")
        # Create the QComboBox and add some items

        layout = QHBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        self.query_le = QLineEdit()
        layout.addWidget(self.query_le, 70)

        self.submit_btn = QPushButton("submit")
        self.submit_btn.clicked.connect(self.filter_table)
        layout.addWidget(self.submit_btn, 30)
        center_window(self)

    def filter_table(self):
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
            print(e)
            return None
