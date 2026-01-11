from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from fonticon_mdi6 import MDI6
from superqt import QSearchableComboBox
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import PandasModel


class MergeOneHotWidget(CelldetectiveWidget):

    def __init__(self, parent_window, selected_columns=None):

        super().__init__()
        self.parent_window = parent_window
        self.selected_columns = selected_columns

        self.setWindowTitle("Merge one-hot encoded columns...")
        # Create the QComboBox and add some items
        center_window(self)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(30, 30, 30, 30)

        if self.selected_columns is not None:
            n_cols = len(self.selected_columns)
        else:
            n_cols = 2

        name_hbox = QHBoxLayout()
        name_hbox.addWidget(QLabel("New categorical column: "), 33)
        self.new_col_le = QLineEdit()
        self.new_col_le.setText("categorical_")
        self.new_col_le.textChanged.connect(self.allow_merge)
        name_hbox.addWidget(self.new_col_le, 66)
        self.layout.addLayout(name_hbox)

        self.layout.addWidget(QLabel("Source columns: "))

        self.cbs = [QSearchableComboBox() for _ in range(n_cols)]
        self.cbs_layout = QVBoxLayout()

        for i in range(n_cols):
            lay = QHBoxLayout()
            lay.addWidget(QLabel(f"column {i}: "), 33)
            self.cbs[i].addItems(["--"] + list(self.parent_window.data.columns))
            if self.selected_columns is not None:
                self.cbs[i].setCurrentText(self.selected_columns[i])
            lay.addWidget(self.cbs[i], 66)
            self.cbs_layout.addLayout(lay)

        self.layout.addLayout(self.cbs_layout)

        hbox = QHBoxLayout()
        self.add_col_btn = QPushButton("Add column")
        self.add_col_btn.clicked.connect(self.add_col)
        self.add_col_btn.setStyleSheet(self.button_add)
        self.add_col_btn.setIcon(icon(MDI6.plus, color="black"))

        hbox.addWidget(QLabel(""), 50)
        hbox.addWidget(self.add_col_btn, 50, alignment=Qt.AlignRight)
        self.layout.addLayout(hbox)

        self.submit_btn = QPushButton("Merge")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.merge_cols)
        self.layout.addWidget(self.submit_btn, 30)

        self.setAttribute(Qt.WA_DeleteOnClose)

    def add_col(self):
        self.cbs.append(QSearchableComboBox())
        self.cbs[-1].addItems(["--"] + list(self.parent_window.data.columns))
        lay = QHBoxLayout()
        lay.addWidget(QLabel(f"column {len(self.cbs)-1}: "), 33)
        lay.addWidget(self.cbs[-1], 66)
        self.cbs_layout.addLayout(lay)

    def merge_cols(self):

        self.parent_window.data[self.new_col_le.text()] = self.parent_window.data.loc[
            :, list(self.selected_columns)
        ].idxmax(axis=1)
        self.parent_window.model = PandasModel(self.parent_window.data)
        self.parent_window.table_view.setModel(self.parent_window.model)
        self.close()

    def allow_merge(self):

        if self.new_col_le.text() == "":
            self.submit_btn.setEnabled(False)
        else:
            self.submit_btn.setEnabled(True)
