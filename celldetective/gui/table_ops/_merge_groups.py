from typing import List

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.gui_utils import PandasModel
from celldetective.gui.base.utils import center_window


class MergeGroupWidget(CelldetectiveWidget):
    def __init__(self, parent_window, columns: List[str] = [], n_cols_init: int = 3):

        super().__init__()
        self.parent_window = parent_window

        self.setWindowTitle("Merge classifications")
        self.group_cols = [
            c
            for c in list(self.parent_window.data.columns)
            if c.startswith("group_") or c.startswith("status_")
        ]
        self.group_cols.insert(0, "--")
        if len(columns) > n_cols_init:
            n_cols_init = len(columns)

        center_window(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 10, 30, 30)

        label = QLabel(
            "Merge several binary or multi-label classification features into a multi-label classification feature where each state is one of the possible combinations.\n"
        )

        label.setWordWrap(True)  # enables automatic line breaking
        label.setTextInteractionFlags(
            Qt.TextSelectableByMouse
        )  # optional, to allow copy
        label.setStyleSheet("color: gray;")  # optional style

        layout.addWidget(label)

        self.name_le = QLineEdit("group_multilabel")
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("name: "), 25)
        name_layout.addWidget(self.name_le, 75)
        layout.addLayout(name_layout)

        self.cbs_layout = QVBoxLayout()
        self.cbs_layout.setContentsMargins(0, 10, 0, 0)

        self.cbs = []
        for i in range(n_cols_init):
            cb_i = QComboBox()
            cb_i.addItems(self.group_cols)
            if i < len(columns):
                selection = columns[i]
                idx = cb_i.findText(selection)
                if idx >= 0:
                    cb_i.setCurrentIndex(idx)
                else:
                    cb_i.setCurrentIndex(0)
            self.cbs.append(cb_i)

            col_layout = QHBoxLayout()
            col_layout.addWidget(QLabel(f"state {i}: "), 25)
            col_layout.addWidget(cb_i, 75)
            self.cbs_layout.addLayout(col_layout)

        layout.addLayout(self.cbs_layout)

        self.add_feature_btn = QPushButton()
        self.add_feature_btn.setIcon(icon(MDI6.plus, color=self.help_color))
        self.add_feature_btn.setIconSize(QSize(20, 20))
        self.add_feature_btn.clicked.connect(self.add_col)
        self.add_feature_btn.setStyleSheet(self.button_select_all)
        layout.addWidget(self.add_feature_btn, alignment=Qt.AlignRight)

        self.submit_btn = QPushButton("Compute")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.compute)
        layout.addWidget(self.submit_btn, 30)

        self.setAttribute(Qt.WA_DeleteOnClose)

    def add_col(self):
        cb_i = QComboBox()
        cb_i.addItems(self.group_cols)
        self.cbs.append(cb_i)

        col_layout = QHBoxLayout()
        col_layout.addWidget(QLabel(f"state {len(self.cbs)}: "), 25)
        col_layout.addWidget(cb_i, 75)

        self.cbs_layout.addLayout(col_layout)

    def compute(self):

        cols_to_merge = [
            cb_i.currentText() for cb_i in self.cbs if cb_i.currentText() != "--"
        ]
        name = self.name_le.text()
        if " " in name:
            name.replace(" ", "_")
        if name == "":
            name = "multilabel"
        if not name.startswith("group_"):
            name = "group_" + name

        if len(cols_to_merge) > 1:
            print(
                "Computing a multi-label classification from the classification feature sources..."
            )
            bases = [int(self.parent_window.data[c].max()) + 1 for c in cols_to_merge]
            multipliers = np.concatenate(([1], np.cumprod(bases[:-1])))
            self.parent_window.data[name] = (
                self.parent_window.data[cols_to_merge] * multipliers
            ).sum(axis=1)
            self.parent_window.data.loc[
                self.parent_window.data[cols_to_merge].isna().any(axis=1), name
            ] = np.nan

            self.parent_window.model = PandasModel(self.parent_window.data)
            self.parent_window.table_view.setModel(self.parent_window.model)
            self.close()
        elif len(cols_to_merge) == 1:
            print("Only one classification feature was selected, nothing to merge...")
        else:
            print("No classification feature was selected...")
