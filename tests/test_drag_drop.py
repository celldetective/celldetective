import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QTableView,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PyQt5.QtCore import Qt, QAbstractTableModel


class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return self._data.index[col]
        return None


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        self.model = PandasModel(df)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)

        self.table_view.horizontalHeader().setSectionsMovable(True)
        self.table_view.horizontalHeader().setDragEnabled(True)
        self.table_view.horizontalHeader().setDragDropMode(self.table_view.InternalMove)

        # Add a button to dump the columns based on logical vs visual order
        self.btn = QPushButton("Print order")
        self.btn.clicked.connect(self.print_order)

        w = QWidget()
        l = QVBoxLayout()
        l.addWidget(self.table_view)
        l.addWidget(self.btn)
        w.setLayout(l)
        self.setCentralWidget(w)

    def print_order(self):
        h = self.table_view.horizontalHeader()
        cols = list(self.model._data.columns)
        new_order = [cols[h.logicalIndex(i)] for i in range(h.count())]
        print(f"Visual order maps to logical columns: {new_order}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = TestWindow()
    w.show()
    w.print_order()
    # app.exec()
