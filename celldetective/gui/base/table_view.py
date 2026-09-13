"""
The table view the data tables of celldetective are shown in.
"""

from typing import List, Optional

from PyQt5.QtCore import QAbstractItemModel, QItemSelection, Qt, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QKeySequence
from PyQt5.QtWidgets import QAbstractItemView, QApplication, QTableView, QWidget

from celldetective.gui.base.styles import TABLE_STYLE


class DataTableView(QTableView):
    """
    A table view in the look of the software, sized for measurement tables.

    Compared to a bare QTableView it has a light grid and a flat header, rows
    of a steady height, columns fitted to their content but kept from growing
    past a readable width, and Ctrl+C copying the selection as tab separated
    values, ready to paste in a spreadsheet.

    The tables of the software replace their model whenever a column is added
    or removed (``table_view.setModel(PandasModel(data))``). The view refits
    its columns and keeps following the selection on every such change, so
    that the call sites do not have to.
    """

    # Emitted once a new model is in place.
    model_changed = pyqtSignal()
    # Emitted whenever the selection changes, whichever model is shown.
    selection_changed = pyqtSignal()

    min_column_width = 56
    max_column_width = 240
    row_padding = 10
    # Rows sampled to fit a column: enough to be representative, few enough
    # for a table of a hundred thousand cells to open at once.
    fit_precision = 250

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the view.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.setStyleSheet(TABLE_STYLE)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideMiddle)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setCornerButtonEnabled(True)

        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setHighlightSections(True)
        header.setSectionsMovable(True)
        header.setDragEnabled(True)
        header.setDragDropMode(QAbstractItemView.InternalMove)
        header.setResizeContentsPrecision(self.fit_precision)
        header.setMinimumSectionSize(self.min_column_width)

        rows = self.verticalHeader()
        rows.setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rows.setSectionsClickable(True)
        # A selected column has a cell on every row: highlighting the rows too
        # would light up the whole index for a single column.
        rows.setHighlightSections(False)
        height = self.fontMetrics().height() + self.row_padding
        rows.setMinimumSectionSize(height)
        rows.setDefaultSectionSize(height)

    def setModel(self, model: QAbstractItemModel) -> None:
        """
        Show a model, fit the columns to it and follow its selection.

        Parameters
        ----------
        model : QAbstractItemModel
            The model to show.
        """

        super().setModel(model)

        if self.selectionModel() is not None:
            self.selectionModel().selectionChanged.connect(self._on_selection_changed)

        self.fit_columns()
        self.model_changed.emit()
        self.selection_changed.emit()

    def _on_selection_changed(
        self, selected: QItemSelection, deselected: QItemSelection
    ) -> None:
        """Relay a change of selection."""

        self.selection_changed.emit()

    def fit_columns(self) -> None:
        """Fit the columns to their content, within a readable width."""

        if self.model() is None:
            return

        self.resizeColumnsToContents()
        header = self.horizontalHeader()
        for column in range(header.count()):
            if header.sectionSize(column) > self.max_column_width:
                header.resizeSection(column, self.max_column_width)

    def selected_columns(self) -> List[int]:
        """
        Return the columns holding at least one selected cell.

        The selection is read as ranges rather than cell by cell: a column
        selected from its header of a large table is a single range, and
        listing its cells would take as long as the table has rows.

        Returns
        -------
        list of int
            The logical indices of the columns, sorted.
        """

        if self.selectionModel() is None:
            return []

        columns = set()
        for selection_range in self.selectionModel().selection():
            columns.update(range(selection_range.left(), selection_range.right() + 1))

        return sorted(columns)

    def selected_cell_count(self) -> int:
        """Return the number of selected cells."""

        if self.selectionModel() is None:
            return 0

        return sum(
            r.width() * r.height() for r in self.selectionModel().selection()
        )

    def copy_selection(self) -> str:
        """
        Copy the selected cells to the clipboard as tab separated values.

        The cells are written at full precision, in the order the columns are
        shown, under a line of column names. Cells left out of a ragged
        selection are written empty, so that the pasted block stays aligned.

        Returns
        -------
        str
            The text put on the clipboard, empty when nothing is selected.
        """

        model = self.model()
        if model is None or self.selectionModel() is None:
            return ""

        indexes = self.selectionModel().selectedIndexes()
        if not indexes:
            return ""

        header = self.horizontalHeader()
        rows = sorted({index.row() for index in indexes})
        columns = sorted(
            {index.column() for index in indexes}, key=header.visualIndex
        )
        cells = {(index.row(), index.column()) for index in indexes}

        def value(row: int, column: int) -> str:
            data = model.data(model.index(row, column), Qt.EditRole)
            if data is None:
                data = model.data(model.index(row, column), Qt.DisplayRole)
            return "" if data is None else str(data)

        lines = [
            "\t".join(
                str(model.headerData(column, Qt.Horizontal, Qt.DisplayRole))
                for column in columns
            )
        ]
        for row in rows:
            lines.append(
                "\t".join(
                    value(row, column) if (row, column) in cells else ""
                    for column in columns
                )
            )

        text = "\n".join(lines)
        QApplication.clipboard().setText(text)

        return text

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Copy the selection on the copy shortcut of the platform."""

        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return

        super().keyPressEvent(event)
