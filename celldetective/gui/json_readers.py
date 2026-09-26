"""
The editor of the configuration of an experiment, its ``config.ini``.

The file keeps the format the rest of the software reads: the well labels are
the options of the ``[Labels]`` section, each holding one value per well joined
by commas, and ``[Metadata]`` holds free key/value pairs shared by the whole
experiment. Both are shown the way they are meant, rather than as raw options:
the labels as a table with one row per well and one column per label, the
metadata as a table of keys and values that can grow.
"""

import configparser
import logging
import os
import re
import subprocess
from subprocess import Popen
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QRegularExpression, QStringListModel, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeyEvent, QKeySequence, QRegularExpressionValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCompleter,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from fonticon_mdi6 import MDI6

from celldetective.gui.base.components import (
    CelldetectiveWidget,
    ToolButton,
    generic_message,
    hint_label,
    tool_strip,
)
from celldetective.gui.base.styles import DANGER_COLOR, MUTED_INK, TABLE_STYLE
from celldetective.gui.base.threads import start_tracked
from celldetective.gui.base.utils import center_window
from celldetective.utils.experiment import (
    count_movies_matching_prefix,
    get_movie_prefix_candidates,
    list_movies_per_position,
)

logger = logging.getLogger("celldetective")

LABELS_SECTION = "Labels"
METADATA_SECTION = "Metadata"

# The prefix telling which stack of a position folder is the movie: that one
# field is offered what the experiment holds (see MoviePrefixField).
MOVIE_SECTION = "MovieSettings"
MOVIE_PREFIX_KEY = "movie_prefix"

# The labels the software reads by name (see utils.experiment): they can be
# edited but neither renamed nor removed.
REQUIRED_LABELS = ("cell_types", "antibodies", "concentrations", "pharmaceutical_agents")

# The values of a label are stored on one line, one per well.
LABEL_SEPARATOR = ","

# Characters configparser would read as syntax in an option name.
FORBIDDEN_KEY_CHARACTERS = re.compile(r"[=:\[\],;#]")


def normalize_key(name: str) -> str:
    """
    Turn a name typed by the user into the option name stored in the file.

    configparser lowercases the option names it reads, so the name is shown
    the way it will come back.

    Parameters
    ----------
    name : str
        The name as typed.

    Returns
    -------
    str
        The name lowercased, with its runs of spaces turned into underscores.
    """

    return re.sub(r"\s+", "_", name.strip().lower())


def key_error(key: str, taken: List[str]) -> Optional[str]:
    """
    Tell what is wrong with an option name, if anything.

    Parameters
    ----------
    key : str
        The normalized name.
    taken : list of str
        The names already used in the same section.

    Returns
    -------
    str or None
        A sentence explaining the problem, None when the name can be used.
    """

    if not key:
        return "The name cannot be empty."
    if FORBIDDEN_KEY_CHARACTERS.search(key):
        return f"'{key}' cannot contain any of = : [ ] , ; #."
    if key in taken:
        return f"'{key}' is already used."
    return None


def read_well_labels(config: configparser.ConfigParser, n_wells: int) -> Dict[str, List[str]]:
    """
    Read the labels of the wells, one list of values per label.

    Parameters
    ----------
    config : configparser.ConfigParser
        The parsed configuration.
    n_wells : int
        The number of wells of the experiment.

    Returns
    -------
    dict
        The values of each label, exactly one per well. A label holding another
        number of values is padded or cut, and a missing required label is given
        the index of each well, as a new experiment would.
    """

    labels = {}

    if config.has_section(LABELS_SECTION):
        for key, value in config.items(LABELS_SECTION):
            values = value.split(LABEL_SEPARATOR)
            if len(values) != n_wells:
                logger.warning(
                    f"Label '{key}' holds {len(values)} values for {n_wells} wells."
                )
                values = (values + [""] * n_wells)[:n_wells]
            labels[key] = values

    for key in REQUIRED_LABELS:
        labels.setdefault(key, [str(i) for i in range(n_wells)])

    return labels


class ElidedLabel(QLabel):
    """
    A label cutting its text in the middle rather than widening its window.

    A long experiment path would otherwise set the width of the whole editor.
    The full text stays in the tooltip.
    """

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the label.

        Parameters
        ----------
        text : str
            The full text.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)
        self.full_text = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(1)

    def resizeEvent(self, event) -> None:
        """Cut the text to the new width."""

        super().resizeEvent(event)
        self.setText(
            self.fontMetrics().elidedText(self.full_text, Qt.ElideMiddle, self.width())
        )


class NoSeparatorDelegate(QStyledItemDelegate):
    """Edit the cells of a table without letting a comma be typed in."""

    def createEditor(self, parent, option, index):
        """Create the usual editor, refusing the separator of the labels."""

        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            editor.setValidator(
                QRegularExpressionValidator(
                    QRegularExpression(f"[^{LABEL_SEPARATOR}]*"), editor
                )
            )
        return editor


class EditableTable(QTableWidget):
    """
    A table in the look of the software, filled like a spreadsheet.

    A block copied from a spreadsheet is pasted from the current cell, and
    Delete clears the selected cells, so that the labels of a plate do not have
    to be typed one well at a time.
    """

    def __init__(self, rows: int, columns: int, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the table.

        Parameters
        ----------
        rows : int
            The number of rows.
        columns : int
            The number of columns.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(rows, columns, parent)

        self.setStyleSheet(TABLE_STYLE)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.EditKeyPressed
            | QAbstractItemView.AnyKeyPressed
        )
        self.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def text(self, row: int, column: int) -> str:
        """Return the text of a cell, empty when it was never filled."""

        item = self.item(row, column)
        return "" if item is None else item.text().strip()

    def set_text(self, row: int, column: int, text: str) -> None:
        """Set the text of a cell."""

        self.setItem(row, column, QTableWidgetItem(text))

    def paste(self, text: str) -> None:
        """
        Paste tab separated values from the current cell.

        Parameters
        ----------
        text : str
            The block to paste, one line per row. What falls past the edges of
            the table is dropped.
        """

        start_row = max(self.currentRow(), 0)
        start_column = max(self.currentColumn(), 0)

        for i, line in enumerate(text.rstrip("\r\n").splitlines()):
            for j, cell in enumerate(line.split("\t")):
                row, column = start_row + i, start_column + j
                if row < self.rowCount() and column < self.columnCount():
                    self.set_text(row, column, cell.strip())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Paste on the paste shortcut, clear the selection on Delete."""

        if event.matches(QKeySequence.Paste):
            self.paste(QApplication.clipboard().text())
            event.accept()
            return

        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.state() != QAbstractItemView.EditingState:
            for index in self.selectedIndexes():
                self.set_text(index.row(), index.column(), "")
            event.accept()
            return

        super().keyPressEvent(event)


class MovieScanThread(QThread):
    """
    Read the stacks of an experiment and the prefixes they offer.

    Scanning the movie folder of every position takes a moment on a large
    experiment or a network share, so it is done off the GUI thread.
    """

    # The stacks of each position (None when they could not be read), the
    # prefixes they offer, and what went wrong, if anything.
    scanned = pyqtSignal(object, list, str)

    def __init__(self, exp_dir: str) -> None:
        """
        Prepare the scan of an experiment.

        Parameters
        ----------
        exp_dir : str
            The experiment folder to scan.
        """

        super().__init__()
        self.exp_dir = exp_dir

    def run(self) -> None:
        """Scan the experiment and hand the result over."""

        try:
            movies = list_movies_per_position(self.exp_dir)
            candidates = get_movie_prefix_candidates(movies)
        except Exception:
            logger.exception("Could not list the stacks of the experiment.")
            self.scanned.emit(
                None, [], "The stacks of the experiment could not be read: see the log."
            )
            return

        self.scanned.emit(movies, candidates, "")


class MoviePrefixField(CelldetectiveWidget):
    """
    The movie prefix field, dressed with the prefixes the experiment holds.

    The names of the stacks sitting in the movie folders are cut into the
    prefixes that would select them and offered as completions, from the field
    or from the button next to it. A line under the field tells what the
    prefix currently typed matches, so that a prefix leaving positions without
    a movie is seen here rather than at the first segmentation.
    """

    def __init__(
        self, field: QLineEdit, exp_dir: str, parent: Optional[QWidget] = None
    ) -> None:
        """
        Dress a prefix field with what the experiment holds.

        Parameters
        ----------
        field : QLineEdit
            The field holding the prefix, kept by the editor that saves it.
        exp_dir : str
            The experiment folder whose stacks are proposed.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.field = field

        # The stacks of the experiment, None until the scan started on opening
        # is over, so that the prefix stored is checked without being edited.
        self.movies_per_position = None
        self.scan_error = ""

        field.setPlaceholderText("any stack of the movie folder")

        self.model = QStringListModel(self)
        self.completer = QCompleter(self.model, field)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        field.setCompleter(self.completer)

        self.suggest_btn = ToolButton(
            MDI6.text_search, "Show the prefixes of the stacks of the experiment."
        )
        self.suggest_btn.clicked.connect(self.show_suggestions)

        self.hint = hint_label("")
        self.hint.setTextFormat(Qt.PlainText)
        self._warning = False
        field.textChanged.connect(self.update_hint)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        row.addLayout(tool_strip(self.suggest_btn))

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addLayout(row)
        box.addWidget(self.hint)

        self.update_hint()
        self.scan_thread = MovieScanThread(exp_dir)
        self.scan_thread.scanned.connect(self._on_scanned)
        start_tracked(self.scan_thread)

    def _on_scanned(self, movies: Optional[dict], candidates: list, error: str) -> None:
        """Keep what the scan read, and check the prefix against it."""

        self.movies_per_position = movies
        self.scan_error = error
        self.model.setStringList(candidates)
        self.update_hint()

    def show_suggestions(self) -> None:
        """Open the list of the prefixes the experiment holds."""

        self.field.setFocus()
        # Every prefix, whatever the field already holds.
        self.completer.setCompletionPrefix("")
        self.completer.complete()

    def update_hint(self) -> None:
        """Tell what the prefix currently typed matches in the experiment."""

        self._show_hint(*self._describe_prefix())

    def _describe_prefix(self) -> Tuple[str, bool]:
        """Return the line telling what the prefix matches, and if it warns."""

        if self.scan_error:
            return self.scan_error, True
        if self.movies_per_position is None:
            return "Reading the stacks of the experiment…", False

        total = len(self.movies_per_position)
        if total == 0:
            return "No position found in the experiment folder.", True
        if not any(self.movies_per_position.values()):
            return (
                f"No stack in the movie folder of any of the {total} positions.",
                True,
            )

        positions, stacks = count_movies_matching_prefix(
            self.movies_per_position, self.field.text().strip()
        )

        if positions == 0:
            return f"No stack of the {total} positions matches this prefix.", True
        if positions < total:
            return (
                f"{total - positions} of the {total} positions hold no matching stack.",
                True,
            )
        if stacks > positions:
            return (
                f"{stacks} stacks over {total} positions: which one of a position "
                "is loaded is left to chance.",
                True,
            )
        return f"One stack in each of the {total} positions.", False

    def _show_hint(self, text: str, warning: bool) -> None:
        """Write the line of feedback under the field."""

        # Restyling repolishes the label: only when the ink changes.
        if warning != self._warning:
            self._warning = warning
            color = DANGER_COLOR if warning else MUTED_INK
            self.hint.setStyleSheet(f"color: {color};")
        self.hint.setText(text)


class ConfigEditor(CelldetectiveWidget):
    """
    Edit the configuration of an experiment.

    The settings of the movies, the channels and the populations are forms, one
    per section. The well labels are a table, one row per well and one column
    per label, and the metadata a table of keys and values.
    """

    def __init__(self, parent_window: QMainWindow) -> None:
        """
        Load and edit the experiment config.

        Parameters
        ----------
        parent_window : QMainWindow
            The parent window containing the experiment configuration path.
        """

        super().__init__()

        self.parent_window = parent_window
        self.config_path = self.parent_window.exp_config

        self.setWindowTitle("Configuration")

        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(self.config_path)

        self.well_names = self._well_names()
        self.fields = {}

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        self.path_label = ElidedLabel(os.path.realpath(self.config_path))
        self.path_label.setStyleSheet(f"color: {MUTED_INK};")
        header.addWidget(self.path_label, 1)
        self.edit_config_btn = ToolButton(
            MDI6.file_cog, "Open the file in a text editor."
        )
        self.edit_config_btn.clicked.connect(self.edit_in_text_editor)
        header.addLayout(tool_strip(self.edit_config_btn))
        layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_labels_tab(), "Well labels")
        self.tabs.addTab(self._build_metadata_tab(), "Metadata")
        layout.addWidget(self.tabs, 1)
        # A page losing the focus hands it to the next widget in the chain, the
        # first tool button of the new page, which then shows its focus ring:
        # give it to what the page is for instead.
        self.tabs.currentChanged.connect(self._focus_page)

        buttons = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setStyleSheet(self.button_style_sheet_2)
        self.cancel_button.clicked.connect(self.close)
        buttons.addWidget(self.cancel_button, 50)
        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet(self.button_primary)
        self.save_button.setShortcut(QKeySequence.Save)
        self.save_button.setToolTip("Save the configuration (Ctrl+S).")
        self.save_button.clicked.connect(self.save_config)
        buttons.addWidget(self.save_button, 50)
        layout.addLayout(buttons)

        # As wide as the control panel it is opened from (see its scroll area).
        self.resize(460, 640)
        center_window(self)

        # Otherwise the window hands the initial focus to the first button,
        # which then shows its focus ring on opening.
        self._focus_page(self.tabs.currentIndex())

    def _focus_page(self, index: int) -> None:
        """Give the focus to the main control of a tab."""

        if index == 0 and self.fields:
            next(iter(self.fields.values())).setFocus()
        elif index == 1:
            self.labels_table.setFocus()
        elif index == 2:
            self.metadata_table.setFocus()

    def _well_names(self) -> List[str]:
        """Return the names of the wells, in the order of the labels."""

        wells = getattr(self.parent_window, "wells", None)
        if wells is not None and len(wells) > 0:
            return [os.path.basename(os.path.normpath(str(w))) for w in wells]

        # No wells known: trust the longest label.
        n_wells = 0
        if self.config.has_section(LABELS_SECTION):
            n_wells = max(
                (len(v.split(LABEL_SEPARATOR)) for _, v in self.config.items(LABELS_SECTION)),
                default=0,
            )
        return [f"W{i + 1}" for i in range(n_wells)]

    def _build_settings_tab(self) -> QWidget:
        """Build one form per plain section of the file."""

        content = QWidget()
        box = QVBoxLayout(content)
        # Stays None when the file holds no movie prefix.
        self.prefix_widget = None

        for section in self.config.sections():
            if section in (LABELS_SECTION, METADATA_SECTION):
                continue

            title = QLabel(section)
            title.setStyleSheet(self.block_title)
            box.addWidget(title)

            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setContentsMargins(10, 0, 0, 10)
            for key, value in self.config.items(section):
                field = QLineEdit(value)
                self.fields[(section, key)] = field
                if section == MOVIE_SECTION and key == MOVIE_PREFIX_KEY:
                    self.prefix_widget = MoviePrefixField(
                        field, self.parent_window.exp_dir
                    )
                    form.addRow(key, self.prefix_widget)
                else:
                    form.addRow(key, field)
            box.addLayout(form)

        box.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_labels_tab(self) -> QWidget:
        """Build the table of the well labels."""

        labels = read_well_labels(self.config, len(self.well_names))
        self.label_keys = list(labels.keys())

        self.labels_table = EditableTable(len(self.well_names), len(self.label_keys))
        self.labels_table.setItemDelegate(NoSeparatorDelegate(self.labels_table))
        self.labels_table.setVerticalHeaderLabels(self.well_names)
        self.labels_table.setHorizontalHeaderLabels(self.label_keys)
        self.labels_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.labels_table.horizontalHeader().setDefaultSectionSize(140)
        for column, key in enumerate(self.label_keys):
            for row, value in enumerate(labels[key]):
                self.labels_table.set_text(row, column, value)
        # Wide enough for the name of each label, which the default cuts.
        self.labels_table.resizeColumnsToContents()
        header = self.labels_table.horizontalHeader()
        for column in range(header.count()):
            header.resizeSection(column, max(header.sectionSize(column) + 16, 120))

        self.labels_table.horizontalHeader().sectionDoubleClicked.connect(
            self.rename_label
        )
        self.labels_table.currentCellChanged.connect(
            lambda *_: self._update_label_tools()
        )

        self.add_label_btn = ToolButton(MDI6.table_column_plus_after, "Add a label.")
        self.add_label_btn.clicked.connect(lambda: self.add_label())
        self.rename_label_btn = ToolButton(MDI6.pencil, "Rename the selected label.")
        self.rename_label_btn.clicked.connect(
            lambda: self.rename_label(self.labels_table.currentColumn())
        )
        self.remove_label_btn = ToolButton(
            MDI6.table_column_remove, "Remove the selected label.", hover_color=DANGER_COLOR
        )
        self.remove_label_btn.clicked.connect(
            lambda: self.remove_label(self.labels_table.currentColumn())
        )

        page = QWidget()
        box = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(
            hint_label(
                "One row per well. Paste a block from a spreadsheet with Ctrl+V; "
                "values cannot contain commas."
            ),
            1,
        )
        top.addLayout(
            tool_strip(self.add_label_btn, self.rename_label_btn, self.remove_label_btn)
        )
        box.addLayout(top)
        box.addWidget(self.labels_table)

        self._update_label_tools()
        return page

    def _build_metadata_tab(self) -> QWidget:
        """Build the table of the metadata."""

        items = (
            self.config.items(METADATA_SECTION)
            if self.config.has_section(METADATA_SECTION)
            else []
        )

        self.metadata_table = EditableTable(len(items), 2)
        self.metadata_table.setHorizontalHeaderLabels(["key", "value"])
        self.metadata_table.verticalHeader().hide()
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.metadata_table.horizontalHeader().resizeSection(0, 180)
        for row, (key, value) in enumerate(items):
            self.metadata_table.set_text(row, 0, key)
            self.metadata_table.set_text(row, 1, value)

        self.metadata_table.currentCellChanged.connect(
            lambda *_: self.remove_entry_btn.setEnabled(self.metadata_table.currentRow() >= 0)
        )

        self.add_entry_btn = ToolButton(MDI6.table_row_plus_after, "Add an entry.")
        self.add_entry_btn.clicked.connect(self.add_metadata_entry)
        self.remove_entry_btn = ToolButton(
            MDI6.table_row_remove, "Remove the selected entry.", hover_color=DANGER_COLOR
        )
        self.remove_entry_btn.clicked.connect(self.remove_metadata_entry)
        self.remove_entry_btn.setEnabled(False)

        page = QWidget()
        box = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(
            hint_label(
                "Values shared by every well of the experiment, e.g. date = 2026-09-15. "
                "Each key becomes a column of the measurement tables."
            ),
            1,
        )
        top.addLayout(tool_strip(self.add_entry_btn, self.remove_entry_btn))
        box.addLayout(top)
        box.addWidget(self.metadata_table)
        return page

    def _update_label_tools(self) -> None:
        """Offer renaming and removal only for a label the user added."""

        column = self.labels_table.currentColumn()
        custom = 0 <= column < len(self.label_keys) and self.label_keys[column] not in REQUIRED_LABELS
        self.rename_label_btn.setEnabled(custom)
        self.remove_label_btn.setEnabled(custom)

    def _ask_label_name(self, title: str, current: str = "") -> Optional[str]:
        """Ask for the name of a label, None when the user cancels."""

        name, ok = QInputDialog.getText(self, title, "Label name:", text=current)
        return normalize_key(name) if ok else None

    def add_label(self, name: Optional[str] = None) -> bool:
        """
        Add a label column, empty for every well.

        Parameters
        ----------
        name : str, optional
            The name of the label. Asked for when not given.

        Returns
        -------
        bool
            True if the label was added.
        """

        if name is None:
            name = self._ask_label_name("Add a label")
            if name is None:
                return False
        name = normalize_key(name)

        error = key_error(name, self.label_keys)
        if error:
            generic_message(error, "warning")
            return False

        column = len(self.label_keys)
        self.label_keys.append(name)
        self.labels_table.insertColumn(column)
        self.labels_table.setHorizontalHeaderItem(column, QTableWidgetItem(name))
        self.labels_table.setCurrentCell(0, column)
        return True

    def rename_label(self, column: int, name: Optional[str] = None) -> bool:
        """
        Rename a label the user added.

        Parameters
        ----------
        column : int
            The column of the label.
        name : str, optional
            The new name. Asked for when not given.

        Returns
        -------
        bool
            True if the label was renamed.
        """

        if not 0 <= column < len(self.label_keys) or self.label_keys[column] in REQUIRED_LABELS:
            return False

        if name is None:
            name = self._ask_label_name("Rename the label", self.label_keys[column])
            if name is None:
                return False
        name = normalize_key(name)

        others = self.label_keys[:column] + self.label_keys[column + 1 :]
        error = key_error(name, others)
        if error:
            generic_message(error, "warning")
            return False

        self.label_keys[column] = name
        self.labels_table.setHorizontalHeaderItem(column, QTableWidgetItem(name))
        return True

    def remove_label(self, column: int) -> bool:
        """
        Remove a label the user added.

        Parameters
        ----------
        column : int
            The column of the label.

        Returns
        -------
        bool
            True if the label was removed.
        """

        if not 0 <= column < len(self.label_keys) or self.label_keys[column] in REQUIRED_LABELS:
            return False

        del self.label_keys[column]
        self.labels_table.removeColumn(column)
        self._update_label_tools()
        return True

    def add_metadata_entry(self) -> None:
        """Add an empty entry and start typing its key."""

        row = self.metadata_table.rowCount()
        self.metadata_table.insertRow(row)
        self.metadata_table.set_text(row, 0, "")
        self.metadata_table.set_text(row, 1, "")
        self.metadata_table.setCurrentCell(row, 0)
        self.metadata_table.editItem(self.metadata_table.item(row, 0))

    def remove_metadata_entry(self) -> None:
        """Remove the selected entries."""

        rows = sorted({index.row() for index in self.metadata_table.selectedIndexes()})
        if not rows and self.metadata_table.currentRow() >= 0:
            rows = [self.metadata_table.currentRow()]
        for row in reversed(rows):
            self.metadata_table.removeRow(row)

    def edit_in_text_editor(self):
        """
        Open the configuration file in the system's default text editor.
        """
        path = self.config_path
        try:
            Popen(["explorer", os.path.realpath(path)])
        except Exception:
            try:
                subprocess.run(["xdg-open", path], check=False)
            except Exception:
                return None

    def build_config(self) -> configparser.ConfigParser:
        """
        Gather what the editor shows into a configuration, without writing it.

        Returns
        -------
        configparser.ConfigParser
            The configuration, its sections in the order of the file.

        Raises
        ------
        ValueError
            If a label value holds a comma, or a metadata key is missing,
            invalid or repeated.
        """

        config = configparser.ConfigParser(interpolation=None)

        sections = self.config.sections()
        for section in (LABELS_SECTION, METADATA_SECTION):
            if section not in sections:
                sections.append(section)

        for section in sections:
            config.add_section(section)

            if section == LABELS_SECTION:
                for column, key in enumerate(self.label_keys):
                    values = [
                        self.labels_table.text(row, column)
                        for row in range(self.labels_table.rowCount())
                    ]
                    for row, value in enumerate(values):
                        if LABEL_SEPARATOR in value:
                            raise ValueError(
                                f"The value of '{key}' for {self.well_names[row]} "
                                f"contains a comma: '{value}'."
                            )
                    config.set(section, key, LABEL_SEPARATOR.join(values))

            elif section == METADATA_SECTION:
                keys = []
                for row in range(self.metadata_table.rowCount()):
                    key = normalize_key(self.metadata_table.text(row, 0))
                    value = self.metadata_table.text(row, 1)
                    if not key and not value:
                        continue
                    error = key_error(key, keys)
                    if error:
                        raise ValueError(f"Metadata entry {row + 1}: {error}")
                    keys.append(key)
                    config.set(section, key, value)

            else:
                for key, _ in self.config.items(section):
                    config.set(section, key, self.fields[(section, key)].text().strip())

        return config

    def save_config(self) -> bool:
        """
        Save the edited configuration to the file and reload it in the parent window.

        Returns
        -------
        bool
            True if the configuration was saved.
        """

        try:
            config = self.build_config()
        except ValueError as e:
            generic_message(str(e), "warning")
            return False

        # Written next to the file first, so that a failed write cannot leave
        # the experiment with half a configuration.
        temporary = self.config_path + ".tmp"
        with open(temporary, "w") as f:
            config.write(f)
        os.replace(temporary, self.config_path)

        try:
            self.parent_window.load_configuration()
        except Exception as e:
            logger.exception("Could not reload the configuration.")
            generic_message(
                f"The configuration was saved but could not be read back: {e}",
                "critical",
            )
            return False

        self.close()
        return True
