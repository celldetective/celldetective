"""
The decision helpers of the software.

Several panels offer a helper that walks the user through a few questions and
ends on a suggestion: which segmentation strategy to use, how to preprocess,
which features to track on. The questions live in the JSON trees of
``celldetective/gui/help``, a nested dict whose keys are the questions and
whose "yes"/"no" branches lead either to another question or to a suggestion.

They used to be asked one modal message box at a time, with no way back and no
sense of where one was. :class:`HelpPanel` asks them in a single window
instead, keeping the answers in view, and ends on the suggestion with a link to
the matching page of the documentation.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from PyQt5.QtCore import QSize, Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective import get_package_location
from celldetective.gui.base.styles import (
    ACCENT_SOFT,
    CELLDETECTIVE_BLUE,
    INK_COLOR,
    SURFACE_BORDER,
    Styles,
    TOOL_BUTTON_SIZE,
    TOOL_ICON_SIZE,
    TOOL_IDLE_COLOR,
    button_style,
)

logger = logging.getLogger("celldetective")

HELP_DIR = os.path.join(get_package_location(), "gui", "help")

# The color a help button rests in: present, but quieter than the actions it
# sits next to. It takes the accent color under the mouse. The tone is the one
# every tool button rests in, the helper being one of them.
HELP_IDLE_COLOR = TOOL_IDLE_COLOR


def help_tree(name: str) -> Dict[str, Any]:
    """
    Load one of the question trees of the software.

    Parameters
    ----------
    name : str
        File name of the tree, e.g. "tracking.json".

    Returns
    -------
    dict
        The question tree.
    """

    with open(os.path.join(HELP_DIR, name)) as f:
        return json.load(f)


def tree_depth(tree: Any) -> int:
    """
    Return the longest run of questions left in a tree.

    Parameters
    ----------
    tree : dict or str
        A question tree, or a suggestion.

    Returns
    -------
    int
        The number of questions on its longest branch.
    """

    if not isinstance(tree, dict):
        return 0

    branches = list(tree.values())[0]

    return 1 + max(tree_depth(branch) for branch in branches.values())


class HelpButton(QPushButton):
    """
    The button opening a helper.

    Quieter than the actions around it until the mouse reaches it, where it
    takes the accent color.
    """

    size = TOOL_BUTTON_SIZE
    icon_size = TOOL_ICON_SIZE

    def __init__(self, tooltip: str, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the button.

        Parameters
        ----------
        tooltip : str
            What the helper is about, as a sentence: the tooltip of the button.
        parent : QWidget, optional
            The parent widget.
        """

        super().__init__(parent)

        self.setToolTip(tooltip)
        self.setFixedSize(self.size, self.size)
        self.setIconSize(QSize(self.icon_size, self.icon_size))
        # The `tool` role, like the cogs and the eyes it shares a strip with:
        # the helper is one of them, not a button of its own kind.
        self.setStyleSheet(button_style("tool"))
        self._paint_icon(HELP_IDLE_COLOR)

        policy = self.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)

    def _paint_icon(self, color: str) -> None:
        """
        Draw the question mark in the given color.

        Parameters
        ----------
        color : str
            The color of the icon.
        """

        self.setIcon(icon(MDI6.help_circle_outline, color=color))

    def enterEvent(self, event) -> None:
        """Take the accent color under the mouse."""

        self._paint_icon(CELLDETECTIVE_BLUE)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        """Go back to resting color."""

        self._paint_icon(HELP_IDLE_COLOR)
        super().leaveEvent(event)


class HelpPanel(QWidget):
    """
    A window walking the user through a question tree.

    One question at a time, the answers given so far kept in view, a way back,
    and a suggestion at the end.
    """

    def __init__(
        self,
        tree: Dict[str, Any],
        title: str,
        docs_url: Optional[str] = None,
        phrasing: Optional[str] = "{suggestion}",
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the panel.

        Parameters
        ----------
        tree : dict
            The question tree to walk.
        title : str
            What the helper is about, shown as the title of the window.
        docs_url : str, optional
            The page of the documentation the suggestion points to.
        phrasing : str, optional
            How to read the suggestion out, as a format string taking
            ``suggestion``.
        parent : QWidget, optional
            The window the helper was opened from.
        """

        super().__init__()

        self.styles = Styles()
        self.full_tree = tree
        self.tree = tree
        self.docs_url = docs_url
        self.phrasing = phrasing
        self.trail = []

        self.setWindowTitle("Helper")
        self.setWindowIcon(self.styles.celldetective_icon)
        self.setMinimumWidth(470)

        self._build(title)
        self.restart()

    def _build(self, title: str) -> None:
        """
        Lay the window out.

        Parameters
        ----------
        title : str
            The title of the helper.
        """

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)

        badge = QLabel()
        badge.setPixmap(icon(MDI6.lifebuoy, color=CELLDETECTIVE_BLUE).pixmap(22, 22))
        head.addWidget(badge)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {CELLDETECTIVE_BLUE};"
        )
        head.addWidget(self.title_lbl)
        head.addStretch(1)

        self.step_lbl = QLabel("")
        self.step_lbl.setStyleSheet(f"color: {HELP_IDLE_COLOR}; font-size: 10px;")
        head.addWidget(self.step_lbl)
        outer.addLayout(head)

        self.trail_lbl = QLabel("")
        self.trail_lbl.setWordWrap(True)
        self.trail_lbl.setStyleSheet(f"color: {HELP_IDLE_COLOR}; font-size: 10px;")
        outer.addWidget(self.trail_lbl)

        self.card = QFrame()
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(16, 14, 16, 14)

        self.text_lbl = QLabel("")
        self.text_lbl.setWordWrap(True)
        self.text_lbl.setTextFormat(Qt.RichText)
        self.text_lbl.setOpenExternalLinks(True)
        self.text_lbl.setStyleSheet(
            f"font-size: 13px; color: {INK_COLOR}; border: none; background: transparent;"
        )
        card_layout.addWidget(self.text_lbl)
        outer.addWidget(self.card)

        row = QHBoxLayout()
        row.setSpacing(8)

        self.back_btn = QPushButton("Back")
        self.back_btn.setIcon(icon(MDI6.arrow_left, color=INK_COLOR))
        self.back_btn.setStyleSheet(self.styles.button_ghost)
        self.back_btn.clicked.connect(self.go_back)
        row.addWidget(self.back_btn)
        row.addStretch(1)

        self.no_btn = QPushButton("No")
        self.no_btn.setMinimumWidth(92)
        self.no_btn.setStyleSheet(self.styles.button_secondary)
        self.no_btn.clicked.connect(lambda: self.answer("no"))
        row.addWidget(self.no_btn)

        self.yes_btn = QPushButton("Yes")
        self.yes_btn.setMinimumWidth(92)
        self.yes_btn.setStyleSheet(self.styles.button_primary)
        self.yes_btn.clicked.connect(lambda: self.answer("yes"))
        row.addWidget(self.yes_btn)

        self.docs_btn = QPushButton("Read the tutorial")
        self.docs_btn.setIcon(icon(MDI6.book_open_variant, color=CELLDETECTIVE_BLUE))
        self.docs_btn.setStyleSheet(self.styles.button_secondary)
        self.docs_btn.clicked.connect(self.open_docs)
        self.docs_btn.hide()
        row.addWidget(self.docs_btn)

        self.restart_btn = QPushButton("Start over")
        self.restart_btn.setIcon(icon(MDI6.restart, color=INK_COLOR))
        self.restart_btn.setStyleSheet(self.styles.button_ghost)
        self.restart_btn.clicked.connect(self.restart)
        self.restart_btn.hide()
        row.addWidget(self.restart_btn)

        outer.addLayout(row)

    @staticmethod
    def question_of(tree: Dict[str, Any]) -> str:
        """
        Return the question a tree opens on.

        Parameters
        ----------
        tree : dict
            The tree to read.

        Returns
        -------
        str
            Its question.
        """

        return list(tree.keys())[0]

    def restart(self) -> None:
        """Go back to the first question."""

        self.tree = self.full_tree
        self.trail = []
        self.show_question()

    def _set_card(self, background: str, border: str) -> None:
        """
        Color the card holding the question or the suggestion.

        Parameters
        ----------
        background : str
            The background color.
        border : str
            The border color.
        """

        self.card.setStyleSheet(
            f"background: {background}; border: 1px solid {border};"
            " border-radius: 8px;"
        )

    def show_question(self) -> None:
        """Show the question the walk has reached."""

        question = self.question_of(self.tree)
        self.text_lbl.setText(question)

        left = tree_depth(self.tree)
        self.step_lbl.setText(
            f"question {len(self.trail) + 1} · {left} left at most"
            if left > 1
            else "last question"
        )
        self.trail_lbl.setText(self._trail_text())

        self.back_btn.setEnabled(bool(self.trail))
        self.yes_btn.show()
        self.no_btn.show()
        self.docs_btn.hide()
        self.restart_btn.hide()
        self._set_card("#FAFCFF", SURFACE_BORDER)
        self._fit()

    def _fit(self) -> None:
        """
        Give the window the height the step it shows asks for.

        Questions and suggestions are of very different lengths, and a window
        sized for the first one would cut the others off.
        """

        self.layout().activate()
        self.setMinimumHeight(self.sizeHint().height())
        self.resize(self.width(), self.sizeHint().height())

    def _trail_text(self) -> str:
        """Return the answers given so far, in one line."""

        if not self.trail:
            return " "

        return "   ·   ".join(
            f"{question[:34]}{'…' if len(question) > 34 else ''} <b>{answer}</b>"
            for question, answer in self.trail
        )

    def answer(self, value: str) -> None:
        """
        Take an answer and move on.

        Parameters
        ----------
        value : str
            Either "yes" or "no".
        """

        question = self.question_of(self.tree)
        self.trail.append((question, value))
        branch = self.tree[question][value]

        if isinstance(branch, dict):
            self.tree = branch
            self.show_question()
        else:
            self.show_suggestion(branch)

    def go_back(self) -> None:
        """Undo the last answer."""

        if not self.trail:
            return

        answers = [answer for _, answer in self.trail[:-1]]
        self.tree = self.full_tree
        self.trail = []
        self.show_question()
        for answer in answers:
            self.answer(answer)

    def show_suggestion(self, suggestion: str) -> None:
        """
        Show what the walk has led to.

        Parameters
        ----------
        suggestion : str
            The suggestion the tree ended on.
        """

        logger.info(f"Help suggestion: {suggestion}")

        text = self.phrasing.format(suggestion=suggestion)
        self.text_lbl.setText(f"<b>Suggestion</b><br><br>{text}")
        self.step_lbl.setText(f"{len(self.trail)} questions answered")
        self.trail_lbl.setText(self._trail_text())

        self.yes_btn.hide()
        self.no_btn.hide()
        self.docs_btn.setVisible(self.docs_url is not None)
        self.restart_btn.show()
        self.back_btn.setEnabled(True)
        self._set_card(ACCENT_SOFT, "#C9DCF3")
        self._fit()

    def open_docs(self) -> None:
        """Open the page of the documentation the suggestion points to."""

        if self.docs_url is not None:
            QDesktopServices.openUrl(QUrl(self.docs_url))


class HelpMenu(QWidget):
    """
    A small window offering the several helpers of a panel.

    Picking one closes the menu and opens that helper.
    """

    def __init__(self, title: str, entries: list) -> None:
        """
        Initialize the menu.

        Parameters
        ----------
        title : str
            What the helpers are about.
        entries : list
            The helpers, as (label, callback) pairs.
        """

        super().__init__()

        self.styles = Styles()
        self.setWindowTitle("Helper")
        self.setWindowIcon(self.styles.celldetective_icon)
        self.setMinimumWidth(380)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)
        badge = QLabel()
        badge.setPixmap(icon(MDI6.lifebuoy, color=CELLDETECTIVE_BLUE).pixmap(22, 22))
        head.addWidget(badge)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {CELLDETECTIVE_BLUE};"
        )
        head.addWidget(title_lbl)
        head.addStretch(1)
        outer.addLayout(head)

        for label, callback in entries:
            button = QPushButton(label)
            button.setStyleSheet(self.styles.button_secondary_plain)
            button.setIcon(icon(MDI6.chevron_right, color=CELLDETECTIVE_BLUE))
            button.clicked.connect(self._picker(callback))
            outer.addWidget(button)

    def _picker(self, callback):
        """
        Return the slot running a helper and closing the menu.

        Parameters
        ----------
        callback : callable
            The helper to run.
        """

        def pick():
            self.close()
            callback()

        return pick


def open_help_menu(
    title: str, entries: list, parent: Optional[QWidget] = None
) -> HelpMenu:
    """
    Offer the several helpers of a panel.

    Parameters
    ----------
    title : str
        What the helpers are about.
    entries : list
        The helpers, as (label, callback) pairs.
    parent : QWidget, optional
        The window the menu is opened from. The menu is kept alive on it.

    Returns
    -------
    HelpMenu
        The menu.
    """

    menu = HelpMenu(title, entries)

    if parent is not None:
        parent._help_menu = menu

    from celldetective.gui.base.utils import center_window

    menu.show()
    center_window(menu)

    return menu


def open_help(
    name: str,
    title: str,
    docs_url: Optional[str] = None,
    phrasing: Optional[str] = "{suggestion}",
    parent: Optional[QWidget] = None,
) -> Optional[HelpPanel]:
    """
    Open the helper of a question tree.

    Parameters
    ----------
    name : str
        File name of the tree in ``celldetective/gui/help``.
    title : str
        What the helper is about.
    docs_url : str, optional
        The page of the documentation the suggestion points to.
    phrasing : str, optional
        How to read the suggestion out, as a format string taking
        ``suggestion``.
    parent : QWidget, optional
        The window the helper is opened from. The panel is kept alive on it.

    Returns
    -------
    HelpPanel or None
        The panel, or None if the tree could not be read.
    """

    try:
        tree = help_tree(name)
    except (OSError, ValueError) as e:
        logger.error(f"The help tree '{name}' could not be read: {e}")
        return None

    panel = HelpPanel(tree, title, docs_url=docs_url, phrasing=phrasing, parent=parent)

    if parent is not None:
        # Kept on the caller, so that the window is not collected on return.
        parent._help_panel = panel

    from celldetective.gui.base.utils import center_window

    panel.show()
    center_window(panel)

    return panel
