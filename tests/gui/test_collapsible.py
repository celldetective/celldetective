"""
Unit tests for the collapsible blocks in celldetective.gui.base.collapsible.

Covers the opening and closing animation of :class:`CollapsibleFrame`, and in
particular that the content keeps its size for the whole of it.
"""

import logging
import pytest
from PyQt5.QtCore import Qt, QEventLoop, QTimer
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

from celldetective.gui.base.collapsible import CollapsibleFrame, UNCONSTRAINED


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def build_card(qtbot, entries=4):
    """
    Return an open card, shown, with a few buttons as content.

    Parameters
    ----------
    qtbot : QtBot
        The bot of the test, holding the widget for its lifetime.
    entries : int, optional
        How many buttons to put in the content.

    Returns
    -------
    tuple
        The (host, card, content) widgets.
    """

    host = QWidget()
    qtbot.addWidget(host)
    box = QVBoxLayout(host)

    card = CollapsibleFrame("Section")
    content = QWidget()
    layout = QVBoxLayout(content)
    for index in range(entries):
        layout.addWidget(QPushButton(f"button {index}"))

    card.set_content(content)
    box.addWidget(card)
    box.addStretch(1)

    host.resize(320, 400)
    host.show()
    qtbot.waitExposed(host)

    return host, card, content


def sample_content_height(card, content, expanded, period=10):
    """
    Toggle the card and record the height of its content while it animates.

    Parameters
    ----------
    card : CollapsibleFrame
        The card to open or close.
    content : QWidget
        The content of the card.
    expanded : bool
        True to open the card, False to close it.
    period : int, optional
        The time between two samples, in milliseconds.

    Returns
    -------
    list
        The heights measured while the animation ran.
    """

    heights = []
    loop = QEventLoop()

    timer = QTimer()
    timer.timeout.connect(lambda: heights.append(content.height()))
    timer.start(period)

    card.set_expanded(expanded, animate=True)
    QTimer.singleShot(card.duration * 3, loop.quit)
    loop.exec_()
    timer.stop()

    return heights


class TestCollapseAnimation:
    """Tests for the animation opening and closing a card."""

    def test_content_keeps_its_size_while_closing(self, qtbot):
        """
        The content is clipped by the shrinking card, never squeezed by it.

        Left free to follow the card, it was flattened into a band a few pixels
        tall on the way down, which read as a line flashing across the card
        just before it closed.
        """

        _, card, content = build_card(qtbot)
        card.set_expanded(True, animate=False)
        qtbot.wait(20)

        full = content.height()
        assert full > 0

        heights = sample_content_height(card, content, False)

        assert heights
        assert min(heights) == full

    def test_content_keeps_its_size_while_opening(self, qtbot):
        """The content is at its full size from the first frame of an opening."""

        _, card, content = build_card(qtbot)

        heights = sample_content_height(card, content, True)

        assert heights
        assert min(heights) == max(heights)

    def test_constraints_are_dropped_once_open(self, qtbot):
        """Neither constraint of the animation outlives it."""

        _, card, content = build_card(qtbot)

        with qtbot.waitSignal(card.animation_finished, timeout=2000):
            card.set_expanded(True, animate=True)

        assert content.isVisible()
        assert content.minimumHeight() == 0
        assert card.maximumHeight() == UNCONSTRAINED

    def test_content_is_hidden_once_closed(self, qtbot):
        """A folded content is hidden, so it takes no room and no clicks."""

        _, card, content = build_card(qtbot)
        card.set_expanded(True, animate=False)
        qtbot.wait(20)

        with qtbot.waitSignal(card.animation_finished, timeout=2000):
            card.set_expanded(False, animate=True)

        assert not content.isVisible()
        assert content.minimumHeight() == 0
        assert card.maximumHeight() == UNCONSTRAINED

    def test_settle_stops_a_running_animation(self, qtbot):
        """A card going away jumps to the state it was heading for."""

        _, card, content = build_card(qtbot)
        card.set_expanded(True, animate=True)
        card.settle()

        assert card.animation.state() != card.animation.Running
        assert content.isVisible()
        assert content.minimumHeight() == 0
