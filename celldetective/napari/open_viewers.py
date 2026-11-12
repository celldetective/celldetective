"""
Keep a single napari viewer open on the positions at a time.

Two viewers on positions can write over each other's masks or tracks: saving the
masks in one while the other still shows, and later saves, the old ones. Opening
a position viewer is therefore refused while another one is open. The viewer for
correcting a segmentation annotation works on a file outside the positions, and
is not concerned.

This module does not import napari, so the GUI can check before loading anything.
"""

import os
import weakref
from typing import NamedTuple, Optional, Tuple

from PyQt5.QtWidgets import QMessageBox, QWidget

from celldetective import get_logger
from celldetective.gui.base.utils import is_alive

logger = get_logger()


class OpenViewer(NamedTuple):
    """A position viewer that is open."""

    window: QWidget
    position: str
    population: str
    kind: str


# The open viewer, as a weak reference to its window and what it shows: the
# window holds the viewer and, through its layers, the whole stack, which must
# go when it does.
_current: Optional[Tuple["weakref.ref[QWidget]", str, str, str]] = None


def _same_position(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(
        os.path.normpath(b)
    )


def register_viewer(window: QWidget, position: str, population: str, kind: str) -> None:
    """
    Record that a viewer window is open on a position, until it is closed.

    Parameters
    ----------
    window : QWidget
        The napari main window.
    position : str
        The position directory shown.
    population : str
        The population shown.
    kind : str
        What the viewer is for, as shown to the user ("segmentation", "tracks").
    """
    global _current
    _current = (weakref.ref(window), position, population, kind)


def open_viewer() -> Optional[OpenViewer]:
    """
    The position viewer that is open, if any.

    Returns
    -------
    OpenViewer or None
        The open viewer, or None when no position viewer is open.
    """
    global _current
    if _current is None:
        return None
    window_ref, position, population, kind = _current
    window = window_ref()
    try:
        alive = window is not None and is_alive(window) and window.isVisible()
    except RuntimeError:
        alive = False
    if not alive:
        _current = None
        return None
    return OpenViewer(window, position, population, kind)


def refusal_message(position: str, opened: OpenViewer) -> str:
    """
    Why another viewer cannot be opened on `position` while `opened` is.

    Parameters
    ----------
    position : str
        The position a viewer was asked for.
    opened : OpenViewer
        The viewer already open.

    Returns
    -------
    str
        The message to show.
    """
    name = os.path.basename(os.path.normpath(opened.position))
    shown = f"The {opened.kind} of the {opened.population} population of position {name}"
    if _same_position(position, opened.position):
        reason = (
            "Two viewers on the same position can save over each other's "
            "corrections."
        )
    else:
        reason = "Only one napari viewer can be open on the positions at a time."
    return (
        f"{shown} is already open in napari.\n\n{reason} Close that viewer "
        "before opening another one."
    )


def refuse_if_viewer_open(parent: Optional[QWidget], position: str) -> bool:
    """
    Refuse, with a message box, to open a viewer while a position viewer is open.

    The open viewer is brought to the front.

    Parameters
    ----------
    parent : QWidget or None
        The window the message box belongs to.
    position : str
        The position a viewer was asked for.

    Returns
    -------
    bool
        True when a viewer is open and the new one must not be opened.
    """
    opened = open_viewer()
    if opened is None:
        return False
    logger.info(
        f"Refused to open a napari viewer on {position}: one is open on "
        f"{opened.position}."
    )
    try:
        if opened.window.isMinimized():
            opened.window.showNormal()
        opened.window.raise_()
        opened.window.activateWindow()
    except Exception as e:
        logger.debug(f"Could not bring the open napari viewer to the front: {e}")
    QMessageBox.information(
        parent, "A napari viewer is already open", refusal_message(position, opened)
    )
    return True
