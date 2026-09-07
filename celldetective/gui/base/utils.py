import logging
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QMainWindow, QWidget
from PyQt5.QtCore import QRect
from typing import Union
from prettytable import PrettyTable

logger = logging.getLogger("celldetective")


def get_current_screen_geometry(widget=None) -> QRect:
    """
    Return the available geometry of the screen where the mouse cursor
    currently resides (or where the widget is located, if provided and visible).

    Unlike ``QApplication.primaryScreen().availableGeometry()``, this
    correctly handles multi-monitor setups.

    Returns
    -------
    QRect
        The available geometry (excluding taskbar) of the active screen.
    """
    desktop = QApplication.desktop()
    if widget is not None and getattr(widget, "isVisible", lambda: False)():
        screen_number = desktop.screenNumber(widget)
    else:
        screen_number = desktop.screenNumber(desktop.cursor().pos())
    return desktop.availableGeometry(screen_number)


def center_window(window: Union[QMainWindow, QWidget]) -> None:
    """
    Centers the given window in the middle of the screen.

    This function calculates the current screen's geometry and moves the
    specified window to the center of the screen. It works by retrieving the
    frame geometry of the window, identifying the screen where the cursor is
    currently located, and adjusting the window's position to be centrally
    aligned on that screen.

    Parameters
    ----------
    window : QMainWindow or QWidget
            The window or widget to be centered on the screen.
    """

    frameGm = window.frameGeometry()
    screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
    centerPoint = QApplication.desktop().screenGeometry(screen).center()
    frameGm.moveCenter(centerPoint)
    window.move(frameGm.topLeft())


def pretty_table(dct: dict):
    """
    Print a dictionary as a pretty table.

    Parameters
    ----------
    dct : dict
        The dictionary to print.
    """
    table = PrettyTable()
    for c in dct.keys():
        table.add_column(str(c), [])
    table.add_row([dct.get(c, "") for c in dct.keys()])
    logger.debug(str(table))



def flush_layout_events(widget: QWidget) -> None:
    """
    Settle a widget's own pending layout events, without running the event loop.

    The replacement for a ``QApplication.processEvents()`` at the end of a
    ``populate_*`` / ``_build_layouts`` / ``__init__``. That call was only ever
    meant to let the window lay itself out before the next line measured it, but
    it re-enters the event loop with the widget half-built and dispatches
    *everything* that happens to be queued -- including the ``DeferredDelete`` of
    any window closed earlier. Qt frees those C++ objects while events queued
    behind the deletion are still addressed to them, and delivering one of those
    is an access violation attributed to whatever is on the stack at the time.

    ``sendPostedEvents`` restricted to `widget` does the useful half only: it
    delivers this widget's and its children's posted events, and Qt excludes
    ``DeferredDelete`` from it unless that type is asked for by name, so nothing
    can be destroyed underneath the caller.

    Parameters
    ----------
    widget : QWidget
        The widget whose pending events should be delivered.
    """

    QApplication.sendPostedEvents(widget, 0)
