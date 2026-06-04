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

