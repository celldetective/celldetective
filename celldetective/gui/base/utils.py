import logging
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QDesktopWidget,
    QMainWindow,
    QWidget,
)
from PyQt5.QtCore import QEvent, QObject, QRect
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


class _ScrollBarSpaceKeeper(QObject):
    """
    Keeps the width of a scroll area free of its vertical scroll bar.

    A scroll bar that comes and goes takes its width from the viewport when it
    appears, which shifts the whole content of the area sideways. The filter
    watches the bar and gives the viewport a right margin of exactly its width
    whenever it is hidden, so that the content keeps the same width and the
    same place in both states.
    """

    def __init__(self, area: QAbstractScrollArea) -> None:
        """
        Initialize the filter.

        Parameters
        ----------
        area : QAbstractScrollArea
            The scroll area to keep steady.
        """

        super().__init__(area)

        self.area = area
        self.bar = area.verticalScrollBar()
        self.bar.installEventFilter(self)
        self.reserve()

    def reserve(self) -> None:
        """Give the viewport the margin the hidden scroll bar would take."""

        margin = 0 if self.bar.isVisible() else self.bar.sizeHint().width()
        self.area.setViewportMargins(0, 0, margin, 0)

    def eventFilter(self, source: QObject, event: QEvent) -> bool:
        """
        Follow the scroll bar as it appears and disappears.

        Parameters
        ----------
        source : QObject
            The event source.
        event : QEvent
            The event.
        """

        if source is self.bar and event.type() in (QEvent.Show, QEvent.Hide):
            self.reserve()

        return super().eventFilter(source, event)


def keep_scrollbar_space(area: QAbstractScrollArea) -> _ScrollBarSpaceKeeper:
    """
    Stop the content of a scroll area from shifting with its scroll bar.

    Parameters
    ----------
    area : QAbstractScrollArea
        The scroll area to keep steady.

    Returns
    -------
    _ScrollBarSpaceKeeper
        The filter, parented to the area.
    """

    return _ScrollBarSpaceKeeper(area)


def keep_window_on_screen(window: Union[QMainWindow, QWidget]) -> None:
    """
    Move a window back inside the screen it sits on, if it pokes out of it.

    Only the offending edges are corrected: a window the user has placed
    somewhere keeps its position as long as it fits.

    Parameters
    ----------
    window : QMainWindow or QWidget
        The window to bring back on screen.
    """

    screen = get_current_screen_geometry(window)
    frame = window.frameGeometry()

    x, y = frame.x(), frame.y()
    if frame.bottom() > screen.bottom():
        y = max(screen.top(), screen.bottom() - frame.height() + 1)
    if frame.top() < screen.top():
        y = screen.top()
    if frame.right() > screen.right():
        x = max(screen.left(), screen.right() - frame.width() + 1)
    if frame.left() < screen.left():
        x = screen.left()

    if (x, y) != (frame.x(), frame.y()):
        window.move(x, y)


def fit_window_to_content(
    window: Union[QMainWindow, QWidget],
    area: QAbstractScrollArea,
    screen_fraction: float = 0.9,
) -> None:
    """
    Give a window the height its scrolled content asks for, within the screen.

    The window is resized to the height of the widget inside the scroll area,
    plus whatever chrome sits around it, and never beyond a fraction of the
    screen: past that the content scrolls instead. The window is then brought
    back on screen if it now pokes out of it.

    Parameters
    ----------
    window : QMainWindow or QWidget
        The window to resize.
    area : QAbstractScrollArea
        The scroll area holding the content of the window.
    screen_fraction : float, optional
        The largest share of the height of the screen the window may take.
    """

    content = area.widget() if hasattr(area, "widget") else None
    if content is None:
        return

    screen = get_current_screen_geometry(window)
    max_height = int(screen_fraction * screen.height())
    window.setMaximumHeight(max_height)

    # Everything that is not the viewport: title bar aside, the margins of the
    # window, the frame of the area and its horizontal scroll bar, if any.
    chrome = window.height() - area.viewport().height()
    wanted = content.sizeHint().height() + chrome

    window.resize(window.width(), max(window.minimumHeight(), min(wanted, max_height)))
    keep_window_on_screen(window)


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


_SIP_MODULES = None


def _sip_modules() -> tuple:
    """
    The sip modules that might own a Qt wrapper, most likely first.

    PyQt5 wraps its objects with the ``PyQt5.sip`` module it was built against.
    A top-level ``sip`` module is a different build -- conda ships one, and it is
    importable alongside a pip-installed PyQt5 -- whose ``isdeleted`` rejects
    foreign wrappers outright. Trying the bundled module first means the common
    case never depends on that rejection being handled.

    Returns
    -------
    tuple
        The importable sip modules, in the order they should be tried.
    """

    global _SIP_MODULES
    if _SIP_MODULES is None:
        modules = []
        try:
            from PyQt5 import sip as pyqt5_sip

            modules.append(pyqt5_sip)
        except ImportError:
            pass
        try:
            import sip as top_level_sip

            if top_level_sip not in modules:
                modules.append(top_level_sip)
        except ImportError:
            pass
        _SIP_MODULES = tuple(modules)
    return _SIP_MODULES


def is_alive(obj) -> bool:
    """
    Whether a Qt object's underlying C++ object is still there.

    A Python wrapper outlives the object it wraps: every window here carries
    ``WA_DeleteOnClose``, so closing one has Qt delete the C++ side while the
    attribute holding it stays perfectly usable-looking. Calling into that is
    sometimes a clean ``RuntimeError`` -- but when the deletion was partial, a
    parent still standing with its children freed, it is an access violation
    instead, which no ``except`` can catch. Asking first is the only guard.

    When no sip module can answer -- none importable, or none that recognises
    the wrapper -- this returns True. The guard is then no worse than the
    unguarded code it replaced, whereas a False would silently disable whatever
    it protects on an environment that is otherwise perfectly healthy.

    Parameters
    ----------
    obj : QObject or None
        The object to check.

    Returns
    -------
    bool
        True when `obj` exists and can still be used.
    """

    if obj is None:
        return False

    for module in _sip_modules():
        try:
            return not module.isdeleted(obj)
        except TypeError:
            # Wrong sip build for this wrapper -- it cannot see the object at
            # all, which says nothing about whether the object is alive. Ask the
            # next one rather than reporting a live widget dead: every caller
            # here uses a False to skip an update, so guessing wrong loses a
            # slider sync or a redraw with nothing logged.
            continue
        except RuntimeError:
            return False
    return True
