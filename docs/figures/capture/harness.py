"""
Launch celldetective the way ``__main__`` does and capture its windows for the docs.

The captures run on the ``demo_ricm`` demo (File > Open Demo in the software).
Point ``CELLDETECTIVE_DOCS_EXP`` to a *copy* of it: the scripts never save, but
opening an experiment writes logs next to it. The windows are captured with
their title bar, as they are on screen, so this runs on Windows (a window is
pinned on top while it is captured) at a 100 % display scale;
on Linux the window is rendered by Qt and given the Ubuntu title bar\n(titlebar_ubuntu.png), so a locked or busy screen does not matter.
"""

import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

EXP = os.environ.get("CELLDETECTIVE_DOCS_EXP")
if not EXP or not os.path.isdir(EXP):
    sys.exit("Set CELLDETECTIVE_DOCS_EXP to a copy of the demo_ricm experiment.")
# Fluorescence captures (nuclei, spots, background) use a copy of demo_adcc (File >
# Open Demo > ADCC Demo): targets are MCF7 nuclei, effectors primary NK cells.
ADCC = os.environ.get("CELLDETECTIVE_DOCS_ADCC")


def require_adcc():
    if not ADCC or not os.path.isdir(ADCC):
        sys.exit("Set CELLDETECTIVE_DOCS_ADCC to a copy of the demo_adcc experiment.")
    return ADCC
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")
os.makedirs(OUT, exist_ok=True)

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

from celldetective.gui.base.app_style import CelldetectiveStyle
from celldetective.gui.base.styles import SCROLLBAR_STYLE, TOOLTIP_STYLE

App = QApplication(sys.argv)
App.setStyle(CelldetectiveStyle("Fusion"))
App.setStyleSheet(TOOLTIP_STYLE + SCROLLBAR_STYLE)

from celldetective import get_software_location


def pump(seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        App.processEvents()
        time.sleep(0.02)


if sys.platform == "win32":
    import ctypes

    _user32 = ctypes.windll.user32
    HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
    SWP_NOSIZE, SWP_NOMOVE, SWP_SHOWWINDOW = 0x1, 0x2, 0x40

    def topmost(win, on=True):
        """Pin a window above the others: Windows refuses to hand the focus to a script."""
        _user32.SetWindowPos(
            ctypes.c_void_p(int(win.winId())),
            ctypes.c_void_p(HWND_TOPMOST if on else HWND_NOTOPMOST),
            0, 0, 0, 0,
            SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW,
        )


def titled(pix, title):
    """
    Put an Ubuntu (Yaru) title bar above a rendered window, as the window manager
    draws it. The bar is cut from a real capture (titlebar_ubuntu.png): left
    corner, a column stretched to the width, and the window buttons.
    """
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap

    tpl = QPixmap(os.path.join(os.path.dirname(os.path.abspath(__file__)), "titlebar_ubuntu.png"))
    bar_h, left, right = tpl.height(), 12, 110
    w, h = pix.width(), pix.height()
    out = QPixmap(w, h + bar_h)
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.drawPixmap(0, 0, tpl, 0, 0, left, bar_h)
    for x in range(left, w - right):
        p.drawPixmap(x, 0, tpl, 20, 0, 1, bar_h)
    p.drawPixmap(w - right, 0, tpl, tpl.width() - right, 0, right, bar_h)
    font = QFont("Ubuntu")
    font.setPointSize(11)
    font.setWeight(QFont.Bold)
    p.setFont(font)
    p.setPen(QColor("#ffffff"))
    # Centred on the bar when it fits between the corner and the buttons.
    tw = p.fontMetrics().horizontalAdvance(title)
    x = max(left, min((w - tw) // 2, w - right - tw))
    p.drawText(QRect(x, 0, w - right - x, bar_h - 1), Qt.AlignVCenter | Qt.AlignLeft,
               p.fontMetrics().elidedText(title, Qt.ElideRight, w - right - x))
    p.drawPixmap(0, bar_h, pix)
    p.end()
    return out


def _marks(win, marks, left, top):
    """Rectangles [x, y, w, h] of the given widgets, in pixels of the capture."""
    from PyQt5.QtCore import QPoint

    out = {}
    for key, w in (marks or {}).items():
        if not hasattr(w, "mapTo"):  # a layout-like helper holding its widget
            w = getattr(w, "list_widget", None)
        if w is None or not w.isVisible():
            continue
        p = w.mapTo(win, QPoint(0, 0))
        out[key] = [p.x() + left, p.y() + top, w.width(), w.height()]
    return out


def grab(widget, name, frame=True, keep_on_top=False, marks=None):
    """
    Capture a top-level window, title bar included, as it is on screen.

    marks : {key: widget}, optional
        Widgets to point at in the figure: their rectangles in the capture are
        written to screenshots/<name>.json, for build_figures.py.
    """
    import json

    win = widget.window()
    if sys.platform != "win32":
        # Render the window itself (the screen may be locked or covered) and
        # draw the window manager's title bar above it.
        win.raise_()
        # The mouse may rest on a plot: drop the coordinates its toolbar prints.
        from matplotlib.backends.backend_qt import NavigationToolbar2QT

        for tb in win.findChildren(NavigationToolbar2QT):
            tb.set_message("")
        pump(0.8)
        pix = win.grab()
        left = top = 0
        if frame:
            pix = titled(pix, win.windowTitle())
            top = pix.height() - win.height()
    else:
        topmost(win, True)
        win.raise_()
        win.activateWindow()
        pump(0.8)
        left = top = 0
        if frame:
            g = win.frameGeometry()
            left, top = win.geometry().x() - g.x(), win.geometry().y() - g.y()
            screen = App.screenAt(g.center()) or App.primaryScreen()
            # grabWindow takes logical coordinates relative to the virtual desktop.
            pix = screen.grabWindow(0, g.x(), g.y(), g.width(), g.height())
        else:
            pix = win.grab()
        if not keep_on_top:
            topmost(win, False)
    path = os.path.join(OUT, name + ".png")
    pix.save(path)
    if marks:
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(_marks(win, marks, left, top), f, indent=1)
    print("saved", path, pix.width(), pix.height(), "dpr", pix.devicePixelRatio())
    return path


def open_experiment(path=EXP):
    from celldetective.gui.InitWindow import AppInitWindow

    init = AppInitWindow(App, software_location=get_software_location())
    pump(1)
    init.experiment_path_selection.setText(path)
    init.validate_button.click()
    pump(3)
    return init, init.control_panel


def move(widget, x, y, w=None, h=None):
    win = widget.window()
    if w and h:
        win.resize(w, h)
    win.move(x, y)
    pump(0.4)
