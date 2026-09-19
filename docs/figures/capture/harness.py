"""
Launch celldetective the way ``__main__`` does and capture its windows for the docs.

The captures run on the ``demo_ricm`` demo (File > Open Demo in the software).
Point ``CELLDETECTIVE_DOCS_EXP`` to a *copy* of it: the scripts never save, but
opening an experiment writes logs next to it. The windows are captured with
their title bar, as they are on screen, so this runs on Windows (a window is
pinned on top while it is captured) at a 100 % display scale.
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


def grab(widget, name, frame=True, keep_on_top=False):
    """Capture a top-level window, title bar included, as it is on screen."""
    win = widget.window()
    topmost(win, True)
    win.raise_()
    win.activateWindow()
    pump(0.8)
    if frame:
        g = win.frameGeometry()
        screen = App.screenAt(g.center()) or App.primaryScreen()
        # grabWindow takes logical coordinates relative to the virtual desktop.
        pix = screen.grabWindow(0, g.x(), g.y(), g.width(), g.height())
    else:
        pix = win.grab()
    if not keep_on_top:
        topmost(win, False)
    path = os.path.join(OUT, name + ".png")
    pix.save(path)
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
