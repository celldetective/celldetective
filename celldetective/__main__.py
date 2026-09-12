#!/usr/bin/env python3
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap

# os.environ['QT_DEBUG_PLUGINS'] = '1'


def check_update():
    """
    Check for software updates on PyPI.

    Fetches the latest version from PyPI and compares it with the current version.
    Logs a warning if a newer version is available.
    """
    from celldetective import logger

    try:
        import requests
        import re
        from celldetective import __version__

        package = "celldetective"
        response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=5)
        latest_version = response.json()["info"]["version"]

        latest_version_num = re.sub("[^0-9]", "", latest_version)
        current_version_num = re.sub("[^0-9]", "", __version__)

        if len(latest_version_num) != len(current_version_num):
            max_length = max([len(latest_version_num), len(current_version_num)])
            latest_version_num = int(
                latest_version_num.zfill(max_length - len(latest_version_num))
            )
            current_version_num = int(
                current_version_num.zfill(max_length - len(current_version_num))
            )

        if latest_version_num > current_version_num:
            logger.warning(
                "Update is available...\nPlease update using `pip install --upgrade celldetective`..."
            )
    except Exception as e:
        logger.error(
            f"Update check failed... Please check your internet connection: {e}"
        )


def main():
    """
    Entry point of the celldetective GUI.

    Starts the Qt application, shows the splash screen while the heavy
    libraries load, and opens the initial window.

    Returns
    -------
    int
            Qt exit code, suitable for `sys.exit`.
    """

    show_splash = True
    from celldetective import logger
    from celldetective import get_package_location
    from celldetective import get_software_location

    logger.info("Loading the libraries...")

    from PyQt5.QtCore import Qt

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    from celldetective.gui.base.app_style import CelldetectiveStyle
    from celldetective.gui.base.styles import (
        SCROLLBAR_STYLE,
        TOOLTIP_STYLE,
        TOOLTIP_FONT_SIZE,
    )

    App = QApplication(sys.argv)
    App.setStyle(CelldetectiveStyle("Fusion"))
    App.setStyleSheet(TOOLTIP_STYLE + SCROLLBAR_STYLE)

    # Set with the widget font rather than through the style sheet, so that the
    # tooltips are laid out with the font they are painted with.
    from PyQt5.QtWidgets import QToolTip

    tooltip_font = App.font()
    tooltip_font.setPointSize(TOOLTIP_FONT_SIZE)
    QToolTip.setFont(tooltip_font)

    software_location = get_software_location()

    splash = None
    if show_splash:
        splash_pix = QPixmap(
            os.path.join(get_package_location(), "icons", "splash.png")
        )
        if splash_pix.isNull():
            # A null pixmap still yields a real, zero-sized top level window,
            # so skip the splash entirely rather than show that ghost.
            logger.warning("Could not load the splash screen image...")
        else:
            splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
            splash.setMask(splash_pix.mask())
            splash.show()
            App.processEvents()

    def splash_message(message):
        """
        Write a status line on the splash screen and repaint it.

        The imports below block the event loop for several seconds; without
        these pumps the splash is never redrawn and the window manager paints
        it as an unresponsive white rectangle.

        Parameters
        ----------
        message : str
                Status line to display.
        """

        if splash is None:
            return
        splash.showMessage(
            message, Qt.AlignBottom | Qt.AlignHCenter, Qt.white
        )
        App.processEvents()

    # Update check in background
    import threading

    update_thread = threading.Thread(target=check_update)
    update_thread.daemon = True
    update_thread.start()

    window = None
    try:
        splash_message("Loading the libraries...")
        from celldetective.gui.InitWindow import AppInitWindow

        logger.info("Libraries successfully loaded...")

        from celldetective.gui.base.utils import center_window

        splash_message("Starting celldetective...")

        # AppInitWindow shows itself in its constructor; splash.finish() below
        # relies on that, as it waits for the window to be displayed.
        window = AppInitWindow(App, software_location=software_location)
        center_window(window)
    finally:
        if splash is not None:
            if window is not None:
                splash.finish(window)
            else:
                # Startup blew up: close the splash so it does not sit on top
                # of the traceback until the interpreter exits.
                splash.close()

    return App.exec()


if __name__ == "__main__":
    sys.exit(main())
