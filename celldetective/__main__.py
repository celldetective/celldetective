#!/usr/bin/env python3
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from os import sep

# os.environ['QT_DEBUG_PLUGINS'] = '1'

if __name__ == "__main__":

    splash = True
    from celldetective import logger
    from celldetective import get_software_location

    logger.info("Loading the libraries...")

    App = QApplication(sys.argv)
    App.setStyle("Fusion")

    software_location = get_software_location()

    if splash:
        splash_pix = QPixmap(
            sep.join([software_location, "celldetective", "icons", "splash.png"])
        )
        splash = QSplashScreen(splash_pix)
        splash.setMask(splash_pix.mask())
        splash.show()
        App.processEvents()

    # Update check in background
    def check_update():
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

    import threading

    update_thread = threading.Thread(target=check_update)
    update_thread.daemon = True
    update_thread.start()

    from celldetective.gui.InitWindow import AppInitWindow

    logger.info("Libraries successfully loaded...")

    from celldetective.gui.base.utils import center_window

    window = AppInitWindow(App, software_location=software_location)
    center_window(window)

    if splash:
        splash.finish(window)

    sys.exit(App.exec())
