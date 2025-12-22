#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from os import sep

#os.environ['QT_DEBUG_PLUGINS'] = '1'

if __name__ == "__main__":

	splash=True
	from celldetective import logger
	from celldetective.gui.gui_utils import center_window
	from celldetective.utils import get_software_location
	logger.info('Loading the libraries...')

	App = QApplication(sys.argv)
	App.setStyle("Fusion")

	# --- Recorder Setup ---
	# try:
	# 	# from celldetective.gui.recorder import ActionRecorder
	# 	import shutil
	# 	import os

	# 	home_dir = os.path.expanduser("~")
	# 	session_dir = os.path.join(home_dir, ".celldetective", "sessions")
	# 	if not os.path.exists(session_dir):
	# 		os.makedirs(session_dir)

	# 	# Old rotation logic removed in favor of timestamped logs and auto-cleanup
	# 	# recorder = ActionRecorder(output_dir=session_dir)
	# 	# recorder.start()

	# 	# Save on exit
	# 	# App.aboutToQuit.connect(lambda: recorder.save())

	# 	# Save on crash
	# 	sys_excepthook = sys.excepthook
	# 	def hook(exctype, value, traceback):
	# 		logger.error("Unhandled exception!", exc_info=(exctype, value, traceback))
	# 		recorder.save()
	# 		sys_excepthook(exctype, value, traceback)
	# 	sys.excepthook = hook
	# except Exception as e:
	# 	logger.error(f"Failed to initialize recorder: {e}")
	# ----------------------

	software_location = get_software_location()

	if splash:
		splash_pix = QPixmap(sep.join([software_location,'celldetective','icons','splash.png']))
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
			
			package = 'celldetective'
			response = requests.get(f'https://pypi.org/pypi/{package}/json', timeout=5)
			latest_version = response.json()['info']['version']

			latest_version_num = re.sub('[^0-9]','', latest_version)
			current_version_num = re.sub('[^0-9]','',__version__)

			if len(latest_version_num)!=len(current_version_num):
				max_length = max([len(latest_version_num),len(current_version_num)])
				latest_version_num = int(latest_version_num.zfill(max_length - len(latest_version_num)))
				current_version_num = int(current_version_num.zfill(max_length - len(current_version_num)))

			if latest_version_num > current_version_num:
				logger.warning('Update is available...\nPlease update using `pip install --upgrade celldetective`...')
		except Exception as e:
			logger.error(f"Update check failed... Please check your internet connection: {e}")

	import threading
	update_thread = threading.Thread(target=check_update)
	update_thread.daemon = True
	update_thread.start()

	from celldetective.gui.InitWindow import AppInitWindow

	logger.info('Libraries successfully loaded...')

	window = AppInitWindow(App, software_location=software_location)
	center_window(window)

	if splash:
		splash.finish(window)

	sys.exit(App.exec())