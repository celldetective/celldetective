#!/usr/bin/env python3
import sys
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from os import sep

from celldetective.gui.gui_utils import center_window

from celldetective.utils import get_software_location
from time import time, sleep

#os.environ['QT_DEBUG_PLUGINS'] = '1'

if __name__ == "__main__":

	splash=True
	print('Loading the libraries...')

	App = QApplication(sys.argv)
	App.setStyle("Fusion")

	software_location = get_software_location()

	if splash:
		start = time()
		splash_pix = QPixmap(sep.join([software_location,'celldetective','icons','splash.png']))
		splash = QSplashScreen(splash_pix)
		splash.setMask(splash_pix.mask())
		splash.show()
		#App.processEvents(QEventLoop.AllEvents, 300)
		while time() - start < 1:
			sleep(0.001)
			App.processEvents()

	try:
		
		import requests
		import re
		from celldetective import __version__

		package = 'celldetective'
		response = requests.get(f'https://pypi.org/pypi/{package}/json')
		latest_version = response.json()['info']['version']

		latest_version_num = re.sub('[^0-9]','', latest_version)
		current_version_num = re.sub('[^0-9]','',__version__)

		if len(latest_version_num)!=len(current_version_num):
			max_length = max([len(latest_version_num),len(current_version_num)])
			latest_version_num = int(latest_version_num.zfill(max_length - len(latest_version_num)))
			current_version_num = int(current_version_num.zfill(max_length - len(current_version_num)))

		if latest_version_num > current_version_num:
			print('Update is available...\nPlease update using `pip install --upgrade celldetective`...')
	
	except Exception as e:

		print(f"{e=}")

	from celldetective.gui.InitWindow import AppInitWindow

	print('Libraries successfully loaded...')

	window = AppInitWindow(App, software_location=software_location)
	center_window(window)

	if splash:
		splash.finish(window)

	sys.exit(App.exec())