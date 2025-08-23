from multiprocessing import Queue
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QRunnable, QObject, pyqtSignal, QThreadPool, QSize, Qt

from celldetective.gui.base_components import CelldetectiveDialog
from celldetective.gui.gui_utils import center_window
import math

class ProgressWindow(CelldetectiveDialog):

	def __init__(self, process=None, parent_window=None, title="", position_info=True, process_args=None):

		super().__init__()
		#QDialog.__init__(self)

		self.setWindowTitle(f'{title} Progress')
		self.__process = process
		self.parent_window = parent_window

		self.position_info = position_info
		if self.position_info:
			self.pos_name = self.parent_window.pos_name

		#self.__btn_run = QPushButton("Start")
		self.__btn_stp = QPushButton("Cancel")
		if self.position_info:
			self.position_label = QLabel(f'Processing position {self.pos_name}...')
		self.__label   = QLabel("Idle")
		self.time_left_lbl = QLabel('')

		self.progress_bar = QProgressBar()
		self.progress_bar.setValue(0)

		self.__runner  = Runner(process=self.__process, progress_bar=self.progress_bar, process_args=process_args, remaining_time_label=self.time_left_lbl)
		self.pool = QThreadPool.globalInstance()
		#print("Multithreading with maximum %d threads" % self.pool.maxThreadCount())

		#self.__btn_run.clicked.connect(self.__run_net)
		self.__btn_stp.clicked.connect(self.__stp_net)
		self.__runner.signals.finished.connect(self.__on_finished)
		self.__runner.signals.finished.connect(self.__on_error)
		self.__runner.signals.update.connect(self.progress_bar.setValue)
		self.__runner.signals.update_time.connect(self.time_left_lbl.setText)

		self.__btn_stp.setDisabled(True)

		self.layout = QVBoxLayout()
		if self.position_info:
			self.layout.addWidget(self.position_label)
		self.layout.addWidget(self.time_left_lbl)
		self.layout.addWidget(self.progress_bar)
		self.btn_layout = QHBoxLayout()
		self.btn_layout.addWidget(self.__btn_stp)
		self.btn_layout.addWidget(self.__label)
		self.layout.addLayout(self.btn_layout)

		self.setLayout(self.layout)
		self.setFixedSize(QSize(250, 130))
		self.__run_net()
		self.setModal(True)
		center_window(self)

	def closeEvent(self, evnt):
		# if self._want_to_close:
		# 	super(MyDialog, self).closeEvent(evnt)
		# else:
		evnt.ignore()
		self.setWindowState(Qt.WindowMinimized)

	def __run_net(self):
		#self.__btn_run.setDisabled(True)
		self.__btn_stp.setEnabled(True)
		self.__label.setText("Running...")
		self.pool.start(self.__runner)

	def __stp_net(self):
		self.__runner.close()
		print('\n Job cancelled... Abort.')
		self.reject()

	def __on_finished(self):
		self.__btn_stp.setDisabled(True)
		self.__label.setText("\nFinished!")
		self.__runner.close()
		self.accept()

	def __on_error(self):
		self.__btn_stp.setDisabled(True)
		self.__label.setText("\nError")
		self.__runner.close()
		self.accept()

class Runner(QRunnable):

	def __init__(self, process=None, progress_bar=None, remaining_time_label=None, process_args=None):
		QRunnable.__init__(self)

		print(f"{process_args=}")
		self.progress_bar = progress_bar
		#self.parent_window = parent_window
		self.remaining_time_label = remaining_time_label
		self.__queue = Queue()
		self.__process = process(self.__queue, process_args=process_args)
		self.signals = RunnerSignal()

	def run(self):
		self.__process.start()
		while True:
			try:
				data = self.__queue.get()
				if len(data)==2:
					progress, time = data
					time = int(time)
					remaining_minutes = time // 60
					remaining_seconds = int(time % 60)
					time_left = "About "+str(remaining_minutes)+" min and "+str(remaining_seconds)+" s remaining"
					if remaining_minutes == 0:
						time_left = "About "+str(remaining_seconds)+" s remaining"
					self.signals.update_time.emit(time_left)
					if isinstance(progress, (float,int)):
						self.signals.update.emit(math.ceil(progress))
				if data == "finished":
					self.signals.finished.emit()
					break
				elif data == "error":
					self.signals.error.emit()
			except Exception as e:
				print(e)
				pass

	def close(self):
		self.__process.end_process()


class RunnerSignal(QObject):

	update = pyqtSignal(int)
	update_time = pyqtSignal(str)
	finished = pyqtSignal()
	error = pyqtSignal()