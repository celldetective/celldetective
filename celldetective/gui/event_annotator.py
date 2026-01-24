from PyQt5.QtWidgets import (
    QComboBox,
    QLabel,
    QRadioButton,
    QFileDialog,
    QApplication,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QShortcut,
    QLineEdit,
    QSlider,
    QAction,
    QMenu,
)
from celldetective.gui.interactive_timeseries_viewer import InteractiveEventViewer
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QKeySequence, QIntValidator

from celldetective.gui.gui_utils import color_from_state
from celldetective.gui.base.utils import center_window, pretty_table
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider, QSearchableComboBox
from celldetective.utils.image_loaders import (
    locate_labels,
    load_frames,
    _get_img_num_per_channel,
)
from celldetective.utils.experiment import (
    get_experiment_metadata,
    get_experiment_labels,
)
from celldetective.gui.gui_utils import (
    color_from_status,
    color_from_class,
)
from celldetective.gui.base.figure_canvas import FigureCanvas
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from matplotlib.animation import FuncAnimation
from matplotlib.cm import tab10
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.gui.base_annotator import BaseAnnotator


class StackLoaderThread(QThread):
    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, annotator):
        super().__init__()
        self.annotator = annotator
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        def callback(progress, status=""):
            if self._is_cancelled:
                return False
            self.progress.emit(progress)
            if status:
                self.status_update.emit(status)
            return True

        try:
            self.annotator.prepare_stack(progress_callback=callback)
            if not self._is_cancelled:
                self.finished.emit()
        except Exception as e:
            print(f"Error in loader thread: {e}")
            self.finished.emit()


class EventAnnotator(BaseAnnotator):
    """
    UI to set tracking parameters for bTrack.

    """

    def __init__(self, *args, lazy_load=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Signal annotator")

        # default params
        self.class_name = "class"
        self.time_name = "t0"
        self.status_name = "status"

        # self.locate_stack()
        if not self.proceed:
            self.close()
        else:
            if not lazy_load:
                self.prepare_stack()
                self.finalize_init()

    def finalize_init(self):
        self.frame_lbl = QLabel("frame: ")
        self.looped_animation()
        self.init_event_buttons()
        self.populate_window()

        self.outliers_check.hide()
        if hasattr(self, "contrast_slider"):
            self.im.set_clim(
                self.contrast_slider.value()[0], self.contrast_slider.value()[1]
            )

    def init_event_buttons(self):

        self.event_btn = QRadioButton("event")
        self.event_btn.setStyleSheet(self.button_style_sheet_2)
        self.event_btn.toggled.connect(self.enable_time_of_interest)

        self.no_event_btn = QRadioButton("no event")
        self.no_event_btn.setStyleSheet(self.button_style_sheet_2)
        self.no_event_btn.toggled.connect(self.enable_time_of_interest)

        self.else_btn = QRadioButton("else")
        self.else_btn.setStyleSheet(self.button_style_sheet_2)
        self.else_btn.toggled.connect(self.enable_time_of_interest)

        self.suppr_btn = QRadioButton("remove")
        self.suppr_btn.setToolTip(
            "Mark for deletion. Upon saving, the cell\nwill be removed from the tables."
        )
        self.suppr_btn.setStyleSheet(self.button_style_sheet_2)
        self.suppr_btn.toggled.connect(self.enable_time_of_interest)

        self.time_of_interest_label = QLabel("time of interest: ")
        self.time_of_interest_le = QLineEdit()

    def populate_options_layout(self):

        # clear options hbox
        for i in reversed(range(self.options_hbox.count())):
            self.options_hbox.itemAt(i).widget().setParent(None)

        options_layout = QVBoxLayout()
        # add new widgets

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.event_btn, 25, alignment=Qt.AlignCenter)
        btn_hbox.addWidget(self.no_event_btn, 25, alignment=Qt.AlignCenter)
        btn_hbox.addWidget(self.else_btn, 25, alignment=Qt.AlignCenter)
        btn_hbox.addWidget(self.suppr_btn, 25, alignment=Qt.AlignCenter)

        time_option_hbox = QHBoxLayout()
        time_option_hbox.setContentsMargins(0, 5, 100, 10)
        time_option_hbox.addWidget(self.time_of_interest_label, 10)
        time_option_hbox.addWidget(self.time_of_interest_le, 15)
        time_option_hbox.addWidget(QLabel(""), 75)

        options_layout.addLayout(btn_hbox)
        options_layout.addLayout(time_option_hbox)

        self.options_hbox.addLayout(options_layout)

        self.annotation_btns_to_hide = [
            self.event_btn,
            self.no_event_btn,
            self.else_btn,
            self.time_of_interest_label,
            self.time_of_interest_le,
            self.suppr_btn,
        ]
        self.hide_annotation_buttons()

    def populate_window(self):
        """
        Create the multibox design.

        """

        super().populate_window()
        self.populate_options_layout()

        self.del_shortcut = QShortcut(Qt.Key_Delete, self)  # QKeySequence("s")
        self.del_shortcut.activated.connect(self.shortcut_suppr)
        self.del_shortcut.setEnabled(False)

        self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
        self.no_event_shortcut.activated.connect(self.shortcut_no_event)
        self.no_event_shortcut.setEnabled(False)

        # Right side
        animation_buttons_box = QHBoxLayout()
        animation_buttons_box.addWidget(self.frame_lbl, 20, alignment=Qt.AlignLeft)

        self.first_frame_btn = QPushButton()
        self.first_frame_btn.clicked.connect(self.set_first_frame)
        self.first_frame_btn.setShortcut(QKeySequence("f"))
        self.first_frame_btn.setIcon(icon(MDI6.page_first, color="black"))
        self.first_frame_btn.setStyleSheet(self.button_select_all)
        self.first_frame_btn.setFixedSize(QSize(60, 60))
        self.first_frame_btn.setIconSize(QSize(30, 30))

        self.last_frame_btn = QPushButton()
        self.last_frame_btn.clicked.connect(self.set_last_frame)
        self.last_frame_btn.setShortcut(QKeySequence("l"))
        self.last_frame_btn.setIcon(icon(MDI6.page_last, color="black"))
        self.last_frame_btn.setStyleSheet(self.button_select_all)
        self.last_frame_btn.setFixedSize(QSize(60, 60))
        self.last_frame_btn.setIconSize(QSize(30, 30))

        self.stop_btn = QPushButton()
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setIcon(icon(MDI6.stop, color="black"))
        self.stop_btn.setStyleSheet(self.button_select_all)
        self.stop_btn.setFixedSize(QSize(60, 60))
        self.stop_btn.setIconSize(QSize(30, 30))

        self.start_btn = QPushButton()
        self.start_btn.clicked.connect(self.start)
        self.start_btn.setIcon(icon(MDI6.play, color="black"))
        self.start_btn.setFixedSize(QSize(60, 60))
        self.start_btn.setStyleSheet(self.button_select_all)
        self.start_btn.setIconSize(QSize(30, 30))
        self.start_btn.hide()

        animation_buttons_box.addWidget(
            self.first_frame_btn, 5, alignment=Qt.AlignRight
        )
        animation_buttons_box.addWidget(self.stop_btn, 5, alignment=Qt.AlignRight)
        animation_buttons_box.addWidget(self.start_btn, 5, alignment=Qt.AlignRight)
        animation_buttons_box.addWidget(self.last_frame_btn, 5, alignment=Qt.AlignRight)

        self.right_panel.addLayout(animation_buttons_box, 5)
        self.right_panel.addWidget(self.fcanvas, 90)

        if not self.rgb_mode:
            contrast_hbox = QHBoxLayout()
            contrast_hbox.setContentsMargins(150, 5, 150, 5)
            self.contrast_slider = QLabeledDoubleRangeSlider()
            self.contrast_slider.setSingleStep(0.001)
            self.contrast_slider.setTickInterval(0.001)
            self.contrast_slider.setOrientation(Qt.Horizontal)
            self.contrast_slider.setRange(
                *[
                    np.nanpercentile(self.stack, 0.001),
                    np.nanpercentile(self.stack, 99.999),
                ]
            )
            self.contrast_slider.setValue(
                [np.nanpercentile(self.stack, 1), np.nanpercentile(self.stack, 99.99)]
            )
            self.contrast_slider.valueChanged.connect(self.contrast_slider_action)
            contrast_hbox.addWidget(QLabel("contrast: "))
            contrast_hbox.addWidget(self.contrast_slider, 90)
            self.right_panel.addLayout(contrast_hbox, 5)

        if self.class_choice_cb.currentText() != "":
            self.compute_status_and_colors(0)

        self.no_event_shortcut = QShortcut(QKeySequence("n"), self)  # QKeySequence("s")
        self.no_event_shortcut.activated.connect(self.shortcut_no_event)
        self.no_event_shortcut.setEnabled(False)

        QApplication.processEvents()

        # Add Menu for Interactive Plotter
        menubar = self.menuBar()
        viewMenu = menubar.addMenu("View")

        openPlotterAct = QAction("Interactive Plotter", self)
        openPlotterAct.setShortcut("Ctrl+P")
        openPlotterAct.setStatusTip("Open interactive signal plotter for corrections")
        openPlotterAct.triggered.connect(self.launch_interactive_viewer)
        viewMenu.addAction(openPlotterAct)

    def launch_interactive_viewer(self):
        if (
            not hasattr(self, "plotter")
            or self.plotter is None
            or not self.plotter.isVisible()
        ):
            label = None
            if hasattr(self, "class_name") and self.class_name.startswith("class_"):
                label = self.class_name.replace("class_", "")

            # Create with shared DF and callback
            self.plotter = InteractiveEventViewer(
                self.trajectories_path,
                df=self.df_tracks,
                event_label=label,
                callback=self.on_viewer_update,
                parent=self,
            )
        self.plotter.show()
        self.plotter.activateWindow()

    def on_viewer_update(self):
        """Callback from interactive viewer to refresh annotator."""
        self.compute_status_and_colors(0)
        self.update_scatters_only()
        self.fcanvas.canvas.draw_idle()

    def update_scatters_only(self):
        """Update only the scatters without reloading the image."""
        self.status_scatter.set_offsets(self.positions[self.framedata])
        self.status_scatter.set_color(self.colors[self.framedata][:, 1])
        self.class_scatter.set_offsets(self.positions[self.framedata])
        self.class_scatter.set_edgecolor(self.colors[self.framedata][:, 0])

    def compute_status_and_colors(self, i):

        self.class_name = self.class_choice_cb.currentText()
        if self.class_name == "":
            self.class_name = "class"

        suffix = self.class_name.replace("class", "").replace("_", "", 1)
        if suffix != "":
            self.expected_status = "status_" + suffix
            self.expected_time = "t_" + suffix
        else:
            self.expected_status = "status"
            self.expected_time = "t0"
        self.expected_class = self.class_name

        self.time_name = self.expected_time
        self.status_name = self.expected_status

        cols = list(self.df_tracks.columns)

        if (
            self.time_name in cols
            and self.class_name in cols
            and not self.status_name in cols
        ):
            # only create the status column if it does not exist to not erase static classification results
            self.make_status_column()
        elif (
            self.time_name in cols
            and self.class_name in cols
            and self.df_tracks[self.status_name].isnull().all()
        ):
            self.make_status_column()
        elif self.time_name in cols and self.class_name in cols:
            # all good, do nothing
            pass
        else:
            if not self.status_name in self.df_tracks.columns:
                self.df_tracks[self.status_name] = 0
                self.df_tracks["status_color"] = color_from_status(0)
                self.df_tracks["class_color"] = color_from_class(1)

        if not self.class_name in self.df_tracks.columns:
            self.df_tracks[self.class_name] = 1
        if not self.time_name in self.df_tracks.columns:
            self.df_tracks[self.time_name] = -1

        self.df_tracks["status_color"] = [
            color_from_status(i) for i in self.df_tracks[self.status_name].to_numpy()
        ]
        self.df_tracks["class_color"] = [
            color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()
        ]

        self.extract_scatter_from_trajectories()
        if len(self.selection) > 0:
            self.select_single_cell(self.selection[0][0], self.selection[0][1])

        self.fcanvas.canvas.draw()

    # def close_without_new_class(self):
    # 	self.newClassWidget.close()

    def cancel_selection(self):

        super().cancel_selection()
        try:
            for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
                self.colors[t][idx, 1] = self.previous_color[k][1]
        except Exception as e:
            pass

    def hide_annotation_buttons(self):

        for a in self.annotation_btns_to_hide:
            a.hide()
        for b in [self.event_btn, self.no_event_btn, self.else_btn, self.suppr_btn]:
            b.setChecked(False)
        self.time_of_interest_label.setEnabled(False)
        self.time_of_interest_le.setText("")
        self.time_of_interest_le.setEnabled(False)

    def enable_time_of_interest(self):

        if self.event_btn.isChecked():
            self.time_of_interest_label.setEnabled(True)
            self.time_of_interest_le.setEnabled(True)
        else:
            self.time_of_interest_label.setEnabled(False)
            self.time_of_interest_le.setEnabled(False)

    def show_annotation_buttons(self):

        for a in self.annotation_btns_to_hide:
            a.show()

        cclass = self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, self.class_name
        ].to_numpy()[0]
        t0 = self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, self.time_name
        ].to_numpy()[0]

        if cclass == 0:
            self.event_btn.setChecked(True)
            self.time_of_interest_le.setText(str(t0))
        elif cclass == 1:
            self.no_event_btn.setChecked(True)
        elif cclass == 2:
            self.else_btn.setChecked(True)
        elif cclass > 2:
            self.suppr_btn.setChecked(True)

        self.enable_time_of_interest()
        self.correct_btn.setText("submit")

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.apply_modification)

    def apply_modification(self):

        t0 = -1
        if self.event_btn.isChecked():
            cclass = 0
            try:
                t0 = float(self.time_of_interest_le.text().replace(",", "."))
                self.line_dt.set_xdata([t0, t0])
                self.cell_fcanvas.canvas.draw_idle()
            except Exception as e:
                print(f"L 598 {e=}")
                t0 = -1
                cclass = 2
        elif self.no_event_btn.isChecked():
            cclass = 1
        elif self.else_btn.isChecked():
            cclass = 2
        elif self.suppr_btn.isChecked():
            cclass = 42

        self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, self.class_name
        ] = cclass
        self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, self.time_name
        ] = t0

        indices = self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, self.class_name
        ].index
        timeline = self.df_tracks.loc[
            self.df_tracks["TRACK_ID"] == self.track_of_interest, "FRAME"
        ].to_numpy()
        status = np.zeros_like(timeline)
        if t0 > 0:
            status[timeline >= t0] = 1.0
        if cclass == 2:
            status[:] = 2
        if cclass > 2:
            status[:] = 42

        status_color = [color_from_status(s, recently_modified=True) for s in status]
        class_color = [
            color_from_class(cclass, recently_modified=True) for i in range(len(status))
        ]

        # self.df_tracks['status_color'] = [color_from_status(i) for i in self.df_tracks[self.status_name].to_numpy()]
        # self.df_tracks['class_color'] = [color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()]

        self.df_tracks.loc[indices, self.status_name] = status
        self.df_tracks.loc[indices, "status_color"] = status_color
        self.df_tracks.loc[indices, "class_color"] = class_color

        # self.make_status_column()
        self.extract_scatter_from_trajectories()
        self.give_cell_information()

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.show_annotation_buttons)
        # self.cancel_btn.click()

        self.hide_annotation_buttons()
        self.correct_btn.setEnabled(False)
        self.correct_btn.setText("correct")
        self.cancel_btn.setEnabled(False)
        self.del_shortcut.setEnabled(False)
        self.no_event_shortcut.setEnabled(False)

        self.selection.pop(0)

    def make_status_column(self):

        print(
            f"Generating status information for class `{self.class_name}` and time `{self.time_name}`..."
        )
        for tid, group in self.df_tracks.groupby("TRACK_ID"):

            indices = group.index
            t0 = group[self.time_name].to_numpy()[0]
            cclass = group[self.class_name].to_numpy()[0]
            timeline = group["FRAME"].to_numpy()
            status = np.zeros_like(timeline)

            if t0 > 0:
                status[timeline >= t0] = 1.0
            # if cclass == 2:
            # 	status[:] = 1.
            if cclass > 2:
                status[:] = 42

            status_color = [color_from_status(s) for s in status]
            class_color = [color_from_class(cclass) for i in range(len(status))]

            self.df_tracks.loc[indices, self.status_name] = status
            self.df_tracks.loc[indices, "status_color"] = status_color
            self.df_tracks.loc[indices, "class_color"] = class_color

    def generate_signal_choices(self):

        self.signal_choice_cb = [QSearchableComboBox() for i in range(self.n_signals)]
        self.signal_choice_label = [
            QLabel(f"signal {i + 1}: ") for i in range(self.n_signals)
        ]
        # self.log_btns = [QPushButton() for i in range(self.n_signals)]

        signals = list(self.df_tracks.columns)

        to_remove = [
            "TRACK_ID",
            "FRAME",
            "x_anim",
            "y_anim",
            "t",
            "state",
            "generation",
            "root",
            "parent",
            "class_id",
            "class",
            "t0",
            "POSITION_X",
            "POSITION_Y",
            "position",
            "well",
            "well_index",
            "well_name",
            "pos_name",
            "index",
            "class_color",
            "status_color",
            "dummy",
            "group_color",
        ]

        meta = get_experiment_metadata(self.exp_dir)
        if meta is not None:
            keys = list(meta.keys())
            to_remove.extend(keys)

        labels = get_experiment_labels(self.exp_dir)
        if labels is not None:
            keys = list(labels.keys())
            to_remove.extend(labels)

        for c in to_remove:
            if c in signals:
                signals.remove(c)

        for i in range(len(self.signal_choice_cb)):
            self.signal_choice_cb[i].addItems(["--"] + signals)
            self.signal_choice_cb[i].setCurrentIndex(i + 1)
            self.signal_choice_cb[i].currentIndexChanged.connect(self.plot_signals)

    def plot_signals(self):

        range_values = []

        try:
            yvalues = []
            for i in range(len(self.signal_choice_cb)):

                signal_choice = self.signal_choice_cb[i].currentText()
                lbl = signal_choice
                n_cut = 35
                if len(lbl) > n_cut:
                    lbl = lbl[: (n_cut - 3)] + "..."
                self.lines[i].set_label(lbl)

                if signal_choice == "--":
                    self.lines[i].set_xdata([])
                    self.lines[i].set_ydata([])
                else:
                    xdata = self.df_tracks.loc[
                        self.df_tracks["TRACK_ID"] == self.track_of_interest, "FRAME"
                    ].to_numpy()
                    ydata = self.df_tracks.loc[
                        self.df_tracks["TRACK_ID"] == self.track_of_interest,
                        signal_choice,
                    ].to_numpy()

                    range_values.extend(ydata)

                    xdata = xdata[ydata == ydata]  # remove nan
                    ydata = ydata[ydata == ydata]

                    yvalues.extend(ydata)
                    self.lines[i].set_xdata(xdata)
                    self.lines[i].set_ydata(ydata)
                    self.lines[i].set_color(tab10(i / 3.0))

            self.configure_ylims()

            min_val, max_val = self.cell_ax.get_ylim()
            t0 = self.df_tracks.loc[
                self.df_tracks["TRACK_ID"] == self.track_of_interest, self.expected_time
            ].to_numpy()[0]
            self.line_dt.set_xdata([t0, t0])
            self.line_dt.set_ydata([min_val, max_val])

            self.cell_ax.legend(fontsize=8)
            self.cell_fcanvas.canvas.draw()
        except Exception as e:
            print(e)
            pass

        if len(range_values) > 0:
            range_values = np.array(range_values)
            if len(range_values[range_values == range_values]) > 0:
                if len(range_values[range_values > 0]) > 0:
                    self.value_magnitude = np.nanpercentile(range_values, 1)
                else:
                    self.value_magnitude = 1
                self.non_log_ymin = 0.98 * np.nanmin(range_values)
                self.non_log_ymax = np.nanmax(range_values) * 1.02
                if self.cell_ax.get_yscale() == "linear":
                    self.cell_ax.set_ylim(self.non_log_ymin, self.non_log_ymax)
                else:
                    self.cell_ax.set_ylim(self.value_magnitude, self.non_log_ymax)

    def extract_scatter_from_trajectories(self):

        self.positions = []
        self.colors = []
        self.tracks = []

        for t in np.arange(self.len_movie):
            self.positions.append(
                self.df_tracks.loc[
                    self.df_tracks["FRAME"] == t, ["x_anim", "y_anim"]
                ].to_numpy()
            )
            self.colors.append(
                self.df_tracks.loc[
                    self.df_tracks["FRAME"] == t, ["class_color", "status_color"]
                ].to_numpy()
            )
            self.tracks.append(
                self.df_tracks.loc[self.df_tracks["FRAME"] == t, "TRACK_ID"].to_numpy()
            )

    # def load_annotator_config(self):
    #
    # 	"""
    # 	Load settings from config or set default values.
    # 	"""
    #
    # 	if os.path.exists(self.instructions_path):
    # 		with open(self.instructions_path, 'r') as f:
    #
    # 			instructions = json.load(f)
    #
    # 			if 'rgb_mode' in instructions:
    # 				self.rgb_mode = instructions['rgb_mode']
    # 			else:
    # 				self.rgb_mode = False
    #
    # 			if 'percentile_mode' in instructions:
    # 				self.percentile_mode = instructions['percentile_mode']
    # 			else:
    # 				self.percentile_mode = True
    #
    # 			if 'channels' in instructions:
    # 				self.target_channels = instructions['channels']
    # 			else:
    # 				self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
    #
    # 			if 'fraction' in instructions:
    # 				self.fraction = float(instructions['fraction'])
    # 			else:
    # 				self.fraction = 0.25
    #
    # 			if 'interval' in instructions:
    # 				self.anim_interval = int(instructions['interval'])
    # 			else:
    # 				self.anim_interval = 1
    #
    # 			if 'log' in instructions:
    # 				self.log_option = instructions['log']
    # 			else:
    # 				self.log_option = False
    # 	else:
    # 		self.rgb_mode = False
    # 		self.log_option = False
    # 		self.percentile_mode = True
    # 		self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
    # 		self.fraction = 0.25
    # 		self.anim_interval = 1

    def prepare_stack(self, progress_callback=None):

        self.img_num_channels = _get_img_num_per_channel(
            self.channels, self.len_movie, self.nbr_channels
        )
        self.stack = []
        disable_tqdm = not len(self.target_channels) > 1

        # Calculate total frames for progress
        total_frames = 0
        for ch in self.target_channels:
            target_ch_name = ch[0]
            indices = self.img_num_channels[
                self.channels[np.where(self.channel_names == target_ch_name)][0]
            ]
            total_frames += len(indices)

        current_frame = 0

        for ch in tqdm(self.target_channels, desc="channel", disable=disable_tqdm):
            target_ch_name = ch[0]
            if self.percentile_mode:
                normalize_kwargs = {"percentiles": (ch[1], ch[2]), "values": None}
            else:
                normalize_kwargs = {"values": (ch[1], ch[2]), "percentiles": None}

            if self.rgb_mode:
                normalize_kwargs.update({"amplification": 255.0, "clip": True})

            chan = []
            indices = self.img_num_channels[
                self.channels[np.where(self.channel_names == target_ch_name)][0]
            ]

            if progress_callback:
                if not progress_callback(
                    int((current_frame / total_frames) * 100),
                    f"Loading channel {target_ch_name}...",
                ):
                    return

            for t in tqdm(range(len(indices)), desc="frame"):

                if progress_callback:
                    if not progress_callback(int((current_frame / total_frames) * 100)):
                        return
                    current_frame += 1

                if self.rgb_mode:
                    f = load_frames(
                        indices[t],
                        self.stack_path,
                        scale=self.fraction,
                        normalize_input=True,
                        normalize_kwargs=normalize_kwargs,
                    )
                    f = f.astype(np.uint8)
                else:
                    f = load_frames(
                        indices[t],
                        self.stack_path,
                        scale=self.fraction,
                        normalize_input=False,
                    )

                chan.append(f[:, :, 0])

            self.stack.append(chan)

        self.stack = np.array(self.stack)
        if self.rgb_mode:
            self.stack = np.moveaxis(self.stack, 0, -1)
        else:
            self.stack = self.stack[0]
            if self.log_option:
                self.stack[np.where(self.stack > 0.0)] = np.log(
                    self.stack[np.where(self.stack > 0.0)]
                )

    def closeEvent(self, event):

        try:
            self.stop()
            del self.stack
            gc.collect()
        except:
            pass

    def looped_animation(self):
        """
        Load an image.

        """

        self.framedata = 0

        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.fcanvas = FigureCanvas(self.fig, interactive=True)
        self.ax.clear()

        self.im = self.ax.imshow(self.stack[0], cmap="gray", interpolation="none")
        self.status_scatter = self.ax.scatter(
            self.positions[0][:, 0],
            self.positions[0][:, 1],
            marker="x",
            c=self.colors[0][:, 1],
            s=50,
            picker=True,
            pickradius=100,
        )
        self.class_scatter = self.ax.scatter(
            self.positions[0][:, 0],
            self.positions[0][:, 1],
            marker="o",
            facecolors="none",
            edgecolors=self.colors[0][:, 0],
            s=200,
        )

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal")

        self.fig.set_facecolor("none")  # or 'None'
        self.fig.canvas.setStyleSheet("background-color: black;")

        self.anim = FuncAnimation(
            self.fig,
            self.draw_frame,
            frames=self.len_movie,  # better would be to cast np.arange(len(movie)) in case frame column is incomplete
            interval=self.anim_interval,  # in ms
            blit=True,
        )

        self.fig.canvas.mpl_connect("pick_event", self.on_scatter_pick)
        self.fcanvas.canvas.draw()

    def select_single_cell(self, index, timepoint):

        self.correct_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.del_shortcut.setEnabled(True)
        self.no_event_shortcut.setEnabled(True)

        self.track_of_interest = self.tracks[timepoint][index]
        print(f"You selected cell #{self.track_of_interest}...")
        self.give_cell_information()
        self.plot_signals()

        self.loc_t = []
        self.loc_idx = []
        for t in range(len(self.tracks)):
            indices = np.where(self.tracks[t] == self.track_of_interest)[0]
            if len(indices) > 0:
                self.loc_t.append(t)
                self.loc_idx.append(indices[0])

        self.previous_color = []
        for t, idx in zip(self.loc_t, self.loc_idx):
            self.previous_color.append(self.colors[t][idx].copy())
            self.colors[t][idx] = "lime"

    def shortcut_no_event(self):
        self.correct_btn.click()
        self.no_event_btn.click()
        self.correct_btn.click()

    def configure_ylims(self):

        try:
            min_values = []
            max_values = []
            feats = []
            for i in range(len(self.signal_choice_cb)):
                signal = self.signal_choice_cb[i].currentText()
                if signal == "--":
                    continue
                else:
                    maxx = np.nanpercentile(
                        self.df_tracks.loc[:, signal].to_numpy().flatten(), 99
                    )
                    minn = np.nanpercentile(
                        self.df_tracks.loc[:, signal].to_numpy().flatten(), 1
                    )
                    min_values.append(minn)
                    max_values.append(maxx)
                    feats.append(signal)

            smallest_value = np.amin(min_values)
            feat_smallest_value = feats[np.argmin(min_values)]
            min_feat = self.df_tracks[feat_smallest_value].min()
            max_feat = self.df_tracks[feat_smallest_value].max()
            pad_small = (max_feat - min_feat) * 0.05
            if pad_small == 0:
                pad_small = 0.05

            largest_value = np.amax(max_values)
            feat_largest_value = feats[np.argmax(max_values)]
            min_feat = self.df_tracks[feat_largest_value].min()
            max_feat = self.df_tracks[feat_largest_value].max()
            pad_large = (max_feat - min_feat) * 0.05
            if pad_large == 0:
                pad_large = 0.05

            if len(min_values) > 0:
                self.cell_ax.set_ylim(
                    smallest_value - pad_small, largest_value + pad_large
                )
        except Exception as e:
            print(f"L1170 {e=}")
            pass

    def draw_frame(self, framedata):
        """
        Update plot elements at each timestep of the loop.
        """

        self.framedata = framedata
        self.frame_lbl.setText(f"frame: {self.framedata}")
        self.im.set_array(self.stack[self.framedata])
        self.status_scatter.set_offsets(self.positions[self.framedata])
        self.status_scatter.set_color(self.colors[self.framedata][:, 1])

        self.class_scatter.set_offsets(self.positions[self.framedata])
        self.class_scatter.set_edgecolor(self.colors[self.framedata][:, 0])

        return (
            self.im,
            self.status_scatter,
            self.class_scatter,
        )

    def stop(self):
        # # On stop we disconnect all of our events.
        self.stop_btn.hide()
        self.start_btn.show()
        self.anim.pause()
        self.stop_btn.clicked.connect(self.start)

    def give_cell_information(self):

        cell_selected = f"cell: {self.track_of_interest}\n"
        cell_class = f"class: {self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.class_name].to_numpy()[0]}\n"
        cell_time = f"time of interest: {self.df_tracks.loc[self.df_tracks['TRACK_ID'] == self.track_of_interest, self.time_name].to_numpy()[0]}\n"
        self.cell_info.setText(cell_selected + cell_class + cell_time)

    def save_trajectories(self):

        if self.normalized_signals:
            self.normalize_features_btn.click()
        if self.selection:
            self.cancel_selection()

        self.df_tracks = self.df_tracks.drop(
            self.df_tracks[self.df_tracks[self.class_name] > 2].index
        )
        self.df_tracks.to_csv(self.trajectories_path, index=False)
        print("Table successfully exported...")
        if self.class_choice_cb.currentText() != "":
            self.compute_status_and_colors(0)
        self.extract_scatter_from_trajectories()

    def set_last_frame(self):

        self.last_frame_btn.setEnabled(False)
        self.last_frame_btn.disconnect()

        self.last_key = len(self.stack) - 1
        while len(np.where(self.stack[self.last_key].flatten() == 0)[0]) > 0.99 * len(
            self.stack[self.last_key].flatten()
        ):
            self.last_key -= 1
        self.anim._drawn_artists = self.draw_frame(self.last_key)
        self.anim._drawn_artists = sorted(
            self.anim._drawn_artists, key=lambda x: x.get_zorder()
        )
        for a in self.anim._drawn_artists:
            a.set_visible(True)

        self.fig.canvas.draw()
        self.anim.event_source.stop()

        # self.cell_plot.draw()
        self.stop_btn.hide()
        self.start_btn.show()
        self.stop_btn.clicked.connect(self.start)
        self.start_btn.setShortcut(QKeySequence("l"))

    def set_first_frame(self):

        self.first_frame_btn.setEnabled(False)
        self.first_frame_btn.disconnect()

        self.first_key = 0
        self.anim._drawn_artists = self.draw_frame(0)
        self.vmin = self.contrast_slider.value()[0]
        self.vmax = self.contrast_slider.value()[1]
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)

        self.anim._drawn_artists = sorted(
            self.anim._drawn_artists, key=lambda x: x.get_zorder()
        )
        for a in self.anim._drawn_artists:
            a.set_visible(True)

        self.fig.canvas.draw()
        self.anim.event_source.stop()

        # self.cell_plot.draw()
        self.stop_btn.hide()
        self.start_btn.show()
        self.stop_btn.clicked.connect(self.start)
        self.start_btn.setShortcut(QKeySequence("f"))


