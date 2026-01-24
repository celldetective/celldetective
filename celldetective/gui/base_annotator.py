from glob import glob
import os
import numpy as np
import pandas as pd
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QHBoxLayout,
    QRadioButton,
    QShortcut,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
)
from PyQt5.QtCore import QSize, Qt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import MinMaxScaler
from superqt import QSearchableComboBox

from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import json

from celldetective.gui.base.styles import Styles
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    CelldetectiveMainWindow,
)
from celldetective import get_software_location
from celldetective.utils.image_loaders import auto_load_number_of_frames, load_frames
from celldetective.utils.experiment import (
    extract_experiment_channels,
    get_experiment_metadata,
    get_experiment_labels,
)
from celldetective.gui.gui_utils import (
    color_from_status,
    color_from_class,
    ExportPlotBtn,
)
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.utils import center_window
import gc
from celldetective import get_logger

logger = get_logger(__name__)


class BaseAnnotator(CelldetectiveMainWindow, Styles):

    def __init__(self, parent_window=None, read_config=True):

        super().__init__()
        self.parent_window = parent_window
        self.read_config = read_config

        self.mode = self.parent_window.mode
        self.pos = self.parent_window.parent_window.pos
        self.exp_dir = self.parent_window.exp_dir
        self.PxToUm = self.parent_window.parent_window.PxToUm
        self.n_signals = 3
        self.soft_path = get_software_location()
        self.recently_modified = False
        self.selection = []
        self.MinMaxScaler = MinMaxScaler()
        self.fraction = 1
        self.log_scale = False

        self.instructions_path = self.exp_dir + os.sep.join(
            ["configs", f"signal_annotator_config_{self.mode}.json"]
        )
        self.trajectories_path = self.pos + os.sep.join(
            ["output", "tables", f"trajectories_{self.mode}.csv"]
        )

        self.screen_height = (
            self.parent_window.parent_window.parent_window.screen_height
        )
        self.screen_width = self.parent_window.parent_window.parent_window.screen_width
        self.value_magnitude = 1

        self.setMinimumWidth(int(0.8 * self.screen_width))
        self.setMinimumHeight(int(0.8 * self.screen_height))

        self.proceed = True
        self.locate_stack()
        if not self.proceed:
            self.close()
        else:
            if self.read_config:
                self.load_annotator_config()
            self.locate_tracks()
            self._init_base_widgets()

    def _init_base_widgets(self):
        self._init_base_widgets_left()
        self._init_base_widgets_right()

    def _init_base_widgets_right(self):
        pass

    def _init_base_widgets_left(self):

        self.class_label = QLabel("event: ")
        self.class_choice_cb = QComboBox()

        cols = np.array(self.df_tracks.columns)
        self.class_cols = np.array(
            [c.startswith("class") for c in list(self.df_tracks.columns)]
        )

        self.class_cols = list(cols[self.class_cols])
        try:
            self.class_cols.remove("class_id")
        except Exception:
            pass
        try:
            self.class_cols.remove("class_color")
        except Exception:
            pass

        self.class_choice_cb.addItems(self.class_cols)
        self.class_choice_cb.currentIndexChanged.connect(self.compute_status_and_colors)

        self.add_class_btn = QPushButton("")
        self.add_class_btn.setStyleSheet(self.button_select_all)
        self.add_class_btn.setIcon(icon(MDI6.plus, color="black"))
        self.add_class_btn.setToolTip("Add a new event class")
        self.add_class_btn.setIconSize(QSize(20, 20))
        self.add_class_btn.clicked.connect(self.create_new_event_class)

        self.del_class_btn = QPushButton("")
        self.del_class_btn.setStyleSheet(self.button_select_all)
        self.del_class_btn.setIcon(icon(MDI6.delete, color="black"))
        self.del_class_btn.setToolTip("Delete an event class")
        self.del_class_btn.setIconSize(QSize(20, 20))
        self.del_class_btn.clicked.connect(self.del_event_class)

        self.cell_info = QLabel("")

        self.correct_btn = QPushButton("correct")
        self.correct_btn.setIcon(icon(MDI6.redo_variant, color="white"))
        self.correct_btn.setIconSize(QSize(20, 20))
        self.correct_btn.setStyleSheet(self.button_style_sheet)
        self.correct_btn.clicked.connect(self.show_annotation_buttons)
        self.correct_btn.setEnabled(False)

        self.cancel_btn = QPushButton("cancel")
        self.cancel_btn.setStyleSheet(self.button_style_sheet_2)
        self.cancel_btn.setShortcut(QKeySequence("Esc"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_selection)

        self._init_cell_fig_widgets()

        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet(self.button_style_sheet)
        self.save_btn.clicked.connect(self.save_trajectories)

        self.export_btn = QPushButton("")
        self.export_btn.setStyleSheet(self.button_select_all)
        self.export_btn.clicked.connect(self.export_signals)
        self.export_btn.setIcon(icon(MDI6.export, color="black"))
        self.export_btn.setIconSize(QSize(25, 25))

    def _init_cell_fig_widgets(self):

        self.generate_signal_choices()
        self.create_cell_signal_canvas()
        self.cell_fcanvas.setMinimumHeight(int(0.2 * self.screen_height))

        self.outliers_check = QCheckBox("Show outliers")
        self.outliers_check.toggled.connect(self.show_outliers)

        self.normalize_features_btn = QPushButton("")
        self.normalize_features_btn.setStyleSheet(self.button_select_all)
        self.normalize_features_btn.setIcon(
            icon(MDI6.arrow_collapse_vertical, color="black")
        )
        self.normalize_features_btn.setIconSize(QSize(25, 25))
        self.normalize_features_btn.setFixedSize(QSize(30, 30))
        self.normalize_features_btn.clicked.connect(self.normalize_features)
        self.normalized_signals = False

        self.log_btn = QPushButton()
        self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
        self.log_btn.setStyleSheet(self.button_select_all)
        self.log_btn.clicked.connect(self.switch_to_log)

        self.export_plot_btn = ExportPlotBtn(self.cell_fig, export_dir=self.exp_dir)

    def close_without_new_class(self):
        self.newClassWidget.close()

    def init_class_selection_block(self):

        self.class_hbox = QHBoxLayout()
        self.class_hbox.setContentsMargins(0, 0, 0, 0)

        self.class_hbox.addWidget(self.class_label, 25)
        self.class_hbox.addWidget(self.class_choice_cb, 70)
        self.class_hbox.addWidget(self.add_class_btn, 5)
        self.class_hbox.addWidget(self.del_class_btn, 5)

    def init_options_block(self):
        self.options_hbox = QHBoxLayout()
        self.options_hbox.setContentsMargins(0, 0, 0, 0)

    def init_correction_block(self):
        self.action_hbox = QHBoxLayout()
        self.action_hbox.setContentsMargins(0, 0, 0, 0)

        self.action_hbox.addWidget(self.correct_btn)
        self.action_hbox.addWidget(self.cancel_btn)

    def init_plot_buttons_block(self):
        self.plot_buttons_hbox = QHBoxLayout()
        self.plot_buttons_hbox.setContentsMargins(0, 0, 0, 0)
        self.plot_buttons_hbox.addWidget(QLabel(""), 90)
        self.plot_buttons_hbox.addWidget(self.outliers_check, 5)
        self.plot_buttons_hbox.addWidget(self.normalize_features_btn, 5)
        self.plot_buttons_hbox.addWidget(self.log_btn, 5)
        self.plot_buttons_hbox.addWidget(self.export_plot_btn, 5)

    def init_save_btn_block(self):

        self.btn_hbox = QHBoxLayout()
        self.btn_hbox.setContentsMargins(0, 10, 0, 0)
        self.btn_hbox.addWidget(self.save_btn, 90)
        self.btn_hbox.addWidget(self.export_btn, 10)

    def populate_window(self):
        """
        Create the multibox design.

        """

        # Main layout
        self.button_widget = CelldetectiveWidget()
        self.main_layout = QHBoxLayout()
        self.button_widget.setLayout(self.main_layout)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Left panel
        self.left_panel = QVBoxLayout()
        self.left_panel.setContentsMargins(30, 5, 30, 5)
        self.left_panel.setSpacing(3)

        self.init_class_selection_block()
        self.left_panel.addLayout(self.class_hbox, 5)
        self.left_panel.addWidget(self.cell_info, 10)

        self.init_options_block()
        self.init_correction_block()
        self.left_panel.addLayout(self.options_hbox, 5)
        self.left_panel.addLayout(self.action_hbox, 5)

        self.left_panel.addWidget(self.cell_fcanvas, 45)

        self.init_plot_buttons_block()
        self.left_panel.addLayout(self.plot_buttons_hbox, 5)

        signal_choice_vbox = QVBoxLayout()
        signal_choice_vbox.setContentsMargins(30, 0, 30, 0)
        for i in range(len(self.signal_choice_cb)):
            hlayout = QHBoxLayout()
            hlayout.addWidget(self.signal_choice_label[i], 20)
            hlayout.addWidget(self.signal_choice_cb[i], 75)
            signal_choice_vbox.addLayout(hlayout)

        self.left_panel.addLayout(signal_choice_vbox, 15)

        self.init_save_btn_block()
        self.left_panel.addLayout(self.btn_hbox, 5)

        # Right panel
        self.right_panel = QVBoxLayout()

        # add left/right to main layout
        self.main_layout.addLayout(self.left_panel, 35)
        self.main_layout.addLayout(self.right_panel, 65)
        self.button_widget.adjustSize()
        # self.compute_status_and_colors(0)

        self.setCentralWidget(self.button_widget)
        # self.show()

        self.del_shortcut = QShortcut(Qt.Key_Delete, self)  # QKeySequence("s")
        self.del_shortcut.activated.connect(self.shortcut_suppr)
        self.del_shortcut.setEnabled(False)

        QApplication.processEvents()

    def generate_signal_choices(self):

        self.signal_choice_cb = [QSearchableComboBox() for i in range(self.n_signals)]
        self.signal_choice_label = [
            QLabel(f"signal {i + 1}: ") for i in range(self.n_signals)
        ]
        # self.log_btns = [QPushButton() for i in range(self.n_signals)]

        signals = [
            c
            for c in self.df_tracks.columns
            if pd.api.types.is_numeric_dtype(self.df_tracks[c])
        ]

        to_remove = [
            "FRAME",
            "ID",
            "POSITION_X",
            "POSITION_Y",
            "TRACK_ID",
            "antibody",
            "cell_type",
            "class",
            "class_color",
            "class_id",
            "concentration",
            "dummy",
            "generation",
            "group",
            "group_color",
            "index",
            "parent",
            "pharmaceutical_agent",
            "pos_name",
            "position",
            "root",
            "state",
            "status",
            "status_color",
            "t",
            "t0",
            "well",
            "well_index",
            "well_name",
            "x_anim",
            "y_anim",
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
            if i + 1 < self.signal_choice_cb[i].count():
                self.signal_choice_cb[i].setCurrentIndex(i + 1)
            else:
                self.signal_choice_cb[i].setCurrentIndex(0)
            self.signal_choice_cb[i].currentIndexChanged.connect(self.plot_signals)

    def on_scatter_pick(self, event):

        self.event = event

        self.correct_btn.disconnect()
        self.correct_btn.clicked.connect(self.show_annotation_buttons)

        logger.info(f"{self.selection=}")

        ind = event.ind

        if len(ind) > 1:
            # More than one point in vicinity
            datax, datay = [self.positions[self.framedata][i, 0] for i in ind], [
                self.positions[self.framedata][i, 1] for i in ind
            ]
            msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
            dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
            ind = [ind[np.argmin(dist)]]

        if len(ind) > 0 and (len(self.selection) == 0):

            self.selection.append([ind[0], self.framedata])
            self.select_single_cell(ind[0], self.framedata)

        elif len(ind) > 0 and len(self.selection) == 1:
            self.cancel_btn.click()
        else:
            pass

        # self.draw_frame(self.current_frame)
        # self.fcanvas.canvas.draw()
        #

    def load_annotator_config(self):
        """
        Load settings from config or set default values.
        """

        if os.path.exists(self.instructions_path):
            with open(self.instructions_path, "r") as f:

                instructions = json.load(f)

                if "rgb_mode" in instructions:
                    self.rgb_mode = instructions["rgb_mode"]
                else:
                    self.rgb_mode = False

                if "percentile_mode" in instructions:
                    self.percentile_mode = instructions["percentile_mode"]
                else:
                    self.percentile_mode = True

                if "channels" in instructions:
                    self.target_channels = instructions["channels"]
                else:
                    self.target_channels = [[self.channel_names[0], 0.01, 99.99]]

                if "fraction" in instructions:
                    self.fraction = float(instructions["fraction"])
                else:
                    self.fraction = 0.25

                if "interval" in instructions:
                    self.anim_interval = int(instructions["interval"])
                else:
                    self.anim_interval = 1

                if "log" in instructions:
                    self.log_option = instructions["log"]
                else:
                    self.log_option = False
        else:
            self.rgb_mode = False
            self.log_option = False
            self.percentile_mode = True
            self.target_channels = [[self.channel_names[0], 0.01, 99.99]]
            self.fraction = 0.25
            self.anim_interval = 1

    def locate_stack(self):
        """
        Locate the target movie.

        """

        movies = glob(
            self.pos
            + os.sep.join(
                ["movie", f"{self.parent_window.parent_window.movie_prefix}*.tif"]
            )
        )

        if len(movies) == 0:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText(
                "No movie is detected in the experiment folder.\nPlease check the stack prefix..."
            )
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                self.proceed = False
                self.close()
            else:
                self.close()
        else:
            self.stack_path = movies[0]
            self.len_movie = self.parent_window.parent_window.len_movie
            len_movie_auto = auto_load_number_of_frames(self.stack_path)
            if len_movie_auto is not None:
                self.len_movie = len_movie_auto
            exp_config = self.exp_dir + "config.ini"
            self.channel_names, self.channels = extract_experiment_channels(
                self.exp_dir
            )
            self.channel_names = np.array(self.channel_names)
            self.channels = np.array(self.channels)
            self.nbr_channels = len(self.channels)
            self.img = load_frames(0, self.stack_path, normalize_input=False)

    def create_cell_signal_canvas(self):

        self.cell_fig, self.cell_ax = plt.subplots(tight_layout=True)
        self.cell_fcanvas = FigureCanvas(self.cell_fig, interactive=True)
        self.cell_ax.clear()

        spacing = 0.5
        minorLocator = MultipleLocator(1)
        self.cell_ax.xaxis.set_minor_locator(minorLocator)
        self.cell_ax.xaxis.set_major_locator(MultipleLocator(5))
        self.cell_ax.grid(which="major")
        self.cell_ax.set_xlabel("time [frame]")
        self.cell_ax.set_ylabel("signal")

        self.cell_fig.set_facecolor("none")  # or 'None'
        self.cell_fig.canvas.setStyleSheet("background-color: transparent;")

        self.lines = [
            self.cell_ax.plot(
                [np.linspace(0, self.len_movie - 1, self.len_movie)],
                [np.zeros((self.len_movie))],
            )[0]
            for i in range(len(self.signal_choice_cb))
        ]
        for i in range(len(self.lines)):
            self.lines[i].set_label(f"signal {i}")

        min_val, max_val = self.cell_ax.get_ylim()
        (self.line_dt,) = self.cell_ax.plot(
            [-1, -1], [min_val, max_val], c="k", linestyle="--"
        )

        self.cell_ax.set_xlim(0, self.len_movie)
        self.cell_ax.legend(fontsize=8)
        self.cell_fcanvas.canvas.draw()

    def resizeEvent(self, event):

        super().resizeEvent(event)
        try:
            self.cell_fig.tight_layout()
        except:
            pass

    def locate_tracks(self):
        """
        Locate the tracks.
        """

        if not os.path.exists(self.trajectories_path):

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("The trajectories cannot be detected.")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Yes:
                self.close()
        else:

            # Load and prep tracks
            self.df_tracks = pd.read_csv(self.trajectories_path)
            self.df_tracks = self.df_tracks.sort_values(by=["TRACK_ID", "FRAME"])
            self.df_tracks.replace([np.inf, -np.inf], np.nan, inplace=True)

            cols = np.array(self.df_tracks.columns)
            self.class_cols = np.array(
                [c.startswith("class") for c in list(self.df_tracks.columns)]
            )
            self.class_cols = list(cols[self.class_cols])
            try:
                self.class_cols.remove("class_id")
            except:
                pass
            try:
                self.class_cols.remove("class_color")
            except:
                pass
            if len(self.class_cols) > 0:
                self.class_name = self.class_cols[0]
                self.expected_status = "status"
                suffix = self.class_name.replace("class", "").replace("_", "")
                if suffix != "":
                    self.expected_status += "_" + suffix
                    self.expected_time = "t_" + suffix
                else:
                    self.expected_time = "t0"
                self.time_name = self.expected_time
                self.status_name = self.expected_status
            else:
                self.class_name = "class"
                self.time_name = "t0"
                self.status_name = "status"

            if (
                self.time_name in self.df_tracks.columns
                and self.class_name in self.df_tracks.columns
                and not self.status_name in self.df_tracks.columns
            ):
                # only create the status column if it does not exist to not erase static classification results
                self.make_status_column()
            elif (
                self.time_name in self.df_tracks.columns
                and self.class_name in self.df_tracks.columns
            ):
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
                color_from_status(i)
                for i in self.df_tracks[self.status_name].to_numpy()
            ]
            self.df_tracks["class_color"] = [
                color_from_class(i) for i in self.df_tracks[self.class_name].to_numpy()
            ]

            self.df_tracks = self.df_tracks.dropna(subset=["POSITION_X", "POSITION_Y"])
            self.df_tracks["x_anim"] = self.df_tracks["POSITION_X"] * self.fraction
            self.df_tracks["y_anim"] = self.df_tracks["POSITION_Y"] * self.fraction
            self.df_tracks["x_anim"] = self.df_tracks["x_anim"].astype(int)
            self.df_tracks["y_anim"] = self.df_tracks["y_anim"].astype(int)

            self.extract_scatter_from_trajectories()
            self.track_of_interest = self.df_tracks["TRACK_ID"].min()

            self.loc_t = []
            self.loc_idx = []
            for t in range(len(self.tracks)):
                indices = np.where(self.tracks[t] == self.track_of_interest)[0]
                if len(indices) > 0:
                    self.loc_t.append(t)
                    self.loc_idx.append(indices[0])

            self.MinMaxScaler = MinMaxScaler()
            self.columns_to_rescale = list(self.df_tracks.columns)

            cols_to_remove = [
                "group",
                "group_color",
                "status",
                "status_color",
                "class_color",
                "TRACK_ID",
                "FRAME",
                "x_anim",
                "y_anim",
                "t",
                "dummy",
                "group_color",
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
                "concentration",
                "cell_type",
                "antibody",
                "pharmaceutical_agent",
                "ID",
            ] + self.class_cols

            meta = get_experiment_metadata(self.exp_dir)
            if meta is not None:
                keys = list(meta.keys())
                cols_to_remove.extend(keys)

            labels = get_experiment_labels(self.exp_dir)
            if labels is not None:
                keys = list(labels.keys())
                cols_to_remove.extend(labels)

            cols = np.array(list(self.df_tracks.columns))
            time_cols = np.array([c.startswith("t_") for c in cols])
            time_cols = list(cols[time_cols])
            cols_to_remove += time_cols

            status_cols = np.array([c.startswith("status_") for c in cols])
            status_cols = list(cols[status_cols])
            cols_to_remove += status_cols

            for tr in cols_to_remove:
                try:
                    self.columns_to_rescale.remove(tr)
                except:
                    pass

            x = self.df_tracks[self.columns_to_rescale].values
            self.MinMaxScaler.fit(x)

    def del_event_class(self):

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(
            f"You are about to delete the class {self.class_choice_cb.currentText()}. The associated time and\nstatus will also be deleted. Do you still want to proceed?"
        )
        msgBox.setWindowTitle("Warning")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.No:
            return None
        else:
            class_to_delete = self.class_choice_cb.currentText()
            time_to_delete = class_to_delete.replace("class", "t")
            status_to_delete = class_to_delete.replace("class", "status")
            cols_to_delete = [class_to_delete, time_to_delete, status_to_delete]
            for c in cols_to_delete:
                try:
                    self.df_tracks = self.df_tracks.drop([c], axis=1)
                except Exception as e:
                    logger.error(e)
            item_idx = self.class_choice_cb.findText(class_to_delete)
            self.class_choice_cb.removeItem(item_idx)

    def normalize_features(self):

        x = self.df_tracks[self.columns_to_rescale].values

        if not self.normalized_signals:
            x = self.MinMaxScaler.transform(x)
            self.df_tracks[self.columns_to_rescale] = x
            self.plot_signals()
            self.normalized_signals = True
            self.normalize_features_btn.setIcon(
                icon(MDI6.arrow_collapse_vertical, color="#1565c0")
            )
            self.normalize_features_btn.setIconSize(QSize(25, 25))
        else:
            x = self.MinMaxScaler.inverse_transform(x)
            self.df_tracks[self.columns_to_rescale] = x
            self.plot_signals()
            self.normalized_signals = False
            self.normalize_features_btn.setIcon(
                icon(MDI6.arrow_collapse_vertical, color="black")
            )
            self.normalize_features_btn.setIconSize(QSize(25, 25))

    def switch_to_log(self):
        """
        Better would be to create a log(quantity) and plot it...
        """

        try:
            if self.cell_ax.get_yscale() == "linear":
                ymin, ymax = self.cell_ax.get_ylim()
                self.cell_ax.set_yscale("log")
                self.log_btn.setIcon(icon(MDI6.math_log, color="#1565c0"))
                self.cell_ax.set_ylim(self.value_magnitude, ymax)
                self.log_scale = True
            else:
                self.cell_ax.set_yscale("linear")
                self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
                self.log_scale = False
        except Exception as e:
            logger.error(e)

        # self.cell_ax.autoscale()
        self.cell_fcanvas.canvas.draw_idle()

    def start(self):
        """
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        """
        self.start_btn.setShortcut(QKeySequence(""))

        self.last_frame_btn.setEnabled(True)
        self.last_frame_btn.clicked.connect(self.set_last_frame)

        self.first_frame_btn.setEnabled(True)
        self.first_frame_btn.clicked.connect(self.set_first_frame)

        self.start_btn.hide()
        self.stop_btn.show()

        self.anim.event_source.start()
        self.stop_btn.clicked.connect(self.stop)

    def contrast_slider_action(self):
        """
        Recontrast the imshow as the contrast slider is moved.
        """

        self.vmin = self.contrast_slider.value()[0]
        self.vmax = self.contrast_slider.value()[1]
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.fcanvas.canvas.draw_idle()

    def show_outliers(self):
        if self.outliers_check.isChecked():
            self.show_fliers = True
            self.plot_signals()
        else:
            self.show_fliers = False
            self.plot_signals()

    def export_signals(self):

        auto_dataset_name = (
            self.pos.split(os.sep)[-4] + "_" + self.pos.split(os.sep)[-2] + ".npy"
        )

        if self.normalized_signals:
            self.normalize_features_btn.click()

        training_set = []
        cols = self.df_tracks.columns
        tracks = np.unique(self.df_tracks["TRACK_ID"].to_numpy())

        for track in tracks:
            # Add all signals at given track
            signals = {}
            for c in cols:
                signals.update(
                    {
                        c: self.df_tracks.loc[
                            self.df_tracks["TRACK_ID"] == track, c
                        ].to_numpy()
                    }
                )
            time_of_interest = self.df_tracks.loc[
                self.df_tracks["TRACK_ID"] == track, self.time_name
            ].to_numpy()[0]
            cclass = self.df_tracks.loc[
                self.df_tracks["TRACK_ID"] == track, self.class_name
            ].to_numpy()[0]
            signals.update({"time_of_interest": time_of_interest, "class": cclass})
            # Here auto add all available channels
            training_set.append(signals)

        pathsave = QFileDialog.getSaveFileName(
            self, "Select file name", self.exp_dir + auto_dataset_name, ".npy"
        )[0]
        if pathsave != "":
            if not pathsave.endswith(".npy"):
                pathsave += ".npy"
            try:
                np.save(pathsave, training_set)
                logger.info(f"File successfully written in {pathsave}.")
            except Exception as e:
                logger.error(f"Error {e}...")

    def create_new_event_class(self):

        # display qwidget to name the event
        self.newClassWidget = CelldetectiveWidget()
        self.newClassWidget.setWindowTitle("Create new event class")

        layout = QVBoxLayout()
        self.newClassWidget.setLayout(layout)
        name_hbox = QHBoxLayout()
        name_hbox.addWidget(QLabel("event name: "), 25)
        self.class_name_le = QLineEdit("event")
        name_hbox.addWidget(self.class_name_le, 75)
        layout.addLayout(name_hbox)

        class_labels = ["event", "no event", "else"]
        layout.addWidget(QLabel("prefill: "))
        radio_box = QHBoxLayout()
        self.class_option_rb = [QRadioButton() for i in range(3)]
        for i, c in enumerate(self.class_option_rb):
            if i == 0:
                c.setChecked(True)
            c.setText(class_labels[i])
            radio_box.addWidget(c, 33, alignment=Qt.AlignCenter)
        layout.addLayout(radio_box)

        btn_hbox = QHBoxLayout()
        submit_btn = QPushButton("submit")
        cancel_btn = QPushButton("cancel")
        btn_hbox.addWidget(cancel_btn, 50)
        btn_hbox.addWidget(submit_btn, 50)
        layout.addLayout(btn_hbox)

        submit_btn.clicked.connect(self.write_new_event_class)
        cancel_btn.clicked.connect(self.close_without_new_class)

        self.newClassWidget.show()
        center_window(self.newClassWidget)

    def shortcut_suppr(self):
        self.correct_btn.click()
        self.suppr_btn.click()
        self.correct_btn.click()

    def closeEvent(self, event):
        # result = QMessageBox.question(self,
        # 			  "Confirm Exit...",
        # 			  "Are you sure you want to exit ?",
        # 			  QMessageBox.Yes| QMessageBox.No,
        # 			  )
        # del self.img
        gc.collect()

    def save_trajectories(self):
        # specific to signal/static annotator
        logger.info(
            "Save trajectory function not implemented for BaseAnnotator class..."
        )

    def cancel_selection(self):
        self.hide_annotation_buttons()
        self.correct_btn.setEnabled(False)
        self.correct_btn.setText("correct")
        self.cancel_btn.setEnabled(False)

        try:
            self.selection.pop(0)
        except Exception as e:
            pass

        try:
            for k, (t, idx) in enumerate(zip(self.loc_t, self.loc_idx)):
                self.colors[t][idx, 0] = self.previous_color[k][0]
                # self.colors[t][idx, 1] = self.previous_color[k][1]
        except Exception as e:
            pass
