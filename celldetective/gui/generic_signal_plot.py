from PyQt5.QtWidgets import (
    QMessageBox,
    QGridLayout,
    QButtonGroup,
    QCheckBox,
    QLineEdit,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QDoubleValidator

from celldetective.gui.gui_utils import ExportPlotBtn, QuickSliderLayout
from celldetective.gui.base.figure_canvas import FigureCanvas
from celldetective.gui.base.utils import center_window
from celldetective.gui.tableUI import TableUI
from celldetective.utils.experiment import collect_experiment_metadata

from superqt.fonticon import icon
from superqt import QLabeledSlider, QLabeledDoubleSlider
from fonticon_mdi6 import MDI6
import numpy as np
import json
import os
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
from glob import glob
from matplotlib.cm import tab10
from celldetective.gui.base.components import CelldetectiveWidget
import matplotlib.cm as mcm
import pandas as pd

from lifelines.utils import qth_survival_times


class GenericSignalPlotWidget(CelldetectiveWidget):

    def __init__(
        self,
        df=None,
        df_pos_info=None,
        df_well_info=None,
        feature_selected=None,
        parent_window=None,
        title="plot",
        *args,
        **kwargs,
    ):

        super().__init__()
        center_window(self)

        self.parent_window = parent_window
        self.setWindowTitle(title)

        self.show_ci = False
        self.legend_visible = True
        self.show_cell_lines = False
        self.alpha_setting = 0.5
        self.scaling_factor = 1
        self.target_class = [0]
        self.float_validator = QDoubleValidator()

        cmap_lbl = self.parent_window.cbs[-1].currentText()
        self.cmap = getattr(mcm, cmap_lbl)

        self.feature_selected = feature_selected
        self.df = df
        self.df_pos_info = df_pos_info
        self.df_well_info = df_well_info

        self.layout = QVBoxLayout()
        self.layout.setSpacing(3)
        self.populate_widget()

        self.ci_btn.click()
        # self.legend_btn.click()

        self.fig.tight_layout()
        self.setLayout(self.layout)

    def populate_widget(self):

        self.plot_options = [QRadioButton() for i in range(3)]
        self.radio_labels = ["well", "position", "both"]
        radio_hbox = QHBoxLayout()
        radio_hbox.setContentsMargins(15, 15, 15, 0)

        self.group_lbl = QLabel("grouping: ")
        radio_hbox.addWidget(self.group_lbl, 25)
        radio_subhbox = QHBoxLayout()
        radio_hbox.addLayout(radio_subhbox, 75)

        self.plot_btn_group = QButtonGroup()
        for i in range(3):
            self.plot_options[i].setText(self.radio_labels[i])
            self.plot_btn_group.addButton(self.plot_options[i])
            radio_subhbox.addWidget(self.plot_options[i], 33, alignment=Qt.AlignCenter)

        if self.parent_window.position_indices is not None:
            # at least a position is selected
            if (
                len(self.parent_window.well_indices) > 1
                and len(self.parent_window.position_indices) == 1
            ):
                # several wells but one position
                self.plot_btn_group.buttons()[0].click()
                # for i in [1,2]:
                # 	self.plot_options[i].setEnabled(False)
            elif len(self.parent_window.well_indices) > 1:
                self.plot_btn_group.buttons()[0].click()
            elif (
                len(self.parent_window.well_indices) == 1
                and len(self.parent_window.position_indices) == 1
            ):
                self.plot_btn_group.buttons()[1].click()
                for i in [0, 2]:
                    self.plot_options[i].setEnabled(False)
            elif (
                len(self.parent_window.well_indices) == 1
                and len(self.parent_window.position_indices) > 1
            ):
                # one well, several positions
                self.plot_btn_group.buttons()[2].click()
        else:
            if len(self.parent_window.well_indices) > 1:
                self.plot_btn_group.buttons()[0].click()
            elif len(self.parent_window.well_indices) == 1:
                self.plot_btn_group.buttons()[2].click()

        self.layout.addLayout(radio_hbox)

        plot_buttons_hbox = QHBoxLayout()
        plot_buttons_hbox.setContentsMargins(10, 10, 5, 0)
        plot_buttons_hbox.addWidget(QLabel(""), 80, alignment=Qt.AlignLeft)

        self.legend_btn = QPushButton("")
        self.legend_btn.setIcon(icon(MDI6.text_box, color=self.help_color))
        self.legend_btn.setStyleSheet(self.button_select_all)
        self.legend_btn.setToolTip("Show or hide the legend")
        self.legend_btn.setIconSize(QSize(20, 20))
        plot_buttons_hbox.addWidget(self.legend_btn, 5, alignment=Qt.AlignRight)

        self.log_btn = QPushButton("")
        self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
        self.log_btn.setStyleSheet(self.button_select_all)
        self.log_btn.setIconSize(QSize(20, 20))
        self.log_btn.setToolTip("Enable or disable log scale")
        plot_buttons_hbox.addWidget(self.log_btn, 5, alignment=Qt.AlignRight)

        self.ci_btn = QPushButton("")
        self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal, color="black"))
        self.ci_btn.setStyleSheet(self.button_select_all)
        self.ci_btn.setIconSize(QSize(20, 20))
        self.ci_btn.setToolTip("Show or hide confidence intervals.")
        plot_buttons_hbox.addWidget(self.ci_btn, 5, alignment=Qt.AlignRight)

        self.cell_lines_btn = QPushButton("")
        self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color="black"))
        self.cell_lines_btn.setStyleSheet(self.button_select_all)
        self.cell_lines_btn.setToolTip("Show or hide individual cell signals.")
        self.cell_lines_btn.setIconSize(QSize(20, 20))
        plot_buttons_hbox.addWidget(self.cell_lines_btn, 5, alignment=Qt.AlignRight)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(4, 3))
        self.plot_widget = FigureCanvas(self.fig, title="")
        self.plot_widget.setContentsMargins(0, 0, 0, 0)
        self.initialize_axis()
        plt.tight_layout()

        self.export_btn = ExportPlotBtn(self.fig, export_dir=self.parent_window.exp_dir)
        plot_buttons_hbox.addWidget(self.export_btn, 5, alignment=Qt.AlignRight)
        self.layout.addLayout(plot_buttons_hbox)

        self.export_tabular_btn = QPushButton("")
        self.export_tabular_btn.setIcon(icon(MDI6.table, color="black"))
        self.export_tabular_btn.setStyleSheet(self.button_select_all)
        self.export_tabular_btn.setToolTip("Tabulate survival values.")
        self.export_tabular_btn.setIconSize(QSize(20, 20))
        plot_buttons_hbox.addWidget(self.export_tabular_btn, 5, alignment=Qt.AlignRight)
        self.export_tabular_btn.hide()

        self.ax.set_prop_cycle(
            "color",
            [
                self.cmap(i)
                for i in np.linspace(0, 1, len(self.parent_window.well_indices))
            ],
        )

        self.fig.set_facecolor("none")  # or 'None'
        self.fig.canvas.setStyleSheet("background-color: transparent;")
        self.plot_widget.canvas.draw()

        self.layout.addWidget(self.plot_widget)

        self.plot_btn_group.buttonClicked[int].connect(self.plot_signals)
        self.legend_btn.clicked.connect(self.show_hide_legend)
        self.log_btn.clicked.connect(self.switch_to_log)
        self.ci_btn.clicked.connect(self.switch_ci)
        self.cell_lines_btn.clicked.connect(self.switch_cell_lines)

        self.class_selection_widget = CelldetectiveWidget()

        self.class_selection_lbl = QLabel("class of interest:")
        class_hbox = QHBoxLayout()
        self.class_selection_widget.setLayout(class_hbox)
        class_hbox.addWidget(self.class_selection_lbl, 25, alignment=Qt.AlignLeft)

        class_subhbox = QHBoxLayout()
        class_hbox.addLayout(class_subhbox, 75)

        self.all_btn = QRadioButton("*")
        self.event_btn = QRadioButton("event")
        self.event_btn.setChecked(True)
        self.no_event_btn = QRadioButton("no event")
        self.class_btn_group = QButtonGroup()
        for btn in [self.all_btn, self.event_btn, self.no_event_btn]:
            self.class_btn_group.addButton(btn)

        self.class_btn_group.buttonClicked[int].connect(self.set_class_to_plot)

        class_subhbox.addWidget(self.all_btn, 33, alignment=Qt.AlignCenter)
        class_subhbox.addWidget(self.event_btn, 33, alignment=Qt.AlignCenter)
        class_subhbox.addWidget(self.no_event_btn, 33, alignment=Qt.AlignCenter)

        self.layout.addWidget(self.class_selection_widget)  # Layout(class_hbox)

        # Rescale
        self.rescale_widget = CelldetectiveWidget()

        scale_hbox = QHBoxLayout()
        self.rescale_widget.setLayout(scale_hbox)

        scale_hbox.addWidget(QLabel("scaling factor: "), 25)
        self.scaling_factor_le = QLineEdit("1")
        self.scaling_factor_le.setValidator(self.float_validator)
        scale_hbox.addWidget(self.scaling_factor_le, 65)

        self.rescale_btn = QPushButton("rescale")
        self.rescale_btn.setStyleSheet(self.button_style_sheet_2)
        self.rescale_btn.clicked.connect(self.rescale_y_axis)
        scale_hbox.addWidget(self.rescale_btn, 10)
        # self.layout.addLayout(scale_hbox)
        self.layout.addWidget(self.rescale_widget)

        # Rescale
        self.cell_lines_alpha_wdg = CelldetectiveWidget()
        alpha_hbox = QHBoxLayout()

        self.alpha_slider = QLabeledDoubleSlider()
        alpha_hbox = QuickSliderLayout(
            label="single-cell\nsignal alpha: ",
            slider=self.alpha_slider,
            slider_initial_value=self.alpha_setting,
            slider_range=(0, 1),
            decimal_option=True,
            precision=5,
        )
        self.alpha_slider.valueChanged.connect(self.submit_alpha)
        self.cell_lines_alpha_wdg.setLayout(alpha_hbox)

        # self.submit_alpha_btn = QPushButton('submit')
        # self.submit_alpha_btn.setStyleSheet(self.button_style_sheet_2)
        # self.submit_alpha_btn.clicked.connect(self.submit_alpha)
        # alpha_hbox.addWidget(self.submit_alpha_btn, 10)
        self.layout.addWidget(self.cell_lines_alpha_wdg)

        self.select_option = [QRadioButton() for i in range(2)]
        self.select_label = ["by name", "spatially"]

        select_hbox = QHBoxLayout()
        select_hbox.addWidget(QLabel("select position: "), 25)

        select_subhbox = QHBoxLayout()
        select_hbox.addLayout(select_subhbox, 75)

        self.select_btn_group = QButtonGroup()
        for i in range(2):
            self.select_option[i].setText(self.select_label[i])
            self.select_btn_group.addButton(self.select_option[i])
            select_subhbox.addWidget(
                self.select_option[i], 33, alignment=Qt.AlignCenter
            )
        self.select_option[0].setChecked(True)
        self.select_btn_group.buttonClicked[int].connect(self.switch_selection_mode)
        self.layout.addLayout(select_hbox)

        self.look_for_metadata()
        if self.metadata_found:
            self.fig_scatter, self.ax_scatter = plt.subplots(
                1, 1, figsize=(4, 2)
            )  # ,figsize=(4,3)
            self.position_scatter = FigureCanvas(self.fig_scatter)
            self.load_coordinates()
            self.plot_spatial_location()
            self.ax_scatter.spines["top"].set_visible(False)
            self.ax_scatter.spines["right"].set_visible(False)
            self.ax_scatter.set_aspect("equal")
            self.ax_scatter.set_xticks([])
            self.ax_scatter.set_yticks([])
            plt.tight_layout()

            self.fig_scatter.set_facecolor("none")  # or 'None'
            self.fig_scatter.canvas.setStyleSheet("background-color: transparent;")
            self.layout.addWidget(self.position_scatter)

        self.generate_pos_selection_widget()
        self.select_btn_group.buttons()[0].click()

    def submit_alpha(self, value):

        alpha = value
        try:
            alpha = float(alpha)
        except:
            return None
        if alpha > 1.0:
            alpha = 1.0
        elif alpha < 0.0:
            alpha = 0.0
        self.alpha_setting = alpha
        self.plot_signals(0)

    def rescale_y_axis(self):
        new_scale = self.scaling_factor_le.text().replace(",", ".")
        if new_scale == "":
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please set a valid scaling factor...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None
        else:
            self.scaling_factor = float(new_scale)
            self.plot_signals(0)

    def switch_selection_mode(self, id):

        for i in range(2):
            if self.select_option[i].isChecked():
                self.selection_mode = self.select_label[i]
        if self.selection_mode == "by name":
            if len(self.metafiles) > 0:
                self.position_scatter.hide()
            self.line_choice_widget.show()
        else:
            if len(self.metafiles) > 0:
                self.position_scatter.show()
            self.line_choice_widget.hide()

    def set_class_to_plot(self):

        if self.all_btn.isChecked():
            self.target_class = [0, 1]
        elif self.event_btn.isChecked():
            self.target_class = [0]
        else:
            self.target_class = [1]

        self.plot_signals(0)

    def generate_pos_selection_widget(self):

        self.well_names = self.df["well_name"].unique()
        self.pos_names = self.df_pos_info[
            "pos_name"
        ].unique()  # pd.DataFrame(self.ks_estimators_per_position)['position_name'].unique()

        self.usable_well_labels = []
        for name in self.well_names:
            for lbl in self.parent_window.well_labels:
                if name + ":" in lbl:
                    self.usable_well_labels.append(lbl)

        thresh = 20
        self.well_name_truncated = [
            w[: thresh - 3] + "..." if len(w) > thresh else w
            for w in self.usable_well_labels
        ]

        self.line_choice_widget = CelldetectiveWidget()
        self.line_check_vbox = QGridLayout()
        self.line_choice_widget.setLayout(self.line_check_vbox)

        if len(self.parent_window.well_indices) > 1:
            self.well_display_options = [
                QCheckBox(self.well_name_truncated[i])
                for i in range(len(self.well_name_truncated))
            ]
            for i in range(len(self.well_names)):
                self.line_check_vbox.addWidget(
                    self.well_display_options[i],
                    i % 4,
                    i // 4,
                    1,
                    1,
                    alignment=Qt.AlignCenter,
                )
                self.well_display_options[i].setChecked(True)
                self.well_display_options[i].setStyleSheet("font-size: 12px;")
                self.well_display_options[i].toggled.connect(self.select_lines)
                self.well_display_options[i].setToolTip(self.usable_well_labels[i])
        else:
            self.pos_display_options = [
                QCheckBox(self.pos_names[i]) for i in range(len(self.pos_names))
            ]
            for i in range(len(self.pos_names)):
                self.line_check_vbox.addWidget(
                    self.pos_display_options[i],
                    i % 4,
                    i // 4,
                    1,
                    1,
                    alignment=Qt.AlignCenter,
                )
                self.pos_display_options[i].setChecked(True)
                self.pos_display_options[i].setStyleSheet("font-size: 12px;")
                self.pos_display_options[i].toggled.connect(self.select_lines)

        self.layout.addWidget(self.line_choice_widget)
        # self.layout.addLayout(self.line_check_vbox)

    def look_for_metadata(self):
        self.metadata_found = False
        self.metafiles = (
            glob(
                self.parent_window.exp_dir
                + os.sep.join([f"W*", "*", "movie", "*metadata.txt"])
            )
            + glob(
                self.parent_window.exp_dir + os.sep.join([f"W*", "*", "*metadata.txt"])
            )
            + glob(self.parent_window.exp_dir + os.sep.join([f"W*", "*metadata.txt"]))
            + glob(self.parent_window.exp_dir + "*metadata.txt")
        )
        print(f"Found {len(self.metafiles)} metadata files...")
        if len(self.metafiles) > 0:
            self.metadata_found = True

    def load_coordinates(self):
        """
        Read metadata and try to extract position coordinates
        """

        self.no_meta = False
        try:
            with open(self.metafiles[0], "r") as f:
                data = json.load(f)
                positions = data["Summary"]["InitialPositionList"]
        except Exception as e:
            print(f"Trouble loading metadata: error {e}...")
            return None

        for k in range(len(positions)):
            pos_label = positions[k]["Label"]
            try:
                coords = positions[k]["DeviceCoordinatesUm"]["XYStage"]
            except:
                try:
                    coords = positions[k]["DeviceCoordinatesUm"]["PIXYStage"]
                except:
                    self.no_meta = True

            if not self.no_meta:
                files = self.df_pos_info["stack_path"].values
                pos_loc = [pos_label in f for f in files]
                self.df_pos_info.loc[pos_loc, "x"] = coords[0]
                self.df_pos_info.loc[pos_loc, "y"] = coords[1]
                self.df_pos_info.loc[pos_loc, "metadata_tag"] = pos_label

    def plot_spatial_location(self):

        try:
            self.sc = self.ax_scatter.scatter(
                self.df_pos_info["x"].values,
                self.df_pos_info["y"].values,
                picker=True,
                pickradius=1,
                color=self.select_color(self.df_pos_info["select"].values),
            )
            self.scat_labels = self.df_pos_info["metadata_tag"].values
            self.ax_scatter.invert_xaxis()

            self.annot = self.ax_scatter.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )
            self.annot.set_visible(False)

            xmin, xmax = self.ax_scatter.get_xlim()
            ymin, ymax = self.ax_scatter.get_ylim()
            xdatarange = xmax - xmin
            ydatarange = ymax - ymin

            self.ax_scatter.set_xlim(xmin - 0.1 * xdatarange, xmax + 0.1 * xdatarange)
            self.ax_scatter.set_ylim(ymin - 0.1 * ydatarange, ymax + 0.1 * ydatarange)

            # xmin,xmax = self.ax_scatter.get_xlim()
            # ymin,ymax = self.ax_scatter.get_ylim()
            # desired_a = 4
            # new_x_max = xmin + desired_a * (ymax - ymin)
            # self.ax_scatter.set_xlim(xmin - (new_x_max - xmin)/2.0, new_x_max - (new_x_max - xmin)/2.0)

            self.fig_scatter.tight_layout()
            self.fig_scatter.canvas.mpl_connect("motion_notify_event", self.hover)
            self.fig_scatter.canvas.mpl_connect("pick_event", self.unselect_position)
        except Exception as e:
            pass

    def update_annot(self, ind):

        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = self.scat_labels[ind["ind"][0]]
        self.annot.set_text(text)
        self.annot.get_bbox_patch().set_facecolor("k")
        self.annot.get_bbox_patch().set_alpha(0.4)

    def hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax_scatter:
            cont, ind = self.sc.contains(event)
            if cont:
                self.update_annot(ind)
                self.annot.set_visible(True)
                self.fig_scatter.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig_scatter.canvas.draw_idle()

    def unselect_position(self, event):

        ind = event.ind  # index of selected position
        well_idx = self.df_pos_info.iloc[ind]["well_index"].values[0]
        selectedPos = self.df_pos_info.iloc[ind]["pos_path"].values[0]
        currentSelState = self.df_pos_info.iloc[ind]["select"].values[0]
        if self.plot_options[0].isChecked() or self.plot_options[2].isChecked():
            self.df_pos_info.loc[
                self.df_pos_info["well_index"] == well_idx, "select"
            ] = not currentSelState
            self.df_well_info.loc[
                self.df_well_info["well_index"] == well_idx, "select"
            ] = not currentSelState
            if len(self.parent_window.well_indices) > 1:
                self.well_display_options[well_idx].setChecked(not currentSelState)
            else:
                for p in self.pos_display_options:
                    p.setChecked(not currentSelState)
        else:
            self.df_pos_info.loc[
                self.df_pos_info["pos_path"] == selectedPos, "select"
            ] = not currentSelState
            if len(self.parent_window.well_indices) <= 1:
                self.pos_display_options[ind[0]].setChecked(not currentSelState)

        self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
        self.position_scatter.canvas.draw_idle()
        self.plot_signals(0)

    def select_color(self, selection):
        colors = [tab10(0) if s else tab10(0.1) for s in selection]
        return colors

    def initialize_axis(self):

        previous_ymin, previous_ymax = self.ax.get_ylim()
        previous_legend = self.legend_visible
        is_log = self.ax.get_yscale()

        self.ax.clear()
        self.ax.plot([], [])

        # Labels
        self.ax.set_xlabel("time [min]")
        self.ax.set_ylabel(self.feature_selected)

        # Spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.grid(
            which="major", color="black", linestyle="-", linewidth=0.8, alpha=0.2
        )
        self.ax.grid(
            which="minor", color="lightgray", linestyle="--", linewidth=0.5, alpha=0.1
        )
        # Lims
        safe_df = self.df.dropna(subset=self.feature_selected)
        values = safe_df[self.feature_selected].values
        if len(values) > 0:
            self.ax.set_ylim(
                np.percentile(values, 1) * self.scaling_factor,
                np.percentile(values, 99) * self.scaling_factor,
            )
        self.ax.set_xlim(
            -(self.df["FRAME"].max() + 2) * self.parent_window.FrameToMin,
            (self.df["FRAME"].max() + 2) * self.parent_window.FrameToMin,
        )

        if is_log == "log":
            self.ax.set_yscale("log")
        if previous_legend:
            leg = self.ax.get_legend()
            if leg is not None:
                leg.set_visible(True)

    def show_hide_legend(self):

        if self.legend_visible:
            leg = self.ax.get_legend()
            leg.set_visible(False)
            self.legend_visible = False
            self.legend_btn.setIcon(icon(MDI6.text_box, color="black"))
        else:
            leg = self.ax.get_legend()
            leg.set_visible(True)
            self.legend_visible = True
            self.legend_btn.setIcon(icon(MDI6.text_box, color=self.help_color))

        self.plot_widget.canvas.draw_idle()

    def switch_to_log(self):
        """
        Switch threshold histogram to log scale. Auto adjust.
        """

        if self.ax.get_yscale() == "linear":
            self.ax.set_yscale("log")
            self.log_btn.setIcon(icon(MDI6.math_log, color=self.help_color))
            # self.ax.set_ylim(0.01,1.05)
        else:
            self.ax.set_yscale("linear")
            self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
            # self.ax.set_ylim(0.01,1.05)

        # self.ax.autoscale()
        self.plot_widget.canvas.draw_idle()

    def plot_signals(self, id):

        for i in range(3):
            if self.plot_options[i].isChecked():
                self.plot_mode = self.radio_labels[i]

        if self.target_class == [0, 1]:
            mean_signal_type = "mean_all"
            std_signal = "std_all"
            matrix = "matrix_all"
        elif self.target_class == [0]:
            mean_signal_type = "mean_event"
            std_signal = "std_event"
            matrix = "matrix_event"
        else:
            mean_signal_type = "mean_no_event"
            std_signal = "std_no_event"
            matrix = "matrix_no_event"

        colors = np.array(
            [self.cmap(i / len(self.df_pos_info)) for i in range(len(self.df_pos_info))]
        )
        well_color = [
            self.cmap(i / len(self.df_well_info)) for i in range(len(self.df_well_info))
        ]

        if self.plot_mode == "position":
            self.initialize_axis()
            lines = self.df_pos_info.loc[self.df_pos_info["select"], "signal"].values
            pos_labels = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_name"
            ].values
            pos_indices = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_index"
            ].values
            well_index = self.df_pos_info.loc[
                self.df_pos_info["select"], "well_index"
            ].values
            for i in range(len(lines)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines[i],
                        colors[pos_indices[i]],
                        pos_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=self.show_ci,
                        cell_lines_option=self.show_cell_lines,
                        matrix=matrix,
                    )
                else:
                    self.plot_line(
                        lines[i],
                        well_color[well_index[i]],
                        pos_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=self.show_ci,
                        cell_lines_option=self.show_cell_lines,
                        matrix=matrix,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=3, fontsize="x-small")

        elif self.plot_mode == "well":
            self.initialize_axis()
            lines = self.df_well_info.loc[self.df_well_info["select"], "signal"].values
            well_index = self.df_well_info.loc[
                self.df_well_info["select"], "well_index"
            ].values
            well_labels = self.df_well_info.loc[
                self.df_well_info["select"], "well_name"
            ].values
            for i in range(len(lines)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines[i],
                        "k",
                        well_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=self.show_ci,
                        cell_lines_option=self.show_cell_lines,
                        matrix=matrix,
                    )
                else:
                    self.plot_line(
                        lines[i],
                        well_color[well_index[i]],
                        well_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=self.show_ci,
                        cell_lines_option=self.show_cell_lines,
                        matrix=matrix,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=2, fontsize="x-small")

        elif self.plot_mode == "both":

            self.initialize_axis()
            lines_pos = self.df_pos_info.loc[
                self.df_pos_info["select"], "signal"
            ].values
            lines_well = self.df_well_info.loc[
                self.df_well_info["select"], "signal"
            ].values

            pos_indices = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_index"
            ].values
            well_index_pos = self.df_pos_info.loc[
                self.df_pos_info["select"], "well_index"
            ].values
            well_index = self.df_well_info.loc[
                self.df_well_info["select"], "well_index"
            ].values
            well_labels = self.df_well_info.loc[
                self.df_well_info["select"], "well_name"
            ].values
            pos_labels = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_name"
            ].values

            for i in range(len(lines_pos)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines_pos[i],
                        colors[pos_indices[i]],
                        pos_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=self.show_ci,
                        cell_lines_option=self.show_cell_lines,
                        matrix=matrix,
                    )
                else:
                    self.plot_line(
                        lines_pos[i],
                        well_color[well_index_pos[i]],
                        None,
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=False,
                    )

            for i in range(len(lines_well)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines_well[i],
                        "k",
                        "pool",
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=False,
                    )
                else:
                    self.plot_line(
                        lines_well[i],
                        well_color[well_index[i]],
                        well_labels[i],
                        mean_signal_type,
                        std_signal=std_signal,
                        ci_option=False,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=3, fontsize="x-small")

        self.plot_widget.canvas.draw()

    def plot_line(
        self,
        line,
        color,
        label,
        mean_signal_type,
        ci_option=True,
        cell_lines_option=False,
        alpha_ci=0.5,
        std_signal=None,
        matrix=None,
    ):

        # Plot a signal
        if line == line:
            self.ax.plot(
                line["timeline"] * self.parent_window.FrameToMin,
                line[mean_signal_type] * self.scaling_factor,
                color=color,
                label=label,
            )

            if ci_option and std_signal is not None:

                self.ax.fill_between(
                    line["timeline"] * self.parent_window.FrameToMin,
                    [
                        a - b
                        for a, b in zip(
                            line[mean_signal_type] * self.scaling_factor,
                            line[std_signal] * self.scaling_factor,
                        )
                    ],
                    [
                        a + b
                        for a, b in zip(
                            line[mean_signal_type] * self.scaling_factor,
                            line[std_signal] * self.scaling_factor,
                        )
                    ],
                    color=color,
                    alpha=alpha_ci,
                )
            if cell_lines_option and matrix is not None:
                # Show individual cell signals
                mat = line[matrix]
                for i in range(mat.shape[0]):
                    self.ax.plot(
                        line["timeline"] * self.parent_window.FrameToMin,
                        mat[i, :] * self.scaling_factor,
                        color=color,
                        alpha=self.alpha_setting,
                    )

    def switch_ci(self):

        # Show the confidence interval / STD

        if self.show_ci:
            self.ci_btn.setIcon(icon(MDI6.arrow_expand_horizontal, color="black"))
        else:
            self.ci_btn.setIcon(
                icon(MDI6.arrow_expand_horizontal, color=self.help_color)
            )
        self.show_ci = not self.show_ci
        try:
            self.plot_signals(0)
        except Exception as e:
            print(f"{e=}")

    def switch_cell_lines(self):

        # Show individual cell signals

        if self.show_cell_lines:
            self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color="black"))
        else:
            self.cell_lines_btn.setIcon(icon(MDI6.view_headline, color=self.help_color))
        self.show_cell_lines = not self.show_cell_lines
        self.plot_signals(0)

    def select_lines(self):

        if len(self.parent_window.well_indices) > 1:
            for i in range(len(self.well_display_options)):
                self.df_well_info.loc[
                    self.df_well_info["well_index"] == i, "select"
                ] = self.well_display_options[i].isChecked()
                self.df_pos_info.loc[self.df_pos_info["well_index"] == i, "select"] = (
                    self.well_display_options[i].isChecked()
                )
        else:
            for i in range(len(self.pos_display_options)):
                self.df_pos_info.loc[self.df_pos_info["pos_index"] == i, "select"] = (
                    self.pos_display_options[i].isChecked()
                )

        if len(self.metafiles) > 0:
            self.sc.set_color(self.select_color(self.df_pos_info["select"].values))
            self.position_scatter.canvas.draw_idle()
        self.plot_signals(0)


class SurvivalPlotWidget(GenericSignalPlotWidget):

    def __init__(self, *args, **kwargs):

        super(SurvivalPlotWidget, self).__init__(*args, **kwargs)
        self.cell_lines_btn.hide()
        self.class_selection_widget.hide()
        self.rescale_widget.hide()
        self.cell_lines_alpha_wdg.hide()
        self.export_tabular_btn.show()
        self.export_tabular_btn.clicked.connect(self.set_table_options)

    def switch_to_log(self):
        """
        Switch threshold histogram to log scale. Auto adjust.
        """

        if self.ax.get_yscale() == "linear":
            ymin, _ = self.ax.get_ylim()
            self.ax.set_ylim(max(ymin, 0.01), 1.05)
            self.ax.set_yscale("log")
            self.log_btn.setIcon(icon(MDI6.math_log, color=self.help_color))
        else:
            self.ax.set_yscale("linear")
            self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
            # self.ax.set_ylim(0.01,1.05)

        # self.ax.autoscale()
        self.plot_widget.canvas.draw_idle()

    def initialize_axis(self):

        previous_legend = self.legend_visible
        is_log = self.ax.get_yscale()

        self.ax.clear()
        self.ax.plot([], [])

        # Labels
        self.ax.set_xlabel("time [min]")
        self.ax.set_ylabel("survival")

        # Spines
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Lims
        self.ax.set_xlim(
            0 * self.parent_window.FrameToMin,
            (self.df["FRAME"].max() + 2) * self.parent_window.FrameToMin,
        )
        if is_log == "log":
            ymin = 0.1
        else:
            ymin = 0.0
        self.ax.set_ylim(ymin, 1.05)

        if is_log == "log":
            self.ax.set_yscale("log")
        if previous_legend:
            leg = self.ax.get_legend()
            if leg is not None:
                leg.set_visible(True)

    def plot_signals(self, id):

        for i in range(3):
            if self.plot_options[i].isChecked():
                self.plot_mode = self.radio_labels[i]

        colors = np.array(
            [self.cmap(i / len(self.df_pos_info)) for i in range(len(self.df_pos_info))]
        )
        well_color = [
            self.cmap(i / len(self.df_well_info)) for i in range(len(self.df_well_info))
        ]

        if self.plot_mode == "position":
            self.initialize_axis()
            lines = self.df_pos_info.loc[
                self.df_pos_info["select"], "survival_fit"
            ].values
            pos_labels = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_name"
            ].values
            pos_indices = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_index"
            ].values
            well_index = self.df_pos_info.loc[
                self.df_pos_info["select"], "well_index"
            ].values
            for i in range(len(lines)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines[i],
                        colors[pos_indices[i]],
                        pos_labels[i],
                        ci_option=self.show_ci,
                    )
                else:
                    self.plot_line(
                        lines[i],
                        well_color[well_index[i]],
                        pos_labels[i],
                        ci_option=self.show_ci,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=3, fontsize="x-small")

        elif self.plot_mode == "well":
            self.initialize_axis()
            lines = self.df_well_info.loc[
                self.df_well_info["select"], "survival_fit"
            ].values
            well_index = self.df_well_info.loc[
                self.df_well_info["select"], "well_index"
            ].values
            well_labels = self.df_well_info.loc[
                self.df_well_info["select"], "well_name"
            ].values
            for i in range(len(lines)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines[i],
                        "k",
                        well_labels[i],
                        ci_option=self.show_ci,
                        legend=True,
                    )
                else:
                    self.plot_line(
                        lines[i],
                        well_color[well_index[i]],
                        well_labels[i],
                        ci_option=self.show_ci,
                        legend=True,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=2, fontsize="x-small")

        elif self.plot_mode == "both":

            self.initialize_axis()
            lines_pos = self.df_pos_info.loc[
                self.df_pos_info["select"], "survival_fit"
            ].values
            lines_well = self.df_well_info.loc[
                self.df_well_info["select"], "survival_fit"
            ].values

            pos_indices = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_index"
            ].values
            well_index_pos = self.df_pos_info.loc[
                self.df_pos_info["select"], "well_index"
            ].values
            well_index = self.df_well_info.loc[
                self.df_well_info["select"], "well_index"
            ].values
            well_labels = self.df_well_info.loc[
                self.df_well_info["select"], "well_name"
            ].values
            pos_labels = self.df_pos_info.loc[
                self.df_pos_info["select"], "pos_name"
            ].values

            for i in range(len(lines_pos)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines_pos[i],
                        colors[pos_indices[i]],
                        pos_labels[i],
                        ci_option=self.show_ci,
                        legend=True,
                    )
                else:
                    self.plot_line(
                        lines_pos[i],
                        well_color[well_index_pos[i]],
                        None,
                        ci_option=False,
                    )

            for i in range(len(lines_well)):
                if len(self.parent_window.well_indices) <= 1:
                    self.plot_line(
                        lines_well[i], "k", "pool", ci_option=False, legend=True
                    )
                else:
                    self.plot_line(
                        lines_well[i],
                        well_color[well_index[i]],
                        well_labels[i],
                        ci_option=False,
                        legend=True,
                    )
            if self.legend_visible:
                self.ax.legend(ncols=3, fontsize="x-small")

        self.plot_widget.canvas.draw()

    def plot_line(self, line, color, label, ci_option=True, legend=None, alpha_ci=0.5):

        # Plot a signal
        if line == line:
            line.plot_survival_function(
                ci_show=ci_option,
                ax=self.ax,
                legend=legend,
                color=color,
                label=label,
                xlabel="timeline [min]",
            )

    def set_table_options(self):

        self.config_table_wg = CelldetectiveWidget()
        self.config_table_wg.setMinimumWidth(480)
        self.config_table_wg.setWindowTitle("Survival data")

        layout = QVBoxLayout()
        self.config_table_wg.setLayout(layout)

        self.all_values_rb = QRadioButton("tabulate all values")
        self.single_timepoint_rb = QRadioButton("survival at single timepoint [min]: ")
        self.ec_rb = QRadioButton(r"EC N% survival: ")
        self.all_values_rb.toggled.connect(self.activate_sliders)
        self.single_timepoint_rb.toggled.connect(self.activate_sliders)

        self.single_timepoint_slider = QLabeledSlider()
        self.single_timepoint_slider.setRange(
            0, int(self.df["FRAME"].max() * self.parent_window.FrameToMin)
        )
        self.single_timepoint_slider.setValue(
            int(self.df["FRAME"].max() * self.parent_window.FrameToMin)
        )

        self.ec_slider = QLabeledSlider()
        self.ec_slider.setRange(0, 100)
        self.ec_slider.setValue(50)

        self.ec_rb.toggled.connect(self.activate_sliders)
        self.all_values_rb.click()

        self.set_btn = QPushButton("Set")
        self.set_btn.setStyleSheet(self.button_style_sheet)
        self.set_btn.clicked.connect(self.assemble_survival_data)

        layout.addWidget(self.all_values_rb)

        single_tp_layout = QHBoxLayout()
        single_tp_layout.addWidget(self.single_timepoint_rb, 33)
        single_tp_layout.addWidget(self.single_timepoint_slider, 66)
        layout.addLayout(single_tp_layout)

        ec_layout = QHBoxLayout()
        ec_layout.addWidget(self.ec_rb, 33)
        ec_layout.addWidget(self.ec_slider, 66)
        layout.addLayout(ec_layout)

        layout.addWidget(self.set_btn)
        center_window(self.config_table_wg)
        self.config_table_wg.show()

    def activate_sliders(self):
        if self.all_values_rb.isChecked():
            self.single_timepoint_slider.setEnabled(False)
            self.ec_slider.setEnabled(False)
        elif self.single_timepoint_rb.isChecked():
            self.single_timepoint_slider.setEnabled(True)
            self.ec_slider.setEnabled(False)
        elif self.ec_rb.isChecked():
            self.ec_slider.setEnabled(True)
            self.single_timepoint_slider.setEnabled(False)

    def assemble_survival_data(self):

        if self.plot_options[0].isChecked():
            data = self.df_well_info
            groupby = ["well_path"]
        if self.plot_options[1].isChecked():
            data = self.df_pos_info
            groupby = ["pos_path"]
        if self.plot_options[2].isChecked():
            print(
                'Not implemented yet... Please select "well" or "position" as grouping...'
            )
            return None

        if self.all_values_rb.isChecked():

            survival_table = []
            tid = 0
            for name, group in data.groupby(groupby):
                print(name)
                if groupby[0] == "pos_path":
                    metadata = collect_experiment_metadata(pos_path=name[0])
                elif groupby[0] == "well_path":
                    metadata = collect_experiment_metadata(well_path=name[0])
                ks_estimator = group["survival_fit"].values[0]
                if ks_estimator != ks_estimator:
                    continue
                timeline = list(ks_estimator.survival_function_.index)
                survival = ks_estimator.survival_function_["KM_estimate"].values
                lower_error = ks_estimator.confidence_interval_[
                    "KM_estimate_lower_0.95"
                ].values
                upper_error = ks_estimator.confidence_interval_[
                    "KM_estimate_upper_0.95"
                ].values
                for k in range(len(timeline)):
                    dico = metadata.copy()
                    dico.update(
                        {
                            "TRACK_ID": tid,
                            "FRAME": int(timeline[k] / self.parent_window.FrameToMin),
                            "timeline": timeline[k],
                            "survival": survival[k],
                            "event_fraction": 1 - survival[k],
                            "KM_estimate_lower_0.95": lower_error[k],
                            "KM_estimate_upper_0.95": upper_error[k],
                        }
                    )
                    survival_table.append(dico)
                tid += 1

            survival_table = pd.DataFrame(survival_table)
            self.table = TableUI(
                survival_table, f"Survival data", plot_mode="plot_track_signals"
            )
            self.table.show()

        elif self.single_timepoint_rb.isChecked():

            survival_table = []
            tid = 0
            for name, group in data.groupby(groupby):
                print(name)
                if groupby[0] == "pos_path":
                    metadata = collect_experiment_metadata(pos_path=name[0])
                elif groupby[0] == "well_path":
                    metadata = collect_experiment_metadata(well_path=name[0])
                ks_estimator = group["survival_fit"].values[0]
                if ks_estimator != ks_estimator:
                    continue
                survival = ks_estimator.survival_function_at_times(
                    self.single_timepoint_slider.value()
                ).values[0]
                dico = metadata.copy()
                dico.update(
                    {
                        "timepoint": self.single_timepoint_slider.value(),
                        "survival": survival,
                        "event_fraction": 1 - survival,
                    }
                )
                survival_table.append(dico)
                tid += 1

            survival_table = pd.DataFrame(survival_table)
            self.table = TableUI(survival_table, f"Survival data", plot_mode="static")
            self.table.show()

        elif self.ec_rb.isChecked():

            survival_table = []
            tid = 0
            for name, group in data.groupby(groupby):
                print(name)
                if groupby[0] == "pos_path":
                    metadata = collect_experiment_metadata(pos_path=name[0])
                elif groupby[0] == "well_path":
                    metadata = collect_experiment_metadata(well_path=name[0])
                ks_estimator = group["survival_fit"].values[0]
                if ks_estimator != ks_estimator:
                    continue
                survival = ks_estimator.survival_function_
                ecN = qth_survival_times(
                    float(self.ec_slider.value()) / 100.0, survival
                )
                dico = metadata.copy()
                dico.update(
                    {
                        "qth": int(self.ec_slider.value()),
                        f"EC{int(self.ec_slider.value())}% [min]": ecN,
                    }
                )
                survival_table.append(dico)
                tid += 1

            survival_table = pd.DataFrame(survival_table)
            self.table = TableUI(survival_table, f"Survival data", plot_mode="static")
            self.table.show()
