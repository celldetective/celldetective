import os
from functools import partial
from glob import glob

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QLineEdit, QHBoxLayout
from fonticon_mdi6 import MDI6
from superqt import QSearchableComboBox
from superqt.fonticon import icon

from celldetective.gui.base.styles import Styles


class ChannelNormGenerator(QVBoxLayout, Styles):
    """Generator for list of channels"""

    def __init__(self, parent_window=None, init_n_channels=4, mode="signals", *args):
        super().__init__(*args)

        self.parent_window = parent_window
        self.mode = mode
        self.init_n_channels = init_n_channels

        if hasattr(self.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window
        elif hasattr(self.parent_window.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window.parent_window
        else:
            self.attr_parent = (
                self.parent_window.parent_window.parent_window.parent_window
            )

        self.channel_names = self.attr_parent.exp_channels
        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_cbs = [QSearchableComboBox() for i in range(self.init_n_channels)]
        self.channel_labels = [QLabel() for i in range(self.init_n_channels)]

        self.normalization_mode_btns = [
            QPushButton("") for i in range(self.init_n_channels)
        ]
        self.normalization_mode = [True for i in range(self.init_n_channels)]
        self.normalization_clip_btns = [
            QPushButton("") for i in range(self.init_n_channels)
        ]
        self.clip_option = [False for i in range(self.init_n_channels)]

        for i in range(self.init_n_channels):
            self.normalization_mode_btns[i].setIcon(
                icon(MDI6.percent_circle, color="#1565c0")
            )
            self.normalization_mode_btns[i].setIconSize(QSize(20, 20))
            self.normalization_mode_btns[i].setStyleSheet(self.button_select_all)
            self.normalization_mode_btns[i].setToolTip(
                "Switch to absolute normalization values."
            )
            self.normalization_mode_btns[i].clicked.connect(
                partial(self.switch_normalization_mode, i)
            )

            self.normalization_clip_btns[i].setIcon(
                icon(MDI6.content_cut, color="black")
            )
            self.normalization_clip_btns[i].setIconSize(QSize(20, 20))
            self.normalization_clip_btns[i].setStyleSheet(self.button_select_all)
            self.normalization_clip_btns[i].clicked.connect(
                partial(self.switch_clipping_mode, i)
            )
            self.normalization_clip_btns[i].setToolTip("clip")

        self.normalization_min_value_lbl = [
            QLabel("Min %: ") for i in range(self.init_n_channels)
        ]
        self.normalization_min_value_le = [
            QLineEdit("0.1") for i in range(self.init_n_channels)
        ]
        self.normalization_max_value_lbl = [
            QLabel("Max %: ") for i in range(self.init_n_channels)
        ]
        self.normalization_max_value_le = [
            QLineEdit("99.99") for i in range(self.init_n_channels)
        ]

        if self.mode == "signals":
            tables = glob(
                self.parent_window.exp_dir
                + os.sep.join(
                    [
                        "W*",
                        "*",
                        "output",
                        "tables",
                        f"trajectories_{self.parent_window.mode}.csv",
                    ]
                )
            )
            all_measurements = []
            for tab in tables:
                import pandas as pd

                cols = pd.read_csv(tab, nrows=1).columns.tolist()
                all_measurements.extend(cols)
            all_measurements = np.unique(all_measurements)

        if self.mode == "signals":
            generic_measurements = [
                "brightfield_channel",
                "live_nuclei_channel",
                "dead_nuclei_channel",
                "effector_fluo_channel",
                "adhesion_channel",
                "fluo_channel_1",
                "fluo_channel_2",
                "area",
                "area_bbox",
                "area_convex",
                "area_filled",
                "major_axis_length",
                "minor_axis_length",
                "eccentricity",
                "equivalent_diameter_area",
                "euler_number",
                "extent",
                "feret_diameter_max",
                "orientation",
                "perimeter",
                "perimeter_crofton",
                "solidity",
                "angular_second_moment",
                "contrast",
                "correlation",
                "sum_of_square_variance",
                "inverse_difference_moment",
                "sum_average",
                "sum_variance",
                "sum_entropy",
                "entropy",
                "difference_variance",
                "difference_entropy",
                "information_measure_of_correlation_1",
                "information_measure_of_correlation_2",
                "maximal_correlation_coefficient",
                "POSITION_X",
                "POSITION_Y",
            ]
        elif self.mode == "channels":
            generic_measurements = [
                "brightfield_channel",
                "live_nuclei_channel",
                "dead_nuclei_channel",
                "effector_fluo_channel",
                "adhesion_channel",
                "fluo_channel_1",
                "fluo_channel_2",
                "None",
            ]

        if self.mode == "channels":
            all_measurements = []
            exp_ch = self.attr_parent.exp_channels
            for c in exp_ch:
                all_measurements.append(c)

        self.channel_items = np.unique(generic_measurements + list(all_measurements))
        self.channel_items = np.insert(self.channel_items, 0, "--")

        self.add_col_btn = QPushButton("Add channel")
        self.add_col_btn.clicked.connect(self.add_channel)
        self.add_col_btn.setStyleSheet(self.button_add)
        self.add_col_btn.setIcon(icon(MDI6.plus, color="black"))

    def add_channel(self):

        self.channel_cbs.append(QSearchableComboBox())
        self.channel_labels.append(QLabel())
        self.channel_cbs[-1].addItems(self.channel_items)
        self.channel_cbs[-1].currentIndexChanged.connect(self.check_valid_channels)
        self.channel_labels[-1].setText(f"channel {len(self.channel_cbs)-1}: ")

        self.normalization_mode_btns.append(QPushButton(""))
        self.normalization_mode.append(True)
        self.normalization_clip_btns.append(QPushButton(""))
        self.clip_option.append(False)

        self.normalization_mode_btns[-1].setIcon(
            icon(MDI6.percent_circle, color="#1565c0")
        )
        self.normalization_mode_btns[-1].setIconSize(QSize(20, 20))
        self.normalization_mode_btns[-1].setStyleSheet(self.button_select_all)
        self.normalization_mode_btns[-1].setToolTip(
            "Switch to absolute normalization values."
        )
        self.normalization_mode_btns[-1].clicked.connect(
            partial(self.switch_normalization_mode, len(self.channel_cbs) - 1)
        )

        self.normalization_clip_btns[-1].setIcon(icon(MDI6.content_cut, color="black"))
        self.normalization_clip_btns[-1].setIconSize(QSize(20, 20))
        self.normalization_clip_btns[-1].setStyleSheet(self.button_select_all)
        self.normalization_clip_btns[-1].clicked.connect(
            partial(self.switch_clipping_mode, len(self.channel_cbs) - 1)
        )
        self.normalization_clip_btns[-1].setToolTip("clip")

        self.normalization_min_value_lbl.append(QLabel("Min %: "))
        self.normalization_min_value_le.append(QLineEdit("0.1"))
        self.normalization_max_value_lbl.append(QLabel("Max %: "))
        self.normalization_max_value_le.append(QLineEdit("99.99"))

        ch_layout = QHBoxLayout()
        ch_layout.addWidget(self.channel_labels[-1], 30)
        ch_layout.addWidget(self.channel_cbs[-1], 70)
        self.channels_vb.addLayout(ch_layout)

        channel_norm_options_layout = QHBoxLayout()
        channel_norm_options_layout.addWidget(QLabel(""), 30)
        ch_norm_sublayout = QHBoxLayout()
        ch_norm_sublayout.addWidget(self.normalization_min_value_lbl[-1])
        ch_norm_sublayout.addWidget(self.normalization_min_value_le[-1])
        ch_norm_sublayout.addWidget(self.normalization_max_value_lbl[-1])
        ch_norm_sublayout.addWidget(self.normalization_max_value_le[-1])
        ch_norm_sublayout.addWidget(self.normalization_clip_btns[-1])
        ch_norm_sublayout.addWidget(self.normalization_mode_btns[-1])
        channel_norm_options_layout.addLayout(ch_norm_sublayout, 70)

        self.channels_vb.addLayout(channel_norm_options_layout)

    def add_to_layout(self):

        self.channels_vb = QVBoxLayout()
        self.channel_option_layouts = []
        for i in range(len(self.channel_cbs)):

            ch_layout = QHBoxLayout()
            self.channel_labels[i].setText(f"channel {i}: ")
            ch_layout.addWidget(self.channel_labels[i], 30)
            self.channel_cbs[i].addItems(self.channel_items)
            self.channel_cbs[i].currentIndexChanged.connect(self.check_valid_channels)
            ch_layout.addWidget(self.channel_cbs[i], 70)
            self.channels_vb.addLayout(ch_layout)

            channel_norm_options_layout = QHBoxLayout()
            # channel_norm_options_layout.setContentsMargins(130,0,0,0)
            channel_norm_options_layout.addWidget(QLabel(""), 30)
            ch_norm_sublayout = QHBoxLayout()
            ch_norm_sublayout.addWidget(self.normalization_min_value_lbl[i])
            ch_norm_sublayout.addWidget(self.normalization_min_value_le[i])
            ch_norm_sublayout.addWidget(self.normalization_max_value_lbl[i])
            ch_norm_sublayout.addWidget(self.normalization_max_value_le[i])
            ch_norm_sublayout.addWidget(self.normalization_clip_btns[i])
            ch_norm_sublayout.addWidget(self.normalization_mode_btns[i])
            channel_norm_options_layout.addLayout(ch_norm_sublayout, 70)
            self.channels_vb.addLayout(channel_norm_options_layout)

        self.addLayout(self.channels_vb)

        add_hbox = QHBoxLayout()
        add_hbox.addWidget(QLabel(""), 66)
        add_hbox.addWidget(self.add_col_btn, 33, alignment=Qt.AlignRight)
        self.addLayout(add_hbox)

    def switch_normalization_mode(self, index):
        """
        Use absolute or percentile values for the normalization of each individual channel.

        """

        currentNormMode = self.normalization_mode[index]
        self.normalization_mode[index] = not currentNormMode

        if self.normalization_mode[index]:
            self.normalization_mode_btns[index].setIcon(
                icon(MDI6.percent_circle, color="#1565c0")
            )
            self.normalization_mode_btns[index].setIconSize(QSize(20, 20))
            self.normalization_mode_btns[index].setStyleSheet(self.button_select_all)
            self.normalization_mode_btns[index].setToolTip(
                "Switch to absolute normalization values."
            )
            self.normalization_min_value_lbl[index].setText("Min %: ")
            self.normalization_max_value_lbl[index].setText("Max %: ")
            self.normalization_min_value_le[index].setText("0.1")
            self.normalization_max_value_le[index].setText("99.99")

        else:
            self.normalization_mode_btns[index].setIcon(
                icon(MDI6.percent_circle_outline, color="black")
            )
            self.normalization_mode_btns[index].setIconSize(QSize(20, 20))
            self.normalization_mode_btns[index].setStyleSheet(self.button_select_all)
            self.normalization_mode_btns[index].setToolTip(
                "Switch to percentile normalization values."
            )
            self.normalization_min_value_lbl[index].setText("Min: ")
            self.normalization_min_value_le[index].setText("0")
            self.normalization_max_value_lbl[index].setText("Max: ")
            self.normalization_max_value_le[index].setText("1000")

    def switch_clipping_mode(self, index):

        currentClipMode = self.clip_option[index]
        self.clip_option[index] = not currentClipMode

        if self.clip_option[index]:
            self.normalization_clip_btns[index].setIcon(
                icon(MDI6.content_cut, color="#1565c0")
            )
            self.normalization_clip_btns[index].setIconSize(QSize(20, 20))
            self.normalization_clip_btns[index].setStyleSheet(self.button_select_all)

        else:
            self.normalization_clip_btns[index].setIcon(
                icon(MDI6.content_cut, color="black")
            )
            self.normalization_clip_btns[index].setIconSize(QSize(20, 20))
            self.normalization_clip_btns[index].setStyleSheet(self.button_select_all)

    def check_valid_channels(self):

        if hasattr(self.parent_window, "submit_btn"):
            if np.all([cb.currentText() == "--" for cb in self.channel_cbs]):
                self.parent_window.submit_btn.setEnabled(False)

        if hasattr(self.parent_window, "spatial_calib_le") and hasattr(
            self.parent_window, "submit_btn"
        ):
            if self.parent_window.spatial_calib_le.text() != "--":
                self.parent_window.submit_btn.setEnabled(True)
        elif hasattr(self.parent_window, "submit_btn"):
            self.parent_window.submit_btn.setEnabled(True)
