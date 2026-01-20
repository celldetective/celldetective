"""
Copyright © 2023 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

from PyQt5.QtWidgets import (
    QComboBox,
    QLabel,
    QRadioButton,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt, QSize
from celldetective.gui.base.components import QHSeperationLine
from superqt import QLabeledDoubleSlider, QLabeledSlider
from celldetective.utils.experiment import extract_experiment_channels
import json
import numpy as np
from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
import os
from celldetective.gui.settings._settings_base import CelldetectiveSettingsPanel


class SettingsSignalAnnotator(CelldetectiveSettingsPanel):
    """
    UI to set normalization and animation parameters for the annotator tool.

    """

    def __init__(self, parent_window=None):

        self.parent_window = parent_window
        self.mode = self.parent_window.mode
        self.exp_dir = self.parent_window.exp_dir
        self.percentile_mode = False

        self.instructions_path = (
            self.parent_window.exp_dir
            + f"configs/signal_annotator_config_{self.mode}.json"
        )
        if self.mode == "pairs":
            self.instructions_path = (
                self.parent_window.exp_dir
                + "configs/signal_annotator_config_neighborhood.json"
            )

        self.channel_names, self.channels = extract_experiment_channels(self.exp_dir)
        self.channel_names = np.array(self.channel_names)
        self.channels = np.array(self.channels)
        self.log_option = False

        super().__init__(title="Configure signal annotator")

        self._add_to_layout()
        self._load_previous_instructions()

        self._adjust_size()
        self.resize(int(self.width() * 1.4), int(self._screen_height * 0.65))

    def _add_to_layout(self):

        sub_layout = QVBoxLayout()
        sub_layout.setContentsMargins(10, 10, 10, 20)
        sub_layout.setContentsMargins(30, 30, 30, 30)
        sub_layout.addWidget(self._modality_lbl)

        # Create radio buttons
        option_layout = QHBoxLayout()
        option_layout.addWidget(self.gs_btn, alignment=Qt.AlignCenter)
        option_layout.addWidget(self.rgb_btn, alignment=Qt.AlignCenter)
        sub_layout.addLayout(option_layout)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(QLabel(""), 90)
        btn_hbox.addWidget(self.log_btn, 5, alignment=Qt.AlignRight)
        btn_hbox.addWidget(self.percentile_btn, 5, alignment=Qt.AlignRight)
        sub_layout.addLayout(btn_hbox)

        for i in range(3):
            hlayout = QHBoxLayout()
            hlayout.addWidget(self.channel_cbs_lbls[i], 20)
            hlayout.addWidget(self.channel_cbs[i], 80)
            sub_layout.addLayout(hlayout)

            hlayout2 = QHBoxLayout()
            hlayout2.addWidget(self.min_val_lbls[i], 20)
            hlayout2.addWidget(self.min_val_les[i], 80)
            sub_layout.addLayout(hlayout2)

            hlayout3 = QHBoxLayout()
            hlayout3.addWidget(self.max_val_lbls[i], 20)
            hlayout3.addWidget(self.max_val_les[i], 80)
            sub_layout.addLayout(hlayout3)

        sub_layout.addWidget(self.hsep)
        hbox_frac = QHBoxLayout()
        hbox_frac.addWidget(self._fraction_lbl, 20)
        hbox_frac.addWidget(self.fraction_slider, 80)
        sub_layout.addLayout(hbox_frac)

        hbox_interval = QHBoxLayout()
        hbox_interval.addWidget(self._interval_lbl, 20)
        hbox_interval.addWidget(self.interval_slider, 80)
        sub_layout.addLayout(hbox_interval)

        self._layout.addLayout(sub_layout)

        self._layout.addWidget(self.submit_btn)

    def _create_widgets(self):
        """
        Create the widgets.

        """
        super()._create_widgets()

        self._modality_lbl = QLabel("Modality: ")
        self._fraction_lbl = QLabel("fraction: ")
        self._interval_lbl = QLabel("interval [ms]: ")

        self.gs_btn = QRadioButton("grayscale")
        self.gs_btn.setChecked(True)

        self.rgb_btn = QRadioButton("RGB")

        self.percentile_btn = QPushButton()
        self.percentile_btn.setIcon(icon(MDI6.percent_circle_outline, color="black"))
        self.percentile_btn.setIconSize(QSize(20, 20))
        self.percentile_btn.setStyleSheet(self.button_select_all)
        self.percentile_btn.setToolTip("Switch to percentile normalization values.")
        self.percentile_btn.clicked.connect(self.switch_to_absolute_normalization_mode)

        self.log_btn = QPushButton()
        self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
        self.log_btn.setStyleSheet(self.button_select_all)
        self.log_btn.clicked.connect(self.switch_to_log)
        self.log_btn.setToolTip("Log-transform the intensities.")
        self.log_btn.setIconSize(QSize(20, 20))

        self.channel_cbs = [QComboBox() for _ in range(3)]
        self.channel_cbs_lbls = [QLabel() for _ in range(3)]

        self.min_val_les = [QLineEdit("0") for _ in range(3)]
        self.min_val_lbls = [QLabel("Min value: ") for _ in range(3)]
        self.max_val_les = [QLineEdit("10000") for _ in range(3)]
        self.max_val_lbls = [QLabel("Max value: ") for _ in range(3)]

        self.rgb_text = ["R: ", "G: ", "B: "]

        for i in range(3):

            self.channel_cbs[i].addItems(self.channel_names)
            self.channel_cbs[i].setCurrentIndex(i)
            self.channel_cbs_lbls[i].setText(self.rgb_text[i])

            self.min_val_les[i].setValidator(self._floatValidator)
            self.max_val_les[i].setValidator(self._floatValidator)

        self.enable_channels()

        self.gs_btn.toggled.connect(self.enable_channels)
        self.rgb_btn.toggled.connect(self.enable_channels)

        self.hsep = QHSeperationLine()

        self.fraction_slider = QLabeledDoubleSlider()
        self.fraction_slider.setSingleStep(0.05)
        self.fraction_slider.setTickInterval(0.05)
        self.fraction_slider.setSingleStep(1)
        self.fraction_slider.setOrientation(Qt.Horizontal)
        self.fraction_slider.setRange(0.1, 1)
        self.fraction_slider.setValue(0.25)

        self.interval_slider = QLabeledSlider()
        self.interval_slider.setSingleStep(1)
        self.interval_slider.setTickInterval(1)
        self.interval_slider.setSingleStep(1)
        self.interval_slider.setOrientation(Qt.Horizontal)
        self.interval_slider.setRange(1, 1000)
        self.interval_slider.setValue(1)

    def enable_channels(self):
        """
        Enable three channels when RGB mode is checked.

        """

        if self.gs_btn.isChecked():

            self.log_btn.setEnabled(True)
            self.percentile_btn.setEnabled(False)

            for k in range(1, 3):
                self.channel_cbs[k].setEnabled(False)
                self.channel_cbs_lbls[k].setEnabled(False)

            for k in range(3):
                self.min_val_les[k].setEnabled(False)
                self.min_val_lbls[k].setEnabled(False)
                self.max_val_les[k].setEnabled(False)
                self.max_val_lbls[k].setEnabled(False)

        elif self.rgb_btn.isChecked():

            self.log_btn.setEnabled(False)
            self.percentile_btn.setEnabled(True)

            for k in range(3):
                self.channel_cbs[k].setEnabled(True)
                self.channel_cbs_lbls[k].setEnabled(True)

                self.min_val_les[k].setEnabled(True)
                self.min_val_lbls[k].setEnabled(True)
                self.max_val_les[k].setEnabled(True)
                self.max_val_lbls[k].setEnabled(True)

    def switch_to_absolute_normalization_mode(self):
        """
        Use absolute or percentile values for the normalization of each individual channel.

        """

        if self.percentile_mode:
            self.percentile_mode = False
            self.percentile_btn.setIcon(
                icon(MDI6.percent_circle_outline, color="black")
            )
            self.percentile_btn.setIconSize(QSize(20, 20))
            self.percentile_btn.setToolTip("Switch to percentile normalization values.")
            for k in range(3):
                self.min_val_lbls[k].setText("Min value: ")
                self.min_val_les[k].setText("0")
                self.max_val_lbls[k].setText("Max value: ")
                self.max_val_les[k].setText("10000")
        else:
            self.percentile_mode = True
            self.percentile_btn.setIcon(icon(MDI6.percent_circle, color="black"))
            self.percentile_btn.setIconSize(QSize(20, 20))
            self.percentile_btn.setToolTip("Switch to absolute normalization values.")
            for k in range(3):
                self.min_val_lbls[k].setText("Min percentile: ")
                self.min_val_les[k].setText("0.01")
                self.max_val_lbls[k].setText("Max percentile: ")
                self.max_val_les[k].setText("99.99")

    def _write_instructions(self):
        """
        Save the current configuration.

        """

        instructions = {
            "rgb_mode": self.rgb_btn.isChecked(),
            "percentile_mode": self.percentile_mode,
            "fraction": float(self.fraction_slider.value()),
            "interval": int(self.interval_slider.value()),
            "log": self.log_option,
        }
        max_i = 3 if self.rgb_btn.isChecked() else 1
        channels = []
        for i in range(max_i):
            channels.append(
                [
                    self.channel_cbs[i].currentText(),
                    float(self.min_val_les[i].text().replace(",", ".")),
                    float(self.max_val_les[i].text().replace(",", ".")),
                ]
            )
        instructions.update({"channels": channels})

        print("Instructions: ", instructions)
        file_name = self.instructions_path
        with open(file_name, "w") as f:
            json.dump(instructions, f, indent=4)
        print("Done.")
        self.close()

    def _load_previous_instructions(self):
        """
        Read and set the widgets to the last configuration.

        """

        print("Reading instructions..")
        if os.path.exists(self.instructions_path):
            with open(self.instructions_path, "r") as f:
                instructions = json.load(f)
                print(instructions)

                if "rgb_mode" in instructions:
                    rgb_mode = instructions["rgb_mode"]
                    if rgb_mode:
                        self.rgb_btn.setChecked(True)
                        self.gs_btn.setChecked(False)

                if "percentile_mode" in instructions:
                    percentile_mode = instructions["percentile_mode"]
                    if percentile_mode:
                        self.percentile_btn.click()

                if "channels" in instructions:
                    channels = instructions["channels"]

                    if len(channels) == 1:
                        max_iter = 1
                    else:
                        max_iter = 3

                    for i in range(max_iter):
                        idx = self.channel_cbs[i].findText(channels[i][0])
                        self.channel_cbs[i].setCurrentIndex(idx)
                        self.min_val_les[i].setText(
                            str(channels[i][1]).replace(".", ",")
                        )
                        self.max_val_les[i].setText(
                            str(channels[i][2]).replace(".", ",")
                        )

                if "fraction" in instructions:
                    fraction = instructions["fraction"]
                    self.fraction_slider.setValue(fraction)

                if "interval" in instructions:
                    interval = instructions["interval"]
                    self.interval_slider.setValue(interval)

                if "log" in instructions:
                    self.log_option = not instructions["log"]
                    self.switch_to_log()

    def switch_to_log(self):
        """
        Switch threshold histogram to log scale. Auto adjust.
        """

        if not self.log_option:
            self.log_btn.setIcon(icon(MDI6.math_log, color="#1565c0"))
            self.log_option = True
        else:
            self.log_btn.setIcon(icon(MDI6.math_log, color="black"))
            self.log_option = False
