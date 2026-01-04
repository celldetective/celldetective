from PyQt5.QtWidgets import (
    QCheckBox,
    QLineEdit,
    QListWidget,
    QTabWidget,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QRadioButton,
    QLabel,
    QButtonGroup,
    QSizePolicy,
    QComboBox,
    QSpacerItem,
    QGridLayout,
)
from celldetective.gui.gui_utils import ThresholdLineEdit, QuickSliderLayout
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from superqt import (
    QLabeledRangeSlider,
    QLabeledSlider,
    QLabeledDoubleRangeSlider,
    QSearchableComboBox,
)

from superqt.fonticon import icon
from fonticon_mdi6 import MDI6
from celldetective.utils.parsing import _extract_channel_indices_from_config
from celldetective.gui.base.styles import Styles
from celldetective.gui.base.components import CelldetectiveWidget

from functools import partial
from glob import glob
import os

import numpy as np


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


class BackgroundFitCorrectionLayout(QGridLayout, Styles):
    """docstring for ClassName"""

    def __init__(self, parent_window=None, *args):
        super().__init__(*args)

        self.parent_window = parent_window

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

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.thresh_lbl = QLabel("Threshold: ")
        self.thresh_lbl.setToolTip(
            "Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation."
        )
        self.threshold_viewer_btn = QPushButton()
        self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.threshold_viewer_btn.setStyleSheet(self.button_select_all)
        self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)
        self.threshold_viewer_btn.setToolTip("Set the threshold graphically.")

        self.model_lbl = QLabel("Model: ")
        self.model_lbl.setToolTip("2D model to fit the background with.")
        self.models_cb = QComboBox()
        self.models_cb.addItems(["paraboloid", "plane"])
        self.models_cb.setToolTip("2D model to fit the background with.")

        self.corrected_stack_viewer = QPushButton("")
        self.corrected_stack_viewer.setStyleSheet(self.button_select_all)
        self.corrected_stack_viewer.setIcon(icon(MDI6.eye_outline, color="black"))
        self.corrected_stack_viewer.setToolTip("View corrected image")
        self.corrected_stack_viewer.clicked.connect(self.preview_correction)
        self.corrected_stack_viewer.setIconSize(QSize(20, 20))

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.threshold_le = ThresholdLineEdit(
            init_value=2,
            connected_buttons=[
                self.threshold_viewer_btn,
                self.corrected_stack_viewer,
                self.add_correction_btn,
            ],
        )

    def add_to_layout(self):

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_lbl, 25)
        channel_layout.addWidget(self.channels_cb, 75)
        self.addLayout(channel_layout, 0, 0, 1, 3)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.thresh_lbl, 25)
        subthreshold_layout = QHBoxLayout()
        subthreshold_layout.addWidget(self.threshold_le, 95)
        subthreshold_layout.addWidget(self.threshold_viewer_btn, 5)

        threshold_layout.addLayout(subthreshold_layout, 75)
        self.addLayout(threshold_layout, 1, 0, 1, 3)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_lbl, 25)
        model_layout.addWidget(self.models_cb, 75)
        self.addLayout(model_layout, 2, 0, 1, 3)

        self.operation_layout = OperationLayout()
        self.addLayout(self.operation_layout, 3, 0, 1, 3)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.add_correction_btn, 95)
        correction_layout.addWidget(self.corrected_stack_viewer, 5)
        self.addLayout(correction_layout, 4, 0, 1, 3)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.addItem(verticalSpacer, 5, 0, 1, 3)

    def add_instructions_to_parent_list(self):

        self.generate_instructions()
        self.parent_window.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_list.addItem(correction_description)

    def generate_instructions(self):

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "fit",
            "model": self.models_cb.currentText(),
            "threshold_on_std": self.threshold_le.get_threshold(),
            "operation": operation,
            "clip": clip,
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def set_threshold_graphically(self):
        from celldetective.gui.viewers import ThresholdedStackVisualizer

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:
            self.viewer = ThresholdedStackVisualizer(
                initial_threshold=thresh,
                parent_le=self.threshold_le,
                preprocessing=[["gauss", 2], ["std", 4]],
                stack_path=self.attr_parent.current_stack,
                n_channels=len(self.channel_names),
                target_channel=self.target_channel,
                window_title="Set the exclusion threshold",
            )
            self.viewer.show()

    def preview_correction(self):
        from celldetective.preprocessing import correct_background_model
        from celldetective.gui.viewers import StackVisualizer

        if (
            self.attr_parent.well_list.isMultipleSelection()
            or not self.attr_parent.well_list.isAnySelected()
            or self.attr_parent.position_list.isMultipleSelection()
            or not self.attr_parent.position_list.isAnySelected()
        ):

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Please select a single position...")
            msgBox.setWindowTitle("Critical")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        corrected_stack = correct_background_model(
            self.attr_parent.exp_dir,
            well_option=self.attr_parent.well_list.getSelectedIndices(),  # +1 ??
            position_option=self.attr_parent.position_list.getSelectedIndices(),  # +1??
            target_channel=self.channels_cb.currentText(),
            model=self.models_cb.currentText(),
            threshold_on_std=self.threshold_le.get_threshold(),
            operation=operation,
            clip=clip,
            export=False,
            return_stacks=True,
            activation_protocol=[["gauss", 2], ["std", 4]],
            show_progress_per_well=True,
            show_progress_per_pos=False,
        )

        if corrected_stack:
            self.viewer = StackVisualizer(
                stack=corrected_stack[0],
                window_title="Corrected channel",
                target_channel=self.channels_cb.currentIndex(),
                frame_slider=True,
                contrast_slider=True,
            )
            self.viewer.show()
        else:
            print("Corrected stack could not be generated... No stack available...")


class LocalCorrectionLayout(BackgroundFitCorrectionLayout):
    """docstring for ClassName"""

    def __init__(self, *args):

        super().__init__(*args)

        if hasattr(self.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window
        elif hasattr(self.parent_window.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window.parent_window
        else:
            self.attr_parent = (
                self.parent_window.parent_window.parent_window.parent_window
            )

        self.thresh_lbl.setText("Distance: ")
        self.thresh_lbl.setToolTip(
            "Distance from the cell mask over which to estimate local intensity."
        )

        self.models_cb.clear()
        self.models_cb.addItems(["mean", "median"])

        self.threshold_le.set_threshold(5)
        self.threshold_le.connected_buttons = [
            self.threshold_viewer_btn,
            self.add_correction_btn,
        ]
        self.threshold_le.setValidator(QIntValidator())

        self.threshold_viewer_btn.disconnect()
        self.threshold_viewer_btn.clicked.connect(self.set_distance_graphically)

        self.corrected_stack_viewer.hide()

    def set_distance_graphically(self):
        from celldetective.gui.viewers import CellEdgeVisualizer

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:

            self.viewer = CellEdgeVisualizer(
                cell_type=self.parent_window.parent_window.mode,
                stack_path=self.attr_parent.current_stack,
                parent_le=self.threshold_le,
                n_channels=len(self.channel_names),
                target_channel=self.channels_cb.currentIndex(),
                edge_range=(0, 30),
                initial_edge=int(thresh),
                invert=True,
                window_title="Set an edge distance to estimate local intensity",
                channel_cb=False,
                PxToUm=1,
            )
            self.viewer.show()

    def generate_instructions(self):

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "local",
            "model": self.models_cb.currentText(),
            "distance": int(self.threshold_le.get_threshold()),
            "operation": operation,
            "clip": clip,
        }


class OperationLayout(QVBoxLayout):
    """docstring for ClassName"""

    def __init__(self, ratio=(0.25, 0.75), *args):

        super().__init__(*args)

        self.ratio = ratio
        self.generate_widgets()
        self.generate_layout()

    def generate_widgets(self):

        self.operation_lbl = QLabel("Operation: ")
        self.operation_group = QButtonGroup()
        self.subtract_btn = QRadioButton("Subtract")
        self.divide_btn = QRadioButton("Divide")
        self.subtract_btn.toggled.connect(self.activate_clipping_options)
        self.divide_btn.toggled.connect(self.activate_clipping_options)

        self.operation_group.addButton(self.subtract_btn)
        self.operation_group.addButton(self.divide_btn)

        self.clip_group = QButtonGroup()
        self.clip_btn = QRadioButton("Clip")
        self.clip_not_btn = QRadioButton("Do not clip")

        self.clip_group.addButton(self.clip_btn)
        self.clip_group.addButton(self.clip_not_btn)

    def generate_layout(self):

        operation_layout = QHBoxLayout()
        operation_layout.addWidget(self.operation_lbl, 100 * int(self.ratio[0]))
        operation_layout.addWidget(
            self.subtract_btn, 100 * int(self.ratio[1]) // 2, alignment=Qt.AlignCenter
        )
        operation_layout.addWidget(
            self.divide_btn, 100 * int(self.ratio[1]) // 2, alignment=Qt.AlignCenter
        )
        self.addLayout(operation_layout)

        clip_layout = QHBoxLayout()
        clip_layout.addWidget(QLabel(""), 100 * int(self.ratio[0]))
        clip_layout.addWidget(
            self.clip_btn, 100 * int(self.ratio[1]) // 4, alignment=Qt.AlignCenter
        )
        clip_layout.addWidget(
            self.clip_not_btn, 100 * int(self.ratio[1]) // 4, alignment=Qt.AlignCenter
        )
        clip_layout.addWidget(QLabel(""), 100 * int(self.ratio[1]) // 2)
        self.addLayout(clip_layout)

        self.subtract_btn.click()
        self.clip_not_btn.click()

    def activate_clipping_options(self):

        if self.subtract_btn.isChecked():
            self.clip_btn.setEnabled(True)
            self.clip_not_btn.setEnabled(True)
        else:
            self.clip_btn.setEnabled(False)
            self.clip_not_btn.setEnabled(False)


class ProtocolDesignerLayout(QVBoxLayout, Styles):
    """Multi tabs and list widget configuration for background correction
    in preprocessing and measurements
    """

    def __init__(
        self,
        parent_window=None,
        tab_layouts=[],
        tab_names=[],
        title="",
        list_title="",
        *args,
    ):

        super().__init__(*args)

        self.title = title
        self.parent_window = parent_window
        self.channel_names = self.parent_window.channel_names
        self.tab_layouts = tab_layouts
        self.tab_names = tab_names
        self.list_title = list_title
        self.protocols = []
        assert len(self.tab_layouts) == len(self.tab_names)

        self.generate_widgets()
        self.generate_layout()

    def generate_widgets(self):

        self.title_lbl = QLabel(self.title)
        self.title_lbl.setStyleSheet(
            """
			font-weight: bold;
			padding: 0px;
			"""
        )

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for k in range(len(self.tab_layouts)):
            wg = CelldetectiveWidget()
            self.tab_layouts[k].parent_window = self
            wg.setLayout(self.tab_layouts[k])
            self.tabs.addTab(wg, self.tab_names[k])

        self.protocol_list_lbl = QLabel(self.list_title)
        self.protocol_list = QListWidget()

        self.delete_protocol_btn = QPushButton("")
        self.delete_protocol_btn.setStyleSheet(self.button_select_all)
        self.delete_protocol_btn.setIcon(icon(MDI6.trash_can, color="black"))
        self.delete_protocol_btn.setToolTip("Remove.")
        self.delete_protocol_btn.setIconSize(QSize(20, 20))
        self.delete_protocol_btn.clicked.connect(self.remove_protocol_from_list)

    def generate_layout(self):

        self.correction_layout = QVBoxLayout()

        self.background_correction_layout = QVBoxLayout()
        self.background_correction_layout.setContentsMargins(0, 0, 0, 0)
        self.title_layout = QHBoxLayout()
        self.title_layout.addWidget(self.title_lbl, 100, alignment=Qt.AlignCenter)
        self.background_correction_layout.addLayout(self.title_layout)
        self.background_correction_layout.addWidget(self.tabs)
        self.correction_layout.addLayout(self.background_correction_layout)

        self.addLayout(self.correction_layout)

        self.list_layout = QVBoxLayout()
        list_header_layout = QHBoxLayout()
        list_header_layout.addWidget(self.protocol_list_lbl)
        list_header_layout.addWidget(self.delete_protocol_btn, alignment=Qt.AlignRight)
        self.list_layout.addLayout(list_header_layout)
        self.list_layout.addWidget(self.protocol_list)

        self.addLayout(self.list_layout)

    def remove_protocol_from_list(self):

        current_item = self.protocol_list.currentRow()
        if current_item > -1:
            del self.protocols[current_item]
            self.protocol_list.takeItem(current_item)


class ChannelOffsetOptionsLayout(QVBoxLayout, Styles):

    def __init__(self, parent_window=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.parent_window = parent_window
        if hasattr(self.parent_window.parent_window, "exp_config"):
            self.attr_parent = self.parent_window.parent_window
        else:
            self.attr_parent = self.parent_window.parent_window.parent_window

        self.channel_names = self.attr_parent.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.shift_lbl = QLabel("Shift: ")
        self.shift_h_lbl = QLabel("(h): ")
        self.shift_v_lbl = QLabel("(v): ")

        self.set_shift_btn = QPushButton()
        self.set_shift_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.set_shift_btn.setStyleSheet(self.button_select_all)
        self.set_shift_btn.setToolTip("Set the channel shift.")
        self.set_shift_btn.clicked.connect(self.open_offset_viewer)

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.vertical_shift_le = ThresholdLineEdit(
            init_value=0,
            connected_buttons=[self.add_correction_btn],
            placeholder="vertical shift [pixels]",
            value_type="float",
        )
        self.horizontal_shift_le = ThresholdLineEdit(
            init_value=0,
            connected_buttons=[self.add_correction_btn],
            placeholder="vertical shift [pixels]",
            value_type="float",
        )

    def add_to_layout(self):

        channel_ch_hbox = QHBoxLayout()
        channel_ch_hbox.addWidget(self.channel_lbl, 25)
        channel_ch_hbox.addWidget(self.channels_cb, 75)
        self.addLayout(channel_ch_hbox)

        shift_hbox = QHBoxLayout()
        shift_hbox.addWidget(self.shift_lbl, 25)

        shift_subhbox = QHBoxLayout()
        shift_subhbox.addWidget(self.shift_h_lbl, 10)
        shift_subhbox.addWidget(self.horizontal_shift_le, 75 // 2)
        shift_subhbox.addWidget(self.shift_v_lbl, 10)
        shift_subhbox.addWidget(self.vertical_shift_le, 75 // 2)
        shift_subhbox.addWidget(self.set_shift_btn, 5)

        shift_hbox.addLayout(shift_subhbox, 75)
        self.addLayout(shift_hbox)

        btn_hbox = QHBoxLayout()
        btn_hbox.addWidget(self.add_correction_btn, 95)
        self.addLayout(btn_hbox)

    def add_instructions_to_parent_list(self):

        self.generate_instructions()
        self.parent_window.protocol_layout.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_layout.protocol_list.addItem(correction_description)

    def generate_instructions(self):

        self.instructions = {
            "correction_type": "offset",
            "target_channel": self.channels_cb.currentText(),
            "correction_horizontal": self.horizontal_shift_le.get_threshold(),
            "correction_vertical": self.vertical_shift_le.get_threshold(),
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def open_offset_viewer(self):
        from celldetective.gui.viewers import ChannelOffsetViewer

        self.attr_parent.locate_image()
        self.set_target_channel()

        if self.attr_parent.current_stack is not None:
            self.viewer = ChannelOffsetViewer(
                parent_window=self,
                stack_path=self.attr_parent.current_stack,
                channel_names=self.attr_parent.exp_channels,
                n_channels=len(self.channel_names),
                channel_cb=True,
                target_channel=self.target_channel,
                window_title="offset viewer",
            )
            self.viewer.show()


class BackgroundModelFreeCorrectionLayout(QGridLayout, Styles):
    """docstring for ClassName"""

    def __init__(self, parent_window=None, *args):
        super().__init__(*args)

        self.parent_window = parent_window

        if hasattr(self.parent_window.parent_window, "exp_config"):
            self.attr_parent = self.parent_window.parent_window
        else:
            self.attr_parent = self.parent_window.parent_window.parent_window

        self.channel_names = self.attr_parent.exp_channels

        self.setContentsMargins(15, 15, 15, 15)
        self.generate_widgets()
        self.add_to_layout()

    def generate_widgets(self):

        self.channel_lbl = QLabel("Channel: ")
        self.channels_cb = QComboBox()
        self.channels_cb.addItems(self.channel_names)

        self.acquistion_lbl = QLabel("Stack mode: ")
        self.acq_mode_group = QButtonGroup()
        self.timeseries_rb = QRadioButton("timeseries")
        self.timeseries_rb.setChecked(True)
        self.tiles_rb = QRadioButton("tiles")
        self.acq_mode_group.addButton(self.timeseries_rb, 0)
        self.acq_mode_group.addButton(self.tiles_rb, 1)

        self.frame_range_slider = QLabeledRangeSlider(parent=None)

        self.timeseries_rb.toggled.connect(self.activate_time_range)
        self.tiles_rb.toggled.connect(self.activate_time_range)

        self.thresh_lbl = QLabel("Threshold: ")
        self.thresh_lbl.setToolTip(
            "Threshold on the STD-filtered image.\nPixel values above the threshold are\nconsidered as non-background and are\nmasked prior to background estimation."
        )
        self.threshold_viewer_btn = QPushButton()
        self.threshold_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.threshold_viewer_btn.setStyleSheet(self.button_select_all)
        self.threshold_viewer_btn.clicked.connect(self.set_threshold_graphically)

        self.background_viewer_btn = QPushButton()
        self.background_viewer_btn.setIcon(icon(MDI6.image_check, color="k"))
        self.background_viewer_btn.setStyleSheet(self.button_select_all)
        self.background_viewer_btn.setToolTip("View reconstructed background.")

        self.corrected_stack_viewer_btn = QPushButton("")
        self.corrected_stack_viewer_btn.setStyleSheet(self.button_select_all)
        self.corrected_stack_viewer_btn.setIcon(icon(MDI6.eye_outline, color="black"))
        self.corrected_stack_viewer_btn.setToolTip("View corrected image")
        self.corrected_stack_viewer_btn.clicked.connect(self.preview_correction)
        self.corrected_stack_viewer_btn.setIconSize(QSize(20, 20))

        self.add_correction_btn = QPushButton("Add correction")
        self.add_correction_btn.setStyleSheet(self.button_style_sheet_2)
        self.add_correction_btn.setIcon(icon(MDI6.plus, color="#1565c0"))
        self.add_correction_btn.setToolTip("Add correction.")
        self.add_correction_btn.setIconSize(QSize(25, 25))
        self.add_correction_btn.clicked.connect(self.add_instructions_to_parent_list)

        self.threshold_le = ThresholdLineEdit(
            init_value=2,
            connected_buttons=[
                self.threshold_viewer_btn,
                self.background_viewer_btn,
                self.corrected_stack_viewer_btn,
                self.add_correction_btn,
            ],
        )

        self.well_slider = QLabeledSlider(parent=None)

        self.background_viewer_btn.clicked.connect(self.estimate_bg)

        self.regress_cb = QCheckBox("Optimize for each frame?")
        self.regress_cb.toggled.connect(self.activate_coef_options)
        self.regress_cb.setChecked(False)

        self.coef_range_slider = QLabeledDoubleRangeSlider(parent=None)
        self.coef_range_layout = QuickSliderLayout(
            label="Coef. range: ",
            slider=self.coef_range_slider,
            slider_initial_value=(0.95, 1.05),
            slider_range=(0.75, 1.25),
            slider_tooltip="Coefficient range to increase or decrease the background intensity level...",
        )

        self.nbr_coefs_lbl = QLabel("Nbr of coefs: ")
        self.nbr_coefs_lbl.setToolTip(
            "Number of coefficients to be tested within range.\nThe more, the slower."
        )

        self.nbr_coef_le = QLineEdit()
        self.nbr_coef_le.setText("100")
        self.nbr_coef_le.setValidator(QIntValidator())
        self.nbr_coef_le.setPlaceholderText("nbr of coefs")

        self.coef_widgets = [
            self.coef_range_layout.qlabel,
            self.coef_range_slider,
            self.nbr_coefs_lbl,
            self.nbr_coef_le,
        ]
        for c in self.coef_widgets:
            c.setEnabled(False)

        self.interpolate_check = QCheckBox("interpolate NaNs")

    def add_to_layout(self):

        channel_layout = QHBoxLayout()
        channel_layout.addWidget(self.channel_lbl, 25)
        channel_layout.addWidget(self.channels_cb, 75)
        self.addLayout(channel_layout, 0, 0, 1, 3)

        acquisition_layout = QHBoxLayout()
        acquisition_layout.addWidget(self.acquistion_lbl, 25)
        acquisition_layout.addWidget(
            self.timeseries_rb, 75 // 2, alignment=Qt.AlignCenter
        )
        acquisition_layout.addWidget(self.tiles_rb, 75 // 2, alignment=Qt.AlignCenter)
        self.addLayout(acquisition_layout, 1, 0, 1, 3)

        frame_selection_layout = QuickSliderLayout(
            label="Time range: ",
            slider=self.frame_range_slider,
            slider_initial_value=(0, 5),
            slider_range=(0, self.attr_parent.len_movie),
            slider_tooltip="frame [#]",
            decimal_option=False,
        )
        frame_selection_layout.qlabel.setToolTip(
            "Frame range for which the background\nis most likely to be observed."
        )
        self.time_range_options = [
            self.frame_range_slider,
            frame_selection_layout.qlabel,
        ]
        self.addLayout(frame_selection_layout, 2, 0, 1, 3)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.thresh_lbl, 25)
        subthreshold_layout = QHBoxLayout()
        subthreshold_layout.addWidget(self.threshold_le, 95)
        subthreshold_layout.addWidget(self.threshold_viewer_btn, 5)
        threshold_layout.addLayout(subthreshold_layout, 75)
        self.addLayout(threshold_layout, 3, 0, 1, 3)

        background_layout = QuickSliderLayout(
            label="QC for well: ",
            slider=self.well_slider,
            slider_initial_value=1,
            slider_range=(1, len(self.attr_parent.wells)),
            slider_tooltip="well [#]",
            decimal_option=False,
            layout_ratio=(0.25, 0.70),
        )
        background_layout.addWidget(self.background_viewer_btn, 5)
        self.addLayout(background_layout, 4, 0, 1, 3)

        self.addWidget(self.regress_cb, 5, 0, 1, 3)

        self.addLayout(self.coef_range_layout, 6, 0, 1, 3)

        coef_nbr_layout = QHBoxLayout()
        coef_nbr_layout.addWidget(self.nbr_coefs_lbl, 25)
        coef_nbr_layout.addWidget(self.nbr_coef_le, 75)
        self.addLayout(coef_nbr_layout, 7, 0, 1, 3)

        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Offset: "), 25)
        self.camera_offset_le = QLineEdit("0")
        self.camera_offset_le.setPlaceholderText("camera black level")
        self.camera_offset_le.setValidator(QDoubleValidator())
        offset_layout.addWidget(self.camera_offset_le, 75)
        self.addLayout(offset_layout, 8, 0, 1, 3)

        self.operation_layout = OperationLayout()
        self.addLayout(self.operation_layout, 9, 0, 1, 3)

        self.addWidget(self.interpolate_check, 10, 0, 1, 1)

        correction_layout = QHBoxLayout()
        correction_layout.addWidget(self.add_correction_btn, 95)
        correction_layout.addWidget(self.corrected_stack_viewer_btn, 5)
        self.addLayout(correction_layout, 11, 0, 1, 3)

        # verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.addItem(verticalSpacer, 5, 0, 1, 3)

    def add_instructions_to_parent_list(self):

        self.generate_instructions()
        self.parent_window.protocols.append(self.instructions)
        correction_description = ""
        for index, (key, value) in enumerate(self.instructions.items()):
            if index > 0:
                correction_description += ", "
            correction_description += str(key) + " : " + str(value)
        self.parent_window.protocol_list.addItem(correction_description)

    def generate_instructions(self):

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"

        if self.regress_cb.isChecked():
            optimize_option = True
            opt_coef_range = self.coef_range_slider.value()
            opt_coef_nbr = int(self.nbr_coef_le.text())
        else:
            optimize_option = False
            opt_coef_range = None
            opt_coef_nbr = None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        if self.camera_offset_le.text() == "":
            offset = None
        else:
            offset = float(self.camera_offset_le.text().replace(",", "."))

        self.instructions = {
            "target_channel": self.channels_cb.currentText(),
            "correction_type": "model-free",
            "threshold_on_std": self.threshold_le.get_threshold(),
            "frame_range": self.frame_range_slider.value(),
            "mode": mode,
            "optimize_option": optimize_option,
            "opt_coef_range": opt_coef_range,
            "opt_coef_nbr": opt_coef_nbr,
            "operation": operation,
            "clip": clip,
            "offset": offset,
            "fix_nan": self.interpolate_check.isChecked(),
        }

    def set_target_channel(self):

        channel_indices = _extract_channel_indices_from_config(
            self.attr_parent.exp_config, [self.channels_cb.currentText()]
        )
        self.target_channel = channel_indices[0]

    def set_threshold_graphically(self):
        from celldetective.gui.viewers import ThresholdedStackVisualizer

        self.attr_parent.locate_image()
        self.set_target_channel()
        thresh = self.threshold_le.get_threshold()

        if self.attr_parent.current_stack is not None and thresh is not None:
            self.viewer = ThresholdedStackVisualizer(
                initial_threshold=thresh,
                parent_le=self.threshold_le,
                preprocessing=[["gauss", 2], ["std", 4]],
                stack_path=self.attr_parent.current_stack,
                n_channels=len(self.channel_names),
                target_channel=self.target_channel,
                window_title="Set the exclusion threshold",
            )
            self.viewer.show()

    def preview_correction(self):
        from celldetective.preprocessing import correct_background_model_free
        from celldetective.gui.viewers import StackVisualizer

        if (
            self.attr_parent.well_list.isMultipleSelection()
            or not self.attr_parent.well_list.isAnySelected()
            or self.attr_parent.position_list.isMultipleSelection()
            or not self.attr_parent.position_list.isAnySelected()
        ):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setText("Please select a single position...")
            msgBox.setWindowTitle("Warning")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return None

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"

        if self.regress_cb.isChecked():
            optimize_option = True
            opt_coef_range = self.coef_range_slider.value()
            opt_coef_nbr = int(self.nbr_coef_le.text())
        else:
            optimize_option = False
            opt_coef_range = None
            opt_coef_nbr = None

        if self.operation_layout.subtract_btn.isChecked():
            operation = "subtract"
        else:
            operation = "divide"
            clip = None

        if (
            self.operation_layout.clip_btn.isChecked()
            and self.operation_layout.subtract_btn.isChecked()
        ):
            clip = True
        else:
            clip = False

        corrected_stacks = correct_background_model_free(
            self.attr_parent.exp_dir,
            well_option=self.attr_parent.well_list.getSelectedIndices(),  # +1 ??
            position_option=self.attr_parent.getSelectedIndices(),  # +1??
            target_channel=self.channels_cb.currentText(),
            mode=mode,
            threshold_on_std=self.threshold_le.get_threshold(),
            frame_range=self.frame_range_slider.value(),
            optimize_option=optimize_option,
            opt_coef_range=opt_coef_range,
            opt_coef_nbr=opt_coef_nbr,
            operation=operation,
            clip=clip,
            export=False,
            return_stacks=True,
            fix_nan=self.interpolate_check.isChecked(),
            show_progress_per_well=True,
            show_progress_per_pos=False,
        )

        self.viewer = StackVisualizer(
            stack=corrected_stacks[0],
            window_title="Corrected channel",
            frame_slider=True,
            contrast_slider=True,
            target_channel=self.channels_cb.currentIndex(),
        )
        self.viewer.show()

    def activate_time_range(self):

        if self.timeseries_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(True)
        elif self.tiles_rb.isChecked():
            for wg in self.time_range_options:
                wg.setEnabled(False)

    def activate_coef_options(self):

        if self.regress_cb.isChecked():
            for c in self.coef_widgets:
                c.setEnabled(True)
        else:
            for c in self.coef_widgets:
                c.setEnabled(False)

    def estimate_bg(self):
        from celldetective.preprocessing import estimate_background_per_condition
        from celldetective.gui.viewers import StackVisualizer

        if self.timeseries_rb.isChecked():
            mode = "timeseries"
        elif self.tiles_rb.isChecked():
            mode = "tiles"

        bg = estimate_background_per_condition(
            self.attr_parent.exp_dir,
            well_option=self.well_slider.value() - 1,
            frame_range=self.frame_range_slider.value(),
            target_channel=self.channels_cb.currentText(),
            show_progress_per_pos=True,
            threshold_on_std=self.threshold_le.get_threshold(),
            mode=mode,
        )
        bg = bg[0]
        bg = bg["bg"]
        print(bg)
        if len(bg) > 0:

            self.viewer = StackVisualizer(
                stack=[bg],
                window_title="Reconstructed background",
                frame_slider=False,
            )
            self.viewer.show()
