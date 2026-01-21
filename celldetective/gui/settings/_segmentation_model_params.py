import json
import os

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QVBoxLayout, QComboBox, QPushButton, QHBoxLayout, QLabel
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.utils.model_loaders import locate_segmentation_model


class SegModelParamsWidget(CelldetectiveWidget):

    def __init__(
        self, parent_window=None, model_name="SD_versatile_fluo", *args, **kwargs
    ):

        super().__init__(*args)
        self.setWindowTitle("Channels")
        self.parent_window = parent_window
        self.model_name = model_name
        self.locate_model_path()
        self.required_channels = self.input_config["channels"]
        self.onlyFloat = QDoubleValidator()

        # Setting up references to parent window attributes
        if hasattr(self.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window
        elif hasattr(self.parent_window.parent_window.parent_window, "locate_image"):
            self.attr_parent = self.parent_window.parent_window.parent_window
        else:
            self.attr_parent = (
                self.parent_window.parent_window.parent_window.parent_window
            )

        # Set up layout and widgets
        self.layout = QVBoxLayout()
        self.populate_widgets()
        self.setLayout(self.layout)
        center_window(self)

    def locate_model_path(self):

        self.model_complete_path = locate_segmentation_model(self.model_name)
        if self.model_complete_path is None:
            print("Model could not be found. Abort.")
            self.abort_process()
        else:
            print(f"Model path: {self.model_complete_path}...")

        if not os.path.exists(self.model_complete_path + "config_input.json"):
            print(
                "The configuration for the inputs to the model could not be located. Abort."
            )
            self.abort_process()

        with open(self.model_complete_path + "config_input.json") as config_file:
            self.input_config = json.load(config_file)

    def populate_widgets(self):

        self.n_channels = len(self.required_channels)
        self.channel_cbs = [QComboBox() for i in range(self.n_channels)]

        # Button to view the current stack with a scale bar
        self.view_diameter_btn = QPushButton()
        self.view_diameter_btn.setStyleSheet(self.button_select_all)
        self.view_diameter_btn.setIcon(icon(MDI6.image_check, color="black"))
        self.view_diameter_btn.setToolTip("View stack.")
        self.view_diameter_btn.setIconSize(QSize(20, 20))
        self.view_diameter_btn.clicked.connect(self.view_current_stack_with_scale_bar)

        # Line edit for entering cell diameter
        self.diameter_le = ThresholdLineEdit(
            init_value=40,
            connected_buttons=[self.view_diameter_btn],
            placeholder="cell diameter in µm",
            value_type="float",
        )

        available_channels = list(self.attr_parent.exp_channels) + ["None"]
        # Populate the comboboxes with available channels from the experiment
        for k in range(self.n_channels):
            hbox_channel = QHBoxLayout()
            hbox_channel.addWidget(QLabel(f"channel {k+1}: "), 33)

            ch_vbox = QVBoxLayout()
            ch_vbox.addWidget(
                QLabel(f"Req: {self.required_channels[k]}"), alignment=Qt.AlignLeft
            )
            ch_vbox.addWidget(self.channel_cbs[k])

            self.channel_cbs[k].addItems(
                available_channels
            )  # Give none option for more than one channel input
            idx = self.channel_cbs[k].findText(self.required_channels[k])

            if idx >= 0:
                self.channel_cbs[k].setCurrentIndex(idx)
            else:
                self.channel_cbs[k].setCurrentIndex(len(available_channels) - 1)

            hbox_channel.addLayout(ch_vbox, 66)
            self.layout.addLayout(hbox_channel)

        if "cell_size_um" in self.input_config:

            # Layout for diameter input and button
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel("cell size [µm]: "), 33)
            hbox.addWidget(self.diameter_le, 61)
            hbox.addWidget(self.view_diameter_btn)
            self.layout.addLayout(hbox)

            self.diameter_le.set_threshold(self.input_config["cell_size_um"])

            # size_hbox = QHBoxLayout()
            # size_hbox.addWidget(QLabel('cell size [µm]: '), 33)
            # self.size_le = QLineEdit(str(self.input_config['cell_size_um']).replace('.',','))
            # self.size_le.setValidator(self.onlyFloat)
            # size_hbox.addWidget(self.size_le, 66)
            # self.layout.addLayout(size_hbox)

        # Button to apply the StarDist settings
        self.set_btn = QPushButton("set")
        self.set_btn.setStyleSheet(self.button_style_sheet)
        self.set_btn.clicked.connect(
            self.parent_window.set_selected_channels_for_segmentation
        )
        self.layout.addWidget(self.set_btn)

    def view_current_stack_with_scale_bar(self):
        """
        Displays the current image stack with a scale bar, allowing users to visually estimate cell diameters.
        """

        self.attr_parent.locate_image()
        if self.attr_parent.current_stack is not None:
            max_size = np.amax([self.attr_parent.shape_x, self.attr_parent.shape_y])
            self.viewer = CellSizeViewer(
                initial_diameter=float(self.diameter_le.text().replace(",", ".")),
                parent_le=self.diameter_le,
                stack_path=self.attr_parent.current_stack,
                window_title=f"Position {self.attr_parent.position_list.currentText()}",
                diameter_slider_range=(0, max_size * self.attr_parent.PxToUm),
                frame_slider=True,
                contrast_slider=True,
                channel_cb=True,
                channel_names=self.attr_parent.exp_channels,
                n_channels=self.attr_parent.nbr_channels,
                PxToUm=self.attr_parent.PxToUm,
            )
            self.viewer.show()
