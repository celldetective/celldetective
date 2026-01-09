import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QVBoxLayout, QComboBox, QHBoxLayout, QLabel, QPushButton

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.utils.model_loaders import locate_signal_model


class SignalModelParamsWidget(CelldetectiveWidget):

    def __init__(self, parent_window=None, model_name=None, *args, **kwargs):

        super().__init__(*args)
        self.setWindowTitle("Signals")
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

        self.model_complete_path = locate_signal_model(self.model_name)
        if self.model_complete_path is None:
            raise ValueError(f"Model {self.model_name} could not be found.")
        else:
            print(f"Model path: {self.model_complete_path}...")

        config_path = os.path.join(self.model_complete_path, "config_input.json")
        if not os.path.exists(config_path):
            raise ValueError(
                f"The configuration for the inputs to the model could not be located at {config_path}."
            )

        with open(config_path) as config_file:
            self.input_config = json.load(config_file)

    def populate_widgets(self):

        self.n_channels = len(self.required_channels)
        self.channel_cbs = [QComboBox() for i in range(self.n_channels)]

        self.parent_window.load_available_tables()
        available_channels = list(self.parent_window.signals) + ["None"]
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

        # Button to apply the StarDist settings
        self.set_btn = QPushButton("set")
        self.set_btn.setStyleSheet(self.button_style_sheet)
        self.set_btn.clicked.connect(
            self.parent_window.set_selected_signals_for_event_detection
        )
        self.layout.addWidget(self.set_btn)
