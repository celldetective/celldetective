import json
import logging
import os
from typing import Optional

logger = logging.getLogger("celldetective")

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QMainWindow

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.model_channel_selection import ModelChannelSelection
from celldetective.gui.base.utils import center_window
from celldetective.utils.model_loaders import locate_signal_model


class SignalModelParamsWidget(CelldetectiveWidget):

    def __init__(
        self,
        parent_window: Optional[QMainWindow] = None,
        model_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the SignalModelParamsWidget.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The parent window.
        model_name : str, optional
            The name of the model.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

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
        """Locate the model path."""
        self.model_complete_path = locate_signal_model(self.model_name)
        if self.model_complete_path is None:
            raise ValueError(f"Model {self.model_name} could not be found.")
        else:
            logger.info(f"Model path: {self.model_complete_path}...")

        config_path = os.path.join(self.model_complete_path, "config_input.json")
        if not os.path.exists(config_path):
            raise ValueError(
                f"The configuration for the inputs to the model could not be located at {config_path}."
            )

        with open(config_path) as config_file:
            self.input_config = json.load(config_file)

    def populate_widgets(self):
        """Populate the widgets."""
        self.parent_window.load_available_tables()

        # The same rows as the segmentation channel dialog, over the measured
        # signals rather than the experiment's channels: a signal model maps its
        # inputs exactly the way a segmentation model does.
        self.channel_selection = ModelChannelSelection(
            required_channels=self.required_channels,
            available_channels=list(self.parent_window.signals),
            selected_channels=self.input_config.get("selected_channels"),
        )
        self.channel_cbs = self.channel_selection.channel_cbs
        self.n_channels = len(self.channel_cbs)
        self.layout.addWidget(self.channel_selection)

        # Button to apply the StarDist settings
        self.set_btn = QPushButton("set")
        self.set_btn.setStyleSheet(self.button_style_sheet)
        self.set_btn.clicked.connect(
            self.parent_window.set_selected_signals_for_event_detection
        )
        self.layout.addWidget(self.set_btn)
