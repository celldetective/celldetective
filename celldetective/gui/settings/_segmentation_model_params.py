import json
import logging
import os
from typing import Optional

logger = logging.getLogger("celldetective")

import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QMainWindow,
)
from fonticon_mdi6 import MDI6
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.model_channel_selection import ModelChannelSelection
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.utils.model_loaders import (
    locate_segmentation_model,
    trained_cell_size_um,
)


class SegModelParamsWidget(CelldetectiveWidget):

    def __init__(
        self,
        parent_window: Optional[QMainWindow] = None,
        model_name: Optional[str] = "SD_versatile_fluo",
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the SegModelParamsWidget.

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
        """Locate the model path."""
        self.model_complete_path = locate_segmentation_model(self.model_name)
        if self.model_complete_path is None:
            logger.error("Model could not be found. Abort.")
            self.abort_process()
        else:
            logger.info(f"Model path: {self.model_complete_path}...")

        if not os.path.exists(self.model_complete_path + "config_input.json"):
            logger.error(
                "The configuration for the inputs to the model could not be located. Abort."
            )
            self.abort_process()

        with open(self.model_complete_path + "config_input.json") as config_file:
            self.input_config = json.load(config_file)

    def abort_process(self):
        """Abort the widget initialization when the model cannot be loaded."""
        from PyQt5.QtWidgets import QMessageBox

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(
            f"Segmentation model could not be found or initialized.\n\n"
            f"Please verify that the model '{self.model_name}' exists in the 'models/segmentation/' folder."
        )
        msgBox.setWindowTitle("Error")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

        raise ValueError(f"Model {self.model_name} could not be located or loaded.")

    def populate_widgets(self):
        """Populate the widgets."""
        # One dropdown per input slot, seeded from the mapping already stored for
        # this model. Shared with the napari single-frame panel so that a model's
        # inputs are mapped the same way wherever it is run from.
        self.channel_selection = ModelChannelSelection(
            required_channels=self.required_channels,
            available_channels=list(self.attr_parent.exp_channels),
            selected_channels=self.input_config.get("selected_channels"),
        )
        self.channel_cbs = self.channel_selection.channel_cbs
        self.n_channels = len(self.channel_cbs)
        self.layout.addWidget(self.channel_selection)

        # Button to view the current stack with a scale bar
        self.view_diameter_btn = QPushButton()
        self.view_diameter_btn.setStyleSheet(self.button_select_all)
        self.view_diameter_btn.setIcon(icon(MDI6.image_check, color="black"))
        self.view_diameter_btn.setToolTip("View stack.")
        self.view_diameter_btn.setIconSize(QSize(20, 20))
        self.view_diameter_btn.clicked.connect(self.view_current_stack_with_scale_bar)

        # The size the model was trained on, which the cell size entered below is
        # rescaled against. Shared with the library and the napari panel, so a
        # generic Cellpose model -- which states that size in pixels rather than
        # in microns -- gets the row here too, and is rescaled the same way
        # wherever it is run from.
        trained = trained_cell_size_um(self.input_config)

        if trained is not None:

            # Only built when it is shown: `set_selected_channels_for_segmentation`
            # tests for this attribute to decide whether the user has a cell size
            # to save, and would otherwise write the placeholder 40 µm as though it
            # had been asked for -- rescaling every later run against a number
            # nobody entered.
            self.diameter_le = ThresholdLineEdit(
                init_value=40,
                connected_buttons=[self.view_diameter_btn],
                placeholder="cell diameter in µm",
                value_type="float",
            )

            # Layout for diameter input and button
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel("cell size [µm]: "), 33)
            hbox.addWidget(self.diameter_le, 61)
            hbox.addWidget(self.view_diameter_btn)
            self.layout.addLayout(hbox)

            # Reopen on the size last saved for this model, as the channel rows
            # do; only falling back on the trained size the first time round.
            stored = self.input_config.get("target_cell_size_um")
            self.diameter_le.set_threshold(stored if stored is not None else trained)

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
