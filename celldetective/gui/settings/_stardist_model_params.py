from PyQt5.QtWidgets import QVBoxLayout, QComboBox, QHBoxLayout, QLabel, QPushButton

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window


class StarDistParamsWidget(CelldetectiveWidget):
    """
    A widget to configure parameters for StarDist segmentation.

    This widget allows the user to select specific imaging channels for segmentation and adjust
    parameters for StarDist, a neural network-based image segmentation tool designed to segment
    star-convex shapes (typically nuclei).

    Parameters
    ----------
    parent_window : QWidget, optional
            The parent window hosting this widget (default is None).
    model_name : str, optional
            The name of the StarDist model being used, typically 'SD_versatile_fluo' for versatile
            fluorescence or 'SD_versatile_he' for H&E-stained images (default is 'SD_versatile_fluo').
    """

    def __init__(
        self, parent_window=None, model_name="SD_versatile_fluo", *args, **kwargs
    ):

        super().__init__(*args)
        self.setWindowTitle("Channels")
        self.parent_window = parent_window
        self.model_name = model_name

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

    def populate_widgets(self):
        """
        Populates the widget with channel selection comboboxes and a 'set' button to configure
        the StarDist segmentation settings. Handles different models by adjusting the number of
        available channels.
        """

        # Initialize comboboxes based on the selected model
        self.stardist_channel_cb = [QComboBox() for i in range(1)]
        self.stardist_channel_template = ["live_nuclei_channel"]
        max_i = 1

        # If the H&E model is selected, update the combobox configuration
        if self.model_name == "SD_versatile_he":
            self.stardist_channel_template = ["H&E_1", "H&E_2", "H&E_3"]
            self.stardist_channel_cb = [QComboBox() for i in range(3)]
            max_i = 3

        # Populate the comboboxes with available channels from the experiment
        for k in range(max_i):
            hbox_channel = QHBoxLayout()
            hbox_channel.addWidget(QLabel(f"channel {k+1}: "))
            hbox_channel.addWidget(self.stardist_channel_cb[k])
            if k == 1:
                self.stardist_channel_cb[k].addItems(
                    list(self.attr_parent.exp_channels) + ["None"]
                )
            else:
                self.stardist_channel_cb[k].addItems(
                    list(self.attr_parent.exp_channels)
                )

            # Set the default channel based on the template or fallback to the first option
            idx = self.stardist_channel_cb[k].findText(
                self.stardist_channel_template[k]
            )
            if idx > 0:
                self.stardist_channel_cb[k].setCurrentIndex(idx)
            else:
                self.stardist_channel_cb[k].setCurrentIndex(0)

            self.layout.addLayout(hbox_channel)

        # Button to apply the StarDist settings
        self.set_stardist_scale_btn = QPushButton("set")
        self.set_stardist_scale_btn.setStyleSheet(self.button_style_sheet)
        self.set_stardist_scale_btn.clicked.connect(
            self.parent_window.set_stardist_scale
        )
        self.layout.addWidget(self.set_stardist_scale_btn)
