from typing import List

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QComboBox, QHBoxLayout, QLabel
from fonticon_mdi6 import MDI6
from superqt import QLabeledDoubleSlider
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import ThresholdLineEdit
from celldetective.gui.viewers.size_viewer import CellSizeViewer


class CellposeParamsWidget(CelldetectiveWidget):
    """
    A widget to configure parameters for Cellpose segmentation, allowing users to set the cell diameter,
    select imaging channels, and adjust flow and cell probability thresholds for cell detection.

    This widget is designed for estimating cell diameters and configuring parameters for Cellpose,
    a deep learning-based segmentation tool. It also provides functionality to preview the image stack with a scale bar.

    Parameters
    ----------
    parent_window : QWidget, optional
            The parent window that hosts the widget (default is None).
    model_name : str, optional
            The name of the Cellpose model being used, typically 'CP_cyto2' for cytoplasm or 'CP_nuclei' for nuclei segmentation
            (default is 'CP_cyto2').

    Notes
    -----
    - This widget assumes that the parent window or one of its ancestor windows has access to the experiment channels
      and can locate the current image stack via `locate_image()`.
    - This class integrates sliders for flow and cell probability thresholds, as well as a channel selection for running
      Cellpose segmentation.
    - The `view_current_stack_with_scale_bar()` method opens a new window where the user can visually inspect the
      image stack with a superimposed scale bar, to better estimate the cell diameter.

    """

    view_diameter_btn: QPushButton = QPushButton()
    diameter_le: ThresholdLineEdit
    viewer: CellSizeViewer
    cellpose_channel_cb: List[QComboBox]
    cellpose_channel_template: List[str]
    flow_slider: QLabeledDoubleSlider = QLabeledDoubleSlider()
    set_cellpose_scale_btn: QPushButton = QPushButton("set")
    cellprob_slider: QLabeledDoubleSlider = QLabeledDoubleSlider()

    def __init__(self, parent_window=None, model_name="CP_cyto2", *args):

        super().__init__(*args)
        self.setWindowTitle("Estimate diameter")
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

        # Layout and widgets setup
        self.layout = QVBoxLayout()
        self.populate_widgets()
        self.setLayout(self.layout)
        center_window(self)

    def populate_widgets(self):
        """
        Populates the widget with UI elements such as buttons, sliders, and comboboxes to allow configuration
        of Cellpose segmentation parameters.
        """

        # Button to view the current stack with a scale bar
        self.view_diameter_btn.setStyleSheet(self.button_select_all)
        self.view_diameter_btn.setIcon(icon(MDI6.image_check, color="black"))
        self.view_diameter_btn.setToolTip("View stack.")
        self.view_diameter_btn.setIconSize(QSize(20, 20))
        self.view_diameter_btn.clicked.connect(self.view_current_stack_with_scale_bar)

        # Line edit for entering cell diameter
        self.diameter_le = ThresholdLineEdit(
            init_value=40,
            connected_buttons=[self.view_diameter_btn],
            placeholder="cell diameter in pixels",
            value_type="float",
        )

        # Comboboxes for selecting imaging channels
        self.cellpose_channel_cb = [QComboBox() for _ in range(2)]
        self.cellpose_channel_template = ["brightfield_channel", "live_nuclei_channel"]
        if self.model_name == "CP_nuclei":
            self.cellpose_channel_template = ["live_nuclei_channel", "None"]

        for k in range(2):
            hbox_channel = QHBoxLayout()
            hbox_channel.addWidget(QLabel(f"channel {k+1}: "))
            hbox_channel.addWidget(self.cellpose_channel_cb[k])
            if k == 1:
                self.cellpose_channel_cb[k].addItems(
                    list(self.attr_parent.exp_channels) + ["None"]
                )
            else:
                self.cellpose_channel_cb[k].addItems(
                    list(self.attr_parent.exp_channels)
                )
            idx = self.cellpose_channel_cb[k].findText(
                self.cellpose_channel_template[k]
            )
            if idx > 0:
                self.cellpose_channel_cb[k].setCurrentIndex(idx)
            else:
                self.cellpose_channel_cb[k].setCurrentIndex(0)

            if k == 1:
                idx = self.cellpose_channel_cb[k].findText("None")
                self.cellpose_channel_cb[k].setCurrentIndex(idx)

            self.layout.addLayout(hbox_channel)

        # Layout for diameter input and button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("diameter [px]: "), 33)
        hbox.addWidget(self.diameter_le, 61)
        hbox.addWidget(self.view_diameter_btn)
        self.layout.addLayout(hbox)

        # Flow threshold slider
        self.flow_slider.setOrientation(Qt.Horizontal)
        self.flow_slider.setRange(-6, 6)
        self.flow_slider.setValue(0.4)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("flow threshold: "), 33)
        hbox.addWidget(self.flow_slider, 66)
        self.layout.addLayout(hbox)

        # Cell probability threshold slider
        self.cellprob_slider.setOrientation(Qt.Horizontal)
        self.cellprob_slider.setRange(-6, 6)
        self.cellprob_slider.setValue(0.0)
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("cellprob threshold: "), 33)
        hbox.addWidget(self.cellprob_slider, 66)
        self.layout.addLayout(hbox)

        # Button to set the scale for Cellpose segmentation
        self.set_cellpose_scale_btn.setStyleSheet(self.button_style_sheet)
        self.set_cellpose_scale_btn.clicked.connect(
            self.parent_window.set_cellpose_scale
        )
        self.layout.addWidget(self.set_cellpose_scale_btn)

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
                diameter_slider_range=(0, max_size),
                frame_slider=True,
                contrast_slider=True,
                channel_cb=True,
                channel_names=self.attr_parent.exp_channels,
                n_channels=self.attr_parent.nbr_channels,
                PxToUm=1,
            )
            self.viewer.show()
