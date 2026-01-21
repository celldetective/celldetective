import os
from glob import glob
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QMessageBox, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton
from fonticon_mdi6 import MDI6
from natsort import natsorted
from superqt.fonticon import icon

from celldetective.gui.gui_utils import PreprocessingLayout2
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective.utils.image_loaders import load_frames
from celldetective import get_logger

logger = get_logger(__name__)

class SpotDetectionVisualizer(StackVisualizer):

    def __init__(
        self,
        parent_channel_cb=None,
        parent_diameter_le=None,
        parent_threshold_le=None,
        parent_preprocessing_list=None,
        cell_type="targets",
        labels=None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.cell_type = cell_type
        self.labels = labels
        self.detection_channel = self.target_channel
        self.switch_from_channel = False

        self.parent_channel_cb = parent_channel_cb
        self.parent_diameter_le = parent_diameter_le
        self.parent_threshold_le = parent_threshold_le
        self.parent_preprocessing_list = parent_preprocessing_list

        self.spot_sizes = []
        self.floatValidator = QDoubleValidator()
        self.init_scatter()

        self.generate_detection_channel()
        self.detection_channel = self.detection_channel_cb.currentIndex()

        self.generate_spot_detection_params()
        self.generate_add_measurement_btn()
        self.load_labels()
        self.change_frame(self.mid_time)

        self.ax.callbacks.connect("xlim_changed", self.update_marker_sizes)
        self.ax.callbacks.connect("ylim_changed", self.update_marker_sizes)

        self.apply_diam_btn.clicked.connect(self.detect_and_display_spots)
        self.apply_thresh_btn.clicked.connect(self.detect_and_display_spots)

        self.channel_cb.setCurrentIndex(self.target_channel)
        self.detection_channel_cb.setCurrentIndex(self.target_channel)

    def update_marker_sizes(self, event=None):

        # Get axis bounds
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Data-to-pixel scale
        ax_width_in_pixels = self.ax.bbox.width
        ax_height_in_pixels = self.ax.bbox.height

        x_scale = (float(xlim[1]) - float(xlim[0])) / ax_width_in_pixels
        y_scale = (float(ylim[1]) - float(ylim[0])) / ax_height_in_pixels

        # Choose the smaller scale for square pixels
        scale = min(x_scale, y_scale)

        # Convert radius_px to data units
        if len(self.spot_sizes) > 0:

            radius_data_units = self.spot_sizes / float(scale)

            # Convert to scatter `s` size (points squared)
            radius_pts = radius_data_units * (72.0 / self.fig.dpi)
            size = np.pi * (radius_pts**2)

            # Update scatter sizes
            self.spot_scat.set_sizes(size)
            self.fig.canvas.draw_idle()

    def init_scatter(self):
        self.spot_scat = self.ax.scatter(
            [], [], s=50, facecolors="none", edgecolors="tab:red", zorder=100
        )
        self.canvas.canvas.draw()

    def change_frame(self, value):

        super().change_frame(value)
        if not self.switch_from_channel:
            self.reset_detection()

        if self.mode == "virtual":
            from tifffile import imread

            self.init_label = imread(self.mask_paths[value])
            self.target_img = load_frames(
                self.img_num_per_channel[self.detection_channel, value],
                self.stack_path,
                normalize_input=False,
            )[:, :, 0]
        elif self.mode == "direct":
            self.init_label = self.labels[value, :, :]
            self.target_img = self.stack[value, :, :, self.detection_channel].copy()

    def detect_and_display_spots(self):

        self.reset_detection()
        self.control_valid_parameters()  # set current diam and threshold
        # self.change_frame(self.frame_slider.value())
        # self.set_detection_channel_index(self.detection_channel_cb.currentIndex())

        image_preprocessing = self.preprocessing.list.items
        if image_preprocessing == []:
            image_preprocessing = None

        from celldetective.measure import extract_blobs_in_image

        blobs_filtered = extract_blobs_in_image(
            self.target_img,
            self.init_label,
            threshold=self.thresh,
            diameter=self.diameter,
            image_preprocessing=image_preprocessing,
        )
        if blobs_filtered is not None:
            self.spot_positions = np.array([[x, y] for y, x, _ in blobs_filtered])
            if len(self.spot_positions) > 0:
                self.spot_sizes = np.sqrt(2) * np.array(
                    [sig for _, _, sig in blobs_filtered]
                )
            # radius_pts = self.spot_sizes * (self.fig.dpi / 72.0)
            # sizes = np.pi*(radius_pts**2)
            if len(self.spot_positions) > 0:
                self.spot_scat.set_offsets(self.spot_positions)
            else:
                empty_offset = np.ma.masked_array([0, 0], mask=True)
                self.spot_scat.set_offsets(empty_offset)
            # self.spot_scat.set_sizes(sizes)
            if len(self.spot_positions) > 0:
                self.update_marker_sizes()
            self.canvas.canvas.draw()

    def reset_detection(self):

        self.ax.scatter([], []).get_offsets()
        empty_offset = np.ma.masked_array([0, 0], mask=True)
        self.spot_scat.set_offsets(empty_offset)
        self.canvas.canvas.draw()

    def load_labels(self):

        # Load the cell labels
        if self.labels is not None:

            if isinstance(self.labels, list):
                self.labels = np.array(self.labels)

            assert (
                self.labels.ndim == 3
            ), "Wrong dimensions for the provided labels, expect TXY"
            assert len(self.labels) == self.stack_length

            self.mode = "direct"
            self.init_label = self.labels[self.mid_time, :, :]
        else:
            self.mode = "virtual"
            assert isinstance(self.stack_path, str)
            assert self.stack_path.endswith(".tif")
            self.locate_labels_virtual()

    def locate_labels_virtual(self):
        # Locate virtual labels

        labels_path = (
            str(Path(self.stack_path).parent.parent)
            + os.sep
            + f"labels_{self.cell_type}"
            + os.sep
        )
        self.mask_paths = natsorted(glob(labels_path + "*.tif"))

        if len(self.mask_paths) == 0:

            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("No labels were found for the selected cells. Abort.")
            msgBox.setWindowTitle("Critical")
            msgBox.setStandardButtons(QMessageBox.Ok)
            returnValue = msgBox.exec()
            self.close()

        from tifffile import imread

        self.init_label = imread(self.mask_paths[self.frame_slider.value()])

    def generate_detection_channel(self):

        assert self.channel_names is not None
        assert len(self.channel_names) == self.n_channels

        channel_layout = QHBoxLayout()
        channel_layout.setContentsMargins(15, 0, 15, 0)
        channel_layout.addWidget(QLabel("Detection\nchannel: "), 25)

        self.detection_channel_cb = QComboBox()
        self.detection_channel_cb.addItems(self.channel_names)
        self.detection_channel_cb.currentIndexChanged.connect(
            self.set_detection_channel_index
        )
        channel_layout.addWidget(self.detection_channel_cb, 75)

        # self.invert_check = QCheckBox('invert')
        # if self.invert:
        # 	self.invert_check.setChecked(True)
        # self.invert_check.toggled.connect(self.set_invert)
        # channel_layout.addWidget(self.invert_check, 10)

        self.canvas.layout.addLayout(channel_layout)

        self.preprocessing = PreprocessingLayout2(fraction=25, parent_window=self)
        self.preprocessing.setContentsMargins(15, 0, 15, 0)
        self.canvas.layout.addLayout(self.preprocessing)

    # def set_invert(self):
    # 	if self.invert_check.isChecked():
    # 		self.invert = True
    # 	else:
    # 		self.invert = False

    def set_detection_channel_index(self, value):

        self.detection_channel = value
        if self.mode == "direct":
            self.target_img = self.stack[-1, :, :, self.detection_channel]
        elif self.mode == "virtual":
            self.target_img = load_frames(
                self.img_num_per_channel[
                    self.detection_channel, self.frame_slider.value()
                ],
                self.stack_path,
                normalize_input=False,
            ).astype(float)[:, :, 0]

    def generate_spot_detection_params(self):

        self.spot_diam_le = QLineEdit("1")
        self.spot_diam_le.setValidator(self.floatValidator)
        self.apply_diam_btn = QPushButton("Set")
        self.apply_diam_btn.setStyleSheet(self.button_style_sheet_2)

        self.spot_thresh_le = QLineEdit("0")
        self.spot_thresh_le.setValidator(self.floatValidator)
        self.apply_thresh_btn = QPushButton("Set")
        self.apply_thresh_btn.setStyleSheet(self.button_style_sheet_2)

        self.spot_diam_le.textChanged.connect(self.control_valid_parameters)
        self.spot_thresh_le.textChanged.connect(self.control_valid_parameters)

        spot_diam_layout = QHBoxLayout()
        spot_diam_layout.setContentsMargins(15, 0, 15, 0)
        spot_diam_layout.addWidget(QLabel("Spot diameter: "), 25)
        spot_diam_layout.addWidget(self.spot_diam_le, 65)
        spot_diam_layout.addWidget(self.apply_diam_btn, 10)
        self.canvas.layout.addLayout(spot_diam_layout)

        spot_thresh_layout = QHBoxLayout()
        spot_thresh_layout.setContentsMargins(15, 0, 15, 0)
        spot_thresh_layout.addWidget(QLabel("Detection\nthreshold: "), 25)
        spot_thresh_layout.addWidget(self.spot_thresh_le, 65)
        spot_thresh_layout.addWidget(self.apply_thresh_btn, 10)
        self.canvas.layout.addLayout(spot_thresh_layout)

    def generate_add_measurement_btn(self):

        add_hbox = QHBoxLayout()
        self.add_measurement_btn = QPushButton("Add measurement")
        self.add_measurement_btn.clicked.connect(self.set_measurement_in_parent_list)
        self.add_measurement_btn.setIcon(icon(MDI6.plus, color="white"))
        self.add_measurement_btn.setIconSize(QSize(20, 20))
        self.add_measurement_btn.setStyleSheet(self.button_style_sheet)
        add_hbox.addWidget(QLabel(""), 33)
        add_hbox.addWidget(self.add_measurement_btn, 33)
        add_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(add_hbox)

    def control_valid_parameters(self):

        valid_diam = False
        try:
            self.diameter = float(self.spot_diam_le.text().replace(",", "."))
            valid_diam = True
        except:
            valid_diam = False

        valid_thresh = False
        try:
            self.thresh = float(self.spot_thresh_le.text().replace(",", "."))
            valid_thresh = True
        except:
            valid_thresh = False

        if valid_diam and valid_thresh:
            self.apply_diam_btn.setEnabled(True)
            self.apply_thresh_btn.setEnabled(True)
            self.add_measurement_btn.setEnabled(True)
        else:
            self.apply_diam_btn.setEnabled(False)
            self.apply_thresh_btn.setEnabled(False)
            self.add_measurement_btn.setEnabled(False)

    def set_measurement_in_parent_list(self):

        if self.parent_channel_cb is not None:
            self.parent_channel_cb.setCurrentIndex(self.detection_channel)
        if self.parent_diameter_le is not None:
            self.parent_diameter_le.setText(self.spot_diam_le.text())
        if self.parent_threshold_le is not None:
            self.parent_threshold_le.setText(self.spot_thresh_le.text())
        if self.parent_preprocessing_list is not None:
            self.parent_preprocessing_list.clear()
            items = self.preprocessing.list.getItems()
            for item in items:
                self.parent_preprocessing_list.addItemToList(item)
            self.parent_preprocessing_list.items = self.preprocessing.list.items
        self.close()
