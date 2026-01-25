import gc

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QComboBox, QPushButton, QShortcut
from superqt import QLabeledDoubleSlider, QLabeledDoubleRangeSlider

from celldetective.gui.base.components import QHSeperationLine
from celldetective.gui.gui_utils import QuickSliderLayout, ThresholdLineEdit
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective.utils.image_loaders import (
    load_frames,
    auto_load_number_of_frames,
    _get_img_num_per_channel,
)
from celldetective import get_logger

logger = get_logger(__name__)


class ChannelOffsetViewer(StackVisualizer):

    def __init__(self, parent_window=None, *args, **kwargs):

        self.parent_window = parent_window
        self.overlay_target_channel = -1
        self.shift_vertical = 0
        self.shift_horizontal = 0
        self.overlay_init_contrast = False
        super().__init__(*args, **kwargs)

        self.load_stack()

        if self.mode == "direct":
            # Initialize overlay frames for direct mode
            default_overlay_idx = -1
            if self.stack.ndim == 4:
                self.overlay_init_frame = self.stack[
                    self.current_time_index, :, :, default_overlay_idx
                ]
                self.overlay_last_frame = self.stack[-1, :, :, default_overlay_idx]
            else:
                # Should rely on 4D stack assumption from StackVisualizer
                self.overlay_init_frame = self.init_frame
                self.overlay_last_frame = self.last_frame

        self.canvas.layout.addWidget(QHSeperationLine())

        self.generate_overlay_channel_cb()
        self.generate_overlay_imshow()

        self.generate_overlay_alpha_slider()
        if self.create_contrast_slider:
            self.generate_overlay_contrast_slider()

        self.generate_overlay_shift()
        self.generate_add_to_parent_btn()

        if self.overlay_target_channel == -1:
            index = len(self.channel_names) - 1
        else:
            index = self.overlay_target_channel
        self.channels_overlay_cb.setCurrentIndex(index)
        self.frame_slider.valueChanged.connect(self.change_overlay_frame)

        self.define_keyboard_shortcuts()

        self.channels_overlay_cb.setCurrentIndex(
            self.parent_window.channels_cb.currentIndex()
        )
        self.set_channel_index(0)

        self.setAttribute(Qt.WA_DeleteOnClose)

    def generate_overlay_imshow(self):
        self.im_overlay = self.ax.imshow(
            self.overlay_init_frame,
            cmap="Blues",
            interpolation="none",
            alpha=0.5,
            **self.imshow_kwargs,
        )

    def generate_overlay_alpha_slider(self):
        # Generate the contrast slider if enabled

        self.overlay_alpha_slider = QLabeledDoubleSlider()
        alpha_layout = QuickSliderLayout(
            label="Overlay\ntransparency: ",
            slider=self.overlay_alpha_slider,
            slider_initial_value=0.5,
            slider_range=(0, 1.0),
            decimal_option=True,
            precision=5,
        )
        alpha_layout.setContentsMargins(15, 0, 15, 0)
        self.overlay_alpha_slider.valueChanged.connect(self.change_alpha_overlay)
        self.canvas.layout.addLayout(alpha_layout)

    def generate_overlay_contrast_slider(self):
        # Generate the contrast slider if enabled

        self.overlay_contrast_slider = QLabeledDoubleRangeSlider()
        contrast_layout = QuickSliderLayout(
            label="Overlay contrast: ",
            slider=self.overlay_contrast_slider,
            slider_initial_value=[
                np.nanpercentile(self.overlay_init_frame, 0.1),
                np.nanpercentile(self.overlay_init_frame, 99.99),
            ],
            slider_range=(
                np.nanmin(self.overlay_init_frame),
                np.nanmax(self.overlay_init_frame),
            ),
            decimal_option=True,
            precision=5,
        )
        contrast_layout.setContentsMargins(15, 0, 15, 0)
        self.im_overlay.set_clim(
            vmin=np.nanpercentile(self.overlay_init_frame, 0.1),
            vmax=np.nanpercentile(self.overlay_init_frame, 99.99),
        )
        self.overlay_contrast_slider.valueChanged.connect(self.change_contrast_overlay)
        self.canvas.layout.addLayout(contrast_layout)

    def set_overlay_channel_index(self, value):
        # Set the channel index based on dropdown value

        self.overlay_target_channel = value
        self.overlay_init_contrast = True
        if self.mode == "direct":
            self.overlay_last_frame = self.stack[-1, :, :, self.overlay_target_channel]
        elif self.mode == "virtual":
            self.overlay_last_frame = load_frames(
                self.img_num_per_channel[
                    self.overlay_target_channel, self.stack_length - 1
                ],
                self.stack_path,
                normalize_input=False,
            ).astype(float)[:, :, 0]
        self.change_overlay_frame(self.frame_slider.value())
        self.overlay_init_contrast = False

    def generate_overlay_channel_cb(self):

        assert self.channel_names is not None
        assert len(self.channel_names) == self.n_channels

        channel_layout = QHBoxLayout()
        channel_layout.setContentsMargins(15, 0, 15, 0)
        channel_layout.addWidget(QLabel("Overlay channel: "), 25)

        self.channels_overlay_cb = QComboBox()
        self.channels_overlay_cb.addItems(self.channel_names)
        self.channels_overlay_cb.currentIndexChanged.connect(
            self.set_overlay_channel_index
        )
        channel_layout.addWidget(self.channels_overlay_cb, 75)
        self.canvas.layout.addLayout(channel_layout)

    def generate_overlay_shift(self):

        shift_layout = QHBoxLayout()
        shift_layout.setContentsMargins(15, 0, 15, 0)
        shift_layout.addWidget(QLabel("shift (h): "), 20, alignment=Qt.AlignRight)

        self.apply_shift_btn = QPushButton("Apply")
        self.apply_shift_btn.setStyleSheet(self.button_style_sheet_2)
        self.apply_shift_btn.setToolTip("Apply the shift to the overlay channel.")
        self.apply_shift_btn.clicked.connect(self.shift_generic)

        self.set_shift_btn = QPushButton("Set")

        self.horizontal_shift_le = ThresholdLineEdit(
            init_value=self.shift_horizontal,
            connected_buttons=[self.apply_shift_btn, self.set_shift_btn],
            placeholder="horizontal shift [pixels]",
            value_type="float",
        )
        shift_layout.addWidget(self.horizontal_shift_le, 20)

        shift_layout.addWidget(QLabel("shift (v): "), 20, alignment=Qt.AlignRight)

        self.vertical_shift_le = ThresholdLineEdit(
            init_value=self.shift_vertical,
            connected_buttons=[self.apply_shift_btn, self.set_shift_btn],
            placeholder="vertical shift [pixels]",
            value_type="float",
        )
        shift_layout.addWidget(self.vertical_shift_le, 20)

        shift_layout.addWidget(self.apply_shift_btn, 20)

        self.canvas.layout.addLayout(shift_layout)

    def change_overlay_frame(self, value):
        # Change the displayed frame based on slider value

        if self.mode == "virtual":

            self.overlay_init_frame = load_frames(
                self.img_num_per_channel[self.overlay_target_channel, value],
                self.stack_path,
                normalize_input=False,
            ).astype(float)[:, :, 0]
        elif self.mode == "direct":
            self.overlay_init_frame = self.stack[
                value, :, :, self.overlay_target_channel
            ].copy()

        self.im_overlay.set_data(self.overlay_init_frame)

        if self.overlay_init_contrast and self.create_contrast_slider:
            self.im_overlay.autoscale()
            I_min, I_max = self.im_overlay.get_clim()
            self.overlay_contrast_slider.setRange(
                np.nanmin([self.overlay_init_frame, self.overlay_last_frame]),
                np.nanmax([self.overlay_init_frame, self.overlay_last_frame]),
            )
            self.overlay_contrast_slider.setValue((I_min, I_max))

        if self.create_contrast_slider:
            self.change_contrast_overlay(self.overlay_contrast_slider.value())

    def locate_image_virtual(self):
        from tifffile import imread

        # Locate the stack of images if provided as a file
        self.stack_length = auto_load_number_of_frames(self.stack_path)
        if self.stack_length is None:
            stack = imread(self.stack_path)
            self.stack_length = len(stack)
            del stack
            gc.collect()

        self.mid_time = self.stack_length // 2
        self.current_time_index = 0
        self.img_num_per_channel = _get_img_num_per_channel(
            np.arange(self.n_channels), self.stack_length, self.n_channels
        )

        self.init_frame = load_frames(
            self.img_num_per_channel[self.target_channel, self.current_time_index],
            self.stack_path,
            normalize_input=False,
        ).astype(float)[:, :, 0]
        self.last_frame = load_frames(
            self.img_num_per_channel[self.target_channel, self.stack_length - 1],
            self.stack_path,
            normalize_input=False,
        ).astype(float)[:, :, 0]
        self.overlay_init_frame = load_frames(
            self.img_num_per_channel[
                self.overlay_target_channel, self.current_time_index
            ],
            self.stack_path,
            normalize_input=False,
        ).astype(float)[:, :, 0]
        self.overlay_last_frame = load_frames(
            self.img_num_per_channel[
                self.overlay_target_channel, self.stack_length - 1
            ],
            self.stack_path,
            normalize_input=False,
        ).astype(float)[:, :, 0]

    def change_contrast_overlay(self, value):
        # Change contrast based on slider value

        vmin = value[0]
        vmax = value[1]
        self.im_overlay.set_clim(vmin=vmin, vmax=vmax)
        self.fig.canvas.draw_idle()

    def change_alpha_overlay(self, value):
        # Change contrast based on slider value

        alpha = value
        self.im_overlay.set_alpha(alpha)
        self.fig.canvas.draw_idle()

    def define_keyboard_shortcuts(self):

        self.shift_up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self.canvas)
        self.shift_up_shortcut.activated.connect(self.shift_overlay_up)

        self.shift_down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self.canvas)
        self.shift_down_shortcut.activated.connect(self.shift_overlay_down)

        self.shift_left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self.canvas)
        self.shift_left_shortcut.activated.connect(self.shift_overlay_left)

        self.shift_right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self.canvas)
        self.shift_right_shortcut.activated.connect(self.shift_overlay_right)

    def shift_overlay_up(self):
        self.shift_vertical -= 2
        self.vertical_shift_le.set_threshold(self.shift_vertical)
        # self.shift_generic()
        self.apply_shift_btn.click()

    def shift_overlay_down(self):
        self.shift_vertical += 2
        self.vertical_shift_le.set_threshold(self.shift_vertical)
        # self.shift_generic()
        self.apply_shift_btn.click()

    def shift_overlay_left(self):
        self.shift_horizontal -= 2
        self.horizontal_shift_le.set_threshold(self.shift_horizontal)
        # self.shift_generic()
        self.apply_shift_btn.click()

    def shift_overlay_right(self):
        self.shift_horizontal += 2
        self.horizontal_shift_le.set_threshold(self.shift_horizontal)
        # self.shift_generic()
        self.apply_shift_btn.click()

    def shift_generic(self):
        from scipy.ndimage import shift

        self.shift_vertical = self.vertical_shift_le.get_threshold()
        self.shift_horizontal = self.horizontal_shift_le.get_threshold()
        self.shifted_frame = shift(
            self.overlay_init_frame,
            [self.shift_vertical, self.shift_horizontal],
            prefilter=False,
        )
        self.im_overlay.set_data(self.shifted_frame)
        self.fig.canvas.draw_idle()
        self.update_profile()

    def generate_add_to_parent_btn(self):

        add_hbox = QHBoxLayout()
        add_hbox.setContentsMargins(0, 5, 0, 5)
        self.set_shift_btn.clicked.connect(self.set_parent_attributes)
        self.set_shift_btn.setStyleSheet(self.button_style_sheet)
        add_hbox.addWidget(QLabel(""), 33)
        add_hbox.addWidget(self.set_shift_btn, 33)
        add_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(add_hbox)

    def update_profile(self):
        if not self.line_mode or not hasattr(self, "line_x") or not self.line_x:
            return

        # Calculate profile
        x0, y0 = self.line_x[0], self.line_y[0]
        x1, y1 = self.line_x[1], self.line_y[1]
        length_px = np.hypot(x1 - x0, y1 - y0)
        if length_px == 0:
            return

        num_points = int(length_px)
        if num_points < 2:
            num_points = 2

        x, y = np.linspace(x0, x1, num_points), np.linspace(y0, y1, num_points)

        # Use self.init_frame and overlay frame
        profiles = []
        colors = ["black", "tab:blue"]

        # Main channel profile
        if hasattr(self, "init_frame") and self.init_frame is not None:
            from scipy.ndimage import map_coordinates

            profile = map_coordinates(
                self.init_frame, np.vstack((y, x)), order=1, mode="nearest"
            )
            profiles.append(profile)
        else:
            profiles.append(None)

        # Overlay channel profile
        # Use data currently in im_overlay, which accounts for shifts
        overlay_data = self.im_overlay.get_array()
        if overlay_data is not None:
            from scipy.ndimage import map_coordinates

            profile_overlay = map_coordinates(
                overlay_data, np.vstack((y, x)), order=1, mode="nearest"
            )
            profiles.append(profile_overlay)
        else:
            profiles.append(None)

        # Basic setup
        self.ax_profile.clear()
        self.ax_profile.set_facecolor("none")

        # Distance axis
        dist_axis = np.arange(num_points)
        title_str = f"{round(length_px,2)} [px]"
        if self.PxToUm is not None:
            title_str += f" | {round(length_px*self.PxToUm,3)} [µm]"

        # Handle Y-Axis Locking
        current_ylim = None
        if self.lock_y_action.isChecked():
            current_ylim = self.ax_profile.get_ylim()

        # Plot profiles
        for i, (profile, color) in enumerate(zip(profiles, colors)):
            if profile is not None:
                if np.all(np.isnan(profile)):
                    profile = np.zeros_like(profile)
                    profile[:] = np.nan

                self.ax_profile.plot(
                    dist_axis, profile, color=color, linestyle="-", label=f"Ch{i}"
                )

        self.ax_profile.set_xticks([])
        self.ax_profile.set_ylabel("Intensity", fontsize=8)
        self.ax_profile.set_xlabel(title_str, fontsize=8)
        self.ax_profile.tick_params(axis="y", which="major", labelsize=6)

        # Hide spines
        self.ax_profile.spines["top"].set_visible(False)
        self.ax_profile.spines["right"].set_visible(False)
        self.ax_profile.spines["bottom"].set_color("black")
        self.ax_profile.spines["left"].set_color("black")

        self.fig.set_facecolor("none")

        if current_ylim:
            self.ax_profile.set_ylim(current_ylim)

        self.fig.canvas.draw_idle()

    def set_parent_attributes(self):

        idx = self.channels_overlay_cb.currentIndex()
        self.parent_window.channels_cb.setCurrentIndex(idx)
        self.parent_window.vertical_shift_le.set_threshold(
            self.vertical_shift_le.get_threshold()
        )
        self.parent_window.horizontal_shift_le.set_threshold(
            self.horizontal_shift_le.get_threshold()
        )
        self.close()
