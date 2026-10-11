import gc
from typing import Any, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QShortcut,
    QMainWindow,
)
from celldetective.gui.base.sliders import (
    QLabeledDoubleSlider,
    QLabeledDoubleRangeSlider,
)

from celldetective.gui.base.components import QHSeperationLine
from celldetective.gui.gui_utils import QuickSliderLayout, ThresholdLineEdit
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective.utils.contrast import contrast_limits, value_range
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    _get_img_num_per_channel,
)
from celldetective import get_logger

logger = get_logger(__name__)


class ChannelOffsetViewer(StackVisualizer):

    def __init__(
        self,
        *args: Any,
        parent_window: Optional[QMainWindow] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ChannelOffsetViewer.

        Parameters
        ----------
        parent_window : QMainWindow, optional
            The parent window.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.
        """

        self.parent_window = parent_window
        self.overlay_target_channel = -1
        self.shift_vertical = 0
        self.shift_horizontal = 0
        self.overlay_init_contrast = False
        super().__init__(*args, **kwargs)

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
        """Generate the overlay image show."""
        kwargs = self.imshow_kwargs.copy()
        if "cmap" in kwargs:
            kwargs.pop("cmap")

        self.im_overlay = self.ax.imshow(
            self.overlay_init_frame,
            cmap="Blues",
            interpolation="none",
            alpha=0.5,
            **kwargs,
        )

    def generate_overlay_alpha_slider(self):
        """Generate the transparency slider for the overlay."""
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
        """Generate the contrast slider for the overlay."""
        # Generate the contrast slider if enabled

        self.overlay_contrast_slider = QLabeledDoubleRangeSlider()
        overlay_range, (vmin, vmax) = contrast_limits(self.overlay_init_frame)
        contrast_layout = QuickSliderLayout(
            label="Overlay contrast: ",
            slider=self.overlay_contrast_slider,
            slider_initial_value=[vmin, vmax],
            slider_range=overlay_range,
            decimal_option=True,
            precision=5,
        )
        contrast_layout.setContentsMargins(15, 0, 15, 0)
        self.im_overlay.set_clim(vmin=vmin, vmax=vmax)
        self.overlay_contrast_slider.valueChanged.connect(self.change_contrast_overlay)
        self.canvas.layout.addLayout(contrast_layout)

    def set_overlay_channel_index(self, value: int) -> None:
        """
        Set the overlay channel index.

        Parameters
        ----------
        value : int
            The channel index.
        """
        # Set the channel index based on dropdown value

        self.overlay_target_channel = value
        self.overlay_init_contrast = True
        if self.mode == "direct":
            self.overlay_last_frame = self.stack[-1, :, :, self.overlay_target_channel]
        elif self.mode == "virtual":
            self.overlay_last_frame = self.read_frame(
                self.overlay_target_channel, self.stack_length - 1
            ).astype(float)
        self.change_overlay_frame(self.frame_slider.value())
        self.overlay_init_contrast = False

    def generate_overlay_channel_cb(self):
        """Generate the overlay channel combobox."""

        if self.channel_names is None:
            raise ValueError("channel_names must be set before generating the overlay channel combobox.")
        if len(self.channel_names) != self.n_channels:
            raise ValueError("channel_names length does not match n_channels.")

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
        """Generate the overlay shift controls."""

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

    def change_overlay_frame(self, value: int) -> None:
        """
        Change the displayed overlay frame.

        Parameters
        ----------
        value : int
             The frame index.
        """
        # Change the displayed frame based on slider value

        if self.mode == "virtual":
            self.overlay_init_frame = self.read_frame(
                self.overlay_target_channel, value
            ).astype(float)
        elif self.mode == "direct":
            self.overlay_init_frame = self.stack[
                value, :, :, self.overlay_target_channel
            ].copy()

        self.im_overlay.set_data(self.overlay_init_frame)

        if self.overlay_init_contrast and self.create_contrast_slider:
            (init_min, init_max), limits = contrast_limits(self.overlay_init_frame)
            last_min, last_max = value_range(self.overlay_last_frame)
            self.overlay_contrast_slider.setRange(
                min(init_min, last_min), max(init_max, last_max)
            )
            self.overlay_contrast_slider.setValue(limits)

        if self.create_contrast_slider:
            self.change_contrast_overlay(self.overlay_contrast_slider.value())

    def locate_image_virtual(self):
        """Locate virtual images."""
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

        first, last = self.current_time_index, self.stack_length - 1
        self.init_frame = self.read_frame(self.target_channel, first).astype(float)
        self.last_frame = self.read_frame(self.target_channel, last).astype(float)
        self.overlay_init_frame = self.read_frame(
            self.overlay_target_channel, first
        ).astype(float)
        self.overlay_last_frame = self.read_frame(
            self.overlay_target_channel, last
        ).astype(float)

    def change_contrast_overlay(self, value: Tuple[float, float]) -> None:
        """
        Change the contract of the overlay.

        Parameters
        ----------
        value : tuple
             The contrast limits (min, max).
        """
        # Change contrast based on slider value

        vmin = value[0]
        vmax = value[1]
        self.im_overlay.set_clim(vmin=vmin, vmax=vmax)
        self.fig.canvas.draw_idle()

    def change_alpha_overlay(self, value: float) -> None:
        """
        Change the transparency of the overlay.

        Parameters
        ----------
        value : float
            The alpha value.
        """
        # Change contrast based on slider value

        alpha = value
        self.im_overlay.set_alpha(alpha)
        self.fig.canvas.draw_idle()

    def define_keyboard_shortcuts(self):
        """Define keyboard shortcuts for shifting."""

        self.shift_up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self.canvas)
        self.shift_up_shortcut.activated.connect(self.shift_overlay_up)

        self.shift_down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self.canvas)
        self.shift_down_shortcut.activated.connect(self.shift_overlay_down)

        self.shift_left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self.canvas)
        self.shift_left_shortcut.activated.connect(self.shift_overlay_left)

        self.shift_right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self.canvas)
        self.shift_right_shortcut.activated.connect(self.shift_overlay_right)

    def shift_overlay_by(self, vertical: float = 0, horizontal: float = 0) -> None:
        """Move the overlay by a number of pixels, and apply the shift."""
        # Only the field moved is rewritten: the other may hold a value typed in.
        if vertical:
            self.shift_vertical += vertical
            self.vertical_shift_le.set_threshold(self.shift_vertical)
        if horizontal:
            self.shift_horizontal += horizontal
            self.horizontal_shift_le.set_threshold(self.shift_horizontal)
        self.apply_shift_btn.click()

    def shift_overlay_up(self):
        """Shift the overlay up."""
        self.shift_overlay_by(vertical=-2)

    def shift_overlay_down(self):
        """Shift the overlay down."""
        self.shift_overlay_by(vertical=2)

    def shift_overlay_left(self):
        """Shift the overlay left."""
        self.shift_overlay_by(horizontal=-2)

    def shift_overlay_right(self):
        """Shift the overlay right."""
        self.shift_overlay_by(horizontal=2)

    def shift_generic(self):
        """Apply the shift to the overlay."""
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
        """Generate the 'Set' button to apply shifts to parent."""

        self.set_shift_btn.clicked.connect(self.set_parent_attributes)
        self.set_shift_btn.setStyleSheet(self.button_style_sheet)
        self.add_centered_button(self.set_shift_btn, margins=(0, 5, 0, 5))

    def profile_sources(self):
        """The main channel in black and the overlay, as shifted, in blue."""
        sources = super().profile_sources()
        overlay = self.im_overlay.get_array()
        if overlay is not None:
            sources.append((overlay, "tab:blue"))
        return sources

    def set_parent_attributes(self):
        """Set the shift attributes in the parent window."""

        idx = self.channels_overlay_cb.currentIndex()
        self.parent_window.channels_cb.setCurrentIndex(idx)
        self.parent_window.vertical_shift_le.set_threshold(
            self.vertical_shift_le.get_threshold()
        )
        self.parent_window.horizontal_shift_le.set_threshold(
            self.horizontal_shift_le.get_threshold()
        )
        self.close()
