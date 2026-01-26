from collections import OrderedDict

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtWidgets import QHBoxLayout, QAction, QLabel, QComboBox
from fonticon_mdi6 import MDI6
from superqt import QLabeledDoubleRangeSlider, QLabeledSlider
from superqt.fonticon import icon

from celldetective.gui.base.components import CelldetectiveWidget
from celldetective.gui.base.utils import center_window
from celldetective.utils.image_loaders import (
    auto_load_number_of_frames,
    _get_img_num_per_channel,
    load_frames,
)
from celldetective import get_logger

logger = get_logger(__name__)


class StackLoader(QThread):
    frame_loaded = pyqtSignal(int, int, np.ndarray)  # channel, frame_idx, image

    def __init__(self, stack_path, img_num_per_channel, n_channels):
        super().__init__()
        self.stack_path = stack_path
        self.img_num_per_channel = img_num_per_channel
        self.n_channels = n_channels
        self.target_channel = 0
        self.priority_frame = 0
        self.cache_keys = set()
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def update_priority(self, channel, frame, current_cache_keys):
        self.mutex.lock()
        self.target_channel = channel
        self.priority_frame = frame
        self.cache_keys = set(current_cache_keys)
        self.condition.wakeAll()
        self.mutex.unlock()

    def stop(self):
        self.running = False
        self.condition.wakeAll()
        self.wait()

    def run(self):
        while self.running:
            self.mutex.lock()
            if not self.running:
                self.mutex.unlock()
                break

            t_ch = self.target_channel
            p_frame = self.priority_frame
            keys_snapshot = list(self.cache_keys)
            self.mutex.unlock()

            # Determine next frame to load
            # Strategy: look around priority frame
            frame_to_load = -1

            # Search radius
            radius = 10
            found = False

            # Check immediate neighbors first
            check_order = [p_frame]
            for r in range(1, radius + 1):
                check_order.append(p_frame + r)
                check_order.append(p_frame - r)

            # Determine max frames
            max_frames = self.img_num_per_channel.shape[1]

            for f in check_order:
                if 0 <= f < max_frames:
                    if (t_ch, f) not in keys_snapshot:
                        frame_to_load = f
                        found = True
                        break

            if found:
                try:
                    # Load the frame
                    from celldetective.utils.image_loaders import load_frames

                    img = load_frames(
                        self.img_num_per_channel[t_ch, frame_to_load],
                        self.stack_path,
                        normalize_input=False,
                    )[:, :, 0]

                    self.frame_loaded.emit(t_ch, frame_to_load, img)

                    # Update snapshot locally to avoid reloading immediately in next loop
                    self.mutex.lock()
                    self.cache_keys.add((t_ch, frame_to_load))
                    self.mutex.unlock()

                except Exception as e:
                    pass
                    # logger.error(f"Error loading frame {frame_to_load}: {e}")
                    # Prepare to wait to avoid spin loop on error
                    self.msleep(100)

            else:
                # If nothing to load, wait
                self.mutex.lock()
                self.condition.wait(self.mutex, 500)  # Wait 500ms or until new priority

                if not self.running:
                    self.mutex.unlock()
                    break

                self.mutex.unlock()


class StackVisualizer(CelldetectiveWidget):
    """
    A widget for visualizing image stacks with interactive sliders and channel selection.

    Parameters:
    - stack (numpy.ndarray or None): The stack of images.
    - stack_path (str or None): The path to the stack of images if provided as a file.
    - frame_slider (bool): Enable frame navigation slider.
    - contrast_slider (bool): Enable contrast adjustment slider.
    - channel_cb (bool): Enable channel selection dropdown.
    - channel_names (list or None): Names of the channels if `channel_cb` is True.
    - n_channels (int): Number of channels.
    - target_channel (int): Index of the target channel.
    - window_title (str): Title of the window.
    - PxToUm (float or None): Pixel to micrometer conversion factor.
    - background_color (str): Background color of the widget.
    - imshow_kwargs (dict): Additional keyword arguments for imshow function.

    Methods:
    - show(): Display the widget.
    - load_stack(): Load the stack of images.
    - locate_image_virtual(): Locate the stack of images if provided as a file.
    - generate_figure_canvas(): Generate the figure canvas for displaying images.
    - generate_channel_cb(): Generate the channel dropdown if enabled.
    - generate_contrast_slider(): Generate the contrast slider if enabled.
    - generate_frame_slider(): Generate the frame slider if enabled.
    - set_target_channel(value): Set the target channel.
    - change_contrast(value): Change contrast based on slider value.
    - set_channel_index(value): Set the channel index based on dropdown value.
    - change_frame(value): Change the displayed frame based on slider value.
    - closeEvent(event): Event handler for closing the widget.

    Notes:
    - This class provides a convenient interface for visualizing image stacks with frame navigation,
      contrast adjustment, and channel selection functionalities.
    """

    def __init__(
        self,
        stack=None,
        stack_path=None,
        frame_slider=True,
        contrast_slider=True,
        channel_cb=False,
        channel_names=None,
        n_channels=1,
        target_channel=0,
        window_title="View",
        PxToUm=None,
        background_color="transparent",
        imshow_kwargs={},
    ):
        super().__init__()

        # self.setWindowTitle(window_title)
        self.window_title = window_title

        # LRU Cache for virtual mode
        self.frame_cache = OrderedDict()
        self.max_cache_size = 128
        self.current_time_index = 0

        self.stack = stack
        self.stack_path = stack_path
        self.create_frame_slider = frame_slider
        self.background_color = background_color
        self.create_contrast_slider = contrast_slider
        self.create_channel_cb = channel_cb
        self.n_channels = n_channels
        self.channel_names = channel_names
        self.target_channel = target_channel
        self.imshow_kwargs = imshow_kwargs
        self.PxToUm = PxToUm
        self.init_contrast = False
        self.channel_trigger = False
        self.roi_mode = False
        self.line_mode = False
        self.line_artist = None
        self.ax_profile = None
        self._min = 0
        self._max = 0

        self.loader_thread = None
        self.load_stack()
        self.generate_figure_canvas()
        if self.create_channel_cb:
            self.generate_channel_cb()
        if self.create_contrast_slider:
            self.generate_contrast_slider()
        if self.create_frame_slider:
            self.generate_frame_slider()

        self.line_color = "orange"
        self.line_artist = None
        self.ax_profile = None
        self.line_text = None
        self.background = None
        self.is_drawing_line = False
        self.generate_custom_tools()

        self.canvas.layout.setContentsMargins(15, 15, 15, 15)

        center_window(self)

    def generate_custom_tools(self):

        tools_layout = QHBoxLayout()
        tools_layout.setContentsMargins(15, 0, 15, 0)

        actions = self.canvas.toolbar.actions()

        # Create the action
        self.line_action = QAction(
            icon(MDI6.chart_line, color="black"), "Line Profile", self.canvas.toolbar
        )
        self.line_action.setCheckable(True)
        self.line_action.setToolTip("Draw a line to plot intensity profile.")
        self.line_action.triggered.connect(self.toggle_line_mode)

        # Lock Y-Axis Action
        self.lock_y_action = QAction(
            icon(MDI6.lock, color="black"), "Lock Y-Axis", self.canvas.toolbar
        )
        self.lock_y_action.setCheckable(True)
        self.lock_y_action.setToolTip(
            "Lock the Y-axis min/max values for the profile plot."
        )
        self.lock_y_action.setEnabled(False)  # Enable only when line mode is active

        target_action = None
        for action in actions:
            if "Zoom" in action.text() or "Pan" in action.text():
                target_action = action

        if target_action:
            insert_before = None
            for action in actions:
                if "Subplots" in action.text() or "Configure" in action.text():
                    insert_before = action
                    break

            if insert_before:
                self.canvas.toolbar.insertAction(insert_before, self.line_action)
                self.canvas.toolbar.insertAction(insert_before, self.lock_y_action)
            else:
                if len(actions) > 5:
                    self.canvas.toolbar.insertAction(actions[5], self.line_action)
                    self.canvas.toolbar.insertAction(actions[5], self.line_action)
                    self.canvas.toolbar.insertAction(actions[5], self.lock_y_action)
                else:
                    self.canvas.toolbar.addAction(self.line_action)
                    self.canvas.toolbar.addAction(self.lock_y_action)

        self.info_lbl = QLabel("")
        tools_layout.addWidget(self.info_lbl)

        self.canvas.layout.addLayout(tools_layout)

    def toggle_line_mode(self):

        if self.line_action.isChecked():

            self.line_mode = True
            self.lock_y_action.setEnabled(True)
            self.canvas.toolbar.mode = ""

            # Enable manual layout control to prevent tight_layout interference
            if hasattr(self.canvas, "manual_layout"):
                self.canvas.manual_layout = True

            # Connect events
            self.cid_press = self.fig.canvas.mpl_connect(
                "button_press_event", self.on_line_press
            )
            self.cid_move = self.fig.canvas.mpl_connect(
                "motion_notify_event", self.on_line_drag
            )
            self.cid_release = self.fig.canvas.mpl_connect(
                "button_release_event", self.on_line_release
            )

            # Save original position if not saved
            # if not hasattr(self, "ax_original_pos"):
            #     self.ax_original_pos = self.ax.get_position()

            # Disable tight_layout/layout engine to prevent fighting manual positioning
            if hasattr(self.fig, "set_layout_engine"):
                self.fig.set_layout_engine("none")
            else:
                self.fig.set_tight_layout(False)

            # Use GridSpec for robust layout
            # 2 rows: Main Image (top, ~75%), Profile (bottom, ~25%)
            # Add margins to ensure axis labels and text are visible
            import matplotlib.gridspec as gridspec

            gs = gridspec.GridSpec(
                2,
                1,
                height_ratios=[3, 1],
                hspace=0.05,
                left=0.1,
                right=0.9,
                bottom=0.05,
                top=1,
            )

            # Move main axes to top slot
            self.ax.set_subplotspec(gs[0])
            self.ax.set_position(gs[0].get_position(self.fig))

            # create profile axes as a subplot in the bottom slot
            if self.ax_profile is None:
                self.ax_profile = self.fig.add_subplot(gs[1])
            else:
                self.ax_profile.set_subplotspec(gs[1])
                self.ax_profile.set_position(gs[1].get_position(self.fig))

            self.ax_profile.set_visible(True)
            self.ax_profile.set_label("profile_axes")
            self.ax_profile.set_facecolor("none")
            self.ax_profile.tick_params(axis="y", which="major", labelsize=8)
            self.ax_profile.set_xticks([])
            self.ax_profile.set_xlabel("")
            self.ax_profile.set_ylabel("Intensity", fontsize=8)

            # Hide spines initially
            self.ax_profile.spines["top"].set_visible(False)
            self.ax_profile.spines["right"].set_visible(False)
            self.ax_profile.spines["bottom"].set_color("black")
            self.ax_profile.spines["left"].set_color("black")

            # Update Toolbar Home State to match new layout BUT with full field of view
            # 1. Save current zoom
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()

            # 2. Set limits to full extent (Home State)
            if hasattr(self, "im"):
                extent = self.im.get_extent()  # (left, right, bottom, top) or similar
                self.ax.set_xlim(extent[0], extent[1])
                self.ax.set_ylim(extent[2], extent[3])

            # 3. Reset Stack and save Home
            self.canvas.toolbar._nav_stack.clear()
            self.canvas.toolbar.push_current()

            # 4. Restore User Zoom
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

            # 5. Push restored zoom state so "Back"/"Forward" logic works from here?
            # Actually, if we just restore, we are "live" at a new state.
            # If we don't push, "Home" works. "Back" might not exist yet. That's fine.
            self.canvas.toolbar.push_current()

            self.canvas.draw()
        else:
            self.line_mode = False
            self.lock_y_action.setChecked(False)
            self.lock_y_action.setEnabled(False)

            # Disable manual layout control
            if hasattr(self.canvas, "manual_layout"):
                self.canvas.manual_layout = False

            # Disconnect events
            if hasattr(self, "cid_press"):
                self.fig.canvas.mpl_disconnect(self.cid_press)
                self.fig.canvas.mpl_disconnect(self.cid_move)
                self.fig.canvas.mpl_disconnect(self.cid_release)

            # Remove line artist
            if self.line_artist:
                self.line_artist.remove()
                self.line_artist = None

            if hasattr(self, "line_text") and self.line_text:
                self.line_text.remove()
                self.line_text = None

            # Remove profile axes and restore space
            if self.ax_profile is not None:
                self.ax_profile.remove()
                self.ax_profile = None

            # Restore original layout
            # if hasattr(self, "ax_original_pos"):
            # standard 1x1 GridSpec or manual restore
            import matplotlib.gridspec as gridspec

            gs = gridspec.GridSpec(1, 1)
            self.ax.set_subplotspec(gs[0])
            # self.ax.set_position(gs[0].get_position(self.fig))
            self.fig.subplots_adjust(
                top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
            )
            # self.ax.set_position(self.ax_original_pos) # tight layout should fix it

            # Re-enable tight_layout via standard resize event later or explicit call
            self.fig.tight_layout()

            # Reset Toolbar Stack for Standard View
            self.canvas.toolbar._nav_stack.clear()
            self.canvas.toolbar.push_current()

            self.canvas.draw()
            self.info_lbl.setText("")

    def on_line_press(self, event):
        if event.inaxes != self.ax:
            return
        if self.canvas.toolbar.mode:
            return

        self.line_x = [event.xdata]
        self.line_y = [event.ydata]
        self.is_drawing_line = True

        # Initialize line artist if needed
        if self.line_artist is None:
            (self.line_artist,) = self.ax.plot(
                self.line_x,
                self.line_y,
                color=self.line_color,
                linestyle="-",
                linewidth=3,
            )
        else:
            self.line_artist.set_data(self.line_x, self.line_y)
            self.line_artist.set_visible(True)

        # Blitting setup
        self.line_artist.set_animated(True)
        self.canvas.draw()
        self.background = self.canvas.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line_artist)
        self.canvas.canvas.blit(self.ax.bbox)

    def on_line_drag(self, event):
        if not getattr(self, "is_drawing_line", False) or event.inaxes != self.ax:
            return

        self.line_x = [self.line_x[0], event.xdata]
        self.line_y = [self.line_y[0], event.ydata]

        # Blitting update
        if self.background:
            self.canvas.canvas.restore_region(self.background)

        self.line_artist.set_data(self.line_x, self.line_y)
        self.ax.draw_artist(self.line_artist)
        self.canvas.canvas.blit(self.ax.bbox)

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

        # Use self.init_frame as self.im.get_array() might be unreliable or cached
        if hasattr(self, "init_frame") and self.init_frame is not None:
            from scipy.ndimage import map_coordinates

            profile = map_coordinates(
                self.init_frame, np.vstack((y, x)), order=1, mode="nearest"
            )
        else:
            return

        if np.all(np.isnan(profile)):
            # If profile is all NaNs, we can't plot meaningful data, but we should maintain the axes
            profile = np.zeros_like(profile)
            profile[:] = np.nan

        # Distance in microns if available
        dist_axis = np.arange(num_points)
        x_label = "Distance (px)"

        # Only show pixel length, rounded to integer
        title_str = f"{round(length_px,2)} [px]"
        if self.PxToUm is not None:
            title_str += f" | {round(length_px*self.PxToUm,3)} [µm]"

        # Handle Y-Axis Locking
        current_ylim = None
        if self.lock_y_action.isChecked():
            current_ylim = self.ax_profile.get_ylim()

        # Plot profile
        self.ax_profile.clear()
        self.ax_profile.set_facecolor("none")
        if hasattr(self, "profile_line") and self.profile_line:
            try:
                self.profile_line.remove()
            except:
                pass

        # Distance in microns if available
        dist_axis = np.arange(num_points)

        (self.profile_line,) = self.ax_profile.plot(
            dist_axis, profile, color="black", linestyle="-"
        )
        self.ax_profile.set_xticks([])
        self.ax_profile.set_ylabel("Intensity", fontsize=8)
        self.ax_profile.set_xlabel(title_str, fontsize=8)
        self.ax_profile.tick_params(axis="y", which="major", labelsize=6)
        # self.ax_profile.grid(True)

        # Hide spines
        self.ax_profile.spines["top"].set_visible(False)
        self.ax_profile.spines["right"].set_visible(False)
        self.ax_profile.spines["bottom"].set_color("black")
        self.ax_profile.spines["left"].set_color("black")

        self.fig.set_facecolor("none")

        if current_ylim:
            self.ax_profile.set_ylim(current_ylim)

        self.fig.canvas.draw_idle()

    def on_line_release(self, event):
        if not getattr(self, "is_drawing_line", False):
            return
        self.is_drawing_line = False

        if event.inaxes != self.ax:
            return

        # Final update
        self.line_x = [self.line_x[0], event.xdata]
        self.line_y = [self.line_y[0], event.ydata]
        self.line_artist.set_data(self.line_x, self.line_y)

        # Finalize drawing (disable animation for persistence)
        self.line_artist.set_animated(False)
        self.background = None

        self.update_profile()
        self.canvas.canvas.draw_idle()

    def show(self):
        # Display the widget
        self.canvas.show()

    def load_stack(self):
        # Load the stack of images
        if self.stack is not None:
            if isinstance(self.stack, list):
                self.stack = np.asarray(self.stack)

            if self.stack.ndim == 3:
                # Assuming stack is (T, Y, X) -> convert to (T, Y, X, 1) for consistent 4D handling
                logger.info(
                    "StackVisualizer: 3D stack detected (T, Y, X). Adding channel axis."
                )
                self.stack = self.stack[:, :, :, np.newaxis]
                self.target_channel = 0

            self.mode = "direct"
            self.stack_length = len(self.stack)
            self.mid_time = self.stack_length // 2
            self.current_time_index = 0
            self.init_frame = self.stack[
                self.current_time_index, :, :, self.target_channel
            ]
            self.last_frame = self.stack[-1, :, :, self.target_channel]
        else:
            self.mode = "virtual"
            assert isinstance(self.stack_path, str)
            assert self.stack_path.endswith(".tif")
            self.locate_image_virtual()

    def locate_image_virtual(self):
        # Locate the stack of images if provided as a file

        self.stack_length = auto_load_number_of_frames(self.stack_path)
        self.mid_time = self.stack_length // 2
        self.current_time_index = 0
        self.img_num_per_channel = _get_img_num_per_channel(
            np.arange(self.n_channels), self.stack_length, self.n_channels
        )

        # Initialize background loader
        self.loader_thread = StackLoader(
            self.stack_path, self.img_num_per_channel, self.n_channels
        )
        self.loader_thread.frame_loaded.connect(self.on_frame_loaded)
        self.loader_thread.start()

        self.init_frame = load_frames(
            self.img_num_per_channel[self.target_channel, self.current_time_index],
            self.stack_path,
            normalize_input=False,
        )[:, :, 0]
        self.last_frame = load_frames(
            self.img_num_per_channel[self.target_channel, self.stack_length - 1],
            self.stack_path,
            normalize_input=False,
        )[:, :, 0]

    def generate_figure_canvas(self):

        if np.all(np.isnan(self.init_frame)):
            p01, p99 = 0, 1
        else:
            p01 = np.nanpercentile(self.init_frame, 0.1)
            p99 = np.nanpercentile(self.init_frame, 99.9)

        if np.isnan(p01):
            p01 = 0
        if np.isnan(p99):
            p99 = 1

        import matplotlib.pyplot as plt
        from celldetective.gui.base.figure_canvas import FigureCanvas

        self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.ax.margins(0)
        self.fig.patch.set_alpha(0)
        self.canvas = FigureCanvas(self.fig, title=self.window_title, interactive=True)
        self.ax.clear()
        self.im = self.ax.imshow(
            self.init_frame,
            cmap="gray",
            interpolation="none",
            zorder=0,
            vmin=p01,
            vmax=p99,
            **self.imshow_kwargs,
        )
        if self.PxToUm is not None:
            from matplotlib_scalebar.scalebar import ScaleBar

            scalebar = ScaleBar(
                self.PxToUm,
                "um",
                length_fraction=0.25,
                location="upper right",
                border_pad=0.4,
                box_alpha=0.95,
                color="white",
                box_color="black",
                font_properties={"weight": "bold", "size": 10},
            )
            self.ax.add_artist(scalebar)
        self.ax.axis("off")

    def generate_channel_cb(self):

        self.channel_cb = QComboBox()
        if self.channel_names is not None and len(self.channel_names) > 0:
            for name in self.channel_names:
                self.channel_cb.addItem(name)
        else:
            for i in range(self.n_channels):
                self.channel_cb.addItem(f"Channel {i}")
        self.channel_cb.currentIndexChanged.connect(self.set_channel_index)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Channel: "), 15)
        layout.addWidget(self.channel_cb, 85)
        self.canvas.layout.addLayout(layout)

    def set_contrast_decimals(self):
        from celldetective.utils.types import is_integer_array

        if is_integer_array(self.init_frame):
            self.contrast_decimals = 0
        else:
            self.contrast_decimals = 2

    def generate_contrast_slider(self):
        # Generate the contrast slider if enabled

        layout = QHBoxLayout()
        self.set_contrast_decimals()
        self.contrast_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
        if np.all(np.isnan(self.init_frame)):
            min_val, max_val = 0, 1
        else:
            min_val = np.nanmin(self.init_frame)
            max_val = np.nanmax(self.init_frame)

        if np.isnan(min_val):
            min_val = 0
        if np.isnan(max_val):
            max_val = 1

        self.contrast_slider.setRange(min_val, max_val)

        # Set initial value to percentiles to avoid outliers
        if np.all(np.isnan(self.init_frame)):
            p01, p99 = 0, 1
        else:
            p01 = np.nanpercentile(self.init_frame, 0.1)
            p99 = np.nanpercentile(self.init_frame, 99.9)

        if np.isnan(p01):
            p01 = min_val
        if np.isnan(p99):
            p99 = max_val

        if p99 > p01:
            self.contrast_slider.setValue((p01, p99))
        else:
            self.contrast_slider.setValue((min_val, max_val))

        self.contrast_slider.setEdgeLabelMode(
            QLabeledDoubleRangeSlider.EdgeLabelMode.NoLabel
        )
        self.contrast_slider.setDecimals(self.contrast_decimals)

        self.contrast_slider.valueChanged.connect(self.change_contrast)
        layout.addWidget(QLabel("Contrast: "), 15)
        layout.addWidget(self.contrast_slider, 85)
        self.canvas.layout.addLayout(layout)

    def generate_frame_slider(self):
        # Generate the frame slider if enabled

        layout = QHBoxLayout()
        self.frame_slider = QLabeledSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, self.stack_length - 1)
        self.frame_slider.setValue(self.current_time_index)
        self.frame_slider.valueChanged.connect(self.change_frame)
        layout.addWidget(QLabel("Time: "), 15)
        layout.addWidget(self.frame_slider, 85)
        self.canvas.layout.addLayout(layout)

    def set_target_channel(self, value):
        self.target_channel = value
        self.init_frame = self.stack[self.current_time_index, :, :, self.target_channel]
        self.im.set_data(self.init_frame)
        self.canvas.draw()
        self.update_profile()

    def change_contrast(self, value):
        # Change contrast based on slider value
        if not self.init_contrast:
            self.im.set_clim(vmin=value[0], vmax=value[1])
            self.canvas.draw()

    def set_channel_index(self, value):
        self.target_channel = value
        self.channel_trigger = True
        if self.create_frame_slider:
            self.change_frame_from_channel_switch(self.frame_slider.value())
        else:
            if self.stack is not None and self.stack.ndim == 4:
                self.init_frame = self.stack[
                    self.current_time_index, :, :, self.target_channel
                ]
                self.im.set_data(self.init_frame)
                self.canvas.draw()
                self.update_profile()

    def change_frame_from_channel_switch(self, value):
        self._min = 0
        self._max = 0
        self.change_frame(value)
        if self.channel_trigger:
            p01 = np.nanpercentile(self.init_frame, 0.1)
            p99 = np.nanpercentile(self.init_frame, 99.9)
            self.im.set_clim(vmin=p01, vmax=p99)
            if self.create_contrast_slider and hasattr(self, "contrast_slider"):
                self.contrast_slider.setValue((p01, p99))
            self.channel_trigger = False
            self.canvas.draw()

    def change_frame(self, value):

        self.current_time_index = value

        # Update loader priority
        if self.mode == "virtual" and self.loader_thread:
            self.loader_thread.update_priority(
                self.target_channel, value, self.frame_cache.keys()
            )

        if self.mode == "direct":
            self.init_frame = self.stack[value, :, :, self.target_channel]

        elif self.mode == "virtual":
            # Check cache first
            cache_key = (self.target_channel, value)
            if cache_key in self.frame_cache:
                self.init_frame = self.frame_cache[cache_key]
                self.frame_cache.move_to_end(cache_key)  # Mark as recently used
            else:
                self.init_frame = load_frames(
                    self.img_num_per_channel[self.target_channel, value],
                    self.stack_path,
                    normalize_input=False,
                )[:, :, 0]

                # Add to cache
                self.frame_cache[cache_key] = self.init_frame
                # Enforce size limit
                if len(self.frame_cache) > self.max_cache_size:
                    self.frame_cache.popitem(last=False)  # Remove oldest

        self.im.set_data(self.init_frame)
        rescale_contrast = False

        # Optimization: Check min/max on subsampled array for large images
        if self.init_frame.size > 1000000:
            view = self.init_frame[::30, ::30]
        else:
            view = self.init_frame

        curr_min = np.nanmin(view)
        curr_max = np.nanmax(view)

        if curr_min < self._min:
            self._min = curr_min
            rescale_constrast = True
        if curr_max > self._max:
            self._max = curr_max
            rescale_contrast = True

        if rescale_contrast:
            self.contrast_slider.setRange(self._min, self._max)
        self.canvas.canvas.draw_idle()
        self.update_profile()

    def on_frame_loaded(self, channel, frame, image):
        """Callback from loader thread"""
        # Store in cache
        cache_key = (channel, frame)
        if cache_key not in self.frame_cache:
            self.frame_cache[cache_key] = image
            if len(self.frame_cache) > self.max_cache_size:
                self.frame_cache.popitem(last=False)

        # If this is the current frame (user might have scrolled while loading), update display?
        # Usually change_frame handles display. If we are waiting for this frame, we might want to refresh.
        if channel == self.target_channel and frame == self.current_time_index:
            # Refresh
            self.change_frame(self.current_time_index)

    def closeEvent(self, event):
        # Event handler for closing the widget
        if self.loader_thread:
            self.loader_thread.stop()
            self.loader_thread = None
        if hasattr(self, "frame_cache") and isinstance(self.frame_cache, OrderedDict):
            self.frame_cache.clear()
        self.canvas.close()

    def __del__(self):
        try:
            if hasattr(self, "loader_thread") and self.loader_thread:
                self.loader_thread.stop()
        except:
            pass
