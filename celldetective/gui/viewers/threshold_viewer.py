from collections import OrderedDict

import numpy as np
from PyQt5.QtWidgets import QLineEdit, QHBoxLayout, QPushButton, QLabel
from superqt import QLabeledDoubleSlider

from celldetective.gui.gui_utils import QuickSliderLayout
from celldetective.gui.viewers.base_viewer import StackVisualizer
from celldetective import get_logger

logger = get_logger(__name__)


class ThresholdedStackVisualizer(StackVisualizer):
    """
    A widget for visualizing thresholded image stacks with interactive sliders and channel selection.

    Parameters:
    - preprocessing (list or None): A list of preprocessing filters to apply to the image before thresholding.
    - parent_le: The parent QLineEdit instance to set the threshold value.
    - initial_threshold (float): Initial threshold value.
    - initial_mask_alpha (float): Initial mask opacity value.
    - args, kwargs: Additional arguments to pass to the parent class constructor.

    Methods:
    - generate_apply_btn(): Generate the apply button to set the threshold in the parent QLineEdit.
    - set_threshold_in_parent_le(): Set the threshold value in the parent QLineEdit.
    - generate_mask_imshow(): Generate the mask imshow.
    - generate_threshold_slider(): Generate the threshold slider.
    - generate_opacity_slider(): Generate the opacity slider for the mask.
    - change_mask_opacity(value): Change the opacity of the mask.
    - change_threshold(value): Change the threshold value.
    - change_frame(value): Change the displayed frame and update the threshold.
    - compute_mask(threshold_value): Compute the mask based on the threshold value.
    - preprocess_image(): Preprocess the image before thresholding.

    Notes:
    - This class extends the functionality of StackVisualizer to visualize thresholded image stacks
      with interactive sliders for threshold and mask opacity adjustment.
    """

    def __init__(
        self,
        preprocessing=None,
        parent_le=None,
        initial_threshold=5,
        initial_mask_alpha=0.5,
        show_opacity_slider=True,
        show_threshold_slider=True,
        fill_holes=True,
        *args,
        **kwargs,
    ):
        # Initialize the widget and its attributes
        super().__init__(*args, **kwargs)
        self.preprocessing = preprocessing
        self.thresh = initial_threshold
        self.mask_alpha = initial_mask_alpha
        self.fill_holes = fill_holes
        self.parent_le = parent_le
        self.show_opacity_slider = show_opacity_slider
        self.show_threshold_slider = show_threshold_slider
        self.thresholded = False
        self.mask = np.zeros_like(self.init_frame)
        self.thresh_min = 0.0
        self.thresh_max = 30.0

        # Cache for processed images
        self.processed_cache = OrderedDict()
        self.processed_image = None
        self.max_processed_cache_size = 128

        self.generate_threshold_slider()

        # Ensure we start at frame 0 for consistent mask caching and UX
        if self.create_frame_slider and hasattr(self, "frame_slider"):
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.change_frame(0)
        elif self.stack_length > 0:
            self.change_frame(0)

        if self.thresh is not None:
            self.compute_mask(self.thresh)

        self.generate_mask_imshow()
        self.generate_scatter()
        self.generate_opacity_slider()
        if isinstance(self.parent_le, QLineEdit):
            self.generate_apply_btn()

    def generate_apply_btn(self):
        # Generate the apply button to set the threshold in the parent QLineEdit
        apply_hbox = QHBoxLayout()
        self.apply_threshold_btn = QPushButton("Apply")
        self.apply_threshold_btn.clicked.connect(self.set_threshold_in_parent_le)
        self.apply_threshold_btn.setStyleSheet(self.button_style_sheet)
        apply_hbox.addWidget(QLabel(""), 33)
        apply_hbox.addWidget(self.apply_threshold_btn, 33)
        apply_hbox.addWidget(QLabel(""), 33)
        self.canvas.layout.addLayout(apply_hbox)

    def closeEvent(self, event):
        if hasattr(self, "processed_cache") and isinstance(
            self.processed_cache, OrderedDict
        ):
            self.processed_cache.clear()
        super().closeEvent(event)

    def set_threshold_in_parent_le(self):
        # Set the threshold value in the parent QLineEdit
        self.parent_le.set_threshold(self.threshold_slider.value())
        self.close()

    def generate_mask_imshow(self):
        # Generate the mask imshow

        self.im_mask = self.ax.imshow(
            np.ma.masked_where(self.mask == 0, self.mask),
            alpha=self.mask_alpha,
            interpolation="none",
            vmin=0,
            vmax=1,
            cmap="Purples",
        )
        self.canvas.canvas.draw()

    def generate_scatter(self):
        self.scat_markers = self.ax.scatter([], [], color="tab:red")

    def generate_threshold_slider(self):
        # Generate the threshold slider
        self.threshold_slider = QLabeledDoubleSlider()
        if self.thresh is None:
            init_value = 1.0e5
        elif isinstance(self.thresh, (list, tuple, np.ndarray)):
            init_value = self.thresh[0]
        else:
            init_value = self.thresh
        thresh_layout = QuickSliderLayout(
            label="Threshold: ",
            slider=self.threshold_slider,
            slider_initial_value=init_value,
            slider_range=(self.thresh_min, np.amax([self.thresh_max, init_value])),
            decimal_option=True,
            precision=4,
            layout_ratio=(0.15, 0.85),
        )
        self.threshold_slider.valueChanged.connect(self.change_threshold)
        if self.show_threshold_slider:
            self.canvas.layout.addLayout(thresh_layout)

    def generate_opacity_slider(self):
        # Generate the opacity slider for the mask
        self.opacity_slider = QLabeledDoubleSlider()
        opacity_layout = QuickSliderLayout(
            label="Opacity: ",
            slider=self.opacity_slider,
            slider_initial_value=0.5,
            slider_range=(0, 1),
            decimal_option=True,
            precision=3,
            layout_ratio=(0.15, 0.85),
        )
        self.opacity_slider.valueChanged.connect(self.change_mask_opacity)
        if self.show_opacity_slider:
            self.canvas.layout.addLayout(opacity_layout)

    def change_mask_opacity(self, value):
        # Change the opacity of the mask
        self.mask_alpha = value
        self.im_mask.set_alpha(self.mask_alpha)
        self.canvas.canvas.draw_idle()

    def change_threshold(self, value):
        # Change the threshold value
        self.thresh = value

        # Sync slider if value came from external source (like Wizard)
        # to prevent slider from being "stale" and overwriting with old value later
        if hasattr(self, "threshold_slider"):
            display_val = value
            if isinstance(value, (list, tuple, np.ndarray)):
                display_val = value[0]

            try:
                current_val = self.threshold_slider.value()
                # Update slider if significant difference
                if abs(current_val - float(display_val)) > 1e-5:
                    self.threshold_slider.blockSignals(True)
                    self.threshold_slider.setValue(float(display_val))
                    self.threshold_slider.blockSignals(False)
            except Exception:
                pass

        if self.thresh is not None:
            self.compute_mask(self.thresh)
            mask = np.ma.masked_where(self.mask == 0, self.mask)
            if hasattr(self, "im_mask"):
                self.im_mask.set_data(mask)
            self.canvas.canvas.draw_idle()

    def change_frame(self, value):
        # Change the displayed frame and update the threshold
        if self.thresholded:
            self.init_contrast = True
        super().change_frame(value)
        self.processed_image = None

        if self.thresh is not None:
            self.change_threshold(self.thresh)
        else:
            self.change_threshold(self.threshold_slider.value())

        if self.thresholded:
            self.thresholded = False
            self.init_contrast = False

    def compute_mask(self, threshold_value):
        # Compute the mask based on the threshold value
        if self.processed_image is None:
            self.preprocess_image()

        from celldetective.utils.image_transforms import (
            estimate_unreliable_edge,
            threshold_image,
        )

        edge = estimate_unreliable_edge(self.preprocessing or [])

        if isinstance(threshold_value, (list, np.ndarray, tuple)):
            self.mask = threshold_image(
                self.processed_image,
                threshold_value[0],
                threshold_value[1],
                foreground_value=1,
                fill_holes=self.fill_holes,
                edge_exclusion=edge,
            ).astype(int)
        else:
            self.mask = threshold_image(
                self.processed_image,
                threshold_value,
                np.inf,
                foreground_value=1,
                fill_holes=self.fill_holes,
                edge_exclusion=edge,
            ).astype(int)

    def preprocess_image(self):
        # Preprocess the image before thresholding

        # Determine cache key
        target = self.target_channel
        time_idx = getattr(self, "current_time_index", 0)
        cache_key = (target, time_idx, str(self.preprocessing))

        # Check cache
        if self.preprocessing is not None:
            if cache_key in self.processed_cache:
                self.processed_image = self.processed_cache[cache_key]
                self.processed_cache.move_to_end(cache_key)
                # Ensure slider range is updated even on cache hit?
                # Probably redundant if image matches, but safe to skip or do lightweight check.
                return

        # Compute
        if self.preprocessing is not None:
            assert isinstance(self.preprocessing, list)
            from celldetective.filters import filter_image

            self.processed_image = filter_image(
                self.init_frame.copy().astype(float), filters=self.preprocessing
            )

            # Subsampled min/max for slider range
            if self.processed_image.size > 1000000:
                view = self.processed_image[::30, ::30]
            else:
                view = self.processed_image

            min_ = np.nanmin(view)
            max_ = np.nanmax(view)

            if min_ < self.thresh_min:
                self.thresh_min = min_
            if max_ > self.thresh_max:
                self.thresh_max = max_

            self.threshold_slider.setRange(self.thresh_min, self.thresh_max)

            # Store in cache
            self.processed_cache[cache_key] = self.processed_image
            if len(self.processed_cache) > self.max_processed_cache_size:
                self.processed_cache.popitem(last=False)

        else:
            # If no preprocessing, just use init_frame (casted)
            # We don't cache this as it's just a reference or light copy of init_frame
            self.processed_image = self.init_frame.astype(float)

    def set_preprocessing(self, activation_protocol):

        self.preprocessing = activation_protocol
        self.preprocess_image()

        self.im.set_data(self.processed_image)
        vmin = np.nanpercentile(self.processed_image, 1.0)
        vmax = np.nanpercentile(self.processed_image, 99.99)
        self.contrast_slider.setRange(
            np.nanmin(self.processed_image), np.nanmax(self.processed_image)
        )
        self.contrast_slider.setValue((vmin, vmax))
        self.im.set_clim(vmin, vmax)
        self.canvas.canvas.draw()
        self.thresholded = True
