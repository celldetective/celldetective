from typing import Any, Optional

import numpy as np
from celldetective.gui.base.sliders import QLabeledDoubleSlider

from celldetective.gui.gui_utils import QuickSliderLayout
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.utils.registration import disk_taper, radial_distance


class RegistrationROIViewer(CellSizeViewer):
    """
    Tune the correlation disk of the stack registration on a real frame.

    A :class:`CellSizeViewer` whose slider sets a radius, with the circle fixed at the image
    centre as in :func:`celldetective.utils.registration.tukey_window`. It adds a dashed circle
    at the start of the Tukey taper, a shading of the pixels by how little they weigh in the
    correlation (fully shaded pixels are ignored) and a Tukey α slider.

    Parameters
    ----------
    parent_layout : RegistrationOptionsLayout
        Options layout whose radius and Tukey α fields receive the values on "Set".
    initial_radius : float, optional
        Radius in pixels shown at opening. If None, 90 % of the half smallest image side.
    tukey_alpha : float, optional
        Fraction of the disk covered by the taper (default 0.25).
    *args, **kwargs
        Passed to :class:`StackVisualizer`.
    """

    def __init__(
        self,
        parent_layout: Any,
        *args: Any,
        initial_radius: Optional[float] = None,
        tukey_alpha: float = 0.25,
        **kwargs: Any,
    ) -> None:

        self.parent_layout = parent_layout
        self.initial_radius = initial_radius
        self.alpha = float(np.clip(tukey_alpha, 0.0, 1.0))
        super().__init__(
            *args,
            parent_le=parent_layout.radius_le,
            PxToUm=1.0,
            **kwargs,
        )

        self.generate_weight_overlay()
        self.generate_alpha_slider()
        self.update_circle()

    def circle_center(self):
        """Exact image centre, the convention of the registration window."""
        ny, nx = self.init_frame.shape[:2]
        return ((nx - 1) / 2.0, (ny - 1) / 2.0)

    def on_xlims_or_ylims_change(self, event_ax):
        """Keep the disk on the image centre when zooming."""

    def generate_diameter_slider(self):
        """Generate the radius slider, bounded by the frame half-diagonal."""

        ny, nx = self.init_frame.shape[:2]
        max_radius = float(np.hypot(ny, nx) / 2.0)
        radius = self.initial_radius
        if radius is None:
            radius = 0.9 * min(ny, nx) / 2.0
        radius = float(np.clip(radius, 1.0, max_radius))
        self.diameter = 2.0 * radius

        self.diameter_slider = QLabeledDoubleSlider()
        radius_layout = QuickSliderLayout(
            label="Radius: ",
            slider=self.diameter_slider,
            slider_initial_value=radius,
            slider_range=(1.0, max_radius),
            decimal_option=True,
            precision=1,
        )
        radius_layout.setContentsMargins(15, 0, 15, 0)
        self.diameter_slider.valueChanged.connect(self.change_diameter)
        self.canvas.layout.addLayout(radius_layout)

    def change_diameter(self, value: float) -> None:
        """Update the disk from the radius slider."""
        self.diameter = 2.0 * value
        self.update_circle()

    def generate_weight_overlay(self):
        """Add the taper circle and the weight shading to the image axes."""
        import matplotlib.pyplot as plt

        # Distance map cached once, so a slider tick only re-evaluates the taper.
        self.radial_distance = radial_distance(self.init_frame.shape[:2])

        # An explicit RGBA image: matplotlib does not honour a per-pixel alpha array
        # passed to `set_alpha` with interpolation="none", and the overlay then
        # hides the whole frame behind opaque black.
        self.weight_rgba = np.zeros(self.init_frame.shape[:2] + (4,), dtype=np.float32)
        self.im_weight = self.ax.imshow(
            self.weight_rgba,
            interpolation="none",
            zorder=2,
        )
        # Leave the cursor readout to the frame underneath.
        self.im_weight.set_mouseover(False)
        self.circ.set_zorder(3)
        self.circ_taper = plt.Circle(
            self.circle_center(), 1.0, ec="tab:red", fill=False, ls="--", zorder=3
        )
        self.ax.add_patch(self.circ_taper)

    def generate_alpha_slider(self):
        """Generate the Tukey α slider."""

        self.alpha_slider = QLabeledDoubleSlider()
        alpha_layout = QuickSliderLayout(
            label="Tukey α: ",
            slider=self.alpha_slider,
            slider_initial_value=self.alpha,
            slider_range=(0.0, 1.0),
            decimal_option=True,
            precision=2,
        )
        alpha_layout.setContentsMargins(15, 0, 15, 0)
        self.alpha_slider.valueChanged.connect(self.change_alpha)
        # Keep the "Set" button last.
        self.canvas.layout.insertLayout(self.canvas.layout.count() - 1, alpha_layout)

    def roi_weights(self) -> np.ndarray:
        """Weights of the correlation for the current radius and α."""
        return disk_taper(self.radial_distance, self.alpha, self.circle_radius())

    def update_circle(self):
        """Redraw the circles and the weight shading."""
        radius = self.circle_radius()
        self.circ_taper.set_radius(radius * (1.0 - self.alpha))
        self.circ_taper.set_visible(0.0 < self.alpha < 1.0)
        self.weight_rgba[..., 3] = 0.6 * (1.0 - self.roi_weights())
        self.im_weight.set_data(self.weight_rgba)
        super().update_circle()

    def change_alpha(self, value: float) -> None:
        """Update the Tukey taper fraction."""
        self.alpha = float(np.clip(value, 0.0, 1.0))
        self.update_circle()

    def set_threshold_in_parent_le(self):
        """Write the radius and α into the registration options and close."""
        self.parent_layout.radius_le.setText(f"{self.diameter_slider.value():.1f}")
        self.parent_layout.alpha_le.set_threshold(round(self.alpha, 3))
        self.close()
