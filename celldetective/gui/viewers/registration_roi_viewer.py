from typing import Any, Optional

import numpy as np
from superqt import QLabeledDoubleSlider

from celldetective.gui.gui_utils import QuickSliderLayout
from celldetective.gui.viewers.size_viewer import CellSizeViewer
from celldetective.utils.registration import tukey_window


class RegistrationROIViewer(CellSizeViewer):
    """
    Tune the correlation disk of the stack registration on a real frame.

    A :class:`CellSizeViewer` measuring a radius, with the circle fixed at the image centre as in
    :func:`celldetective.utils.registration.tukey_window`. It adds a dashed circle at the start of
    the Tukey taper, a shading of the pixels by how little they weigh in the correlation (fully
    shaded pixels are ignored) and a Tukey α slider.

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
        self.alpha = float(np.clip(tukey_alpha, 0.0, 1.0))
        super().__init__(
            *args,
            initial_diameter=1.0,  # set from the frame shape below
            parent_le=parent_layout.radius_le,
            follow_view_center=False,
            measure="radius",
            PxToUm=1.0,
            **kwargs,
        )

        ny, nx = self.init_frame.shape[:2]
        max_radius = float(np.hypot(ny, nx) / 2.0)
        if initial_radius is None:
            initial_radius = 0.9 * min(ny, nx) / 2.0
        radius = float(np.clip(initial_radius, 1.0, max_radius))

        self.generate_weight_overlay()
        self.generate_alpha_slider()
        self.diameter_slider.setRange(1.0, max_radius)
        self.diameter_slider.setValue(radius)
        self.change_diameter(radius)

    def circle_center(self):
        """Exact image centre, the convention of the registration window."""
        ny, nx = self.init_frame.shape[:2]
        return ((nx - 1) / 2.0, (ny - 1) / 2.0)

    def generate_weight_overlay(self):
        """Add the taper circle and the weight shading to the image axes."""
        import matplotlib.pyplot as plt

        self.im_weight = self.ax.imshow(
            np.ones(self.init_frame.shape[:2], dtype=np.float32),
            cmap="Greys",
            vmin=0,
            vmax=1,
            interpolation="none",
            zorder=2,
        )
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
            precision=5,
        )
        alpha_layout.setContentsMargins(15, 0, 15, 0)
        self.alpha_slider.valueChanged.connect(self.change_alpha)
        # Keep the "Set" button last.
        self.canvas.layout.insertLayout(self.canvas.layout.count() - 1, alpha_layout)

    def roi_weights(self) -> np.ndarray:
        """Weights of the correlation for the current radius and α."""
        return tukey_window(
            self.init_frame.shape[:2], alpha=self.alpha, radius=self.circle_radius()
        )

    def update_circle(self):
        """Redraw the circles and the weight shading."""
        radius = self.circle_radius()
        if hasattr(self, "circ_taper"):
            self.circ_taper.set_radius(radius * (1.0 - self.alpha))
            self.circ_taper.set_visible(0.0 < self.alpha < 1.0)
            self.im_weight.set_alpha(0.6 * (1.0 - self.roi_weights()))
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
