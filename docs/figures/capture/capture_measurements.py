"""
The measurement settings (Configure measurements) and the viewers they open:
local and fit background correction, mask-based, texture, position-based
measurements and spot detection.

Runs on demo_adcc (CELLDETECTIVE_DOCS_ADCC), target population.
"""

from harness import *


def scroll_to(ms, widget, top_margin=10):
    """Scroll the settings window so that ``widget`` sits at its top."""
    area = ms._scroll_area
    y = widget.mapTo(ms._widget, widget.rect().topLeft()).y()
    area.verticalScrollBar().setValue(max(0, y - top_margin))
    pump(0.5)


FRAME = 20
# (x, y) limits of the spot detection view, in pixels of the image
SPOT_ZOOM = ((700, 1400), (1300, 600))


def show_viewer(viewer, x=1040, y=40, size=None):
    """Go to FRAME and return the window of a stack viewer (often its canvas)."""
    canvas = getattr(viewer, "canvas", None)
    win = canvas if canvas is not None and canvas.isWindow() and canvas.isVisible() else viewer
    if size:
        win.resize(*size)
    win.move(x, y)
    pump(1)
    if hasattr(viewer, "frame_slider"):
        viewer.frame_slider.setValue(FRAME)
        pump(3)
    return win


init, cp = open_experiment(require_adcc())
init.hide()
move(cp, 20, 40)
block = cp.ProcessPopulations[1]
block.collapse_btn.click()
pump(1)

block.open_measurement_configuration_ui()
pump(2.5)
ms = block.settings_measurements
move(ms, 480, 40, 520, 760)

# -- Local background correction and its distance viewer ---------------------
local = ms.local_correction_layout
local.channels_cb.setCurrentText("effector_fluo_channel")
local.threshold_le.set_threshold(5)
scroll_to(ms, ms.normalisation_frame)
pl = ms.protocol_layout
grab(ms, "measure_local", marks={
    "tabs": pl.tabs.tabBar(), "channel": local.channels_cb, "distance": local.threshold_le,
    "viewer": local.threshold_viewer_btn, "model": local.models_cb,
    "subtract": local.operation_layout.subtract_btn, "noclip": local.operation_layout.clip_not_btn,
    "add": local.add_correction_btn, "list": pl.protocol_list, "delete": pl.delete_protocol_btn,
})

local.threshold_viewer_btn.click()
pump(4)
viewer = local.viewer
win = show_viewer(viewer)
grab(win, "measure_local_viewer", marks={
    "edge": viewer.edge_slider, "opacity": viewer.opacity_slider, "set": viewer.set_measurement_btn,
})
win.close()
pump(0.5)

# -- Fit background correction and its exclusion threshold viewer ------------
ms.protocol_layout.tabs.setCurrentIndex(1)
pump(0.5)
fit = ms.fit_correction_layout
fit.channels_cb.setCurrentText("dead_nuclei_channel")
fit.threshold_le.set_threshold(2)
fit.operation_layout.subtract_btn.setChecked(True)
fit.operation_layout.clip_not_btn.setChecked(True)
fit.add_correction_btn.click()
pump(0.5)
grab(ms, "measure_fit", marks={
    "tabs": pl.tabs.tabBar(), "channel": fit.channels_cb, "threshold": fit.threshold_le,
    "viewer": fit.threshold_viewer_btn, "model": fit.models_cb, "downsample": fit.downsample_le,
    "subtract": fit.operation_layout.subtract_btn, "noclip": fit.operation_layout.clip_not_btn,
    "preview": fit.corrected_stack_viewer, "add": fit.add_correction_btn, "list": pl.protocol_list,
})

fit.threshold_viewer_btn.click()
pump(4)
viewer = fit.viewer
win = show_viewer(viewer)
grab(win, "measure_fit_viewer", marks={
    "threshold": viewer.threshold_slider, "opacity": viewer.opacity_slider,
    "apply": viewer.apply_threshold_btn, "channel": viewer.channel_cb,
})
win.close()
pump(0.5)
ms.protocol_layout.protocol_list.clear()
ms.protocol_layout.protocols = []

# -- Mask-based measurements and texture -------------------------------------
scroll_to(ms, ms.features_frame)
grab(ms, "measure_features", marks={
    "features": ms.features_list, "add_feature": ms.add_feature_btn, "create_feature": ms.create_feature_btn,
    "contours": ms.contours_list, "add_contour": ms.add_contour_btn, "view_contour": ms.view_contour_btn,
})

ms.activate_haralick_btn.setChecked(True)
ms.haralick_channel_choice.setCurrentText("brightfield_channel")
pump(0.5)
scroll_to(ms, ms.activate_haralick_btn, top_margin=60)
grab(ms, "measure_texture", marks={
    "check": ms.activate_haralick_btn, "channel": ms.haralick_channel_choice,
    "distance": ms.haralick_distance_le, "levels": ms.haralick_n_gray_levels_le,
    "scale": ms.haralick_scale_slider, "hist": ms.haralick_hist_btn, "digit": ms.haralick_digit_btn,
    "pmin": ms.haralick_percentile_min_le, "pmax": ms.haralick_percentile_max_le,
    "norm": ms.haralick_normalization_mode_btn,
})

# -- Position-based measurements ---------------------------------------------
scroll_to(ms, ms.iso_frame)
grab(ms, "measure_position", marks={
    "radii": ms.radii_list, "add_radius": ms.add_radius_btn, "del_radius": ms.del_radius_btn,
    "ops": ms.operations_list, "add_op": ms.add_op_btn, "del_op": ms.del_op_btn,
})

# -- Spot detection -----------------------------------------------------------
ms.spot_check.setChecked(True)
ms.spot_channel.setCurrentText("dead_nuclei_channel")
ms.diameter_value.setText("7")
ms.threshold_value.setText("0.5")
pump(0.5)
scroll_to(ms, ms.spot_detection_frame)
grab(ms, "measure_spots", marks={
    "check": ms.spot_check, "channel": ms.spot_channel, "preprocessing": ms.spot_preprocessing.list,
    "diameter": ms.diameter_value, "threshold": ms.threshold_value, "viewer": ms.spot_viewer_btn,
    "save": ms.submit_btn,
})

ms.spot_viewer_btn.click()
pump(5)
viewer = ms.spot_visual
win = show_viewer(viewer, x=560, size=(1000, 720))
viewer.apply_thresh_btn.click()
pump(3)
# Zoom on a cluster of target cells, where dead nuclei light up.
viewer.ax.set_xlim(*SPOT_ZOOM[0])
viewer.ax.set_ylim(*SPOT_ZOOM[1])
viewer.canvas.canvas.draw()
pump(1)
grab(win, "measure_spots_viewer", marks={
    "channel": viewer.detection_channel_cb, "preprocessing": viewer.preprocessing.list,
    "diameter": viewer.spot_diam_le, "set_diameter": viewer.apply_diam_btn,
    "threshold": viewer.spot_thresh_le, "set_threshold": viewer.apply_thresh_btn,
    "add": viewer.add_measurement_btn, "canvas": viewer.canvas.canvas,
})
win.close()

ms.close()
pump(0.5)
App.quit()
