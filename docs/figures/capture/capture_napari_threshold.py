"""
The Threshold tab of the single-frame panel, on the nuclei of the ADCC demo.

The demo's own ``threshold_config_targets.json`` is applied inside two regions of
interest drawn on the frame, which is what the tab is for: trying a configuration
where it is hard to get right.
"""

from harness import *
import numpy as np
import napari
from celldetective.napari.utils import control_segmentation_napari, launch_segmentation_viewer
from celldetective.napari.threshold_segmentation import ThresholdSegmentationPanel

ADCC = require_adcc()
pos = os.path.join(ADCC, "W1", "100") + os.sep
result = control_segmentation_napari(pos, prefix="Corrected", population="targets", prepare_only=True)
result.pop("flush_memory", None)
launch_segmentation_viewer(**result, block=False, flush_memory=False)
pump(3)
viewer = napari.current_viewer()
qwin = viewer.window._qt_window
qwin.showNormal()
move(qwin, 0, 0, 1560, 860)
viewer.dims.set_current_step(0, 22)
viewer.reset_view()
# The configuration thresholds the live nuclei channel; show it rather than the
# brightfield the viewer opens on.
for layer in viewer.layers:
    if type(layer).__name__ == "Image":
        layer.visible = layer.name == "Image [3]"
pump(1)

panel = qwin.findChildren(ThresholdSegmentationPanel)[0]
tabs = panel.parent()
while tabs is not None and type(tabs).__name__ != "QTabWidget":
    tabs = tabs.parent()
if tabs is not None:
    tabs.setCurrentWidget(panel)
pump(0.5)

panel.load_configs([os.path.join(ADCC, "configs", "threshold_config_targets.json")])

# Two regions where the nuclei are dense, drawn as the user would with the
# rectangle tool of the ROI layer.
panel.add_roi_layer()
rois = viewer.layers[panel.region_cb.currentText()]
t = int(viewer.dims.current_step[0])
for y0, x0, y1, x1 in [(300, 1000, 1150, 1900), (1350, 300, 1900, 900)]:
    rois.add_rectangles(
        np.array([[t, y0, x0], [t, y0, x1], [t, y1, x1], [t, y1, x0]], dtype=float),
        edge_color="yellow",
        edge_width=6,
    )
rois.selected_data = set()
rois.mode = "pan_zoom"
print("ROI edge colors:", rois.edge_color)
viewer.layers.selection.active = viewer.layers["segmentation"]
pump(1)

# The frame as it is before any segmentation: trying a configuration out is
# what one does before running the pipeline over the position.
viewer.layers["segmentation"].data[t] = 0
viewer.layers["segmentation"].refresh()

panel.replace_cb.setChecked(True)
panel.run_btn.click()
t0 = time.time()
while time.time() - t0 < 300:
    pump(1)
    if panel.run_btn.text().startswith("Threshold"):
        break
pump(2)
try:
    qwin.statusBar()._help.setText("")
except Exception:
    pass
grab(
    qwin,
    "napari_threshold_segmentation",
    marks={
        "tabs": tabs.tabBar() if tabs is not None else None,
        "config": panel.config_lbl,
        "load": panel.load_btn,
        "wizard": panel.wizard_btn,
        "region": panel.region_cb,
        "add_roi": panel.add_roi_btn,
        "replace": panel.replace_cb,
        "following": panel.following_cb,
        "run": panel.run_btn,
    },
)
