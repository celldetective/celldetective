from harness import *

init, cp = open_experiment()
init.hide()
move(cp, 40, 10)

pre = cp.PreprocessingPanel
pre.collapse_btn.click()
pump(1)

reg = pre.registration_options_layout
reg.radius_le.setText("420")
reg.reference_cb.setCurrentText("previous")
reg.add_correction_btn.click()
pump(0.5)
# Scroll down to the registration options and the list of corrections.
from PyQt5.QtWidgets import QScrollArea

for sa in cp.findChildren(QScrollArea):
    if sa.isVisible():
        sa.ensureWidgetVisible(pre.protocol_layout.protocol_list, 0, 40)
pump(0.5)
grab(cp, "control_panel_registration_protocol")

reg.open_roi_viewer()
pump(4)
viewer = reg.viewer.canvas
pump(3)
move(viewer, 500, 60)
grab(viewer, "registration_roi_viewer")
