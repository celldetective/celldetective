from harness import *
import napari
from celldetective.napari.utils import control_segmentation_napari, launch_segmentation_viewer
from celldetective.napari.frame_segmentation import FrameSegmentationPanel

pos = os.path.join(EXP, "W1", "100") + os.sep
result = control_segmentation_napari(pos, prefix="Aligned", population="effectors", prepare_only=True)
result.pop("flush_memory", None)
launch_segmentation_viewer(**result, block=False, flush_memory=False)
pump(3)
viewer = napari.current_viewer()
qwin = viewer.window._qt_window
qwin.showNormal()
move(qwin, 0, 0, 1560, 860)
viewer.dims.set_current_step(0, 20)
viewer.reset_view()
pump(1)

panel = qwin.findChildren(FrameSegmentationPanel)[0]
i = panel.model_cb.findText("lymphocytes_ricm")
if i >= 0:
    panel.model_cb.setCurrentIndex(i)
pump(2)
# Run the model on the frame (CPU, under a minute) to show its result.
panel.replace_cb.setChecked(True)
panel.run_btn.click()
t0 = time.time()
while time.time() - t0 < 120:
    pump(1)
    if panel.run_btn.text() == "Segment this frame":
        break
pump(2)
grab(qwin, "napari_frame_segmentation")
