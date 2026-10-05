"""
The movie prefix field of the configuration editor and the stack viewer.

For the prefix suggestions to show more than one entry, the movie folder of the
demo copy is expected to hold, next to the aligned stack of the demo, a raw
stack and a corrected one, as an experiment does after preprocessing: their
content does not matter, only their names (e.g. 2 x 8 x 8 zeros saved as
``R_1_MMStack_Undefined2.ome.tif`` and
``Corrected_Aligned_Normed_R_1_MMStack_Undefined2.ome.tif``).
"""

import json

import numpy as np
from PyQt5.QtCore import QPoint

from harness import *

init, cp = open_experiment()
init.hide()
move(cp, 40, 10)

# Configuration editor: the movie prefix field of the Settings tab, the line
# under it telling what the prefix matches, and the list of its suggestions.
cp.open_config_editor()
pump(1)
ed = cp.cfg_editor
ed.path_label.full_text = r"C:\Users\me\Experiments\demo_ricm\config.ini"
move(ed, 440, 40, 640, 500)
ed.tabs.setCurrentIndex(0)
pump(2)  # the scan of the movie folders runs off the GUI thread
prefix = ed.prefix_widget
grab(
    ed,
    "config_editor_movie_prefix",
    keep_on_top=True,
    marks={"field": prefix.field, "button": prefix.suggest_btn, "hint": prefix.hint},
)
# The popup closes when the editor is activated for its capture, so it is
# captured on its own, with its position relative to the editor's frame.
prefix.show_suggestions()
pump(0.8)
popup = prefix.completer.popup()
if popup.isVisible():
    popup.grab().save(os.path.join(OUT, "config_editor_movie_prefix_popup.png"))
    p, origin = popup.mapToGlobal(QPoint(0, 0)), ed.frameGeometry().topLeft()
    marks_file = os.path.join(OUT, "config_editor_movie_prefix.json")
    with open(marks_file) as f:
        rects = json.load(f)
    rects["popup"] = [p.x() - origin.x(), p.y() - origin.y(), popup.width(), popup.height()]
    with open(marks_file, "w") as f:
        json.dump(rects, f, indent=1)
    print("popup", rects["popup"])
    popup.hide()
ed.close()  # never saved: the demo copy stays as it is

# Stack viewer of the position: contrast refined once with the auto button and
# an intensity profile drawn across the frame.
cp.view_current_stack()
pump(3)
viewer = cp.viewer
move(viewer.canvas, 900, 40, 620, 760)
pump(1)
viewer.auto_contrast()
pump(0.5)
viewer.line_action.setChecked(True)
viewer.toggle_line_mode()
pump(0.5)
ny, nx = np.asarray(viewer.init_frame).shape[:2]
viewer.line_x = [0.2 * nx, 0.8 * nx]
viewer.line_y = [0.35 * ny, 0.65 * ny]
(viewer.line_artist,) = viewer.ax.plot(
    viewer.line_x, viewer.line_y, color=viewer.line_color, linewidth=3
)
viewer.update_profile()
viewer.canvas.draw()
pump(1)
toolbar = viewer.canvas.toolbar
grab(
    viewer.canvas,
    "stack_viewer",
    marks={
        "contrast": viewer.contrast_slider,
        "auto": viewer.auto_contrast_btn,
        "line": toolbar.widgetForAction(viewer.line_action),
        "lock": toolbar.widgetForAction(viewer.lock_y_action),
        "channel": viewer.channel_cb,
        "time": viewer.frame_slider,
    },
)
viewer.canvas.close()
