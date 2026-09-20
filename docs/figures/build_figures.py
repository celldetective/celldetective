"""
Build the annotated figures of the documentation.

    python build_figures.py            # every figure
    python build_figures.py table      # the figures whose name contains "table"

Each function lays out screenshots of ``screenshots/`` (taken by the scripts of
``capture/``) and annotates them. Coordinates are in pixels of the figure; a
screenshot placed at (x, y) shifts its own pixel coordinates by that much, which
is what the ``ox``/``oy`` offsets below stand for. The SVGs are written to
``docs/source/_static/figures``.
"""

import json
import os
import sys

from annotate import RAW, Figure


def marks(shot, ox, oy, pad=3):
    """
    Rectangles of the widgets recorded with a capture (screenshots/<shot>.json),
    moved by the screenshot's position and padded, ready for Figure.callout.
    """
    with open(os.path.join(RAW, shot + ".json")) as f:
        rects = json.load(f)
    return {
        k: (ox + x - pad, oy + y - pad, w + 2 * pad, h + 2 * pad)
        for k, (x, y, w, h) in rects.items()
    }


def grow(r, left=0, top=0, right=0, bottom=0):
    """Enlarge a rectangle, e.g. to take in a label next to the widget."""
    x, y, w, h = r
    return (x - left, y - top, w + left + right, h + top + bottom)


def union(*rects):
    """The rectangle around several rectangles."""
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    return (x0, y0, x1 - x0, y1 - y0)


def registration():
    f = Figure("stack-registration", 1117, 845)
    ox, oy = 20, 20
    f.shot("control_panel_registration_protocol.png", ox, oy)
    vx, vy = 560, 60
    f.shot("registration_roi_viewer.png", vx, vy)

    # The options of the registration, then the protocol list and Submit.
    f.box(ox + 36, oy + 378, 356, 262)
    f.badge(1, ox + 36, oy + 378)
    f.box(ox + 38, oy + 660, 354, 98)
    f.badge(3, ox + 24, oy + 700)
    f.badge(4, ox + 24, oy + 776)

    # The ROI button opens the viewer on a frame of the current position.
    f.badge(2, ox + 362, oy + 437)
    f.arrow(ox + 385, oy + 470, vx - 4, vy + 250, bend=-0.3)
    return f.render()


def config_editor():
    f = Figure("config-editor", 997, 887)
    ax, ay = 20, 20
    f.shot("config_editor_1_well_labels.png", ax, ay)
    bx, by = 330, 330
    f.shot("config_editor_2_metadata.png", bx, by)

    f.callout(1, ax + 524, ay + 109, 97, 30, rounded=False)  # label tools
    f.callout(2, ax + 467, ay + 146, 73, 60, side="right")  # the new column
    f.callout(3, ax + 604, ay + 41, 25, 30, rounded=False)  # text editor button
    f.callout(4, bx + 551, by + 111, 73, 28, rounded=False)  # metadata tools
    f.callout(5, bx + 22, by + 204, 597, 32)  # the new entry
    f.callout(6, bx + 325, by + 490, 303, 30, frame=False)  # Save
    return f.render()


def table_explorer():
    tx, ty = 20, 150
    f = Figure("table-explorer", 1092, ty + 592 + 45)
    f.shot("table_ui.png", tx, ty)

    # Tools of the header, labelled from the right so that no line crosses a label.
    tools = [
        (1024, "save as .csv"),
        (994, "delete the selected columns"),
        (964, "copy the selected cells"),
        (905, "keep the rows matching a query"),
        (875, "collapse each track into one row"),
        (845, "distributions and statistics"),
        (815, "plot the selected columns"),
    ]
    for k, (x, s) in enumerate(tools):
        y = 22 + 18 * k
        f.label(s, tx + x - 10, y, size=14, anchor="end")
        f.leader(tx + x, y - 5, tx + x, ty + 64)

    f.label("menus: File · Edit · Table · Math · Plot", tx, 125, size=14)
    f.leader(tx + 60, 131, tx + 60, ty + 36)
    f.label("right-click a column header", tx + 300, 70, size=14)
    f.label("for the column actions", tx + 300, 88, size=14)
    f.leader(tx + 360, 94, tx + 360, ty + 112)
    f.label("what is selected, and how much", tx + 280, ty + 592 + 32, size=14)
    f.leader(tx + 276, ty + 592 + 27, tx + 150, ty + 578)
    return f.render()


def table_stats():
    f = Figure("table-stats", 827, 683)
    dx, dy = 20, 20
    f.shot("plot_1d_dialog.png", dx, dy)
    px, py = 340, 20
    f.shot("pval_table.png", px, py)
    ex, ey = 340, 360
    f.shot("effect_size_table.png", ex, ey)

    f.box(px + 8, py + 86, 446, 68)
    f.box(ex + 8, ey + 86, 446, 56)
    f.arrow(dx + 272, dy + 604, ex - 4, ey + 150, bend=0.25)
    return f.render()


def help_panel():
    f = Figure("help-panel", 1167, 890)
    cx, cy = 20, 20
    f.shot("control_panel_effectors.png", cx, cy)
    mx, my = 520, 40
    f.shot("help_menu.png", mx, my)
    qx, qy = 500, 280
    f.shot("help_question.png", qx, qy)
    sx, sy = 500, 570
    f.shot("help_suggestion.png", sx, sy)

    # The help buttons of SEGMENT and TRACK, each opening its helper.
    f.callout(1, cx + 364, cy + 325, 26, 26, side="right", rounded=False)
    f.callout(3, cx + 364, cy + 393, 26, 26, side="right", rounded=False)
    f.arrow(cx + 422, cy + 326, mx - 4, my + 90, bend=-0.2)
    f.arrow(cx + 422, cy + 410, qx - 4, qy + 120, bend=0.15)

    f.callout(2, mx + 16, my + 74, 350, 82, side="right")  # the two helpers
    f.callout(4, qx + 18, qy + 114, 606, 68)  # the question
    f.arrow(qx + 600, qy + 222, sx + 600, sy - 4, bend=-0.35)
    f.label("yes, yes", qx + 585, qy + 268, size=15, anchor="end")

    f.callout(5, sx + 16, sy + 93, 386, 20, side="right")  # answers so far
    f.callout(6, sx + 16, sy + 230, 60, 38, side="right", rounded=False)  # Back
    f.callout(7, sx + 383, sy + 233, 159, 33, frame=False)  # Read the tutorial
    return f.render()


def napari_frame_segmentation():
    f = Figure("napari-frame-segmentation", 1607, 937)
    nx, ny = 20, 20
    f.shot("napari_frame_segmentation.png", nx, ny)
    m = marks("napari_frame_segmentation", nx, ny)

    # The rows start at the very edge of the dock, so a badge on their left would
    # sit on the label it points at: those rows are called out from the right.
    f.callout(1, *m["tabs"], side="left", r=16, rounded=False)
    f.callout(2, *grow(m["model"], left=89), side="right", r=16)
    f.callout(3, *m["channels"], side="left", r=16)
    f.callout(4, *m["parameters"], side="left", r=16)
    f.callout(5, *m["replace"], side="right", r=16)
    f.callout(6, *m["run"], frame=False, r=16)
    # The layer the labels land in, in the layer list.
    f.callout(7, nx + 7, ny + 504, 278, 30, side="right", r=16)
    return f.render()


def napari_threshold_segmentation():
    f = Figure("napari-threshold-segmentation", 1607, 937)
    nx, ny = 20, 20
    f.shot("napari_threshold_segmentation.png", nx, ny)
    m = marks("napari_threshold_segmentation", nx, ny)

    # The rows are ~30 px apart, closer than two badges: alternate the sides.
    f.callout(1, *m["tabs"], side="right", r=16, rounded=False)
    f.callout(2, *grow(m["config"], left=65, top=2, bottom=2), side="left", r=16)
    f.callout(3, *union(m["load"], m["wizard"]), side="right", r=16)
    # The frame takes in the "region:" label, so that the badge lands beside the
    # panel rather than on the word.
    f.callout(4, *grow(union(m["region"], m["add_roi"]), left=74), side="left", r=16)
    f.callout(5, *m["replace"], side="right", r=16)
    f.callout(6, *m["following"], side="left", r=16)
    f.callout(7, *m["run"], frame=False, r=16)
    # One of the two regions drawn on the frame.
    f.callout(8, nx + 508, ny + 563, 234, 219, side="left", r=16)
    return f.render()


def background_correction_local():
    f = Figure("background-correction-local", 1180, 800)
    ox, oy = 20, 20
    f.shot("measure_local.png", ox, oy, crop=(0, 0, 520, 480))
    m = marks("measure_local", ox, oy)
    vx, vy = 620, 20
    f.shot("measure_local_viewer.png", vx, vy)
    v = marks("measure_local_viewer", vx, vy)

    f.callout(1, *m["tabs"], side="right")
    f.callout(2, *union(m["distance"], m["viewer"]), side="right")
    vb = m["viewer"]
    f.arrow(vb[0] + vb[2] + 36, vb[1] + vb[3] / 2, vx - 4, vy + 250, bend=-0.2)
    f.callout(3, *v["edge"])
    f.callout(4, *v["set"], frame=False)
    f.callout(5, *m["model"], side="right")
    # The whole operation block: a frame drawn on the two selected buttons alone
    # cut through the label of a third, and the paragraph names all four.
    f.callout(
        6, *union(m["subtract"], m["divide"], m["clip"], m["noclip"]), side="right"
    )
    f.callout(7, *m["add"], frame=False)
    f.callout(8, *m["list"], side="right")
    return f.render()


def background_correction_fit():
    f = Figure("background-correction-fit", 1180, 800)
    ox, oy = 20, 20
    f.shot("measure_fit.png", ox, oy, crop=(0, 0, 520, 480))
    m = marks("measure_fit", ox, oy)
    vx, vy = 620, 20
    f.shot("measure_fit_viewer.png", vx, vy)
    v = marks("measure_fit_viewer", vx, vy)

    f.callout(1, *m["tabs"], side="right")
    f.callout(2, *union(m["threshold"], m["viewer"]), side="right")
    vb = m["viewer"]
    f.arrow(vb[0] + vb[2] + 36, vb[1] + vb[3] / 2, vx - 4, vy + 250, bend=-0.2)
    f.callout(3, *v["threshold"], side="right")
    f.callout(4, *v["apply"], frame=False)
    f.callout(5, *union(m["model"], m["downsample"]), side="right")
    f.callout(
        6, *union(m["subtract"], m["divide"], m["clip"], m["noclip"]), side="right"
    )
    f.callout(7, *m["preview"], side="right", rounded=False)
    f.callout(8, *m["add"], frame=False)
    f.callout(9, *m["list"], side="right")
    return f.render()


def texture_measurements():
    f = Figure("texture-measurements", 600, 400)
    ox, oy = 20, 20
    f.shot("measure_texture.png", ox, oy, crop=(0, 0, 520, 350))
    m = marks("measure_texture", ox, oy)
    f.callout(1, *m["check"])
    f.callout(2, *union(m["hist"], m["digit"]), side="right", rounded=False)
    f.callout(3, *m["channel"], side="right")
    f.callout(4, *union(m["distance"], m["levels"], m["scale"]), side="right")
    f.callout(5, *union(m["pmin"], m["pmax"]))
    f.callout(6, *m["norm"], side="right", rounded=False)
    return f.render()


def contour_measurements():
    f = Figure("contour-measurements", 600, 420)
    ox, oy = 20, 20
    f.shot("measure_features.png", ox, oy, crop=(0, 0, 520, 370))
    m = marks("measure_features", ox, oy)
    f.callout(1, *m["features"], side="right")
    f.callout(2, *union(m["add_feature"], m["create_feature"]), side="right", rounded=False)
    f.callout(3, *union(m["add_contour"], m["view_contour"]), side="right", rounded=False)
    f.callout(4, *m["contours"], side="right")
    return f.render()


def position_measurements():
    f = Figure("position-measurements", 600, 560)
    ox, oy = 20, 20
    f.shot("measure_position.png", ox, oy, crop=(0, 0, 520, 500))
    m = marks("measure_position", ox, oy)
    f.callout(1, *union(m["del_radius"], m["add_radius"]), side="right", rounded=False)
    f.callout(2, *m["radii"], side="right")
    f.callout(3, *union(m["del_op"], m["add_op"]), side="right", rounded=False)
    f.callout(4, *m["ops"], side="right")
    return f.render()


def spot_detection():
    f = Figure("spot-detection", 1600, 840)
    ox, oy = 20, 20
    f.shot("measure_spots.png", ox, oy)
    m = marks("measure_spots", ox, oy)
    vx, vy = 580, 60
    f.shot("measure_spots_viewer.png", vx, vy)
    v = marks("measure_spots_viewer", vx, vy)

    f.callout(1, *m["check"])
    f.callout(2, *union(m["channel"], m["preprocessing"]), side="right")
    f.callout(3, *union(m["diameter"], m["threshold"]), side="right")
    f.callout(4, *m["viewer"], side="right", rounded=False)
    f.arrow(m["viewer"][0] + 36, m["viewer"][1] + 12, vx - 4, vy + 200, bend=-0.3)
    f.callout(5, *v["preprocessing"], side="right")
    # Both fields and both Set buttons: the threshold field reaches further left
    # than the diameter one, and a frame on the diameter alone cut through it.
    f.callout(
        6,
        *union(v["diameter"], v["threshold"], v["set_diameter"], v["set_threshold"]),
        side="right",
    )
    f.callout(7, *v["add"], frame=False)
    f.callout(8, *m["save"], frame=False)
    return f.render()


def apply_segmentation_model():
    f = Figure("apply-segmentation-model", 1060, 760)
    bx, by = 20, 20
    f.shot("process_targets.png", bx, by, crop=(0, 0, 460, 700))
    b = marks("process_targets", bx, by)
    ux, uy = 560, 60
    f.shot("upload_model_cellpose.png", ux, uy)
    u = marks("upload_model_cellpose", ux, uy)

    f.callout(1, *b["upload"], side="top", rounded=False)
    ub = b["upload"]
    f.arrow(ub[0] + ub[2] / 2 + 20, ub[1] - 30, ux - 4, uy + 60, bend=-0.25)
    f.callout(2, *union(u["threshold"], u["cellpose"]), side="right")
    f.callout(3, *u["calibration"], side="right")
    f.callout(4, *union(u["channel0"], u["channel1"]), side="right")
    f.callout(5, *union(u["diameter"], u["flow"]), side="right")
    f.callout(6, *u["choose"], rounded=False)
    f.callout(7, *u["upload"], frame=False)
    f.callout(8, *union(b["segment"], b["seg_models"]))
    f.callout(9, *b["submit"], frame=False)
    return f.render()


def threshold_wizard():
    f = Figure("threshold-wizard", 1440, 900)
    wx, wy = 20, 20
    f.shot("threshold_wizard_1.png", wx, wy)
    w = marks("threshold_wizard_1", wx, wy)
    f.callout(1, *w["preprocessing"], side="right")
    f.callout(2, *w["apply"], frame=False)
    f.callout(3, *union(w["fill"], w["equalize"]), side="top", rounded=False)
    f.callout(4, *w["slider"], side="right")
    f.callout(5, *w["channel"], side="right")
    return f.render()


def threshold_wizard_objects():
    f = Figure("threshold-wizard-objects", 1440, 900)
    wx, wy = 20, 20
    f.shot("threshold_wizard_2.png", wx, wy)
    w = marks("threshold_wizard_2", wx, wy)
    objects = union(w["markers"], w["min_dist"], w["run"], w["watershed"])
    f.callout(1, *grow(objects, bottom=-(w["run"][3] + 4)), side="right")
    f.callout(2, *w["run"], frame=False)
    f.callout(3, *w["watershed"], frame=False)
    f.callout(4, *grow(union(w["props"], w["feature1"]), left=8), side="right")
    f.callout(5, *union(w["query"], w["submit"]), side="right")
    f.callout(6, *w["save"], frame=False)
    return f.render()


def tracking_settings():
    f = Figure("tracking-settings", 1080, 860)
    bx, by = 20, 20
    f.shot("process_targets.png", bx, by, crop=(0, 0, 460, 700))
    b = marks("process_targets", bx, by)
    tx, ty = 540, 30
    f.shot("tracking_settings_btrack.png", tx, ty)
    t = marks("tracking_settings_btrack", tx, ty)

    f.callout(1, *b["track_config"], side="right", rounded=False)
    tb = b["track_config"]
    f.arrow(tb[0] + tb[2] + 36, tb[1] + tb[3] / 2, tx - 4, ty + 200, bend=0.2)
    f.callout(2, *union(t["btrack"], t["trackpy"]), side="right")
    f.callout(3, *t["config"], side="right")
    f.callout(4, *t["features"], side="right")
    f.callout(5, *t["post"], side="right")
    f.callout(6, *t["save"], frame=False)
    f.callout(7, *b["track"])
    f.callout(8, *b["submit"], frame=False)
    return f.render()


def train_segmentation():
    f = Figure("train-segmentation-model", 1180, 940)
    ax, ay = 20, 20
    f.shot("train_segmentation_1.png", ax, ay)
    a = marks("train_segmentation_1", ax, ay)
    bx, by = 600, 40
    f.shot("train_segmentation_2.png", bx, by)
    b = marks("train_segmentation_2", bx, by)

    f.callout(1, *a["name"], side="right")
    f.callout(2, *grow(a["pretrained"], right=135), side="right")
    f.callout(3, *a["channel0"], side="right")
    f.callout(4, *a["add_channel"], side="right", rounded=False)
    f.callout(5, *b["calibration"], side="right")
    f.callout(6, *grow(b["data_folder"], right=135), side="right")
    f.callout(7, *union(b["dataset"], b["validation"]), side="right")
    f.callout(8, *b["hyper"], side="right")
    f.callout(9, *b["train"], frame=False)
    return f.render()


def train_event_model():
    f = Figure("train-event-model", 1180, 940)
    ax, ay = 20, 20
    f.shot("train_event_1.png", ax, ay)
    a = marks("train_event_1", ax, ay)
    bx, by = 600, 40
    f.shot("train_event_2.png", bx, by)
    b = marks("train_event_2", bx, by)

    f.callout(1, *union(a["name"], a["event_name"]), side="right")
    f.callout(2, *grow(a["pretrained"], right=135), side="right")
    f.callout(3, *a["channel0"], side="right")
    f.callout(4, *b["length"], side="right")
    f.callout(5, *grow(b["data_folder"], right=135), side="right")
    f.callout(6, *union(b["dataset"], b["validation"]), side="right")
    f.callout(7, *b["hyper"], side="right")
    f.callout(8, *b["train"], frame=False)
    return f.render()


def classifier():
    f = Figure("classifier", 580, 870)
    cx, cy = 40, 20
    f.shot("classifier_time_correlated.png", cx, cy)
    c = marks("classifier_time_correlated", cx, cy)
    f.callout(1, *c["name"])
    f.callout(2, *c["project"], side="right", rounded=False)
    f.callout(3, *union(c["frame"], c["alpha"]), side="right")
    f.callout(4, *grow(union(c["feature0"], c["feature1"]), right=36), side="right")
    f.callout(5, *union(c["query"], c["preview"]), side="right")
    f.callout(6, *union(c["time_corr"], c["transient"], c["prereq"]), side="right")
    f.callout(7, *c["apply"], frame=False)
    return f.render()


def signal_annotator_settings():
    f = Figure("signal-annotator-settings", 540, 770)
    sx, sy = 40, 20
    f.shot("signal_annotator_settings.png", sx, sy)
    s_ = marks("signal_annotator_settings", sx, sy)
    f.callout(1, *union(s_["grayscale"], s_["rgb"]), side="right")
    f.callout(2, *union(s_["log"], s_["percentile"]), side="right", rounded=False)
    f.callout(3, *s_["channel"], side="right")
    f.callout(4, *grow(s_["fraction"], top=-22, bottom=-22), side="right")
    f.callout(5, *s_["save"], frame=False)
    return f.render()


def _plot_controls(f, px, py, n, extra=()):
    """Callouts on the controls of a survival or signal plot window."""
    f.callout(n, px + 22, py + 60, 366, 28, side="right")  # grouping
    f.callout(n + 1, px + 240, py + 98, 184, 30, side="left", rounded=False)  # toolbar
    for k, r in enumerate(extra):
        f.callout(n + 2 + k, px + r[0], py + r[1], r[2], r[3], side="right")


def survival():
    f = Figure("survival", 1380, 640)
    ax, ay = 20, 20
    f.shot("analyze_tab.png", ax, ay, crop=(0, 0, 460, 470))
    a = marks("analyze_tab", ax, ay)
    ox, oy = 530, 40
    f.shot("survival_options.png", ox, oy)
    o = marks("survival_options", ox, oy)
    px, py = 920, 60
    f.shot("survival_plot.png", px, py)

    f.callout(1, *a["survival"], frame=False)
    sb = a["survival"]
    f.arrow(sb[0] + sb[2] - 60, sb[1] - 6, ox - 4, oy + 90, bend=-0.3)
    f.callout(2, *o["population"], side="right")
    f.callout(3, *union(o["reference"], o["interest"]), side="right")
    f.callout(4, *o["query"], side="right")
    f.callout(5, *o["calibration"], side="right")
    f.callout(6, *o["submit"], frame=False)
    ob = o["submit"]
    f.arrow(ob[0] + ob[2] - 40, ob[1] + ob[3] + 8, px - 4, py + 420, bend=0.3)
    _plot_controls(f, px, py, 7, extra=[(186, 502, 64, 28)])
    return f.render()


def signals():
    f = Figure("synchronized-signals", 1390, 780)
    ax, ay = 20, 20
    f.shot("analyze_tab.png", ax, ay, crop=(0, 0, 460, 470))
    a = marks("analyze_tab", ax, ay)
    ox, oy = 530, 40
    f.shot("signals_options.png", ox, oy)
    o = marks("signals_options", ox, oy)
    fx, fy = 530, 600
    f.shot("signals_feature.png", fx, fy)
    s_ = marks("signals_feature", fx, fy)
    px, py = 930, 60
    f.shot("signals_plot.png", px, py)

    f.callout(1, *a["signals"], frame=False)
    sb = a["signals"]
    f.arrow(sb[0] + sb[2] - 60, sb[1] - 6, ox - 4, oy + 90, bend=-0.3)
    f.callout(2, *union(o["population"], o["interest"]), side="right")
    f.callout(3, *o["absolute"], side="right")
    f.callout(4, *union(o["query"], o["pool"]), side="right")
    f.callout(5, *o["submit"], frame=False)
    f.arrow(o["submit"][0] + 120, o["submit"][1] + o["submit"][3] + 6, fx + 120, fy - 6, bend=0)
    f.callout(6, *s_["feature"], side="right")
    f.callout(7, *s_["set"], frame=False)
    f.arrow(fx + s_["set"][2] + 20, s_["set"][1] + 14, px - 4, py + 420, bend=0.3)
    _plot_controls(f, px, py, 8, extra=[(12, 500, 412, 30), (12, 540, 412, 40), (186, 646, 64, 28)])
    return f.render()


FIGURES = [
    registration,
    config_editor,
    table_explorer,
    table_stats,
    help_panel,
    napari_frame_segmentation,
    napari_threshold_segmentation,
    background_correction_local,
    background_correction_fit,
    texture_measurements,
    contour_measurements,
    position_measurements,
    spot_detection,
    apply_segmentation_model,
    threshold_wizard,
    threshold_wizard_objects,
    tracking_settings,
    train_segmentation,
    train_event_model,
    classifier,
    signal_annotator_settings,
    survival,
    signals,
]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    for build in FIGURES:
        if not wanted or any(w in build.__name__ for w in wanted):
            build()
