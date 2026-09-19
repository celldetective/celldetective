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

import sys

from annotate import Figure


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
    for n, y in [(1, 100), (2, 157), (3, 255), (4, 324), (5, 356)]:
        f.badge(n, nx + 1268, ny + y, r=16)
    f.badge(6, nx + 1062, ny + 398, r=16)
    return f.render()


FIGURES = [
    registration,
    config_editor,
    table_explorer,
    table_stats,
    help_panel,
    napari_frame_segmentation,
]

if __name__ == "__main__":
    wanted = sys.argv[1:]
    for build in FIGURES:
        if not wanted or any(w in build.__name__ for w in wanted):
            build()
