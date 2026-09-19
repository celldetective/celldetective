# Capturing celldetective windows

Scripts start with `from harness import *` and run from `docs/figures/capture`
with `CELLDETECTIVE_DOCS_EXP` set to a scratch copy of `demo_ricm`
(single well W1, position 100, population `effectors`, RICM `adhesion_channel`,
40 frames, tracked and measured; `trajectories_effectors.csv` has
`class_firstdetection` 0/2 and `adhesion_channel_mean`, good for group
comparisons — `status_firstdetection == 0` rows have no measurements).

Each script takes ~30–60 s; run them one by one with a 3-minute timeout and
check the `saved …` lines, then `Read` the PNGs.

## Harness essentials

- `init, cp = open_experiment()` → start window and ControlPanel; `init.hide()`.
- `move(widget, x, y, w=None, h=None)` positions/resizes the top-level window.
- `grab(widget, "name")` captures the **frame** (title bar included) from the
  screen, after pinning the window topmost with Win32 `SetWindowPos`
  (Windows refuses focus to scripts; without it other apps cover the capture).
- `pump(seconds)` processes events; wait after every click that opens or
  loads something (1–4 s; model runs up to 120 s in a polling loop).

## Recipes (object paths as of Sept 2026 — verify with Grep if they fail)

| Window | How to reach it |
|---|---|
| Start window | `AppInitWindow(App, software_location=get_software_location())` |
| Population block | `cp.ProcessPopulations[0]`; expand with `.collapse_btn.click()` |
| Preprocessing block | `cp.PreprocessingPanel.collapse_btn.click()`; registration options `pre.registration_options_layout`; protocol list `pre.protocol_layout.protocol_list`; scroll with `QScrollArea.ensureWidgetVisible` |
| Registration ROI viewer | `reg.open_roi_viewer()`; the window is `reg.viewer.canvas` |
| Table Explorer | `block.view_tab_btn.click()` → `block.tab_ui`; select columns via `table_view.selectColumn` / selection model; then scroll bars back to 0 |
| 1D plot dialog / stats | `tab.plot_inst_action.trigger()` → `tab.plot1Dparams`; cards `tab.plot_selector.cards[name].setChecked(True)`, `tab.stats_selector.cards`; `x_cb/y_cb/hue_cb.setCurrentText`; `tab.plot1d_btn.click()` → `tab.pval_table`, `tab.effect_size_table` |
| Help panels | `block.help_seg_btn.click()` opens a `HelpMenu` (find it in `App.topLevelWidgets()`); `block.help_tracking()` → `block._help_panel`; `.answer("yes")` |
| Config editor | `cp.open_config_editor()` → `cp.cfg_editor`; tabs `ed.tabs`; `ed.add_label(name)`; `ed.labels_table`, `ed.metadata_table`; **never press Save** |
| napari segmentation viewer | `control_segmentation_napari(pos, prefix="Aligned", population="effectors", prepare_only=True)` then `launch_segmentation_viewer(**result, block=False, flush_memory=False)`; `napari.current_viewer().window._qt_window`; panel `qwin.findChildren(FrameSegmentationPanel)[0]` (`model_cb`, `replace_cb`, `run_btn`) |

## Staging tricks

- Private paths: the config editor's path is an `ElidedLabel`; set
  `ed.path_label.full_text` (setting the text is undone on resize).
- `setCurrentCell` scrolls a table to that cell; after switching tabs call
  `resizeColumnsToContents()` and reset the horizontal scroll bar.
- napari's status bar is overwritten by the mouse position; don't annotate a
  status message, point at the result on the canvas instead.
- Demo values that make a point: pick groups that both have data, and check
  the numbers you quote in a caption against the capture.
- Close windows you opened; the scripts never save configurations or labels,
  and the demo copy should stay unchanged (`git`-style check: compare
  `config.ini` before/after if in doubt).
