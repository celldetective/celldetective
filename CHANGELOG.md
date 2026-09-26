# Changelog

All notable changes to Celldetective are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The movie prefix of the experiment configuration suggests the prefixes the
  experiment actually holds. The names of the stacks sitting in the movie
  folders are cut around their separators and their numbering, and the pieces
  that select a stack in the most positions are offered as completions, from
  the field or from the button next to it. A line under the field tells what
  the prefix typed matches, so a prefix leaving positions without a movie, or
  matching several stacks in one, is seen there rather than at the first
  segmentation.
- Image viewers have an auto contrast button next to the contrast slider. As
  in Fiji, each click narrows the contrast to the 1st-99th percentiles of the
  pixels in view (the zoomed region) within the current limits, peeling off outliers such as the diverging
  values of a background division; the fourth click restores the full range.


### Changed
- Applying the model-free background correction is faster and lighter. The
  optimal coefficient is found by a binary search on the convex loss instead of
  one evaluation per coefficient, with the same result, and *interpolate NaNs*
  runs once on the background, then on a corrected frame only if NaNs are left
  in it. On 2048 x 2048 frames, a frame takes about 1.5 s instead of 7.5 s, and
  22 s instead of 39 s when every frame still has NaNs to interpolate. The
  corrected stack is held in float32, halving its memory.
### Fixed
- Answering *yes* to "No labels can be found for this position. Do you want to
  annotate from scratch?" wrote the empty labels transposed on any non-square
  image (`shape_x` rows by `shape_y` columns). Every later write into them then
  failed to broadcast: segmenting or thresholding the frame from napari, and
  correcting a mask. The labels now have the shape of the movie.
- Thresholding the frame from napari inside a region of interest raised
  `operands could not be broadcast together` when the labels of the position did
  not have the shape of its movie. The region is now drawn at the shape of the
  image, and a position whose labels do not match it says so and what to do
  about it.
- The threshold configuration wizard kept its histogram — and with it the range
  of the threshold slider — on the channel it opened with: picking another
  channel changed the image but not the intensities the threshold was set on.
  The histogram now follows the viewer, both when the channel changes (the
  threshold is set afresh, the intensity domain being another one) and when the
  frame does (the threshold being tuned is kept).
- A value label sitting over a handle at either end of a range slider — the
  contrast slider of every viewer at its minimum, for one — lost its first digit
  (`).00` for `0.00`): superqt centres the label on its handle, leaving half of
  it outside the widget, where Qt clips it. Such a label is now moved back just
  inside.
- The single-frame segmentation panels of the napari viewer stretched their rows
  apart to fill the height of their tab: the slack now goes to the bottom of the
  panel, as it did before the two panels were put in tabs.
- The model-free background correction preview now subtracts the camera
  offset, as the correction itself does.

### Documentation
- The single-frame segmentation section of *Segment* is written around the two
  tabs of the panel: **Model** for a segmentation model and the new **Threshold**
  for the pipelines of the threshold configuration wizard, with the region of
  interest, the *Replace the labels in the region* and *Also the following
  frames* options, and what is remembered for the experiment. Both figures are
  captured from the current interface — the model tab on the RICM demo, the
  threshold tab on the nuclei of the ADCC demo, inside two regions drawn on the
  frame.
- The threshold wizard guide no longer says a configuration has to be uploaded
  again every session: the last one used for each population is recorded in
  `configs/last_threshold_configs.json` and loaded again with the experiment. It
  also points at the napari route for trying a pipeline on one frame.
- A *Skipped positions* section in *Troubleshooting* lists the reasons a position
  is skipped, as the progress window now names them.
- The viewers of the background correction, the texture, position and spot
  measurements were captured again, the local one showing the clipped contrast
  label.
- Icon names corrected against the code across the guides: the napari view button
  of the **SEGMENT**, **TRACK**, **MEASURE** and **DETECT EVENTS** rows is
  `eye-check-outline` (it was written as `eye`, `eye-outline`, `eye-check` or the
  non-existent `eye-outline-check`), and the button next to Cellpose's
  **diameter** is `image-check`, which opens the stack with a circle of that
  diameter rather than an "Interactive Diameter Estimator".

## [1.6.4] - 2026-09-20

### Added
- Threshold segmentation can be applied from inside the napari correction
  viewer, the counterpart of the frame segmentation plugin for the classical
  pipeline. One or more configurations written by the threshold configuration
  wizard are applied to the frame the time slider is on, either to the whole
  frame or only within regions of interest drawn as napari shapes, so a
  configuration can be tried where it is hard to get right before a full run.
  The configurations are the files the batch pipeline reads, applied the same
  way, and the ones last used for a population are remembered in the
  experiment.

### Changed
- Batch processing starts faster: the heavy imports of a worker are deferred
  or skipped, and IPython is no longer imported in worker processes at all.
- A position skipped for want of its labels, its configuration or its
  segmentation model now says so, naming the reason, instead of failing the
  run with a generic "Process exited unexpectedly."

### Fixed
- Integer sliders showed at most 99 whatever their value (e.g. 300 epochs, a
  max signal length of 128, the time slider of the viewers on movies of more
  than 100 frames): superqt never passed the slider's range to its label. All
  integer sliders now come from `celldetective.gui.base.sliders`, whose label
  follows the range.
- The value labels of the decimal sliders lost their first digit (`).500`,
  `0000` for a threshold of `2.0000`): they are now sized for the decimals
  they show.
- Typing a decimal value with a comma in a slider label no longer raises an
  error.
- The **Min %** / **Max %** fields of the channel normalization (model upload
  and training windows) were too narrow to show `99.99`.
- The viewer opened from the **Fit** background correction listed the channels
  as "Channel 0, 1…" and did not select the channel on display: it now shows
  the channel names, on the displayed channel.
- The **Local** background correction showed a *Downsample* field that it
  ignores; the field is now hidden there.
- The registration ROI viewer showed black images, and the napari frame
  segmentation panel was too large.
- Every labeled float slider raised `AttributeError: 'SliderLabel' object has
  no attribute 'lineEdit'` on superqt 0.7.8 and later, which took the viewers
  and the control panel down with it: superqt turned the slider label from a
  spin box, which owns a line edit, into a line edit. No version is pinned, so
  both are handled.
- The value labels at the ends of a range slider, the contrast slider of every
  viewer among them, were stretched to more than twice their width: they were
  sized for the widest number the field accepts rather than for the value they
  show.
- The segmentation warm-up could deadlock on the import lock when TensorFlow
  and torch were imported from two threads at once.

### Documentation
- New annotated figures, captured from the current interface on the demo
  projects: background correction (local and fit), mask-based, contour,
  texture and position-based measurements, spot detection, the threshold
  configuration wizard, model upload, tracking settings, segmentation and
  event model training, conditional classification, event annotator settings,
  survival and synchronized signal plots. They replace the screenshots of
  earlier versions (some taken on macOS) and illustrate how-to guides that had
  none.
- How-to guides corrected against the code: buttons are named by their icon
  and label (e.g. the settings button of the **MEASURE** row, not a "Measure
  tab"; **Save**, not **Set**), the local correction changes the intensity
  columns instead of adding suffixed ones, contour bands are set in the **Set
  distances** window, *Recompile* reinitializes the weights of a pretrained
  event model, the event annotator's playback speed is set in the viewer, and
  the threshold wizard's filters are removed with the delete button.
- The figure capture scripts (`docs/figures/capture`) also run on Linux/X11,
  rendering each window with an Ubuntu title bar, and record the rectangles
  of the widgets a figure points at.

## [1.6.3] - 2026-09-19

### Fixed
- Threshold segmentation failed on every frame with
  `name 'interpolate_nan' is not defined` whenever the target channel held NaN
  pixels (for instance after a background correction). The import had been
  lost in the v1.5.0 import cleanup; a regression test now covers NaN frames.

## [1.6.2] - 2026-09-19

### Fixed
- The documentation build on Read the Docs failed with Sphinx 9, which
  `sphinx-hoverxref` does not support: the docs build now installs
  `docs/requirements.txt`, which pins Sphinx below 8 without affecting the
  package's own dependencies.

## [1.6.1] - 2026-09-19

### Fixed
- The documentation failed to build on Read the Docs, which no longer offers
  `ubuntu-20.04`: it now builds on `ubuntu-24.04` with Python 3.11.

## [1.6.0] - 2026-09-19

### Added
- **Stack registration** in the Preprocessing module: the drift of each
  position is estimated on one channel by Fourier phase cross-correlation of
  Tukey-windowed frames (sub-pixel, optional downscaling, against the previous
  or the first frame) and applied to every channel. A viewer sets the
  correlation disk and taper on a frame of the current position. The shifts of
  each frame are saved next to the registered stack
  (`<stack>_registration_shifts.csv`) and the step is logged in
  `log_preprocessing.txt`.
- **Configuration editor** rebuilt around the content of `config.ini`: a
  *Settings* tab (one form per section), a *Well labels* table (one row per
  well, labels can be added, renamed and removed, blocks pasted from a
  spreadsheet) and a *Metadata* table. Saving reloads the configuration in the
  control panel.
- **Help panels**: the help buttons open a single window that asks its
  questions in turn, keeps the answers in view, can go back, and ends on a
  suggestion with a link to the matching page of the documentation.
- **Table Explorer**: a header with the size of the table and a strip of tools
  (plot, distributions and statistics, collapse tracks, query, copy, delete,
  save), a line describing the selection, a right-click menu on the column
  headers, copying cells with `Ctrl+C`, and the actions regrouped in *Edit*,
  *Table*, *Math* and *Plot* menus.
- The p-value and effect-size tables explain how to read them: each cell
  compares its row to its column, and mirror cells test the opposite direction.
- Event detection models scan tracks longer than their input with overlapping
  windows and report the earliest detection, instead of failing. Training cuts
  longer annotated signals to the model length.
- A button restores the cell size a segmentation model was trained on in the
  model parameter dialog.
- The napari annotation corrector (*Plugins > Correct a segmentation
  annotation*) offers the single-frame segmentation panel too, reading the
  channels from the annotation's sidecar.
- Segment a single frame straight from the napari correction viewer: a
  `FrameSegmentationPanel` docked on the right offers model selection, one
  dropdown per model input slot, and the inference parameters that model type
  actually takes. The run happens on a worker thread, with progress indication
  and cancellation, and writes into the `segmentation` layer only - nothing
  reaches disk until the labels are saved, and nothing is written back to
  `config_input.json`.
- `prepare_segmentation_model()` and `segment_frame()` split the setup and
  per-frame halves of `segment()`, so one loaded model can be reused across
  frames that do not arrive as a single stack.
- `segment()` gained `selected_channels`, `target_cell_size` and `diameter`
  arguments, which override what the model configuration stores, and
  `use_stored_mapping`, which opts into what it stores.
- Training a StarDist model pads images smaller than the patch size instead of
  failing, and deepens the U-Net (up to three levels) when the median object is
  larger than the network's field of view.

### Changed
- **Python 3.9 or newer is required** (`python_requires=">=3.9"`), and btrack
  is pinned to `>=0.7,<0.8`.
- The *spatially* selection mode of the plotting windows is only offered when
  the experiment metadata holds stage coordinates for the positions shown.
- New visual style across the interface: buttons, checkboxes, collapsible
  blocks, menus, progress bars, tooltips (only shown on items too long to be
  read) and matplotlib figures.
- The splash screen reports what is loading and closes if the start-up fails.
- `class_id` (the mask label) is never interpolated by the track
  post-processing, so no mask is invented for interpolated positions.
- Tracks left without any mask are dropped when the napari track viewer opens
  and on export.
- **`segment()` can now honour two settings it used to ignore**, matching what
  `SegmentCellDLProcess` has always done, so the library and the pipeline return
  the same masks for the same model: the `selected_channels` mapping stored in
  the model's `config_input.json`, and the `cell_size_um` /
  `target_cell_size_um` rescaling. Both live in the model directory, which is
  installed once and shared by every experiment, so applying them by default
  would let a mapping saved while working on one experiment change the result of
  a `segment()` call made for another. They are opt-in: pass
  `use_stored_mapping=True` for pipeline parity, or `selected_channels` /
  `target_cell_size` to set them explicitly. Existing calls are unaffected.
- `CUDA_VISIBLE_DEVICES` is now set only while a segmentation model is being
  built or run, and restored afterwards, instead of being left pinned for the
  lifetime of the process. `segment_frame()` re-applies the device choice its
  model was prepared with, so `use_gpu=False` holds even if the framework defers
  creating its device context until the first prediction.

### Fixed
- StarDist segmentation no longer hangs at 0 % on large or elongated frames:
  frames up to 12 MP are predicted in a single pass, and larger ones get their
  tile overlap computed analytically instead of by StarDist's receptive-field
  probe. Transfer learning from a large-grid model no longer hangs either.
  Inference now rescales then normalizes, like training.
- The installed `celldetective` command failed with `AttributeError`; only
  `python -m celldetective` worked. Package data is now located through
  `importlib.resources`.
- Closing a window while one of its loaders was still running could crash the
  software with an access violation: running threads are now kept alive and
  stopped cleanly on every teardown path.
- An error on one reference cell no longer discards every pair measurement of
  the position.
- The model parameter dialog refuses a negative cell size, and a non-positive
  target size saved by an earlier build is ignored with a warning instead of
  mirroring or failing the run.
- `Calibrate...` in the Table Explorer no longer takes the `Ctrl+C` shortcut,
  which made the cells of the table impossible to copy.
- The "Read the tutorial" links of the classification and experiment-structure
  helpers pointed to pages that do not exist.
- Cellpose models could not be loaded at all on Windows: the model name was
  derived from the model path with `split("/")[-2]`, which raises `IndexError`
  on an `os.sep`-joined path.
- A channel mapping that feeds the same experiment channel into several of a
  model's input slots no longer leaves all but the first slot black. Channel
  matching in `segment()` is also consistently case-insensitive now, instead of
  resolving indices case-insensitively but transferring pixels case-sensitively.
- Listing the segmentation models no longer deletes model directories or creates
  category directories as a side effect when it is only being asked what is
  available.
- `segment()` no longer crashes when the model cannot be located; it logs and
  returns None.
- Labels written into the napari viewer are refused rather than silently wrapped
  round when they do not fit the layer's integer type, which used to alias a new
  cell onto an existing one.
- Segmenting a frame from napari can be undone with `Ctrl+Z` like any other edit,
  and reports the number of objects rather than the highest label value.
- Declining napari's "are you sure you want to close?" prompt no longer takes the
  frame-segmentation panel down with it. The panel used to tear itself down as
  soon as the close was *attempted*, leaving a viewer that stayed open with a
  vanished, permanently inert panel.
- A model whose download was interrupted no longer shadows the copy on Zenodo
  forever. `locate_segmentation_model()` skips a local directory that has no
  `config_input.json`, so the next run re-downloads it instead of failing with
  "could not be loaded" every time.
- The frame-segmentation panel builds its channel and parameter rows after a
  model is fetched on its first run, instead of leaving the "not downloaded yet"
  placeholder up until the model dropdown is cycled.
- A cell size of zero is rejected with a message naming the field, rather than
  failing the run with `float division by zero`.
- `prepare_segmentation_model()` no longer raises `KeyError` on a Cellpose
  `config_input.json` that omits `diameter`, `cellprob_threshold` or
  `flow_threshold` when the caller passes those values explicitly.
- A cell size set in the model parameter dialog now reaches the full-position
  run for a generalist Cellpose model too. The pipeline read the trained size
  from a `cell_size_um` key those models do not carry, so the setting rescaled
  the napari preview and `segment()` while the run it was previewing ignored it.
  All three now go through `trained_cell_size_um()`.
- Closing the napari viewer during a segmentation run no longer strands the
  worker thread. Detaching the panel's slots cleared the whole `finished`
  signal, including the worker's own cleanup, so a run still in flight kept its
  thread, the image stack and the loaded network alive for the rest of the
  session.

### Documentation
- New guide to register stacks in the software; the Preprocessing page presents
  both in-software and Fiji registration.
- Guides updated for the configuration editor, the Table Explorer menus and
  tools, the reading of the statistical tables, the napari single-frame
  segmentation and annotation corrector, the help panels and sliding-window
  event detection.
- The annotated figures are now SVG files built from scripted captures of the
  software (`docs/figures`), so they can be rebuilt when the interface changes.

## [1.5.3] - 2026-06-04

This release focuses entirely on window sizing and multi-monitor / high-DPI
display handling. Previously, windows were sized against the *primary* screen
and locked with fixed sizes, which left dialogs and panels mis-sized,
off-screen, or with inaccessible buttons when the app was opened on a secondary
or scaled display.

### Added
- New `get_current_screen_geometry()` helper that returns the available
  geometry of the monitor where the cursor (or the window) actually is, rather
  than always using the primary screen.
- High-DPI support: enabled `AA_EnableHighDpiScaling` and `AA_UseHighDpiPixmaps`
  at startup, so the interface and icons scale correctly on high-DPI / 4K
  monitors.

### Changed
- All screen-size lookups now route through `get_current_screen_geometry()`
  across the app: init window, Control Panel, dynamic progress dialog, settings
  panels, pivot table view, and the new-experiment configuration dialog.
- The Control Panel's `screen_width` / `screen_height` are now live properties
  that re-evaluate against the current screen, so collapsing/expanding panels
  recomputes against the correct monitor.
- Windows no longer use `setFixedSize`. They now use sensible minimum sizes plus
  a resize to a fraction of the current screen, so they remain resizable and
  adapt to the active display.
- The Control Panel height is capped at 90% of the current screen, and the cap
  is re-applied dynamically when the process, preprocessing, and
  interactions/neighborhood blocks are collapsed/expanded — keeping action
  buttons reachable on small or scaled screens.
- Annotators and the threshold configuration wizard (`BaseAnnotator`,
  `PairEventAnnotator`, `ThresholdConfigWizard`) now use fixed minimum sizes
  (e.g. 800×600) and resize to 80% of the current screen instead of forcing an
  80%-of-primary-screen minimum.
- The new-experiment condition-labels dialog uses a fixed minimum width and
  resizes to a screen-relative width.
- The init window now uses `setMinimumSize(sizeHint())` instead of a fixed size,
  allowing it to grow when needed.

## [1.5.2] - 2026-05-03

This release adds several new cell measurements, rewrites the custom-measurement
API documentation, and includes a large robustness pass (the "Claude audit"):
silent failures were removed, `print` calls converted to proper logging,
`assert`s replaced with explicit exceptions, and numerous edge cases hardened.

### Added
- New shape measurements: `circularity` (4π·area / perimeter²) and
  `aspect_ratio` (major axis / minor axis), both produced as single scalar
  columns with no channel suffix.
- New per-channel intensity distribution moments: `intensity_skewness` and
  `intensity_kurtosis` (Fisher / excess).
- Mask contact-site intensity measurement and accompanying tests.
- Membrane-to-cytoplasm (M/C) intensity ratio measurement.
- Custom-measurement API now supports three signatures (shape-only,
  per-channel intensity, and single-`target_channel` intensity), documented in
  the "write a custom measurement" guide.
- New `regionprops` reference page in the documentation.
- New test suites: contact-site intensity and extra-properties (including the
  new measurement GUI).

### Changed
- Centralized `COLUMN_LABELS` and aligned feature definitions with the code.
- Converted `print` statements to structured logging and replaced `assert`s
  with explicit, descriptive exceptions across the codebase.
- Replaced `os.abort` with `sys.exit` for clean shutdown.
- btrack pinned to `>=0.5.13,<0.8`, and btrack logs are now surfaced to the
  user.

### Fixed
- Relative-measurements indexing bug, plus bugs in pair, intersection, and
  contact-site measurements.
- Center-of-mass displacement calculation.
- Neighborhood column detection.
- `StackVisualizer` threading regression and intertwined-animation bug when
  resizing the event-annotation window.
- Channel-index handling and a Windows access-violation error.
- Robustness safety nets: torch unavailable, empty channel list, imageio
  versions that mishandle array-type frame indices, and uninitialized
  variables.

## [1.5.1] - 2026-03-28

Adds a suite of interactive table-derived plots and binned-data visualization,
plus a round of bug fixes.

### Added
- **Histogram in the measurement annotator** for quick data inspection.
- **Binned data and binned plots** — group data into bins with dedicated plot
  support for visualizing binned distributions.
- **Parallel-coordinates plot** for multi-dimensional data exploration in the
  table UI.
- **Correlation plot** — correlation matrix for pairwise relationships between
  measurements.
- **Interactive plots from the table** — table-derived plots are now
  interactive, and statistical results (p-values, effect sizes) can be
  exported.
- **Visual selector for plots** with improved colormap behavior.
- **More intuitive 2D plotting** — improved histogram and KDE interactions in
  the table UI.
- Settings windows now enforce a minimum width to prevent layout issues on
  small screens.

### Fixed
- Empty-dataframe crash when no cells are detected on a given frame.
- Styling inconsistencies in the correlation-matrix and parallel-coordinates
  windows.
- Empty-table bug when opening neighborhood settings ([#21]).
- Cellpose model-parameter dialog bug ([#22]).
- Measurement bug ([#23]).
- Empty-table handling, via a shared helper function deployed across all
  relevant widgets.
- Crash when opening the pair annotator on untracked data.
- Diameter type-coercion bug (int/float).
- Classifier-widget behavior.
- Tracking bug.
- Linux floating-point precision bug.
- `groupby` bug on track collapse.
- Progress-bar time estimation for reversed-frame segmentation.

### Documentation
- Switched to a card-based documentation portal.
- Renamed the "Pair Event Annotator" to **Interaction Annotator** and the
  "Static Measurements Annotator" to **Phenotype Annotator**.
- Documented the multi-threshold segmentation pipeline and the OR operation
  for mask union.

## [1.5.0] - 2026-02-12

A major feature release: progress feedback throughout the pipeline, optional
heavy dependencies, faster startup, a reorganized processing UI, and a large
documentation and test overhaul.

### Added
- **Progress bars across the pipeline:** neighborhood computation, background
  correction and preprocessing, tracking, and downloads (demos / models /
  datasets), plus a generic download progress window and an advanced
  stack-loader bar.
- **Live training feedback:** training loss and plots shown directly in the
  progress popup.
- **Viewer improvements:** framerate slider, next/previous-frame buttons,
  interactive time-series integration in the event viewer, and an improved spot
  detection viewer.
- Edit the config file directly from within Celldetective.
- Non-blocking napari with a load progress dialog for annotation correction,
  and stylized napari buttons.
- Freeze U-Net layers for more efficient transfer learning; model-fit preview
  on a single image.
- Recent-projects list capped at the last 10.

### Changed
- **Optional heavy dependencies:** TensorFlow, StarDist, and Cellpose are now
  optional, with graceful fallbacks and partial-install support; clearer
  messages when StarDist/torch are missing.
- **Faster startup** via lazy TensorFlow imports and lightened startup-window
  imports.
- Split the monolithic `process_block` into separate `interactions_block`,
  `preprocessing_block`, and `process_block` panels.
- Major refactor of the measure annotator and edge/stack visualizers; renamed
  the Cellpose/StarDist submodules to reduce confusion.
- Migrated `|` type hints to `Union` and added function typing throughout, with
  auto-generated typing from docstrings in the docs.

### Fixed
- Interactions-table and contour-viewer (incl. local correction) bugs.
- Spot-detection measurement bugs and channel-offset viewer bugs.
- GPU selected while torch is not configured for it.
- Tracking: interpolate time gaps, group-by tracks, and progress when masks are
  missing/empty.
- Circular-import issues with napari and TableUI.
- Keras-version MSE bug and a Windows-specific pretrained-event-model bug;
  Zenodo download path issue.

### Tests / CI
- Large expansion of the GUI and unit test suites (table UI, thresholds,
  survival UI, viewers, segmentation model loader, settings panels, partial
  install, image formats, and more).
- Resolved Windows access-violation, hanging, and stalling test issues; build
  the package on tag and fix the PyPI workflow.

[1.6.4]: https://github.com/celldetective/celldetective/compare/v1.6.3...v1.6.4
[1.6.3]: https://github.com/celldetective/celldetective/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/celldetective/celldetective/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/celldetective/celldetective/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/celldetective/celldetective/compare/v1.5.3...v1.6.0
[1.5.3]: https://github.com/celldetective/celldetective/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/celldetective/celldetective/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/celldetective/celldetective/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/celldetective/celldetective/compare/v1.4.3...v1.5.0
[#21]: https://github.com/celldetective/celldetective/issues/21
[#22]: https://github.com/celldetective/celldetective/issues/22
[#23]: https://github.com/celldetective/celldetective/issues/23
