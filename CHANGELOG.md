# Changelog

All notable changes to Celldetective are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Changed
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

[1.5.3]: https://github.com/celldetective/celldetective/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/celldetective/celldetective/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/celldetective/celldetective/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/celldetective/celldetective/compare/v1.4.3...v1.5.0
[#21]: https://github.com/celldetective/celldetective/issues/21
[#22]: https://github.com/celldetective/celldetective/issues/22
[#23]: https://github.com/celldetective/celldetective/issues/23
