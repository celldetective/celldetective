Settings & Parameters
=====================

This reference page lists the configuration parameters for various Celldetective modules.

.. _ref_segmentation_settings:

Segmentation Data Import
------------------------

These parameters appear in the **Upload Model** window when importing a pretrained model.

**General Settings (All Models)**

*   **Input spatial calibration**: The pixel resolution (in microns) of the images the model was *trained on*.
*   **Channel Mapping**: Map the model's expected inputs (e.g., "Channel 1", "Cyto", "Nuclei") to your experiment's channels. Select ``--`` to ignore.
*   **Normalization**:
    *   **Mode**: Check for percentile-based standard scaling (0-1). Uncheck for raw values.
    *   **Clip**: Check to clip values outside the chosen percentile range.
    *   **Range**: Min/max percentiles for normalization (e.g., 1.0 - 99.8).

**Cellpose Specifics**

*   **Cell Diameter [px]**: The average object diameter in the training data. If set to 30.0 (default), Cellpose assumes standard scaling.
*   **Cellprob Threshold**: Threshold for the confidence map (default 0.0). Lower values increase sensitivity.
*   **Flow Threshold**: Threshold for flow error (default 0.4). Lower values enforce stricter shapes.

.. _ref_runtime_segmentation_settings:

Segmentation Runtime Settings
-----------------------------

These parameters appear when applying a **generalist** model.

**StarDist (Generalist)**

*   **Channel Selection**: Map specific experiment channels (e.g., Nuclei) to the model's input.

**Cellpose (Generalist)**

*   **Channel Mapping**: Select "Cytoplasm" and "Nuclei" channels.
*   **Diameter [px]**: Expected cell diameter. Use the **eye icon** to open the *Interactive Diameter Estimator*.
*   **Flow/Cellprob Thresholds**: Adjust detection sensitivity and shape constraints on the fly.

.. _ref_tracking_settings:

Tracking Settings
-----------------

Accessible via the **Settings** button in the Tracking module.

**Trackers**

*   **bTrack**: Bayesian tracker using Kalman filters and visual features.
*   **trackpy**: Particle tracker based on Crocker-Grier.
    *   **Search range [px]**: Max movement distance per frame.
    *   **Memory [frames]**: Max frames a particle can disappear.

**Feature Extraction**

**Feature Extraction**
 
 *   **Morphological & Intensity**:
     *   **Standard**: ``area``, ``eccentricity``, ``solidity``, ``perimeter``, ``intensity_mean``, ``intensity_max``, ``intensity_min``, etc.
     *   **Advanced**: ``major_axis_length``, ``minor_axis_length``, ``orientation``, ``extent``, ``euler_number``, ``feret_diameter_max``.
     *   **Custom**: Any allowed function from ``skimage.measure.regionprops``.
 *   **Haralick Texture Features**:
     *   **Target channel**: Channel to analyze (must be one of the loaded channels).
     *   **Distance**: Pixel distance for GLCM calculation (default 1).
     *   **# gray levels**: Number of intensity bins for quantization (default 256).
     *   **Scale**: Downscaling factor (0-1) to speed up computation.
     *   **Normalization**:
         *   **Percentile Mode**: Normalize intensities between min/max percentiles (e.g., 1% - 99.9%).
         *   **Absolute Mode**: Normalize intensities between fixed pixel values.
 
 **Post-Processing**
 
 *   **Min. tracklength**: Filter out tracks shorter than this number of frames.
 *   **Remove tracks... (Start)**: Remove tracks that do not start at the first frame.
 *   **Remove tracks... (End)**: Remove tracks that do not end at the last frame.
 *   **Interpolate gaps**: Fill missing detections (gaps) within a track using linear interpolation.
 *   **Extrapolate (Pre)**: Sustain the first detection's position backwards to the start of the movie.
 *   **Extrapolate (Post)**: Sustain the last detection's position forwards to the end of the movie.

.. _ref_neighborhood_settings:

Neighborhood Measurement Settings
---------------------------------

Accessible when selecting **Neighborhood** in Measurements.

**Population Configuration**

*   **Reference / Neighbor**: Select populations.
*   **Filters**:
    *   **Status**: Restrict to specific cell states (e.g., "Alive").
    *   **Time Ref**: Condition on an event time column.
*   **Cumulated Presence**: Compute total duration of neighbor presence.

**Measurement Types**

*   **Distance Threshold**: Radial proximity.
    *   **Radius [px]**: Distance threshold.
*   **Mask Contact**: Boundary proximity.
    *   **Edge Proximity**: Pixel distance between boundaries.

.. _ref_survival_settings:

Survival Analysis Settings
--------------------------

Accessible via **Analyze > Plot Survival**.

**Data Selection**

*   **Population**: Target cell population.
*   **Time of Reference**: Start point (:math:`T=0`, e.g., ``t_firstdetection``).
*   **Time of Interest**: End event (e.g., ``t_death``).

**Filtering**

*   **Query**: Pandas query string helper (e.g., ``TRACK_ID > 10``).
*   **Cut obs. time [min]**: Censoring threshold.

**Visualization**

*   **Time calibration**: Frames-to-minutes conversion.
*   **Cmap**: Colormap for curves.

.. _ref_experiment_config:

Experiment Configuration (config.ini)
-------------------------------------

These tags define the structure of your experiment in the ``config.ini`` file.

*   ``[Populations]``: Comma-separated list of cell populations (e.g., ``nk,rbc``).
*   ``[MovieSettings]``: Image acquisition parameters.
    *   ``pxtoum``: Pixel size in microns.
    *   ``frametomin``: Frame interval in minutes.
    *   ``movie_prefix``: Filename prefix for raw images.
*   ``[Channels]``: Names and order of channels (e.g., ``brightfield_channel = 0``).
*   ``[Labels]``: Experimental conditions per well (e.g., ``concentrations = 0,100``).
*   ``[Metadata]``: Additional experiment metadata.

.. _ref_preprocessing_settings:

Preprocessing Protocols
-----------------------

Accessible via the **Preprocessing** module.

**Background Correction**

*   **Model Fit**: Fits a 2D surface (plane/paraboloid) to the background.
    *   **Threshold**: Standard deviation threshold to exclude cells from fit.
*   **Model Free**: Computes a median background image from multiple positions or timeframes.
    *   **Tiles**: Use all frames/tiles to estimate background (best for time-lapse).

**Channel Offset**

*   **X/Y Offset**: Pixel shift to align channels.
*   **Viewer**: Use the arrow keys to visually align the overlay channel (blue) with reference (gray).

.. _ref_signal_settings:

Signal Analysis
---------------

**Signal Mapping**

Configuration window for Deep Learning signal models.

*   **Required Inputs** (Left): The specific signals expected by the model (e.g., "Nuclei Intensity").
*   **Available Columns** (Right): The columns from your measurement table to map to these inputs.

.. _ref_event_annotation_settings:

Event Annotation
----------------

Configuration for the Single Cell Signal Annotator.

*   **Image Mode**:
    *   **Grayscale**: Single channel visualization.
    *   **Composite**: RGB overlay (requires channel selection and per-channel normalization).
*   **Rescaling**: Downscaling fraction (e.g., 0.5) to reduce memory usage during animation.
*   **Time Interval**: Playback speed (milliseconds between frames).
