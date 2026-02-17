Settings & Parameters
=====================

This reference page lists the configuration parameters for various Celldetective modules.

.. _ref_segmentation_settings:

Segmentation Data Import
------------------------

These parameters appear in the **Upload Model** window when importing a pretrained model.

**General Settings (All Models)**

*   :term:`Input spatial calibration`: The pixel resolution (in microns) of the images the model was *trained on*.
*   :term:`Channel Mapping`: Map the model's expected inputs (e.g., "Channel 1", "Cyto", "Nuclei") to your experiment's channels. Select ``--`` to ignore.
*   :term:`Normalization`:

    *   **Mode**: Check for percentile-based standard scaling (0-1). Uncheck for raw values.
    *   **Clip**: Check to clip values outside the chosen percentile range.
    *   **Range**: Min/max percentiles for normalization (e.g., 1.0 - 99.8).

**Cellpose Specifics**

*   :term:`Cell Diameter` [px]: The average object diameter in the training data. If set to 30.0 (default), Cellpose assumes standard scaling.
*   :term:`Cellprob Threshold`: Threshold for the confidence map (default 0.0). Lower values increase sensitivity.
*   :term:`Flow Threshold`: Threshold for flow error (default 0.4). Lower values enforce stricter shapes.

.. _ref_runtime_segmentation_settings:

Segmentation Runtime Settings
-----------------------------

These parameters appear when applying a **generalist** model.

:term:`StarDist` (Generalist)

*   **Channel Selection**: Map specific experiment channels (e.g., Nuclei) to the model's input.

:term:`Cellpose` (Generalist)

*   **Channel Mapping**: Select "Cytoplasm" and "Nuclei" channels.
*   :term:`Diameter` [px]: Expected cell diameter. Use the :icon:`eye,black` button to open the *Interactive Diameter Estimator*.
*   **Flow/Cellprob Thresholds**: Adjust detection sensitivity and shape constraints on the fly.

.. _ref_tracking_settings:

Tracking Settings
-----------------

Accessible via the :icon:`cog-outline,black` button in the Tracking module.

**Trackers**

*   :term:`bTrack`: Bayesian tracker using Kalman filters and visual features.
*   :term:`trackpy`: Particle tracker based on Crocker-Grier.

    *   :term:`Search range` [px]: Max movement distance per frame.
    *   :term:`Memory` [frames]: Max frames a particle can disappear.

**Feature Extraction**
 
*   :term:`Morphological features <Morphological features>` & Intensity:

    *   **Standard**: ``area``, ``eccentricity``, ``solidity``, ``perimeter``, ``intensity_mean``, ``intensity_max``, ``intensity_min``, etc.
    *   **Advanced**: ``major_axis_length``, ``minor_axis_length``, ``orientation``, ``extent``, ``euler_number``, ``feret_diameter_max``.
    *   **Custom**: Any allowed function from ``skimage.measure.regionprops``.

*   :term:`Haralick Texture Features`:

    *   **Target channel**: Channel to analyze (must be one of the loaded channels).
    *   **Distance**: Pixel distance for GLCM calculation (default 1).
    *   **# gray levels**: Number of intensity bins for quantization (default 256).
    *   **Scale**: Downscaling factor (0-1) to speed up computation.
    *   **Normalization**:

        *   **Percentile Mode**: Normalize intensities between min/max percentiles (e.g., 1% - 99.9%).
        *   **Absolute Mode**: Normalize intensities between fixed pixel values.
 
**Post-Processing**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Setting
     - Description
   * - :term:`Min. tracklength <Track length>`
     - Filter out tracks shorter than this number of frames.
   * - **Remove tracks... (Start)**
     - Remove tracks that do not start at the first frame.
   * - **Remove tracks... (End)**
     - Remove tracks that do not end at the last frame.
   * - :term:`Interpolate gaps`
     - Fill missing detections (gaps) within a track using linear interpolation.
   * - :term:`Extrapolate` (Pre)
     - Sustain the first detection's position backwards to the start of the movie.
   * - :term:`Extrapolate` (Post)
     - Sustain the last detection's position forwards to the end of the movie.

.. _ref_neighborhood_settings:

Neighborhood Measurement Settings
---------------------------------

Accessible when selecting **Neighborhood** in Measurements.

**Population Configuration**

*   :term:`Reference <Reference population>` / :term:`Neighbor <Neighbor population>`: Select the two populations to analyze (can be the same for self-neighborhood).
*   **Filters**:

    *   **Status**: Restrict analysis to cells with a specific status (e.g., "Alive", "Positive").
    *   **Not**: Check the **"Not"** button (:icon:`alert-circle-outline,black`) to invert the status selection (e.g., Select "Alive" and check "Not" to target "Dead" cells).
    *   **Event Time**: Correlate measurements with a specific event (e.g., ``t_death``). This creates event-aligned neighborhood metrics.

*   :term:`Cumulated Presence`: If checked, computes the total duration (in frames or time) that a neighbor has been present within the defined threshold.

**Measurement Types**

*   **Distance Threshold**: Detects neighbors within a fixed radial distance from the cell centroid.

    *   **Distance [px]**: The radius of the neighborhood circle. Can add multiple distances.

*   :term:`Mask Contact`: Detects neighbors whose boundaries are within a specific proximity.

    *   **Distance [px]**: The maximum distance between cell boundaries to be considered "in contact" (often 0 for touching or small positive value for near-contact).

**General Options**

*   **Clear Previous**: If checked, removes all previously computed neighborhood columns from the data tables before saving new ones. Essential when re-running analysis with different parameters to avoid clutter.

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

.. _ref_single_cell_measurements:

Single Cell Measurements
------------------------

Accessible via the **Analyze > Measure** tab.

**Isotropic Measurements**

Measurements taken within circular or ring-shaped ROIs centered on the cell.

*   **Radii [px]**: List of radii (e.g., ``10``) or rings (e.g., ``10-20``) defining the ROIs.
*   **Operations**: Statistical operations to perform within the ROI (``mean``, ``std``, ``sum``, ``median``, ``min``, ``max``).

**Contour Measurements**

Measurements taken within a band relative to the cell boundary.

*   **Distances [px]**: List of distances from the mask edge. Positive values are inside (erosion), negative values are outside (dilation). Pairs (e.g., ``(0, 5)``) define a band.

**Spot Detection**

Detection of intracellular spots (e.g., FISH probes) using Laplacian of Gaussian.

*   **Channel**: Target channel for spot detection.
*   **Diameter [px]**: Expected diameter of the spots.
*   **Threshold**: Sensitivity threshold for detection.
*   **Preprocessing**: filters to apply before detection (e.g., ``smooth``, ``denoise``).

.. _ref_segmentation_training:

Segmentation Model Training
---------------------------

Accessible via **Train > Segmentation Model**.

**Model Selection**

*   **Model Type**:

    *   **StarDist**: Best for round/convex objects (nuclei).
    *   **Cellpose**: Best for complex shapes and cytoplasm.

*   **Pretrained Model**: Initialize weights from an existing model (Generic or Custom).
*   **Model Name**: Unique name for the new model.

**Training Data**

*   **Training Data**: Folder containing images and masks (e.g., from an annotated experiment).
*   **Include Dataset**: Select a built-in dataset to augment training.
*   :term:`Augmentation Factor <Augmentation>`: Multiplier for data augmentation (rotation, flip, zoom). Default ``2.0``.
*   :term:`Validation Split`: Fraction of data reserved for validation (e.g., ``0.2``).

**Hyperparameters**

*   :term:`Learning Rate`: Step size for the optimizer (StarDist default: ``0.0003``, Cellpose default: ``0.01``).
*   :term:`Batch Size`: Number of images per training step (default ``8``).
*   :term:`Epochs`: Number of training iterations (StarDist default: ``100``-``500``, Cellpose default: ``100``-``10000``).

.. _ref_experiment_config:

Experiment Configuration (config.ini)
-------------------------------------

The ``config.ini`` file is created automatically when you set up a new experiment
(see :ref:`new-experiment-guide`).
It uses the standard INI format and is located at the root of the experiment folder.
Below is a complete reference of every section and key.

``[Populations]``
~~~~~~~~~~~~~~~~~

Declares which cell populations are included in the experiment.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``populations``
     - string
     - Comma-separated list of population names (e.g. ``targets,effectors``).
       These names match the population folders created inside each position directory.

``[MovieSettings]``
~~~~~~~~~~~~~~~~~~~

Image-acquisition and stack geometry parameters.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``pxtoum``
     - float
     - Spatial calibration: how many micrometres one pixel represents (default ``1.0``).
   * - ``frametomin``
     - float
     - Temporal calibration: the interval in minutes between two consecutive frames
       (default ``1.0``). For single-time-point data, leave at ``1.0``.
   * - ``len_movie``
     - int
     - Number of frames in the movie. Used as a fallback when automatic frame-count
       extraction fails. For variable-length stacks, set a conservative (lower) estimate.
   * - ``movie_prefix``
     - string
     - Filename prefix that stack files must start with to be loaded (e.g. ``Experiment``).
       Leave blank if filenames have no common prefix.
   * - ``shape_x``
     - int
     - Image width in pixels (default ``2048``).
   * - ``shape_y``
     - int
     - Image height in pixels (default ``2048``).

``[Channels]``
~~~~~~~~~~~~~~

Maps channel names to their stack index (0-based).
Each key is a channel name and each value is the integer index of that channel in
the multi-channel stack, or ``nan`` if the channel is not present.

**Example**

.. code-block:: ini

   [Channels]
   brightfield_channel = 0
   adhesion_channel = 1
   fitc_channel = 2
   cy5_channel = nan

Built-in channel names include ``brightfield_channel``, ``live_nuclei_channel``,
``dead_nuclei_channel``, ``effector_fluo_channel``, ``adhesion_channel``,
``fluo_channel_1``, ``fluo_channel_2``.
Custom channel names can be added during experiment creation.

``[Labels]``
~~~~~~~~~~~~

Per-well biological condition labels. Each value is a comma-separated list whose
length equals the number of wells in the experiment.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``cell_types``
     - string
     - Cell type for each well (e.g. ``NK,NK,T-cell,T-cell``).
   * - ``antibodies``
     - string
     - Antibody used in each well (e.g. ``anti-CD4,anti-CD4,none,none``).
   * - ``concentrations``
     - string
     - Antibody or drug concentration for each well (e.g. ``0,100,0,100``).
   * - ``pharmaceutical_agents``
     - string
     - Pharmaceutical agent applied in each well (e.g. ``none,dextran,none,dextran``).
       Fields can be left blank (defaults to well index).

``[Metadata]``
~~~~~~~~~~~~~~

Additional experiment-level metadata.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Key
     - Type
     - Description
   * - ``concentration_units``
     - string
     - Unit for concentration values in ``[Labels]`` (default ``pM``).

**Full example**

.. code-block:: ini

   [Populations]
   populations = targets,effectors

   [MovieSettings]
   pxtoum = 0.325
   frametomin = 3.0
   len_movie = 120
   movie_prefix = Experiment
   shape_x = 2048
   shape_y = 2048

   [Channels]
   brightfield_channel = 0
   adhesion_channel = 1
   fitc_channel = 2

   [Labels]
   cell_types = NK,NK,T-cell,T-cell
   antibodies = anti-CD16,anti-CD16,none,none
   concentrations = 0,100,0,100
   pharmaceutical_agents = none,dextran,none,dextran

   [Metadata]
   concentration_units = pM


.. _ref_preprocessing_settings:

Preprocessing Protocols
-----------------------

Accessible via the **Preprocessing** module.

**General Correction Settings**

*   :term:`Operation`:

    *   **Subtract**: Subtract the estimated background from the image.
    *   **Divide**: Divide the image by the background (flat-field correction).

*   :term:`Clip`: (Subtract mode only) Clip negative values to zero after subtraction.
*   :term:`Offset`: Camera black level/offset. Subtracted prior to background estimation.
*   :term:`Interpolate NaNs`: Fill missing or NaN pixels using neighboring values.

**Background Correction**

*   **Model Fit**: Fits a 2D surface (plane/paraboloid) to the background.

    *   **Model type**: ``paraboloid`` (best for curved illumination) or ``plane`` (best for simple gradients).
    *   **Threshold**: Standard deviation threshold to exclude cells/objects from the fit.
    *   **Downsample**: Factor to downsample images for faster surface fitting (default: 10).

*   **Model Free**: Computes a median background image from multiple positions or timeframes.
    
    *   **Stack mode**:

        *   ``timeseries``: Estimates background from a range of frames in the current position.
        *   ``tiles``: Estimates background across all positions/tiles (best for global background).
    
    *   **Time range**: Specific frames to use for estimation (only in ``timeseries`` mode).
    *   **Threshold**: Standard deviation threshold to mask cells during estimation.
    *   **Optimization**:

        *   **Optimize for each frame**: If checked, performs a linear regression to adjust the background level per-frame.
        *   **Coef. range**: Range of scaling factors allowed during optimization (e.g., 0.95 - 1.05).
        *   **Nbr of coefs**: Number of values to test within the coefficient range.

**Local Correction**

*   **Distance**: The radial distance (in pixels) from the cell mask boundary used to estimate local background.
*   **Model**: ``mean`` or ``median`` of intensity within the boundary band.

**Channel Offset**

*   **Shift (h)/(v)**: Pixel shift (horizontal and vertical) to align the target channel with the reference.
*   **Viewer**: Use the :icon:`image-check,black` button to open the *Offset Viewer*. Use arrow keys to visually align the channels.

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
