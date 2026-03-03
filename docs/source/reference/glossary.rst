Glossary
========

This glossary defines the key concepts used in the documentation.

General Terms
-------------

.. glossary::
    :sorted:

    event
        An occurrence that happens to single cell at a specific point in time, such as a state change, an environmental change or the onset of a process.

    event time
        The time point at which a cell is observed to begin or undergo the :term:`event` of interest.

    event class
        The classification of a cell relative to an :term:`event`, categorized as "event," "no event," or "else".

    first detection event
        An automatically detected event marking the first time a cell is observed in the field of view, provided it appears after the first frame and away from the image edges. This effectively identifies cells that "land" or appear during the experiment, excluding those already present or entering from the boundaries.

    cell population
        A group of cells for which a single-cell description is computed. Typically refers to a cell type (e.g., T cells, cancer cells) but can be based on any shared characteristic.

    instance segmentation
        A computer vision task that takes a multidimensional image as input and outputs a labeled image, where each individual object is assigned a unique pixel mask and label.

    neighborhood
        A spatial proximity relationship matching a reference cell and surrounding neighbor cells, determined by a specific method (isotropic or mask-contact) and distance parameter. See :ref:`ref_neighborhood_measurements`.

    neighbor counts
        Metrics quantifying the number of neighboring cells within a defined :term:`neighborhood`, calculated using one of three methods: inclusive, exclusive, or intermediate (weighted).

    threshold-based event detection
        A method for detecting events by applying conditions on cell features. The conditions are evaluated at each time point, producing a binary signal per cell. The time of event is extracted by fitting a model to the binary signal.

    experiment project
        A **Celldetective experiment** is a directory containing all configuration and data for a single experiment.

        .. code-block:: text

            MyExperiment/
            ├── config.ini             # Experiment configuration file
            ├── W1/                    # Well folder 1
            ├── W2/                    # Well folder 2
            └── ...

    well
        A collection of positions sharing the same biological condition, often associated with a physical well from a multi-well plate.

        **Naming Convention**

        *   **Pattern**: ``W{number}`` (e.g., ``W1``, ``W12``).
        *   **Regex**: ``^W\d+/?$``

        **Structure**

        A well folder contains subfolders for each imaging **Position** (Field of View).

        .. code-block:: text

            W1/
            ├── 100/                   # Position folder (0)
            ├── 101/                   # Position folder (1)
            └── ...

    position
        A field of view (microscopy image stack or "movie") taken within a well. Interchangeably the spatial position and the associated image data. The naming convention usually follows ``{WellNumber}0{PositionIndex}``.

        **Contents**

        .. code-block:: text

            100/
            ├── movie/                 # Input image stacks
            │   └── images.tif
            ├── labels_nuclei/         # Segmentation masks (generated)
            ├── output/                # Analysis output
            │   └── tables/            # CSV results
            └── metadata.json          # Position metadata

    characteristic group
        A ensemble of instantaneous cell :term:`phenotypes <phenotype>`.

    phenotype
        A ensemble of cells sharing similar features at time :math:`t`. An ensemble of phenotypes forms a characteristic group.

    single-cell measurement
        A property measured from either an image or dynamical information associated with a unique cell at time :math:`t`. See :ref:`ref_measurements`.

    survival
        Probability that a cell exhibits an event as a function of time :math:`\Delta t_\textrm{event}`.

    background correction
        A process that flattens or removes the non-cell part of an image.

    preprocessing
        Any operation performed on the raw microscopy images before segmentation and measurements.

    alignment
        The correction of microscopy image drifts over time and offsets so that non-moving objects do not appear to be moving up to tracking errors in case of tracking.

    input spatial calibration
        Pixel resolution of the training images (in microns). Used to rescale input images to match the model's expected scale.

    morphological features
        Geometric properties of a segmented object, such as area, perimeter, eccentricity, and solidity. See :ref:`ref_morphological_measurements`.

    texture features
        Texture descriptors calculated from the Gray Level Co-occurrence Matrix (:term:`GLCM`), quantifying properties like contrast, correlation, and homogeneity. See :ref:`ref_texture_measurements`.

    reference population
        The set of cells *for which* neighborhood metrics are computed (the "center" cells).

    neighbor population
        The set of cells *that are counted* around the reference cells.

    cumulated presence
        A metric that sums the duration (in time or frames) that a specific neighbor (or any neighbor) has been present within the neighborhood of a reference cell.

    mask contact
        A neighborhood definition based on the distance between the boundaries (masks) of two cells, rather than their centroids.

    censoring
        In survival analysis, the condition where the event of interest has not occurred by the end of the observation period (monitor/cut-off time).

    isotropic measurements
        Measurements computed within circular or ring-shaped Regions of Interest (:term:`ROI`) centered on a cell.

    contour measurements
        Measurements computed within a band (dilation/erosion) defined relative to the cell's segmentation mask boundary.

    label
        Can refer to:
        1. The unique integer ID assigned to a segmented object in a label image.
        2. A metadata tag associated with an experiment condition (e.g., "Drug A").

    signal
        A quantitative measurement derived from an image channel (e.g., "Mean Intensity", "GFP Fluorescence") associated with a segmented object. See :ref:`ref_intensity_measurements`.

    spreading event
        A cellular event where a lymphocyte (e.g., T-cell) flattens and increases its contact area upon interaction with a stimulating surface acting as a proxy for an antigen-presenting cell.
    
    cell type
        A classification of cells based on their morphology, function, or molecular markers (e.g., HeLa, T-cells).

    antibody
        A protein used to label specific cellular targets in immunofluorescence experiments.

    concentration
        The amount of a substance (e.g., drug, antibody) present in a solution, often expressed in µM, ng/mL, etc.

    pharmaceutical agents
        Drugs or chemical compounds applied to the cells to perturb their behavior or state.

    reference cell
        A single cell belonging to a :term:`reference population`. It acts as the focal point for neighborhood calculations.

    neighbor cell
        A single cell belonging to a :term:`neighbor population`. It is identified based on its proximity or relation to a :term:`reference cell`.


Graphical Tools
---------------

.. glossary::
    :sorted:

    Event Annotator
        An interactive tool for manually annotating cell :term:`events <event>` and their occurrence times (:term:`event time`) in single-cell trajectories, used for characterization and building training sets for event detection models. See :ref:`ref_signal_annotator_shortcuts`.

    Classifier Widget
        An interactive tool for creating custom cell classifications or characteristic groups based on quantitative features. It allows users to filter cells using logical queries (e.g., ``area > 100``), visualize populations in 2D scatter plots, and propagate labels over time to define events or states. See :ref:`ref_classifier_widget`.

    Table Explorer
        A spreadsheet-like interface for viewing and interacting with single-cell measurement data. It supports 1D and 2D plotting, statistical analysis, data aggregation (track collapsing), and file export. See :ref:`ref_table_explorer_menus`.

    Phenotype Annotator
        An interactive tool for manually assigning :term:`phenotypes <phenotype>` (integer labels) to cells within a specific :term:`characteristic group` (attribute column). Unlike the :term:`Event Annotator`, this tool does not require time-lapse movies and can be used on static snapshots or unconnected timepoints. See :ref:`ref_phenotype_annotator`.

    Interaction Annotator
        An interactive tool for annotating interactions between a specific :term:`reference cell` and a :term:`neighbor cell`. Unlike single-cell annotators, it stores event data (e.g., contact timing) in the **pair table**, linking the event to the unique relationship between the two cells rather than their individual tracks. See :ref:`ref_interaction_annotator`.