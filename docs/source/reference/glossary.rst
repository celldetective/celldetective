Glossary
========

This glossary defines the key concepts used in the documentation.


.. glossary::
    :sorted:

    event
        An occurrence that happens to single cell at a specific point in time, such as a state change, an environmental change or the onset of a process.

    event time
        The time point at which a cell is observed to begin or undergo the :term:`event` of interest.

    event class
        The classification of a cell relative to an :term:`event`, categorized as "event," "no event," or "else".

    cell population
        A group of cells for which a single-cell description is computed. Typically refers to a cell type (e.g., T cells, cancer cells) but can be based on any shared characteristic.

    instance segmentation
        A computer vision task that takes a multidimensional image as input and outputs a labeled image, where each individual object is assigned a unique pixel mask and label.

    neighborhood
        A spatial proximity relationship matching a reference cell and surrounding neighbor cells, determined by a specific method (isotropic or mask-contact) and distance parameter.

    neighbor counts
        Metrics quantifying the number of neighboring cells within a defined :term:`neighborhood`, calculated using one of three methods: inclusive, exclusive, or intermediate (weighted).

    threshold-based event detection
        A method for detecting events by applying conditions on cell features. The conditions are evaluated at each time point, producing a binary signal per cell. The time of event is extracted by fitting a model to the binary signal.

    experiment project
        A **Celldetective experiment** consists of a folder and a configuration file in ``.ini`` format. The folder is organized hierarchically to support data from multiple wells and positions:

        #. **Experiment folder**: Contains individual well folders (one per well) and the configuration file.
        #. **Well folder**: Includes subfolders corresponding to position within that well.
        #. **Position folder**: Contains a single ``movie/`` subfolder where the user drops the stack associated with that position.

        See :doc:`project-structure` for detailed directory layout and file naming conventions.

    well
        A collection of positions sharing the same biological condition, often associated with a physical well from a multi-well plate.

    position
        A field of view (microscopy image stack or "movie") taken within a well. Interchangeably the spatial position and the associated image data.

    characteristic group
        A ensemble of instantaneous cell :term:`phenotypes <phenotype>`.

    phenotype
        A ensemble of cells sharing similar features at time :math:`t`. An ensemble of phenotypes forms a characteristic group.

    single-cell measurement
        A property measured from either an image or dynamical information associated with a unique cell at time :math:`t`.

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
        Geometric properties of a segmented object, such as area, perimeter, eccentricity, and solidity.

    texture features
        Texture descriptors calculated from the Gray Level Co-occurrence Matrix (:term:`GLCM`), quantifying properties like contrast, correlation, and homogeneity.

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
        A quantitative measurement derived from an image channel (e.g., "Mean Intensity", "GFP Fluorescence") associated with a segmented object.

    cell type
        A classification of cells based on their morphology, function, or molecular markers (e.g., HeLa, T-cells).

    antibody
        A protein used to label specific cellular targets in immunofluorescence experiments.

    concentration
        The amount of a substance (e.g., drug, antibody) present in a solution, often expressed in µM, ng/mL, etc.

    pharmaceutical agents
        Drugs or chemical compounds applied to the cells to perturb their behavior or state.