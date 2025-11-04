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

    experiment project
        A **Celldetective experiment** consists of a folder and a configuration file in ``.ini`` format. The folder is organized hierarchically to support data from multiple wells and positions:

        #. **Experiment folder**: Contains individual well folders (one per well) and the configuration file.
        #. **Well folder**: Includes subfolders corresponding to position within that well.
        #. **Position folder**: Contains a single ``movie/`` subfolder where the user drops the stack associated with that position.

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