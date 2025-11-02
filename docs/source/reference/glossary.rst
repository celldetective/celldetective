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
        The highest organizational level in Celldetective, consisting of a main folder and a configuration file, containing all experimental data.

    well
        A folder within the :term:`experiment project` that groups data from one physical well on a multi-well plate, typically corresponding to a single biological condition.

    position
        A subfolder within a :term:`well` folder, representing a single field of view (microscopy image stack or "movie") taken within that well.