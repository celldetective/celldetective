Data organization
=================

.. _data-organization:

**Reference keys:** :term:`experiment project`, :term:`well`, :term:`position`

Structure
---------

Celldetective structures experimental data into nested **well** and **position** folders, mimicking the spatial segregation of a multi-well plate, often used in biological assays. The **well** partitioning is used by experimentalists to test multiple biological conditions in parallel.

Since resolving single-cells requires high magnification, capturing all cells within a well in a single image is rarely feasible. Instead, experimentalists typically select multiple imaging **positions** (or fields of view) within the well, thus sampling the well.

A direct consequence of this structural organization is that there is a spatial scale relationship between positions, wells and experiment: :math:`\mathcal{A}_\textrm{position} \leq \mathcal{A}_\textrm{well} \leq \mathcal{A}_\textrm{experiment}` , where :math:`\mathcal{A}` is the spatial area. At the same time, a position can be conceptually understood as a repetition of a biological condition, a well the biological condition and the experiment as a collection of biological conditions.

.. figure:: ../_static/glass-slide.png
    :align: center
    :alt: exp_folder_mimics_glass_slide

    The experiment folder mimics the organization of the glass slide into wells and fields of view within wells.

By convention, an experiment should be associated with data produced during a unique, timestamped, experiment. It is technically possible to create **meta-experiments** where positions or wells are pulled from more than one experiment, which is convenient to assess the reproducibility of an outcome.

Processing
----------

In Celldetective, single-cell detection is performed independently for each position. The software allows looping through multiple positions or wells, enabling streamlined analysis. Higher-level representations, such as population responses, can aggregate single-cell data from all positions within a well to provide a comprehensive overview.

In Celldetective, "processing a movie" is synonymous with processing a position. This approach standardizes workflows and ensures clear data organization.

Metadata
--------

Biological conditions and experimental metadata can be injected into the single-cell data by modifying the experiment configuration file. The items of the ``Labels`` section are structured to given a unique information per well. In practise, whenever you have information that is different between at least two wells, use this section. The ``Metadata`` section, on the other hand, can be used to add some global variables to the single cell data (antibody concentration units, temperature and CO:math:`_2` conditions of the experiment, etc).