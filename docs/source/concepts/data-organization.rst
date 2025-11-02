Data organization
=================

.. _data-organization:

**Reference keys:** :term:`experiment project`, :term:`well`, :term:`position`

We designed a software that structures experimental data into nested :term:`well` and :term:`position` folders, mimicking the spatial segregation in a multi-well plate. The :term:`well` partitioning allows experimentalists to test in parallel multiple biological conditions, such as different cell types, drugs or antibodies at different concentrations, pre-treatments on the cells or surfaces and so on.

.. figure:: ../_static/glass-slide.png
    :align: center
    :alt: exp_folder_mimics_glass_slide

    The experiment folder mimics the organization of the glass slide into wells and fields of view within wells.

Since cells are microscopic objects observed under high magnification, capturing all cells within a :term:`well` in a single image is rarely feasible. Instead, experimentalists typically select multiple imaging **positions** (or fields of view) within the :term:`well`, aiming for a representative sample of the entire :term:`well`.

In Celldetective, single-cell detection is performed independently for each :term:`position`. The software allows looping through multiple :term:`positions <position>` or :term:`wells <well>`, enabling streamlined analysis. Higher-level representations, such as population responses, can aggregate single-cell data from all :term:`positions <position>` within a :term:`well` to provide a comprehensive overview.

A **Celldetective experiment** (:term:`experiment project`) consists of a folder and a configuration file in ``.ini`` format. The folder is organized hierarchically to support data from multiple :term:`wells <well>` and :term:`positions <position>`:

#. **Experiment folder**: Contains individual :term:`well` folders (one per :term:`well`) and the configuration file.
#. **Well folder**: Includes subfolders corresponding to :term:`positions <position>` within that :term:`well`.
#. **Position folder**: Contains a single ``movie/`` subfolder where the user drops the stack associated with that :term:`position`.

In Celldetective, "processing a movie" is synonymous with processing a :term:`position`. This approach standardizes workflows and ensures clear data organization.

Biological conditions and experimental metadata can be injected into the single-cell data by modifying the experiment configuration file. The items of the ``Labels`` section are structured to given a unique information per well. In practise, whenever you have information that is different between at least two wells, use this section. The ``Metadata`` section, on the other hand, can be used to add some global variables to the single cell data (antibody concentration units, temperature and CO2 conditions of the experiment, etc).