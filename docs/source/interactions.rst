Interactions
============

.. _interactions:

Prerequisites
-------------

You must perform the segmentation and measurements of the cell populations for which you want to compute the neighborhood (at least one).


Neighborhood
------------

Neighborhood analysis quantifies the spatial relationships between two cell populations — essential for studying cell-cell interactions (e.g., immune cell targeting, tissue organization). It is configured in the dedicated **Interactions** section of the control panel, where you define reference and neighbor populations and choose a detection method (Distance Threshold or Mask Contact).
    
.. seealso::
    :doc:`how-to-guides/basics/measure-cell-interactions` for a step-by-step configuration guide. |
    :ref:`Neighborhood Measurement Settings <ref_neighborhood_settings>` for parameter details.


Pair measurements
-----------------

If the **MEASURE PAIRS** option is selected, all computed neighborhoods are detected automatically from the pickle files for both target and effector cell populations. All pairs existing at least once, at one timepoint, are identified. The complete signals (before/after entering the neighborhood) are recovered for the two cells of interest. Several quantities are computed (relative distance, velocity, angle, angular velocity, in-neighborhood or not). 

If the center of mass displacements were computed for the neighbor population, an additional pair measurement computed automatically is the scalar product between the center of mass displacement vector on the neighbor cell and the cell-pair vector, as well as the cosine of the angle between the two vectors. 

A unique pair is identified by four columns: ``REFERENCE_ID``, ``NEIGHBOR_ID``, ``reference_population``, and ``neighbor_population``. The pair measurements can be explored in the Table Explorer, as well as in a dedicated signal annotator viewer.


Pair signal viewer
------------------

The **Pair Signal Viewer** allows you to inspect the spatial and temporal dynamics of interacting cells. It visualizes the reference cell, the neighbor cell, and the connecting segment simultaneously.

.. seealso::
    :doc:`how-to-guides/basics/visualize-cell-interactions` for a guide on using the viewer.

