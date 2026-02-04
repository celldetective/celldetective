Interactions
============

.. _interactions:

Prerequisites
-------------

You must perform the segmentation and measurements of the cell populations for which you want to compute the neighborhood (at least one).

Neighborhood
------------

Neighborhood analysis is configured in the **Measurements** module, allowing you to define reference and neighbor populations and choose detection methods (Distance Threshold or Mask Contact).
    
For full configuration instructions, see the :doc:`Neighborhood Measurements Guide <measure>`.


Pair measurements
-----------------

If the ``MEASURE PAIRS`` option is selected, all the computed neighborhoods are detected automatically from the pickle files for both target and effector cell populations. All pairs existing at least once, at one timepoint, are identified. The complete signals (before/after entering the neighborhood) are recovered for the two cells of interest. Several quantities are computed (relative distance, velocity, angle, angular velocity, in-neighborhood or not). 

If the center of mass displacements were computed for the neighbor population, an additional pair measurement computed automatically is the scalar product between the center of mass displacement vector on the neighbor cell and the cell-pair vector, as well as the cosine of the angle between the two vectors. 

A unique pair is identified by four columns: a ``REFERENCE_ID`` (the ``TRACK_ID`` or ``ID`` of the reference cell), a ``NEIGHBOR_ID`` (the ``TRACK_ID`` or ``ID`` of the neighbor cell), a ``reference_population`` and a ``neighbor_population`` column. 

The measurements of the pairs can be explored in a table UI like for the trajectory tables, as well as a signal annotator viewer, designed for interactions.

Pair signal analysis
--------------------

The viewer for single-cell signal analysis is revisited here to view simultaneously the reference/neighbor population of interest determined by the user at the top. The color code for each population can be selected independently from the available "class" and "status" attributes of each population. 

By convention, the reference cell has to be clicked first. Upon clicking, all other reference cells are colored in black (hidden), and the pairs around the reference cell of interest are explicitly represented as clickable dynamic segments. If you click on the symbol at the center of the segment or directly on the neighbor cell, the pair is selected and can be annotated. 

The signals of the reference or neighbor cell or the pair can be viewed simultaneously in the plot canvas on the left side. 

