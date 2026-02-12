Signals and events
==================

.. _signals_and_events:

Prerequisites
-------------

Perform segmentation, tracking, and measurements for either target or effector cells. Select a single position.


Overview
--------

After measuring single-cell features over time, the next step is to characterize dynamic behaviors — detecting when and whether specific events occur for each cell. Celldetective offers two complementary strategies for this: deep learning signal analysis and threshold-based event detection.


Deep-learning signal analysis
-----------------------------

Celldetective provides a zoo of deep-learning models that take single-cell signal traces as input and predict an event class and time of event for each cell. These models work similarly to segmentation models — select one, map your measurement columns to the model's expected inputs, and submit.

For a detailed list of signal mapping parameters, see the :ref:`Signal Analysis Reference <ref_signal_settings>`.


Threshold-based event detection
-------------------------------

As an alternative to deep learning, you can define feature-based classification rules (e.g., ``PI_intensity_mean > 500``) that produce a binary signal per cell. For tracked cells with time-correlated events, a sigmoid is fitted to extract the event time. The quality of the fit is assessed by an :math:`R^2` score.

.. seealso::
    :doc:`how-to-guides/basics/detect-an-event-with-conditions` for a step-by-step guide.


Single-cell signal viewer
--------------------------

Celldetective ships a powerful viewer for exploring single-cell signals and manually annotating events. This tool allows you to:

*   Visualize single-cell signal traces (intensity, morphology) synchronized with the movie.
*   Manually annotate event times and classes.
*   Curate datasets to train deep-learning event detection models.

.. figure:: _static/signal-annotator.gif
    :width: 800px
    :align: center
    :alt: signal_annotator

    Application on an ADCC system of MCF-7 breast cancer cells co-cultured with human primary NK cells.

The viewer displays the movie with cell centroids marked by their current event status. Clicking a cell reveals its full temporal signal trace.

.. seealso::
    :doc:`how-to-guides/basics/annotate-an-event` for a step-by-step annotation guide. |
    :ref:`Event Annotation Settings <ref_event_annotation_settings>` for viewer configuration.



