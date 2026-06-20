How to detect an event using conditions
---------------------------------------

This guide shows you how to detect time-correlated events (e.g., cell death, spreading) by applying feature-based classification rules that produce a binary signal, which is then fitted with a sigmoid to extract the event time.

Reference keys: :term:`event`, :term:`event class`, :term:`event time`, :term:`threshold-based event detection`

**Prerequisite:** You have accurately segmented, **tracked**, and measured a cell population of interest. This guide only applies to dynamic data.


Define the classification rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. In the **CLASSIFY** section, click the :icon:`scatter-plot,black` button to open the classifier.

#. Enter a name for the event (e.g., ``death``).

#. Project features of interest to identify the transition signal. For example, plot ``PI_intensity_mean`` over time to see when cells become PI-positive.

#. Write the classification condition for the event (e.g., ``PI_intensity_mean > 500``), then press **Preview** to check which cells match (matching cells turn red).

#. Check the **Time correlated** option and select **Irreversible event**. This fits a sigmoid to the resulting binary signal to extract the event time.

#. Press **Save config** to store the classification as a reusable config. The classifier has no "Apply" button: back in the population block, click the :icon:`playlist-plus,black` button to import the saved config, then tick **DETECT EVENTS** and run the block. Time-correlated configs are routed to the **DETECT EVENTS** step automatically.


How the sigmoid fitting works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classification condition is evaluated at each time point, producing a binary signal per cell:

*   A **completely null** signal → no event detected.
*   A **completely positive** signal → the event occurred before imaging started.
*   A **sigmoid-like switch** → a transition. The time of event is extracted by fitting a sigmoid. An :math:`R^2 > 0.7` validates the fit; otherwise the cell is classified as "else" for manual correction.

.. figure:: ../../_static/classify.gif
    :width: 400px
    :align: center
    :alt: static_class

    The window to perform a feature-based classification on either static detections or trajectories.


Refine event annotations
~~~~~~~~~~~~~~~~~~~~~~~~~

After automatic detection, use the Event Annotator to verify and correct event times:

.. seealso::
    :doc:`annotate-an-event` for a step-by-step guide on manual event annotation and correction.