How to detect an event using conditions
---------------------------------------

This guide shows you how to detect time-correlated events (e.g., cell death, spreading) by applying feature-based classification rules that produce a binary signal, which is then fitted with a sigmoid to extract the event time.

Reference keys: :term:`event`, :term:`event class`, :term:`event time`, :term:`threshold-based event detection`

**Prerequisite:** You have accurately segmented, **tracked**, and measured a cell population of interest. This guide only applies to dynamic data.


Define the classification rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. In the **Measurements** section, click **Classify data**.

#. Enter a name for the event (e.g., ``death``).

#. Project features of interest to identify the transition signal. For example, plot ``PI_intensity_mean`` over time to see when cells become PI-positive.

#. Write the classification condition for the event (e.g., ``PI_intensity_mean > 500``).

#. Check the **Time correlated event** option. This triggers sigmoid fitting on the resulting binary signal.

#. Click **Apply**.


How the sigmoid fitting works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classification condition is evaluated at each time point, producing a binary signal per cell. Only the frames where the cell is observed are read: from its first detection, or after the pre-event if one is set.

*   A **completely null** signal → no event detected (class 1).
*   A **completely positive** signal → the event occurred before the cell was first observed (class 2).
*   A **sigmoid-like switch** → a transition. The time of event is extracted by fitting a sigmoid. An :math:`R^2` above the threshold (0.5 by default) validates the fit (class 0). A fit at or before the first observed frame means the transition was not observed (class 2).
*   A **poor fit** → the cell is classified by its median state: no event (class 1) if mostly negative, otherwise "else" (class 2) with no event time, for manual correction.

The ``status_`` column then follows from the class and time: 0 before the event and 1 from it on, and empty for an "else" cell with no event time. The per-frame result of the condition is kept in the ``inst_status_`` column.

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