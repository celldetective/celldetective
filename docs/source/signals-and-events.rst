Signals and events
==================

.. _signals_and_events:

Prerequisites
-------------

Perform segmentation, tracking and measurements for either target or effector cells. Select a single position.


Deep-learning signal analysis
-----------------------------

We provide several Deep-learning models that take select single-cell signals as their input and determine the event class and time of event (if any) for all cells. Exactly as for the segmentation models, we provide a zoo of such models that can be applied to positions or wells.

**Workflow**

1.  Navigate to the **Signal Analysis** section in the control panel.
2.  Review the **Available Models** list and select the appropriate one.
3.  Click **Submit**.

**Signal Mapping**

If the model requires specific input signals (e.g., "Nuclei Intensity", "Cytoplasm Morphology"), a **Signals** configuration window will appear.
You must map the model's **Required Inputs** to the columns available in your current measurement tables.

For detailed mapping instructions, see the :ref:`Signal Analysis Reference <ref_signal_settings>`.

Click **Set** to launch the prediction process. The results (event class and time) will be added to your data tables. 


Threshold-based event characterization
--------------------------------------

We also provide a binary classification module, allowing you to classify cells to any number of classes or groups of interest in a "static" way, one frame at a time, based on feature values. This classification process yields a binary signal, which can be interpreted:

#. a perfectly null signal is associated to the absence of event
#. a completely positive signal is associated to an event that already happened
#. a sigmoid-like switch is a transition from an absence of event to an event. The time of event is extracted by fitting a sigmoid on the binary classification signal. A :math:`R^2` score is computed. If the score is higher than 0.7, the time is kept, otherwise the cell is classified as "else", to be corrected.

Click on the ``Classify data`` button of the measurements section. Define a name for the class/event/group. Project as many features of interest and write down the classification conditions for the event of interest. Apply to write in the tables the new class/group. If the ``TRACK_ID`` column is in the table (tracked data) and the ``Time correlated event`` option is checked, the sigmoid-fitting process will be triggered to extract the event times and create a class. When ``Time correlated event`` option is not checked, instead of a class/event, a characteristic group will be created and the phenotypes will be assigned accordingly.

.. figure:: _static/classify.gif
    :width: 400px
    :align: center
    :alt: static_class

    The window to perform a feature-based classification on either static detections or trajectories.



Single-cell signal visualization
--------------------------------

Single annotator configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Celldetective ships a powerful viewer for single-cell signals. If a single position is set and tracking has been performed, the eye icon in the signal analysis section becomes active.

Click the configuration button next to the eye to adjust settings (e.g., Image Mode, Playback Speed).
For details on these parameters, see the :ref:`Event Annotation Settings Reference <ref_event_annotation_settings>`.

Upon saving, a configuration file is created for future sessions. Click the eye to proceed.


Single annotator UI
~~~~~~~~~~~~~~~~~~~

.. figure:: _static/signal-annotator.gif
    :width: 800px
    :align: center
    :alt: signal_annotator

    Application on an ADCC system of MCF-7 breast cancer cells co-cultured with human primary NK cells where the Hoechst nuclear stain is blue, PI nuclear stain in red and CFSE marks the NK cells in green. Upon killing by the NK cells, the nuclei of MCF-7 cells turn red. 

In this application, blue cell nuclei turn red when a target cell is killed by a cell from the green population. You can zoom in the animation, move around, and click on any single cell of interest. The center of mass of cells is coded by a dynamic double scatter plot. The cross symbol encodes the cell class. The circle around the cross shows the current cell status (whether the event happened before the current frame or not). Upon clicking, the signals are updated in the left side panel, to show the measurements of the selected cell. You can view simultaneously up to three signals. Since quantities can be wildly different, normalization and log-rescaling buttons can be triggered to rescale the signals.

In the top part of the left side panel, you select the event of interest. Changing the event updates the colors attributed to each cell marker in the animation. You can decide to create a brand new event and annotate cells (almost) from scratch: you set an initial class for all cells to be modified. The class of a single-cell with respect to an event can be modified. Similarly, the time estimated in the case of an event can be changed to a different value. Don't forget to click on "Save" to save all modifications.

.. note::

   Cells marked for deletion (key Del), are temporarily colored in black. Upon saving, they are removed completely and cannot be recovered again.


If all cells have been annotated for an event of interest, you can decide to export a training set for signal analysis with a Deep Learning model, by clicking on the export button on the right side of the "Save" button. 


Keyboard shortcuts 
~~~~~~~~~~~~~~~~~~

The Single Annotator supports various keyboard shortcuts for navigation (``Space``, ``f``, ``l``) and annotation (``Del``, ``n``, ``Esc``).

For a complete list of shortcuts and mouse interactions, see the :ref:`Signal Annotator Shortcuts Reference <ref_signal_annotator_shortcuts>`.
