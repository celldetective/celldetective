Tracking
========

.. _track:

The Tracking module links segmented cells across frames to create trajectories. This allows you to analyze cell motility, lineage, and dynamic behaviors.

Prerequisite
------------

*   **Segmentation**: You must have segmented the cells (masks) before tracking.

.. _configure_tracker:

How to Configure the Tracker
----------------------------

1.  Navigate to the **Tracking** module in the main processing panel.
2.  Click the **Settings** button |settings_icon| to open the configuration window.
3.  **Select a Tracker**:
    *   Choose **bTrack** (default) for complex behaviors (division, apoptosis) and crowded scenes. It uses a Bayesian approach with motion prediction.
    *   Choose **trackpy** for simple particle tracking (Brownian motion).
4.  **Add Features** (Optional):
    *   Click **Add features** to calculate morphological (e.g., area) or intensity features during tracking.
    *   Enable **Haralick texture features** if you need texture analysis (computationally expensive).
5.  **Configure Post-Processing** (Optional):
    *   Enable options to filter short tracks, fill gaps, or extrapolate positions.
6.  Click **Save** to apply your settings.

.. tip::
    For a detailed explanation of every parameter, see the :ref:`Tracking Settings Reference <ref_tracking_settings>`.

.. _run_tracking:

How to Run Tracking
-------------------

1.  In the **Tracking** module control panel, check the **TRACK** box.
2.  Ensure you have selected the wells/positions you wish to process.
3.  Click **Submit**.
4.  Celldetective will:
    *   Load the segmentation masks.
    *   Run the selected tracker.
    *   Compute the requested features.
    *   Save the results as ``trajectories_targets.csv`` (or ``_effectors``) in the ``output/tables`` folder of each position.

.. _visualize_tracks:

How to Visualize Tracks
-----------------------

1.  Select a single position in the file list.
2.  Click the **Eye** button |eye_icon| in the Tracking module.
3.  Napari will open with the following layers:
    *   ``image``: Raw microscopy data.
    *   ``segmentation``: Labeled cell masks (color-coded by ID).
    *   ``tracks``: Trajectory lines connecting cell positions over time.
    *   ``points``: Centroids of detected cells.

.. _correct_tracks:

How to Correct Tracking Errors
------------------------------

You can manually correct tracking mistakes (e.g., identity switches) using the Napari viewer.

1.  **Identify the Error**: Scroll through the timeline to find where a cell's ID changes incorrectly.
2.  **Select the Correct ID**:
    *   Activate the ``segmentation`` layer.
    *   Select the **picker tool** (pipette).
    *   Click on the cell *before* the error (the correct ID). The label value will be selected.
3.  **Apply to Future Frames**:
    *   Advance to the frame where the error occurs.
    *   **Double-click** on the cell with the wrong ID.
    *   A confirmation dialog will appear. Click **Yes**.
    *   The software will assign the selected ID to this cell and propagate it to all subsequent frames.
    *   Any conflicting track that previously held this ID will be assigned a new, unique ID to prevent merging.
4.  **Save Changes**:
    *   Once finished, click the **Export the modified tracks...** button in the Napari dock widget.
    *   This updates the CSV file and re-runs any post-processing (e.g., smoothing, velocity).

.. |settings_icon| image:: _static/settings_icon_placeholder.png
    :height: 1em
.. |eye_icon| image:: _static/eye_icon_placeholder.png
    :height: 1em

References
----------

.. [#] Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. Frontiers in Computer Science 3, (2021).
