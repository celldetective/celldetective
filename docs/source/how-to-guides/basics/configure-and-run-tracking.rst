How to configure and run tracking
----------------------------------

This guide shows you how to set up a tracker and run it on your segmented data.

Reference keys: :term:`tracking`, :term:`cell population`

**Prerequisite:** You must have segmented masks for the population of interest.


Configure the tracker
~~~~~~~~~~~~~~~~~~~~~

#. Navigate to the **Tracking** module in the main processing panel.

#. Click the **Settings** :icon:`cog-outline` button to open the configuration window.

#. **Select a Tracker**:

   *   Choose :term:`bTrack` (default) for complex behaviors (division, apoptosis) and crowded scenes. It uses a Bayesian approach with motion prediction.
   *   Choose :term:`trackpy` for simple particle tracking (Brownian motion).

#. **Add Features** (Optional): Click **Add features** to calculate morphological (e.g., area) or intensity features during tracking. Enable **:term:`Haralick Texture Features`** if you need texture analysis (computationally expensive).

#. **Configure Post-Processing** (Optional): Enable options to filter short tracks, fill gaps, or extrapolate positions.

#. Click **Save** to apply your settings.

For a detailed explanation of every parameter, see the :ref:`Tracking Settings Reference <ref_tracking_settings>`.


Run tracking
~~~~~~~~~~~~

#. In the **Tracking** module control panel, check the **TRACK** box.

#. Ensure you have selected the wells/positions you wish to process.

#. Click **Submit**.

Celldetective will load the segmentation masks, run the selected tracker, compute the requested features, and save the results as ``trajectories_targets.csv`` (or ``_effectors``) in the ``output/tables`` folder of each position.


Visualize tracks
~~~~~~~~~~~~~~~~

#. Select a single position in the file list.

#. Click the **Eye** button in the Tracking module.

#. Napari will open with the following layers:

   *   ``image``: Raw microscopy data.
   *   ``segmentation``: Labeled cell masks (color-coded by ID).
   *   ``tracks``: Trajectory lines connecting cell positions over time.
   *   ``points``: Centroids of detected cells.
