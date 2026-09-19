How to measure locally corrected intensities
---------------------------------------------

This guide shows you how to correct single-cell intensity measurements by subtracting or dividing by the local background around each cell.

Reference keys: **local correction**, :term:`single-cell measurement`

**Prerequisite:** You must have segmented the cells. Tracking is recommended but not required.


Enable local background correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. In the block of your population of interest, click the :icon:`cog-outline,black` button of the **MEASURE** row to open the measurement settings.

#. In the **BACKGROUND CORRECTION** section at the top, select the **Local** tab.


Configure the correction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../_static/figures/background-correction-local.svg
    :width: 100%
    :target: ../../_static/figures/background-correction-local.svg
    :align: center
    :alt: The Local tab of the background correction and the viewer that sets the band distance

    **Local background correction.** The Local tab of the measurement settings (ADCC demo) and the viewer that sets the distance of the background band around each cell.

#. **Channel**: select the intensity channel to correct.

#. **Distance**: set the distance (in pixels) from the edge of the mask over which the background is estimated (2). The background is sampled in a band outside each cell, up to this distance.

#. (Optional) Click :icon:`image-check,black` next to the distance to set it visually: the viewer draws the band around the cells of the current position; move the **Edge** slider (3) until the band covers background only, then press :blue:`Set` (4) to write the distance back.

#. **Model**: choose how to estimate the background intensity in the band (5):

   *   ``mean``: average intensity in the band.
   *   ``median``: median intensity (more robust to neighbouring cells).

#. **Operation**: choose how to apply the correction (6):

   *   **Subtract**: subtract the estimated background from the cell intensity. **Clip** sets the negative values to zero.
   *   **Divide**: divide the cell intensity by the estimated background.

#. Press :icon:`plus,#1565c0` :blue:`Add correction` (7). The correction appears in the **Corrections to apply** list (8); remove a selected one with :icon:`delete,black`.


Run the measurements
~~~~~~~~~~~~~~~~~~~~

#. Scroll down and click :blue:`Save` to save the configuration.

#. In the control panel, check the **MEASURE** box and click **Submit**.

The correction is applied to the image of each frame before the measurements: the intensity columns of the table (e.g. ``effector_fluo_channel_mean``) are then measured on the corrected channel.

.. tip::
    Use the viewer to ensure the background band does not overlap with neighboring cells. Decrease the distance if cells are densely packed.
