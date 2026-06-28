How to correct a single-cell track
----------------------------------

This guide shows you how to correct manually a single-cell track that missed detections or was truncated.

Reference keys: **tracking**

**Prerequisite:** You have accurately segmented, and **tracked** a cell population of interest. This guide only applies to dynamic data.

#. Go to the TRACK section for the cell population of interest. Click on the :icon:`eye-check,black` icon on the right side to open napari and explore tracking results.

#. On the left side, click on the Image layer associated with your channel of interest. With multi-channel data, click on the :icon:`eye,black` icon to disable other Image layers. Adjust the *contrast limits* with the double slider (top left).

#. Select the segmentation layer.

#. Use the time axis slider below the image to find a track cut early with more detections available at later times.

#. Put the slider at the last frame where the track was still uninterrupted. Pick the pipette tool (:icon:`eyedropper,black`) in the top left of part of napari. With the pipette selected, click on the mask of the cell of interest.

#. Pick the Pan/Zoom button (:icon:`cursor-move`).

#. Use the time axis slider to go to the next frame where a cell mask related to this track is observed.

#. Double click on the cell. Say yes to the pop-up to propagate the identity of the cell from the previous frame to the current frame. If the mask is associated with another track, time propagation will be performed automatically.

#. Optional: above the export button, tune the *track post-processing* options applied on save -- removing tracks that do not start at the beginning or do not end at the end, interpolating missed detections, sustaining the first/last position to the movie boundaries, interpolating missing values, and a minimum tracklength. These widgets are pre-filled from the position's tracking configuration, so by default they reproduce the pipeline you already set up. Leave everything unchecked (and the minimum tracklength at ``0``) to export the tracks exactly as corrected.

#. Press the *Export the modified tracks...* button to save the changes in the position table. The post-processing is applied and the track and point layers refresh in place, so removed tracks disappear and interpolated/extrapolated positions appear immediately.

.. note::

    The post-processing affects the trajectory (track and point) layers and the saved table. The displayed segmentation masks are not pruned for removed tracks, so a mask may remain visible for a track that was filtered out of the table on export.



