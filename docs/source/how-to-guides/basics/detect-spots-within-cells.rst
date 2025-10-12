How to detect sub-cellular spots
--------------------------------

This guide shows you how to count and measure spots inside your cell masks.

Reference keys: *single-cell measurements*

**Prerequisite**: you have segmented your cell population of interest accurately.

#. Go to the MEASURE section and click on the :icon:`cog-outline,black` icon to enter measurement settings.

#. Scroll down to the **SPOT DETECTION** section.

#. Tick the *Perform spot detection* option.

#. Press the :icon:`image-check,black` icon on the right side to set up spot detection visually.

#. Set up the channel interest and adjust contrast to see the spots clearly.

#. Set the detection channel to the same channel as above.

#. Estimate visually the average spot diameter. Set the detection threshold to 0. Press Set (any of the two buttons).

#. Assess whether single-spot circles reflect accurately single spot size (you can have many false positive detections at this stage, which is normal, leave the threshold at 0).

#. Increase the detection threshold to remove as many false positive detections as possible.

#. Once the detection is satisfactory, press :icon:`plus,black` Add measurement.

#. Save the new measurement settings.

#. Check the MEASURE option and press *Submit* to measure.

See :py:func:`celldetective.measure.extract_blobs_in_image` for more information about the algorithm used for single-spot detection.