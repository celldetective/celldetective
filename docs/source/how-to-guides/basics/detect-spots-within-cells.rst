How to detect sub-cellular spots
--------------------------------

This guide shows you how to count and measure spots inside your cell masks.

Reference keys: :term:`single-cell measurements`

**Prerequisite**: you have segmented your cell population of interest accurately.

#. Go to the MEASURE section and click on the :icon:`cog-outline,black` icon to enter measurement settings.

#. Scroll down to the **SPOT DETECTION** section.

#. Tick the *Perform spot detection* option.

#. Press the :icon:`image-check,black` icon on the right side to set up spot detection visually.

#. Set up the channel interest using the controls on the **Right Panel**. You can change the displayed frame and adjust contrast to see the spots clearly.
    
    .. note::
        The viewer is split into two panels:
        - **Left Panel**: Contains all detection settings (Channels, Thresholds, Preprocessing).
        - **Right Panel**: Displays the image and visualization controls.

#. In the **Left Panel**, set the detection channel to the same channel as above.

   .. tip::
       If the image is noisy or the background is uneven, or if the spots are dark (e.g., RICM), use the **Preprocessing** options below the channel selection.
       
       - For **noisy images**: Add a `gaussian` or `median` filter (e.g., sigma=1 or size=3).
       - For **uneven background**: Add a `tophat` filter (white tophat) to isolate bright spots.
       - For **dark spots**: Add an `invert` filter roughly at the bit-depth max (e.g., 255 or 65535) to make spots bright.

       You can check the **Preview** box (below the Preprocessing list) to see the effect of your filters on the image.

#. Estimate visually the average spot diameter (in pixels). You can zoom in on the image.

#. Set the **Detection threshold** to 0 initially.

#. Press **Set** (any of the two buttons) to preview the detection. 
   
   - **Visual Feedback**: Detected spots will appear as red circles. 
   - **Note**: At threshold 0, you will likely see many false positives (background noise detected as spots). This is normal.

#. Gradually **increase the detection threshold** and press **Set** again to update the preview.
   
   - The goal is to filter out the false positives until only the real spots remain circled.
   - If spots are not detected even at threshold 0, try adjusting the diameter or checking your preprocessing.

#. Once the detection is satisfactory, press :icon:`plus,black` Add measurement.

#. Save the new measurement settings.

#. Check the MEASURE option and press *Submit* to measure.

See :py:func:`celldetective.measure.extract_blobs_in_image` for more information about the algorithm used for single-spot detection.