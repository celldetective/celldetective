How to detect sub-cellular spots
--------------------------------

This guide shows you how to count and measure spots inside your cell masks.

Reference keys: :term:`single-cell measurement`

**Prerequisite**: you have segmented your cell population of interest accurately.

#. Go to the MEASURE section and click on the :icon:`cog-outline,black` icon to enter measurement settings.

#. Scroll down to the **SPOT DETECTION** section.

#. Tick the *Perform spot detection* option.

#. Press the :icon:`image-check,black` icon on the right side to set up spot detection visually.

   .. figure:: ../../_static/figures/spot-detection.svg
       :width: 100%
       :target: ../../_static/figures/spot-detection.svg
       :align: center
       :alt: The spot detection block of the measurement settings and its viewer

       **Spot detection.** The SPOT DETECTION block of the measurement settings and the viewer it opens, here on the dead nuclei channel of the ADCC demo.

   The block holds the option (1), the channel and preprocessing (2), the diameter and threshold (3) and the viewer button (4). In the viewer, the filters (5) and the two **Set** buttons (6) drive the detection shown in red, and :icon:`plus,#1565c0` :blue:`Add measurement` (7) copies the values back to the block, saved with :blue:`Save` (8).

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

       You can check the **Preview** box (below the Preprocessing list) to see the effect of your filters on the image (e.g. smoothing, inversion). This preview does not show the detected spots, only the enhanced image.

#. Estimate visually the average spot diameter (in pixels). You can zoom in on the image.

#. Set the **Detection threshold** to 0 initially.

#. Press **Set** (next to Diameter or Threshold) to run the spot detection.
   
   - **Visual Feedback**: Detected spots will appear as red circles. 
   - **Note**: At threshold 0, you will likely see many false positives (background noise detected as spots). This is normal.

#. Gradually **increase the detection threshold** and press **Set** again to update the preview.
   
   - The goal is to filter out the false positives until only the real spots remain circled.
   - If spots are not detected even at threshold 0, try adjusting the diameter or checking your preprocessing.
   - **Note**: The detection uses the preprocessed image if filters are listed, regardless of whether the "Preview" checkbox is ticked.

#. Once the detection is satisfactory, press :icon:`plus,#1565c0` :blue:`Add measurement`.

#. Scroll down and press :blue:`Save` in the measurement settings.

#. Check the MEASURE option and press *Submit* to measure.

See :py:func:`celldetective.measure.extract_blobs_in_image` for more information about the algorithm used for single-spot detection.