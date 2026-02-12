How to measure edge intensity
-----------------------------

This guide shows you how to measure intensity features within specific contour bands relative to the cell boundary (e.g., peripheral or peri-cellular intensity).

Reference keys: :term:`contour`, :term:`single-cell measurement`

**Prerequisite:** You must have segmented the cells. Tracking is recommended but not required.


Enable contour measurements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open the **Measure** tab for your population of interest.

#. In the measurement settings, locate the **Contour** section.

#. Check the **Contour** option.


Configure the contour band
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. **Channel**: Select the intensity channel to measure.

#. **Distance** :math:`d` (pixels): define the offset from the mask edge.

   *   Positive values (:math:`d > 0`) measure **inside** the cell (erosion from the boundary).
   *   Negative values (:math:`d < 0`) measure **outside** the cell (dilation beyond the boundary).

#. **Range** ``(min, max)``: define a specific ring band using distance transforms. For example, ``(0, 5)`` measures a 5-pixel-wide ring at the cell edge.


Run the measurements
~~~~~~~~~~~~~~~~~~~~

#. Click **Set** to save the configuration.

#. In the control panel, check the **MEASURE** box and click **Submit**.

The contour intensity features (e.g., ``intensity_mean``, ``intensity_max``) for the selected band will be appended to your measurement table with a suffix indicating the contour distance.

.. tip::
    Combine a positive and negative distance to measure both inside and outside the cell boundary, which is useful for quantifying membrane-associated signals.