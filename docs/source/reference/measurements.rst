Measurements Reference
======================

.. _ref_measurements:

This page provides a comprehensive reference for all single-cell measurements available in Celldetective.
The output table (e.g., `trajectories.csv` or `tracked_data.csv`) contains one row per cell per time point, with columns corresponding to these measurements.

.. note::
   **Channel Naming**: For multi-channel images, intensity-based measurements are prefixed with the channel name (e.g., ``GFP_mean``, ``RFP_max``).

Morphological Measurements
--------------------------

Standard shape descriptors computed from the cell's segmentation mask using ``scikit-image.regionprops``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Measurement Name
     - Description
   * - ``area``
     - Number of pixels in the region.
   * - ``perimeter``
     - Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
   * - ``eccentricity``
     - Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is between 0 and 1.
   * - ``solidity``
     - Ratio of pixels in the region to pixels of the convex hull image.
   * - ``extent``
     - Ratio of pixels in the region to pixels in the total bounding box.
   * - ``orientation``
     - Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region (in radians).
   * - ``axis_major_length``
     - The length of the major axis of the ellipse that has the same normalized second central moments as the region.
   * - ``axis_minor_length``
     - The length of the minor axis of the ellipse that has the same normalized second central moments as the region.
   * - ``equivalent_diameter_area``
     - The diameter of a circle with the same area as the region.
   * - ``feret_diameter_max``
     - Maximum Feret diameter (longest distance between any two points on the region contour).
   * - ``euler_number``
     - Euler characteristic of the set of non-zero pixels. Computed as number of objects minus number of holes (using 8-connectivity).
   * - ``convex_area``
     - Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.
   * - ``centroid-0``, ``centroid-1``
     - Coordinate of the centroid (row, col). **Note**: In tracked data, these are replaced by ``POSITION_Y`` and ``POSITION_X``.

Intensity Measurements
----------------------

Statistics of pixel intensities within the cell mask.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Measurement Name
     - Description
   * - ``{channel}_mean``
     - Mean intensity. NaN values are ignored.
   * - ``{channel}_max``
     - Maximum intensity.
   * - ``{channel}_min``
     - Minimum intensity.
   * - ``{channel}_std``
     - Standard deviation of intensity.

Center of Mass Measurements
---------------------------

Quantifies the spatial distribution of intensity relative to the geometric center.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Measurement Name
     - Description
   * - ``{channel}_center_of_mass_distance``
     - Euclidean shift between geometric centroid and intensity center of mass.
   * - ``{channel}_center_of_mass_angle``
     - Angle of the shift vector (radians).
   * - ``{channel}_center_of_mass_dx``
     - Shift along X-axis (pixels).
   * - ``{channel}_center_of_mass_dy``
     - Shift along Y-axis (pixels).

Radial Intensity Measurements
-----------------------------

Analysis of intensity distribution from center to periphery.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Measurement Name
     - Description
   * - ``{channel}_radial_intensity_slope``
     - Slope of linear regression (intensity vs radius).
   * - ``{channel}_radial_intensity_intercept``
     - Intercept at center.
   * - ``{channel}_radial_intensity_r2``
     - $R^2$ goodness-of-fit to linear gradient model.

Texture Measurements (Haralick)
-------------------------------

.. note::
   Texture features are computed using the **Haralick** method (via the `mahotas` library).
   These features are calculated on a per-channel basis.

**Naming Convention:**

``haralick_{feature}_{channel}``

where ``{feature}`` is one of the following 13 Haralick features:

*   ``angular_second_moment``
*   ``contrast``
*   ``correlation``
*   ``sum_of_square_variance``
*   ``inverse_difference_moment``
*   ``sum_average``
*   ``sum_variance``
*   ``sum_entropy``
*   ``entropy``
*   ``difference_variance``
*   ``difference_entropy``
*   ``information_measure_of_correlation_1``
*   ``information_measure_of_correlation_2``
*   ``maximal_correlation_coefficient``

**Example:**
For a channel named ``EGFP``, the contrast would be: ``haralick_contrast_EGFP``.

.. tip::
   Haralick features are computed using a specific **distance** (offset) defined in the settings.
   However, the distance is not included in the feature name. If you process multiple distances, keep in mind they might overwrite each other if output to the same table.

Spot Detection
--------------

.. note::
   Spot detection counts "blobs" or spots within the cell area (e.g., vesicles, granules) using LoG (Laplacian of Gaussian) or DoG (Difference of Gaussian).

**Naming Convention:**

*   ``{channel}_spot_count``: Total number of spots detected in the cell for the given channel.
*   ``{channel}_mean_spot_intensity``: Mean intensity of the detected spots (pixels belonging to spots) for the given channel.

**Example:**
For a channel named ``Lysosomes``, you will get:

*   ``Lysosomes_spot_count``
*   ``Lysosomes_mean_spot_intensity``

Extra Properties (Custom)
-------------------------

Celldetective includes several built-in "extra properties" that provide more advanced morphological or intensity descriptors.

**Area/Fraction in Intensity Threshold:**

*   ``area_detected_in_{channel}``: Area (in pixels) of the cell that is above a certain intensity threshold in ``{channel}``.
*   ``fraction_of_area_detected_in_{channel}``: Fraction of the cell's total area that is above the threshold (0.0 to 1.0).
*   ``area_dark_in_{channel}``: Area (in pixels) of the cell that is *below* a certain intensity threshold.
*   ``fraction_of_area_dark_in_{channel}``: Fraction of the cell's total area that is *below* the threshold.

**Other Custom Properties:**

*   ``intensity_nanmean``: Mean intensity ignoring NaNs (useful for masked arrays).
*   ``feret_diameter_max``: Maximum Feret diameter.

Neighborhood Measurements
-------------------------

Neighborhood measurements describe the local environment of a cell, quantifying its neighbors.
These are often computed between two populations (e.g., Targets vs Effectors) or within a single population.

**Naming Convention:**

``{metric}_count_neighborhood_({pop1}-{pop2})_{method}_{radius}_px``

Where:

*   ``{metric}`` is one of:

    *   ``inclusive``: Total number of neighbors.
    *   ``exclusive``: Number of neighbors for which the current reference cell is the absolute *closest*.
    *   ``intermediate``: A weighted count sum of weights, where $w = 1/N_{neighborhoods}$. A neighbor belonging to 2 references contributes 0.5 to each.

*   ``{pop1}-{pop2}``: The pair of populations interacting (e.g., ``targets-effectors``).
*   ``{method}``: The neighborhood definition method:

    *   ``circle``: Neighbors within a fixed Euclidean distance.
    *   ``contact``: Neighbors touching or within a small dilation distance (mask-based).

*   ``{radius}``: The search radius or duality distance in pixels.

**Example:**
``inclusive_count_neighborhood_(targets-effectors)_circle_30_px``

**Status Decomposition:**

If the analysis distinguishes between neighbor states (e.g., "live" vs "dead"):

*   ``_s1_``: Positive status (e.g., live neighbors).

    *   Example: ``inclusive_count_s1_neighborhood_(targets-effectors)_circle_30_px``

*   ``_s0_``: Negative status (e.g., dead/no-event neighbors).

    *   Example: ``inclusive_count_s0_neighborhood_(targets-effectors)_circle_30_px``

**Mean Neighborhood Relative to Events:**

When analyzing tracks with detected events (e.g., cell division, death, or a specific signal change), neighborhood metrics can be aggregated relative to the event time ($t_{event}$) **of the reference cell**.

*   ``mean_count_{metric}_{neigh_col}_before_event``

    *   **Description**: The average of the ``{metric}`` (e.g., ``inclusive``, ``intermediate``) calculated over all frames **up to and including** the event frame of the reference cell ($t \le t_{event}$).
    *   **Note**: If no event is detected for the reference cell (or no event column is specified), this metric is calculated over the **entire duration** of the track ($t \le t_{max}$).
    *   **Implicit Status**: This metric specifically averages the counts of neighbors with **positive status** (``_s1_``), e.g., live cells.

*   ``mean_count_{metric}_{neigh_col}_after_event``

    *   **Description**: The average of the ``{metric}`` calculated over all frames **strictly after** the event frame of the reference cell ($t > t_{event}$).
    *   **Note**: If no event is detected for the reference cell, or if the event occurs at the last frame, this value will not be computed (NaN).
    *   **Implicit Status**: This metric specifically averages the counts of neighbors with **positive status** (``_s1_``).

For details on neighborhood analysis, see :doc:`../how-to-guides/basics/measure-cell-interactions`.
