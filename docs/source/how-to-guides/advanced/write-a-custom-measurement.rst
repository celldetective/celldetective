How to write a custom measurement
=================================

This guide shows you how to create your own single-cell measurement in Python, ready-to-be-used in the software.

Reference keys: **ROI**, **Mask**, **Features**

.. admonition:: Prerequisite

    You need access to the source code of Celldetective or at least the `celldetective/extra_properties.py` file.

.. seealso::
    :doc:`../../reference/regionprops` — technical details of the custom ``regionprops``
    implementation and how it differs from scikit-image's original.

Introduction
------------

Celldetective allows you to extend its measurement capabilities by adding custom Python functions. These functions are automatically discovered and applied to every cell during the measurement process. This is useful for specific needs like:

*   Measuring the area of dark regions within a cell.
*   Computing specific intensity percentiles.
*   Calculating shape descriptors not included in the standard library.

Step 1: Locate the definitions file
-----------------------------------

1.  Navigate to your Celldetective installation folder.
2.  Open the file `celldetective/extra_properties.py` in a text editor or IDE.

Step 2: Define your function
----------------------------

There are two valid signatures depending on whether you want to measure **every channel** or **one specific channel**.

**Measure every channel:**

.. code-block:: python

    def my_custom_measurement(regionmask, intensity_image, **kwargs):
        # Called once per cell per channel.
        # intensity_image is a 2-D single-channel crop.
        return scalar_value

**Measure one specific channel:**

.. code-block:: python

    def my_channel_measurement(regionmask, intensity_image, target_channel='my_channel', **kwargs):
        # Called once per cell, only for the channel named 'my_channel'.
        # All other channel output slots are automatically set to NaN.
        return scalar_value

**Arguments:**

*   ``regionmask`` (*ndarray*): A binary mask of the cell within its bounding box.
*   ``intensity_image`` (*ndarray*): The intensity image crop. Unlike scikit-image's default ``regionprops``, the background is **not zeroed** — threshold-based analysis within the bounding box is valid.
*   ``target_channel`` (*str, optional*): If present, its **default value** declares which channel this function applies to. Celldetective reads the default via ``inspect.signature`` and calls the function only for that channel; you never pass it yourself.
*   ``**kwargs``: Required to absorb extra arguments passed by the framework.

**Return Value:**

*   Must return a single **scalar** (float or int).
*   Returning ``NaN`` is allowed.

**Example: Measuring the max intensity**

.. code-block:: python

    import numpy as np

    def max_intensity(regionmask, intensity_image, **kwargs):
        # Select pixels within the cell mask
        masked_pixels = intensity_image[regionmask]
        # Return the maximum value
        return np.max(masked_pixels)

Step 3: Naming your function
----------------------------

The name of your function determines the column name in the output table.

*   **Automatic Renaming**: If your function name contains ``intensity``, it will be replaced by the actual channel name.
    *   *Example:* ``max_intensity`` becomes ``max_red_channel`` (if measuring the red channel).
*   **Avoid Conflicts**: Do not use simple numbers (e.g., ``measure_1``) to avoid confusion with channel indices.

Step 4: Use it in Celldetective
-------------------------------

1.  Save the `extra_properties.py` file.
2.  Restart Celldetective.
3.  Go to the **Measurements** module settings.
4.  Your new function will appear in the **Extra features** list. Checks the box to enable it.
