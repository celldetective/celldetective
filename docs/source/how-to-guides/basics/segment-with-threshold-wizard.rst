How to segment with the Threshold Configuration Wizard
-------------------------------------------------------

This guide shows you how to build a traditional segmentation pipeline interactively using filters, thresholds, and morphological operations — without a Deep Learning model.

Reference keys: :term:`instance segmentation`, :term:`cell population`


Open the Threshold Configuration Wizard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Select a specific position within your experiment.

#. In the **Segmentation** section of the Control Panel, locate the population of interest (e.g., Targets or Effectors).

#. Click the :icon:`upload,black` button.

#. Toggle the **Threshold** option.

#. Click the **Threshold Config Wizard** button to open the interface.

.. image:: ../../_static/tcw.png
    :align: center
    :alt: threshold_config_wizard

*The Threshold Configuration Wizard interface showing preprocessing, thresholding, and object detection controls.*


Step 1: Preprocessing
~~~~~~~~~~~~~~~~~~~~~

Enhance the image to make objects easier to detect.

*   **Add Filter:** Select a filter (e.g., ``gauss``, ``median``, ``std``) and kernel size, then click **Add**.
*   **Remove Filter:** Double-click a filter in the list to remove it.
*   **Apply:** Click **Apply** to see the effect on the image.


Step 2: Thresholding
~~~~~~~~~~~~~~~~~~~~

Binarize the image to separate foreground (cells) from background.

*   **Slider:** Adjust the min/max sliders to define the intensity range of fit.
*   **Histogram:** Use the histogram to identify intensity peaks. Toggle **Log scale** for better visibility of low-intensity pixels.
*   **Fill Holes:** Keep this checked to automatically fill holes inside detected objects.


Step 3: Object Detection (Split / Merge)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert the binary mask into individual objects.

**Option A: Markers (Watershed)**
Best for touching cells or nuclei.

*   **Footprint:** Adjust the size of the local region used to find distinct peaks. Larger values merge peaks; smaller values split them.
*   **Min Distance:** Set the minimum allowed distance between two object centers.
*   **Run:** Click **Run** to detect markers (shown as red dots).
*   **Watershed:** Click **Watershed** to expand markers into object boundaries.

**Option B: All Objects**
Best for well-separated objects.

*   **Select:** Choose **all non-contiguous objects**.
*   **Watershed:** Click **Watershed** to label all connected components directly.


Step 4: Property Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove false positives based on morphology or intensity.

*   **Visualize:** Use the dropdowns to plot two properties (e.g., ``area`` vs ``solidity``) on the scatter plot.
*   **Select:** Click points on the scatter plot to highlight the corresponding object in the viewer.
*   **Query:** Enter a filtering query in the text box (e.g., ``area > 100`` or ``solidity > 0.9``).
*   **Filter:** Click **Submit Query** to remove objects that don't match the criteria.


Save and apply the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Click **Save**. The configuration is saved as a ``.json`` file in your experiment's ``configs/`` folder.

#. The wizard closes, and the config file path is automatically loaded into the **Upload Model** window.

#. Click **Upload** to confirm.

#. To process the entire position or experiment, select **Threshold** in the segmentation zoo and click **Submit**.

.. note::

    You must reload the threshold config file if you reopen the experiment later.
