Tutorial: Your First Experiment
===============================

.. _first_experiment:

In this tutorial, you will go through a complete workflow: loading a demo dataset, segmenting cells, tracking them, and visualizing the results.

.. note::
    **Prerequisites**: Ensure you have installed Celldetective following the :doc:`Get Started <get-started>` guide.

Step 1: Get the demo data
-------------------------

We have a built-in shortcut to download a demo dataset (Spreading Assay).

1.  Open your terminal and run:

    .. code-block:: console

        $ python -m celldetective

2.  In the startup window's menu bar, go to **File > Open Demo > Spreading Assay Demo**.
3.  Select a folder where you want to save the data.
4.  The software will download the ``demo_ricm`` dataset and automatically load it.

.. figure:: _static/maingui.png
    :align: center
    :alt: main_gui
    :width: 600px

    The main window. Use the **File** menu to access demos.


Step 2: Segment Cells
---------------------

Now we will detect the cells in the images. The demo dataset contains images of immune cells spreading on a surface.

1.  In the **Experiment Overview** tab (center panel), click on **Run Analysis** to open the main interface.
2.  Locate the **Process Effectors** block in the processing panel (expand it if necessary).
3.  Check the **Segment** box.
4.  In the **Model zoo** dropdown, select **lymphocytes_ricm**.
5.  Click **Submit** to run segmentation.

    .. tip::
        You can visualize the results immediately by clicking the :icon:`eye,black` button next to the segmentation entry. This opens Napari with the image and mask layers.


Step 3: Track Cells
-------------------

Once cells are segmented, we can link them over time.

1.  Check the **Track** box.
2.  Click the :icon:`cog-outline,black` button next to it.
3.  Select **trackpy** in the tracking options.
4.  Close the configuration window and click **Submit**.
5.  The software will link detections frame-by-frame and generate trajectories.


Step 4: Measure Features
------------------------

To analyze cellular dynamics, we need to extract quantitative features.

1.  Check the **Measure** box.
2.  Click the :icon:`cog-outline,black` button next to it.
3.  Ensure **area** and **intensity_mean** are listed in the features list.
4.  Close the configuration window and click **Submit**.
5.  Celldetective will extract these features for every cell at every time point.


Step 5: Analyze Signals
-----------------------

This is the core of Celldetective: analyzing how single-cell features change over time.

1.  Scroll to the **Signal Analysis** section.
2.  Click the :icon:`eye,black` button (Signal Annotator) to open the interactive viewer.
3.  **Click on any cell** in the movie.
4.  The panel on the left will display its **Signal Traces** (e.g., Intensity vs Time).
5.  You can inspect distinct behaviors, identifying events like cell division or death based on these traces.


Step 6: Visualize Results
-------------------------

1.  Go back to the **Tracking** section.
2.  Click the :icon:`eye,black` button.
3.  Napari will open showing the image, masks, and tracks.
4.  Use the slider to watch the cells and their trails!

Congratulations! You have successfully processed your first experiment.

Next Steps
----------

*   Learn how to :doc:`create your own experiment <how-to-guides/basics/create-an-experiment>`.
*   Try :doc:`conditional classification <how-to-guides/basics/perform-conditional-cell-classification>` to identify cell states.
*   Perform :doc:`survival analysis <how-to-guides/basics/plot-survival>` on your tracked cells.
*   Explore :doc:`measurement options <measure>`.
