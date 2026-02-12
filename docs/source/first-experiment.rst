Your First Experiment
=====================

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

1.  Locate the **Process Effectors** block in the processing panel and expand it.
2.  Check the **Segment** box.
3.  In the **Model zoo** dropdown, select **lymphocytes_ricm**.
4.  Click **Submit** to run segmentation.

    .. tip::
        You can visualize and correct the segmentation results by clicking the :icon:`eye-outline,black` button next to the segmentation entry. This opens napari with the image and mask layers.


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
5.  Celldetective will measure these features for every cell at every time point.


Step 5: Analyze Time-Series
---------------------------

This is the core of Celldetective: analyzing how single-cell features change over time.

1.  Scroll to the **Signal Analysis** section.
2.  Click the :icon:`eye,black` button (Event Annotator) to open the interactive viewer.
3.  **Click on any cell** in the movie.
4.  The panel on the left will display its feature time-series (e.g., Intensity vs Time).

.. tip::
    **Detect Spreading Events**: In this demo, cells become dark (low intensity) when they spread. Let's annotate this using the **Classifier Widget**:

    1.  In the **Measure** section, click the :icon:`scatter-plot,black` button (Classifier Widget).
    2.  Set **class name** to ``spreading``.
    3.  In the **classify** field, type ``intensity_mean < 1``.
    4.  Check **Time correlated** and select **irreversible event**.
    5.  Click **apply** to detect this event for all tracks.
    6.  Reopen the Event Annotator (Step 5) to see the vertical lines marking the spreading time.


Step 6: Explore Results
-----------------------

You can now use dedicated tools to analyze your data:

1.  **Survival Analysis**: Predict the time between first detection and spreading using the :doc:`Survival Plot <how-to-guides/basics/plot-survival>` (Start Event: ``t_firstdetection``, End Event: ``t_spreading``).
2.  **Table Exploration**: Inspect feature distributions and correlations with the :doc:`Table Explorer <table_exploration>`.

Congratulations! You have successfully processed your first experiment.

Next Steps
----------

*   Learn how to :doc:`create your own experiment <how-to-guides/basics/create-an-experiment>`.
*   Try :doc:`conditional classification <how-to-guides/basics/perform-conditional-cell-classification>` to identify cell states.
*   Explore :doc:`measurement options <measure>`.
