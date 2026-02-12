A Two-Population Experiment
===========================

.. _adcc-example:

In this tutorial, you will analyze a co-culture experiment: MCF-7 cancer cells (targets) and primary NK cells (effectors). We will focus on the targets to detect lysis events.

Step 1: Get the demo data
-------------------------

We provide a demo experiment in `Zenodo`_ (ADCC experiment).

.. _Zenodo : https://zenodo.org/records/10650279

1.  Open your terminal and run:

    .. code-block:: console

        $ python -m celldetective

2.  In the menu bar, go to **File > Open Demo > Cytotoxicity Assay Demo**.
3.  The project will download and load automatically.

Step 2: Segment Targets
-----------------------

We will process the first position of the first well.

1.  In the top control panel, select **Well** ``W1`` and **Position** ``100``.
2.  Expand the **PROCESS TARGETS** block.
3.  Check the **Segment** box.
4.  In the **Model zoo**, select ``mcf7_nuc_stardist_transfer``.
5.  Click **Submit** to run segmentation.

    .. tip::
        Click the :icon:`eye-outline,black` button to visualize the segmentation results in Napari.

Step 3: Track Targets
---------------------

1.  Check the **Track** box.
2.  Click the :icon:`cog-outline,black` button to configure tracking.
3.  Set parameters:

    *   **Minimum tracklength**: ``10``
    *   Check **Remove tracks that do not start at the beginning**.
    *   Check **Interpolate missed detections**.
    *   Check **Sustain last position**.
    *   Uncheck other post-processing options.

4.  Click **Save**.
5.  Ensure **Segment** is unchecked and Click **Submit**.

Step 4: Measure Targets & Detect Events
---------------------------------------

We will detect cell death (lysis) using the uptake of Propidium Iodide (PI).

1.  Check the **Measure** box.
2.  Check the **Detect Events** box.
3.  In the **Signal models** list, select ``lysis_PI_area``.
4.  Click **Submit**.

Step 5: Process Effectors
-------------------------

We will segment and measure the primary NK cells (Effectors) without tracking them, as the time resolution is low.

1.  Expand the **PROCESS EFFECTORS** block.
2.  Check the **Segment** box.
3.  In the **Model zoo**, select ``primNK_cfse``.
4.  Ensure **Track** is **unchecked**.
5.  Check the **Measure** box.
6.  Click **Submit**.

Step 6: Analyze Interactions
----------------------------

Compute the proximity between Targets and Effectors.

1.  Expand the **INTERACTIONS** block.
2.  Check the **Neighborhoods** box.
3.  Click the :icon:`plus,black` button next to **ISOTROPIC DISTANCE THRESHOLD**.
4.  Set **Distance** to ``30`` pixels.
5.  Click **Add**.
6.  Click **Submit**.

Step 7: Analyze Time-Series
---------------------------

Visualize the single-cell signals and the detected lysis events.

1.  Click the :icon:`cog-outline,black` button in the **Detect Events** section (Event Annotator settings).
2.  Configure the RGB representation if needed and click **Save**.
3.  Click the :icon:`eye,black` button (Event Annotator) to open the viewer.
4.  You can inspect the traces and see the detected lysis times.

.. note::
    Since we computed neighborhoods, you can also visualize the number of effectors in contact/proximity with each target cell. Look for the ``inclusive_count_neighborhood_(targets-effectors)_circle_30_px`` signal in the list.

.. figure:: _static/signal-annotator.gif
    :width: 800px
    :align: center
    :alt: signal_annotator

    Visualize single-cell signals (e.g. intensity, neighbors) with the signal annotator.

Step 8: Explore Results
-----------------------

1.  **Survival Analysis**:

    *   Go to the **Analyze** tab.
    *   Click **Plot survival**.
    *   Select **Population**: ``targets``.
    *   **Time of Interest**: ``t_lysis``.
    *   **Time of Reference**: ``0`` (beginning of experiment).
    *   Click **Submit** to view the Kaplan-Meier survival curve.

2.  **Signal Synchronization**:

    *   Click **Plot signals**.
    *   Select **Population**: ``targets``.
    *   **Class**: ``class_lysis``.
    *   **Time**: ``t_lysis`` (to align signals at the moment of death).
    *   Click **Submit** to view the average lysis signature.

Next Steps
----------

*   Learn more about :doc:`survival analysis <how-to-guides/basics/plot-survival>`.
*   Explore :doc:`measurement options <measure>`.
