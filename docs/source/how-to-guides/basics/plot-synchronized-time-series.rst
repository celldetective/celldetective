How to synchronize single-cell timeseries over a population
-----------------------------------------------------------

This guide shows you how to collapse single-cell signal traces into population-averaged time series, aligned to a reference event time.

Reference keys: *mean signal*, *signal response*, *synchronize*, *population average*

**Prerequisite:** You have segmented, tracked, measured, and annotated events for a cell population.


Step 1: Configure the signal plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Go to the **Analyze** tab and click **Plot signals**.

#. In the **Options** window, configure the following:

    *   **Population**: The cell population (or pair) to analyze.
    *   **Class**: The column used to segregate cells (e.g., ``class_event``). This determines the "event" vs "no event" grouping.
    *   **Time of interest**: The event time column (e.g., ``t_event``) used to align the traces (t=0).
    *   **Cmap**: (Optional) Select a colormap for the curves.
    *   **Absolute time**: Check this to ignore the event time and synchronize signals using an absolute frame number (set via the slider).
    *   **Query**: (Optional) Enter a pandas query to filter cells (e.g., ``TRACK_ID > 10``).
    *   **Time calibration**: Frame-to-minute conversion factor.
    *   **Pool projection**: Choose how to aggregate the population (``mean`` or ``median``).
    *   **Min # cells for pool**: Minimum number of cells required to calculate a valid data point.

#. Click **Submit**.


Step 2: Select the signal
~~~~~~~~~~~~~~~~~~~~~~~~~

#. A second window appears ("Select numeric feature").

#. Select the measurement you want to plot (e.g., ``mean_intensity``).

#. Click **Set**.


Step 3: Interact with the plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plot window displays the synchronized signals. You can interact with it using the controls:

*   **Grouping**: Switch between **Well** (pooled per well), **Position** (per position), or **Both**.

*   **Toolbar Buttons**:

    *   **Legend**: Toggle legend visibility.
    *   **Log**: Toggle log-scale for Y-axis.
    *   **CI**: Toggle 95% confidence intervals.
    *   **Cell lines**: Toggle display of individual single-cell traces.
    *   **Export**: Save the figure or export tabular data.

*   **Class of interest**: Filter the displayed curves by class:

    *   ``*``: Show all cells.
    *   ``event``: Show only cells belonging to the event class (class 0).
    *   ``no event``: Show only cells belonging to the non-event class (class 1).
    
*   **Rescale**: Manually set a scaling factor for the Y-axis.
*   **Single-cell signal alpha**: Adjust the transparency of individual cell traces.
*   **Select position**: Choose which positions/wells to display, either **by name** (checkboxes) or **spatially** (clicking on the position map).