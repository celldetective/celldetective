How to synchronize single-cell timeseries over a population
-----------------------------------------------------------

This guide shows you how to collapse single-cell signal traces into population-averaged time series, aligned to a reference event time.

Reference keys: **mean signal**, **signal response**, **population average**

**Prerequisite:** You have segmented, tracked, measured, and annotated events for a cell population.


Step 1: Configure the signal plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../../_static/figures/synchronized-signals.svg
    :width: 100%
    :target: ../../_static/figures/synchronized-signals.svg
    :align: center
    :alt: The Analyze tab, the signal plot options, the feature choice and the plot

    **Synchronized signals.** The mean RICM intensity of the effector cells of the spreading demo, aligned on their spreading time.

#. Go to the **Analyze** tab and click **plot signals** (1).

#. In the **Options** window, configure the following:

    *   **population**: The cell population (or pair) to analyze (2).
    *   **class**: The column used to segregate cells (e.g., ``class_spread``). This determines the "event" vs "no event" grouping.
    *   **time of interest**: The event time column (e.g., ``t_spread``) used to align the traces (t=0).
    *   **cmap**: (Optional) Select a colormap for the curves.
    *   **absolute time**: Check this to ignore the event time and synchronize signals using an absolute frame number (set via the slider) (3).
    *   **select cells with query**: (Optional) Enter a pandas query to filter cells (e.g., ``TRACK_ID > 10``) (4).
    *   **time calibration (frame to min)**: Frame-to-minute conversion factor.
    *   **pool projection**: Choose how to aggregate the population (``mean`` or ``median``).
    *   **min # cells for pool**: Minimum number of cells required to calculate a valid data point.

#. Click **Submit** (5).


Step 2: Select the signal
~~~~~~~~~~~~~~~~~~~~~~~~~

#. A second window appears ("Select numeric feature").

#. Select the measurement you want to plot (e.g., ``adhesion_channel_mean``) (6).

#. Click **set** (7).


Step 3: Interact with the plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The plot window displays the synchronized signals. You can interact with it using the controls:

*   **Grouping** (8): Switch between **well** (pooled per well), **position** (per position), or **both**.

*   **Toolbar Buttons** (9):

    *   **Legend**: Toggle legend visibility.
    *   **Log**: Toggle log-scale for Y-axis.
    *   **CI**: Show or hide the band around the curves (see **Band** below).
    *   **Cell lines**: Toggle display of individual single-cell traces.
    *   **Export**: Save the figure or export tabular data.
    *   **Compare** (balance icon): Test the wells against each other (see below).

*   **Class of interest** (10): Filter the displayed curves by class:

    *   ``*``: Show all cells.
    *   ``event``: Show only cells belonging to the event class (class 0).
    *   ``no event``: Show only cells belonging to the non-event class (class 1).
    
*   **Rescale** (11): Set a **scaling factor** for the Y-axis and press **rescale**.
*   **Single-cell signal alpha**: Adjust the transparency of individual cell traces.
*   **Select position** (12): Choose which positions/wells to display, either **by name** (checkboxes) or **spatially** (clicking on the position map).

*   **Band**: What the band around a curve shows:

    *   ``95% CI of the mean`` (default): the uncertainty of the mean curve, counting each
        cell as independent. Offered with the ``mean`` projection only.
    *   ``spread across cells``: the SD around a mean, the median absolute deviation around a
        median. It describes the cells: it does not shrink with more cells and is not an
        uncertainty.

Compare the wells
~~~~~~~~~~~~~~~~~

Click **Compare** to test whether the wells respond differently. Each cell gets a
**response**: its mean over a window after the synchronizing time minus its mean over a
baseline window before it (by default the 10 frames on each side, in minutes). Cells observed
over less than half of either window are left out.

Two matrices open, as for the tests of the table explorer: the one-sided KS test p-values
and the effect sizes (Cliff's delta) of the responses between wells.

Responses are only defined for cells aligned on their event: show the class ``event``, or
synchronize on an absolute time to include all cells.

.. note::

   The tests count each cell as independent, so they compare **wells**. When each condition
   has one well, a difference may come from the well rather than its condition: repeat the
   experiment to conclude on a condition.

