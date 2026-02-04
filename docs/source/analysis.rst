Analysis
========

.. _analysis:


Mean signal response
--------------------

Go to the ``Analyze`` tab, click on the ``plot signals`` button. Configure the cell population of interest, set the class to segregate the cells (the naming convention should follow what you annotated or correct in the signal annotator). Set the associated event time. You can also show the signals using an absolute timepoint to synchronize the signals. Click on ``Submit``. A second window asks for the signal of interest. Pick one in the list and validate. 


.. figure:: _static/mean-signal-response.png
    :align: center
    :alt: mean_signal_response
    
    **An interface to collapse signals with respect to events.** a) The cell population, class to
    segregate the cells and time of the event is set. Upon submission, a second window asks
    to select a single signal among the signals measured for that cell population. The control
    panel header informs about the data selection, between a single position, multiple positions
    and multiple wells. b) In multiple position mode, the mean signal trace (plus or minus the
    standard deviation) is generated for each position, as well as a pool trace pooling cells from
    all positions. The cells can be filtered between the ones that experienced the event (b) and
    the ones that did not (c), affecting the mean traces.



Survival response
-----------------

Configure a survival function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Go to the ``Analyze`` tab and click on the ``plot survival`` button to open the configuration window.

**1. Data Selection**

   * **Population**: Select the cell population to analyze (e.g., "targets", "effectors"). You can also select "pairs" to analyze interaction durations (requires prior pair measurement).
   * **Time of Reference**: Define the starting point (:math:`T=0`) for the survival analysis.
     
     * `t_firstdetection`: The time the cell first appeared in the movie.
     * `t0`: Absolute time 0 of the experiment.
     * Any other event time column (e.g., `t_mitosis`).
   * **Time of Interest**: Define the event that marks the "death" or end of the state being measured.
     
     * `t_death`: Time of cell death.
     * Any other event time column.
     * **Note**: The reference time and time of interest must be different.

**2. Filtering (Optional)**

   * **Select cells with query**: Enter a Pandas query string to filter the cells before analysis.
     
     * *Example*: ``TRACK_ID > 10`` or ``mean_intensity > 500``
   * **Cut obs. time [min]**: Set a maximum observation duration. Events occurring after this time will be considered right-censored (i.e., the cell "survived" past this point).

**3. Visualization Settings**

   * **Time calibration (frame to min)**: Conversion factor from frames to minutes. Defaults to the experiment setting.
   * **Cmap**: Select the colormap for the output curves.

**4. Run Analysis**

   Click **Submit** to generate the survival curves.

Output
^^^^^^

A new window will appear displaying the Kaplan-Meier survival curves.

* **Multiple Positions**: Shows individual curves for each position (with 95% confidence intervals) and a pooled curve for the well.
* **Multiple Wells**: Shows pooled curves for each well to allow comparison between conditions.



.. figure:: _static/survival-analysis.png
    :align: center
    :alt: survival_analysis
    
    **An interface to represent survival functions at multiple scales.** a) An analysis modules pilots
    the making of survival functions. A cell population of interest is set, a reference time and an
    event time are picked from the list of events available for that population. The control panel
    header informs about the data selection, between a single position, multiple positions and
    multiple wells. b) In multiple-positions mode, each position’s survival function is plotted with
    its 95 % confidence interval, as well as the pooled survival function for the well. Positions
    can be added or removed from the plot. c) In multiple-wells mode, individual positions are
    still shown but without the 95 % confidence interval. Emphasis is put on the pooled survival
    functions that can be compared across wells. As before, wells can be added or removed
    from the plot.



Start exploring data
--------------------

For a detailed guide on how to interact with the data tables, perform calculations, and collapse tracks, please refer to the :ref:`table_exploration` page.

