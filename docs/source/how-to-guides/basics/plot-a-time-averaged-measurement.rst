How to compare collapsed measurements across conditions
=======================================================

This guide explains how to aggregate single-cell track data (e.g., computing the time-average of a signal) and compare it across different experimental conditions using Boxplots and Stripplots.

**Prerequisites**

*   You have **tracked** cells and generated measurement tables.
*   Your experiment has **conditions** (e.g., different treatments per well) loaded as columns (e.g., derived from ``[Labels]`` in ``config.ini``).

**Step 1: Open the Table Explorer**

1.  Select your experiment or specific position.
2.  Open the **Analysis** tab.
3.  Click **Table Explorer** (or access it via the Measurement results).

**Step 2: Collapse Tracks**

Transform the data from "one row per timepoint" to "one row per track".

1.  Press ``Ctrl+G`` or select **Table > Collapse tracks...** (also the :icon:`arrow-collapse-vertical,black` button above the table).
2.  In the dialog:
    *   **Operation**: Select ``mean`` (for time-average), ``max``, or another statistic.
3.  Click **OK**.
    *   *Result*: The table updates. Rows now represent individual cells (Tracks). New columns like ``mean_mean_intensity`` appear. Columns that are constant per track (like ``well_id`` or ``treatment``) are preserved.

**Step 3: Plot by Condition**

1.  Press ``Ctrl+I`` or select **Plot > Distributions and statistics...** (also the :icon:`chart-bell-curve,black` button above the table). The **Set 1D plot parameters** window opens.
2.  Set **x** to the condition column (categorical, e.g., ``well_name``) and **y** to the measurement column (numerical, e.g., ``mean_mean_intensity``).
3.  (Optional) Use the **hue** dropdown to subdivide the plot by another category (e.g., ``replicate``).

**Step 4: Configure the Visualization**

1.  Under **Representations**, click the **boxplot** card. Click the **strip** card as well to show the individual data points on top of the boxes. Cards can be combined.
2.  (Optional) Under **Statistical Tests**, click the cards of the tests to compute between the groups: the KS test p-value and the Cliff's Delta effect size. Each opens a table of pairwise comparisons (see :ref:`table_exploration`).
3.  Press **set**.

**Step 5: Export**

*   Click the **Save Icon** in the plot window to export the figure.
*   Or press ``Ctrl+S`` (**File > Save as...**) in the Table Explorer to save the collapsed data table to CSV.