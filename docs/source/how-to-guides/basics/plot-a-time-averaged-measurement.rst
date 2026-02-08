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

1.  Press ``Ctrl+G`` or select **File > Collapse tracks...**.
2.  In the dialog:
    *   **Operation**: Select ``mean`` (for time-average), ``max``, or another statistic.
3.  Click **OK**.
    *   *Result*: The table updates. Rows now represent individual cells (Tracks). New columns like ``mean_mean_intensity`` appear. Columns that are constant per track (like ``well_id`` or ``treatment``) are preserved.

**Step 3: Plot by Condition**

1.  Select the **Condition Column** first (Categorical, e.g., ``well_name``).
2.  Hold ``Ctrl`` and select the **Measurement Column** (Numerical, e.g., ``mean_mean_intensity``).
    *   *Note*: The order of selection often determines X and Y axes. Condition (X) vs Measurement (Y).
3.  Press ``Ctrl+P`` or select **File > Plot...**.

**Step 4: Configure the Visualization**

1.  The plotting window opens.
2.  In the **Plot Type** dropdown, select **Box Plot**.
3.  To show individual data points, check the **Stripplot** option (or select **Strip Plot** to view it alone).
4.  (Optional) Use the **Hue** dropdown to subdivide the boxplots by another category (e.g., ``replicate``).
5.  Check **Show stats** if you wish to display statistical comparisons (e.g., p-values between boxes).

**Step 5: Export**

*   Click the **Save Icon** in the plot window to export the figure.
*   Or press ``Ctrl+S`` in the Table Explorer to save the collapsed data table to CSV.