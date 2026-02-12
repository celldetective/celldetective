How to compute the duration between two events
==============================================

This guide shows you how to compute the time difference between two events (e.g. ``t_spread`` - ``t_firstdetection``) and visualize it as a stripplot against a biological condition.

**Prerequisite**: You must have a table with two numeric columns representing the time of two events (e.g. ``t_firstdetection`` and ``t_spread``).

Reference keys: *event duration*

**Step-by-step:**

1. Open your project and load the experiment.

2. Select the wells and positions of interest in the **Control Panel**.

3. Click on the :icon:`table,#1565c0` **Explore table** button to open the table view.

4. Go to **Math > Subtract...**.

5. In the dialog:

    - Set **Column 1** to the later event time (e.g., ``t_spread``).
    - Set **Column 2** to the earlier event time (e.g., ``t_firstdetection``).
    - Click **Compute**.

    A new column named ``t_spread-t_firstdetection`` (or similar) will be added to the table.

6. (Optional) Rename the new column for clarity:

    - Select the new column header.
    - Go to **Edit > Rename...**.
    - Enter a new name, for example ``hovering duration``.

7. (For tracked data) Go to **File > Collapse tracks...**:

    - In the dialog, keep **global operation** selected.
    - Choose **mean** (or **first**) as the operation to ensure one value per track.
    - Click **Apply**.

    A **new table window** will open with the collapsed data (one row per track). Continue the next steps in this new window.

8. In the new window, go to **File > Plot...** (or press ``Ctrl+p``).

9. In the **Set 1D plot parameters** dialog:

    - Check the representation you want, for example **strip** (stripplot), **boxplot**, or **violin**.
    - Set **x** to the grouping variable (e.g., ``well_name`` or ``antibody``).
    - Set **y** to the duration column (e.g., ``hovering duration``).
    - (Optional) Set **hue** to color by another variable.

10. Click **set**.

    A new plot window will open with the requested representation.
