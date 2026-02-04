UI Menus & Shortcuts
====================

This reference lists the available menu actions and keyboard shortcuts for various Celldetective interfaces.

.. _ref_table_explorer_menus:

Table Explorer (TableUI)
------------------------

**File Menu**

*   **Save as...** (``Ctrl+S``): Export table to CSV.
*   **Save inplace...**: Overwrite the original source file.
*   **Plot...** (``Ctrl+P``): Open the Plotting Interface.
*   **Plot instantaneous...** (``Ctrl+I``): Toggle row-selection plotting for track signals.
*   **Collapse tracks...** (``Ctrl+G``): Open the Track Collapsing dialog.
*   **Collapse pairs in neighborhood...**: Aggregate interaction data (Pair tables only).
*   **Group by frames...** (``Ctrl+T``): Aggregate data by time frame.
*   **Query...**: Filter rows using Pandas query syntax.

**Edit Menu**

*   **Delete...** (``Del``): Delete selected columns.
*   **Rename...**: Rename the selected column.

**Math Menu**

*   **Calibrate...** (``Ctrl+C``): Multiply a column by a factor.
*   **Merge states...**: Combine multiple status columns.
*   **Differentiate...** (``Ctrl+D``): Compute derivatives (e.g., speed from position).
    *   *Options*: Window size, Mode (forward/backward/central).
*   **Absolute value...**: Compute ``abs(x)``.
*   **Log (decimal)...**: Compute ``log10(x)``.
*   **Arithmetic**: Add, Subtract, Multiply, or Divide two columns.
