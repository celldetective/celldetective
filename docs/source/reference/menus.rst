UI Menus & Shortcuts
====================

This reference lists the available menu actions and keyboard shortcuts for various Celldetective interfaces.

.. _ref_table_explorer_menus:

Table Explorer (TableUI)
------------------------

**File Menu**

*   **Save as...** (:kbd:`Ctrl+S`): Export table to CSV.
*   **Save inplace...**: Overwrite the original source file.
*   **Plot...** (:kbd:`Ctrl+P`): Open the Plotting Interface.
*   **Plot instantaneous...** (:kbd:`Ctrl+I`): Toggle row-selection plotting for track signals.
*   **Collapse tracks...** (:kbd:`Ctrl+G`): Open the Track Collapsing dialog.
*   **Collapse pairs in neighborhood...**: Aggregate interaction data (Pair tables only).
*   **Group by frames...** (:kbd:`Ctrl+T`): Aggregate data by time frame.
*   **Query...**: Filter rows using Pandas query syntax.

**Edit Menu**

*   **Delete...** (:kbd:`Del`): Delete selected columns.
*   **Rename...**: Rename the selected column.

**Math Menu**

*   **Calibrate...** (:kbd:`Ctrl+C`): Multiply a column by a factor.
*   **Merge states...**: Combine multiple status columns.
*   **Differentiate...** (:kbd:`Ctrl+D`): Compute derivatives (e.g., speed from position).

    *   *Options*: Window size, Mode (forward/backward/central).
    
*   **Absolute value...**: Compute ``abs(x)``.
*   **Log (decimal)...**: Compute ``log10(x)``.
*   **Arithmetic**: Add, Subtract, Multiply, or Divide two columns.

.. _ref_signal_annotator_shortcuts:

Signal Annotator / Viewer
-------------------------

**Keyboard Shortcuts**

| Shortcut            | Action                                            |
+=====================+===================================================+
| :kbd:`Space`        | Play / Stop animation                             |
+---------------------+---------------------------------------------------+
| :kbd:`f`            | Jump to **First** frame                           |
+---------------------+---------------------------------------------------+
| :kbd:`l`            | Jump to **Last** frame                            |
+---------------------+---------------------------------------------------+
| :kbd:`Ctrl+P`       | Open interactive signal plotter                   |
+---------------------+---------------------------------------------------+
| :kbd:`Esc`          | Cancel cell selection                             |
+---------------------+---------------------------------------------------+
| :kbd:`Del`          | Mark selected cell for deletion (cannot undo)     |
+---------------------+---------------------------------------------------+
| :kbd:`n`            | Reset cell class to "No Event"                    |
+---------------------+---------------------------------------------------+

**Mouse Interactions**

*   **Left Click (Cell)**: Select a cell to view its signals.
*   **Left Click (Timeline)**: Jump to a specific timepoint.

.. _ref_interactive_plotter_shortcuts:

Interactive Plotter
-------------------

Accessed via :kbd:`Ctrl+P` from the Signal Annotator or Table Explorer.

**Keyboard Shortcuts**

+---------------------+---------------------------------------------------+
| Shortcut            | Action                                            |
+=====================+===================================================+
| :kbd:`Left / Right` | Shift event time for selected traces              |
+---------------------+---------------------------------------------------+
| :kbd:`Ctrl+S`       | Save changes                                      |
+---------------------+---------------------------------------------------+

**Mouse Interactions**

*   **Click + Drag**: Draw a box to select multiple traces (highlighted in red).
