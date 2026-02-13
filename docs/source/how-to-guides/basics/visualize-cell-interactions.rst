.. _visualize-cell-interactions:

How to visualize cell-cell interactions
---------------------------------------

This guide shows you how to use the **Pair Signal Viewer** to inspect the signals of interacting cell pairs.

Reference keys: :term:`neighborhood`, :term:`pair measurements`

**Prerequisite:** You must have computed :doc:`neighborhoods <measure-cell-interactions>` and enabled **MEASURE PAIRS**.

Open the Pair Signal Viewer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open the **Interactions** section in the control panel.

#. Click the :icon:`eye-check,black` button next to the **MEASURE PAIRS** checkbox (or the relevant pair protocol).

    .. note::
        The viewer adapts the standard Signal Annotator interface to show both reference and neighbor populations simultaneously.

Visualizing Pairs
~~~~~~~~~~~~~~~~~

The viewer displays the movie with both cell populations overlaid.

1.  **Select a Reference Cell**: Click on a cell from the reference population.
    
    *   All *other* reference cells will be hidden (colored black) to reduce clutter.
    *   The selected cell remains visible.
    *   Valid pairs involving this cell are displayed as **dynamic segments** connecting the reference cell to its neighbors.

2.  **Select a Pair**: Click on the **segment center** or directly on the **neighbor cell**.
    
    *   The signal plot panel (left) updates to show traces for this specific pair.

3.  **Inspect Signals**: The plot canvas displays:

    *   **Reference Signal**: The signal of the reference cell.
    *   **Neighbor Signal**: The signal of the neighbor cell.
    *   **Pair Signal**: Derived pair metrics (e.g., distance, angle) if selected.

    You can toggle visibility of each trace using the legend or dropdown menus.

Customizing the View
~~~~~~~~~~~~~~~~~~~~

*   **Color Coding**: Use the control panel to change the color coding for each population independently (e.g., by "class" or "status").
*   **Navigation**: Use the standard :ref:`Signal Annotator Shortcuts <ref_signal_annotator_shortcuts>` (Space to play, arrows to step).
