How to create a multi-label characteristic group from exclusive classifications
-------------------------------------------------------------------------------

This guide shows you how to assemble multiple one-hot-encoded classifications into a multi-label characteristic group where each state combination is a phenotype.

**Prerequisite**: the segmentation and measurements for at least one population.

Reference keys: :term:`single-cell measurement`, :term:`characteristic group`, :term:`phenotype`

**Step-by-step:**

#. Perform successive threshold classifications to isolate your phenotypes of interest from the rest. For each phenotype, classify the phenotype vs all other cells.

#. Click on the :icon:`table,#1565c0` :blue:`Explore table` button to open the table view.

#. Select your groups or status classification-like features to merge them into a single characteristic group by Ctrl+Left-Clicking on the column to combine.

#. Go to **Math > Merge states...**. Set a name for the new group. Check the feature names to merge.

#. Compute. The new column is added add the end of the table view.

#. Go to **File > Save inplace...** to write this new feature in all of the position tables.

#. Enter the :icon:`eye,black` viewer in the MEASURE section to explore the new group.