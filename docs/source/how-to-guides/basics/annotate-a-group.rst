How to annotate a group of cells
================================

This guide shows you how to create a characteristic group where each state is associated with a specific cell phenotype (within a population).

Reference keys: :term:`characteristic group`, :term:`phenotype`

**Prerequisite:** You have accurately segmented a cell population of interest, tracked it if relevant, and measured each single cell.

**Step-by-step:**

#. Go to the MEASURE section for the cell population of interest. Click on the :icon:`eye-check,black` icon on the right side to view single-cell measurements in-situ.

#. Set a frame of interest in the top right part of the viewer, using the slider.

#. Set a channel of interest, adjust the contrast of the image.

#. On the top-left side, create a new characteristic group by pressing the :icon:`plus,black` icon.

#. Give a name to the new group, press submit.

#. All single cells should have a blue circle. It means that the associated phenotype for this group is initialized at 0.

#. Click on a single cell to modify its phenotype.

#. Press :icon:`redo-variant,black` correct on the left side to change the phenotype. Set all similar cells to 1.

#. Repeat the same operation for the next phenotype (2). Annotate **all cells** in the current frame.

#. Once you have fully annotated the current frame for this characteristic group, press the :icon:`export,black` button at the bottom of the left side to export an annotation (in ``.npy`` format).

#. To update the position table with the new group, press Save. There is no time propagation, so the phenotype of cells from other frames will be 0, if you do not annotate them.

