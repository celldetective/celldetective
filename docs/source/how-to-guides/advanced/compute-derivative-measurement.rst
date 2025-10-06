How to differentiate a feature
------------------------------

This guide shows you how to compute and save the derivative of a single-cell measurement.

**Prerequisite**: the segmentation, tracking and measurement steps must be done for at least one population.

Reference keys: *single-cell measurement*

**Step-by-step:**

#. Open a project.

#. Set in the header the wells and positions for which you want to compute that derivative.

#. Expand the block associated with your cell population.

#. Click on the :icon:`table_chart,#1565c0` **Explore table** button to open the table view.

#. Go to **Math > Differentiate...**. If the option is disabled, check that you have cell tracks.

#. Set up the derivative computation (see :py:func:`celldetective.utils.derivative` for implementation details):

   - Select the measurement of interest :math:`m` (e.g. ``area``).
   - Set the window size to ``1``.
   - Set the derivative mode to ``forward``.

#. Compute. A new feature **d/dt.m**, is written at the end of the table.

#. Go to **File > Save inplace...** to write this new feature in all of the position tables.
