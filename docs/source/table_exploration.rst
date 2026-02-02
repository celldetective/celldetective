Table Explorer (Table UI)
=======================

.. _table_exploration:

The Table Explorer (internally ``TableUI``) is a powerful interface for interacting with your measurement data. It allows you to visualize data structure, perform calculations, generate plots, and aggregate track information.

Overview
--------

The Table UI provides a spreadsheet-like view of your data with extensive capabilities for data manipulation and visualization. It is accessible from various parts of the application where measurement tables are displayed.

Menu Actions
------------

The menu bar provides several categories of actions:

File
~~~~
- **Save as...** (``Ctrl+S``): Save the current table to a CSV file.
- **Save inplace...**: Save the table back to its original location (if applicable). Use this only if you created new measurements that you would like to keep with the data (e.g., a difference of times).
- **Plot...** (``Ctrl+P``): Open the plotting interface for 1D/2D data visualization.
- **Plot instantaneous...** (``Ctrl+I``): Toggle between static plots and interactive track signal plots.
- **Collapse tracks...** (``Ctrl+G``): Open the dialogue to aggregate track data (see `Track Collapsing`_).
- **Collapse pairs in neighborhood...**: (For pair tables) Aggregate interaction data based on defined neighborhoods.
- **Group by frames...** (``Ctrl+T``): Aggregate data by time frame to see population-level temporal trends.
- **Query...**: Filter the table using SQL-like queries.

Edit
~~~~
- **Delete...** (``Del``): Delete selected columns.
- **Rename...**: Rename the selected column.

Math
~~~~
Perfom mathematical operations on columns:

- **Calibrate...** (``Ctrl+C``): Apply a calibration factor to a column.
- **Merge states...**: Merge multiple classification/status columns.
- **Differentiate...** (``Ctrl+D``): Compute the derivative of a column (requires tracks).
- **Absolute value...**: Compute the absolute value of a column.
- **Log (decimal)...**: Compute the base-10 logarithm of a column.
- **Divide/Multiply/Add/Subtract...**: Perform arithmetic operations between two selected columns.

Plotting
--------

The Table UI offers versatile plotting capabilities. Select columns in the table and use **File > Plot...** (``Ctrl+P``) to visualize them.

1D Plotting
~~~~~~~~~~~
If one column is selected (or no specific column), the 1D plot interface opens. It supports:

- **Distributions**: Histogram, KDE plot, ECDF plot.
- **Categorical**: Countplot, Swarm plot, Violin plot, Strip plot, Box plot, Boxenplot.
- **Stats**: Option to compute KS test p-values and Cliff's Delta effect size.
- **Grouping**: Select X, Y, and Hue variables to group data.

2D Plotting
~~~~~~~~~~~
If two columns are selected, a scatter plot is automatically generated comparing the two variables.

Time Series / Track Signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~
When viewing track data, you can visualize signals over time:
- **Plot instantaneous...** (``Ctrl+I``): Switches the plotting mode to show track signals when rows are selected, allowing you to see the temporal evolution of metrics for selected cells.

Track Collapsing
----------------

One of the most powerful features is the ability to aggregate data at the track level. Use **File > Collapse tracks...** (``Ctrl+G``) to open the projection mode dialog.

Global Operation
~~~~~~~~~~~~~~~~
Collapse the entire track into a single value using an aggregation function:
- **Operations**: `mean`, `median`, `min`, `max`, `first`, `last`, `prod`, `sum`.
- **Example**: Calculate the *max* intensity or *mean* speed of a cell over its entire track.

@ Event Time
~~~~~~~~~~~~
Extract measurement values at a specific event time.
- **Use case**: "What was the cell's area *at the moment of division* (t0)?"
- Requires event times (columns starting with `t_` or `t0`) to be present in the data.

Per Status
~~~~~~~~~~
Aggregate measurements independently for each cell state/status.
- **Use case**: "What is the mean speed of the cell while it is in *G1 phase* vs *S phase*?"
- Requires status/classification columns.
