How to perform a model-free background correction
-------------------------------------------------

This guide shows you how to sample a background model across multiple positions and correct your images. This guide is particularly applicable to label-free images such as brightfield and RICM.

Reference keys: :term:`background correction`

Correction protocol
~~~~~~~~~~~~~~~~~~~

#. Launch the software. Open a project.

#. Expand the **PROCESSING** block. In the **BACKGROUND CORRECTION** section, click on the Model-free tab.

#. Set the channel to correct.

#. If your image stack is a timeseries, set the stack mode to timeseries and adjust the time range to take the frames with the least amount of cells (often the first few frames). Else, tick tiles (all frames will be used).

#. Click on the :icon:`image-check,black` icon next to the threshold field to estimate this threshold value visually.

#. Use the threshold slider to select as many cells as possible (in purple) while leaving most of the background regions unselected (not purple).

#. In the QC section, select a well of interest and click on the :icon:`image-check,black` to perform a visual assessment of the sampled background for this well. Repeat for all wells of interest.

#. Tick the "Optimize for each frame?" option.

#. Leave default values:

	- coefficient range between 0.95 and 1.05
	- number of coefficients of 100

#. Set the black level of the image in offset (e.g. minimum pixel value observed if the microscope side port shutter or light source shutter is closed).

#. Tick the ``Divide`` option.


Apply correction
~~~~~~~~~~~~~~~~

#. Press :icon:`plus,#1565c0` :blue:`Add correction`.

#. Once all preprocessing protocols have been defined (background correction, other channel alignments), scroll down and press Submit. To apply the same protocol to all positions, accept the first popup.

#. At the end of the preprocessing change the movie stack prefix in the experiment configuration to ``Corrected_``.
