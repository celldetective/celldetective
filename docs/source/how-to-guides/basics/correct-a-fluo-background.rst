How to correct a fluorescent background
=======================================

This guide shows you how to subtract the background bulk fluorescence from a fluorescence image.

Reference keys: *background correction*

Correction protocol
~~~~~~~~~~~~~~~~~~~

#. Launch the software. Open a project.

#. Expand the **PROCESSING** block. In the **BACKGROUND CORRECTION** section, click on the Fit tab.

#. Set the fluorescence channel to correct.

#. Click on the :icon:`image-check,black` icon next to the threshold field to estimate this threshold value visually.

#. Use the threshold slider to select as many cells as possible (in purple) while leaving most of the background regions unselected (not purple).

#. Apply.

#. Set the ``paraboloid`` model.

#. Tick the subtract operation.

#. Do not clip.

See :py:func:`celldetective.preprocessing.field_correction` to understand what the threshold represents.
See :py:func:`celldetective.preprocessing.paraboloid` for a definition of the paraboloid model.

.. note::
	Not clipping allows you to perform quality control on the resulting intensities. If the model is poor you should see many negative values.

Apply correction
~~~~~~~~~~~~~~~~

#. Press :icon:`plus,#1565c0` :blue:`Add correction`.

#. Once all preprocessing protocols have been defined (background correction, other channel alignments), scroll down and press Submit. To apply the same protocol to all positions, accept the first popup.

#. At the end of the preprocessing change the movie stack prefix in the experiment configuration to ``Corrected_``.
