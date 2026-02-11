How to align two channels
=========================

This guide shows you how to correct a spatial offset between two channels.

Reference keys: :term:`preprocessing`, :term:`alignment`

Alignment protocol
~~~~~~~~~~~~~~~~~~

#. Launch the software. Open a project.

#. Expand the **PROCESSING** block, scroll down to **CHANNEL OFFSET CORRECTION**.

#. Click on the :icon:`image-check,black` icon to set visually the shift values with a dedicated viewer.

#. Set the overlay transparency slider to 0.

#. Set the "reference" channel (Channel). Adjust the contrast to over saturate contours.

#. Set the overlay transparency slider to 1.

#. Select your overlay channel (the channel image to move relative to the first). Adjust the contrast to saturate contours.

#. Set the overlay transparency slider to 0.5.

#. Use keyboard arrows to move the overlay image and achieve perfect alignment in most places (see note). To achieve higher precision, fine-tune the shift values in the respective shift fields (and apply to view the result).

#. Press set.

.. note::
	A good alignment at the center does not guarantee a good alignment at the edges. This correction method only allows for 2D rigid translations, which may not be true for your data.

Apply correction
~~~~~~~~~~~~~~~~

#. Press :icon:`plus,#1565c0` :blue:`Add correction`.

#. Once all preprocessing protocols have been defined (background correction, other channel alignments), press Submit. To apply the same protocol to all positions, accept the first popup.

#. At the end of the preprocessing change the movie stack prefix in the experiment configuration to ``Corrected_``.
