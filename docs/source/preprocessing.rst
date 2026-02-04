Preprocessing
=============

.. _preprocessing:


Overview
--------

Preprocessing is an essential step to prepare your microscopy data for analysis in Celldetective. It includes both off-software and in-software methods to align, correct, and optimize your stacks for segmentation and downstream measurements.


Off-software preprocessing
--------------------------

Registration
~~~~~~~~~~~~

We highly recommend aligning your movies before using Celldetective. A common tool for this is the **Linear Stack Alignment with SIFT Multichannel** plugin available in Fiji [#]_ , which can be activated by enabling the PTBIOP update site (see discussion here_).

.. _here: https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16

To facilitate this step, we provide `a macro`_ that can be reused for preprocessing tasks in the ``movie/`` subfolder of each position folder.

.. _`a macro`: align_macro.html


In-software preprocessing
-------------------------

Sometimes, preprocessing your images directly within Celldetective can simplify segmentation and produce more controlled measurements. The **Preprocessing** module allows you to batch-correct stacks through the following steps:


**Workflow: Applying a correction**

1.  **Select a target channel** and define correction parameters (manually or visually).
2.  **Add the correction protocol** to the list.
3.  Repeat for other channels if needed.
4.  **Submit** to apply corrections.

The corrected stacks are saved with the prefix ``Corrected_``.

Background correction
~~~~~~~~~~~~~~~~~~~~~

**Concept**

Background correction removes uneven illumination or artifacts. Celldetective offers two approaches:

1.  **Model Fit**: Fits a 2D mathematical surface (plane/paraboloid) to the image, excluding cells. Best for single images with clear background.
2.  **Model Free**: Computes a median background from multiple positions or timeframes. Best for batch processing well plates.

For technical details on these methods (Thresholds, Tile options), see the :ref:`Preprocessing Protocols Reference <ref_preprocessing_settings>`.


Channel offset correction
~~~~~~~~~~~~~~~~~~~~~~~~~

In some optical microscopy setups, offsets between modalities can affect intensity measurements by misaligning channel data.

With the **channel offset correction** module, you can:

#. Estimate the offset between two channels.

#. Use the Viewer to visualize a reference channel in grayscale and the overlayed channel to correct in blue.

#. Adjust the overlay position using keyboard arrows until alignment is satisfactory.

#. Apply the correction and add it to the protocol list.


Bibliography
------------

.. [#] Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019
