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

We highly recommend aligning your movies before using Celldetective to correct for stage drift. This is typically done using external tools like Fiji (ImageJ).

.. seealso::
    :doc:`how-to-guides/basics/register-stacks-with-fiji` for a step-by-step guide on using the customized Fiji macro for batch registration.



In-software preprocessing
--------------------------

The **Preprocessing** module lets you batch-correct stacks directly within Celldetective. Corrected stacks are saved with the prefix ``Corrected_``.

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Task
     - Description
     - Guide
   * - Background correction (Model Fit)
     - Fits a 2D surface to the background, excluding cells. Best for single images with clear background.
     - :doc:`how-to <how-to-guides/basics/correct-a-fluo-background>`
   * - Background correction (Model Free)
     - Computes a median background from multiple positions or timeframes. Best for batch processing well plates.
     - :doc:`how-to <how-to-guides/basics/perform-model-free-background-correction>`
   * - Channel offset correction
     - Aligns channels that have pixel shifts between modalities.
     - :doc:`how-to <how-to-guides/basics/align-channels>`

For a full list of parameters (thresholds, tile options, etc.), see the :ref:`Preprocessing Protocols Reference <ref_preprocessing_settings>`.


Bibliography
------------

 Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019
