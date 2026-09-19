Preprocessing
=============

.. _preprocessing:


Overview
--------

Preprocessing is an essential step to prepare your microscopy data for analysis in Celldetective. It includes both off-software and in-software methods to align, correct, and optimize your stacks for segmentation and downstream measurements.


Registration
------------

Stage drift and shaking make still objects move across the frames, which misleads tracking and blurs time-averaged measurements. We highly recommend registering the movies before segmentation. Celldetective registers stacks by phase cross-correlation, directly in the **Preprocessing** module (see the table below). The stacks can also be aligned before being imported, with external tools like Fiji (ImageJ), which can correct rotations too.

.. seealso::
    :doc:`how-to-guides/basics/register-stacks` to register the stacks in Celldetective, and :doc:`how-to-guides/basics/register-stacks-with-fiji` for a step-by-step guide on using the customized Fiji macro for batch registration.



In-software preprocessing
--------------------------

The **Preprocessing** module lets you batch-correct stacks directly within Celldetective. Corrected stacks are saved with the prefix ``Corrected_``. The corrections added to the list are applied in order: the first one reads the raw movie, the next ones correct the ``Corrected_`` stack in place. Each step is recorded with its parameters in the ``log_preprocessing.txt`` file of the position.

.. list-table::
   :widths: 30 50 20
   :header-rows: 1

   * - Task
     - Description
     - Guide
   * - Background correction (Model Fit)
     - Fits a 2D surface to the background, excluding cells. Best for fluorescence images.
     - :doc:`how-to <how-to-guides/basics/correct-a-fluo-background>`
   * - Background correction (Model Free)
     - Computes a median background from multiple positions or timeframes. Best for brightfield/reflection microscopy.
     - :doc:`how-to <how-to-guides/basics/perform-model-free-background-correction>`
   * - Channel offset correction
     - Aligns channels that have pixel shifts between modalities.
     - :doc:`how-to <how-to-guides/basics/align-channels>`
   * - Stack registration
     - Corrects the drift of the field over time, estimated on one channel and applied to all.
     - :doc:`how-to <how-to-guides/basics/register-stacks>`

For a full list of parameters (thresholds, tile options, etc.), see the :ref:`Preprocessing Protocols Reference <ref_preprocessing_settings>`.


Bibliography
------------

 Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012). https://doi.org/10.1038/nmeth.2019
