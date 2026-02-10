Segmentation
============

.. _segment:


I/O
---

The **segmentation module** takes a stack of microscopy images as input and produces a stack of **instance segmentation masks**, delineating each cell in the images. The results are saved frame by frame in a ``labels_*population*`` subfolder within each position folder.


Overview
--------

The process of instance segmentation takes an image (multichannel or not) as its input and yields a label image (2D), where each segmented object is attributed a single label. 

In Celldetective, you may want to specifically segment up to two cell populations of interest on the images (typically target and effector cells but it could be anything). Segmentation can be carried out using traditional segmentation pipelines (based on filters and thresholds) or using Deep-learning models trained for such task. Celldetective proposes both options and allows a cross-talk between the two. As illustrated below, the output of a traditional segmentation can be corrected and used as an input to a DL model directly in Celldetective. That is only one of many paths to perform segmentation in Celldetective.

.. figure:: _static/segmentation-options.png
    :align: center
    :alt: seg_options
    
    **Overview of segmentation options in Celldetective.** Celldetective provides several entry points (black arrows) to perform segmentation, with the intent of segmenting specifically a cell population (left : effectors, right : targets). The masks output from each segmentation technique can be visualized and manually corrected in napari. Exporting these corrections into a paired image and masks dataset can be used either to fit a generalist model (transfer learning) or train one from scratch. Once the segmentation is satisfactory enough, the user can decide to proceed with the tracking and measurement modules.


Traditional segmentation
------------------------

In many applications, cell or nucleus segmentation can be achieved through the use of filters and thresholds, without having to resort to a Deep Learning model. Adapting such a model to a new system can be time-consuming and computationally expensive, as it usually requires numerous annotations. To ensure a user-friendly experience with Celldetective, we developed a robust framework for traditional segmentation as a potent alternative to calling a Deep Learning model.

We call this UI the ``Threshold Configuration Wizard`` (TCW). This interface allows you to interactively build a segmentation pipeline step-by-step.

.. image:: _static/tcw.png
    :align: center
    :alt: threshold_config_wizard

*The Threshold Configuration Wizard interface showing preprocessing, thresholding, and object detection controls.*

The wizard guides you through four stages:

1. **Preprocessing** — enhance the image with filters (``gauss``, ``median``, ``std``, etc.) to make objects easier to detect.
2. **Thresholding** — binarize the image to separate foreground from background.
3. **Object Detection** — split touching objects using a watershed or label all connected components.
4. **Property Filtering** — remove false positives based on morphology or intensity queries (e.g., ``area > 100``).

The pipeline can be saved as a ``.json`` config file, which can be loaded later via the **Upload Model** window.

For a complete step-by-step walkthrough, see :doc:`How to segment with the Threshold Configuration Wizard <how-to-guides/basics/segment-with-threshold-wizard>`.



Deep learning segmentation
--------------------------

Models
~~~~~~

Celldetective ships with Deep-learning segmentation models trained with the :term:`StarDist` [#]_ or :term:`Cellpose` [#]_ [#]_ algorithm. They are split in two families: 

#. **Generalist models** — models published in the literature that have been trained on thousands of images with one or two channels, on general tasks such as segmenting all nuclei visible on the images. In some cases, more than one modality was passed in the channel slots during training to force the model to generalize and be less sensitive to the modality. 
#. **Population-specific models** — models that we trained from scratch on brand new multimodal data to achieve more specific tasks such as detecting the nuclei of a population in the presence of another. In this configuration, accurate segmentation often requires to look at multiple channels at once, *i.e.* performing a multimodal interpretation.


.. figure:: _static/table-generalist-models.png
    :align: center
    :alt: table_generalist
    
    **Generalist models.** This table lists the different generalist models (:term:`Cellpose` or :term:`StarDist`) which can be called natively in Celldetective. The images have been sampled from their respective datasets, cropped to ( 200 × 200 ) px and rescaled homogeneously to fit in the table.


.. figure:: _static/target-models.png
    :align: center
    :alt: table_target_models
    
    **Target models.** MCF-7 nuclei segmentation models that we developed for our application. The models have been trained on the ``db_mcf7_nuclei_w_primary_NK`` dataset available in Zenodo.

.. figure:: _static/effector-models.png
    :align: center
    :alt: table_effector_models
    
    **Effector models.** Primary NK segmentation models that we developed for our application. The models have been trained on the ``db_primary_NK_w_mcf7`` dataset available in Zenodo.


Importing and applying models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models are imported via the :icon:`upload,black` button in the Segmentation panel. This creates a configuration file that maps your experiment's channels to the model's expected inputs, including spatial calibration and normalization.

For a detailed list of all import parameters, see the :ref:`Segmentation Data Import Reference <ref_segmentation_settings>`.

For a complete step-by-step walkthrough (including generalist model configuration), see :doc:`How to apply a segmentation model <how-to-guides/basics/apply-a-segmentation-model>`.



Mask visualization and correction
---------------------------------

Once a position is segmented, the results can be visualized in **napari** by clicking the :icon:`eye,black` button in the segmentation section. This overlays the segmented masks on the original images.

With napari, segmentation mistakes can be corrected using the brush, eraser, and fill tools. Celldetective provides two plugins:

#. **Save the modified labels** — overwrite the masks in place.
#. **Export a training sample** — create an annotated pair (image + mask) to train a Deep Learning model on your data.

For a step-by-step annotation workflow, see :doc:`How to annotate for segmentation <how-to-guides/basics/annotate-for-segmentation>`.
To train a model on your annotations, see :doc:`How to train a segmentation model <how-to-guides/advanced/train-a-segmentation-model-from-scratch>`.


.. figure:: _static/napari.png
    :align: center
    :alt: napari
    
    **napari**. napari provides the basic requirements of image manipulation software, namely a brush, rubber, bucket and pipette, to work on the segmentation layer. In this RICM image of spreading NK cells, two couples of cells have been mistakenly segmented as one object and must be separated. On the right panel, two plugins specific to Celldetective allow 1) the export of the modified masks directly in the position folder, and 2) to create automatically an annotation consisting of the current multichannel frame, the modified mask and a configuration file specifying the modality content of the image and its spatial calibration.


References
----------

.. [#] Florian KROMP, Eva BOZSAKY, Fikret RIFATBEGOVIC, Lukas FISCHER, Magdalena AMBROS, Maria BERNEDER, Tamara WEISS, Daria LAZIC, Wolfgang DÖRR, Allan HANBURY, Klaus BEISKE et al. « An Annotated Fluorescence Image Dataset for Training Nuclear Segmentation Methods ». In : Scientific Data 7.1 (1 11 août 2020), p. 262. ISSN : 2052-4463. DOI : 10.1038/s41597-020-00608-w . URL : https://www.nature.com/articles/s41597-020-00608-w.

.. [#] Ahlers, J. et al. napari: a multi-dimensional image viewer for Python. Zenodo https://doi.org/10.5281/zenodo.8115575 (2023).

.. [#] Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. Cell Detection with Star-Convex Polygons. in Medical Image Computing and Computer Assisted Intervention – MICCAI 2018 (eds. Frangi, A. F., Schnabel, J. A., Davatzikos, C., Alberola-López, C. & Fichtinger, G.) 265–273 (Springer International Publishing, Cham, 2018). doi:10.1007/978-3-030-00934-2_30.

.. [#] Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021).

.. [#] Pachitariu, M. & Stringer, C. Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641 (2022).
