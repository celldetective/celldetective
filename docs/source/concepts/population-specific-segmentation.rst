Population-specific segmentation
--------------------------------

**Reference keys:** :term:`cell population`, :term:`instance segmentation`

Independent segmentation
========================

The purpose of Celldetective is to achieve single-cell resolution through instance segmentation. In co-cultures, different cell types can be in spatial co-presence on a 2D projection, making single-cell quantifications challenging.

One way to fix this problem, particularly in effector/target systems, is to segment independently each **cell population**, in order to have masks as complete as possible for each population separately. Measurements can thus be performed as cleanly as possible, within the constraints of 2D images.

Another way to put this is that if you have several cell populations on your images, you should repeat the segmentation task for each population of interest, with a segmentation method as appropriate as possible for the population you want. A consequence of this is that what we call a "cell population" is a group of cells that were segmented using the same method (a population does not have to strictly relate to cell type; it can be based on different cell states in a mono culture, for example).

Strategies
==========

.. The process of :term:`instance segmentation` takes an image (multichannel or not) as its input and yields a label image (2D), where each segmented object is attributed a single label.

.. In Celldetective, you may want to specifically segment up to two :term:`cell populations` of interest on the images (typically target and effector cells, but it could be anything). Segmentation can be carried out using traditional segmentation pipelines (based on filters and thresholds) or using deep learning models trained for such a task. Celldetective proposes both options and allows a cross-talk between the two. As illustrated below, the output of a traditional segmentation can be corrected and used as an input to a DL model directly in Celldetective. That is only one of many paths to perform segmentation in Celldetective.

.. figure:: ../_static/segmentation-options.png
    :align: center
    :alt: seg_options

    **Overview of segmentation options in Celldetective.** Celldetective provides several entry points (black arrows) to perform segmentation, with the intent of segmenting specifically a cell population (left: effectors, right: targets). The masks output from each segmentation technique can be visualized and manually corrected in napari. Exporting these corrections into a paired image and masks dataset can be used either to fit a generalist model (transfer learning) or train one from scratch. Once the segmentation is satisfactory enough, the user can decide to proceed with the tracking and measurement modules.

