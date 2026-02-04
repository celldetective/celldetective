Overview
========

.. _overview:


Description
-----------

.. figure:: _static/workflow.png
    :width: 100%
    :align: center
    :alt: celldetective_workflow
   

Despite notable efforts in the development of user-friendly software that integrate state-of-the-art solutions to perform single cell analysis, very few are designed for time-lapse data and even less for multimodal problems where cells populations are mixed and can only be separated through the use of multimodal information. Few software solutions provide, to our knowledge, the extraction of response functions from single cell events, such as the dynamic survival of a population directly in the GUI, as coding skills are usually required to do so. We want to study complex data which is often multimodal time lapse microscopy images of interacting cell populations, without loss of generality. With a high need for an easy-to-use, no-coding-skill-required software adapted for images and intended for biologists, we introduce **Celldetective**, an open-source python-based software with the following highlight features:

* **Comprehensive single-cell image analysis** : Celldetective ships segmentation, tracking, and measurement modules, as well as event detection from single-cell signals, for up to two populations of interest.
* **Integration of state-of-the-art solutions** : Celldetective harnesses state-of-the-art segmentation techniques (StarDist [#]_, Cellpose [#]_ , [#]_) and tracking algorithm (bTrack [#]_), as well as the napari viewer [#]_ where applicable. These algorithms are interfaced to be well integrated and accessible for the target audience, in the context of complex biological applications.
* **A framework for event description and annotations** : we propose a broad and intuitive framework to annotate and automate the detection of events from single-cell signals through Deep Learning signal classification and regression. The event formulation is directly exploited to define the population's survival responses.
* **A neighborhood scheme to study cell-cell interactions** : we introduce a neighborhood scheme to relate the spatio-temporal distribution and measurements of two cell populations, allowing the study of how cell-cell interactions affect single-cell and population responses.
* **Deep Learning customization in GUI** : Celldetective simplifies the specialization of Deep Learning models or the creation of new ones adapted to user data, by facilitating the creation of training sets and the training of such models, without having to write a single line of code.
* **In-software analysis** : Celldetective ships visualization tools to collapse single-cell signals with respect to an event, build survival curves, compare measurement distributions across biological conditions.
* **A library of segmentation and signal models**: we created specific models to investigate a co-culture of MCF-7 cells and primary NK cells, that are available directly in the software with a large collection of generalist models developed by the StarDist and Cellpose teams, which are a perfect starting point to segment single cells in a new biological system. 
* **Accessible and open source** : Celldetective does not require any coding skills. The software, its models and datasets are made fully open source to encourage transparency and reproducibility.


 System requirements
-------------------

Celldetective is designed to run on standard workstations as well as high-performance clusters. It supports GPU acceleration (NVIDIA) for Deep Learning tasks but can run fully on CPU.

For detailed hardware and software specifications, see the :ref:`System Requirements Reference <reference/system-requirements>`. 


Bibliography
------------

.. [#] Schmidt, U., Weigert, M., Broaddus, C. & Myers, G. Cell Detection with Star-Convex Polygons. in Medical Image Computing and Computer Assisted Intervention – MICCAI 2018 (eds. Frangi, A. F., Schnabel, J. A., Davatzikos, C., Alberola-López, C. & Fichtinger, G.) 265–273 (Springer International Publishing, Cham, 2018). doi:10.1007/978-3-030-00934-2_30.

.. [#] Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106 (2021).

.. [#] Pachitariu, M. & Stringer, C. Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641 (2022).

.. [#] Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. Frontiers in Computer Science 3, (2021).

.. [#] Ahlers, J. et al. napari: a multi-dimensional image viewer for Python. Zenodo https://doi.org/10.5281/zenodo.8115575 (2023).
