First project
=============

.. _first_experiment:

Input
-----

Celldetective is designed to process multichannel time-lapse microscopy data saved as ``tif`` stacks. Lower-dimensional data is also compatible. Notably, Z-stacks are not supported at the moment, although such images can be passed to Celldetective with a trick (see the note below). The files may have the following formats:

- XY   (2D) frame: single-timepoint & single-channel image.
- CXY  (3D) stack: single-timepoint & multichannel image.
- PXY  (3D) stack: multi-position, single-channel & single-timepoint images.
- TXY  (3D) stack: time-lapse images.
- PCXY (4D) stack: multi-position, multichannel-channel & single-timepoint images.
- TCXY (4D) stack: multi-channel time-lapse images.

.. note::
    A Z-axis can be passed to Celldetective as a substitute to the time-axis, in which case each slice can be segmented and measured independently. There is no compatibility with both Z-stacks and time-lapse data. 


Pre-Processing Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Microscopy data acquired through :math:`\mu` Manager [#]_ often interlaces the channel dimension with the time dimension to preserve their separation. Before using these stacks in Celldetective, they must be disentangled to ensure proper functionality of the Celldetective viewers.

Before loading your data into Celldetective, we recommend opening the raw stacks in **ImageJ** (or a similar tool) to verify that the stack dimensions (time, channels, spatial axes) are correctly set.

For large stacks exceeding 5 GB, we recommend using the **Bio-Formats Exporter** plugin in ImageJ to save the stacks. This format optimizes the data for efficient processing and visualization in Celldetective.


Creating a new experiment
-------------------------

.. figure:: _static/maingui.png
    :align: center
    :alt: exp_folder_mimics_glass_slide

    Startup window (top). Panels to create (left) or process (right) an experiment.

To create a new experiment, follow this :ref:`how-to guide <new-experiment-guide>`.

Configuration file example
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: ini

   # config.ini

    [Populations]
    populations = nk,rbc
   
    [MovieSettings]
    pxtoum = 0.3112
    frametomin = 2.75
    len_movie = 44
    shape_x = 2048
    shape_y = 2048
    movie_prefix = Aligned

    [Channels]
    brightfield_channel = 0
    live_nuclei_channel = 3
    dead_nuclei_channel = 1
    effector_fluo_channel = 2
    adhesion_channel = nan
    fluo_channel_1 = nan
    fluo_channel_2 = nan

    [Labels]
    cell_types = MCF7-HER2+primary NK,MCF7-HER2+primary NK
    antibodies = None,Ab
    concentrations = 0,100
    pharmaceutical_agents = None,None

    [Metadata]
    concentration_units = pM
    cell_donor = 01022022


Configuration file tags
~~~~~~~~~~~~~~~~~~~~~~~

The configuration file defines the structure of your experiment (populations, image parameters, channel names, etc.).

For a detailed explanation of each tag (``[Populations]``, ``[MovieSettings]``, etc.), see the :ref:`Experiment Configuration Reference <ref_experiment_config>`.

Quick access to the experiment folder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an experiment is opened in Celldetective, you can quickly access its folder by clicking the **folder icon** next to the experiment name in the top menu.


Drag and drop the movies
------------------------

To prepare your data for processing, you need to place each movie into its corresponding position folder, specifically in the ``movie/`` subfolder (e.g., ``W1/100/movie/``).

This step is **not automated**, as variations in acquisition protocols and naming conventions make it difficult to provide a universal solution. If manual placement is too time-consuming, we recommend creating a custom script tailored to your specific data organization.

Once the movies are placed in their respective folders, you can proceed to image processing. Detailed instructions on processing are provided in the next sections.


Bibliography
------------

.. [#] Arthur D Edelstein, Mark A Tsuchida, Nenad Amodaj, Henry Pinkard, Ronald D Vale, and Nico Stuurman (2014), Advanced methods of microscope control using μManager software. Journal of Biological Methods 2014 1(2):e11.
