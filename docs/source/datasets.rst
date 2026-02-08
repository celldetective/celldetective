Datasets
========

.. _datasets:

Installation
------------

ADCCFactory can be installed using:

.. code-block:: console

	$ pip install adccfactory
	
Running the GUI
---------------

Once the pip installation is complete, open a terminal and run:

.. code-block:: console

	$ python -m adccfactory

A first window of the GUI will open, asking for the path to the ADCC experiment folder to be loaded.


Configure your first experiment folder
--------------------------------------

ADCCFactory requires a specific folder tree, that mimics the organization of a `glass slide`_ into wells (main folders) and positions within the wells (subfolders). A configuration file, common to the whole experiment, is read to provide the relevant information unique to each experiment. 

.. _`glass slide`: Microscopy

.. figure:: _static/glass-slide.png
    :align: center
    :alt: exp_folder_mimics_glass_slide
    
    The experiment folder mimics the organization of the glass slide into wells and fields of view within wells.

To generate automatically such a folder tree, open ADCCFactory and go to File>New experiment... or press Ctrl+N.

.. figure:: _static/startup_new_exp.gif
    :width: 400px
    :align: center
    :alt: startup_new_experiment
    
    Press Ctrl+N or go to File>New experiment... to configure a new experiment folder
   
A dialog window will ask you where you want to create the experiment folder. Then a second window will ask for complementary information needed to fill the configuration file.     
   
.. image:: _static/configure_experiment.png
    :width: 350px
    :align: center
    :alt: configure_experiment

Once you press "Submit", these parameters create the experiment folder named "ExpLambda" in home/. At the root of the experiment folder is a configuration file that looks as follows:

.. code-block:: ini

   # Configuration for ExpLambda/ following user input
   
   [MovieSettings]
   pxtoum = 0.1
   frametomin = 1.0
   len_movie = 60
   shape_x = 2048
   shape_y = 2048
   transmission = 0
   blue_channel = 3
   red_channel = 1
   green_channel = -1
   movie_prefix = Aligned

   [SearchRadii]
   search_radius_tc = 100
   search_radius_nk = 75

   [BinningParameters]
   time_dilation = 1

   [Thresholds]
   cell_nbr_threshold = 10
   intensity_measurement_radius = 26
   intensity_measurement_radius_nk = 10
   minimum_tracklength = 0
   model_signal_length = 128
   hide_frames_for_tracking = 

   [Labels]
   concentrations = 0,1,10,100,100,10,1,0
   cell_types = WT,WT,WT,WT,HER2+,HER2+,HER2+,HER2+

   [Paths]
   modelpath = /home/limozin/Documents/GitHub/ADCCFactory/models/

   [Display]
   blue_percentiles = 1,99
   red_percentiles = 1,99.5
   fraction = 4

Detailed information about the role of each parameter is provided in "Configuration file".

Drag and drop movies
--------------------

.. note::

   Unfortunately, putting the movies in their respective folders is a manual task

The user can now drag and drop the movie associated to each field of view of each well in its respective folder (typical path: "ExpFolder/well/fov/movie/"). The movie should be in TIF format and be organized in time-X-Y-channel or channel-time-X-Y order. 

We highly recommend that you align the movie beforehand using for example, the "Linear Stack Alignment with SIFT Multichannel" tool available in Fiji, when activating the PTBIOP update site [#]_ (see discussion here_). We also put `a macro`_ at your disposal to facilitate this preliminary step.

.. _`a macro`: Align_Macro


.. _here: https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16

Usually, the alive target nucleus florescence channel works as a great reference for alignment, since the target cells are quasi-static. 

.. figure:: _static/align-stack-sift.gif
    :align: center
    :alt: sift_align
    
    Demonstration of the of the SIFT multichannel tool on FIJI

Load an experiment folder
-------------------------

Once you have filled up an experiment folder with some ADCC movies, you can open ADCCFactory, browse to the folder and press "Submit" to open the Control Panel.


References
----------

.. [#] https://www.epfl.ch/research/facilities/ptbiop/
