How to align two channels
=========================

This guide shows you how to create an experiment project and import your data. Learn more about the structure of an experiment project :doc:`here <data-organization>`.

Reference keys: *experiment project*, *well*, *position*, *population*, *cell type*, *antibody*, *concentration*, *pharmaceutical agents*

New project
~~~~~~~~~~~

#. Launch the software and go to **File > New Experiment...**

#. Set a folder where the project will be stored.

#. Fill the information requested. Refer to the reference guide for more precision.
 - Name for the experiment. Avoid spaces in the name.

 - Number of wells.

 - Number of positions per well. If variable, put the highest number in a well.

 - Conversion factor from pixel to :math:`\mu m` on the images.

 - Time interval between two frames in minutes if time-series data. Else, leave at default value. If time-series but variable, put an average estimate.

 - Number of frames. If variable, put a lower estimate. Value read directly from the stacks in most cases.

 - Prefix to find the image stack file. Leave blank if filenames have variable prefixes.

 - Image width/height in pixels.

#. Select your channels and specify their index in the stack with the slider on the right side (0 is first, 1 is the second channel, etc). Use existing channels if appropriate. Else, create your own channel. Avoid spaces in the name.

#. Select your cell population(s). If you have an immune cell population, select `effectors`. If you have cancer cells, select `targets`. Else create appropriate populations.

#. Submit.

#. In the pop-up window, fill the information for each well (cell type, antibody, concentration, pharmaceutical agents). Fields can be left blank.

#. After submitting:
 - The dialog closes.

 - The path to the newly created experiment is automatically loaded in the startup window. Click **Open** to access it.

 - On the disk, the experiment folder is created with a configuration file that looks like the example below.

Drag and drop the image stacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open the project in Celldetective.  Click on the :icon:`folder,black` icon next to the experiment name in the top menu to open the experiment folder.

#. Drag and drop each TIF stack file in its its corresponding position folder, specifically in the ``movie/`` subfolder (e.g., ``W1/100/movie/``). Step **not automated**.

