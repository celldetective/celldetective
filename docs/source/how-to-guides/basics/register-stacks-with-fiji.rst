How to register stacks with Fiji
================================

This guide shows you how to align your microscopy movies (registration) before importing them into Celldetective.

**Prerequisite:** You have installed `Fiji (ImageJ) <https://fiji.sc/>`_.

We highly recommend adhering to the :doc:`Data Organization <../../concepts/data-organization>` guidelines to structure your folders.

Overview
--------

Registration corrects for stage drift or shaking during acquisition. A common tool for this is the **Linear Stack Alignment with SIFT Multichannel** plugin available in Fiji [#]_.


Step 1: Install the SIFT plugin
-------------------------------

1.  Open Fiji.
2.  Go to **Help > Update...**.
3.  Click **Manage update sites**.
4.  Check the **PTBIOP** update site.
5.  Click **Close** and then **Apply changes**.
6.  Restart Fiji.

For more details on this plugin, see the `Image.sc discussion <https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16>`_.


Step 2: Register a stack (Manual)
---------------------------------

1.  Open your stack in Fiji.
2.  Go to **Plugins > BIOP > Linear Stack Alignment with SIFT Multichannel**.
3.  Select the transformation mode (usually "Translation" for simple drift).
4.  Run the alignment.
5.  Save the registered stack as a new TIFF file.


Step 3: Batch registration (Macro)
----------------------------------

To facilitate batch processing, we provide an ImageJ macro that can be used to process multiple positions automatically.

1.  Download the `alignment macro <../../_static/macros/align_stack.ijm>`_ (or copy the code below).
2.  Open the macro in Fiji.
3.  Update the input/output directories in the script.
4.  Run the macro.

.. code-block:: java
    :caption: align_stack.ijm

    // TODO: Add macro content here or link to file

.. note::
    Ensure your registered files are saved in the correct `movie/` subfolder of each position if you are following the recommended folder structure.


References
----------

.. [#] Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012).
