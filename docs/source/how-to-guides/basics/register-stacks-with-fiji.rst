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

1. Copy the code below (or download the `alignment macro <../../_static/macros/align_stack.ijm>`_) into Fiji's macro editor (**Plugins > New > Macro**).
2. Update the variables in the code (lines 6-9) to match your data:
    * ``target_channel``: the channel used for registration.
    * ``prefix``: the prefix of the stacks to be aligned (e.g., "After" or "Raw").
3. Run the macro.
4. When prompted, select the **root folder** of your Celldetective experiment.

.. code-block:: javascript
    :caption: align_stack.ijm

    run("Collect Garbage");
    experiment = getDirectory("Experiment folder containing movies to align...");
    wells = getFileList(experiment);

    octave_steps = "7";
    target_channel = "4";
    target_channel_int = 4;
    prefix = "After"

    for (i=0;i<wells.length;i++){
        
        well = wells[i];
        
        if(endsWith(well, File.separator)){
            
            positions = getFileList(experiment+well);
            
            for (j=0;j<positions.length;j++) {
                
                pos = positions[j];
                movie = getFileList(experiment+well+pos+"movie"+File.separator);
                
                for (k=0;k<movie.length;k++) {
                    if (startsWith(movie[k], prefix)) {
                        
                        // Open stack
                        open(experiment+well+pos+"movie"+File.separator+movie[k]);
                        Stack.setDisplayMode("grayscale");

                        // Here write the preprocessing steps
                        Stack.setChannel(target_channel_int);
                        run("Enhance Contrast", "saturated=0.35");
                        run("Linear Stack Alignment with SIFT MultiChannel", "registration_channel="+target_channel+" initial_gaussian_blur=1.60 steps_per_scale_octave="+octave_steps+" minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");

                        // Save output
                        saveAs("Tiff", experiment+well+pos+"movie"+File.separator+"Aligned_"+movie[k]);
                        close();
                        close();
                        run("Collect Garbage");
                    }
                }
            }
            
        }
    }

    print("Done.");

.. note::
    Ensure your registered files are saved in the correct `movie/` subfolder of each position if you are following the recommended folder structure.


References
----------

.. [#] Schindelin, J., Arganda-Carreras, I., Frise, E. et al. Fiji: an open-source platform for biological-image analysis. Nat Methods 9, 676–682 (2012).
