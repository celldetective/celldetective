How to transfer a segmentation model
====================================

This guide shows you how to export a custom segmentation model from one computer to another.

Reference keys: :term:`Cellpose`, :term:`StarDist`, :term:`Experiment`

Context
-------

When you train a model in Celldetective, it is saved locally in the `celldetective/models` directory. To use this model on another machine (e.g., moving from a training workstation to an analysis laptop), you need to manually transfer the model files.

Step 1: Locate the model
------------------------

1.  Navigate to your Celldetective installation folder.
2.  Go to `celldetective/models/segmentation`.
    *   If you trained a specialized model (e.g., for effectors), check `models/segmentation_effectors`.
3.  Find the folder with your model's name.

Step 2: Export the model
------------------------

1.  **Zip the folder**: Right-click the model folder and compress it into a `.zip` archive.
    *   Ensure the zip file contains the model configuration (JSON files) and weights.

Step 3: Import on the target machine
------------------------------------

1.  Transfer the `.zip` file to the target computer (USB, network share, etc.).
2.  Open Celldetective on the target machine.
3.  Go to the **Segmentation** module.
4.  Click **UPLOAD**.
5.  Select :term:`StarDist` or :term:`Cellpose` depending on your model type.
6.  Click **Choose File** and select your `.zip` file (or the unzipped folder).
7.  Click **Upload**.

The model is now available in the "My Models" list on the new machine.