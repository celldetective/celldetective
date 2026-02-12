How to transfer an event detection model
========================================

This guide shows you how to export a trained event detection model for use on another machine.

Reference keys: :term:`Event`, :term:`Signal`, :term:`Model`

Context
-------

Event detection models (used for spotting peaks or specific signal patterns) are saved in the `celldetective/models` directory. You can transfer them to other computers to reuse the same analysis parameters.

Step 1: Locate the model
------------------------

1.  Navigate to your Celldetective installation folder.
2.  Go to `celldetective/models/signal_detection`.
    *   For paired signals (two channels), look in `models/pair_signal_detection`.
3.  Find the folder matching your model's name.

Step 2: Export the model
------------------------

1.  **Compress the folder**: Create a `.zip` archive of the model folder.
2.  Transfer this file to the destination computer.

Step 3: Import on the target machine
------------------------------------

1.  **Unzip** the model folder.
2.  Place it manually into the corresponding directory on the target machine:
    *   `celldetective/models/signal_detection` (for single signals)
    *   `celldetective/models/pair_signal_detection` (for paired signals)
3.  **Restart Celldetective**.

The model will now appear in the dropdown menu of the **Event Detection** module.
