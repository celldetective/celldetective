How to annotate an event
------------------------------

This guide shows you how to annotate a single-cell event, either for manual characterisation or to create a dataset to train a deep learning event detection model.

Reference keys: *event*, *event class*, *event time*


**Prerequisite:** You have accurately segmented, **tracked** and measured a cell population of interest. This guide only applies to dynamic data.

Prepare the viewer
~~~~~~~~~~~~~~~~~~

#. Go to the DETECT EVENTS section for the cell population of interest. Click on the :icon:`cog-outline,black` button to set up the viewer.

#. Set the modality to "grayscale".

#. Set the channel of interest.

#. Adapt the fraction value to your data (e.g. 0.25). At a fraction of 1, you will see the original image (but the viewer will take longer to open and may be slow).

#. Set a time interval between frames (e.g. 100 ms).

#. Save and exit.

Annotate in the viewer
~~~~~~~~~~~~~~~~~~~~~~

#. Press the :icon:`eye-check,black` button of the DETECT EVENT section to open the event viewer.

#. Adjust the image contrast.

#. On the top-left side, create a new event by pressing the :icon:`plus,black` button.

#. Give the event a name and set the initialization class to "no event".

#. Click on a single cell to modify its event class.

#. Press :icon:`redo-variant,black` **correct** on the left side to change the class. Choose a class among "event", "no event", "else" and "remove". If you pick "event", determine the exact time (in frame unit) at which the event occurs. Use the time-series of the single cell to observe inflexion points and determine precisely this time.

#. Click on :icon:`redo-variant,black` **submit**, a vertical line will show you the event time on the time series.

#. Proceed to annotate all single-cells with the proper class.

#. Press the :icon:`export,black` button at the bottom of the left side to export a time-series dataset (in ``.npy`` format). Put the file in a folder containing all of your annotations specific to this event.

#. Press the Save button to update the position table with the new event class and the values for each cell.