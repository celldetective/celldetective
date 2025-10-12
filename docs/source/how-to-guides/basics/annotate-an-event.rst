How to annotate an event
------------------------------

This guide shows you how to annotate a single-cell event, either for manual characterisation or to create a dataset to train a deep learning event detection model.

Reference keys: *event*, *event class*, *event time*


**Prerequisite:** You have accurately segmented, **tracked** and measured a cell population of interest. This guide only applies to dynamic data.

**Step-by-step:**

#. Go to the DETECT EVENTS section for the cell population of interest. Click on the :icon:`cog-outline,black` button to set up the viewer.

#. Set the modality to "grayscale".

#. Set the channel of interest.

#. Adapt the fraction value to your data (e.g. 0.25). At a fraction of 1, you will see the original image (but the viewer will take longer to open and may be slow).

#. Set a time interval between frames (e.g. 100 ms).

#. Save and exit.

#. Press the :icon:`eye-check,black` button of the DETECT EVENT section to open the event viewer.

#. Adjust the image contrast.

#. On the top-left side, create a new event by pressing the :icon:`plus,black` button.

#. Give the event a name and set the initialization class to "no event".

#. Click on a single cell to modify its event class.

#. Press :icon:`redo-variant,black` correct on the left side to change the class. Choose a class among "event", "no event", "else" and "remove". If you pick "event", you must determine the exact time (in frame) at which the event occurs. Use the time-series of the single cell to observe inflexion points and determine precisely this time.

#. Click on :icon:`redo-variant,black`, a vertical line will show you the event time on the time series. 
