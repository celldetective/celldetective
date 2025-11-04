How to perform conditional cell classification
==============================================

This guide shows you how classify cells from their features using conditional expressions.

**Reference keys:** :term:`characteristic group`, :term:`phenotype`

**Prerequisite:** You have accurately segmented and measured a :term:`cell population`.

1. Open an :term:`experiment project`. In the header part, select the :term:`wells <well>` and :term:`positions <position>` you want to classify.

2. Expand the block associated with your :term:`cell population` of interest. Click on the triple dots icon in the *MEASURE* section to launch the classifier utility.

3. Name the classification to create. For **dynamic data** this becomes the name of the event. For **static** data, it is the name of the :term:`group`.

4. Select two features that can clusterize the cells (e.g. area and adhesion channel intensity in a spreading classification).

5. Use the slider to see the measurements for different frames. You can press the :icon:`math-integral,black` button to show all measurements on the same plane.

6. Type a condition in the classify field. Examples include: ``area > 500``, ``adhesion_channel_mean_intensity < 1 and area > 500``, etc. The condition expressions follow this convention: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html . You may need to use backticks on complex feature names (e.g. ```d/dt.area` > 1``. Make sure to type properly the feature name (do a copy and paste from the fields above preferably).

7. Press the **Preview** button to see the classification result in the feature space (red: condition is True, blue: condition is False). You can switch features to see for as many projections of the feature space as needed the validity of the classification.

8. Press **Apply** to write the classification in the tables, creating a new column that labels the assigned :term:`phenotype`.