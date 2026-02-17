How to perform conditional cell classification
==============================================

This guide shows you how classify cells from their features using conditional expressions.

**Prerequisite:** You have accurately segmented and measured a :term:`cell population`.

**Reference keys:** :term:`characteristic group`, :term:`phenotype`

1. Open an :term:`experiment project`. In the header part, select the :term:`wells <well>` and :term:`positions <position>` you want to classify.

2. Expand the block associated with your :term:`cell population` of interest. Click on the triple dots icon in the *MEASURE* section to launch the classifier utility.

3. Name the classification to create. For **dynamic data** this becomes the name of the event. For **static** data, it is the name of the :term:`group`.

4. Select two features that can clusterize the cells (e.g. area and adhesion channel intensity in a spreading classification).

6. **Explore Your Data:**
    - Use the **Frame Slider** at the bottom to visualize the population feature distribution frame by frame.
    - Click the **Project Times** button :icon:`math-integral,black` to superimpose all timepoints on the same plot (useful for checking overall population clusters).
    - Use the **Log Scale** buttons :icon:`math-log,black` next to each feature selector to switch between linear and log scales.
    - Adjust the **Transparency Slider** (bottom right) if points are too dense.

7. **Define the Class:**
    Type a condition in the classify field. The syntax supports numeric comparisons, logic operators, and string matching.

    *   **Numeric conditions:** ``area > 500``, ``intensity < 200``
    *   **Combinations:** ``area > 500 and intensity < 200``, ``area > 500 or circularity > 0.8``
    *   **String/Category matching:** ``well == "W1"``, ``label != "A"`` (use quotes for strings)
    *   **Complex columns:** Use backticks for columns with special characters: ```d/dt.area` > 0``

8. **Preview:**
    Press the **Preview** button.
    
    - **Red points:** Cells matching your condition (Positive).
    - **Blue points:** Cells not matching (Negative).
    
    *Tip: Change the x/y features to verify that your classification makes sense in other dimensions.*

9. **Apply (Static vs. Time-Correlated):**

    - **Static Group (Default):**
      If **Time correlated** is unchecked, clicking **Apply** creates a standard :term:`group` or status column. This is a frame-by-frame classification.

    - **Time Correlated Event (For Tracked Data):**
      If your data is tracked (contains ``TRACK_ID``), you can check **Time correlated**. This fits a sigmoid to the binary signal of each track to detect *when* an event happens (e.g., cell death, specific state entry).
      
      Select the event type:
      
      *   **Unique state:** The cell enters a state and stays there (or doesn't).
      *   **Irreversible event:** A definitive transition (like death).
      *   **Transient event:** A state that can be entered and exited (e.g., calcium pulse).
      
      *Note: The **R2 tolerance** slider defines how well the sigmoid must fit the data to accept the event time.*

10. Press **Apply** to finalize. A new column (e.g., ``status_my_class``) and potentially event times (``t_my_class``) will be added to your data.