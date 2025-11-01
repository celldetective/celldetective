Single-Cell Event
=================

**Reference keys:** *single-cell event*, *event time*, *event class*

A **single-cell event** is defined as a specific occurrence that happens to a cell at a particular time point. This can include:

* A discrete state change (e.g., alive to dead)
* The onset of a longer process (e.g., a cell beginning to spread)
* A contextual change (e.g., the arrival of a neighboring cell, a change in local cell density)

An event often leaves a detectable **signature** in one or more single-cell time-series (e.g., a sudden change in fluorescence, shape, texture), though this is not mandatory. An event may be imperceptible if the chosen measurements do not capture the relevant biological change.

Events are, by definition, **dynamical**. This concept requires time-lapse data with single-cell tracking and cannot be applied to static datasets.

Event Classification and Time
-----------------------------

Our framework for classification borrows directly from survival theory, which is often used in medical studies. In this approach, each single cell is treated like an individual patient in a clinical trial. We classify each cell into one of three states relative to a specific event:

* **Class "event":** Cells that experience the event of interest *during* the observation window (medical analogy: a patient who develops the disease *during* the study).
* **Class "no event":** Cells that do *not* experience the event by the time the observation ends (medical analogy: a patient who remains healthy for the entire study duration).
* **Class "else":** Cells that cannot be classified as "event" or "no event." This primarily includes cells that experienced the event *before* the observation began (medical analogy: a patient who already had the disease when the study started).

Cells belonging to the "event" class are associated with an **event time** (:math:`t_{\text{event}}`). This is interpreted as the time at which the event begins. By convention, the onset of the event is used, which is a requirement for performing survival studies.

Applications and Limitations
----------------------------

In Celldetective, events are fundamental for several analyses. For example, you can build a survival function (using up to two events) or synchronize multiple single-cell time-series relative to the time of an event.

A key **limitation** is that the event concept in Celldetective is not designed to handle cyclic events. Such processes must be decomposed into a series of discrete, successive events (e.g., "start of mitosis," "end of mitosis"). Frequency characterization is not supported at this time.