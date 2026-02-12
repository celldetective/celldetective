How to plot survival between two events
---------------------------------------

This guide shows you how to generate Kaplan-Meier survival curves between two annotated events.

Reference keys: :term:`survival`, :term:`event time`

**Prerequisite:** You have segmented, tracked, measured, and annotated events for a cell population. At least two events must be defined (a start reference and an end event).


Configure the survival plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Go to the **Analyze** tab and click **Plot survival**.

#. Select the **population** of interest.

#. Set the **Start Event** (Time of Reference) — the event that marks the beginning of the observation window (e.g., ``t_firstdetection``).

#. Set the **End Event** (Time of Interest) — the event whose occurrence you want to measure (e.g., ``t_death``).

#. (Optional) Enter a **Query** to filter the population (e.g., ``TRACK_ID > 10`` or ``treatment == "drug_A"``).

#. Click **Submit**.

For a full description of all fields, see the :ref:`Survival Analysis Settings Reference <ref_survival_settings>`.


Interpret the output
~~~~~~~~~~~~~~~~~~~~~

A window appears with the Kaplan-Meier survival curves.

*   **Single position**: A single survival curve for the selected position.
*   **Multiple positions**: Individual curves per position (with 95% confidence intervals) and a pooled curve for the well.
*   **Multiple wells**: Pooled curves per well for comparing conditions. Individual positions are shown without confidence intervals.

You can add or remove positions/wells from the plot interactively.