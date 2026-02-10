How to measure cell-cell interactions
--------------------------------------

This guide shows you how to quantify spatial relationships between two cell populations using the **Interactions** section of the control panel.

Reference keys: *neighborhood*, *cell-cell interactions*

**Prerequisite:** Both populations must be segmented. Tracking is recommended but not required.


Add a neighborhood protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Open the **Interactions** section in the control panel.

#. Check the **NEIGHBORHOODS** box to enable neighborhood computation.

#. Choose a neighborhood type and click its :icon:`plus,black` button to open the configuration dialog:

   *   **ISOTROPIC DISTANCE THRESHOLD** — detects neighbors whose center of mass falls within a threshold distance of the reference cell.
   *   **MASK CONTACT** — detects neighbors whose segmentation masks touch or nearly touch the reference cell, within a threshold edge distance.


Configure the reference population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the configuration dialog that opens:

#. Under **REFERENCE**, select the reference **population** (e.g., ``Targets``).

#. (Optional) Select a **status** filter to restrict analysis to a subset of reference cells (e.g., only ``Alive`` cells). Use the :icon:`invert-colors,black` button to invert the status selection.

#. (Optional) Select an **event time** column. If set, average neighborhood metrics will be computed before and after this event.


Configure the neighbor population
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Under **NEIGHBOR**, select the neighbor **population** (e.g., ``Effectors``).

#. (Optional) Select a **status** filter to restrict which neighbor cells are considered. Use the :icon:`invert-colors,black` button to invert the selection.

#. (Optional) Check **cumulated presence** to compute the total contact duration of each neighbor around a reference cell over time.


Set the neighborhood distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Under **NEIGHBORHOOD**, click the :icon:`plus,black` button to add a distance value (in pixels). You can add multiple distances to analyze different ranges simultaneously.

#. (Optional) Click the :icon:`image-check,black` viewer button to preview the neighborhood radius or edge distance on your stack.

#. To remove a distance, select it in the list and click the :icon:`delete,black` button.

#. Click **Save** to add this neighborhood protocol to the computation list.


Run and manage protocols
~~~~~~~~~~~~~~~~~~~~~~~~~

You can add multiple neighborhood protocols (e.g., distances at 30 px and 50 px, plus a mask contact). They appear in the **Neighborhoods to compute** list.

*   To remove a protocol, select it and click the :icon:`delete,black` button next to the list.
*   To run all protocols, click **Submit**.

For a complete list of parameters, see the :ref:`Neighborhood Measurement Settings Reference <ref_neighborhood_settings>`.


Output columns
~~~~~~~~~~~~~~

For a distance threshold of ``d`` px between populations ``A`` and ``B``, the following columns are written to the reference population's table. When the reference and neighbor populations are the same, ``(A-B)`` is replaced by ``self`` in all column names below.

**Per-frame counts** (in the CSV table):

*   ``inclusive_count_neighborhood_(A-B)_circle_{d}_px`` — total number of neighbors.
*   ``exclusive_count_neighborhood_(A-B)_circle_{d}_px`` — number of neighbors for which this reference cell is the closest.
*   ``intermediate_count_neighborhood_(A-B)_circle_{d}_px`` — attention-weighted neighbor count (each neighbor contributes ``1/N`` where ``N`` is the number of reference cells it is close to).

Each metric is also decomposed by neighbor status when available: ``..._s0_...`` (status 0) and ``..._s1_...`` (status 1).

**Per-track averages** (if tracking and an event time column are set):

*   ``mean_count_inclusive_neighborhood_(A-B)_circle_{d}_px_before_event``
*   ``mean_count_inclusive_neighborhood_(A-B)_circle_{d}_px_after_event``
*   (Same pattern for ``exclusive`` and ``intermediate``.)

**Pickle-only column** (not in CSV):

*   ``neighborhood_(A-B)_circle_{d}_px`` — per-frame list of neighbor dictionaries, each containing ``id``, ``distance``, ``status``, ``weight``, and ``closest``.
