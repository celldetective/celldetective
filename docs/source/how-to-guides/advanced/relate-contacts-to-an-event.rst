How to relate contacts to an event with a contact hazard model
==============================================================

This guide shows you how to test whether the contacts a cell receives (from
effector cells, say) raise the hazard of one of its events (its death, say),
and whether the answer depends on how a contact is defined.

Reference keys: :term:`neighborhood`, :term:`mask contact`, :term:`contact episode`, :term:`hazard ratio`, :term:`survival`

**Prerequisite:**

*   The reference population (e.g. ``targets``) is tracked and has the event
    of interest: a ``t_death`` column with its ``class_death`` (``0`` the event
    happened, ``1`` it did not, ``2`` otherwise, e.g. a cell already dead when first seen),
    from :doc:`event detection <../basics/detect-an-event-with-conditions>` or
    :doc:`annotation <../basics/annotate-an-event>`.
*   At least one neighborhood between the reference and the neighbor
    population is computed (:doc:`../basics/measure-cell-interactions`), so
    that the table of the reference population has its
    ``inclusive_count_neighborhood_...`` columns. Several mask-contact
    distances let you check the robustness of the result.
*   For the analysis of contact episodes, the pairs of that neighborhood are
    measured (**MEASURE PAIRS**), which writes ``trajectories_pairs.csv``.


Why a hazard model with time-varying contacts
----------------------------------------------

The tempting analysis groups the cells by their total contact time and compares
the survival curves of the groups. It is biased in two ways:

*   **Immortal time.** A cell has to stay alive to accumulate contact, and a
    contact usually ends when the cell dies. Long-contacted cells then look
    protected, or the opposite, whatever the contacts do.
*   **Reverse causation.** Effectors may stay on, or be drawn to, cells that
    are already dying. Contact then goes with death without causing it.

The contact hazard model follows each cell frame by frame and relates the
hazard of the event at a time to the contacts the cell has received *before*
that time only. The result is a :term:`hazard ratio` per covariate: by how
much the instantaneous risk of the event is multiplied when the covariate
rises by one unit (one more neighbor, one more minute of contact). A ratio
whose 95% interval lies above 1 means a higher risk.

.. note::
    Images alone show association, not causation. To claim that the contacts
    *cause* the deaths, add evidence the model cannot give: a dose-response
    (the ratio grows with the contact time), a delay between contact and death
    that is consistent across cells, and above all perturbations in sister
    wells (blocking antibodies, inhibitors of the lytic machinery, a range of
    effector:target ratios, targets alone for the spontaneous death rate).
    Selecting these wells in the control panel and giving each its own baseline
    hazard (**baseline per** ``well``) compares them in one model.


Fit the model
-------------

.. figure:: ../../_static/figures/contact-hazard.svg
    :width: 100%
    :target: ../../_static/figures/contact-hazard.svg
    :align: center
    :alt: The Analyze tab, the contact hazard options and the forest plot of the hazard ratios

    **Contact hazard of death.** Synthetic targets killed at random during
    their contacts with effectors: being in contact multiplies the hazard of
    death, at both contact distances, while the contact time accumulated
    before adds nothing.

#. Select the wells and positions to analyze in the control panel.

#. Go to the **Analyze** tab and click **contact hazard**, under
   **Interactions** (1).

#. Select the **reference population** (2). Only the populations with a
   computed neighborhood are offered.

#. Check one or more **neighborhoods** (3). The model is fitted once per
   neighborhood: with several contact distances, the forest plot shows
   whether the conclusion depends on the definition of a contact.

#. Select the **event** (``t_death`` by default when it exists) and, in
   **follow from**, when each cell starts to be at risk: its **first frame**,
   or an event time such as ``t_firstdetection`` (4).

#. Check the **covariates** (5); ``in_contact`` and ``cumulative_contact`` are
   checked by default:

   *   ``in_contact``: 1 when the cell has at least one neighbor, else 0;
   *   ``n_neighbors``: the number of neighbors;
   *   ``weighted_neighbors``: the neighbors, each counted 1 / the number of
       reference cells it touches;
   *   ``cumulative_contact``: the time spent with at least one neighbor so far;
   *   ``ongoing_contact``: the time since the current contact began, 0 out of
       contact;
   *   ``contact_<channel>_mean``: the contact-site intensities, when they were
       measured with the neighborhood, 0 out of contact.

#. (Optional) Set a **lag [frames]** (6): the hazard of the interval
   ``(t, t + 1]`` is then related to the exposure at frame ``t - lag``. Use it
   when the event is seen some time after its cause, e.g. the uptake of a
   death dye a few frames after the lethal hit. The first ``lag`` frames of
   each cell, without past exposure, are left out.

#. Choose the **duration unit** (``min``, ``h`` or ``frame``) of the contact
   durations: a hazard ratio of ``cumulative_contact`` is per unit, e.g. per
   minute of contact.

#. (Optional) In **baseline per**, give each ``well`` or ``position`` its own
   baseline hazard, so that cells are only compared with cells of the same
   well or position. Do it when wells differ in conditions that change the
   death rate by themselves.

#. Keep the **penalty** at ``0.01`` unless you have a reason (7). When every
   event follows a contact (complete separation), the unpenalized ratio is
   infinite; a small penalty keeps it finite. Larger penalties pull all the
   ratios towards 1.

#. (Optional) Restrict the cells with a **query**, check the **time
   calibration**, and check **show the long table** to see the data the model
   is fitted on.

#. Click **Submit** (8).


Read the results
----------------

A table opens, titled after the event and the population, with one row per
neighborhood and covariate:

*   ``neighborhood`` and its ``distance`` [px];
*   ``hazard_ratio`` with its 95% interval ``ci_lower`` – ``ci_upper``, the
    coefficient ``coef`` (its logarithm) and the ``p`` value;
*   ``n_cells`` and ``n_events``, the cells followed and the events seen;
*   ``error``, for a neighborhood that could not be fitted (no event, or a
    covariate that never varies).

The **Hazard ratios** window plots them, one panel per covariate and one line
per neighborhood, on a log scale with a dashed line at 1 (9). Like every table
of the software, the results can be saved, sorted and plotted further.

The ratios of covariates fitted together are adjusted for each other. In the
figure, being in contact raises the hazard, and once this is accounted for, the
contact time accumulated before adds nothing: the synthetic targets were
killed with the same chance at every frame of contact.

.. tip::
    Few events give wide intervals. Check ``n_events`` before reading a ratio,
    and prefer one or two covariates to many.


The long table
--------------

With **show the long table**, a table opens with one row per cell and frame
interval ``(start, stop]`` (in the duration unit) for the first neighborhood
checked: the ``cell`` (well, position and track), its exposure at the start of
the interval (or ``lag`` frames before), and ``event``, 1 on the interval
where the event happened. It is the input of the model, in the "counting
process" format of survival analysis with time-varying covariates; save it to
fit other models elsewhere.

A frame missing from a track counts as without contact. A cell whose event
happened before it was followed (``class_death`` = 2, or an event before the
**follow from** time) is left out.


Relate contact episodes to the event
------------------------------------

.. figure:: ../../_static/figures/contact-episodes.svg
    :width: 100%
    :target: ../../_static/figures/contact-episodes.svg
    :align: center
    :alt: The contact episode options and the fraction of episodes followed by a death per class of duration

    **Death after contact episodes.** The fraction of contact episodes followed
    by the death of the target within 10 frames of their start, per class of
    episode duration.

The model answers whether contact raises the hazard. A complementary,
descriptive view asks how often the event follows a contact episode, depending
on its duration.

#. In the **Contact episodes** part of the window, check **event after each
   contact episode** (1). It needs the pair measurements.

#. Set the **window [frames]** in which the event counts (2), and after the
   ``end`` or the ``start`` of the episode (3):

   *   ``end``: among the cells alive at the end of the episode. The duration
       of the contact is fully seen, but a cell dying during the contact is
       left out.
   *   ``start``: among the cells alive at its start. The deaths during the
       contact count, but a death can cut a contact short, which makes long
       contacts look safer than they are.

#. (Optional) Bridge short interruptions with **max gap [frames]**, and set
   the **duration bins** (in frames, ``inf`` for no upper bound) (4).

#. Click **Submit**.

A table of the episodes kept opens, with their ``outcome`` (1 when the event
happened in the window), and the **Event after contact episodes** window plots
the fraction of episodes followed by the event per class of duration, with its
95% interval and the counts above (5). An episode is kept when the cell is
seen for the whole window or has the event in it.

.. warning::
    The episodes of one cell, or of one effector, are not independent: read
    this plot as a description, and the hazard model for the test.

.. seealso::
    :ref:`Contact episodes in the Table Explorer <pair_collapsing>` to inspect
    the episodes themselves. |
    :ref:`Contact Hazard Settings <ref_contact_hazard_settings>` for all the
    options.


Check the analysis on synthetic data
------------------------------------

Celldetective can write a synthetic effector-target experiment whose contacts
and deaths are known exactly, to check that an analysis finds an effect where
there is one, and none where there is none:

.. code-block:: console

    python -m celldetective.utils.synthetic_interactions SyntheticCausal --size 320 320 --frames 60 --targets 16 --effectors 16 --death-probability 0.08 --with-labels --with-tables
    python -m celldetective.utils.synthetic_interactions SyntheticNull --size 320 320 --frames 60 --targets 16 --effectors 16 --background-death 0.01 --with-labels --with-tables

With ``--death-probability``, a target dies with this chance after each frame
it spends in contact with an effector: the contacts cause the deaths. With
``--background-death``, every target dies with this chance at each frame,
whatever its contacts: there is no effect to find. ``--with-labels`` and
``--with-tables`` write the true masks and trajectories (with ``t_death`` and
``class_death``), so you can open the experiment, compute a mask-contact
neighborhood between ``targets`` and ``effectors`` and run the contact hazard
directly. The true contacts and contact episodes are in the
``ground_truth`` folder of the position.
