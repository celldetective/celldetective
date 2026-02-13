Tracking
========

.. _track:

The Tracking module links segmented cells across frames to create trajectories. This allows you to analyze cell motility, lineage, and dynamic behaviors.


Overview
--------

After segmentation, individual cell detections exist independently in each frame. Tracking connects these detections across time to form trajectories, assigning a persistent identity to each cell. This is essential for any time-resolved analysis — measuring speed, detecting events such as division or death, and studying interactions between populations.


Available trackers
------------------

Celldetective integrates two tracking algorithms:

*   :term:`bTrack` [#]_ (default) — a Bayesian tracker that uses Kalman filters and cell features to predict motion. It handles complex behaviors such as division and apoptosis, and is the recommended choice for crowded scenes.
*   **trackpy** — a Crocker–Grier particle tracker well-suited for simple Brownian motion.

Both trackers produce a table of cell positions, identities, and (optionally) morphological or intensity features per frame. Results are saved as a CSV file (``trajectories_<population>.csv``) in the ``output/tables`` folder of each position.


Post-processing
~~~~~~~~~~~~~~~

After tracking, optional post-processing can be applied to clean up results:

*   Filter out short tracks.
*   Interpolate gaps (missing detections within a track).
*   Extrapolate positions backwards or forwards to the movie boundaries.

For a full list of post-processing and tracker parameters, see the :ref:`Tracking Settings Reference <ref_tracking_settings>`.


How-to guides
-------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Task
     - Guide
   * - Configure a tracker and run it on your data
     - :doc:`how-to <how-to-guides/basics/configure-and-run-tracking>`
   * - Correct a tracking error
     - :doc:`how-to <how-to-guides/basics/correct-a-track>`


References
----------

.. [#] Ulicna, K., Vallardi, G., Charras, G. & Lowe, A. R. Automated Deep Lineage Tree Analysis Using a Bayesian Single Cell Tracking Approach. Frontiers in Computer Science 3, (2021).
