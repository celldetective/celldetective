How to inject metadata in the tables?
=====================================

This guide shows you how to inject per-well or global metadata into the single cell tables.

Reference keys: :term:`single-cell measurement`

**Step-by-step:**

#. Open a project.

#. Press the :icon:`cog-outline,black` button to view the experiment configuration parameters.

#. Press the :icon:`file-cog,black` button in the top-right corner to edit the configuration file with a text editor.

**Case 1: add a metadata label per-well:**

#. Find the ``[Labels]`` section. If it does not exist, create it. It usually includes ``cell_types``, ``antibodies``, ``concentrations`` and ``pharmaceutical_agents``.

#. Add a new entry in the section following the same template as existing. Example for three wells: ``new_label = val 1, val 2, val 3``. Example for one well: ``new_label = val 1``. Ensure that you have strictly as many values as there are wells.

#. Type :kbd:`Ctrl+S` to save the configuration file.

**Case 2: add global metadata:**

#. Find the ``[Metadata]`` section. If it does not exist, create it.

#. Add a new entry in the section following the template ``new_metadata = value``.

#. Type :kbd:`Ctrl+S` to save the configuration file.


To see the updated config, close the experiment configuration editor window (both the text editor and the Celldetective config window). Reopen the window by pressing the :icon:`cog-outline,black` button.

After computing single-cell measurements, you should be able to see and write these metadata by pressing the :icon:`table,#1565c0` :blue:`Explore table` button for the population of interest.