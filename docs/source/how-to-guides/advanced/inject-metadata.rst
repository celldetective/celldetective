How to inject metadata in the tables?
=====================================

This guide shows you how to inject per-well or global metadata into the single cell tables.

Reference keys: :term:`single-cell measurement`

The metadata live in the experiment configuration file, ``config.ini`` (see :ref:`ref_experiment_config`). Celldetective edits it for you in the **Configuration** window.

.. figure:: ../../_static/figures/config-editor.svg
    :width: 100%
    :target: ../../_static/figures/config-editor.svg
    :align: center
    :alt: the configuration editor

    **The configuration editor.** The *Well labels* tab holds one row per well and one column per label; the *Metadata* tab holds the values shared by the whole experiment.

In the *Well labels* tab, the tools of the header add, rename or remove a label (1), and each label is a column holding one value per well (2). In the *Metadata* tab, the tools add or remove an entry (4), and each entry holds a value shared by every well (5). The :icon:`file-cog,black` button opens ``config.ini`` in a text editor instead (3). Nothing is written to the file before **Save** (6).

**Step-by-step:**

#. Open a project.

#. Press the :icon:`cog-outline,black` button of the control panel to open the **Configuration** window.

**Case 1: add a metadata label per well:**

#. Go to the **Well labels** tab.

#. Press the :icon:`table-column-plus-after,black` button and name the new label. Names are stored in lowercase, with underscores instead of spaces.

#. Fill in one value per well. A block of cells copied from a spreadsheet can be pasted with :kbd:`Ctrl+V`. Values cannot contain commas.

The four default labels (``cell_types``, ``antibodies``, ``concentrations`` and ``pharmaceutical_agents``) can be edited but not renamed or removed. A label you added can be renamed (:icon:`pencil,black`) or removed (:icon:`table-column-remove,black`).

**Case 2: add global metadata:**

#. Go to the **Metadata** tab.

#. Press the :icon:`table-row-plus-after,black` button and type a key and its value, e.g. ``date`` and ``2024-03-27``.

Press **Save** (or :kbd:`Ctrl+S`). The configuration is written to ``config.ini`` and reloaded at once in the control panel.

After computing single-cell measurements, you should be able to see these metadata as columns of the tables by pressing the :icon:`table,#1565c0` :blue:`Explore table` button for the population of interest.

.. tip::

    The **Settings** tab gives access to the other sections of the file, such as the movie prefix, the calibration or the channel indices.

.. _change-movie-prefix:

Change the movie prefix
~~~~~~~~~~~~~~~~~~~~~~~

The movie prefix tells which stack of a position's ``movie/`` folder is the movie: the ``.tif`` file whose name starts with it. Preprocessing writes new stacks next to the original one (``Corrected_`` for a background correction or a registration), and the prefix is what switches the analysis over to them.

.. figure:: ../../_static/figures/movie-prefix.svg
    :width: 85%
    :target: ../../_static/figures/movie-prefix.svg
    :align: center
    :alt: the movie prefix field of the configuration editor and the prefixes it suggests

    **The movie prefix field.** The prefixes cut from the names of the stacks the experiment holds, and the line telling what the prefix typed matches.

#. Open the **Configuration** window (:icon:`cog-outline,black` button of the control panel) on its **Settings** tab, section *MovieSettings*.

#. Press the :icon:`text-search,black` button next to ``movie_prefix`` (1) to list the prefixes the stacks of the experiment offer, the ones selecting a stack in the most positions first, or start typing to get the matching ones. The names of the stacks are cut around their separators and their numbering to build them.

#. Read the line under the field (2): it tells, as you type, how many positions hold a matching stack. It turns red when a prefix leaves positions without a movie, or matches several stacks in a position, in which case which one is loaded is left to chance. Such a prefix is better fixed here than found out at the first segmentation.

#. Press **Save**.
