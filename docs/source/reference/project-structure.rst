Project Structure
=================

.. _ref_project_structure:

This reference documents the directory structure and file organization of a Celldetective experiment.

Experiment Project
------------------

An **Experiment Project** is a directory that contains all configuration and data for a single experiment.

.. code-block:: text

    MyExperiment/
    ├── config.ini             # Experiment configuration file
    ├── W1/                    # Well folder 1
    ├── W2/                    # Well folder 2
    └── ...

.. _well-folders:

Well Folders
------------

Each **Well Folder** represents a single acquisition site or experimental unit.

**Naming Convention**

*   **Pattern**: ``W{number}`` (e.g., ``W1``, ``W12``).
*   **Regex**: ``^W\d+/?$``

**Structure**

A well folder contains subfolders for each imaging **Position** (Field of View).

.. code-block:: text

    W1/
    ├── 100/                   # Position folder (0)
    ├── 101/                   # Position folder (1)
    └── ...

Position Folders
----------------

Each **Position Folder** corresponds to a specific Field of View (FOV). The naming convention usually follows ``{WellNumber}0{PositionIndex}``.

**Contents**

.. code-block:: text

    100/
    ├── movie/                 # Input image stacks
    │   └── images.tif
    ├── labels_nuclei/         # Segmentation masks (generated)
    ├── output/                # Analysis output
    │   └── tables/            # CSV results
    └── metadata.json          # Position metadata

.. seealso::
    :doc:`../concepts/data-organization` for the conceptual overview of this hierarchy.
