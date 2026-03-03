Installation Reference
======================

This page details system requirements, advanced installation options and troubleshooting steps.

System Requirements
-------------------

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

Celldetective is versatile and can run on standard workstations or high-performance clusters.

*   **CPU**: Modern multi-core processor (Intel Core i7/i9 or equivalent).
*   **RAM**:

    *   Minimum: Sufficient to load a single movie stack into memory (dependent on image size).
    *   Recommended: 16 GB+ for smooth visualization in Napari.

*   **GPU (Optional but Recommended)**:

    *   NVIDIA GPU with CUDA support (e.g., RTX 3070, 8GB VRAM).
    *   Greatly accelerates Deep Learning inference (:term:`StarDist`, :term:`Cellpose`).
    *   *Note*: CPU-only mode is fully supported but slower.

Software Requirements
~~~~~~~~~~~~~~~~~~~~~

*   **OS**:

    *   Windows 10/11
    *   Linux (Ubuntu 20.04 LTS recommended)
    *   MacOS (Experimental, TensorFlow setup varies)
    
*   **Python**: Version 3.9 to 3.11.
*   **Dependencies**: managed via pip/conda (see :doc:`/get-started`).

.. _ref_dev_installation:

Standard Installation
---------------------

We recommend using `conda` to create a clean environment for Celldetective.

1.  **Create an environment** (Python 3.9 - 3.11):

    .. code-block:: console

        $ conda create -n celldetective python=3.11 pyqt
        $ conda activate celldetective

2.  **Install Celldetective**:

    .. code-block:: console

        $ pip install celldetective[all]


Development Version
-------------------

To run the latest development version:

1.  Clone the repository:

    .. code-block:: console

        $ git clone git://github.com/celldetective/celldetective.git
        $ cd celldetective

2.  Create and activate environment:

    .. code-block:: console

        $ conda create -n celldetective python=3.11 pyqt
        $ conda activate celldetective

3.  Install in editable mode:

    .. code-block:: console

        $ pip install -r requirements.txt
        $ pip install -e .

Direct Install from GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    $ pip install git+https//github.com/celldetective/celldetective.git

.. _ref_install_troubleshooting:

Troubleshooting
---------------

**Microsoft Visual C++ (Windows)**

The installation of ``mahotas`` on Windows requires Microsoft Visual C++ 14.0 or greater.
Download it from the `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.

**NVIDIA GPU Support**

To use your NVIDIA GPU, ensure you have installed:
*   Proper NVIDIA Drivers
*   CUDA Toolkit
*   cuDNN libraries

We recommend installing TensorFlow with CUDA support via conda or pip (e.g., ``pip install tensorflow[and-cuda]``).
