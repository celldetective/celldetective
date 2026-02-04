Installation Reference
======================

This page details advanced installation options and troubleshooting steps.

.. _ref_dev_installation:

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
