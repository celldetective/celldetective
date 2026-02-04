Get started
===========

.. _get_started:


Installation
------------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

We recommend using `conda` to create a clean environment for Celldetective.

1.  **Create an environment** (Python 3.9 - 3.11):

    .. code-block:: console

        $ conda create -n celldetective python=3.11 pyqt
        $ conda activate celldetective

2.  **Install Celldetective**:

    .. code-block:: console

        $ pip install celldetective

**That's it!** You are ready to launch.

*For development versions, troubleshooting (e.g., Visual C++ errors), or manual installation, see the :ref:`Installation Reference <ref_dev_installation>`.*


Launching the GUI
-----------------

Once the pip installation is complete, open a terminal and run:

.. code-block:: console

	$ python -m celldetective


.. figure:: _static/launch.gif
    :width: 100%
    :align: center
    :alt: static_class

    How to launch the software from a terminal, here, without an environment


A startup image is displayed during the loading of the python libraries. Upon completion, the first window of the software opens. 

You can either create a new experiment (button New or shortcut Ctrl+N) or load one.