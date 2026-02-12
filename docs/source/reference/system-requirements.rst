System Requirements
===================

Hardware Requirements
---------------------

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
---------------------

*   **OS**:

    *   Windows 10/11
    *   Linux (Ubuntu 20.04 LTS recommended)
    *   MacOS (Experimental, TensorFlow setup varies)
    
*   **Python**: Version 3.9 to 3.11.
*   **Dependencies**: managed via pip/conda (see :doc:`/get-started`).
