Concepts
========

.. _concepts:

This section provides background information and theoretical context for Celldetective. Unlike practical guides or tutorials, these pages explain **why** things are done a certain way and define the core abstractions used throughout the software.

You will find explanations of:

*   **Data Organization**: How Celldetective structures projects, experiments, and metadata.
*   **Segmentation Strategies**: The logic behind separating cell populations and refining masks.
*   **Events**: The theoretical framework for defining and detecting single-cell events (e.g., division, death).
*   **Survival Analysis**: How survival theory is adapted for cellular event duration.
*   **Neighborhoods**: Definitions of cellular interaction and proximity metrics.

.. toctree::
   :maxdepth: 1
   :caption: Concepts:

   concepts/data-organization
   concepts/population-specific-segmentation
   concepts/single-cell-event
   concepts/survival
