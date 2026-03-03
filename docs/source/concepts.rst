Concepts
========

.. _concepts:

This section provides background information and theoretical context for Celldetective. Unlike practical guides or tutorials, these pages explain **why** things are done a certain way and define the core abstractions used throughout the software.

.. grid:: 2
    :gutter: 3

    .. grid-item-card:: :icon:`folder-network` Data Organization
        :link: concepts/data-organization
        :link-type: doc

        How Celldetective structures projects, experiments, well folders, and metadata.

    .. grid-item-card:: :icon:`shape-outline` Segmentation Strategies
        :link: concepts/population-specific-segmentation
        :link-type: doc

        The logic behind separating cell populations and refining masks.

    .. grid-item-card:: :icon:`calendar-clock` Events
        :link: concepts/single-cell-event
        :link-type: doc

        The theoretical framework for defining and detecting single-cell events.

    .. grid-item-card:: :icon:`chart-bell-curve-cumulative` Survival Analysis
        :link: concepts/survival
        :link-type: doc

        How survival theory is adapted for cellular event duration analysis.

    .. grid-item-card:: :icon:`account-network` Neighborhoods
        :link: concepts/neighborhood
        :link-type: doc

        Definitions of cellular interaction and proximity metrics.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Concepts:

   concepts/data-organization
   concepts/population-specific-segmentation
   concepts/single-cell-event
   concepts/survival
   concepts/neighborhood
