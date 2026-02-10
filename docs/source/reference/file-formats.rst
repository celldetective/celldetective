Supported File Formats
======================

.. _ref_file_formats:

Input Formats
-------------

Celldetective supports multichannel time-lapse microscopy data saved as TIFF stacks.

Supported dimensions:

*   **XY** (2D): Single-timepoint & single-channel image.
*   **CXY** (3D): Single-timepoint & multichannel image.
*   **PXY** (3D): Multi-position, single-channel & single-timepoint images.
*   **TXY** (3D): Time-lapse images (single channel).
*   **PCXY** (4D): Multi-position, multichannel & single-timepoint images.
*   **TCXY** (4D): Multi-channel time-lapse images.

Supported file types:

*   Standard TIFF (``.tif``, ``.tiff``)
*   OME-TIFF (``.ome.tif``, ``.ome.tiff``)

.. note::
    **Z-stacks**: Native Z-stack support is not available. However, a Z-axis can be treated as a Time-axis to segment each slice independently. There is no compatibility for data having both Z-slices and Time-points.


Metadata Handling
-----------------

*   **Standard TIFF**: Dimensions are inferred or manually configured during experiment creation.
*   **OME-TIFF**: Axis metadata (XYZCT) is read automatically.
*   **Micro-Manager**: Multichannel stacks acquired with Micro-Manager are deinterlaced automatically.

.. tip::
    For large stacks (> 5 GB), use the **Bio-Formats Exporter** plugin in ImageJ to ensure efficient read performance.
