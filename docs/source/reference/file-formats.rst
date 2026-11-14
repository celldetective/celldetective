Supported File Formats
======================

.. _ref_file_formats:

Input Formats
-------------

Celldetective reads multichannel time-lapse microscopy data saved as TIFF stacks, one file per position, in the ``movie`` folder of the position.

Supported file types:

*   Standard TIFF (``.tif``, ``.tiff``), with or without ImageJ metadata
*   OME-TIFF (``.ome.tif``, ``.ome.tiff``), including sets split over several files, and files written by Bio-Formats
*   Micro-Manager stacks (MMStack and NDTiff)

The movie of a position is the file whose name starts with the movie prefix of the experiment. If several files match, the first in natural order is used and a warning names the others.

Supported dimensions:

*   **XY** (2D): Single-timepoint & single-channel image.
*   **CXY** (3D): Single-timepoint & multichannel image.
*   **TXY** (3D): Time-lapse images (single channel).
*   **TCXY** (4D): Multi-channel time-lapse images.
*   **ZXY**, **ZCXY**: Z-stacks without time. Each slice is read as a frame, so it is segmented on its own.
*   **TZXY**, **TZCXY**: Z-stacks over time. Each time point is read as the maximum intensity projection of its slices.
*   **RGB** images and time-lapses: the red, green and blue samples are read as three channels.
*   **PXY**, **PCXY**: several single-timepoint positions stacked in one file are read as the frames of one movie.

A file holding several positions as separate series (an OME-TIFF with several ``Image`` elements, for example) is read for its first position only. Export one file per position instead.


Metadata Handling
-----------------

*   **Axes**: Celldetective reads the axis labels of the ImageJ or OME metadata. A TIFF without metadata is read as frames, then channels, along its leading axes.
*   **Channels**: The number of channels comes from the experiment (its ``config.ini``). When a file has a channel axis but no time and no Z axis, as many 3D files written without axis labels do, that axis is read as channels if its size equals the number of channels of the experiment, and as frames otherwise. Outside an experiment, an axis of more than 5 planes is read as frames.
*   **Plane order**: Each image is read where the metadata says it is stored, whatever the order of the planes in the file. This covers channel-first files, Bio-Formats files, multi-file OME-TIFF sets and NDTiff.

When a file cannot be read as it is labelled, Celldetective reads what it can and logs a warning:

*   other positions of the file are dropped (see above);
*   missing planes, such as a missing file of a multi-file set or an acquisition stopped early, are read as black images;
*   the slices of a movie with both Z and time are projected;
*   axes other than time, Z and channels (Micro-Manager Magellan tiles, for example) are read at their first index only;
*   if the metadata cannot be parsed, or tifffile cannot read the dimensions of the file, its pages are read in order, as frames of the experiment's channels;
*   a file that cannot be opened at all is reported with a message asking to re-save it.

.. tip::
    For large stacks, save the movie as an uncompressed TIFF with ImageJ (*File > Save As > Tiff...*). Such a file is memory-mapped, and napari opens it lazily, frame by frame. A compressed movie, or one written with metadata between planes (as Bio-Formats does), is read again for every frame shown, which is slower.
