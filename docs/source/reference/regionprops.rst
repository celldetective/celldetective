Custom ``regionprops`` Implementation
======================================

.. _ref_regionprops:

Celldetective uses a custom ``regionprops`` implementation
(``celldetective.regionprops``) in place of scikit-image's
``skimage.measure.regionprops_table`` for all single-cell measurements.
This page documents how it differs from the original and why.

Overview
--------

The custom implementation is a thin subclass of scikit-image's
``RegionProperties``, extended with three capabilities that the original
does not support:

1. **Unmasked intensity image** — thresholding inside the bounding box works correctly.
2. **Named channel support** — channel indices are associated with semantic names.
3. **Channel-targeted extra properties** — a function can declare which channel it applies to via the default value of a ``target_channel`` parameter.

Additionally, ``regionprops_table`` silently drops properties whose values
are all ``NaN`` across all cells, rather than propagating empty columns.

Difference 1: Unmasked intensity image
---------------------------------------

In scikit-image's original implementation, the ``image_intensity`` property
returns the bounding-box crop **multiplied by the cell mask**:

.. code-block:: python

    # scikit-image original
    return self._intensity_image[self.slice] * image   # image = boolean mask

Background pixels (outside the cell) are zeroed before the measurement
function sees them.

Celldetective's ``CustomRegionProps`` overrides this property to return the
raw bounding-box crop **without masking**:

.. code-block:: python

    # CustomRegionProps
    return self._intensity_image[self.slice]            # no masking

**Why:** threshold-based custom measurements (e.g. ``area_dark_intensity``)
need to classify pixels as "dark" relative to the actual intensity distribution
inside the cell. If the background were zeroed, those zero-valued pixels would
be misclassified as dark signal, inflating the result. With unmasked crops, the
threshold is applied to real pixel values; the cell mask is then used separately
to restrict which pixels are counted.

.. warning::

    Because background pixels are not zeroed, any custom function that uses
    raw pixel values (e.g. mean, sum) without applying ``regionmask`` will
    include pixels outside the cell boundary. Always restrict computations to
    the region of interest:

    .. code-block:: python

        def my_measurement(regionmask, intensity_image, **kwargs):
            return np.mean(intensity_image[regionmask])   # correct
            # not: np.mean(intensity_image)               # would include background


Difference 2: Named channel support
-------------------------------------

Standard ``regionprops_table`` has no concept of channel names — channels are
identified only by integer index.

``regionprops_table`` in Celldetective accepts an optional ``channel_names``
list that is passed down to every ``CustomRegionProps`` instance. When
present, ``channel_names`` is used to:

- Validate that the number of names matches the number of channels in the
  image (raises ``ValueError`` if not).
- Enable channel targeting in extra-property functions (see below).
- Drive the column-renaming step in ``data_cleaning.py``, which replaces
  the raw ``intensity`` prefix with the actual channel name in output
  column names (e.g. ``intensity_mean-0`` → ``GFP_mean``).


Difference 3: Channel-targeted extra properties
-------------------------------------------------

Extra-property functions follow one of two calling conventions:

**All-channel functions** (no ``target_channel`` parameter)
    Called once per cell per channel. ``intensity_image`` receives a
    2-D single-channel crop for each iteration:

    .. code-block:: python

        def my_measurement(regionmask, intensity_image, **kwargs):
            return np.mean(intensity_image[regionmask])

**Channel-targeted functions** (``target_channel`` with a default value)
    The function declares which channel it applies to through the **default
    value** of ``target_channel``:

    .. code-block:: python

        def my_measurement(regionmask, intensity_image, target_channel='GFP', **kwargs):
            return np.mean(intensity_image[regionmask])

    The framework uses ``inspect.signature`` to read the default value at
    runtime. The function is then called **once**, receiving only that
    channel's 2-D crop. All other channel output slots are filled with
    ``NaN``. ``target_channel`` is **never passed as an argument**; only its
    default is inspected.

    If the named channel is absent from the experiment's ``channel_names``,
    a warning is logged and all output slots remain ``NaN``.

**Decision table:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Signature
     - Number of calls per cell
     - ``intensity_image`` content
   * - No ``target_channel``
     - Once per channel (N calls)
     - Single-channel 2-D crop for each channel
   * - ``target_channel='name'``
     - Once (for the named channel)
     - Single-channel 2-D crop for ``'name'`` only; other slots → NaN


Difference 4: All-NaN property filtering
-----------------------------------------

Scikit-image's ``regionprops_table`` returns a column for every requested
property regardless of whether any cell produced a valid value.

Celldetective's version tests each property across all cells before
assembling the output table:

.. code-block:: python

    good_props = []
    for prop in properties:
        try:
            nan_test = [np.isnan(getattr(r, prop)) for r in regions]
            if not np.all(nan_test):
                good_props.append(prop)
        except AttributeError:
            logger.warning(f"Could not measure {prop}... Skip...")

Properties where every cell returned ``NaN`` (e.g. a channel-targeted
function whose channel is absent from the experiment) are silently excluded
from the output DataFrame. This keeps the measurement table free of
uninformative all-NaN columns without requiring the caller to clean up.

.. seealso::

    :doc:`../how-to-guides/advanced/write-a-custom-measurement` — how to write
    a custom measurement function that integrates with this system.

    :ref:`ref_measurements` — reference for all built-in measurement column names.
