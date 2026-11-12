"""
Write small OME-TIFF files laid out as Bio-Formats writes them.

``tifffile.imwrite(..., ome=True)`` stores a series as one contiguous block,
which Bio-Formats does not. A file exported by Bio-Formats (bfconvert, Fiji's
Bio-Formats Exporter, most acquisition software) has:

- one IFD per plane, written between the planes, so the series is not one
  contiguous block of pixels;
- strips of one row;
- the OME-XML in the first IFD, mapping every plane to its (Z, C, T) with
  ``TiffData`` elements, in the plane order given by ``DimensionOrder``;
- for a multi-file set, the same OME-XML in every file, with ``UUID``
  elements naming the file that holds each plane;
- for several positions, one ``Image`` per position in the same file.

The official samples (https://downloads.openmicroscopy.org/images/OME-TIFF/,
e.g. ``bioformats-artificial`` and ``tubhiswt-4D``) behave in celldetective
as the files written here do; the tests use these instead to stay offline.
"""

import itertools
import os
import uuid

import numpy as np
from tifffile import TiffWriter

OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _planes_in_order(dimension_order, sizes):
    """``(z, c, t)`` of every plane, in file order. In ``DimensionOrder``
    the first letter after ``XY`` varies fastest."""
    dims = dimension_order[2:]
    slowest_first = list(reversed(dims))
    for combo in itertools.product(*(range(sizes[d]) for d in slowest_first)):
        index = dict(zip(slowest_first, combo))
        yield index["Z"], index["C"], index["T"]


def write_bioformats_ome(
    path, images, dimension_order="XYZCT", one_file_per_channel=False
):
    """
    Write ``images`` as a Bio-Formats OME-TIFF.

    Parameters
    ----------
    path : str
        Path ending in ``.ome.tif``. With ``one_file_per_channel``, the files
        are ``<stem>_C<c>.ome.tif`` next to it.
    images : list of ndarray
        One ``(T, C, Z, Y, X)`` uint16 array per position (OME ``Image``).
    dimension_order : str
        OME ``DimensionOrder``, e.g. ``"XYZCT"`` or ``"XYZTC"``.
    one_file_per_channel : bool
        Write a multi-file set, one file per channel.

    Returns
    -------
    list of str
        The files written.
    """
    folder = os.path.dirname(path)
    stem = os.path.basename(path)[: -len(".ome.tif")]
    n_channels = images[0].shape[1]
    if one_file_per_channel:
        keys = list(range(n_channels))
        names = {c: f"{stem}_C{c}.ome.tif" for c in keys}
    else:
        keys = [None]
        names = {None: os.path.basename(path)}
    uuids = {k: f"urn:uuid:{uuid.uuid4()}" for k in keys}
    planes = {k: [] for k in keys}

    xml_images = []
    for i, data in enumerate(images):
        size_t, size_c, size_z, size_y, size_x = data.shape
        sizes = {"T": size_t, "C": size_c, "Z": size_z}
        tiff_data = []
        for z, c, t in _planes_in_order(dimension_order, sizes):
            key = c if one_file_per_channel else None
            ifd = len(planes[key])
            planes[key].append(np.ascontiguousarray(data[t, c, z]))
            tiff_data.append(
                f'<TiffData FirstC="{c}" FirstT="{t}" FirstZ="{z}" IFD="{ifd}" '
                f'PlaneCount="1"><UUID FileName="{names[key]}">{uuids[key]}'
                f"</UUID></TiffData>"
            )
        channels = "".join(
            f'<Channel ID="Channel:{i}:{c}" SamplesPerPixel="1"/>'
            for c in range(size_c)
        )
        xml_images.append(
            f'<Image ID="Image:{i}" Name="position {i}">'
            f'<Pixels BigEndian="false" DimensionOrder="{dimension_order}" '
            f'ID="Pixels:{i}" SizeC="{size_c}" SizeT="{size_t}" SizeX="{size_x}" '
            f'SizeY="{size_y}" SizeZ="{size_z}" Type="uint16">'
            f'{channels}{"".join(tiff_data)}</Pixels></Image>'
        )

    written = []
    for key in keys:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<OME xmlns="{OME_NS}" Creator="OME Bio-Formats 6.0.0" '
            f'UUID="{uuids[key]}">{"".join(xml_images)}</OME>'
        )
        file_path = os.path.join(folder, names[key])
        with TiffWriter(file_path) as tif:
            for j, plane in enumerate(planes[key]):
                tif.write(
                    plane,
                    contiguous=False,
                    rowsperstrip=1,
                    metadata=None,
                    description=xml if j == 0 else None,
                )
        written.append(file_path)
    return written
