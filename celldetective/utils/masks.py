from typing import Union, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk
from celldetective.log_manager import get_logger

logger = get_logger(__name__)


def contour_of_instance_segmentation(label, distance, sdf=None, voronoi_map=None):
    """

    Generate an instance mask containing the contour of the segmented objects.

    This updated version uses a Signed Distance Field (SDF) and Voronoi tessellation approach
    Generic enough to handle Inner contours, Outer contours, and arbitrary "stripes" (annuli).

    Parameters
    ----------
    label : ndarray
            The instance segmentation labels.
    distance : int, float, list, tuple, or str
            The distance specification.
            - Scalar > 0: Inner contour (Erosion) from boundary to 'distance' pixels inside. Range [0, distance].
            - Scalar < 0: Outer contour (Dilation) from 'distance' pixels outside to boundary. Range [distance, 0].
            - Tuple/List (a, b): Explicit range in SDF space.
               - Positive values are inside the object.
               - Negative values are outside methods.
               - Example: (5, 10) -> Inner ring 5 to 10px deep.
               - Example: (-10, -5) -> Outer ring 5 to 10px away.
            - String "rad1-rad2": Interpretation for Batch Measurements (Outer Rings).
               - Interpreted as an annular region OUTSIDE the object.
               - "5-10" -> Range [-10, -5] in SDF space (5 to 10px away).
    sdf : ndarray, optional
            Pre-computed Signed Distance Field (dist_in - dist_out).
            If provided, avoids recomputing EDT.
    voronoi_map : ndarray, optional
            Pre-computed Voronoi map of instance labels.
            Required if sdf is provided and outer contours are needed.

    Returns
    -------
    border_label : ndarray
            An instance mask containing the contour of the segmented objects.
            Outer contours preserve instance identity via Voronoi propagation.

    """
    from scipy.ndimage import distance_transform_edt

    # helper to parse string "rad1-rad2"
    if isinstance(distance, str):
        try:
            # Check for stringified tuple "(a, b)"
            distance = distance.strip()
            if distance.startswith("(") and distance.endswith(")"):
                import ast

                val_tuple = ast.literal_eval(distance)
                if isinstance(val_tuple, (list, tuple)) and len(val_tuple) == 2:
                    min_r = val_tuple[0]
                    max_r = val_tuple[1]
                else:
                    raise ValueError("Tuple string must have 2 elements")
            else:
                try:
                    val = float(distance)
                    # It's a scalar string like "5" or "-5"
                    if val >= 0:
                        min_r = 0
                        max_r = val
                    else:
                        min_r = val
                        max_r = 0
                except ValueError:
                    # It's a range string "5-10"
                    parts = distance.split("-")
                    # Assumption: "A-B" where A, B positive radii for OUTER annulus.
                    r1 = float(parts[0])
                    r2 = float(parts[1])
                    min_r = -max(r1, r2)
                    max_r = -min(r1, r2)

        except Exception:
            logger.warning(
                f"Could not parse contour string '{distance}'. returning empty."
            )
            return np.zeros_like(label)

    elif isinstance(distance, (list, tuple)):
        min_r = distance[0]
        max_r = distance[1]

    elif isinstance(distance, (int, float)):
        if distance >= 0:
            min_r = 0
            max_r = distance
        else:
            min_r = distance
            max_r = 0
    else:
        return np.zeros_like(label)

    if sdf is None or voronoi_map is None:
        # Compute SDF maps
        # We need SDF = dist_in - dist_out
        # inside > 0, outside < 0

        # 1. Dist In (Inside object)
        dist_in = distance_transform_edt(label > 0)

        # 2. Dist Out (Outside object) + Voronoi
        dist_out, indices = distance_transform_edt(label == 0, return_indices=True)

        # Voronoi Map
        voronoi_map = label[indices[0], indices[1]]

        # Composite SDF
        sdf = dist_in - dist_out

    # Create Mask
    mask = (sdf >= min_r) & (sdf <= max_r)

    # Result
    border_label = voronoi_map * mask

    return border_label


def create_patch_mask(h, w, center=None, radius=None):
    """

    Create a circular patch mask of given dimensions.
    Adapted from alkasm on https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    Parameters
    ----------
    h : int
            Height of the mask. Prefer odd value.
    w : int
            Width of the mask. Prefer odd value.
    center : tuple, optional
            Coordinates of the center of the patch. If not provided, the middle of the image is used.
    radius : int or float or list, optional
            Radius of the circular patch. If not provided, the smallest distance between the center and image walls is used.
            If a list is provided, it should contain two elements representing the inner and outer radii of a circular annular patch.

    Returns
    -------
    numpy.ndarray
            Boolean mask where True values represent pixels within the circular patch or annular patch, and False values represent pixels outside.

    Notes
    -----
    The function creates a circular patch mask of the given dimensions by determining which pixels fall within the circular patch or annular patch.
    The circular patch or annular patch is centered at the specified coordinates or at the middle of the image if coordinates are not provided.
    The radius of the circular patch or annular patch is determined by the provided radius parameter or by the minimum distance between the center and image walls.
    If an annular patch is desired, the radius parameter should be a list containing the inner and outer radii respectively.

    Examples
    --------
    >>> mask = create_patch_mask(100, 100, center=(50, 50), radius=30)
    >>> print(mask)

    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    if isinstance(radius, int) or isinstance(radius, float):
        mask = dist_from_center <= radius
    elif isinstance(radius, list):
        mask = (dist_from_center <= radius[1]) * (dist_from_center >= radius[0])
    else:
        print("Please provide a proper format for the radius")
        return None

    return mask
