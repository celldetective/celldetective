from typing import Union, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import disk


def contour_of_instance_segmentation(
    label: np.ndarray, distance: Union[int, List[int], Tuple[int]]
):
    """

    Generate an instance mask containing the contour of the segmented objects.

    Parameters
    ----------
    label : ndarray
            The instance segmentation labels.
    distance : int, float, list, or tuple
            The distance or range of distances from the edge of each instance to include in the contour.
            If a single value is provided, it represents the maximum distance. If a tuple or list is provided,
            it represents the minimum and maximum distances.

    Returns
    -------
    border_label : ndarray
            An instance mask containing the contour of the segmented objects.

    Notes
    -----
    This function generates an instance mask representing the contour of the segmented instances in the label image.
    It use the distance_transform_edt function from the scipy.ndimage module to compute the Euclidean distance transform.
    The contour is defined based on the specified distance(s) from the edge of each instance.
    The resulting mask, `border_label`, contains the contour regions, while the interior regions are set to zero.

    Examples
    --------
    >>> border_label = contour_of_instance_segmentation(label, distance=3)
    # Generate a binary mask containing the contour of the segmented instances with a maximum distance of 3 pixels.

    """
    if isinstance(distance, (list, tuple)):

        edt = distance_transform_edt(label)

        if isinstance(distance, list) or isinstance(distance, tuple):
            min_distance = distance[0]
            max_distance = distance[1]

        elif isinstance(distance, (int, float)):
            min_distance = 0
            max_distance = distance

        thresholded = (edt <= max_distance) * (edt > min_distance)
        border_label = np.copy(label)
        border_label[np.where(thresholded == 0)] = 0

    else:
        size = (2 * abs(int(distance)) + 1, 2 * abs(int(distance)) + 1)
        dilated_image = ndimage.grey_dilation(
            label, footprint=disk(int(abs(distance)))
        )  # size=size,
        border_label = np.copy(dilated_image)
        matching_cells = np.logical_and(dilated_image != 0, label == dilated_image)
        border_label[np.where(matching_cells == True)] = 0
        border_label[label != 0] = 0.0

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
