"""
Neighborhood Module
===================

This module provides improved algorithms for defining and analyzing local neighborhoods of cells.
It supports various definitions of "neighborhood," including distance-based and contact-based criteria.

Key Features
------------
-   **Distance-Based Neighbors**: Identifies neighbors within a specified radius.
-   **Contact-Based Neighbors**: Identifies neighbors that are physically touching (based on segmentation masks).
-   **Neighborhood Metrics**: Computes statistics about the local environment, such as density or neighbor types.

Main Functions
--------------
-   `distance_cut_neighborhood`: Main function to compute distance-based neighborhoods for a set of objects.
-   `contact_neighborhood`: Computes contact-based neighborhoods using mask adjacency.
-   `compute_attention_weight`: Calculates weights based on proximity, useful for weighted neighborhood metrics.

Integrations
------------
The output of this module (neighborhood lists) is often used by `relative_measurements.py` to compute detailed pair-wise statistics.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import os
from itertools import product

from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.utils.data_cleaning import extract_identity_col
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation
from celldetective.utils.image_loaders import (
    locate_labels,
    locate_stack,
    locate_stack_lazy,
)
from celldetective.utils.data_loaders import get_position_table, get_position_pickle

logger = logging.getLogger("celldetective")

# Distance given to the pairs of cells that are not neighbors (masks not in
# contact, or a cell and itself), so that no cut-off distance keeps them.
_NOT_NEIGHBOR = 1.0e06
# Cut-off of the contact neighborhood: every pair in contact, whatever the
# distance between the cell centers.
_ANY_CONTACT = 1.0e05

# The neighborhood counts, in the order their columns are written.
_METRICS = ("inclusive", "intermediate", "exclusive")


def _neigh_col(mode: str, kind: str, d: float) -> str:
    """Name of the neighbor-list column of a ``circle`` or ``contact``
    neighborhood of distance ``d``, before it is named after its populations."""
    return f"neighborhood_{'self' if mode == 'self' else '2'}_{kind}_{d}_px"


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _frames(
    timeline: np.ndarray, progress_callback: Optional[Callable[[int, int], None]]
) -> Iterator[int]:
    """Yields the frames of the timeline, and reports each one to
    ``progress_callback`` (frames done, number of frames) once processed."""
    for i, t in enumerate(tqdm(timeline)):
        yield t
        if progress_callback is not None:
            progress_callback(i + 1, len(timeline))


def _fill_neighborhood_at_t(
    time_index: int,
    setA: pd.DataFrame,
    setB: pd.DataFrame,
    dist_map: np.ndarray,
    cutoff: float,
    intersection_map: Optional[np.ndarray] = None,
    attention_weight: Optional[bool] = None,
    include_dead_weight: bool = False,
    symmetrize: bool = False,
    compute_cum_sum: bool = False,
    weights: Optional[np.ndarray] = None,
    closest_A: Optional[np.ndarray] = None,
    neigh_col: str = "",
    column_labelsA: Optional[Dict[str, str]] = None,
    column_labelsB: Optional[Dict[str, str]] = None,
    statusA: Optional[str] = None,
    statusB: Optional[str] = None,
    presence: Optional[Dict[Any, Dict[Any, Tuple[int, float]]]] = None,
) -> None:
    """
    Writes the neighbors of the cells of set A at one time point, for a
    distance or a contact neighborhood.

    Parameters
    ----------
    time_index : int
        The time point to process.
    setA : pandas.DataFrame
        Dataset for population A.
    setB : pandas.DataFrame
        Dataset for population B.
    dist_map : ndarray
        Distance matrix between the cells of setA and setB at ``time_index``.
    cutoff : float
        Cells of B at a distance up to ``cutoff`` are neighbors.
    intersection_map : ndarray, optional
        Overlap (in pixels) between the masks of A and B, for a contact
        neighborhood. When given, each neighbor also lists its
        ``intersection``.
    attention_weight : bool, optional
        Whether to compute attention weights.
    include_dead_weight : bool, optional
        Whether to include dead cells in weight computation.
    symmetrize : bool, optional
        Whether to update setB with neighborhood info from setA.
    compute_cum_sum : bool, optional
        Whether to compute cumulative presence.
    weights : ndarray, optional
        Precomputed attention weights.
    closest_A : ndarray, optional
        IDs of closest cells in setA.
    neigh_col : str
        Column name to store neighborhood data.
    column_labelsA : dict, optional
        Column mapping for set A.
    column_labelsB : dict, optional
        Column mapping for set B.
    statusA : str, optional
        Column name for status in set A.
    statusB : str, optional
        Column name for status in set B.
    presence : dict, optional
        For the cumulated presence: for each track of A, the number of
        frames each neighbor has been around and the sum of its weights, over
        the frames filled so far. The same dict is given for all the frames
        of a neighborhood, in time order, and is updated here.
    """

    def frame_data(df, labels, status):
        at_t = df[labels["time"]] == time_index
        # IDs and statuses are taken from one array, so that they share its dtype.
        columns = [labels["x"], labels["y"], labels["track"], status]
        data = df.loc[at_t, columns].to_numpy()
        return df.index[at_t], data[:, 2], data[:, -1]

    index_A, ids_A, status_A = frame_data(setA, column_labelsA, statusA)
    index_B, ids_B, status_B = frame_data(setB, column_labelsB, statusB)

    for k in range(dist_map.shape[0]):

        row = dist_map[k, :]
        sel = np.flatnonzero(row <= cutoff)
        # Rebuilt from a list, so that an object array gets a numeric dtype.
        neighs_B = np.array(ids_B[sel].tolist())
        status_neigh_B = np.array(status_B[sel].tolist())
        dist_B = [round(row[i], 2) for i in sel]
        intersect_B = None
        if intersection_map is not None:
            intersect_B = [round(intersection_map[k, i], 2) for i in sel]
        closest_B_cell = neighs_B[np.argmin(dist_B)] if len(sel) > 0 else None

        if symmetrize and attention_weight:
            if include_dead_weight:
                neigh_count = len(neighs_B)
            else:
                neigh_count = np.count_nonzero(status_neigh_B == 1)
            weight_A = 1.0 / neigh_count if neigh_count > 0 else np.nan
            if not include_dead_weight and status_A[k] == 0:
                weight_A = 0

        if compute_cum_sum:
            if len(sel) > 0 and column_labelsB["track"] != "TRACK_ID":
                raise ValueError(
                    "The set B does not seem to contain tracked data. The cumulative time will be meaningless."
                )
            past = presence.setdefault(ids_A[k], {})

        neighs = []
        for n, j in enumerate(sel):

            # Write the minimum info about neighboring cell B
            neigh_dico = {
                "id": neighs_B[n],
                "distance": dist_B[n],
                "status": status_neigh_B[n],
            }
            if intersect_B is not None:
                neigh_dico["intersection"] = intersect_B[n]
            if attention_weight:
                neigh_dico.update(
                    {"weight": weights[j], "closest": bool(closest_A[j] == ids_A[k])}
                )

            if compute_cum_sum:
                # Integrated presence of the neighboring cell B
                count, weight_sum = past.get(neighs_B[n], (0, 0.0))
                neigh_dico["cumulated_presence"] = count + 1
                if attention_weight:
                    neigh_dico["cumulated_presence_weighted"] = weight_sum + weights[j]

            if symmetrize:
                sym_dico = {
                    "id": ids_A[k],
                    "distance": dist_B[n],
                    "status": status_A[k],
                }
                if intersect_B is not None:
                    sym_dico["intersection"] = intersect_B[n]
                if attention_weight:
                    sym_dico.update(
                        {"weight": weight_A, "closest": bool(neighs_B[n] == closest_B_cell)}
                    )
                sym_neigh = setB.at[index_B[j], neigh_col]
                if not isinstance(sym_neigh, list):
                    sym_neigh = []
                sym_neigh.append(sym_dico)
                setB.at[index_B[j], neigh_col] = sym_neigh

            neighs.append(neigh_dico)

        setA.at[index_A[k], neigh_col] = neighs

        if compute_cum_sum:
            for neigh in neighs:
                count, weight_sum = past.get(neigh["id"], (0, 0.0))
                if attention_weight:
                    weight_sum = weight_sum + neigh["weight"]
                past[neigh["id"]] = (count + 1, weight_sum)


def _contact_site_mask(
    labelsA: np.ndarray,
    labelsB: Optional[np.ndarray],
    mask_id_A: int,
    mask_id_B: int,
    border: int = 3,
) -> np.ndarray:
    """
    Returns a binary mask of the pixels in cell A that lie within ``border``
    pixels of cell B's mask.

    This defines the contact zone from cell A's perspective: the portion of
    A's segmentation mask that faces B.  Used for contact-site intensity
    sampling.

    Parameters
    ----------
    labelsA : ndarray
        Label image for population A.
    labelsB : ndarray or None
        Label image for population B.  Pass ``None`` for a self-contact
        computation (both cells come from ``labelsA``).
    mask_id_A : int
        Label value of the reference cell in ``labelsA``.
    mask_id_B : int
        Label value of the contact neighbor.
    border : int, optional
        Dilation radius in pixels used to probe proximity.  Should match the
        ``distance`` parameter used when computing the contact neighborhood.
        Default is 3.

    Returns
    -------
    ndarray
        Boolean array of the same shape as ``labelsA``.  True pixels belong
        to cell A and are within ``border`` pixels of cell B.
    """
    lB = labelsA if labelsB is None else labelsB
    mask_B = lB == mask_id_B
    zone = np.zeros(labelsA.shape, dtype=bool)
    rows, cols = np.nonzero(mask_B)
    if rows.size == 0:
        return zone
    # Dilate only around B: no pixel further than the radius can be reached.
    radius = max(1, border)
    box = (
        slice(max(rows.min() - radius, 0), rows.max() + radius + 1),
        slice(max(cols.min() - radius, 0), cols.max() + radius + 1),
    )
    zone[box] = binary_dilation(mask_B[box], iterations=radius) & (
        labelsA[box] == mask_id_A
    )
    return zone


def _measure_contact_intensity_at_t(
    time_index: int,
    setA: pd.DataFrame,
    setA_t: pd.DataFrame,
    setB_t: pd.DataFrame,
    labelsA: np.ndarray,
    labelsB: Optional[np.ndarray],
    intensity_image: np.ndarray,
    channel_names: List[str],
    dist_map: np.ndarray,
    border: float,
    column_labelsA: Dict[str, str],
    column_labelsB: Dict[str, str],
) -> None:
    """
    Computes contact-site intensity statistics and writes them back to
    ``setA`` in-place.

    For each cell in ``setA_t``, the closest contact neighbor in ``setB_t``
    is identified from ``dist_map``.  The contact zone (pixels of cell A
    within ``border`` pixels of the neighbor) is extracted and, for each
    channel in ``channel_names``, the mean, max and std of the intensity
    are written as new columns:

    ``contact_{channel}_mean``, ``contact_{channel}_max``,
    ``contact_{channel}_std``.

    Cells with no contact neighbor, or whose contact zone is empty, receive
    NaN for all channel stats.

    Parameters
    ----------
    time_index : int
        Current frame index (used only for logging).
    setA : pd.DataFrame
        Full (all-timepoints) DataFrame for population A.  Results are
        written here.
    setA_t : pd.DataFrame
        Subset of ``setA`` at ``time_index``.
    setB_t : pd.DataFrame
        Subset of setB at ``time_index``.
    labelsA : ndarray
        Label image for population A at ``time_index``.
    labelsB : ndarray or None
        Label image for population B at ``time_index``.
    intensity_image : ndarray, shape (Y, X, C)
        Multi-channel intensity image at ``time_index``.
    channel_names : list of str
        Names of the C channels in ``intensity_image``.
    dist_map : ndarray, shape (nA, nB)
        Distance matrix; non-contact pairs have value ``_NOT_NEIGHBOR``.
    border : float
        Contact border size in pixels (passed to ``_contact_site_mask``).
    column_labelsA : dict
        Column label mapping for setA (keys: 'mask_id', etc.).
    column_labelsB : dict
        Column label mapping for setB (keys: 'mask_id', etc.).
    """
    mask_ids_A = setA_t[column_labelsA["mask_id"]].to_numpy()
    mask_ids_B = setB_t[column_labelsB["mask_id"]].to_numpy()
    border_int = max(1, int(round(border)))

    for k, (idx, mask_id_A) in enumerate(zip(setA_t.index, mask_ids_A)):
        row_dist = dist_map[k, :]
        if not np.any(row_dist < _ANY_CONTACT):
            continue

        mask_id_B = mask_ids_B[int(np.argmin(row_dist))]
        contact_zone = _contact_site_mask(
            labelsA, labelsB, int(mask_id_A), int(mask_id_B), border=border_int
        )
        if not np.any(contact_zone):
            continue

        for ch_idx, ch_name in enumerate(channel_names):
            ch_img = (
                intensity_image[..., ch_idx]
                if intensity_image.ndim == 3
                else intensity_image
            )
            pixels = ch_img[contact_zone].astype(float)
            setA.at[idx, f"contact_{ch_name}_mean"] = float(np.mean(pixels))
            setA.at[idx, f"contact_{ch_name}_max"] = float(np.max(pixels))
            setA.at[idx, f"contact_{ch_name}_std"] = float(np.std(pixels))


def _compute_mask_contact_dist_maps(
    setA: pd.DataFrame,
    setB: pd.DataFrame,
    labelsA: np.ndarray,
    labelsB: Optional[np.ndarray],
    distances: List[float],
    mode: str = "self",
    column_labelsA: Optional[Dict[str, str]] = None,
    column_labelsB: Optional[Dict[str, str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Computes, for each distance, a distance map based on mask contact between
    two sets of cells. What does not depend on the distance is computed once.

    Parameters
    ----------
    setA : pandas.DataFrame
        Cells of set A at one time point.
    setB : pandas.DataFrame
        Cells of set B at the same time point.
    labelsA : numpy.ndarray
        Labels for setA.
    labelsB : numpy.ndarray or None
        Labels for setB. None for a self neighborhood.
    distances : list of float
        Distances in pixels up to which masks are in contact.
    mode : str, optional
        Distance mode ("self", "inter"). Default is "self".
    column_labelsA : dict, optional
        Column labels for setA. Default is None.
    column_labelsB : dict, optional
        Column labels for setB. Default is None.

    Returns
    -------
    list of (dist_map, intersection_map), one per distance
        ``dist_map``, shape (nA, nB): distance between the cell centers of
        the pairs in contact, ``_NOT_NEIGHBOR`` for the other pairs.
        ``intersection_map``, shape (nA, nB): overlap of the masks of the
        pairs in contact, in pixels.
    """

    coordinates_A = setA.loc[:, [column_labelsA["x"], column_labelsA["y"]]].to_numpy()
    coordinates_B = setB.loc[:, [column_labelsB["x"], column_labelsB["y"]]].to_numpy()
    ids_A = setA.loc[:, column_labelsA["track"]].to_numpy()
    ids_B = setB.loc[:, column_labelsB["track"]].to_numpy()
    mask_ids_A = setA.loc[:, column_labelsA["mask_id"]].to_numpy()
    mask_ids_B = setB.loc[:, column_labelsB["mask_id"]].to_numpy()

    center_dist = cdist(coordinates_A, coordinates_B, metric="euclidean")

    # Only the masks of the cells in the tables can be in contact.
    labelsA = np.where(np.isin(labelsA, mask_ids_A), labelsA, 0.0)
    if labelsB is not None:
        labelsB = np.where(np.isin(labelsB, mask_ids_B), labelsB, 0.0)

    # Overlap, in pixels, of every pair of masks of A and B.
    overlaps = {}
    if labelsB is not None:
        both = (labelsA != 0) & (labelsB != 0)
        pairs, counts = np.unique(
            np.stack([labelsA[both], labelsB[both]], axis=1),
            axis=0,
            return_counts=True,
        )
        overlaps = {(int(a), int(b)): int(c) for (a, b), c in zip(pairs, counts)}

    # Row of the first cell with each mask label.
    row_A, row_B = {}, {}
    for i, m in enumerate(mask_ids_A):
        row_A.setdefault(m, i)
    for i, m in enumerate(mask_ids_B):
        row_B.setdefault(m, i)

    maps = []
    for distance in distances:
        contact_pairs = contact_neighborhood(
            labelsA, labelsB=labelsB, border=distance, connectivity=2
        )
        dist_map = center_dist.copy()
        intersection_map = np.zeros_like(dist_map, dtype=float)
        not_in_contact = np.ones_like(dist_map, dtype=bool)
        for mask_A, mask_B in np.abs(contact_pairs):
            idx_A = row_A.get(int(mask_A))
            idx_B = row_B.get(int(mask_B))
            if idx_A is None or idx_B is None:
                logger.debug(
                    f"Contact pair ({mask_A}, {mask_B}) not found in DataFrame mask IDs; skipping."
                )
                continue
            intersection = overlaps.get((int(mask_A), int(mask_B)), 0)
            not_in_contact[idx_A, idx_B] = False
            if mode == "self":
                not_in_contact[idx_B, idx_A] = False
            intersection_map[idx_A, idx_B] = intersection
            logger.debug(
                f"Ref cell #{ids_A[idx_A]} matched with neigh. cell #{ids_B[idx_B]}..."
            )
            logger.debug(f"Computed intersection: {intersection} px...")

        dist_map[not_in_contact] = _NOT_NEIGHBOR
        maps.append((dist_map, intersection_map))
    return maps


def _resolve_status(
    df: pd.DataFrame, column: Optional[str], invert: Optional[bool]
) -> str:
    """Prepares the status column of one set and returns the column to use.

    No column (or ``live_status``) makes every cell alive. Otherwise, when
    ``invert`` is given, status 2 (event already happened) becomes 1, and if
    ``invert`` is True the status is inverted into ``not_{column}``.
    """
    if column is None or column == "live_status":
        df.loc[:, "live_status"] = 1
        return "live_status"
    if invert is None or column not in df.columns:
        # A missing column is reported by `set_live_status`.
        return column
    df.loc[df[column] == 2, column] = 1  # already happened events become event
    if not invert:
        return column
    df.loc[:, "not_" + column] = [
        not a if a == 0 or a == 1 else np.nan for a in df.loc[:, column].values
    ]
    return "not_" + column


def set_live_status(
    setA: pd.DataFrame,
    setB: pd.DataFrame,
    status: Optional[List[str]],
    not_status_option: Optional[List[bool]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[List[str]]]:
    """
    Updates the live status for cells in two datasets based on specified status columns and options.

    This function assigns a live status to cells in two datasets (setA and setB) based on the provided
    status columns and options. If no status column is provided, all cells are marked as live. Otherwise,
    the function updates the datasets based on the status criteria, potentially inverting the status
    based on the `not_status_option`.

    Parameters
    ----------
    setA : pandas.DataFrame
            The first dataset containing trajectory or position information for cells.
    setB : pandas.DataFrame
            The second dataset containing trajectory or position information for cells.
    status : list or None
            A list containing the names of the columns in setA and setB that classify cells as alive (1) or dead (0).
            If None, all cells are considered alive. The list should contain exactly two elements.
    not_status_option : list
            A list containing boolean values indicating whether to invert the status for setA and setB, respectively.
            True means the status should be inverted; False means it should not.

    Returns
    -------
    tuple
            A tuple containing the updated setA and setB DataFrames, along with the final status column names
            used to classify cells in each set.

    """

    logger.debug(f"Provided statuses: {status}...")
    if status is None:
        status = [None, None]
    if isinstance(status, list):
        if len(status) != 2:
            raise ValueError("Please provide only two columns to classify cells as alive or dead.")
        # With a list of options, an option left to None does not invert.
        invert = (
            [bool(option) for option in not_status_option]
            if isinstance(not_status_option, list)
            else [None, None]
        )
        # A new list: the caller's stays as given, so that the same options
        # can be used again.
        status = [
            _resolve_status(setA, status[0], invert[0]),
            _resolve_status(setB, status[1], invert[1]),
        ]
        if status[0] not in setA.columns:
            raise KeyError(f"Status column '{status[0]}' not found in set A.")
        if status[1] not in setB.columns:
            raise KeyError(f"Status column '{status[1]}' not found in set B.")

    setA = setA.reset_index(drop=True)
    setB = setB.reset_index(drop=True)

    return setA, setB, status


_COUNT_PREFIXES = tuple(f"{m}_count_" for m in _METRICS) + ("mean_count_",)


def _population_column(column: str, population: List[str]) -> str:
    """Name a two-population neighborhood column after its populations:
    ``neighborhood_2_`` becomes ``neighborhood_(A-B)_``. Self neighborhoods
    keep their name."""
    if population[0] == population[1]:
        return column
    return column.replace(
        "neighborhood_2_", f"neighborhood_({population[0]}-{population[1]})_", 1
    )


def _rename_counts_to_populations(
    df: pd.DataFrame, population: List[str]
) -> pd.DataFrame:
    """Rename the count and mean columns of a two-population neighborhood
    after its populations."""
    counts = [c for c in df.columns if c.startswith(_COUNT_PREFIXES)]
    return df.rename(columns={c: _population_column(c, population) for c in counts})


def _drop_recomputed_columns(
    df: pd.DataFrame, kind: str, distance: List[float], population: List[str]
) -> pd.DataFrame:
    """Drop the neighbor, count and mean columns of the neighborhoods about
    to be computed, left by a previous run, so that computing one again
    replaces it instead of adding a duplicate column."""
    mode = "self" if population[0] == population[1] else "two-pop"
    names = [_population_column(_neigh_col(mode, kind, d), population) for d in distance]
    stale = [
        c
        for c in df.columns
        if any(c == n or c.endswith("_" + n) or f"_{n}_" in c for n in names)
    ]
    return df.drop(columns=stale)


def compute_attention_weight(
    dist_matrix: np.ndarray,
    cut_distance: float,
    opposite_cell_status: np.ndarray,
    opposite_cell_ids: np.ndarray,
    axis: int = 1,
    include_dead_weight: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the attention weight for each cell based on its proximity to cells of an opposite type within a specified distance.

    This function calculates the attention weight for cells by considering the distance to the cells of an opposite type
    within a given cutoff distance. It optionally considers only the 'live' opposite cells based on their status. The function
    returns two arrays: one containing the attention weights and another containing the IDs of the closest opposite cells.

    Parameters
    ----------
    dist_matrix : ndarray
            A 2D array representing the distance matrix between cells of two types.
    cut_distance : float
            The cutoff distance within which opposite cells will influence the attention weight.
    opposite_cell_status : ndarray
            An array indicating the status (e.g., live or dead) of each opposite cell. Only used when `include_dead_weight` is False.
    opposite_cell_ids : ndarray
            An array containing the IDs of the opposite cells.
    axis : int, optional
            The axis along which to compute the weights (default is 1). Axis 0 corresponds to rows, and axis 1 corresponds to columns.
    include_dead_weight : bool, optional
            If True, includes all opposite cells within the cutoff distance in the weight calculation, regardless of their status.
            If False, only considers opposite cells that are 'live' (default is True).

    Returns
    -------
    tuple of ndarrays
            A tuple containing two arrays: `weights` and `closest_opposite`. `weights` is an array of attention weights for each cell,
            and `closest_opposite` is an array of the IDs of the closest opposite cells within the cutoff distance.

    """

    # One column per cell, one row per opposite cell.
    dist = dist_matrix if axis == 1 else dist_matrix.T
    weights = np.full(dist.shape[1], np.nan)
    closest_opposite = np.full(dist.shape[1], np.nan)
    if dist.shape[1] == 0:
        return weights, closest_opposite

    within = dist <= cut_distance
    if include_dead_weight:
        nbr_opposite = within.sum(axis=0)
        closest_opposite[:] = opposite_cell_ids[np.argmin(dist, axis=0)]
    else:
        alive = opposite_cell_status == 1
        nbr_opposite = within[alive].sum(axis=0)
        if np.any(alive):
            closest_opposite[:] = opposite_cell_ids[alive][
                np.argmin(dist[alive], axis=0)
            ]

    has_opposite = nbr_opposite > 0
    weights[has_opposite] = 1.0 / nbr_opposite[has_opposite]
    return weights, closest_opposite


def distance_cut_neighborhood(
    setA: pd.DataFrame,
    setB: pd.DataFrame,
    distance: Union[float, List[float]],
    mode: str = "two-pop",
    status: Optional[List[str]] = None,
    not_status_option: Optional[List[bool]] = None,
    compute_cum_sum: bool = True,
    attention_weight: bool = True,
    symmetrize: bool = True,
    include_dead_weight: bool = True,
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Match neighbors in set A and B within a circle of radius d.

    Parameters
    ----------
    setA : pandas.DataFrame
        Trajectory or position set A.
    setB : pandas.DataFrame
        Trajectory or position set B.
    distance : float or list of float
            Cut-distance in pixels to match neighboring pairs.
    mode: str
            neighboring mode, between 'two-pop' (e.g. target-effector) and 'self' (target-target or effector-effector).
    status : list of str, optional
            Status columns of set A and set B classifying cells as alive (1) or dead (0); see `set_live_status` (default is None, all alive).
    not_status_option : list of bool, optional
            Whether to invert the status of set A and set B (default is None).
    compute_cum_sum : bool, optional
            Compute cumulated time of presence of neighbours (only if trajectories available for both sets) (default is True).
    attention_weight : bool, optional
            Compute the attention weight (how much a cell of set B is shared across cells of set A) (default is True).
    symmetrize : bool, optional
            Write in set B the neighborhood of set A (default is True).
    include_dead_weight : bool, optional
            Do not count dead cells when establishing attention weight (default is True).
    column_labels : dict, optional
            Dictionary specifying column names for 'track', 'time', 'x', and 'y'.
            Default is {'track': 'TRACK_ID', 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.
    progress_callback : callable, optional
            Called after each frame of each distance with the number of frames done and the number of frames.
    """

    if setA is None or setB is None:
        return None, None
    if mode not in ("two-pop", "self"):
        logger.error("Please provide a valid mode between `two-pop` and `self`...")
        return None, None
    setA, setB, status = set_live_status(setA, setB, status, not_status_option)

    for d in _as_list(distance):

        neigh_col = _neigh_col(mode, "circle", d)

        cl = []
        for s in [setA, setB]:
            # Check whether data can be tracked
            labels = {**column_labels, "track": extract_identity_col(s)}
            if labels["track"] == "ID":
                compute_cum_sum = False  # without tracks, no cumulated presence
            cl.append(labels)

            # Remove nan tracks (cells that do not belong to a track)
            s[neigh_col] = np.nan
            s[neigh_col] = s[neigh_col].astype(object)
            s.dropna(subset=[labels["track"]], inplace=True)

        # Loop over each available timestep
        timeline = np.unique(
            np.concatenate(
                [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
            )
        ).astype(int)
        presence = {}
        for t in _frames(timeline, progress_callback):

            at_t_A = setA[cl[0]["time"]] == t
            at_t_B = setB[cl[1]["time"]] == t
            coordinates_A = setA.loc[at_t_A, [cl[0]["x"], cl[0]["y"]]].to_numpy()
            coordinates_B = setB.loc[at_t_B, [cl[1]["x"], cl[1]["y"]]].to_numpy()
            if len(coordinates_A) == 0 or len(coordinates_B) == 0:
                continue

            dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")
            if mode == "self":
                np.fill_diagonal(dist_map, _NOT_NEIGHBOR)

            weights = None
            closest_A = None
            if attention_weight:
                weights, closest_A = compute_attention_weight(
                    dist_map,
                    d,
                    setA.loc[at_t_A, status[0]].to_numpy(),
                    setA.loc[at_t_A, cl[0]["track"]].to_numpy(),
                    axis=1,
                    include_dead_weight=include_dead_weight,
                )

            _fill_neighborhood_at_t(
                t,
                setA,
                setB,
                dist_map,
                d,
                attention_weight=attention_weight,
                include_dead_weight=include_dead_weight,
                symmetrize=symmetrize,
                compute_cum_sum=compute_cum_sum,
                weights=weights,
                closest_A=closest_A,
                neigh_col=neigh_col,
                column_labelsA=cl[0],
                column_labelsB=cl[1],
                statusA=status[0],
                statusB=status[1],
                presence=presence,
            )

    return setA, setB


def _merge_pickled_neighborhoods(
    df: pd.DataFrame, df_pkl: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Brings back into a table the neighborhood columns kept only in its pickle."""
    if df_pkl is None:
        return df
    id_col = extract_identity_col(df_pkl)
    cols = [c for c in df_pkl.columns if c.startswith("neighborhood")]
    cols += ["FRAME", id_col]
    logger.debug(f"Recover {cols} from the pickle file...")
    try:
        df = pd.merge(df, df_pkl.loc[:, cols], how="outer", on=[id_col, "FRAME"])
        logger.debug(f"Merged columns: {list(df.columns)}")
    except Exception as e:
        logger.warning(f"Failure to merge pickle and csv files: {e}")
    return df


def _load_position_tables(
    pos: str, population: List[str], clear_neigh: bool
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
    """Loads the tables of both populations with the neighborhoods saved in
    their pickles, or deletes the pickles and the pair table first if
    ``clear_neigh``. Returns None if a table is missing."""
    df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
    df_B, path_B = get_position_table(pos, population=population[1], return_path=True)
    if df_A is None or df_B is None:
        return None

    if clear_neigh:
        for path in (path_A, path_B):
            if os.path.exists(path.replace(".csv", ".pkl")):
                os.remove(path.replace(".csv", ".pkl"))
        df_pair, pair_path = get_position_table(
            pos, population="pairs", return_path=True
        )
        if df_pair is not None:
            os.remove(pair_path)

    df_A = _merge_pickled_neighborhoods(
        df_A, get_position_pickle(pos, population=population[0], return_path=False)
    )
    df_B = _merge_pickled_neighborhoods(
        df_B, get_position_pickle(pos, population=population[1], return_path=False)
    )
    if clear_neigh:
        # The counts and means of the CSVs go too.
        df_A = df_A.drop(columns=df_A.columns[df_A.columns.str.contains("neighborhood")])
        df_B = df_B.drop(columns=df_B.columns[df_B.columns.str.contains("neighborhood")])
    return df_A, df_B, path_A, path_B


def _save_position_tables(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    path_A: str,
    path_B: str,
    population: List[str],
    kind: str,
    distance: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Names the new neighborhoods of A after the populations and writes both
    tables: the pickles keep the neighbor lists, the CSVs do not. The
    neighbor table B of a two-population neighborhood does not keep the
    symmetrized lists."""
    two_pop = population[0] != population[1]
    if two_pop:
        for d in distance:
            neigh_col = _neigh_col("two-pop", kind, d)
            df_A = df_A.rename(columns={neigh_col: _population_column(neigh_col, population)})
            df_B = df_B.drop(columns=[neigh_col])
        df_B.to_pickle(path_B.replace(".csv", ".pkl"))

    df_A = _rename_counts_to_populations(df_A, population)
    df_A.to_pickle(path_A.replace(".csv", ".pkl"))

    def without_neighbor_lists(df):
        return df.drop(columns=df.columns[df.columns.str.startswith("neighborhood_")])

    without_neighbor_lists(df_A).to_csv(path_A, index=False)
    if two_pop:
        without_neighbor_lists(df_B).to_csv(path_B, index=False)
    return df_A, df_B


def _neighborhood_at_position(
    kind: str,
    pos: str,
    distance: Union[float, List[float]],
    population: Union[str, List[str]],
    theta_dist: Optional[Union[float, List[float]]],
    return_tables: bool,
    clear_neigh: bool,
    event_time_col: Optional[str],
    neighborhood_kwargs: Dict[str, Any],
    channel_names: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes the ``circle`` or ``contact`` neighborhoods of a position,
    with their counts and means, and writes the tables. See
    `compute_neighborhood_at_position`."""

    pos = pos.replace("\\", "/")
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"

    if isinstance(population, str):
        population = [population, population]
    distance = _as_list(distance)
    if theta_dist is not None and len(_as_list(theta_dist)) != len(distance):
        raise ValueError("Incompatible number of distances and number of edge thresholds.")

    mode = "self" if population[0] == population[1] else "two-pop"
    # A copy: neither the caller's options nor the shared default are changed.
    neighborhood_kwargs = {**neighborhood_kwargs, "mode": mode}

    tables = _load_position_tables(pos, population, clear_neigh)
    if tables is None:
        return None
    df_A, df_B, path_A, path_B = tables

    # Computing a neighborhood again replaces it.
    df_A = _drop_recomputed_columns(df_A, kind, distance, population)
    df_B = _drop_recomputed_columns(df_B, kind, distance, population)

    if kind == "circle":
        metrics = ["inclusive", "exclusive", "intermediate"]
        df_A, df_B = distance_cut_neighborhood(
            df_A,
            df_B,
            distance,
            progress_callback=progress_callback,
            **neighborhood_kwargs,
        )
    else:
        metrics = ["inclusive", "intermediate"]
        labelsA = locate_labels(pos, population=population[0])
        labelsB = None if mode == "self" else locate_labels(pos, population=population[1])

        # Intensity stack (T, Y, X, C) for the contact-site intensity, read
        # frame by frame when it can be memory-mapped.
        intensity_images = None
        if channel_names is not None:
            try:
                intensity_images = locate_stack_lazy(pos)
                if intensity_images is None:
                    intensity_images = locate_stack(pos)
            except Exception as e:
                logger.warning(
                    f"Could not load intensity stack for contact-site measurements: {e}. "
                    "Contact-site intensity columns will be absent."
                )

        logger.debug(f"Distance: {distance} for mask contact")
        df_A, df_B = mask_contact_neighborhood(
            df_A,
            df_B,
            labelsA,
            labelsB,
            distance,
            intensity_images=intensity_images,
            channel_names=channel_names,
            progress_callback=progress_callback,
            **neighborhood_kwargs,
        )

    if df_A is None or df_B is None or len(df_A) == 0:
        return None

    for d in distance:
        neigh_col = _neigh_col(mode, kind, d)
        if kind == "contact":
            df_A.loc[df_A["class_id"].isnull(), neigh_col] = np.nan

        logger.info("Count neighborhood...")
        df_A = compute_neighborhood_metrics(
            df_A, neigh_col, metrics=metrics, decompose_by_status=True
        )
        logger.info("Neighborhood metrics computed.")

        if "TRACK_ID" in df_A.columns and not np.all(df_A["TRACK_ID"].isnull()):
            logger.info("Estimate average neighborhood before/after event...")
            df_A = mean_neighborhood_before_event(
                df_A, neigh_col, event_time_col, metrics=metrics
            )
            if event_time_col is not None:
                df_A = mean_neighborhood_after_event(
                    df_A, neigh_col, event_time_col, metrics=metrics
                )
            logger.info("Average neighborhood estimation done.")

    df_A, df_B = _save_position_tables(
        df_A, df_B, path_A, path_B, population, kind, distance
    )

    if return_tables:
        return df_A, df_B


def compute_neighborhood_at_position(
    pos: str,
    distance: Union[float, List[float]],
    population: Union[str, List[str]] = ["targets", "effectors"],
    theta_dist: Optional[Union[float, List[float]]] = None,
    img_shape: Tuple[int, int] = (2048, 2048),
    return_tables: bool = False,
    clear_neigh: bool = False,
    event_time_col: Optional[str] = None,
    neighborhood_kwargs: Dict[str, Any] = {
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    },
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]]:
    """
    Computes neighborhood metrics for specified cell populations within a given position, based on distance criteria and additional parameters.

    This function assesses the neighborhood interactions between two specified cell populations (or within a single population) at a given position.
    It computes various neighborhood metrics based on specified distances.
    The results are optionally cleared of previous neighborhood calculations and can be returned as updated tables.

    Parameters
    ----------
    pos : str
            The path to the position directory where the analysis is to be performed.
    distance : float or list of float
            The distance(s) in pixels to define neighborhoods.
    population : list of str, optional
            Names of the cell populations to analyze. If a single population is provided, it is used for both populations in the analysis (default is ['targets', 'effectors']).
    theta_dist : float or list of float, optional
            Edge threshold(s) in pixels, one per distance. Edge filtering is currently disabled: the thresholds are only checked against the distances.
    img_shape : tuple of int, optional
            The dimensions (height, width) of the images in pixels (default is (2048, 2048)). Currently unused.
    return_tables : bool, optional
            If True, returns the updated data tables for both populations (default is False).
    clear_neigh : bool, optional
            If True, clears existing neighborhood columns from the data tables before computing new metrics (default is False).
    event_time_col : str, optional
            The column name indicating the event time for each cell, required if mean neighborhood metrics are to be computed before events.
    neighborhood_kwargs : dict, optional
            Additional keyword arguments for neighborhood computation, including mode, status options, and metrics (default includes mode 'two-pop', and symmetrization).
    progress_callback : callable, optional
            Called after each frame with the number of frames done and the number of frames.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
            If `return_tables` is True, returns the updated data tables for the specified populations. If only one population is analyzed, both returned data frames will be identical.

    Raises
    ------
    FileNotFoundError
            If the specified position path does not exist.
    ValueError
            If the number of distances and edge thresholds do not match.

    """
    return _neighborhood_at_position(
        "circle",
        pos,
        distance,
        population,
        theta_dist,
        return_tables,
        clear_neigh,
        event_time_col,
        neighborhood_kwargs,
        progress_callback=progress_callback,
    )


def _sort_by_track(neigh_table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Returns the table sorted by track and frame, and the columns that
    identify a track."""
    neigh_table = neigh_table.reset_index(drop=True)
    groupbycols = ["position"] if "position" in neigh_table.columns else []
    groupbycols.append(extract_identity_col(neigh_table))
    neigh_table.sort_values(by=groupbycols + ["FRAME"], inplace=True)
    return neigh_table, groupbycols


def compute_neighborhood_metrics(
    neigh_table: pd.DataFrame,
    neigh_col: str,
    metrics: List[str] = ["inclusive", "exclusive", "intermediate"],
    decompose_by_status: bool = False,
) -> pd.DataFrame:
    """
    Computes and appends neighborhood metrics to a dataframe based on specified neighborhood characteristics.

    This function iterates through a dataframe grouped by either 'TRACK_ID' or ['position', 'TRACK_ID'] (if 'position' column exists)
    and computes various neighborhood metrics (inclusive, exclusive, intermediate counts) for each cell. It can also decompose these
    metrics by cell status (e.g., live or dead) if specified.

    Parameters
    ----------
    neigh_table : pandas.DataFrame
            A dataframe containing neighborhood information for each cell, including position, track ID, frame, and a specified neighborhood column.
    neigh_col : str
            The column name in `neigh_table` that contains neighborhood information (e.g., a list of neighbors with their attributes).
    metrics : list of str, optional
            The metrics to be computed from the neighborhood information. Possible values include 'inclusive', 'exclusive', and 'intermediate'.
            Default is ['inclusive', 'exclusive', 'intermediate'].
    decompose_by_status : bool, optional
            If True, the metrics are computed separately for different statuses (e.g., live or dead) of the neighboring cells. Default is False.

    Returns
    -------
    pandas.DataFrame
            The input dataframe with additional columns for each of the specified metrics, and, if `decompose_by_status` is True, separate
            metrics for each status.

    Notes
    -----
    - 'inclusive' count refers to the total number of neighbors.
    - 'exclusive' count refers to the number of neighbors that are closest.
    - 'intermediate' count refers to the sum of weights attributed to neighbors, representing a weighted count.
    - If `decompose_by_status` is True, metrics are appended with '_s0' or '_s1' to indicate the status they correspond to.

    Examples
    --------
    >>> neigh_table = pd.DataFrame({
    ...     'TRACK_ID': [1, 1, 2, 2],
    ...     'FRAME': [1, 2, 1, 2],
    ...     'neighborhood_info': [{'weight': 1, 'status': 1, 'closest': 1}, ...]  # example neighborhood info
    ... })
    >>> neigh_col = 'neighborhood_info'
    >>> updated_neigh_table = compute_neighborhood_metrics(neigh_table, neigh_col, metrics=['inclusive'], decompose_by_status=True)
    # Computes the inclusive count of neighbors for each cell, decomposed by cell status.

    """

    neigh_table, groupbycols = _sort_by_track(neigh_table)
    metrics = [m for m in _METRICS if m in metrics]
    # (column suffix, status of the neighbors counted); None counts them all.
    statuses = [("", None)]
    if decompose_by_status:
        statuses += [("s0_", 0.0), ("s1_", 1.0)]
    # Columns in the order they are written: all neighbors, then by status.
    columns = [(m, "") for m in metrics]
    columns += [(m, s) for m in metrics for s, _ in statuses[1:]]

    for _, group in neigh_table.groupby(groupbycols):
        group = group.dropna(subset=[neigh_col])
        neighbors = group[neigh_col].to_numpy()

        counts = {
            (m, s): np.zeros(len(neighbors), dtype=int if (m, s) == ("inclusive", "") else float)
            for m, s in columns
        }
        for t, neighs_at_t in enumerate(neighbors):
            weights = np.array([n.get("weight", np.nan) for n in neighs_at_t], dtype=float)
            status = [n.get("status", np.nan) for n in neighs_at_t]
            closest = np.array([n.get("closest", False) == 1.0 for n in neighs_at_t], dtype=bool)

            for suffix, s in statuses:
                if s is None:
                    kept = np.ones(len(neighs_at_t), dtype=bool)
                else:
                    kept = np.array([x == s for x in status], dtype=bool)
                values = {
                    "inclusive": np.count_nonzero(kept),
                    "intermediate": np.sum(weights[kept]),
                    "exclusive": np.count_nonzero(closest & kept),
                }
                for m in metrics:
                    counts[(m, suffix)][t] = values[m]

        for m, s in columns:
            neigh_table.loc[group.index, f"{m}_count_{s}{neigh_col}"] = counts[(m, s)]

    return neigh_table


def _mean_neighborhood_around_event(
    neigh_table: pd.DataFrame,
    neigh_col: str,
    event_time_col: Optional[str],
    metrics: List[str],
    after: bool,
) -> pd.DataFrame:
    """Writes the mean of the neighborhood counts of each track up to its
    event (``after=False``) or after it (``after=True``). The counts of the
    live neighbors (``s1``) are averaged when they exist."""

    neigh_table, groupbycols = _sort_by_track(neigh_table)
    suffix = "_after_event" if after else "_before_event"

    if event_time_col is None:
        # Without an event, the mean up to it covers the whole observation
        # time, and there is nothing after it.
        if not after:
            logger.info(
                "No event time was provided... Estimating the mean neighborhood over the whole observation time..."
            )
        neigh_table.loc[:, "event_time_temp"] = (
            None if after else neigh_table["FRAME"].max()
        )
        event_time_col = "event_time_temp"
        suffix = ""

    for _, group in neigh_table.groupby(groupbycols):

        group = group.dropna(subset=[neigh_col])
        if len(group) == 0:
            continue

        event_time = group[event_time_col].to_numpy()[0]
        if after:
            if event_time is None or not event_time >= 0.0:
                continue
            in_window = group["FRAME"] > event_time
        else:
            if event_time < 0.0:
                event_time = group["FRAME"].max()
            in_window = group["FRAME"] <= event_time

        for metric in ("intermediate", "inclusive", "exclusive"):
            if metric not in metrics:
                continue
            column = f"{metric}_count_s1_{neigh_col}"
            if column not in group.columns:
                column = f"{metric}_count_{neigh_col}"
            counts = group.loc[in_window, column].to_numpy()
            if np.any(counts == counts):  # any count that is not NaN
                neigh_table.loc[
                    group.index, f"mean_count_{metric}_{neigh_col}{suffix}"
                ] = np.nanmean(counts)

    if event_time_col == "event_time_temp":
        neigh_table = neigh_table.drop(columns="event_time_temp")
    return neigh_table


def mean_neighborhood_before_event(
    neigh_table: pd.DataFrame,
    neigh_col: str,
    event_time_col: Optional[str],
    metrics: List[str] = ["inclusive", "exclusive", "intermediate"],
) -> pd.DataFrame:
    """
    Computes the mean neighborhood metrics for each cell track before a specified event time.

    This function calculates the mean values of specified neighborhood metrics (inclusive, exclusive, intermediate)
    for each cell track up to and including the frame of an event. The function requires the neighborhood metrics to
    have been previously computed and appended to the input dataframe. It operates on grouped data based on position
    and track ID, handling cases with or without position information.

    Parameters
    ----------
    neigh_table : pandas.DataFrame
            A dataframe containing cell track data with precomputed neighborhood metrics and event time information.
    neigh_col : str
            The base name of the neighborhood metric columns in `neigh_table`.
    event_time_col : str or None
            The column name indicating the event time for each cell track. If None, the maximum frame number in the
            dataframe is used as the event time for all tracks.
    metrics : list, optional
            List of metrics to compute. Default is ["inclusive", "exclusive", "intermediate"].

    Returns
    -------
    pandas.DataFrame
            The input dataframe with added columns for the mean neighborhood metrics before the event for each cell track.
            The new columns are named as 'mean_count_{metric}_{neigh_col}_before_event', where {metric} is one of
            'inclusive', 'exclusive', 'intermediate'.

    """
    return _mean_neighborhood_around_event(
        neigh_table, neigh_col, event_time_col, metrics, after=False
    )


def mean_neighborhood_after_event(
    neigh_table: pd.DataFrame,
    neigh_col: str,
    event_time_col: Optional[str],
    metrics: List[str] = ["inclusive", "exclusive", "intermediate"],
) -> pd.DataFrame:
    """
    Computes the mean neighborhood metrics for each cell track after a specified event time.

    This function calculates the mean values of specified neighborhood metrics (inclusive, exclusive, intermediate)
    for each cell track after the event time. The function requires the neighborhood metrics to
    have been previously computed and appended to the input dataframe. It operates on grouped data based on position
    and track ID, handling cases with or without position information.

    Parameters
    ----------
    neigh_table : pandas.DataFrame
            A dataframe containing cell track data with precomputed neighborhood metrics and event time information.
    neigh_col : str
            The base name of the neighborhood metric columns in `neigh_table`.
    event_time_col : str or None
            The column name indicating the event time for each cell track. If None, nothing is computed, as there is
            no event to be after.
    metrics : list, optional
            List of metrics to compute. Default is ["inclusive", "exclusive", "intermediate"].

    Returns
    -------
    pandas.DataFrame
            The input dataframe with added columns for the mean neighborhood metrics after the event for each cell track.
            The new columns are named as 'mean_count_{metric}_{neigh_col}_after_event', where {metric} is one of
            'inclusive', 'exclusive', 'intermediate'.

    """
    return _mean_neighborhood_around_event(
        neigh_table, neigh_col, event_time_col, metrics, after=True
    )


# Direct cell-cell contact neighborhood


def contact_neighborhood(
    labelsA: np.ndarray,
    labelsB: Optional[np.ndarray] = None,
    border: int = 3,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Identifies pairs of cells that are in contact or close proximity.

    Parameters
    ----------
    labelsA : ndarray
        Label image for the first population.
    labelsB : ndarray, optional
        Label image for the second population.
    border : int, optional
        Border expansion size to check for contact. Default is 3.
    connectivity : int, optional
        Connectivity for pixel graph. Default is 2.

    Returns
    -------
    ndarray, shape (n_pairs, 2)
        Unique pairs of contacting cell labels, each sorted. With ``labelsB``,
        the labels of A are negative.
    """

    labelsA = labelsA.astype(float)
    if labelsB is not None:
        labelsB = labelsB.astype(float)

    if border > 0:
        # Grow every mask by the border, so that masks this close touch.
        for labels in [labelsA] if labelsB is None else [labelsA, labelsB]:
            edge = contour_of_instance_segmentation(
                label=labels, distance=-border
            ).astype(float)
            on_edge = edge > 0
            labels[on_edge] = edge[on_edge]

    if labelsB is not None:
        # A labels negative, so that pairs across populations have opposite signs.
        labelsA[labelsA != 0] *= -1
        label_cases = [merge_labels(labelsA, labelsB), merge_labels(labelsB, labelsA)]
    else:
        label_cases = [labelsA]

    pairs = np.concatenate(
        [find_contact_neighbors(lbl, connectivity=connectivity) for lbl in label_cases]
    )
    pairs = np.sort(pairs, axis=1)
    if labelsB is not None:
        pairs = pairs[np.sign(pairs[:, 0]) != np.sign(pairs[:, 1])]

    return np.unique(pairs, axis=0)


def merge_labels(labelsA: np.ndarray, labelsB: np.ndarray) -> np.ndarray:
    """
    Merges two label images into one.

    Parameters
    ----------
    labelsA : ndarray
        First label image.
    labelsB : ndarray
        Second label image.

    Returns
    -------
    ndarray
        Merged label image (float) where non-zero pixels from labelsB overwrite labelsA.
    """
    return np.where(labelsB != 0, labelsB, labelsA).astype(float)


def find_contact_neighbors(labels: np.ndarray, connectivity: int = 2) -> np.ndarray:
    """
    Finds touching neighbors in a label image.

    Parameters
    ----------
    labels : ndarray
        Label image.
    connectivity : int, optional
        1 for the 4 nearest pixels, 2 to add the diagonals. Default is 2.

    Returns
    -------
    ndarray
        The labels of every pair of adjacent foreground pixels with different
        labels (touching masks), in both orders.
    """

    if labels.ndim != 2:
        raise ValueError("Wrong dimension for labels...")
    height, width = labels.shape
    pairs = []
    for dy, dx in product((-1, 0, 1), repeat=2):
        if (dy, dx) == (0, 0) or abs(dy) + abs(dx) > connectivity:
            continue
        # Each pixel and its neighbor at (dy, dx).
        center = labels[max(-dy, 0) : height - max(dy, 0), max(-dx, 0) : width - max(dx, 0)]
        neighbor = labels[max(dy, 0) : height + min(dy, 0), max(dx, 0) : width + min(dx, 0)]
        touching = (center != 0) & (neighbor != 0) & (center != neighbor)
        pairs.append(np.column_stack((center[touching], neighbor[touching])))
    return np.concatenate(pairs)


def mask_contact_neighborhood(
    setA: pd.DataFrame,
    setB: pd.DataFrame,
    labelsA: List[np.ndarray],
    labelsB: Optional[List[np.ndarray]],
    distance: Union[float, List[float]],
    mode: str = "two-pop",
    status: Optional[List[str]] = None,
    not_status_option: Optional[List[bool]] = None,
    compute_cum_sum: bool = True,
    attention_weight: bool = True,
    symmetrize: bool = True,
    include_dead_weight: bool = True,
    column_labels: Dict[str, str] = {
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
        "mask_id": "class_id",
    },
    intensity_images: Optional[List[np.ndarray]] = None,
    channel_names: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Match neighbors in set A and B whose masks are in contact, or closer than d pixels.

    Parameters
    ----------
    setA : pandas.DataFrame
        Trajectory or position set A.
    setB : pandas.DataFrame
        Trajectory or position set B.
    labelsA : list of ndarray
            List of label images for set A.
    labelsB : list of ndarray
            List of label images for set B. None for a self neighborhood.
    distance : float or list of float
            Distance in pixels up to which masks are in contact.
    mode : str, optional
            Neighboring mode, between 'two-pop' and 'self' (default is 'two-pop').
    status : list, optional
            Status columns of set A and set B; see `set_live_status` (default is None, all alive).
    not_status_option : list of bool, optional
            Whether to invert the status of set A and set B (default is None).
    compute_cum_sum : bool, optional
            Compute cumulated time of presence of neighbours (only if trajectories available for both sets) (default is True).
    attention_weight : bool, optional
            Compute the attention weight (how much a cell of set B is shared across cells of set A) (default is True).
    symmetrize : bool, optional
            Write in set B the neighborhood of set A (default is True).
    include_dead_weight : bool, optional
            Do not count dead cells when establishing attention weight (default is True).
    column_labels : dict, optional
            Dictionary specifying column names for 'track', 'time', 'x', 'y' and 'mask_id'.
            Default is {'track': 'TRACK_ID', 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y', 'mask_id': 'class_id'}.
    intensity_images : list of ndarray or array of shape (T, Y, X, C), optional
            Per-frame multi-channel intensity images with shape ``(Y, X, C)``
            (a lazy dask stack is read one frame at a time).
            When provided together with ``channel_names``, contact-site
            intensity statistics are computed for each cell and written as new
            columns ``contact_{channel}_mean/max/std``.  Default is None.
    channel_names : list of str, optional
            Names of the C channels in ``intensity_images``.  Must be provided
            when ``intensity_images`` is not None.  Default is None.
    progress_callback : callable, optional
            Called after each frame with the number of frames done and the number of frames.
    """

    if setA is None or setB is None:
        return None, None
    if mode not in ("two-pop", "self"):
        logger.error("Please provide a valid mode between `two-pop` and `self`...")
        return None, None
    setA, setB, status = set_live_status(setA, setB, status, not_status_option)

    measure_contact_intensity = (
        intensity_images is not None and channel_names is not None
    )

    cl = []
    for s in [setA, setB]:
        # Check whether data can be tracked
        labels = {**column_labels, "track": extract_identity_col(s)}
        if labels["track"] == "ID":
            compute_cum_sum = False
        cl.append(labels)

    setA = setA.loc[~setA[cl[0]["track"]].isnull(), :].copy()
    setB = setB.loc[~setB[cl[1]["track"]].isnull(), :].copy()

    if labelsB is None:
        labelsB = [None] * len(labelsA)

    distances = _as_list(distance)
    neigh_cols = [_neigh_col(mode, "contact", d) for d in distances]
    for neigh_col in neigh_cols:
        for s in [setA, setB]:
            s[neigh_col] = np.nan
            s[neigh_col] = s[neigh_col].astype(object)
        # The contact-site intensity columns come after the first neighborhood.
        if measure_contact_intensity:
            for ch_name in channel_names:
                for stat in ("mean", "max", "std"):
                    col = f"contact_{ch_name}_{stat}"
                    if col not in setA.columns:
                        setA[col] = np.nan

    presences = [{} for _ in distances]

    # Loop over each available timestep; the work on the masks that does not
    # depend on the distance is done once per frame.
    timeline = np.unique(
        np.concatenate(
            [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
        )
    ).astype(int)
    for t in _frames(timeline, progress_callback):

        setA_t = setA.loc[setA[cl[0]["time"]] == t, :]
        setB_t = setB.loc[setB[cl[1]["time"]] == t, :]
        if len(setA_t) == 0 or len(setB_t) == 0:
            continue

        maps = _compute_mask_contact_dist_maps(
            setA_t,
            setB_t,
            labelsA[t],
            labelsB[t],
            distances,
            mode=mode,
            column_labelsA=cl[0],
            column_labelsB=cl[1],
        )
        image_t = None
        if measure_contact_intensity and intensity_images[t] is not None:
            # Read once for all the distances (a lazy stack reads it here).
            image_t = np.asarray(intensity_images[t])
        for d, neigh_col, presence, (dist_map, intersection_map) in zip(
            distances, neigh_cols, presences, maps
        ):
            if mode == "self":
                np.fill_diagonal(dist_map, _NOT_NEIGHBOR)

            weights = None
            closest_A = None
            if attention_weight:
                weights, closest_A = compute_attention_weight(
                    dist_map,
                    _ANY_CONTACT,
                    setA_t[status[0]].to_numpy(),
                    setA_t[cl[0]["track"]].to_numpy(),
                    axis=1,
                    include_dead_weight=include_dead_weight,
                )

            _fill_neighborhood_at_t(
                t,
                setA,
                setB,
                dist_map,
                _ANY_CONTACT,
                intersection_map=intersection_map,
                attention_weight=attention_weight,
                include_dead_weight=include_dead_weight,
                symmetrize=symmetrize,
                compute_cum_sum=compute_cum_sum,
                weights=weights,
                closest_A=closest_A,
                neigh_col=neigh_col,
                column_labelsA=cl[0],
                column_labelsB=cl[1],
                statusA=status[0],
                statusB=status[1],
                presence=presence,
            )

            # With several distances, the last one in contact is kept.
            if image_t is not None:
                _measure_contact_intensity_at_t(
                    t,
                    setA,
                    setA_t,
                    setB_t,
                    labelsA[t],
                    labelsB[t],
                    image_t,
                    channel_names,
                    dist_map,
                    d,
                    cl[0],
                    cl[1],
                )

    return setA, setB


def compute_contact_neighborhood_at_position(
    pos: str,
    distance: Union[float, List[float]],
    population: List[str] = ["effectors", "targets"],
    theta_dist: Union[float, List[float]] = 20,
    img_shape: Tuple[int, int] = (2048, 2048),
    return_tables: bool = True,
    clear_neigh: bool = True,
    event_time_col: Optional[str] = None,
    channel_names: Optional[List[str]] = None,
    neighborhood_kwargs: Dict[str, Any] = {
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    },
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Optional[Tuple[pd.DataFrame, ...]]:
    """
    Computes contact neighborhood metrics for specified cell populations within a given position.

    This function assesses the contacts between the masks of two specified cell populations (or within a single population) at a given position,
    up to the specified distances. The results are optionally cleared of previous neighborhood calculations and can be returned as updated tables.

    Parameters
    ----------
    pos : str
            The path to the position directory where the analysis is to be performed.
    distance : float or list of float
            The distance(s) in pixels up to which masks are in contact.
    population : list of str, optional
            Names of the cell populations to analyze. If a single population is provided, it is used for both populations in the analysis (default is ['effectors', 'targets']).
    theta_dist : float or list of float, optional
            Edge threshold(s) in pixels, one per distance (default is 20). Edge filtering is currently disabled: the thresholds are only checked against the distances.
    img_shape : tuple of int, optional
            The dimensions (height, width) of the images in pixels (default is (2048, 2048)). Currently unused.
    return_tables : bool, optional
            If True, returns the updated data tables for both populations (default is True).
    clear_neigh : bool, optional
            If True, clears existing neighborhood columns from the data tables before computing new metrics (default is True).
    event_time_col : str, optional
            The column name indicating the event time for each cell, required if mean neighborhood metrics are to be computed before events.
    channel_names : list of str, optional
            Channels of the position's stack in which to measure the intensity at the contact site; see `mask_contact_neighborhood` (default is None).
    neighborhood_kwargs : dict, optional
            Additional keyword arguments for neighborhood computation, including mode, status options, and metrics (default includes mode 'two-pop', and symmetrization).
    progress_callback : callable, optional
            Called after each frame with the number of frames done and the number of frames.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
            If `return_tables` is True, returns the updated data tables for the specified populations. If only one population is analyzed, both returned data frames will be identical.

    Raises
    ------
    FileNotFoundError
            If the specified position path does not exist.
    ValueError
            If the number of distances and edge thresholds do not match.

    """
    return _neighborhood_at_position(
        "contact",
        pos,
        distance,
        population,
        theta_dist,
        return_tables,
        clear_neigh,
        event_time_col,
        neighborhood_kwargs,
        channel_names=channel_names,
        progress_callback=progress_callback,
    )


def extract_neighborhood_in_pair_table(
    df: pd.DataFrame,
    distance: Optional[int] = None,
    reference_population: str = "targets",
    neighbor_population: str = "effectors",
    mode: str = "circle",
    neighborhood_key: Optional[str] = None,
    contact_only: bool = True,
) -> pd.DataFrame:
    """
    Extracts data from a pair table that matches specific neighborhood criteria based on reference and neighbor
    populations, distance, and mode of neighborhood computation (e.g., circular or contact-based).

    Parameters
    ----------
    df : pandas.DataFrame
            DataFrame containing the pair table, which includes columns for 'reference_population', 'neighbor_population',
            and a column for neighborhood status.
    distance : int, optional
            Radius in pixels for neighborhood calculation, used only if `neighborhood_key` is not provided.
    reference_population : str, default="targets"
            The reference population to consider.
    neighbor_population : str, default="effectors"
            The neighbor population to consider, used only if `neighborhood_key` is not provided.
    mode : str, default="circle"
            Neighborhood computation mode. Options are "circle" for radius-based or "contact" for contact-based neighborhood, used only if `neighborhood_key` is not provided.
    neighborhood_key : str, optional
            A precomputed neighborhood key to identify specific neighborhoods. If provided, this key overrides `distance`,
            `mode`, and `neighbor_population`.
    contact_only : bool, default=True
            If True, only rows indicating contact with the neighbor population (status=1) are kept; if False, both
            contact (status=1) and no-contact (status=0) rows are included.

    Returns
    -------
    pandas.DataFrame
            Filtered DataFrame containing rows that meet the specified neighborhood criteria.

    Notes
    -----
    - When `neighborhood_key` is None, the neighborhood column is generated based on the provided `reference_population`,
      `neighbor_population`, `distance`, and `mode`.
    - The function uses `status_<neigh_col>` to filter rows based on `contact_only` criteria.

    Example
    -------
    >>> neighborhood_data = extract_neighborhood_in_pair_table(df, distance=50, reference_population="targets",
    ...                                                        neighbor_population="effectors", mode="circle")
    >>> neighborhood_data.head()

    Raises
    ------
    ValueError
            If `mode` is not 'circle' or 'contact'.
    KeyError
            If the required neighborhood status column does not exist in `df`.
    """

    if neighborhood_key is None:
        if mode not in ["circle", "contact"]:
            raise ValueError("Please set a valid neighborhood computation mode ('circle' or 'contact')")
        population = [reference_population, neighbor_population]
        pair_mode = "self" if population[0] == population[1] else "two-pop"
        neigh_col = _population_column(_neigh_col(pair_mode, mode, distance), population)
    else:
        neigh_col = neighborhood_key.replace("status_", "")
        if "_(" in neigh_col and ")_" in neigh_col:
            neighbor_population = (
                neigh_col.split("_(")[-1].split(")_")[0].split("-")[-1]
            )
        elif "self" in neigh_col:
            neighbor_population = reference_population
        elif reference_population == "effectors":
            neighbor_population = "targets"
        else:
            neighbor_population = "effectors"

    if "status_" + neigh_col not in df.columns:
        raise KeyError("The selected neighborhood does not appear in the data...")

    logger.debug(f"Neighborhood table preview:\n{df[['reference_population', 'neighbor_population', 'status_' + neigh_col]]}")

    s_keep = [1] if contact_only else [0, 1]
    return df.loc[
        (df["reference_population"] == reference_population)
        & (df["neighbor_population"] == neighbor_population)
        & (df["status_" + neigh_col].isin(s_keep))
    ]
