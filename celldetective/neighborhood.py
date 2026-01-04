import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.graph import pixel_graph
import os
from celldetective.utils.masks import contour_of_instance_segmentation
from celldetective.utils.data_cleaning import extract_identity_col
from scipy.spatial.distance import cdist
from celldetective.utils.image_loaders import locate_labels
from celldetective.utils.data_loaders import get_position_table, get_position_pickle

abs_path = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
)


def _fill_distance_neighborhood_at_t(
    time_index,
    setA,
    setB,
    dist_map,
    attention_weight=None,
    include_dead_weight=False,
    symmetrize=False,
    compute_cum_sum=False,
    weights=None,
    closest_A=None,
    neigh_col="",
    column_labelsA=None,
    column_labelsB=None,
    statusA=None,
    statusB=None,
    distance=10,
):

    index_A = list(setA.loc[setA[column_labelsA["time"]] == time_index].index)
    index_B = list(setB.loc[setB[column_labelsB["time"]] == time_index].index)

    dataA = setA.loc[
        setA[column_labelsA["time"]] == time_index,
        [column_labelsA["x"], column_labelsA["y"], column_labelsA["track"], statusA],
    ].to_numpy()

    ids_A = dataA[:, 2]
    status_A = dataA[:, 3]

    dataB = setB.loc[
        setB[column_labelsB["time"]] == time_index,
        [column_labelsB["x"], column_labelsB["y"], column_labelsB["track"], statusB],
    ].to_numpy()
    ids_B = dataB[:, 2]
    status_B = dataB[:, 3]

    for k in range(dist_map.shape[0]):

        col = dist_map[k, :]
        col[col == 0.0] = 1.0e06

        neighs_B = np.array([ids_B[i] for i in np.where((col <= distance))[0]])
        status_neigh_B = np.array([status_B[i] for i in np.where((col <= distance))[0]])
        dist_B = [round(col[i], 2) for i in np.where((col <= distance))[0]]
        if len(dist_B) > 0:
            closest_B_cell = neighs_B[np.argmin(dist_B)]

        if symmetrize and attention_weight:
            n_neighs = float(len(neighs_B))
            if not include_dead_weight:
                n_neighs_alive = len(np.where(status_neigh_B == 1)[0])
                neigh_count = n_neighs_alive
            else:
                neigh_count = n_neighs
            if neigh_count > 0:
                weight_A = 1.0 / neigh_count
            else:
                weight_A = np.nan

            if not include_dead_weight and status_A[k] == 0:
                weight_A = 0

        neighs = []
        setA.at[index_A[k], neigh_col] = []
        for n in range(len(neighs_B)):

            # index in setB
            n_index = np.where(ids_B == neighs_B[n])[0][0]
            # Assess if neigh B is closest to A
            if attention_weight:
                if closest_A[n_index] == ids_A[k]:
                    closest = True
                else:
                    closest = False

            if symmetrize:
                # Load neighborhood previous data
                sym_neigh = setB.loc[index_B[n_index], neigh_col]
                if neighs_B[n] == closest_B_cell:
                    closest_b = True
                else:
                    closest_b = False
                if isinstance(sym_neigh, list):
                    sym_neigh.append(
                        {"id": ids_A[k], "distance": dist_B[n], "status": status_A[k]}
                    )
                else:
                    sym_neigh = [
                        {"id": ids_A[k], "distance": dist_B[n], "status": status_A[k]}
                    ]
                if attention_weight:
                    sym_neigh[-1].update({"weight": weight_A, "closest": closest_b})

            # Write the minimum info about neighborhing cell B
            neigh_dico = {
                "id": neighs_B[n],
                "distance": dist_B[n],
                "status": status_neigh_B[n],
            }
            if attention_weight:
                neigh_dico.update({"weight": weights[n_index], "closest": closest})

            if compute_cum_sum:
                # Compute the integrated presence of the neighboring cell B
                assert (
                    column_labelsB["track"] == "TRACK_ID"
                ), "The set B does not seem to contain tracked data. The cumulative time will be meaningless."
                past_neighs = [
                    [ll["id"] for ll in l] if len(l) > 0 else [None]
                    for l in setA.loc[
                        (setA[column_labelsA["track"]] == ids_A[k])
                        & (setA[column_labelsA["time"]] <= time_index),
                        neigh_col,
                    ].to_numpy()
                ]
                past_neighs = [item for sublist in past_neighs for item in sublist]

                if attention_weight:
                    past_weights = [
                        [ll["weight"] for ll in l] if len(l) > 0 else [None]
                        for l in setA.loc[
                            (setA[column_labelsA["track"]] == ids_A[k])
                            & (setA[column_labelsA["time"]] <= time_index),
                            neigh_col,
                        ].to_numpy()
                    ]
                    past_weights = [
                        item for sublist in past_weights for item in sublist
                    ]

                cum_sum = len(np.where(past_neighs == neighs_B[n])[0])
                neigh_dico.update({"cumulated_presence": cum_sum + 1})

                if attention_weight:
                    cum_sum_weighted = np.sum(
                        [
                            w if l == neighs_B[n] else 0
                            for l, w in zip(past_neighs, past_weights)
                        ]
                    )
                    neigh_dico.update(
                        {
                            "cumulated_presence_weighted": cum_sum_weighted
                            + weights[n_index]
                        }
                    )

            if symmetrize:
                setB.at[index_B[n_index], neigh_col] = sym_neigh

            neighs.append(neigh_dico)

        setA.at[index_A[k], neigh_col] = neighs


def _fill_contact_neighborhood_at_t(
    time_index,
    setA,
    setB,
    dist_map,
    intersection_map=None,
    attention_weight=None,
    include_dead_weight=False,
    symmetrize=False,
    compute_cum_sum=False,
    weights=None,
    closest_A=None,
    neigh_col="",
    column_labelsA=None,
    column_labelsB=None,
    statusA=None,
    statusB=None,
    d_filter=10,
):

    index_A = list(setA.loc[setA[column_labelsA["time"]] == time_index].index)
    index_B = list(setB.loc[setB[column_labelsB["time"]] == time_index].index)

    dataA = setA.loc[
        setA[column_labelsA["time"]] == time_index,
        [
            column_labelsA["x"],
            column_labelsA["y"],
            column_labelsA["track"],
            column_labelsA["mask_id"],
            statusA,
        ],
    ].to_numpy()

    ids_A = dataA[:, 2]
    status_A = dataA[:, 4]

    dataB = setB.loc[
        setB[column_labelsB["time"]] == time_index,
        [
            column_labelsB["x"],
            column_labelsB["y"],
            column_labelsB["track"],
            column_labelsB["mask_id"],
            statusB,
        ],
    ].to_numpy()
    ids_B = dataB[:, 2]
    status_B = dataB[:, 4]

    for k in range(dist_map.shape[0]):

        col = dist_map[k, :]
        col_inter = intersection_map[k, :]
        col[col == 0.0] = 1.0e06

        neighs_B = np.array([ids_B[i] for i in np.where((col <= d_filter))[0]])
        status_neigh_B = np.array([status_B[i] for i in np.where((col <= d_filter))[0]])
        dist_B = [round(col[i], 2) for i in np.where((col <= d_filter))[0]]
        intersect_B = [round(col_inter[i], 2) for i in np.where((col <= d_filter))[0]]

        if len(dist_B) > 0:
            closest_B_cell = neighs_B[np.argmin(dist_B)]

        if symmetrize and attention_weight:
            n_neighs = float(len(neighs_B))
            if not include_dead_weight:
                n_neighs_alive = len(np.where(status_neigh_B == 1)[0])
                neigh_count = n_neighs_alive
            else:
                neigh_count = n_neighs
            if neigh_count > 0:
                weight_A = 1.0 / neigh_count
            else:
                weight_A = np.nan

            if not include_dead_weight and status_A[k] == 0:
                weight_A = 0

        neighs = []
        setA.at[index_A[k], neigh_col] = []
        for n in range(len(neighs_B)):

            # index in setB
            n_index = np.where(ids_B == neighs_B[n])[0][0]
            # Assess if neigh B is closest to A
            if attention_weight:
                if closest_A[n_index] == ids_A[k]:
                    closest = True
                else:
                    closest = False

            if symmetrize:
                # Load neighborhood previous data
                sym_neigh = setB.loc[index_B[n_index], neigh_col]
                if neighs_B[n] == closest_B_cell:
                    closest_b = True
                else:
                    closest_b = False
                if isinstance(sym_neigh, list):
                    sym_neigh.append(
                        {
                            "id": ids_A[k],
                            "distance": dist_B[n],
                            "status": status_A[k],
                            "intersection": intersect_B[n],
                        }
                    )
                else:
                    sym_neigh = [
                        {
                            "id": ids_A[k],
                            "distance": dist_B[n],
                            "status": status_A[k],
                            "intersection": intersect_B[n],
                        }
                    ]
                if attention_weight:
                    sym_neigh[-1].update({"weight": weight_A, "closest": closest_b})

            # Write the minimum info about neighborhing cell B
            neigh_dico = {
                "id": neighs_B[n],
                "distance": dist_B[n],
                "status": status_neigh_B[n],
                "intersection": intersect_B[n],
            }
            if attention_weight:
                neigh_dico.update({"weight": weights[n_index], "closest": closest})

            if compute_cum_sum:
                # Compute the integrated presence of the neighboring cell B
                assert (
                    column_labelsB["track"] == "TRACK_ID"
                ), "The set B does not seem to contain tracked data. The cumulative time will be meaningless."
                past_neighs = [
                    [ll["id"] for ll in l] if len(l) > 0 else [None]
                    for l in setA.loc[
                        (setA[column_labelsA["track"]] == ids_A[k])
                        & (setA[column_labelsA["time"]] <= time_index),
                        neigh_col,
                    ].to_numpy()
                ]
                past_neighs = [item for sublist in past_neighs for item in sublist]

                if attention_weight:
                    past_weights = [
                        [ll["weight"] for ll in l] if len(l) > 0 else [None]
                        for l in setA.loc[
                            (setA[column_labelsA["track"]] == ids_A[k])
                            & (setA[column_labelsA["time"]] <= time_index),
                            neigh_col,
                        ].to_numpy()
                    ]
                    past_weights = [
                        item for sublist in past_weights for item in sublist
                    ]

                cum_sum = len(np.where(past_neighs == neighs_B[n])[0])
                neigh_dico.update({"cumulated_presence": cum_sum + 1})

                if attention_weight:
                    cum_sum_weighted = np.sum(
                        [
                            w if l == neighs_B[n] else 0
                            for l, w in zip(past_neighs, past_weights)
                        ]
                    )
                    neigh_dico.update(
                        {
                            "cumulated_presence_weighted": cum_sum_weighted
                            + weights[n_index]
                        }
                    )

            if symmetrize:
                setB.at[index_B[n_index], neigh_col] = sym_neigh

            neighs.append(neigh_dico)

        setA.at[index_A[k], neigh_col] = neighs


def _compute_mask_contact_dist_map(
    setA,
    setB,
    labelsA,
    labelsB=None,
    distance=10,
    mode="self",
    column_labelsA=None,
    column_labelsB=None,
):

    coordinates_A = setA.loc[:, [column_labelsA["x"], column_labelsA["y"]]].to_numpy()
    coordinates_B = setB.loc[:, [column_labelsB["x"], column_labelsB["y"]]].to_numpy()
    ids_A = setA.loc[:, column_labelsA["track"]].to_numpy()
    ids_B = setB.loc[:, column_labelsB["track"]].to_numpy()
    mask_ids_A = setA.loc[:, column_labelsA["mask_id"]].to_numpy()
    mask_ids_B = setB.loc[:, column_labelsB["mask_id"]].to_numpy()

    # compute distance matrix
    dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")
    intersection_map = np.zeros_like(dist_map).astype(float)

    # Do the mask contact computation
    labelsA = np.where(np.isin(labelsA, mask_ids_A), labelsA.copy(), 0.0)

    if labelsB is not None:
        labelsB = np.where(np.isin(labelsB, mask_ids_B), labelsB.copy(), 0.0)

    contact_pairs = contact_neighborhood(
        labelsA, labelsB=labelsB, border=distance, connectivity=2
    )

    # Put infinite distance to all non-contact pairs (something like this)
    flatA = labelsA.flatten()
    if labelsB is not None:
        flatB = labelsB.flatten()

    if len(contact_pairs) > 0:
        mask = np.ones_like(dist_map).astype(bool)

        indices_to_keep = []
        for cp in contact_pairs:

            cp = np.abs(cp)
            mask_A, mask_B = cp
            idx_A = np.where(mask_ids_A == int(mask_A))[0][0]
            idx_B = np.where(mask_ids_B == int(mask_B))[0][0]

            intersection = 0
            if labelsB is not None:
                intersection = len(
                    flatA[(flatA == int(mask_A)) & (flatB == int(mask_B))]
                )

            indices_to_keep.append([idx_A, idx_B, intersection])
            print(
                f"Ref cell #{ids_A[idx_A]} matched with neigh. cell #{ids_B[idx_B]}..."
            )
            print(f"Computed intersection: {intersection} px...")

        if len(indices_to_keep) > 0:
            indices_to_keep = np.array(indices_to_keep)
            mask[indices_to_keep[:, 0], indices_to_keep[:, 1]] = False
            if mode == "self":
                mask[indices_to_keep[:, 1], indices_to_keep[:, 0]] = False
            dist_map[mask] = 1.0e06
            intersection_map[indices_to_keep[:, 0], indices_to_keep[:, 1]] = (
                indices_to_keep[:, 2]
            )
        else:
            dist_map[:, :] = 1.0e06
    else:
        dist_map[:, :] = 1.0e06

    return dist_map, intersection_map


def set_live_status(setA, setB, status, not_status_option):
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

    print(f"Provided statuses: {status}...")
    if (
        status is None
        or status == ["live_status", "live_status"]
        or status == [None, None]
    ):
        setA.loc[:, "live_status"] = 1
        setB.loc[:, "live_status"] = 1
        status = ["live_status", "live_status"]
    elif isinstance(status, list):
        assert (
            len(status) == 2
        ), "Please provide only two columns to classify cells as alive or dead."
        if status[0] is None or status[0] == "live_status":
            setA.loc[:, "live_status"] = 1
            status[0] = "live_status"
        elif status[0] is not None and isinstance(not_status_option, list):
            setA.loc[setA[status[0]] == 2, status[0]] = (
                1  # already happened events become event
            )
            if not_status_option[0]:
                setA.loc[:, "not_" + status[0]] = [
                    not a if a == 0 or a == 1 else np.nan
                    for a in setA.loc[:, status[0]].values
                ]
                status[0] = "not_" + status[0]
        if status[1] is None or status[1] == "live_status":
            setB.loc[:, "live_status"] = 1
            status[1] = "live_status"
        elif status[1] is not None and isinstance(not_status_option, list):
            setB.loc[setB[status[1]] == 2, status[1]] = (
                1  # already happened events become event
            )
            if not_status_option[1]:
                setB.loc[:, "not_" + status[1]] = [
                    not a if a == 0 or a == 1 else np.nan
                    for a in setB.loc[:, status[1]].values
                ]
                status[1] = "not_" + status[1]

        assert status[0] in list(setA.columns)
        assert status[1] in list(setB.columns)

    setA = setA.reset_index(drop=True)
    setB = setB.reset_index(drop=True)

    return setA, setB, status


def compute_attention_weight(
    dist_matrix,
    cut_distance,
    opposite_cell_status,
    opposite_cell_ids,
    axis=1,
    include_dead_weight=True,
):
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

    weights = np.empty(dist_matrix.shape[axis])
    closest_opposite = np.empty(dist_matrix.shape[axis])

    for i in range(dist_matrix.shape[axis]):
        if axis == 1:
            row = dist_matrix[:, i]
        elif axis == 0:
            row = dist_matrix[i, :]
        row[row == 0.0] = 1.0e06
        nbr_opposite = len(row[row <= cut_distance])

        if not include_dead_weight:
            stat = opposite_cell_status[np.where(row <= cut_distance)[0]]
            nbr_opposite = len(stat[stat == 1])
            index_subpop = np.argmin(row[opposite_cell_status == 1])
            closest_opposite[i] = opposite_cell_ids[opposite_cell_status == 1][
                index_subpop
            ]
        else:
            closest_opposite[i] = opposite_cell_ids[np.argmin(row)]

        if nbr_opposite > 0:
            weight = 1.0 / float(nbr_opposite)
            weights[i] = weight

    return weights, closest_opposite


def distance_cut_neighborhood(
    setA,
    setB,
    distance,
    mode="two-pop",
    status=None,
    not_status_option=None,
    compute_cum_sum=True,
    attention_weight=True,
    symmetrize=True,
    include_dead_weight=True,
    column_labels={
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
    },
):
    """

    Match neighbors in set A and B within a circle of radius d.

    Parameters
    ----------
    setA,setB : pandas DataFrame
            Trajectory or position sets A and B.
    distance : float
            Cut-distance in pixels to match neighboring pairs.
    mode: str
            neighboring mode, between 'two-pop' (e.g. target-effector) and 'self' (target-target or effector-effector).
    status: None or status
            name to look for cells to ignore (because they are dead). By default all cells are kept.
    compute_cum_sum: bool,
            compute cumulated time of presence of neighbours (only if trajectories available for both sets)
    attention_weight: bool,
            compute the attention weight (how much a cell of set B is shared across cells of set A)
    symmetrize: bool,
            write in set B the neighborhood of set A
    include_dead_weight: bool
            do not count dead cells when establishing attention weight
    """

    # Check live_status option
    if setA is not None and setB is not None:
        setA, setB, status = set_live_status(setA, setB, status, not_status_option)
    else:
        return None, None

    # Check distance option
    if not isinstance(distance, list):
        distance = [distance]

    for d in distance:
        # loop over each provided distance

        if mode == "two-pop":
            neigh_col = f"neighborhood_2_circle_{d}_px"
        elif mode == "self":
            neigh_col = f"neighborhood_self_circle_{d}_px"

        cl = []
        for s in [setA, setB]:

            # Check whether data can be tracked
            temp_column_labels = column_labels.copy()

            id_col = extract_identity_col(s)
            temp_column_labels.update({"track": id_col})
            if id_col == "ID":
                compute_cum_sum = (
                    False  # if no tracking data then cum_sum is not relevant
                )
            cl.append(temp_column_labels)

            # Remove nan tracks (cells that do not belong to a track)
            s[neigh_col] = np.nan
            s[neigh_col] = s[neigh_col].astype(object)
            s.dropna(subset=[cl[-1]["track"]], inplace=True)

        # Loop over each available timestep
        timeline = np.unique(
            np.concatenate(
                [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
            )
        ).astype(int)
        for t in tqdm(timeline):

            coordinates_A = setA.loc[
                setA[cl[0]["time"]] == t, [cl[0]["x"], cl[0]["y"]]
            ].to_numpy()
            ids_A = setA.loc[setA[cl[0]["time"]] == t, cl[0]["track"]].to_numpy()
            status_A = setA.loc[setA[cl[0]["time"]] == t, status[0]].to_numpy()

            coordinates_B = setB.loc[
                setB[cl[1]["time"]] == t, [cl[1]["x"], cl[1]["y"]]
            ].to_numpy()
            ids_B = setB.loc[setB[cl[1]["time"]] == t, cl[1]["track"]].to_numpy()

            if len(ids_A) > 0 and len(ids_B) > 0:

                # compute distance matrix
                dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")

                if attention_weight:
                    weights, closest_A = compute_attention_weight(
                        dist_map,
                        d,
                        status_A,
                        ids_A,
                        axis=1,
                        include_dead_weight=include_dead_weight,
                    )

                _fill_distance_neighborhood_at_t(
                    t,
                    setA,
                    setB,
                    dist_map,
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
                    distance=d,
                )

    return setA, setB


def compute_neighborhood_at_position(
    pos,
    distance,
    population=["targets", "effectors"],
    theta_dist=None,
    img_shape=(2048, 2048),
    return_tables=False,
    clear_neigh=False,
    event_time_col=None,
    neighborhood_kwargs={
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    },
):
    """
    Computes neighborhood metrics for specified cell populations within a given position, based on distance criteria and additional parameters.

    This function assesses the neighborhood interactions between two specified cell populations (or within a single population) at a given position.
    It computes various neighborhood metrics based on specified distances, considering the entire image or excluding edge regions.
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
            Edge threshold(s) in pixels to exclude cells close to the image boundaries from the analysis. If not provided, defaults to 90% of each specified distance.
    img_shape : tuple of int, optional
            The dimensions (height, width) of the images in pixels (default is (2048, 2048)).
    return_tables : bool, optional
            If True, returns the updated data tables for both populations (default is False).
    clear_neigh : bool, optional
            If True, clears existing neighborhood columns from the data tables before computing new metrics (default is False).
    event_time_col : str, optional
            The column name indicating the event time for each cell, required if mean neighborhood metrics are to be computed before events.
    neighborhood_kwargs : dict, optional
            Additional keyword arguments for neighborhood computation, including mode, status options, and metrics (default includes mode 'two-pop', and symmetrization).

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
            If `return_tables` is True, returns the updated data tables for the specified populations. If only one population is analyzed, both returned data frames will be identical.

    Raises
    ------
    AssertionError
            If the specified position path does not exist or if the number of distances and edge thresholds do not match.

    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    assert os.path.exists(pos), f"Position {pos} is not a valid path."

    if isinstance(population, str):
        population = [population, population]

    if not isinstance(distance, list):
        distance = [distance]
    if not theta_dist is None and not isinstance(theta_dist, list):
        theta_dist = [theta_dist]

    if theta_dist is None:
        theta_dist = [0.9 * d for d in distance]
    assert len(theta_dist) == len(
        distance
    ), "Incompatible number of distances and number of edge thresholds."

    if population[0] == population[1]:
        neighborhood_kwargs.update({"mode": "self"})
    if population[1] != population[0]:
        neighborhood_kwargs.update({"mode": "two-pop"})

    df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
    df_B, path_B = get_position_table(pos, population=population[1], return_path=True)
    if df_A is None or df_B is None:
        return None

    if clear_neigh:
        if os.path.exists(path_A.replace(".csv", ".pkl")):
            os.remove(path_A.replace(".csv", ".pkl"))
        if os.path.exists(path_B.replace(".csv", ".pkl")):
            os.remove(path_B.replace(".csv", ".pkl"))
        df_pair, pair_path = get_position_table(
            pos, population="pairs", return_path=True
        )
        if df_pair is not None:
            os.remove(pair_path)

    df_A_pkl = get_position_pickle(pos, population=population[0], return_path=False)
    df_B_pkl = get_position_pickle(pos, population=population[1], return_path=False)

    if df_A_pkl is not None:
        pkl_columns = np.array(df_A_pkl.columns)
        neigh_columns = np.array([c.startswith("neighborhood") for c in pkl_columns])
        cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

        id_col = extract_identity_col(df_A_pkl)
        cols.append(id_col)
        on_cols = [id_col, "FRAME"]

        print(f"Recover {cols} from the pickle file...")
        try:
            df_A = pd.merge(df_A, df_A_pkl.loc[:, cols], how="outer", on=on_cols)
            print(df_A.columns)
        except Exception as e:
            print(f"Failure to merge pickle and csv files: {e}")

    if df_B_pkl is not None and df_B is not None:
        pkl_columns = np.array(df_B_pkl.columns)
        neigh_columns = np.array([c.startswith("neighborhood") for c in pkl_columns])
        cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

        id_col = extract_identity_col(df_B_pkl)
        cols.append(id_col)
        on_cols = [id_col, "FRAME"]

        print(f"Recover {cols} from the pickle file...")
        try:
            df_B = pd.merge(df_B, df_B_pkl.loc[:, cols], how="outer", on=on_cols)
        except Exception as e:
            print(f"Failure to merge pickle and csv files: {e}")

    if clear_neigh:
        unwanted = df_A.columns[df_A.columns.str.contains("neighborhood")]
        df_A = df_A.drop(columns=unwanted)
        unwanted = df_B.columns[df_B.columns.str.contains("neighborhood")]
        df_B = df_B.drop(columns=unwanted)

    df_A, df_B = distance_cut_neighborhood(df_A, df_B, distance, **neighborhood_kwargs)

    if df_A is None or df_B is None or len(df_A) == 0:
        return None

    for td, d in zip(theta_dist, distance):

        if neighborhood_kwargs["mode"] == "two-pop":
            neigh_col = f"neighborhood_2_circle_{d}_px"

        elif neighborhood_kwargs["mode"] == "self":
            neigh_col = f"neighborhood_self_circle_{d}_px"

        # edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
        # edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
        # df_A.loc[~edge_filter_A, neigh_col] = np.nan
        # df_B.loc[~edge_filter_B, neigh_col] = np.nan

        print("Count neighborhood...")
        df_A = compute_neighborhood_metrics(
            df_A,
            neigh_col,
            metrics=["inclusive", "exclusive", "intermediate"],
            decompose_by_status=True,
        )
        # if neighborhood_kwargs['symmetrize']:
        # 	df_B = compute_neighborhood_metrics(df_B, neigh_col, metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
        print("Done...")

        if "TRACK_ID" in list(df_A.columns):
            if not np.all(df_A["TRACK_ID"].isnull()):
                print("Estimate average neighborhood before/after event...")
                df_A = mean_neighborhood_before_event(df_A, neigh_col, event_time_col)
                if event_time_col is not None:
                    df_A = mean_neighborhood_after_event(
                        df_A, neigh_col, event_time_col
                    )
                print("Done...")

    if not population[0] == population[1]:
        # Remove neighborhood column from neighbor table, rename with actual population name
        for td, d in zip(theta_dist, distance):
            if neighborhood_kwargs["mode"] == "two-pop":
                neigh_col = f"neighborhood_2_circle_{d}_px"
                new_neigh_col = neigh_col.replace(
                    "_2_", f"_({population[0]}-{population[1]})_"
                )
                df_A = df_A.rename(columns={neigh_col: new_neigh_col})
            elif neighborhood_kwargs["mode"] == "self":
                neigh_col = f"neighborhood_self_circle_{d}_px"
            df_B = df_B.drop(columns=[neigh_col])
        df_B.to_pickle(path_B.replace(".csv", ".pkl"))

    cols_to_rename = [
        c
        for c in list(df_A.columns)
        if c.startswith("intermediate_count_")
        or c.startswith("inclusive_count_")
        or c.startswith("exclusive_count_")
        or c.startswith("mean_count_")
    ]
    new_col_names = [
        c.replace("_2_", f"_({population[0]}-{population[1]})_") for c in cols_to_rename
    ]
    new_name_map = {}
    for k, c in enumerate(cols_to_rename):
        new_name_map.update({c: new_col_names[k]})
    df_A = df_A.rename(columns=new_name_map)

    df_A.to_pickle(path_A.replace(".csv", ".pkl"))

    unwanted = df_A.columns[df_A.columns.str.startswith("neighborhood_")]
    df_A2 = df_A.drop(columns=unwanted)
    df_A2.to_csv(path_A, index=False)

    if not population[0] == population[1]:
        unwanted = df_B.columns[df_B.columns.str.startswith("neighborhood_")]
        df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
        df_B_csv.to_csv(path_B, index=False)

    if return_tables:
        return df_A, df_B


def compute_neighborhood_metrics(
    neigh_table,
    neigh_col,
    metrics=["inclusive", "exclusive", "intermediate"],
    decompose_by_status=False,
):
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

    neigh_table = neigh_table.reset_index(drop=True)
    if "position" in list(neigh_table.columns):
        groupbycols = ["position"]
    else:
        groupbycols = []

    id_col = extract_identity_col(neigh_table)
    groupbycols.append(id_col)

    neigh_table.sort_values(by=groupbycols + ["FRAME"], inplace=True)

    for tid, group in neigh_table.groupby(groupbycols):
        group = group.dropna(subset=neigh_col)
        indices = list(group.index)
        neighbors = group[neigh_col].to_numpy()

        if "inclusive" in metrics:
            n_inclusive = [len(n) for n in neighbors]

        if "intermediate" in metrics:
            n_intermediate = np.zeros(len(neighbors))
            n_intermediate[:] = np.nan

        if "exclusive" in metrics:
            n_exclusive = np.zeros(len(neighbors))
            n_exclusive[:] = np.nan

        if decompose_by_status:

            if "inclusive" in metrics:
                n_inclusive_status_0 = np.zeros(len(neighbors))
                n_inclusive_status_0[:] = np.nan
                n_inclusive_status_1 = np.zeros(len(neighbors))
                n_inclusive_status_1[:] = np.nan

            if "intermediate" in metrics:
                n_intermediate_status_0 = np.zeros(len(neighbors))
                n_intermediate_status_0[:] = np.nan
                n_intermediate_status_1 = np.zeros(len(neighbors))
                n_intermediate_status_1[:] = np.nan

            if "exclusive" in metrics:
                n_exclusive_status_0 = np.zeros(len(neighbors))
                n_exclusive_status_0[:] = np.nan
                n_exclusive_status_1 = np.zeros(len(neighbors))
                n_exclusive_status_1[:] = np.nan

        for t in range(len(neighbors)):

            neighs_at_t = neighbors[t]
            weights_at_t = [n["weight"] for n in neighs_at_t]
            status_at_t = [n["status"] for n in neighs_at_t]
            closest_at_t = [n["closest"] for n in neighs_at_t]

            if "intermediate" in metrics:
                n_intermediate[t] = np.sum(weights_at_t)
            if "exclusive" in metrics:
                n_exclusive[t] = sum([c == 1.0 for c in closest_at_t])

            if decompose_by_status:

                if "inclusive" in metrics:
                    n_inclusive_status_0[t] = sum([s == 0.0 for s in status_at_t])
                    n_inclusive_status_1[t] = sum([s == 1.0 for s in status_at_t])

                if "intermediate" in metrics:
                    weights_at_t = np.array(weights_at_t)

                    # intermediate
                    weights_status_1 = weights_at_t[
                        np.array([s == 1.0 for s in status_at_t], dtype=bool)
                    ]
                    weights_status_0 = weights_at_t[
                        np.array([s == 0.0 for s in status_at_t], dtype=bool)
                    ]
                    n_intermediate_status_1[t] = np.sum(weights_status_1)
                    n_intermediate_status_0[t] = np.sum(weights_status_0)

                if "exclusive" in metrics:
                    n_exclusive_status_0[t] = sum(
                        [
                            c == 1.0 if s == 0.0 else False
                            for c, s in zip(closest_at_t, status_at_t)
                        ]
                    )
                    n_exclusive_status_1[t] = sum(
                        [
                            c == 1.0 if s == 1.0 else False
                            for c, s in zip(closest_at_t, status_at_t)
                        ]
                    )

        if "inclusive" in metrics:
            neigh_table.loc[indices, "inclusive_count_" + neigh_col] = n_inclusive
        if "intermediate" in metrics:
            neigh_table.loc[indices, "intermediate_count_" + neigh_col] = n_intermediate
        if "exclusive" in metrics:
            neigh_table.loc[indices, "exclusive_count_" + neigh_col] = n_exclusive

        if decompose_by_status:
            if "inclusive" in metrics:
                neigh_table.loc[indices, "inclusive_count_s0_" + neigh_col] = (
                    n_inclusive_status_0
                )
                neigh_table.loc[indices, "inclusive_count_s1_" + neigh_col] = (
                    n_inclusive_status_1
                )
            if "intermediate" in metrics:
                neigh_table.loc[indices, "intermediate_count_s0_" + neigh_col] = (
                    n_intermediate_status_0
                )
                neigh_table.loc[indices, "intermediate_count_s1_" + neigh_col] = (
                    n_intermediate_status_1
                )
            if "exclusive" in metrics:
                neigh_table.loc[indices, "exclusive_count_s0_" + neigh_col] = (
                    n_exclusive_status_0
                )
                neigh_table.loc[indices, "exclusive_count_s1_" + neigh_col] = (
                    n_exclusive_status_1
                )

    return neigh_table


def mean_neighborhood_before_event(
    neigh_table,
    neigh_col,
    event_time_col,
    metrics=["inclusive", "exclusive", "intermediate"],
):
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

    Returns
    -------
    pandas.DataFrame
            The input dataframe with added columns for the mean neighborhood metrics before the event for each cell track.
            The new columns are named as 'mean_count_{metric}_{neigh_col}_before_event', where {metric} is one of
            'inclusive', 'exclusive', 'intermediate'.

    """

    neigh_table = neigh_table.reset_index(drop=True)
    if "position" in list(neigh_table.columns):
        groupbycols = ["position"]
    else:
        groupbycols = []

    id_col = extract_identity_col(neigh_table)
    groupbycols.append(id_col)

    neigh_table.sort_values(by=groupbycols + ["FRAME"], inplace=True)
    suffix = "_before_event"

    if event_time_col is None:
        print(
            "No event time was provided... Estimating the mean neighborhood over the whole observation time..."
        )
        neigh_table.loc[:, "event_time_temp"] = neigh_table["FRAME"].max()
        event_time_col = "event_time_temp"
        suffix = ""

    for tid, group in neigh_table.groupby(groupbycols):

        group = group.dropna(subset=neigh_col)
        indices = list(group.index)

        event_time_values = group[event_time_col].to_numpy()
        if len(event_time_values) > 0:
            event_time = event_time_values[0]
        else:
            continue

        if event_time < 0.0:
            event_time = group["FRAME"].max()

        if "intermediate" in metrics:
            valid_counts_intermediate = group.loc[
                group["FRAME"] <= event_time, "intermediate_count_s1_" + neigh_col
            ].to_numpy()
            if (
                len(
                    valid_counts_intermediate[
                        valid_counts_intermediate == valid_counts_intermediate
                    ]
                )
                > 0
            ):
                neigh_table.loc[
                    indices, f"mean_count_intermediate_{neigh_col}{suffix}"
                ] = np.nanmean(valid_counts_intermediate)
        if "inclusive" in metrics:
            valid_counts_inclusive = group.loc[
                group["FRAME"] <= event_time, "inclusive_count_s1_" + neigh_col
            ].to_numpy()
            if (
                len(
                    valid_counts_inclusive[
                        valid_counts_inclusive == valid_counts_inclusive
                    ]
                )
                > 0
            ):
                neigh_table.loc[
                    indices, f"mean_count_inclusive_{neigh_col}{suffix}"
                ] = np.nanmean(valid_counts_inclusive)
        if "exclusive" in metrics:
            valid_counts_exclusive = group.loc[
                group["FRAME"] <= event_time, "exclusive_count_s1_" + neigh_col
            ].to_numpy()
            if (
                len(
                    valid_counts_exclusive[
                        valid_counts_exclusive == valid_counts_exclusive
                    ]
                )
                > 0
            ):
                neigh_table.loc[
                    indices, f"mean_count_exclusive_{neigh_col}{suffix}"
                ] = np.nanmean(valid_counts_exclusive)

    if event_time_col == "event_time_temp":
        neigh_table = neigh_table.drop(columns="event_time_temp")
    return neigh_table


def mean_neighborhood_after_event(
    neigh_table,
    neigh_col,
    event_time_col,
    metrics=["inclusive", "exclusive", "intermediate"],
):
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
            The column name indicating the event time for each cell track. If None, the maximum frame number in the
            dataframe is used as the event time for all tracks.

    Returns
    -------
    pandas.DataFrame
            The input dataframe with added columns for the mean neighborhood metrics before the event for each cell track.
            The new columns are named as 'mean_count_{metric}_{neigh_col}_before_event', where {metric} is one of
            'inclusive', 'exclusive', 'intermediate'.

    """

    neigh_table = neigh_table.reset_index(drop=True)
    if "position" in list(neigh_table.columns):
        groupbycols = ["position"]
    else:
        groupbycols = []

    id_col = extract_identity_col(neigh_table)
    groupbycols.append(id_col)

    neigh_table.sort_values(by=groupbycols + ["FRAME"], inplace=True)
    suffix = "_after_event"

    if event_time_col is None:
        neigh_table.loc[:, "event_time_temp"] = None  # neigh_table['FRAME'].max()
        event_time_col = "event_time_temp"
        suffix = ""

    for tid, group in neigh_table.groupby(groupbycols):

        group = group.dropna(subset=neigh_col)
        indices = list(group.index)

        event_time_values = group[event_time_col].to_numpy()
        if len(event_time_values) > 0:
            event_time = event_time_values[0]
        else:
            continue

        if event_time is not None and (event_time >= 0.0):

            if "intermediate" in metrics:
                valid_counts_intermediate = group.loc[
                    group["FRAME"] > event_time, "intermediate_count_s1_" + neigh_col
                ].to_numpy()
                if (
                    len(
                        valid_counts_intermediate[
                            valid_counts_intermediate == valid_counts_intermediate
                        ]
                    )
                    > 0
                ):
                    neigh_table.loc[
                        indices, f"mean_count_intermediate_{neigh_col}{suffix}"
                    ] = np.nanmean(valid_counts_intermediate)
            if "inclusive" in metrics:
                valid_counts_inclusive = group.loc[
                    group["FRAME"] > event_time, "inclusive_count_s1_" + neigh_col
                ].to_numpy()
                if (
                    len(
                        valid_counts_inclusive[
                            valid_counts_inclusive == valid_counts_inclusive
                        ]
                    )
                    > 0
                ):
                    neigh_table.loc[
                        indices, f"mean_count_inclusive_{neigh_col}{suffix}"
                    ] = np.nanmean(valid_counts_inclusive)
            if "exclusive" in metrics:
                valid_counts_exclusive = group.loc[
                    group["FRAME"] > event_time, "exclusive_count_s1_" + neigh_col
                ].to_numpy()
                if (
                    len(
                        valid_counts_exclusive[
                            valid_counts_exclusive == valid_counts_exclusive
                        ]
                    )
                    > 0
                ):
                    neigh_table.loc[
                        indices, f"mean_count_exclusive_{neigh_col}{suffix}"
                    ] = np.nanmean(valid_counts_exclusive)

    if event_time_col == "event_time_temp":
        neigh_table = neigh_table.drop(columns="event_time_temp")

    return neigh_table


# New functions for direct cell-cell contact neighborhood


def sign(num):
    return -1 if num < 0 else 1


def contact_neighborhood(labelsA, labelsB=None, border=3, connectivity=2):

    labelsA = labelsA.astype(float)
    if labelsB is not None:
        labelsB = labelsB.astype(float)

    if border > 0:
        labelsA_edge = contour_of_instance_segmentation(
            label=labelsA, distance=border * (-1)
        ).astype(float)
        labelsA[np.where(labelsA_edge > 0)] = labelsA_edge[np.where(labelsA_edge > 0)]
        if labelsB is not None:
            labelsB_edge = contour_of_instance_segmentation(
                label=labelsB, distance=border * (-1)
            ).astype(float)
            labelsB[np.where(labelsB_edge > 0)] = labelsB_edge[
                np.where(labelsB_edge > 0)
            ]

    if labelsB is not None:
        labelsA[labelsA != 0] = -labelsA[labelsA != 0]
        labelsAB = merge_labels(labelsA, labelsB)
        labelsBA = merge_labels(labelsB, labelsA)
        label_cases = [labelsAB, labelsBA]
    else:
        label_cases = [labelsA]

    coocurrences = []
    for lbl in label_cases:
        coocurrences.extend(find_contact_neighbors(lbl, connectivity=connectivity))

    unique_pairs = np.unique(coocurrences, axis=0)

    if labelsB is not None:
        neighs = np.unique(
            [
                tuple(sorted(p))
                for p in unique_pairs
                if p[0] != p[1] and sign(p[0]) != sign(p[1])
            ],
            axis=0,
        )
    else:
        neighs = np.unique(
            [tuple(sorted(p)) for p in unique_pairs if p[0] != p[1]], axis=0
        )

    return neighs


def merge_labels(labelsA, labelsB):

    labelsA = labelsA.astype(float)
    labelsB = labelsB.astype(float)

    labelsAB = labelsA.copy()
    labelsAB[np.where(labelsB != 0)] = labelsB[np.where(labelsB != 0)]

    return labelsAB


def find_contact_neighbors(labels, connectivity=2):

    assert labels.ndim == 2, "Wrong dimension for labels..."
    g, nodes = pixel_graph(labels, mask=labels.astype(bool), connectivity=connectivity)
    g.eliminate_zeros()

    coo = g.tocoo()
    center_coords = nodes[coo.row]
    neighbor_coords = nodes[coo.col]

    center_values = labels.ravel()[center_coords]
    neighbor_values = labels.ravel()[neighbor_coords]
    touching_masks = np.column_stack((center_values, neighbor_values))

    return touching_masks


def mask_contact_neighborhood(
    setA,
    setB,
    labelsA,
    labelsB,
    distance,
    mode="two-pop",
    status=None,
    not_status_option=None,
    compute_cum_sum=True,
    attention_weight=True,
    symmetrize=True,
    include_dead_weight=True,
    column_labels={
        "track": "TRACK_ID",
        "time": "FRAME",
        "x": "POSITION_X",
        "y": "POSITION_Y",
        "mask_id": "class_id",
    },
):
    """

    Match neighbors in set A and B within a circle of radius d.

    Parameters
    ----------
    setA,setB : pandas DataFrame
            Trajectory or position sets A and B.
    distance : float
            Cut-distance in pixels to match neighboring pairs.
    mode: str
            neighboring mode, between 'two-pop' (e.g. target-effector) and 'self' (target-target or effector-effector).
    status: None or status
            name to look for cells to ignore (because they are dead). By default all cells are kept.
    compute_cum_sum: bool,
            compute cumulated time of presence of neighbours (only if trajectories available for both sets)
    attention_weight: bool,
            compute the attention weight (how much a cell of set B is shared across cells of set A)
    symmetrize: bool,
            write in set B the neighborhood of set A
    include_dead_weight: bool
            do not count dead cells when establishing attention weight
    """

    if setA is not None and setB is not None:
        setA, setB, status = set_live_status(setA, setB, status, not_status_option)
    else:
        return None, None

    # Check distance option
    if not isinstance(distance, list):
        distance = [distance]

    cl = []
    for s in [setA, setB]:

        # Check whether data can be tracked
        temp_column_labels = column_labels.copy()

        id_col = extract_identity_col(s)
        temp_column_labels.update({"track": id_col})
        if id_col == "ID":
            compute_cum_sum = False

        cl.append(temp_column_labels)

    setA = setA.loc[~setA[cl[0]["track"]].isnull(), :].copy()
    setB = setB.loc[~setB[cl[1]["track"]].isnull(), :].copy()

    if labelsB is None:
        labelsB = [None] * len(labelsA)

    for d in distance:
        # loop over each provided distance

        if mode == "two-pop":
            neigh_col = f"neighborhood_2_contact_{d}_px"
        elif mode == "self":
            neigh_col = f"neighborhood_self_contact_{d}_px"
        else:
            print("Please provide a valid mode between `two-pop` and `self`...")
            return None

        setA[neigh_col] = np.nan
        setA[neigh_col] = setA[neigh_col].astype(object)

        setB[neigh_col] = np.nan
        setB[neigh_col] = setB[neigh_col].astype(object)

        # Loop over each available timestep
        timeline = np.unique(
            np.concatenate(
                [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
            )
        ).astype(int)
        for t in tqdm(timeline):

            setA_t = setA.loc[setA[cl[0]["time"]] == t, :]
            setB_t = setB.loc[setB[cl[1]["time"]] == t, :]

            if len(setA_t) > 0 and len(setB_t) > 0:
                dist_map, intersection_map = _compute_mask_contact_dist_map(
                    setA_t,
                    setB_t,
                    labelsA[t],
                    labelsB[t],
                    distance=d,
                    mode=mode,
                    column_labelsA=cl[0],
                    column_labelsB=cl[1],
                )

                d_filter = 1.0e05
                if attention_weight:
                    status_A = setA_t[status[0]].to_numpy()
                    ids_A = setA_t[cl[0]["track"]].to_numpy()
                    weights, closest_A = compute_attention_weight(
                        dist_map,
                        d_filter,
                        status_A,
                        ids_A,
                        axis=1,
                        include_dead_weight=include_dead_weight,
                    )
                else:
                    weights = None
                    closest_A = None

                _fill_contact_neighborhood_at_t(
                    t,
                    setA,
                    setB,
                    dist_map,
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
                    d_filter=d_filter,
                )

    return setA, setB


def compute_contact_neighborhood_at_position(
    pos,
    distance,
    population=["targets", "effectors"],
    theta_dist=None,
    img_shape=(2048, 2048),
    return_tables=False,
    clear_neigh=False,
    event_time_col=None,
    neighborhood_kwargs={
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    },
):
    """
    Computes neighborhood metrics for specified cell populations within a given position, based on distance criteria and additional parameters.

    This function assesses the neighborhood interactions between two specified cell populations (or within a single population) at a given position.
    It computes various neighborhood metrics based on specified distances, considering the entire image or excluding edge regions.
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
            Edge threshold(s) in pixels to exclude cells close to the image boundaries from the analysis. If not provided, defaults to 90% of each specified distance.
    img_shape : tuple of int, optional
            The dimensions (height, width) of the images in pixels (default is (2048, 2048)).
    return_tables : bool, optional
            If True, returns the updated data tables for both populations (default is False).
    clear_neigh : bool, optional
            If True, clears existing neighborhood columns from the data tables before computing new metrics (default is False).
    event_time_col : str, optional
            The column name indicating the event time for each cell, required if mean neighborhood metrics are to be computed before events.
    neighborhood_kwargs : dict, optional
            Additional keyword arguments for neighborhood computation, including mode, status options, and metrics (default includes mode 'two-pop', and symmetrization).

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, pandas.DataFrame)
            If `return_tables` is True, returns the updated data tables for the specified populations. If only one population is analyzed, both returned data frames will be identical.

    Raises
    ------
    AssertionError
            If the specified position path does not exist or if the number of distances and edge thresholds do not match.

    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    assert os.path.exists(pos), f"Position {pos} is not a valid path."

    if isinstance(population, str):
        population = [population, population]

    if not isinstance(distance, list):
        distance = [distance]
    if not theta_dist is None and not isinstance(theta_dist, list):
        theta_dist = [theta_dist]

    if theta_dist is None:
        theta_dist = [0 for d in distance]  # 0.9*d
    assert len(theta_dist) == len(
        distance
    ), "Incompatible number of distances and number of edge thresholds."

    if population[0] == population[1]:
        neighborhood_kwargs.update({"mode": "self"})
    if population[1] != population[0]:
        neighborhood_kwargs.update({"mode": "two-pop"})

    df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
    df_B, path_B = get_position_table(pos, population=population[1], return_path=True)
    if df_A is None or df_B is None:
        return None

    if clear_neigh:
        if os.path.exists(path_A.replace(".csv", ".pkl")):
            os.remove(path_A.replace(".csv", ".pkl"))
        if os.path.exists(path_B.replace(".csv", ".pkl")):
            os.remove(path_B.replace(".csv", ".pkl"))
        df_pair, pair_path = get_position_table(
            pos, population="pairs", return_path=True
        )
        if df_pair is not None:
            os.remove(pair_path)

    df_A_pkl = get_position_pickle(pos, population=population[0], return_path=False)
    df_B_pkl = get_position_pickle(pos, population=population[1], return_path=False)

    if df_A_pkl is not None:
        pkl_columns = np.array(df_A_pkl.columns)
        neigh_columns = np.array([c.startswith("neighborhood") for c in pkl_columns])
        cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

        id_col = extract_identity_col(df_A_pkl)
        cols.append(id_col)
        on_cols = [id_col, "FRAME"]

        print(f"Recover {cols} from the pickle file...")
        try:
            df_A = pd.merge(df_A, df_A_pkl.loc[:, cols], how="outer", on=on_cols)
            print(df_A.columns)
        except Exception as e:
            print(f"Failure to merge pickle and csv files: {e}")

    if df_B_pkl is not None and df_B is not None:
        pkl_columns = np.array(df_B_pkl.columns)
        neigh_columns = np.array([c.startswith("neighborhood") for c in pkl_columns])
        cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

        id_col = extract_identity_col(df_B_pkl)
        cols.append(id_col)
        on_cols = [id_col, "FRAME"]

        print(f"Recover {cols} from the pickle file...")
        try:
            df_B = pd.merge(df_B, df_B_pkl.loc[:, cols], how="outer", on=on_cols)
        except Exception as e:
            print(f"Failure to merge pickle and csv files: {e}")

    labelsA = locate_labels(pos, population=population[0])
    if population[1] == population[0]:
        labelsB = None
    else:
        labelsB = locate_labels(pos, population=population[1])

    if clear_neigh:
        unwanted = df_A.columns[df_A.columns.str.contains("neighborhood")]
        df_A = df_A.drop(columns=unwanted)
        unwanted = df_B.columns[df_B.columns.str.contains("neighborhood")]
        df_B = df_B.drop(columns=unwanted)

    print(f"Distance: {distance} for mask contact")
    df_A, df_B = mask_contact_neighborhood(
        df_A, df_B, labelsA, labelsB, distance, **neighborhood_kwargs
    )
    if df_A is None or df_B is None or len(df_A) == 0:
        return None

    for td, d in zip(theta_dist, distance):

        if neighborhood_kwargs["mode"] == "two-pop":
            neigh_col = f"neighborhood_2_contact_{d}_px"
        elif neighborhood_kwargs["mode"] == "self":
            neigh_col = f"neighborhood_self_contact_{d}_px"

        df_A.loc[df_A["class_id"].isnull(), neigh_col] = np.nan

        # edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
        # edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
        # df_A.loc[~edge_filter_A, neigh_col] = np.nan
        # df_B.loc[~edge_filter_B, neigh_col] = np.nan

        df_A = compute_neighborhood_metrics(
            df_A,
            neigh_col,
            metrics=["inclusive", "intermediate"],
            decompose_by_status=True,
        )
        if "TRACK_ID" in list(df_A.columns):
            if not np.all(df_A["TRACK_ID"].isnull()):
                df_A = mean_neighborhood_before_event(
                    df_A,
                    neigh_col,
                    event_time_col,
                    metrics=["inclusive", "intermediate"],
                )
                if event_time_col is not None:
                    df_A = mean_neighborhood_after_event(
                        df_A,
                        neigh_col,
                        event_time_col,
                        metrics=["inclusive", "intermediate"],
                    )
                print("Done...")

    if not population[0] == population[1]:
        # Remove neighborhood column from neighbor table, rename with actual population name
        for td, d in zip(theta_dist, distance):
            if neighborhood_kwargs["mode"] == "two-pop":
                neigh_col = f"neighborhood_2_contact_{d}_px"
                new_neigh_col = neigh_col.replace(
                    "_2_", f"_({population[0]}-{population[1]})_"
                )
                df_A = df_A.rename(columns={neigh_col: new_neigh_col})
            elif neighborhood_kwargs["mode"] == "self":
                neigh_col = f"neighborhood_self_contact_{d}_px"
            df_B = df_B.drop(columns=[neigh_col])
        df_B.to_pickle(path_B.replace(".csv", ".pkl"))

    cols_to_rename = [
        c
        for c in list(df_A.columns)
        if c.startswith("intermediate_count_")
        or c.startswith("inclusive_count_")
        or c.startswith("exclusive_count_")
        or c.startswith("mean_count_")
    ]
    new_col_names = [
        c.replace("_2_", f"_({population[0]}-{population[1]})_") for c in cols_to_rename
    ]
    new_name_map = {}
    for k, c in enumerate(cols_to_rename):
        new_name_map.update({c: new_col_names[k]})
    df_A = df_A.rename(columns=new_name_map)

    print(f"{df_A.columns=}")
    df_A.to_pickle(path_A.replace(".csv", ".pkl"))

    unwanted = df_A.columns[df_A.columns.str.startswith("neighborhood_")]
    df_A2 = df_A.drop(columns=unwanted)
    df_A2.to_csv(path_A, index=False)

    if not population[0] == population[1]:
        unwanted = df_B.columns[df_B.columns.str.startswith("neighborhood_")]
        df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
        df_B_csv.to_csv(path_B, index=False)

    if return_tables:
        return df_A, df_B


def extract_neighborhood_in_pair_table(
    df,
    distance=None,
    reference_population="targets",
    neighbor_population="effectors",
    mode="circle",
    neighborhood_key=None,
    contact_only=True,
):
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
            The reference population to consider. Must be either "targets" or "effectors".
    neighbor_population : str, default="effectors"
            The neighbor population to consider. Must be either "targets" or "effectors", used only if `neighborhood_key` is not provided.
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
    - Ensures that `reference_population` and `neighbor_population` are valid inputs and consistent with the neighborhood
      mode and key.

    Example
    -------
    >>> neighborhood_data = extract_neighborhood_in_pair_table(df, distance=50, reference_population="targets",
                                                                                                                       neighbor_population="effectors", mode="circle")
    >>> neighborhood_data.head()

    Raises
    ------
    AssertionError
            If `reference_population` or `neighbor_population` is not valid, or if the required neighborhood status
            column does not exist in `df`.
    """

    # assert reference_population in ["targets", "effectors"], "Please set a valid reference population ('targets' or 'effectors')"
    if neighborhood_key is None:
        # assert neighbor_population in ["targets", "effectors"], "Please set a valid neighbor population ('targets' or 'effectors')"
        assert mode in [
            "circle",
            "contact",
        ], "Please set a valid neighborhood computation mode ('circle' or 'contact')"
        type = "(" + "-".join([reference_population, neighbor_population]) + ")"
        neigh_col = f"neighborhood_{type}_{mode}_{distance}_px"
    else:
        neigh_col = neighborhood_key.replace("status_", "")
        if "_(" in neigh_col and ")_" in neigh_col:
            neighbor_population = (
                neigh_col.split("_(")[-1].split(")_")[0].split("-")[-1]
            )
        else:
            if "self" in neigh_col:
                neighbor_population = reference_population
            else:
                if reference_population == "effectors":
                    neighbor_population = "targets"
                else:
                    neighbor_population = "effectors"

    assert "status_" + neigh_col in list(
        df.columns
    ), "The selected neighborhood does not appear in the data..."

    print(df[["reference_population", "neighbor_population", "status_" + neigh_col]])

    if contact_only:
        s_keep = [1]
    else:
        s_keep = [0, 1]

    data = df.loc[
        (df["reference_population"] == reference_population)
        & (df["neighbor_population"] == neighbor_population)
        & (df["status_" + neigh_col].isin(s_keep))
    ]

    return data


# def mask_intersection_neighborhood(setA, labelsA, setB, labelsB, threshold_iou=0.5, viewpoint='B'):
# 	# do whatever to match objects in A and B
# 	return setA, setB

if __name__ == "__main__":

    print("None")
    pos = "/home/torro/Documents/Experiments/NKratio_Exp/W5/500"

    test, _ = compute_neighborhood_at_position(
        pos,
        [62],
        population=["targets", "effectors"],
        theta_dist=None,
        img_shape=(2048, 2048),
        return_tables=True,
        clear_neigh=True,
        neighborhood_kwargs={
            "mode": "two-pop",
            "status": ["class", None],
            "not_status_option": [True, False],
            "include_dead_weight": True,
            "compute_cum_sum": False,
            "attention_weight": True,
            "symmetrize": False,
        },
    )

    # test = compute_neighborhood_metrics(test, 'neighborhood_self_circle_150_px', metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
    print(test.columns)
    # print(segment(None,'test'))
