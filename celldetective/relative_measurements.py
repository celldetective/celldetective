"""
Relative Measurements Module
============================

This module calculates quantitative metrics describing the relationship between pairs of objects (reference and neighbor).
It relies on neighborhood definitions provided by the `neighborhood` module.

Key Features
------------
-   **Spatial Metrics**: Computes relative distance, angle, and orientation between cell pairs.
-   **Temporal Metrics**: Calculates relative velocity and interaction duration.
-   **Signal Coupling**: Analyzes how signals correlates between measuring pairs.

Main Functions
--------------
-   `measure_pairs`: Computes instantaneous spatial relationships for identified pairs.
-   `measure_pair_signals_at_position`: Extends pair measurements with signal analysis and temporal derivatives.
-   `rel_measure_at_position`: Wrapper script to execute relative measurements for an entire position.

Input
-----
Requires pre-computed tracking tables (`pkl` or `csv`) containing neighborhood information columns.
"""

from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
from celldetective.utils.maths import derivative
from celldetective.utils.data_cleaning import extract_identity_col
from celldetective.utils.image_loaders import locate_labels, locate_stack
from celldetective.neighborhood import _contact_site_mask
import os
import subprocess
import sys
import logging

logger = logging.getLogger("celldetective")

abs_path = os.sep.join(
    [os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"]
)


def _load_pair_tables(
    pos: str, reference_population: str, neighbor_population: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load reference and neighbor trajectory tables from pkl or csv files."""
    tab_ref = pos + os.sep.join(
        ["output", "tables", f"trajectories_{reference_population}.pkl"]
    )
    if os.path.exists(tab_ref):
        df_reference = pd.read_pickle(tab_ref)
    elif os.path.exists(tab_ref.replace(".pkl", ".csv")):
        df_reference = pd.read_csv(tab_ref.replace(".pkl", ".csv"))
    else:
        df_reference = None

    tab_neigh = tab_ref.replace(reference_population, neighbor_population)
    if os.path.exists(tab_neigh):
        df_neighbor = pd.read_pickle(tab_neigh)
    elif os.path.exists(tab_neigh.replace(".pkl", ".csv")):
        df_neighbor = pd.read_csv(tab_neigh.replace(".pkl", ".csv"))
    else:
        df_neighbor = None

    return df_reference, df_neighbor


def _build_neighbor_timeline(
    group: pd.DataFrame, neighborhood_description: str
) -> Tuple[list, list, pd.DataFrame, dict]:
    """Build per-frame neighbor ID lists and intersection values for one reference cell."""
    neighbor_dicts = group.loc[:, f"{neighborhood_description}"].values
    frames = group["FRAME"].values

    neighbor_ids: list = []
    neighbor_ids_per_t: list = []
    intersection_rows: list = []
    time_of_first_entrance: dict = {}

    for t in range(len(group)):
        neighbors_at_t = neighbor_dicts[t]
        frame_t = int(frames[t])
        neighs_t: list = []
        if not (isinstance(neighbors_at_t, float) or neighbors_at_t != neighbors_at_t):
            for neigh in neighbors_at_t:
                if neigh["id"] not in neighbor_ids:
                    time_of_first_entrance[neigh["id"]] = frame_t
                intersection_rows.append(
                    {
                        "frame": frame_t,
                        "neigh_id": neigh["id"],
                        "intersection": neigh.get("intersection", np.nan),
                    }
                )
                neighbor_ids.append(neigh["id"])
                neighs_t.append(neigh["id"])
        neighbor_ids_per_t.append(neighs_t)

    return neighbor_ids, neighbor_ids_per_t, pd.DataFrame(intersection_rows), time_of_first_entrance


def _compute_pair_geometry(
    coords_reference: np.ndarray,
    coords_neighbor: np.ndarray,
    coords_center_of_mass: list,
    center_of_mass_columns: list,
    timeline_reference: np.ndarray,
    timeline_neighbor: np.ndarray,
    full_timeline: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute relative position vectors, angles, distances, and dot products over time."""
    n = len(full_timeline)
    n_com = len(center_of_mass_columns)

    neighbor_vector = np.full((n, 2), np.nan)
    mass_displacement_vector = np.full((n_com, n, 2), np.nan)
    dot_product_vector = np.full((n_com, n), np.nan)
    cosine_dot_vector = np.full((n_com, n), np.nan)

    ref_idx_map = {frame: i for i, frame in enumerate(timeline_reference)}
    neigh_idx_map = {frame: i for i, frame in enumerate(timeline_neighbor)}

    for t_idx in range(n):
        frame = full_timeline[t_idx]
        if frame in ref_idx_map and frame in neigh_idx_map:
            idx_ref = ref_idx_map[frame]
            idx_neigh = neigh_idx_map[frame]

            neighbor_vector[t_idx, 0] = coords_neighbor[idx_neigh, 0] - coords_reference[idx_ref, 0]
            neighbor_vector[t_idx, 1] = coords_neighbor[idx_neigh, 1] - coords_reference[idx_ref, 1]

            for z in range(n_com):
                mass_displacement_vector[z, t_idx, 0] = (
                    coords_center_of_mass[z][idx_neigh, 0] - coords_neighbor[idx_neigh, 0]
                )
                mass_displacement_vector[z, t_idx, 1] = (
                    coords_center_of_mass[z][idx_neigh, 1] - coords_neighbor[idx_neigh, 1]
                )
                dot_product_vector[z, t_idx] = np.dot(
                    mass_displacement_vector[z, t_idx], -neighbor_vector[t_idx]
                )
                norm_prod = np.linalg.norm(mass_displacement_vector[z, t_idx]) * np.linalg.norm(
                    -neighbor_vector[t_idx]
                )
                cosine_dot_vector[z, t_idx] = dot_product_vector[z, t_idx] / norm_prod

    exclude = neighbor_vector[:, 1] != neighbor_vector[:, 1]
    angle = np.full(n, np.nan)
    angle[~exclude] = np.unwrap(
        np.arctan2(neighbor_vector[:, 1][~exclude], neighbor_vector[:, 0][~exclude])
    )
    relative_distance = np.sqrt(neighbor_vector[:, 0] ** 2 + neighbor_vector[:, 1] ** 2)

    return neighbor_vector, angle, relative_distance, dot_product_vector, cosine_dot_vector, exclude


def _compute_pair_velocities(
    relative_distance: np.ndarray,
    angle: np.ndarray,
    full_timeline: np.ndarray,
    exclude: np.ndarray,
    velocity_kwargs: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute relative velocity and angular velocity at short and long timescales."""
    n = len(full_timeline)
    rel_velocity = derivative(relative_distance, full_timeline, **velocity_kwargs)
    rel_velocity_smooth = derivative(relative_distance, full_timeline, window=7, mode="bi")

    angular_velocity = np.full(n, np.nan)
    angular_velocity_smooth = np.full(n, np.nan)
    angular_velocity[~exclude] = derivative(
        angle[~exclude], full_timeline[~exclude], **velocity_kwargs
    )
    angular_velocity_smooth[~exclude] = derivative(
        angle[~exclude], full_timeline[~exclude], window=7, mode="bi"
    )

    return rel_velocity, rel_velocity_smooth, angular_velocity, angular_velocity_smooth


def _build_pair_row(
    tid: float,
    nc: float,
    reference_population: str,
    neighbor_population: str,
    t: int,
    idx: int,
    relative_distance: np.ndarray,
    rel_velocity: np.ndarray,
    rel_velocity_smooth: np.ndarray,
    angle: np.ndarray,
    angular_velocity: np.ndarray,
    angular_velocity_smooth: np.ndarray,
    inter: float,
    ref_inter_fraction: float,
    neigh_inter_fraction: float,
    status: int,
    cum_sum: int,
    neighborhood_description: str,
    time_of_first_entrance: dict,
    ref_tracked: bool,
    neigh_tracked: bool,
    center_of_mass_labels: list,
    dot_product_vector: np.ndarray,
    cosine_dot_vector: np.ndarray,
) -> dict:
    """Assemble a single measurement row dict for one pair at one timepoint.

    Parameters
    ----------
    t : int
        Actual frame number (used for the FRAME column).
    idx : int
        Position of ``t`` within ``full_timeline`` (used to index geometry arrays).
        When full_timeline = [0, 1, 2, ...], ``idx == t``; they differ when tracks
        start at a frame other than 0.
    """
    row: dict = {
        "REFERENCE_ID": tid,
        "NEIGHBOR_ID": nc,
        "reference_population": reference_population,
        "neighbor_population": neighbor_population,
        "FRAME": t,
        "distance": relative_distance[idx],
        "intersection": inter,
        "reference_frac_area_intersection": ref_inter_fraction,
        "neighbor_frac_area_intersection": neigh_inter_fraction,
        "velocity": rel_velocity[idx],
        "velocity_smooth": rel_velocity_smooth[idx],
        "angle": angle[idx] * 180 / np.pi,
        "angular_velocity": angular_velocity[idx],
        "angular_velocity_smooth": angular_velocity_smooth[idx],
        f"status_{neighborhood_description}": status,
        f"residence_time_in_{neighborhood_description}": cum_sum,
        f"class_{neighborhood_description}": 0,
        f"t0_{neighborhood_description}": time_of_first_entrance[nc],
        "reference_tracked": ref_tracked,
        "neighbors_tracked": neigh_tracked,
    }
    for z, lbl in enumerate(center_of_mass_labels):
        row[lbl + "_center_of_mass_dot_product"] = dot_product_vector[z, idx]
        row[lbl + "_center_of_mass_dot_cosine"] = cosine_dot_vector[z, idx]
    return row


def measure_pairs(pos: str, neighborhood_protocol: dict) -> Optional[pd.DataFrame]:
    """
    Measures properties of cell pairs defined by a neighborhood protocol at a specific position.

    Parameters
    ----------
    pos : str
        Path to the experimental position.
    neighborhood_protocol : dict
        Dictionary containing neighborhood settings (reference, neighbor, type, distance, etc.).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing paired measurements, or None if data is missing.
    """

    reference_population = neighborhood_protocol["reference"]
    neighbor_population = neighborhood_protocol["neighbor"]
    neighborhood_type = neighborhood_protocol["type"]
    neighborhood_distance = neighborhood_protocol["distance"]
    neighborhood_description = neighborhood_protocol["description"]

    relative_measurements = []

    df_reference, df_neighbor = _load_pair_tables(pos, reference_population, neighbor_population)

    if df_reference is None:
        return None

    if str(neighborhood_description) not in df_reference.columns:
        raise KeyError(f"Neighborhood description '{neighborhood_description}' not found in reference columns.")
    neighborhood = df_reference.loc[:, f"{neighborhood_description}"].to_numpy()

    ref_id_col = extract_identity_col(df_reference)
    ref_tracked = False
    if ref_id_col is None:
        return None
    elif ref_id_col == "TRACK_ID":
        ref_tracked = True
    neigh_id_col = extract_identity_col(df_neighbor)
    neigh_tracked = False
    if neigh_id_col is None:
        return None
    elif neigh_id_col == "TRACK_ID":
        neigh_tracked = True

    center_of_mass_columns = [
        (c, c.replace("POSITION_X", "POSITION_Y"))
        for c in list(df_neighbor.columns)
        if c.endswith("center_of_mass_POSITION_X")
    ]
    center_of_mass_labels = [
        c.replace("_center_of_mass_POSITION_X", "")
        for c in list(df_neighbor.columns)
        if c.endswith("center_of_mass_POSITION_X")
    ]

    for t in np.unique(
        list(df_reference["FRAME"].unique()) + list(df_neighbor["FRAME"])
    ):

        group_reference = df_reference.loc[df_reference["FRAME"] == t, :]
        group_neighbors = df_neighbor.loc[df_neighbor["FRAME"] == t, :]

        for tid, group in group_reference.groupby(ref_id_col):

            neighborhood = group.loc[:, f"{neighborhood_description}"].to_numpy()[0]
            coords_reference = group[["POSITION_X", "POSITION_Y"]].to_numpy()[0]

            neighbors = []
            if not (isinstance(neighborhood, float) or neighborhood != neighborhood):
                for neigh in neighborhood:
                    neighbors.append(neigh["id"])

            unique_neigh = list(np.unique(neighbors))
            logger.debug(f"unique_neigh={unique_neigh}")

            neighbor_properties = group_neighbors.loc[
                group_neighbors[neigh_id_col].isin(unique_neigh)
            ]

            for nc, group_neigh in neighbor_properties.groupby(neigh_id_col):

                neighbor_vector = np.zeros(2)
                neighbor_vector[:] = np.nan
                mass_displacement_vector = np.zeros((len(center_of_mass_columns), 2))

                coords_center_of_mass = []
                for col in center_of_mass_columns:
                    coords_center_of_mass.append(
                        group_neigh[[col[0], col[1]]].to_numpy()[0]
                    )

                dot_product_vector = np.zeros((len(center_of_mass_columns)))
                dot_product_vector[:] = np.nan

                cosine_dot_vector = np.zeros((len(center_of_mass_columns)))
                cosine_dot_vector[:] = np.nan

                coords_neighbor = group_neigh[["POSITION_X", "POSITION_Y"]].to_numpy()[
                    0
                ]
                intersection = np.nan
                if "intersection" in list(group_neigh.columns):
                    intersection = group_neigh["intersection"].values[0]

                neighbor_vector[0] = coords_neighbor[0] - coords_reference[0]
                neighbor_vector[1] = coords_neighbor[1] - coords_reference[1]

                if (
                    neighbor_vector[0] == neighbor_vector[0]
                    and neighbor_vector[1] == neighbor_vector[1]
                ):
                    angle = np.arctan2(neighbor_vector[1], neighbor_vector[0])
                    relative_distance = np.sqrt(
                        neighbor_vector[0] ** 2 + neighbor_vector[1] ** 2
                    )

                    for z, cols in enumerate(center_of_mass_columns):

                        mass_displacement_vector[z, 0] = (
                            coords_center_of_mass[z][0] - coords_neighbor[0]
                        )
                        mass_displacement_vector[z, 1] = (
                            coords_center_of_mass[z][1] - coords_neighbor[1]
                        )

                        dot_product_vector[z] = np.dot(
                            mass_displacement_vector[z], -neighbor_vector
                        )
                        cosine_dot_vector[z] = np.dot(
                            mass_displacement_vector[z], -neighbor_vector
                        ) / (
                            np.linalg.norm(mass_displacement_vector[z])
                            * np.linalg.norm(-neighbor_vector)
                        )

                    relative_measurements.append(
                        {
                            "REFERENCE_ID": tid,
                            "NEIGHBOR_ID": nc,
                            "reference_population": reference_population,
                            "neighbor_population": neighbor_population,
                            "FRAME": t,
                            "distance": relative_distance,
                            "intersection": intersection,
                            "angle": angle * 180 / np.pi,
                            f"status_{neighborhood_description}": 1,
                            f"class_{neighborhood_description}": 0,
                            "reference_tracked": ref_tracked,
                            "neighbors_tracked": neigh_tracked,
                        }
                    )
                    for z, lbl in enumerate(center_of_mass_labels):
                        relative_measurements[-1].update(
                            {
                                lbl
                                + "_center_of_mass_dot_product": dot_product_vector[z],
                                lbl
                                + "_center_of_mass_dot_cosine": cosine_dot_vector[z],
                            }
                        )

    df_pairs = pd.DataFrame(relative_measurements)

    return df_pairs


def _measure_contact_site_intensity(
    labelsA_t: np.ndarray,
    labelsB_t: Optional[np.ndarray],
    ref_class_id: int,
    neigh_class_id: int,
    intensity_t: np.ndarray,
    channel_names: List[str],
    border: int = 3,
) -> Dict[str, float]:
    """
    Compute contact-zone intensity statistics for one pair at one timepoint.

    Parameters
    ----------
    labelsA_t : ndarray, shape (H, W)
        Label image for population A at this timepoint.
    labelsB_t : ndarray or None, shape (H, W)
        Label image for population B at this timepoint. If None (self-contact),
        labelsA_t is used for both populations.
    ref_class_id : int
        Label value of the reference cell in labelsA_t.
    neigh_class_id : int
        Label value of the neighbor cell in labelsB_t.
    intensity_t : ndarray, shape (H, W, C)
        Multi-channel intensity image at this timepoint.
    channel_names : list of str
        Names of the C channels in intensity_t.
    border : int
        Dilation radius (pixels) used to define the contact zone.

    Returns
    -------
    dict
        Keys ``contact_{ch}_mean``, ``contact_{ch}_max``, ``contact_{ch}_std``
        for each channel.  Values are NaN when the contact zone is empty.
    """
    if labelsB_t is None:
        labelsB_t = labelsA_t

    result: Dict[str, float] = {}
    try:
        zone = _contact_site_mask(labelsA_t, labelsB_t, ref_class_id, neigh_class_id, border)
    except Exception as e:
        logger.debug(f"_contact_site_mask failed for ids ({ref_class_id}, {neigh_class_id}): {e}")
        zone = np.zeros_like(labelsA_t, dtype=bool)

    for ch_idx, ch_name in enumerate(channel_names):
        if not np.any(zone) or intensity_t.ndim < 3 or ch_idx >= intensity_t.shape[2]:
            result[f"contact_{ch_name}_mean"] = np.nan
            result[f"contact_{ch_name}_max"] = np.nan
            result[f"contact_{ch_name}_std"] = np.nan
        else:
            pixels = intensity_t[:, :, ch_idx][zone].astype(float)
            result[f"contact_{ch_name}_mean"] = float(np.mean(pixels))
            result[f"contact_{ch_name}_max"] = float(np.max(pixels))
            result[f"contact_{ch_name}_std"] = float(np.std(pixels))
    return result


def measure_pair_signals_at_position(
    pos: str,
    neighborhood_protocol: dict,
    velocity_kwargs: dict = {"window": 3, "mode": "bi"},
) -> Optional[pd.DataFrame]:
    """
    Measures signals and temporal properties for cell pairs at a specific position.

    Parameters
    ----------
    pos : str
        Path to the experimental position.
    neighborhood_protocol : dict
        Dictionary containing neighborhood settings.
    velocity_kwargs : dict, optional
        Arguments for velocity calculation (window size, mode). Default is {"window": 3, "mode": "bi"}.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing temporal pair measurements, or None if data is missing.
    """

    reference_population = neighborhood_protocol["reference"]
    neighbor_population = neighborhood_protocol["neighbor"]
    neighborhood_description = neighborhood_protocol["description"]

    df_reference, df_neighbor = _load_pair_tables(pos, reference_population, neighbor_population)

    if df_reference is None:
        return None

    if str(neighborhood_description) not in df_reference.columns:
        raise KeyError(f"Neighborhood description '{neighborhood_description}' not found in reference columns.")

    ref_id_col = extract_identity_col(df_reference)
    if ref_id_col is not None:
        df_reference = df_reference.sort_values(by=[ref_id_col, "FRAME"])

    ref_tracked = False
    if ref_id_col == "TRACK_ID":
        ref_tracked = True
    elif ref_id_col == "ID":
        return measure_pairs(pos, neighborhood_protocol)
    else:
        logger.error("ID or TRACK ID column could not be found in reference table. Abort.")
        return None

    logger.info("Measuring pair signals...")

    neigh_id_col = extract_identity_col(df_neighbor)
    neigh_tracked = False
    if neigh_id_col == "TRACK_ID":
        neigh_tracked = True
    elif neigh_id_col == "ID":
        return measure_pairs(pos, neighborhood_protocol)
    else:
        logger.error("ID or TRACK ID column could not be found in neighbor table. Abort.")
        return None

    # --- Contact-site intensity setup (mask_contact neighborhoods only) -------
    channel_names = neighborhood_protocol.get("channel_names")
    contact_border = int(neighborhood_protocol.get("contact_border", 3))
    labelsA_all = None
    labelsB_all = None
    intensity_stack = None  # list of (H, W, C) arrays, one per frame

    if channel_names:
        try:
            labelsA_all = locate_labels(pos, population=reference_population)
            if neighbor_population != reference_population:
                labelsB_all = locate_labels(pos, population=neighbor_population)
            raw_stack = locate_stack(pos)  # (T, H, W, C)
            if raw_stack is not None:
                intensity_stack = [raw_stack[t] for t in range(raw_stack.shape[0])]
        except Exception as e:
            logger.warning(f"Could not load labels/stack for contact-site intensity: {e}")
            channel_names = None  # disable gracefully

    # Build fast lookup: (track_id, frame) -> class_id for both populations
    ref_class_lookup: Dict[Tuple, int] = {}
    neigh_class_lookup: Dict[Tuple, int] = {}
    if channel_names and "class_id" in df_reference.columns:
        sub = df_reference[["FRAME", ref_id_col, "class_id"]].dropna(subset=["class_id", "FRAME"])
        for _, row in sub.iterrows():
            ref_class_lookup[(row[ref_id_col], int(row["FRAME"]))] = int(row["class_id"])
    if channel_names and "class_id" in df_neighbor.columns:
        sub = df_neighbor[["FRAME", neigh_id_col, "class_id"]].dropna(subset=["class_id", "FRAME"])
        for _, row in sub.iterrows():
            neigh_class_lookup[(row[neigh_id_col], int(row["FRAME"]))] = int(row["class_id"])
    # -------------------------------------------------------------------------

    relative_measurements: list = []

    try:
        for tid, group in df_reference.groupby(ref_id_col):

            timeline_reference = group["FRAME"].to_numpy()
            coords_reference = group[["POSITION_X", "POSITION_Y"]].to_numpy()
            ref_area = (
                group["area"].to_numpy() if "area" in group.columns else [np.nan] * len(coords_reference)
            )

            neighbor_ids, neighbor_ids_per_t, intersection_values, time_of_first_entrance = (
                _build_neighbor_timeline(group, neighborhood_description)
            )

            unique_neigh = list(np.unique(neighbor_ids))
            logger.debug(
                f"Reference cell {tid}: found {len(unique_neigh)} neighbour cells: {unique_neigh}..."
            )

            neighbor_properties = df_neighbor.loc[df_neighbor[neigh_id_col].isin(unique_neigh)]

            center_of_mass_columns = [
                (c, c.replace("POSITION_X", "POSITION_Y"))
                for c in neighbor_properties.columns
                if c.endswith("center_of_mass_POSITION_X")
            ]
            center_of_mass_labels = [
                c.replace("_center_of_mass_POSITION_X", "")
                for c in neighbor_properties.columns
                if c.endswith("center_of_mass_POSITION_X")
            ]

            for nc, group_neigh in neighbor_properties.groupby(neigh_id_col):

                timeline_neighbor = group_neigh["FRAME"].to_numpy()
                coords_neighbor = group_neigh[["POSITION_X", "POSITION_Y"]].to_numpy()
                neigh_area = (
                    group_neigh["area"].to_numpy()
                    if "area" in group_neigh.columns
                    else [np.nan] * len(timeline_neighbor)
                )
                coords_center_of_mass = [
                    group_neigh[[col[0], col[1]]].to_numpy() for col in center_of_mass_columns
                ]

                full_timeline, _, _ = timeline_matching(timeline_reference, timeline_neighbor)

                _, angle, relative_distance, dot_product_vector, cosine_dot_vector, exclude = (
                    _compute_pair_geometry(
                        coords_reference,
                        coords_neighbor,
                        coords_center_of_mass,
                        center_of_mass_columns,
                        timeline_reference,
                        timeline_neighbor,
                        full_timeline,
                    )
                )

                rel_velocity, rel_velocity_smooth, angular_velocity, angular_velocity_smooth = (
                    _compute_pair_velocities(
                        relative_distance, angle, full_timeline, exclude, velocity_kwargs
                    )
                )

                # Pre-build frame→position maps to avoid repeated O(N) list.index() calls
                ref_frame_to_idx = {int(f): i for i, f in enumerate(timeline_reference)}
                neigh_frame_to_idx = {int(f): i for i, f in enumerate(timeline_neighbor)}

                cum_sum = 0
                for idx, t in enumerate(full_timeline):
                    t = int(t)
                    if t not in ref_frame_to_idx or t not in neigh_frame_to_idx:
                        continue

                    idx_reference = ref_frame_to_idx[t]
                    idx_neighbor = neigh_frame_to_idx[t]

                    inter_vals = intersection_values.loc[
                        (intersection_values["neigh_id"] == nc)
                        & (intersection_values["frame"] == t),
                        "intersection",
                    ].values
                    inter = np.nan if len(inter_vals) == 0 else inter_vals[0]

                    neigh_inter_fraction = (
                        inter / neigh_area[idx_neighbor]
                        if inter == inter and neigh_area[idx_neighbor] == neigh_area[idx_neighbor]
                        else np.nan
                    )
                    ref_inter_fraction = (
                        inter / ref_area[idx_reference]
                        if inter == inter and ref_area[idx_reference] == ref_area[idx_reference]
                        else np.nan
                    )

                    in_neighborhood = nc in neighbor_ids_per_t[idx_reference]
                    if in_neighborhood:
                        cum_sum += 1
                    status = 1 if in_neighborhood else 0

                    row = _build_pair_row(
                        tid,
                        nc,
                        reference_population,
                        neighbor_population,
                        t,
                        idx,
                        relative_distance,
                        rel_velocity,
                        rel_velocity_smooth,
                        angle,
                        angular_velocity,
                        angular_velocity_smooth,
                        inter,
                        ref_inter_fraction,
                        neigh_inter_fraction,
                        status,
                        cum_sum,
                        neighborhood_description,
                        time_of_first_entrance,
                        ref_tracked,
                        neigh_tracked,
                        center_of_mass_labels,
                        dot_product_vector,
                        cosine_dot_vector,
                    )

                    # Contact-site intensity (mask_contact only, cells in contact)
                    if channel_names and in_neighborhood and intensity_stack is not None:
                        lA_t = labelsA_all[t] if labelsA_all is not None and t < len(labelsA_all) else None
                        lB_t = labelsB_all[t] if labelsB_all is not None and t < len(labelsB_all) else lA_t
                        img_t = intensity_stack[t] if t < len(intensity_stack) else None
                        ref_cid = ref_class_lookup.get((tid, t))
                        neigh_cid = neigh_class_lookup.get((nc, t))
                        if lA_t is not None and img_t is not None and ref_cid is not None and neigh_cid is not None:
                            contact_stats = _measure_contact_site_intensity(
                                lA_t, lB_t, ref_cid, neigh_cid,
                                img_t, channel_names, contact_border,
                            )
                            row.update(contact_stats)

                    relative_measurements.append(row)

        return pd.DataFrame(relative_measurements)

    except KeyError:
        logger.warning(
            "Neighborhood not found in data frame. Measurements for this neighborhood will not be calculated."
        )


def timeline_matching(
    timeline1: np.ndarray, timeline2: np.ndarray
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Match two timelines and create a unified timeline with corresponding indices.

    Parameters
    ----------
    timeline1 : array-like
            The first timeline to be matched.
    timeline2 : array-like
            The second timeline to be matched.

    Returns
    -------
    tuple
            A tuple containing:

                    - full_timeline : numpy.ndarray
                            The unified timeline spanning from the minimum to the maximum time point in the input timelines.
                    - index1 : list of int
                            The indices of `timeline1` in the `full_timeline`.
                    - index2 : list of int
                            The indices of `timeline2` in the `full_timeline`.

    Examples
    --------
    >>> timeline1 = [1, 2, 5, 6]
    >>> timeline2 = [2, 3, 4, 6]
    >>> full_timeline, index1, index2 = timeline_matching(timeline1, timeline2)
    >>> print(full_timeline)
    [1 2 3 4 5 6]
    >>> print(index1)
    [0, 1, 4, 5]
    >>> print(index2)
    [1, 2, 3, 5]

    Notes
    -----
    - The function combines the two timelines and generates a continuous range from the minimum to the maximum time point.
    - It then finds the indices of the original timelines in this unified timeline.
    - The function assumes that the input timelines consist of integer values.

    """

    min_t = np.amin(np.concatenate((timeline1, timeline2)))
    max_t = np.amax(np.concatenate((timeline1, timeline2)))
    full_timeline = np.arange(min_t, max_t + 1)

    index1 = [list(np.where(full_timeline == int(t))[0])[0] for t in timeline1]
    index2 = [list(np.where(full_timeline == int(t))[0])[0] for t in timeline2]

    return full_timeline, index1, index2


def rel_measure_at_position(pos: str) -> None:
    """
    Executes the relative measurement script for a given position.

    Parameters
    ----------
    pos : str
        Path to the experimental position.
    """

    pos = pos.replace("\\", "/")
    pos = rf"{pos}"
    if not os.path.exists(pos):
        raise FileNotFoundError(f"Position {pos} is not a valid path.")
    if not pos.endswith("/"):
        pos += "/"
    script_path = os.sep.join([abs_path, "scripts", "measure_relative.py"])
    result = subprocess.run(
        [sys.executable, script_path, "--pos", pos],
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"Relative measurement script exited with code {result.returncode} for position {pos}.")
        raise RuntimeError(f"Relative measurement failed for position {pos} (exit code {result.returncode}).")


# def mcf7_size_model(x,x0,x2):
# 	return np.piecewise(x, [x<= x0, (x > x0)*(x<=x2), x > x2], [lambda x: 1, lambda x: -1/(x2-x0)*x + (1+x0/(x2-x0)), 0])
# def sigmoid(x,x0,k):
# 	return 1/(1 + np.exp(-(x-x0)/k))
# def velocity_law(x):
# 	return np.piecewise(x, [x<=-10, x > -10],[lambda x: 0., lambda x: (1*x+10)*(1-sigmoid(x, 1,1))/10])


# def probabilities(pairs,radius_critical=80,radius_max=150):
# 	scores = []
# 	pair_dico=[]
# 	print(f'Found {len(pairs)} TC-NK pairs...')
# 	if len(pairs) > 0:
# 		unique_tcs = np.unique(pairs['tc'].to_numpy())
# 		unique_nks = np.unique(pairs['nk'].to_numpy())
# 		matrix = np.zeros((len(unique_tcs), len(unique_nks)))
# 		for index, row in pairs.iterrows():

# 			i = np.where(unique_tcs == row['tc'])[0]
# 			j = np.where(unique_nks == row['nk'])[0]

# 			d_prob = mcf7_size_model(row['drel'], radius_critical, radius_max)
# 			lamp_prob = sigmoid(row['lamp1'], 1.05, 0.01)
# 			synapse_prob = row['syn_class']
# 			velocity_prob = velocity_law(row['vrel'])  # 1-sigmoid(row['vrel'], 1,1)
# 			time_prob = row['t_residence_rel']

# 			hypotheses = [d_prob, velocity_prob, lamp_prob, synapse_prob,
# 						  time_prob]  # lamp_prob d_prob, synapse_prob, velocity_prob, lamp_prob
# 			s = np.sum(hypotheses) / len(hypotheses)

# 			matrix[i, j] = s  # synapse_prob': synapse_prob,
# 			pair_dico.append(
# 				{ 'tc': row['tc'], 'nk': row['nk'], 'synapse_prob': synapse_prob,
# 				 'd_prob': d_prob, 'lamp_prob': lamp_prob, 'velocity_prob': velocity_prob, 'time_prob': time_prob})
# 		pair_dico = pd.DataFrame(pair_dico)

# 		hypotheses = ['velocity_prob', 'd_prob', 'time_prob', 'lamp_prob', 'synapse_prob']

# 		for i in tqdm(range(2000)):
# 			sample = np.array(random.choices(np.linspace(0, 1, 100), k=len(hypotheses)))
# 			weights = sample / np.sum(sample)

# 			score_i = {}
# 			for k, hyp in enumerate(hypotheses):
# 				score_i.update({'w_' + hyp: weights[k]})
# 			probs=[]
# 			for cells, group in pair_dico.groupby(['tc']):

# 				group['total_prob'] = 0
# 				for hyp in hypotheses:
# 					group['total_prob'] += group[hyp] * score_i['w_' + hyp]
# 					probs.append(group)
# 	return probs


def update_effector_table(
    df_relative: pd.DataFrame, df_effector: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates the effector table to mark effectors that are part of a neighborhood.

    Parameters
    ----------
    df_relative : pandas.DataFrame
        DataFrame containing relative measurements (pairs).
    df_effector : pandas.DataFrame
        DataFrame containing effector data.

    Returns
    -------
    pandas.DataFrame
        Updated effector DataFrame with 'group_neighborhood' column.
    """
    df_effector["group_neighborhood"] = 1
    effectors = np.unique(df_relative["EFFECTOR_ID"].to_numpy())
    for effector in effectors:
        try:
            # Set group_neighborhood to 0 where TRACK_ID matches effector
            df_effector.loc[
                df_effector["TRACK_ID"] == effector, "group_neighborhood"
            ] = 0
        except KeyError:
            df_effector.loc[df_effector["ID"] == effector, "group_neighborhood"] = 0
    return df_effector


def extract_neighborhoods_from_pickles(
    pos: str, populations: List[str] = ["targets", "effectors"]
) -> List[dict]:
    """
    Extract neighborhood protocols from pickle files located at a given position.

    Parameters
    ----------
    pos : str
            The base directory path where the pickle files are located.
    populations : list of str, optional
            List of populations to extract neighborhoods from. Default is ["targets", "effectors"].

    Returns
    -------
    list of dict
            A list of dictionaries, each containing a neighborhood protocol. Each dictionary has the keys:

                    - 'reference' : str
                            The reference population ('targets' or 'effectors').
                    - 'neighbor' : str
                            The neighbor population.
                    - 'type' : str
                            The type of neighborhood ('circle' or 'contact').
                    - 'distance' : float
                            The distance parameter for the neighborhood.
                    - 'description' : str
                            The original neighborhood string.

    Notes
    -----
    - The function checks for the existence of pickle files containing target and effector trajectory data.
    - If the files exist, it loads the data and extracts columns that start with 'neighborhood'.
    - The neighborhood settings are extracted using the `extract_neighborhood_settings` function.
    - The function assumes the presence of subdirectories 'output/tables' under the provided `pos`.

    Examples
    --------
    >>> protocols = extract_neighborhoods_from_pickles('/path/to/data')
    >>> for protocol in protocols:
    >>>		print(protocol)
    {'reference': 'targets', 'neighbor': 'targets', 'type': 'contact', 'distance': 5.0, 'description': 'neighborhood_self_contact_5_px'}

    """

    neighborhood_protocols = []

    for pop in populations:
        tab_pop = pos + os.sep.join(["output", "tables", f"trajectories_{pop}.pkl"])
        if os.path.exists(tab_pop):
            df_pop = pd.read_pickle(tab_pop)
            for column in list(df_pop.columns):
                if column.startswith("neighborhood"):
                    neigh_protocol = extract_neighborhood_settings(
                        column, population=pop
                    )
                    neighborhood_protocols.append(neigh_protocol)

    # tab_tc = pos + os.sep.join(['output', 'tables', 'trajectories_targets.pkl'])
    # if os.path.exists(tab_tc):
    # 	df_targets = np.load(tab_tc, allow_pickle=True)
    # else:
    # 	df_targets = None
    # if os.path.exists(tab_tc.replace('targets','effectors')):
    # 	df_effectors = np.load(tab_tc.replace('targets','effectors'), allow_pickle=True)
    # else:
    # 	df_effectors = None

    # neighborhood_protocols=[]

    # if df_targets is not None:
    # 	for column in list(df_targets.columns):
    # 		if column.startswith('neighborhood'):
    # 			neigh_protocol = extract_neighborhood_settings(column, population='targets')
    # 			neighborhood_protocols.append(neigh_protocol)

    # if df_effectors is not None:
    # 	for column in list(df_effectors.columns):
    # 		if column.startswith('neighborhood'):
    # 			neigh_protocol = extract_neighborhood_settings(column, population='effectors')
    # 			neighborhood_protocols.append(neigh_protocol)

    return neighborhood_protocols


def extract_neighborhood_settings(
    neigh_string: str, population: str = "targets"
) -> dict:
    """
    Extract neighborhood settings from a given string.

    Parameters
    ----------
    neigh_string : str
            The string describing the neighborhood settings. Must start with 'neighborhood'.
    population : str, optional
            The population type ('targets' by default). Can be either 'targets' or 'effectors'.

    Returns
    -------
    dict
            A dictionary containing the neighborhood protocol with keys:

                    - 'reference' : str
                            The reference population.
                    - 'neighbor' : str
                            The neighbor population.
                    - 'type' : str
                            The type of neighborhood ('circle' or 'contact').
                    - 'distance' : float
                            The distance parameter for the neighborhood.
                    - 'description' : str
                            The original neighborhood string.

    Raises
    ------
    AssertionError
            If the `neigh_string` does not start with 'neighborhood'.

    Notes
    -----
    - The function determines the neighbor population based on the given population.
    - The neighborhood type and distance are extracted from the `neigh_string`.
    - The description field in the returned dictionary contains the original neighborhood string.

    Examples
    --------
    >>> extract_neighborhood_settings('neighborhood_self_contact_5_px', 'targets')
    {'reference': 'targets', 'neighbor': 'targets', 'type': 'contact', 'distance': 5.0, 'description': 'neighborhood_self_contact_5_px'}

    """

    if not neigh_string.startswith("neighborhood"):
        raise ValueError(f"Expected a neighborhood string starting with 'neighborhood', got: '{neigh_string}'")
    logger.debug(f"neigh_string={neigh_string}")

    if "_(" in neigh_string and ")_" in neigh_string:
        # determine neigh pop from string
        neighbor_population = neigh_string.split("_(")[-1].split(")_")[0].split("-")[-1]
        logger.debug(f"neighbor_population={neighbor_population}")
    else:
        # Legacy fallback: neighborhood string predates the _(pop)_ encoding.
        # Assume canonical target/effector pairing; for any other population
        # default to self-neighboring (same population).
        if population == "targets":
            neighbor_population = "effectors"
        elif population == "effectors":
            neighbor_population = "targets"
        else:
            neighbor_population = population

    if "self" in neigh_string:

        if "circle" in neigh_string:

            distance = float(neigh_string.split("circle_")[1].replace("_px", ""))
            neigh_protocol = {
                "reference": population,
                "neighbor": population,
                "type": "circle",
                "distance": distance,
                "description": neigh_string,
            }
        elif "contact" in neigh_string:
            distance = float(neigh_string.split("contact_")[1].replace("_px", ""))
            neigh_protocol = {
                "reference": population,
                "neighbor": population,
                "type": "contact",
                "distance": distance,
                "description": neigh_string,
            }
    else:

        if "circle" in neigh_string:

            distance = float(neigh_string.split("circle_")[1].replace("_px", ""))
            neigh_protocol = {
                "reference": population,
                "neighbor": neighbor_population,
                "type": "circle",
                "distance": distance,
                "description": neigh_string,
            }
        elif "contact" in neigh_string:

            distance = float(neigh_string.split("contact_")[1].replace("_px", ""))
            neigh_protocol = {
                "reference": population,
                "neighbor": neighbor_population,
                "type": "contact",
                "distance": distance,
                "description": neigh_string,
            }

    return neigh_protocol


def expand_pair_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Expands a pair table by merging reference and neighbor trajectory data from CSV files based on the specified
    reference and neighbor populations, and their associated positions and frames.

    Parameters
    ----------
    data : pandas.DataFrame
            DataFrame containing the pair table

    Returns
    -------
    pandas.DataFrame
            Expanded DataFrame that includes merged reference and neighbor data, sorted by position, reference population,
            neighbor population, and frame. Rows without values in `REFERENCE_ID`, `NEIGHBOR_ID`, `reference_population`,
            or `neighbor_population` are dropped.

    Notes
    -----
    - For each unique pair of `reference_population` and `neighbor_population`, the function identifies corresponding
      trajectories CSV files based on the position identifier.
    - The function reads the trajectories CSV files, prefixes columns with `reference_` or `neighbor_` to avoid
      conflicts, and merges data from reference and neighbor tables based on `TRACK_ID` or `ID`, and `FRAME`.
    - Merges are performed in an outer join manner to retain all rows, regardless of missing values in the target files.
    - The final DataFrame is sorted and cleaned to ensure only valid pairings are included.

    Example
    -------
    >>> expanded_df = expand_pair_table(pair_table)
    >>> expanded_df.head()

    Raises
    ------
    AssertionError
            If `reference_population` or `neighbor_population` is not found in the columns of `data`.

    """

    if "reference_population" not in data.columns:
        raise KeyError("Please provide a valid pair table...")
    if "neighbor_population" not in data.columns:
        raise KeyError("Please provide a valid pair table...")

    data.__dict__.update(
        data.astype({"reference_population": str, "neighbor_population": str}).__dict__
    )

    expanded_table = []

    for neigh, group in data.groupby(["reference_population", "neighbor_population"]):

        ref_pop = neigh[0]
        neigh_pop = neigh[1]

        for pos, pos_group in group.groupby("position"):

            ref_tab = os.sep.join(
                [pos, "output", "tables", f"trajectories_{ref_pop}.csv"]
            )
            neigh_tab = os.sep.join(
                [pos, "output", "tables", f"trajectories_{neigh_pop}.csv"]
            )

            if os.path.exists(ref_tab):
                df_ref = pd.read_csv(ref_tab)
                if "TRACK_ID" in df_ref.columns:
                    if not np.all(df_ref["TRACK_ID"].isnull()):
                        ref_merge_cols = ["TRACK_ID", "FRAME"]
                    else:
                        ref_merge_cols = ["ID", "FRAME"]
                else:
                    ref_merge_cols = ["ID", "FRAME"]

            if os.path.exists(neigh_tab):
                df_neigh = pd.read_csv(neigh_tab)
                if "TRACK_ID" in df_neigh.columns:
                    if not np.all(df_neigh["TRACK_ID"].isnull()):
                        neigh_merge_cols = ["TRACK_ID", "FRAME"]
                    else:
                        neigh_merge_cols = ["ID", "FRAME"]
                else:
                    neigh_merge_cols = ["ID", "FRAME"]

            df_ref = df_ref.add_prefix("reference_", axis=1)
            df_neigh = df_neigh.add_prefix("neighbor_", axis=1)
            ref_merge_cols = ["reference_" + c for c in ref_merge_cols]
            neigh_merge_cols = ["neighbor_" + c for c in neigh_merge_cols]

            merge_ref = pos_group.merge(
                df_ref,
                how="outer",
                left_on=["REFERENCE_ID", "FRAME"],
                right_on=ref_merge_cols,
                suffixes=("", "_reference"),
            )
            merge_neigh = merge_ref.merge(
                df_neigh,
                how="outer",
                left_on=["NEIGHBOR_ID", "FRAME"],
                right_on=neigh_merge_cols,
                suffixes=("_reference", "_neighbor"),
            )
            expanded_table.append(merge_neigh)

    df_expanded = pd.concat(expanded_table, axis=0, ignore_index=True)
    df_expanded = df_expanded.sort_values(
        by=[
            "position",
            "reference_population",
            "neighbor_population",
            "REFERENCE_ID",
            "NEIGHBOR_ID",
            "FRAME",
        ]
    )
    df_expanded = df_expanded.dropna(
        axis=0,
        subset=[
            "REFERENCE_ID",
            "NEIGHBOR_ID",
            "reference_population",
            "neighbor_population",
        ],
    )

    return df_expanded
