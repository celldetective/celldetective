"""
Checks of the synthetic effector-target movie and, through its ground
truth, of threshold segmentation and of the distance and contact
neighborhoods.
"""

import os

import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage.measure import label

from celldetective.neighborhood import (
    compute_contact_neighborhood_at_position,
    compute_neighborhood_at_position,
)
from celldetective.segmentation import segment_frame_from_thresholds
from celldetective.utils.synthetic_interactions import (
    simulate_interactions,
    write_experiment,
)

SHAPE = (160, 160)


@pytest.fixture(scope="module")
def sim():
    return simulate_interactions(seed=0)


@pytest.fixture
def position(tmp_path, sim):
    return write_experiment(
        str(tmp_path / "SynthExp"), sim, write_labels=True, write_tables=True
    )


def _pairs_in(df, col):
    """(TRACK_ID, neighbor id, FRAME) triplets listed in a neighborhood column."""
    found = set()
    for track, frame, neighbors in zip(df["TRACK_ID"], df["FRAME"], df[col]):
        if isinstance(neighbors, list):
            found |= {(int(track), int(n["id"]), int(frame)) for n in neighbors}
    return found


def _true_contacts(sim):
    c = sim.contacts
    return set(zip(c["TARGET_ID"], c["EFFECTOR_ID"], c["FRAME"]))


def test_same_seed_gives_same_movie():
    a, b = simulate_interactions(seed=3), simulate_interactions(seed=3)
    np.testing.assert_array_equal(a.movie, b.movie)
    pd.testing.assert_frame_equal(a.episodes, b.episodes)


@pytest.mark.parametrize("seed", range(5))
def test_masks_touch_exactly_during_dockings(seed):
    sim = simulate_interactions(seed=seed)
    docked = {
        (r.TARGET_ID, r.EFFECTOR_ID, t)
        for r in sim.episodes.itertuples()
        for t in range(r.t_start, r.t_end + 1)
    }
    assert len(docked) > 0
    assert _true_contacts(sim) == docked
    assert (sim.contacts["overlap"] > 0).all()


def test_contacts_have_short_and_long_durations():
    sim = simulate_interactions(seed=1, contact_durations=(3, 12))
    complete = sim.episodes[sim.episodes["t_end"] < sim.params["n_frames"] - 1]
    assert {3, 12} <= set(complete["duration"])


def test_threshold_segmentation_recovers_every_cell(sim):
    for t in range(0, sim.movie.shape[0], 5):
        frame = np.moveaxis(sim.movie[t], 0, -1).astype(float)
        for channel, truth in ((0, sim.labels_targets[t]), (1, sim.labels_effectors[t])):
            lbl = segment_frame_from_thresholds(
                frame,
                target_channel=channel,
                thresholds=(550, 1e6),
                do_watershed=False,
                edge_exclusion=False,
            )
            assert label(lbl > 0).max() == len(np.unique(truth)) - 1
            agreement = np.mean((lbl > 0) == (truth > 0))
            assert agreement > 0.99


def test_death_channel_marks_dead_targets():
    sim = simulate_interactions(seed=0, death_after=4)
    assert sim.channel_names[-1] == "dead_nuclei_channel"
    dead = sim.targets[sim.targets["status"] == 1]
    assert len(dead) > 0
    t, track = int(dead["FRAME"].iloc[0]), int(dead["TRACK_ID"].iloc[0])
    inside = sim.labels_targets[t] == track
    death = sim.channel_names.index("dead_nuclei_channel")
    assert sim.movie[t, death][inside].mean() > 500


def test_combined_channel_shows_both_populations(sim):
    combined = sim.movie[:, sim.channel_names.index("brightfield_channel")]
    cells = (sim.labels_targets > 0) | (sim.labels_effectors > 0)
    assert np.mean((combined > 325) == cells) > 0.99
    assert np.mean((combined > 775) == (sim.labels_effectors > 0)) > 0.99


def test_combined_channel_can_be_left_out():
    sim = simulate_interactions(seed=0, combined_channel=False)
    assert sim.channel_names == ["live_nuclei_channel", "effector_fluo_channel"]
    assert sim.movie.shape[1] == 2


def test_experiment_layout(position, sim):
    exp = os.path.dirname(os.path.dirname(os.path.dirname(position)))
    assert os.path.exists(os.path.join(exp, "config.ini"))
    movie = tifffile.imread(os.path.join(position, "movie", "synthetic.tif"))
    assert movie.shape == sim.movie.shape
    assert len(os.listdir(os.path.join(position, "labels_effectors"))) == sim.movie.shape[0]
    for name in ("contacts.csv", "contact_episodes.csv", "labels_targets.tif"):
        assert os.path.exists(os.path.join(position, "ground_truth", name))


def test_contact_neighborhood_matches_ground_truth(position, sim):
    df_targets = compute_contact_neighborhood_at_position(
        position,
        distance=0,
        population=["targets", "effectors"],
        img_shape=SHAPE,
        theta_dist=0,
        return_tables=True,
        clear_neigh=True,
    )[0]
    col = "neighborhood_(targets-effectors)_contact_0_px"
    assert _pairs_in(df_targets, col) == _true_contacts(sim)


def test_contact_tolerance_of_two_keeps_its_value_in_the_column_name(position):
    df_targets = compute_contact_neighborhood_at_position(
        position,
        distance=2,
        population=["targets", "effectors"],
        img_shape=SHAPE,
        theta_dist=0,
        return_tables=True,
        clear_neigh=True,
    )[0]
    assert "neighborhood_(targets-effectors)_contact_2_px" in df_targets.columns


def test_distance_neighborhood_matches_centre_distances(position, sim):
    radius = 30
    df_targets = compute_neighborhood_at_position(
        position,
        distance=radius,
        population=["targets", "effectors"],
        img_shape=SHAPE,
        theta_dist=0,
        return_tables=True,
        clear_neigh=True,
    )[0]
    col = f"neighborhood_(targets-effectors)_circle_{radius}_px"
    t, e = sim.targets, sim.effectors
    merged = t.merge(e, on="FRAME", suffixes=("_t", "_e"))
    d = np.hypot(
        merged["POSITION_X_t"] - merged["POSITION_X_e"],
        merged["POSITION_Y_t"] - merged["POSITION_Y_e"],
    )
    close = merged[d <= radius]
    expected = set(zip(close["TRACK_ID_t"], close["TRACK_ID_e"], close["FRAME"]))
    assert len(expected) > len(_true_contacts(sim))
    assert _pairs_in(df_targets, col) == expected
