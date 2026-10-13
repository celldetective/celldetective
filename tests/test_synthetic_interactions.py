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


def test_multiwell_experiment_carries_the_conditions_of_its_wells(tmp_path, sim):
    from celldetective.utils.data_loaders import load_experiment_tables
    from celldetective.utils.experiment import get_experiment_labels

    exp = str(tmp_path / "Dose")
    other = simulate_interactions(seed=1)
    positions = write_experiment(
        exp, [[sim, other], [other]], write_tables=True,
        well_labels={"antibodies": ["none", "bsAb"], "concentrations": [0, 0.5], "et_ratio": [1, 4]},
    )
    assert [[os.path.basename(os.path.normpath(p)) for p in w] for w in positions] == [["100", "101"], ["200"]]
    labels = get_experiment_labels(exp)
    assert labels["concentrations"] == [0.0, 0.5]  # decimals are numbers too
    assert labels["et_ratio"] == [1.0, 4.0]
    assert labels["pharmaceutical_agents"] == ["none", "none"]
    targets = load_experiment_tables(exp + os.sep, population="targets")
    assert targets.groupby("concentrations")["position"].nunique().to_dict() == {0.0: 2, 0.5: 1}
    assert set(targets.loc[targets["concentrations"] == 0.5, "antibodies"]) == {"bsAb"}
    assert set(targets.loc[targets["concentrations"] == 0.5, "et_ratio"]) == {4.0}


def test_multiwell_experiment_refuses_inconsistent_inputs(tmp_path, sim):
    with pytest.raises(ValueError):
        write_experiment(str(tmp_path / "a"), [[sim], [sim]], well_labels={"concentrations": [0]})
    with pytest.raises(ValueError):
        write_experiment(str(tmp_path / "b"), [[sim]], well_labels={"agent": ["a,b"]})
    with pytest.raises(ValueError):
        write_experiment(str(tmp_path / "c"), [[sim], [simulate_interactions(shape=(100, 100))]])


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


@pytest.mark.parametrize("delay", [0, 4])
def test_a_lethal_hit_is_seen_after_the_delay(delay):
    sim = simulate_interactions(seed=0, death_probability=0.1, death_delay=delay)
    per_target = sim.targets.groupby("TRACK_ID")[["t_death", "t_lethal_hit"]].first()
    killed = per_target[per_target["t_lethal_hit"] >= 0]
    assert len(killed) > 0
    np.testing.assert_array_equal(killed["t_death"], killed["t_lethal_hit"] + 1 + delay)
    contacts = set(zip(sim.contacts["TARGET_ID"], sim.contacts["FRAME"]))
    death = sim.channel_names.index("dead_nuclei_channel")
    for track, row in killed.iterrows():
        # The hit is decided during a contact; the target looks alive until its death.
        assert (track, row["t_lethal_hit"]) in contacts
        for t in range(row["t_lethal_hit"], row["t_death"]):
            assert sim.movie[t, death][sim.labels_targets[t] == track].mean() < 500
        assert sim.movie[row["t_death"], death][sim.labels_targets[row["t_death"]] == track].mean() > 500


def test_the_delay_applies_to_deaths_after_a_contact_time():
    sim = simulate_interactions(seed=0, death_after=4, death_delay=3)
    per_target = sim.targets.groupby("TRACK_ID")[["t_death", "t_lethal_hit"]].first()
    killed = per_target[per_target["t_lethal_hit"] >= 0]
    assert len(killed) > 0
    np.testing.assert_array_equal(killed["t_death"], killed["t_lethal_hit"] + 3)
    counts = sim.contacts.groupby("TARGET_ID")["FRAME"]
    for track, row in killed.iterrows():
        # The hit is the frame of the 4th frame in contact.
        assert sorted(counts.get_group(track))[3] == row["t_lethal_hit"]


def test_no_delay_gives_the_same_movie():
    a = simulate_interactions(seed=2, death_probability=0.05)
    b = simulate_interactions(seed=2, death_probability=0.05, death_delay=0)
    np.testing.assert_array_equal(a.movie, b.movie)
    assert (a.targets["t_lethal_hit"] >= -1).all()
    # Contacts never depend on the deaths.
    c = simulate_interactions(seed=2, death_probability=0.05, death_delay=5)
    pd.testing.assert_frame_equal(a.contacts, c.contacts)


# ---------------------------------------------------------------------------
# Killing mechanisms
# ---------------------------------------------------------------------------

DENSE = dict(shape=(320, 320), n_frames=60, n_targets=16, n_effectors=32, contact_durations=(4, 12))


def per_target(sim):
    cols = ["t_death", "t_lethal_hit", "death_cause", "killer", "resistant"]
    return sim.targets.groupby("TRACK_ID")[cols].first()


def docked_since(sim):
    """{(target, effector, frame): frame the docking started} for every docked frame."""
    out = {}
    for ep in sim.episodes.itertuples():
        for f in range(ep.t_start, ep.t_end + 1):
            out[(ep.TARGET_ID, ep.EFFECTOR_ID, f)] = ep.t_start
    return out


def test_mechanisms_needing_per_effector_hits_refuse_without_it():
    with pytest.raises(ValueError):
        simulate_interactions(death_probability=0.1, cooperativity=2)
    with pytest.raises(ValueError):
        simulate_interactions(death_probability=0.1, lethality_per_kill=0.5)


@pytest.mark.parametrize("per_effector_hits", [False, True])
def test_the_killer_is_docked_to_the_target_at_the_hit(per_effector_hits):
    sim = simulate_interactions(seed=0, death_probability=0.1, per_effector_hits=per_effector_hits, **DENSE)
    killed = per_target(sim).query("death_cause == 'contact'")
    assert len(killed) > 3
    docked = docked_since(sim)
    for track, row in killed.iterrows():
        assert (track, row["killer"], row["t_lethal_hit"]) in docked


def test_contact_maturation():
    sim = simulate_interactions(seed=0, death_probability=0.2, contact_maturation=3, **DENSE)
    killed = per_target(sim).query("death_cause == 'contact'")
    assert len(killed) > 3
    docked = docked_since(sim)
    for track, row in killed.iterrows():
        assert row["t_lethal_hit"] - docked[(track, row["killer"], row["t_lethal_hit"])] >= 3


def test_lethality_changes_with_each_kill():
    sim = simulate_interactions(seed=0, death_probability=0.1, per_effector_hits=True, lethality_per_kill=0.5, **DENSE)
    kills = per_target(sim).query("killer > 0")["killer"].value_counts()
    last = sim.effectors[sim.effectors["FRAME"] == sim.effectors["FRAME"].max()].set_index("TRACK_ID")
    assert kills.max() >= 2
    for effector, row in last.iterrows():
        assert row["kills"] == kills.get(effector, 0)
    np.testing.assert_allclose(sim.effectors["lethality"], 0.5 ** sim.effectors["kills"])


def test_cooperativity_raises_the_hits_of_crowded_targets():
    """With strong synergy, a larger share of the kills happens with several
    effectors docked than without."""

    def crowded_share(coop):
        sim = simulate_interactions(seed=0, death_probability=0.02, per_effector_hits=True, cooperativity=coop, **DENSE)
        n_docked = sim.contacts.groupby(["TARGET_ID", "FRAME"]).size()
        killed = per_target(sim).query("death_cause == 'contact'")
        return np.mean([n_docked[(t, f)] > 1 for t, f in zip(killed.index, killed["t_lethal_hit"])])

    assert crowded_share(8.0) > crowded_share(1.0)


def test_resistant_targets_only_die_of_the_background():
    sim = simulate_interactions(
        seed=0, death_probability=0.3, background_death=0.01, resistant_fraction=0.25, **DENSE
    )
    t = per_target(sim)
    assert t["resistant"].sum() == 4
    assert set(t.loc[t["resistant"], "death_cause"]) <= {"", "background"}
    assert (t.loc[~t["resistant"], "death_cause"] == "contact").sum() > 3


def test_bystander_deaths_follow_a_dead_neighbor():
    radius = 90
    sim = simulate_interactions(
        seed=0, death_probability=0.05, bystander_death=0.1, bystander_radius=radius, **DENSE
    )
    t = per_target(sim)
    bystanders = t[t["death_cause"] == "bystander"]
    assert len(bystanders) > 0
    pos = sim.targets.set_index(["TRACK_ID", "FRAME"])[["POSITION_X", "POSITION_Y"]]
    for track, row in bystanders.iterrows():
        f = row["t_death"] - 1  # decided the frame before it is seen dead
        dead_then = t[(t["t_death"] >= 0) & (t["t_death"] <= f)].index
        d = [np.hypot(*(pos.loc[(track, f)] - pos.loc[(o, f)])) for o in dead_then]
        assert min(d) <= radius
        assert row["t_lethal_hit"] == -1 and row["killer"] == -1


def test_recruitment_brings_effectors_near_dead_targets():
    def contacts_near_dead(recruitment):
        sim = simulate_interactions(seed=0, death_probability=0.05, recruitment=recruitment, bystander_radius=90, **DENSE)
        t_death = per_target(sim)["t_death"]
        pos = sim.targets.set_index(["FRAME", "TRACK_ID"])[["POSITION_X", "POSITION_Y"]]
        near = []
        for row in sim.contacts.itertuples():
            dead = [o for o, td in t_death.items() if 0 <= td <= row.FRAME and o != row.TARGET_ID]
            here = pos.loc[(row.FRAME, row.TARGET_ID)].to_numpy()
            near.append(any(np.hypot(*(here - pos.loc[(row.FRAME, o)].to_numpy())) <= 90 for o in dead))
        return np.mean(near)

    assert contacts_near_dead(5.0) > contacts_near_dead(0.0)


def test_a_delay_range_draws_each_delay():
    sim = simulate_interactions(seed=0, death_probability=0.2, death_delay=(2, 6), **DENSE)
    killed = per_target(sim).query("death_cause == 'contact'")
    delays = killed["t_death"] - killed["t_lethal_hit"] - 1
    assert delays.between(2, 6).all()
    assert delays.nunique() > 1
