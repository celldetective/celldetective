"""
Checks of the attraction tests against synthetic movies whose effectors are
drawn to the targets, blind to them, or drawn to them from a given frame on.
"""

import numpy as np
import pandas as pd
import pytest

from celldetective.attraction import approach_bias, arrival_test, shift_tracks
from celldetective.utils.synthetic_interactions import simulate_interactions

# Blind effectors dock to the targets they bump into, from 21 px between
# centres (target and effector radii plus the simulator's 3 px margin): the
# contact distance must reach that far.
SIZE = (320, 320)
CONTACT = 21
SMALL = dict(shape=SIZE, n_frames=60, n_targets=16, n_effectors=16, contact_durations=(4, 12), capture=True)


def tables(seeds=range(4), **kwargs):
    """Targets and effectors of several simulated positions."""
    targets, effectors = [], []
    for seed in seeds:
        sim = simulate_interactions(seed=seed, **{**SMALL, **kwargs})
        targets.append(sim.targets.assign(position=seed))
        effectors.append(sim.effectors.assign(position=seed))
    return pd.concat(targets, ignore_index=True), pd.concat(effectors, ignore_index=True)


@pytest.fixture(scope="module")
def attracted():
    return tables()


@pytest.fixture(scope="module")
def blind():
    return tables(attraction=0)


def test_blind_effectors_still_meet_targets(blind):
    # The trap the tests guard against: blind effectors make contacts too.
    _, effectors = blind
    assert (effectors["state"] == "docked").sum() > 100
    assert (effectors["state"] != "approach").all()


def test_attraction_zero_without_capture_never_docks():
    sim = simulate_interactions(seed=0, attraction=0)
    assert sim.episodes.empty


def test_attraction_out_of_range_is_refused():
    with pytest.raises(ValueError):
        simulate_interactions(attraction=1.5)


def test_arrival_test_finds_the_attraction(attracted):
    result = arrival_test(*attracted, CONTACT, shape=SIZE, n_shifts=99)
    pooled = result[result["position"] == "all"].iloc[0]
    assert pooled["ratio"] > 2
    assert pooled["p_value"] <= 0.02
    assert (result.loc[result["position"] != "all", "ratio"] > 1.5).all()


def test_arrival_test_clears_blind_effectors(blind):
    result = arrival_test(*blind, CONTACT, shape=SIZE, n_shifts=99)
    pooled = result[result["position"] == "all"].iloc[0]
    assert 0.8 < pooled["ratio"] < 1.25
    assert pooled["p_value"] > 0.05


def test_contact_distance_below_capture_range_inflates_blind_arrivals(blind):
    # Grazing passes end in a docking for real effectors and in nothing for
    # shifted ones: blind effectors look more attracted at 18 px than at 21.
    ratio = {d: arrival_test(*blind, d, shape=SIZE, n_shifts=49).iloc[-1]["ratio"] for d in (18, CONTACT)}
    assert ratio[18] > ratio[CONTACT]


def test_arrival_test_dates_a_recruitment_onset():
    attraction = np.r_[np.zeros(30), np.ones(30)]
    result = arrival_test(*tables(attraction=attraction), CONTACT, shape=SIZE, n_shifts=99, time_bins=[0, 30, 60])
    pooled = result[result["position"] == "all"].set_index("window_start")
    assert pooled.loc[0, "p_value"] > 0.05
    assert pooled.loc[30, "ratio"] > 2
    assert pooled.loc[30, "p_value"] <= 0.02


def test_arrival_test_columns_without_positions(attracted):
    targets, effectors = (df[df["position"] == 0].drop(columns="position") for df in attracted)
    result = arrival_test(targets, effectors, CONTACT, shape=SIZE, n_shifts=9)
    assert len(result) == 1
    assert {"arrivals", "free_steps", "rate", "null_rate", "ratio", "p_value"} <= set(result.columns)


def test_approach_bias_points_to_targets(attracted, blind):
    kwargs = dict(shape=SIZE, bins=[27, 39, 51], border=60, n_shifts=49)
    drawn = approach_bias(*attracted, CONTACT, **kwargs)
    assert (drawn["mean_cos"] > drawn["null_high"]).any()
    assert (drawn["mean_cos"] > 0).all()
    unbiased = approach_bias(*blind, CONTACT, **kwargs)
    assert (unbiased["mean_cos"] - unbiased["null_cos"]).abs().max() < 3 * unbiased["sem"].max()


def test_approach_bias_time_windows(attracted):
    bias = approach_bias(*attracted, CONTACT, shape=SIZE, bins=2, max_distance=45, n_shifts=0, time_bins=2)
    assert len(bias) == 4
    assert set(bias["window_start"]) == {0, 30}
    assert "null_cos" not in bias.columns


def test_shift_tracks_keeps_the_motion(attracted):
    _, effectors = attracted
    shifted = shift_tracks(effectors, SIZE, seed=0)
    original = effectors.sort_values(["position", "TRACK_ID", "FRAME"])
    keys = ["position", "TRACK_ID"]
    moved = shifted.groupby(keys)[["POSITION_X", "POSITION_Y"]].diff()
    raw = original.groupby(keys)[["POSITION_X", "POSITION_Y"]].diff()
    same = ~shifted["shift_seam"].to_numpy() & raw.notna().all(axis=1).to_numpy()
    np.testing.assert_allclose(moved.to_numpy()[same], raw.to_numpy()[same], atol=1e-6)
    assert shifted[["POSITION_X", "POSITION_Y"]].min().min() >= 0
    assert (shifted["POSITION_X"] < SIZE[1]).all() and (shifted["POSITION_Y"] < SIZE[0]).all()
    # One offset per position: the arrangement of the effectors is kept.
    xy = ["POSITION_X", "POSITION_Y"]
    for (_, before), (_, after) in zip(original.groupby("position"), shifted.groupby("position")):
        offset = np.mod(after[xy].to_numpy() - before[xy].to_numpy(), [SIZE[1], SIZE[0]])
        assert np.ptp(offset, axis=0).max() < 1e-4
