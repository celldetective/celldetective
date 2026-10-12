"""
Tests of the contact episodes, of the long table of time-varying contact
exposure and of the Cox model fitted to it, on hand-made tables and on
synthetic movies where the deaths are, or are not, caused by the contacts.
"""

import logging
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from celldetective.hazard import (
    binned_event_fraction,
    condition_covariates,
    encode_conditions,
    fit_hazard_per_neighborhood,
    episode_outcomes,
    fit_hazard,
    neighborhood_distance,
    time_varying_table,
)
from celldetective.relative_measurements import measure_pair_signals_at_position
from celldetective.utils.data_cleaning import collapse_pairs_by_episode
from celldetective.utils.synthetic_interactions import simulate_interactions

from tests import test_interactions as ti

NB = "neighborhood_(targets-effectors)_contact_0_px"


@pytest.fixture(autouse=True)
def quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


def reference_table(counts, t_event=-1, cls=None, track=1, **extra):
    """A one-cell reference table from its per-frame neighbor counts."""
    n = len(counts)
    df = pd.DataFrame(
        {
            "TRACK_ID": track,
            "FRAME": np.arange(n),
            f"inclusive_count_{NB}": counts,
            "t_death": t_event,
            "class_death": (0 if t_event >= 0 else 1) if cls is None else cls,
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Long table
# ---------------------------------------------------------------------------


def test_long_table_rows_and_exposure():
    counts = [0, 1, 2, 0, 1, 1, 0, 0]
    tab = time_varying_table(reference_table(counts, t_event=6), NB, "death", duration_unit="frame")
    # Intervals (0,1] ... (5,6], the event in the last one.
    assert list(tab["start"]) == [0, 1, 2, 3, 4, 5]
    assert list(tab["stop"]) == [1, 2, 3, 4, 5, 6]
    assert list(tab["event"]) == [0, 0, 0, 0, 0, 1]
    assert list(tab["in_contact"]) == [0, 1, 1, 0, 1, 1]
    assert list(tab["n_neighbors"]) == [0, 1, 2, 0, 1, 1]
    assert list(tab["cumulative_contact"]) == [0, 1, 2, 2, 3, 4]
    assert list(tab["ongoing_contact"]) == [0, 1, 2, 0, 1, 2]


def test_long_table_lag_uses_earlier_exposure():
    counts = [1, 0, 0, 0, 1, 0]
    tab = time_varying_table(reference_table(counts), NB, "death", lag=2, duration_unit="frame")
    # Censored at the last frame; the first two frames have no lagged exposure.
    assert list(tab["start"]) == [2, 3, 4]
    assert list(tab["in_contact"]) == [1, 0, 0]
    assert tab["event"].sum() == 0


def test_long_table_units_origin_and_exclusions():
    counts = [0, 1, 1, 1, 0, 0]
    df = reference_table(counts, t_event=4.5, t_firstdetection=2)
    tab = time_varying_table(df, NB, "death", origin="t_firstdetection", frame_to_min=5, duration_unit="min")
    assert list(tab["start"]) == [10, 15, 20]
    assert list(tab["event"]) == [0, 0, 1]  # 4 < 4.5 <= 5
    assert list(tab["cumulative_contact"]) == [10, 15, 15]

    dead_before = reference_table(counts, t_event=0, cls=2)
    assert time_varying_table(dead_before, NB, "death").empty

    missing = reference_table(counts, t_event=5).drop(index=[2, 3])
    tab = time_varying_table(missing, NB, "death", duration_unit="frame")
    assert list(tab["in_contact"]) == [0, 1, 0, 0, 0]  # missing frames: no contact


def test_long_table_carries_extra_covariates_and_keys():
    df = reference_table([0, 1, 1, 0], t_event=-1, position="p1", well="w1",
                         contact_fluo_mean=[np.nan, 5.0, 7.0, np.nan])
    tab = time_varying_table(df, NB, "death", extra_covariates=["contact_fluo_mean"])
    assert list(tab["contact_fluo_mean"]) == [0, 5, 7]
    assert set(tab["cell"]) == {"w1/p1/1"}


def test_long_table_without_neighborhood():
    # A population without neighborhood: no contact covariate, the cell
    # measurements only, and one fit named "none".
    df = reference_table([0, 0, 0, 0], t_event=3, area=[10.0, 20.0, 30.0, 40.0]).drop(
        columns=[f"inclusive_count_{NB}"]
    )
    tab = time_varying_table(df, None, "death", extra_covariates=["area"], duration_unit="frame")
    assert "in_contact" not in tab.columns
    assert list(tab["area"]) == [10, 20, 30]
    assert list(tab["event"]) == [0, 0, 1]
    summary = fit_hazard_per_neighborhood(df, [], "death", ["area"], extra_covariates=["area"])
    assert list(summary["neighborhood"]) == ["none"]


def test_long_table_carries_cell_measurements_forward():
    # A frame where the cell is not detected keeps its last area; frames
    # before its first measurement stay missing.
    df = reference_table([0, 1, 1, 0, 0], t_event=-1, area=[np.nan, 50.0, np.nan, 70.0, 80.0])
    tab = time_varying_table(df, NB, "death", extra_covariates=["area"])
    assert np.isnan(tab["area"].iloc[0])
    assert list(tab["area"].iloc[1:]) == [50, 50, 70]


# ---------------------------------------------------------------------------
# Cox model
# ---------------------------------------------------------------------------


def simulated_long_table(beta, n_cells=300, n_frames=40, seed=0):
    """Cells switching in and out of contact, dying at a per-frame rate
    ``h0 * exp(beta * in_contact)``."""
    rng = np.random.default_rng(seed)
    frames = []
    for cell in range(n_cells):
        counts = (rng.random(n_frames) < 0.3).astype(int)
        t_death = -1
        for t in range(n_frames - 1):
            if rng.random() < 0.01 * np.exp(beta * counts[t]):
                t_death = t + 1
                break
        frames.append(reference_table(counts, t_event=t_death, track=cell))
    return time_varying_table(pd.concat(frames), NB, "death", duration_unit="frame")


@pytest.mark.parametrize("beta", [0.0, np.log(4)])
def test_cox_recovers_the_hazard_ratio(beta):
    tab = simulated_long_table(beta)
    _, summary = fit_hazard(tab, ["in_contact"])
    row = summary.iloc[0]
    assert row["ci_lower"] < np.exp(beta) < row["ci_upper"]
    assert row["n_cells"] == 300
    if beta > 0:
        assert row["ci_lower"] > 1


def test_default_penalty_does_not_grow_with_the_number_of_frames():
    # lifelines multiplies its penalty by the number of rows (cell-frames);
    # the default penalty must still leave a well-supported ratio in place.
    tab = simulated_long_table(np.log(4), n_frames=200)
    _, free = fit_hazard(tab, ["in_contact"])
    _, penalized = fit_hazard(tab, ["in_contact"], penalizer=0.5)
    assert penalized.iloc[0]["ci_lower"] < 4 < penalized.iloc[0]["ci_upper"]
    assert penalized.iloc[0]["hazard_ratio"] == pytest.approx(free.iloc[0]["hazard_ratio"], rel=0.1)


def test_cox_refuses_tables_without_events_or_variation():
    tab = simulated_long_table(0.0, n_cells=20)
    with pytest.raises(ValueError):
        fit_hazard(tab.assign(event=0), ["in_contact"])
    with pytest.raises(ValueError):
        fit_hazard(tab.assign(in_contact=1), ["in_contact"])


def test_cox_per_standard_deviation():
    # Standardized, the hazard ratio of a continuous covariate is per SD (that
    # of the unit raised to the SD); a binary covariate is left as it is.
    tab = simulated_long_table(np.log(4))
    tab["area"] = np.random.default_rng(1).normal(500, 80, len(tab))
    _, per_unit = fit_hazard(tab, ["in_contact", "area"])
    _, per_sd = fit_hazard(tab, ["in_contact", "area"], standardize=True)
    per_unit, per_sd = per_unit.set_index("covariate"), per_sd.set_index("covariate")
    assert per_sd.loc["in_contact", "scale"] == 1
    assert per_sd.loc["area", "scale"] == pytest.approx(tab["area"].std())
    assert per_sd.loc["in_contact", "hazard_ratio"] == pytest.approx(per_unit.loc["in_contact", "hazard_ratio"], rel=1e-4)
    assert per_sd.loc["area", "hazard_ratio"] == pytest.approx(
        per_unit.loc["area", "hazard_ratio"] ** tab["area"].std(), rel=1e-3
    )


def test_cox_leaves_out_intervals_without_a_value():
    tab = simulated_long_table(np.log(4))
    tab["area"] = np.random.default_rng(1).normal(500, 80, len(tab))
    tab.loc[tab.index[:50], "area"] = np.nan
    _, summary = fit_hazard(tab, ["in_contact", "area"])
    assert summary["hazard_ratio"].notna().all()


def test_conditions_of_the_wells_as_covariates():
    # A numeric condition is continuous (on its scale or a log one) or by
    # level; a text one only by level; one that does not vary is left out.
    labels = {"concentrations": [0.0, 0.5, 100.0], "antibodies": ["none", "bsAb", "bsAb"],
              "cell_types": ["T", "T", "T"]}
    options = condition_covariates(labels)
    assert list(options) == ["concentrations", "log10(1 + concentrations)", "concentrations (each vs 0)",
                             "antibodies (each vs none)"]
    reference = pd.DataFrame({"concentrations": [0.0, 0.5, 100.0, 100.0],
                              "antibodies": ["none", "bsAb", "bsAb", "bsAb"], "area": 1.0})
    encoded, names = encode_conditions(reference, ["area", *options], options)
    assert names == ["area", "concentrations", "log10(1 + concentrations)", "concentrations = 0.5",
                     "concentrations = 100", "antibodies = bsAb"]
    assert list(encoded["concentrations = 100"]) == [0, 0, 1, 1]
    assert encoded["log10(1 + concentrations)"].iloc[2] == pytest.approx(np.log10(101))
    assert list(encoded["antibodies = bsAb"]) == [0, 1, 1, 1]


def test_fit_split_by_condition_and_effect():
    # Contact kills in one condition only: the split fits tell which, and the
    # summary reads the effect off the interval.
    tab_rows = []
    for cond, beta in (("A", 0.0), ("B", np.log(6))):
        rng = np.random.default_rng(5 if cond == "A" else 2)  # seed 1 is a 5% false positive
        for cell in range(200):
            counts = (rng.random(40) < 0.3).astype(int)
            t_death = -1
            for t in range(39):
                if rng.random() < 0.02 * np.exp(beta * counts[t]):
                    t_death = t + 1
                    break
            tab_rows.append(reference_table(counts, t_event=t_death, track=cell).assign(position=cond, drug=cond))
    reference = pd.concat(tab_rows)
    summary = fit_hazard_per_neighborhood(reference, [NB], "death", ["in_contact"], penalizer=0.5,
                                          split_by="drug", duration_unit="frame").set_index("condition")
    assert summary.loc["drug = B", "effect"] == "higher hazard"
    assert summary.loc["drug = A", "effect"] == "no clear effect"


def test_cox_with_strata():
    tab = simulated_long_table(np.log(4))
    tab["well"] = np.where(tab["cell"].str[-1].isin(list("01234")), "A", "B")
    _, summary = fit_hazard(tab, ["in_contact"], strata=["well"])
    assert summary.iloc[0]["ci_lower"] > 1


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def pair_rows(status, ref=1, neigh=2, **extra):
    df = pd.DataFrame(
        {
            "REFERENCE_ID": ref,
            "NEIGHBOR_ID": neigh,
            "FRAME": np.arange(len(status)),
            f"status_{NB}": status,
            f"t0_{NB}": 3,
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


def test_episodes_of_a_pair():
    status = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    ep = collapse_pairs_by_episode(pair_rows(status, intersection=np.arange(10.0)), NB)
    assert list(ep["t_start"]) == [1, 5, 7]
    assert list(ep["t_end"]) == [2, 5, 9]
    assert list(ep["duration"]) == [2, 1, 3]
    assert ep["gap_before"].isna().iloc[0] and list(ep["gap_before"].iloc[1:]) == [2, 1]
    assert list(ep["right_censored"]) == [False, False, True]
    assert list(ep["intersection"]) == [1.5, 5.0, 8.0]
    assert f"t0_{NB}" not in ep.columns

    bridged = collapse_pairs_by_episode(pair_rows(status), NB, max_gap=1)
    assert list(bridged["t_start"]) == [1, 5]
    assert list(bridged["duration"]) == [2, 5]
    assert list(bridged["frames_in_contact"]) == [2, 4]

    maxed = collapse_pairs_by_episode(pair_rows(status, intersection=np.arange(10.0)), NB, projection="max")
    assert list(maxed["intersection"]) == [2, 5, 9]


def test_episode_outcomes_alignments():
    reference = pd.concat(
        [
            reference_table([0] * 30, t_event=8, track=1),  # dies during the contact
            reference_table([0] * 30, t_event=15, track=2),  # dies after it
            reference_table([0] * 30, t_event=-1, track=3),  # survives
        ]
    )
    episodes = pd.DataFrame({"REFERENCE_ID": [1, 2, 3], "t_start": [5, 5, 5], "t_end": [10, 10, 10],
                             "duration": [6, 6, 6]})
    start = episode_outcomes(episodes, reference, "death", window=12, align="start")
    assert dict(zip(start["REFERENCE_ID"], start["outcome"])) == {1: 1, 2: 1, 3: 0}
    end = episode_outcomes(episodes, reference, "death", window=12, align="end")
    assert dict(zip(end["REFERENCE_ID"], end["outcome"])) == {2: 1, 3: 0}
    short = episode_outcomes(episodes, reference, "death", window=25, align="end")
    assert list(short["REFERENCE_ID"]) == [2]  # cell 3 is not seen long enough

    binned = binned_event_fraction(start, [0, 10])
    assert binned.iloc[0][["n", "events"]].tolist() == [3, 2]
    assert binned.iloc[0]["ci_lower"] < 2 / 3 < binned.iloc[0]["ci_upper"]


# ---------------------------------------------------------------------------
# Synthetic movies: deaths caused by the contacts, or not
# ---------------------------------------------------------------------------

SHAPE = (320, 320)


def movie_position(folder, **deaths):
    sim = simulate_interactions(shape=SHAPE, n_frames=60, n_targets=16, n_effectors=16, seed=0, **deaths)
    pos = ti.write_position(folder, sim)
    for d in (0, 2):
        ti.compute_contact_neighborhood_at_position(
            pos, distance=d, population=["targets", "effectors"], img_shape=SHAPE, theta_dist=0,
            return_tables=False, clear_neigh=d == 0, neighborhood_kwargs=ti.kwargs(),
        )
    reference = pd.read_csv(os.path.join(pos, "output", "tables", "trajectories_targets.csv"))
    return sim, pos, reference


@pytest.fixture(scope="module")
def causal(tmp_path_factory):
    return movie_position(tmp_path_factory.mktemp("causal") / "Exp", death_probability=0.08)


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    return movie_position(tmp_path_factory.mktemp("null") / "Exp", background_death=0.01)


def test_contact_caused_deaths_raise_the_hazard(causal):
    _, _, reference = causal
    tab = time_varying_table(reference, NB, "death")
    _, summary = fit_hazard(tab, ["in_contact"], penalizer=0.5)
    assert summary.iloc[0]["ci_lower"] > 1
    assert summary.iloc[0]["n_events"] >= 8


def test_contact_independent_deaths_do_not(null):
    _, _, reference = null
    tab = time_varying_table(reference, NB, "death")
    _, summary = fit_hazard(tab, ["in_contact"], penalizer=0.5)
    assert summary.iloc[0]["ci_lower"] < 1 < summary.iloc[0]["ci_upper"]


def test_sensitivity_over_contact_distances(causal):
    _, _, reference = causal
    neighborhoods = [NB, NB.replace("_0_px", "_2_px"), NB.replace("_0_px", "_9_px")]
    result = fit_hazard_per_neighborhood(reference, neighborhoods, "death", ["in_contact"], penalizer=0.5)
    fitted = result.dropna(subset=["hazard_ratio"])
    assert list(fitted["distance"]) == [0, 2]
    assert (fitted["ci_lower"] > 1).all()
    assert result.loc[result["distance"] == 9, "error"].notna().all()
    assert neighborhood_distance("neighborhood_self_contact_12.5_px") == 12.5


def test_episodes_from_the_pair_table_match_the_simulation(causal):
    sim, pos, reference = causal
    pairs = measure_pair_signals_at_position(
        pos, {"reference": "targets", "neighbor": "effectors", "type": "contact", "distance": 0,
              "description": NB},
    )
    ep = collapse_pairs_by_episode(pairs, NB)
    found = set(zip(ep["REFERENCE_ID"], ep["NEIGHBOR_ID"], ep["t_start"], ep["t_end"]))
    truth = sim.episodes
    expected = set(zip(truth["TARGET_ID"] + ti.T_OFF, truth["EFFECTOR_ID"] + ti.E_OFF,
                       truth["t_start"], truth["t_end"]))
    assert found == expected
    last = len(sim.movie) - 1
    assert (ep["right_censored"] == (ep["t_end"] == last)).all()
    # The mean overlap of an episode is that of the simulation.
    c = sim.contacts.assign(REFERENCE_ID=sim.contacts["TARGET_ID"] + ti.T_OFF,
                            NEIGHBOR_ID=sim.contacts["EFFECTOR_ID"] + ti.E_OFF)
    for row in ep.itertuples(index=False):
        sel = c[(c["REFERENCE_ID"] == row.REFERENCE_ID) & (c["NEIGHBOR_ID"] == row.NEIGHBOR_ID)
                & (c["FRAME"] >= row.t_start) & (c["FRAME"] <= row.t_end)]
        assert row.intersection == pytest.approx(sel["overlap"].mean())

    outcomes = episode_outcomes(ep, reference, "death", window=10, align="start")
    assert outcomes["outcome"].sum() > 0
