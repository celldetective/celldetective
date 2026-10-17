"""
Regression tests of the interaction features against the exact ground truth
of the synthetic effector-target movie: distance and mask-contact
neighborhoods, their counts, cumulated presence, status decomposition and
event averages, pair measurements, contact-site intensities and the
neighborhood process run by the GUI.

The tables are given track IDs distinct from the mask labels (targets
``label + 100``, effectors ``label + 200``), so that mixing up the two fails.
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

from celldetective.neighborhood import (
    compute_contact_neighborhood_at_position,
    compute_neighborhood_at_position,
    extract_neighborhood_in_pair_table,
)
from celldetective.relative_measurements import (
    expand_pair_table,
    extract_neighborhoods_from_pickles,
    measure_pair_signals_at_position,
)
from celldetective.utils.synthetic_interactions import (
    simulate_interactions,
    write_experiment,
)

SHAPE = (160, 160)
T_OFF, E_OFF = 100, 200
CONTACT_0 = "neighborhood_(targets-effectors)_contact_0_px"
CIRCLE_30 = "neighborhood_(targets-effectors)_circle_30_px"


# ---------------------------------------------------------------------------
# Experiments and ground truth
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    return simulate_interactions(seed=0)


@pytest.fixture(scope="module")
def sim_death():
    return simulate_interactions(seed=0, death_after=4)


def write_position(folder, sim):
    """A one-position experiment with the true masks and trajectories, whose
    track IDs differ from the mask labels and with the mask areas."""
    pos = write_experiment(str(folder), sim, write_labels=True, write_tables=True)
    for population, labels, offset in (
        ("targets", sim.labels_targets, T_OFF),
        ("effectors", sim.labels_effectors, E_OFF),
    ):
        path = os.path.join(pos, "output", "tables", f"trajectories_{population}.csv")
        df = pd.read_csv(path)
        df["TRACK_ID"] = df["class_id"] + offset
        df["area"] = [
            int(np.sum(labels[f] == c)) for f, c in zip(df["FRAME"], df["class_id"])
        ]
        df.to_csv(path, index=False)
    return pos


@pytest.fixture
def position(tmp_path, sim):
    return write_position(tmp_path / "SynthExp", sim)


def kwargs(**overrides):
    """Fresh neighborhood options: the functions write into the dict they get."""
    options = {
        "mode": "two-pop",
        "status": None,
        "not_status_option": None,
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": True,
    }
    options.update(overrides)
    return options


def contact(pos, distance=0, population=("targets", "effectors"), **options):
    extra = {}
    for key in ("event_time_col", "channel_names"):
        if key in options:
            extra[key] = options.pop(key)
    return compute_contact_neighborhood_at_position(
        pos,
        distance=distance,
        population=list(population),
        img_shape=SHAPE,
        theta_dist=0,
        return_tables=True,
        clear_neigh=options.pop("clear_neigh", True),
        neighborhood_kwargs=kwargs(**options),
        **extra,
    )


def circle(pos, distance=30, population=("targets", "effectors"), **options):
    return compute_neighborhood_at_position(
        pos,
        distance=distance,
        population=list(population),
        img_shape=SHAPE,
        theta_dist=0,
        return_tables=True,
        clear_neigh=options.pop("clear_neigh", True),
        neighborhood_kwargs=kwargs(**options),
    )


def neighbor_entries(df, col):
    """{(TRACK_ID, neighbor id, FRAME): neighbor dict} listed in a neighborhood column."""
    found = {}
    for track, frame, neighbors in zip(df["TRACK_ID"], df["FRAME"], df[col]):
        if isinstance(neighbors, (list, np.ndarray)):
            for n in neighbors:
                found[(int(track), int(n["id"]), int(frame))] = n
    return found


def true_contacts(sim):
    c = sim.contacts
    return set(zip(c["TARGET_ID"] + T_OFF, c["EFFECTOR_ID"] + E_OFF, c["FRAME"]))


def reference_contacts(labels_a, labels_b, d, offset_a, offset_b):
    """Pairs whose masks, each grown outward by ``d`` px, touch or overlap
    (8-connectivity), computed independently of the library."""
    same = labels_b is labels_a
    found = set()
    for f in range(len(labels_a)):

        def grown(lbl):
            return {
                int(c): distance_transform_edt(lbl != c) <= d
                for c in np.unique(lbl)
                if c > 0
            }

        grown_a = grown(labels_a[f])
        grown_b = grown_a if same else grown(labels_b[f])
        for a, ga in grown_a.items():
            touching = binary_dilation(ga, structure=np.ones((3, 3)))
            for b, gb in grown_b.items():
                if same and a == b:
                    continue
                if np.any(touching & gb):
                    found.add((a + offset_a, b + offset_b, f))
    return found


def centre_distances(sim, a="targets", b="effectors"):
    """Every pair of cells at every frame with the distance between centres."""
    offsets = {"targets": T_OFF, "effectors": E_OFF}
    tab = {"targets": sim.targets, "effectors": sim.effectors}
    m = tab[a].merge(tab[b], on="FRAME", suffixes=("_a", "_b"))
    m["ID_a"] = m["TRACK_ID_a"] + offsets[a]
    m["ID_b"] = m["TRACK_ID_b"] + offsets[b]
    m["dx"] = m["POSITION_X_b"] - m["POSITION_X_a"]
    m["dy"] = m["POSITION_Y_b"] - m["POSITION_Y_a"]
    m["d"] = np.hypot(m["dx"], m["dy"])
    return m[m["ID_a"] != m["ID_b"]]


def running_count(pairs):
    """For each (a, b, frame) of a set of pairs, how many frames up to this
    one the pair was found."""
    counts, out = {}, {}
    for a, b, f in sorted(pairs, key=lambda p: p[2]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
        out[(a, b, f)] = counts[(a, b)]
    return out


# ---------------------------------------------------------------------------
# Mask-contact neighborhood
# ---------------------------------------------------------------------------


def test_contact_pairs_use_track_ids_not_mask_labels(position, sim):
    df = contact(position)[0]
    assert set(neighbor_entries(df, CONTACT_0)) == true_contacts(sim)


def test_contact_from_the_effectors_side_is_the_transpose(position, sim):
    df = contact(position, population=("effectors", "targets"))[0]
    col = "neighborhood_(effectors-targets)_contact_0_px"
    found = set(neighbor_entries(df, col))
    assert found == {(e, t, f) for t, e, f in true_contacts(sim)}


@pytest.mark.parametrize("d", [1, 2, 3])
def test_contact_distance_grows_both_masks_outward(position, sim, d):
    df = contact(position, distance=d)[0]
    col = f"neighborhood_(targets-effectors)_contact_{d}_px"
    found = set(neighbor_entries(df, col))
    expected = reference_contacts(
        sim.labels_targets, sim.labels_effectors, d, T_OFF, E_OFF
    )
    assert true_contacts(sim) < expected
    assert found == expected


def test_contact_neighbor_carries_overlap_and_centre_distance(position, sim):
    entries = neighbor_entries(contact(position)[0], CONTACT_0)
    for row in sim.contacts.itertuples():
        n = entries[(row.TARGET_ID + T_OFF, row.EFFECTOR_ID + E_OFF, row.FRAME)]
        assert n["intersection"] == row.overlap
        assert n["distance"] == pytest.approx(row.distance, abs=0.01)


def test_contact_counts(position, sim):
    df = contact(position)[0]
    c = sim.contacts
    n_targets_of = c.groupby(["FRAME", "EFFECTOR_ID"])["TARGET_ID"].transform("size")
    c = c.assign(weight=1.0 / n_targets_of)
    inclusive = c.groupby(["TARGET_ID", "FRAME"]).size()
    intermediate = c.groupby(["TARGET_ID", "FRAME"])["weight"].sum()

    for got in df.to_dict("records"):
        key = (got["TRACK_ID"] - T_OFF, got["FRAME"])
        assert got[f"inclusive_count_{CONTACT_0}"] == inclusive.get(key, 0)
        assert got[f"intermediate_count_{CONTACT_0}"] == pytest.approx(
            intermediate.get(key, 0.0)
        )

    mean = df.groupby("TRACK_ID")[f"inclusive_count_{CONTACT_0}"].mean()
    stored = df.groupby("TRACK_ID")[f"mean_count_inclusive_{CONTACT_0}"].first()
    pd.testing.assert_series_equal(stored, mean, check_names=False)


@pytest.mark.parametrize("symmetrize", [True, False])
def test_contact_pairs_do_not_depend_on_symmetrize(position, sim, symmetrize):
    df = contact(position, symmetrize=symmetrize)[0]
    assert set(neighbor_entries(df, CONTACT_0)) == true_contacts(sim)


def test_contact_cumulated_presence(position, sim):
    df = contact(position, compute_cum_sum=True)[0]
    entries = neighbor_entries(df, CONTACT_0)
    expected = running_count(true_contacts(sim))
    assert {k: n["cumulated_presence"] for k, n in entries.items()} == expected

    # Weighted: each frame counts 1 / (number of targets the effector touches).
    c = sim.contacts
    w = 1.0 / c.groupby(["FRAME", "EFFECTOR_ID"])["TARGET_ID"].transform("size")
    weights = {
        (t + T_OFF, e + E_OFF, f): x
        for t, e, f, x in zip(c["TARGET_ID"], c["EFFECTOR_ID"], c["FRAME"], w)
    }
    for (t, e, f), n in entries.items():
        past = sum(x for (a, b, g), x in weights.items() if (a, b) == (t, e) and g <= f)
        assert n["cumulated_presence_weighted"] == pytest.approx(past)


def test_self_contact(position, sim):
    d = 2
    df = contact(position, distance=d, population=("effectors", "effectors"))[0]
    found = set(neighbor_entries(df, f"neighborhood_self_contact_{d}_px"))
    expected = reference_contacts(
        sim.labels_effectors, sim.labels_effectors, d, E_OFF, E_OFF
    )
    assert len(expected) > 0
    assert found == expected
    assert found == {(b, a, f) for a, b, f in found}


def test_contact_status_counts_split_live_and_dead_neighbors(tmp_path, sim_death):
    pos = write_position(tmp_path / "DeathExp", sim_death)
    df = contact(
        pos,
        population=("effectors", "targets"),
        status=[None, "status"],
        not_status_option=[False, True],
        compute_cum_sum=True,
    )[0]
    col = "neighborhood_(effectors-targets)_contact_0_px"
    # The boolean status makes the cell IDs Python objects; presence still counts.
    pairs = {(e, t, f) for t, e, f in true_contacts(sim_death)}
    presence = {k: n["cumulated_presence"] for k, n in neighbor_entries(df, col).items()}
    assert presence == running_count(pairs)
    t_death = sim_death.targets.groupby("TRACK_ID")["t_death"].first()
    alive, dead = {}, {}
    for row in sim_death.contacts.itertuples():
        td = t_death[row.TARGET_ID]
        key = (row.EFFECTOR_ID + E_OFF, row.FRAME)
        is_dead = 0 <= td <= row.FRAME
        (dead if is_dead else alive)[key] = (dead if is_dead else alive).get(key, 0) + 1
    assert len(dead) > 0 and len(alive) > 0

    for track, frame, s1, s0 in zip(
        df["TRACK_ID"],
        df["FRAME"],
        df[f"inclusive_count_s1_{col}"],
        df[f"inclusive_count_s0_{col}"],
    ):
        assert s1 == alive.get((track, frame), 0)
        assert s0 == dead.get((track, frame), 0)


def test_contact_means_before_and_after_the_event(tmp_path, sim_death):
    pos = write_position(tmp_path / "DeathExp", sim_death)
    df = contact(pos, event_time_col="t_death")[0]
    count = f"inclusive_count_{CONTACT_0}"
    before = f"mean_count_inclusive_{CONTACT_0}_before_event"
    after = f"mean_count_inclusive_{CONTACT_0}_after_event"
    n_dead = 0
    for track, group in df.groupby("TRACK_ID"):
        td = group["t_death"].iloc[0]
        if td < 0:
            assert group[before].iloc[0] == pytest.approx(group[count].mean())
            assert group[after].isnull().all()
            continue
        n_dead += 1
        assert group[before].iloc[0] == pytest.approx(
            group.loc[group["FRAME"] <= td, count].mean()
        )
        if (group["FRAME"] > td).any():
            assert group[after].iloc[0] == pytest.approx(
                group.loc[group["FRAME"] > td, count].mean()
            )
    assert n_dead > 0


def test_same_options_give_the_same_result_twice(tmp_path, sim_death):
    pos = write_position(tmp_path / "DeathExp", sim_death)
    options = kwargs(status=[None, "status"], not_status_option=[False, True])
    col = "inclusive_count_s1_neighborhood_(effectors-targets)_contact_0_px"
    runs = []
    for _ in range(2):
        df = compute_contact_neighborhood_at_position(
            pos,
            distance=0,
            population=["effectors", "targets"],
            img_shape=SHAPE,
            theta_dist=0,
            return_tables=True,
            clear_neigh=True,
            neighborhood_kwargs=options,
        )[0]
        runs.append(df.sort_values(["TRACK_ID", "FRAME"])[col].to_numpy())
    np.testing.assert_array_equal(runs[0], runs[1])


def test_contact_site_intensity_in_the_neighborhood(position, sim):
    df = contact(position, channel_names=sim.channel_names)[0]
    mean_col = "contact_live_nuclei_channel_mean"
    assert mean_col in df.columns
    touching = {(t, f) for t, _, f in true_contacts(sim)}
    in_contact = [
        (t, f) in touching for t, f in zip(df["TRACK_ID"], df["FRAME"])
    ]
    in_contact = np.array(in_contact)
    # The zone lies in the target mask, uniformly bright in the target channel.
    assert np.all(np.abs(df.loc[in_contact, mean_col] - 1000) < 20)
    assert df.loc[~in_contact, mean_col].isnull().all()
    # It holds the pixels the effector overlaps, bright in the effector channel.
    assert np.all(df.loc[in_contact, "contact_effector_fluo_channel_max"] > 800)


def test_neighborhoods_accumulate_and_are_found_back(position):
    contact(position)
    contact(position, distance=2, clear_neigh=False)
    circle(position, distance=30, clear_neigh=False)
    protocols = extract_neighborhoods_from_pickles(position, populations=["targets"])
    found = {(p["type"], p["distance"]) for p in protocols}
    assert found == {("contact", 0.0), ("contact", 2.0), ("circle", 30.0)}
    for p in protocols:
        assert (p["reference"], p["neighbor"]) == ("targets", "effectors")


# ---------------------------------------------------------------------------
# Distance neighborhood
# ---------------------------------------------------------------------------


def test_circle_counts_and_closest(position, sim):
    r = 30
    df = circle(position, distance=r, compute_cum_sum=True)[0]
    entries = neighbor_entries(df, CIRCLE_30)

    m = centre_distances(sim)
    close = m[m["d"] <= r]
    expected = set(zip(close["ID_a"], close["ID_b"], close["FRAME"]))
    assert set(entries) == expected
    for a, b, f, d in zip(close["ID_a"], close["ID_b"], close["FRAME"], close["d"]):
        assert entries[(a, b, f)]["distance"] == pytest.approx(d, abs=0.01)

    # Closest: the effector's nearest target, whatever the radius.
    nearest = m.loc[m.groupby(["FRAME", "ID_b"])["d"].idxmin()]
    nearest = set(zip(nearest["ID_a"], nearest["ID_b"], nearest["FRAME"]))
    for key, n in entries.items():
        assert n["closest"] == (key in nearest)

    n_targets = close.groupby(["FRAME", "ID_b"]).size()
    exclusive = {}
    intermediate = {}
    for a, b, f in expected:
        intermediate[(a, f)] = intermediate.get((a, f), 0) + 1 / n_targets[(f, b)]
        exclusive[(a, f)] = exclusive.get((a, f), 0) + ((a, b, f) in nearest)
    for track, frame, ex, im in zip(
        df["TRACK_ID"],
        df["FRAME"],
        df[f"exclusive_count_{CIRCLE_30}"],
        df[f"intermediate_count_{CIRCLE_30}"],
    ):
        assert ex == exclusive.get((track, frame), 0)
        assert im == pytest.approx(intermediate.get((track, frame), 0))

    presence = running_count(expected)
    assert {k: n["cumulated_presence"] for k, n in entries.items()} == presence


def test_self_circle(position, sim):
    r = 25
    df = circle(position, distance=r, population=("effectors", "effectors"))[0]
    found = set(neighbor_entries(df, f"neighborhood_self_circle_{r}_px"))
    m = centre_distances(sim, "effectors", "effectors")
    close = m[m["d"] <= r]
    expected = set(zip(close["ID_a"], close["ID_b"], close["FRAME"]))
    assert len(expected) > 0
    assert found == expected


# ---------------------------------------------------------------------------
# Pair measurements
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contact_position(tmp_path_factory, sim):
    pos = write_position(tmp_path_factory.mktemp("pairs") / "SynthExp", sim)
    contact(pos)
    return pos


def pair_protocol(**extra):
    protocol = {
        "reference": "targets",
        "neighbor": "effectors",
        "type": "contact",
        "distance": 0,
        "description": CONTACT_0,
    }
    protocol.update(extra)
    return protocol


@pytest.fixture(scope="module")
def pairs(contact_position):
    return measure_pair_signals_at_position(contact_position, pair_protocol())


def in_contact(pairs):
    rows = pairs[pairs[f"status_{CONTACT_0}"] == 1]
    return set(zip(rows["REFERENCE_ID"], rows["NEIGHBOR_ID"], rows["FRAME"]))


def test_pairs_status_follows_the_contacts(pairs, sim):
    truth = true_contacts(sim)
    assert in_contact(pairs) == truth
    met = {(a, b) for a, b, _ in truth}
    assert set(zip(pairs["REFERENCE_ID"], pairs["NEIGHBOR_ID"])) == met
    # Every pair that met is followed over the whole movie.
    assert (pairs.groupby(["REFERENCE_ID", "NEIGHBOR_ID"]).size() == len(sim.movie)).all()


def test_pairs_geometry(pairs, sim):
    m = centre_distances(sim).set_index(["ID_a", "ID_b", "FRAME"])
    for row in pairs.itertuples(index=False):
        truth = m.loc[(row.REFERENCE_ID, row.NEIGHBOR_ID, row.FRAME)]
        assert row.distance == pytest.approx(truth["d"])
        angle = np.deg2rad(row.angle)
        assert np.cos(angle) == pytest.approx(truth["dx"] / truth["d"], abs=1e-6)
        assert np.sin(angle) == pytest.approx(truth["dy"] / truth["d"], abs=1e-6)

    # Docked effectors sit at a fixed distance from the target centre.
    docked = pairs[pairs[f"status_{CONTACT_0}"] == 1]
    contact_d = sim.params["target_radius"] + sim.params["effector_radius"] - 2
    np.testing.assert_allclose(docked["distance"], contact_d)


def test_pairs_velocity_is_zero_while_docked(pairs, sim):
    p = pairs.set_index(["REFERENCE_ID", "NEIGHBOR_ID", "FRAME"])
    checked = 0
    for ep in sim.episodes.itertuples():
        for f in range(ep.t_start + 2, ep.t_end - 1):
            key = (ep.TARGET_ID + T_OFF, ep.EFFECTOR_ID + E_OFF, f)
            assert p.loc[key, "velocity"] == pytest.approx(0, abs=1e-6)
            checked += 1
    assert checked > 0


def test_pairs_residence_time_and_first_entrance(pairs, sim):
    truth = true_contacts(sim)
    contact_frames = {}
    for a, b, f in truth:
        contact_frames.setdefault((a, b), []).append(f)
    col_res = f"residence_time_in_{CONTACT_0}"
    col_t0 = f"t0_{CONTACT_0}"
    for (a, b), group in pairs.groupby(["REFERENCE_ID", "NEIGHBOR_ID"]):
        frames = np.array(sorted(contact_frames[(a, b)]))
        expected = [np.sum(frames <= f) for f in group["FRAME"]]
        assert list(group[col_res]) == expected
        assert (group[col_t0] == frames.min()).all()


def test_pairs_intersection(pairs, sim):
    overlap = {
        (t + T_OFF, e + E_OFF, f): o
        for t, e, f, o in zip(
            sim.contacts["TARGET_ID"],
            sim.contacts["EFFECTOR_ID"],
            sim.contacts["FRAME"],
            sim.contacts["overlap"],
        )
    }
    area_t = sim.targets.assign(
        area=[
            np.sum(sim.labels_targets[f] == c)
            for f, c in zip(sim.targets["FRAME"], sim.targets["class_id"])
        ]
    ).set_index(["TRACK_ID", "FRAME"])["area"]
    for row in pairs.itertuples(index=False):
        key = (row.REFERENCE_ID, row.NEIGHBOR_ID, row.FRAME)
        if key in overlap:
            assert row.intersection == overlap[key]
            area = area_t[(row.REFERENCE_ID - T_OFF, row.FRAME)]
            assert row.reference_frac_area_intersection == pytest.approx(
                overlap[key] / area
            )
        else:
            assert np.isnan(row.intersection)


def test_pairs_contact_site_intensity(contact_position, sim):
    df = measure_pair_signals_at_position(
        contact_position,
        pair_protocol(channel_names=sim.channel_names, contact_border=0),
    )
    touching = df[f"status_{CONTACT_0}"] == 1
    mean = df["contact_live_nuclei_channel_mean"]
    assert np.all(np.abs(mean[touching] - 1000) < 20)
    assert mean[~touching].isnull().all()
    assert np.all(df.loc[touching, "contact_effector_fluo_channel_max"] > 800)


def test_pairs_from_neighbor_arrays(tmp_path, contact_position, sim):
    """Neighbor lists stored as arrays, as a pickle may give them back."""
    pos = str(tmp_path / "W1" / "100") + os.sep
    shutil.copytree(contact_position, pos)
    path = os.path.join(pos, "output", "tables", "trajectories_targets.pkl")
    df = pd.read_pickle(path)
    df[CONTACT_0] = [
        np.array(n, dtype=object) if isinstance(n, list) else n for n in df[CONTACT_0]
    ]
    df.to_pickle(path)
    assert in_contact(measure_pair_signals_at_position(pos, pair_protocol())) == true_contacts(sim)


def test_untracked_pairs(tmp_path, contact_position, sim):
    """Without tracks, the pairs are measured frame by frame from the cell IDs."""
    pos = str(tmp_path / "W1" / "100") + os.sep
    shutil.copytree(contact_position, pos)
    for population in ("targets", "effectors"):
        path = os.path.join(pos, "output", "tables", f"trajectories_{population}.pkl")
        df = pd.read_pickle(path).rename(columns={"TRACK_ID": "ID"})
        if population == "targets":
            df[CONTACT_0] = [
                np.array(n, dtype=object) if isinstance(n, list) else n
                for n in df[CONTACT_0]
            ]
        df.to_pickle(path)
    pairs = measure_pair_signals_at_position(pos, pair_protocol())
    assert in_contact(pairs) == true_contacts(sim)


def test_extract_neighborhood_in_pair_table(pairs, sim):
    kept = extract_neighborhood_in_pair_table(
        pairs, distance=0, mode="contact", contact_only=True
    )
    assert set(zip(kept["REFERENCE_ID"], kept["NEIGHBOR_ID"], kept["FRAME"])) == true_contacts(sim)
    by_key = extract_neighborhood_in_pair_table(
        pairs, neighborhood_key=f"status_{CONTACT_0}", contact_only=False
    )
    assert len(by_key) == len(pairs)


def test_expand_pair_table(pairs, contact_position, sim):
    data = pairs.assign(position=contact_position)
    expanded = expand_pair_table(data)
    assert len(expanded) == len(pairs)
    t = sim.targets.assign(TRACK_ID=sim.targets["TRACK_ID"] + T_OFF).set_index(
        ["TRACK_ID", "FRAME"]
    )
    e = sim.effectors.assign(TRACK_ID=sim.effectors["TRACK_ID"] + E_OFF).set_index(
        ["TRACK_ID", "FRAME"]
    )
    for row in expanded.itertuples(index=False):
        assert row.reference_POSITION_X == pytest.approx(
            t.loc[(row.REFERENCE_ID, row.FRAME), "POSITION_X"]
        )
        assert row.neighbor_POSITION_Y == pytest.approx(
            e.loc[(row.NEIGHBOR_ID, row.FRAME), "POSITION_Y"]
        )


def test_relative_measurement_script_merges_neighborhoods(position, sim, monkeypatch):
    import celldetective
    from celldetective.relative_measurements import rel_measure_at_position

    # The script runs in a subprocess: make it import the code under test.
    root = os.path.dirname(os.path.dirname(celldetective.__file__))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]))
    contact(position)
    circle(position, distance=30, clear_neigh=False)
    rel_measure_at_position(position)
    df = pd.read_csv(os.path.join(position, "output", "tables", "trajectories_pairs.csv"))
    assert in_contact(df) == true_contacts(sim)

    m = centre_distances(sim)
    close = m[m["d"] <= 30]
    rows = df[df[f"status_{CIRCLE_30}"] == 1]
    assert set(zip(rows["REFERENCE_ID"], rows["NEIGHBOR_ID"], rows["FRAME"])) == set(
        zip(close["ID_a"], close["ID_b"], close["FRAME"])
    )


# ---------------------------------------------------------------------------
# The neighborhood process run by the GUI
# ---------------------------------------------------------------------------


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        pass


def run_process(pos, **protocol):
    from celldetective.processes.compute_neighborhood import NeighborhoodProcess

    queue = _Queue()
    full = {
        "population": ["targets", "effectors"],
        "clear_neigh": True,
        "event_time_col": None,
        "neighborhood_kwargs": kwargs(symmetrize=False, compute_cum_sum=True),
    }
    full.update(protocol)
    NeighborhoodProcess(
        queue,
        process_args={"batch_structure": {0: {"positions": [pos]}}, "protocols": [full], "img_shape": SHAPE,
                      "measure_pairs": True},
    ).run()
    assert queue.items[-1] == "finished"
    return pd.read_csv(os.path.join(pos, "output", "tables", "trajectories_pairs.csv"))


def test_process_runs_a_batch_in_one_window(tmp_path, sim):
    # Two wells of one position each and two protocols, in one process: every
    # position gets both neighborhoods, the well and position bars only move
    # forward, and the process ends once, at the end of the batch.
    from celldetective.processes.compute_neighborhood import NeighborhoodProcess
    from celldetective.utils.synthetic_interactions import write_experiment

    wells = write_experiment(str(tmp_path / "Batch"), [[sim], [sim]], write_labels=True, write_tables=True)
    protocols = [
        {"population": ["targets", "effectors"], "neighborhood_type": "mask_contact", "distance": [0],
         "clear_neigh": True, "event_time_col": None, "neighborhood_kwargs": kwargs()},
        {"population": ["targets", "effectors"], "neighborhood_type": "distance_threshold", "distance": [30],
         "clear_neigh": False, "event_time_col": None, "neighborhood_kwargs": kwargs()},
    ]
    queue = _Queue()
    NeighborhoodProcess(
        queue,
        process_args={"batch_structure": {k: {"positions": w} for k, w in enumerate(wells)},
                      "protocols": protocols, "img_shape": SHAPE},
    ).run()

    assert queue.items.count("finished") == 1 and queue.items[-1] == "finished"
    for key in ("well_progress", "pos_progress"):
        values = [m[key] for m in queue.items if isinstance(m, dict) and key in m]
        assert values[-1] == 100
    wells_seen = [m["well_progress"] for m in queue.items if isinstance(m, dict) and "well_progress" in m]
    assert wells_seen == sorted(wells_seen)
    for pos in (p for w in wells for p in w):
        df = pd.read_pickle(os.path.join(pos, "output", "tables", "trajectories_targets.pkl"))
        assert CONTACT_0 in df.columns
        assert CIRCLE_30 in df.columns


def test_process_contact_neighborhood_and_pairs(position, sim):
    pairs = run_process(
        position,
        neighborhood_type="mask_contact",
        distance=[0],
        channel_names=sim.channel_names,
    )
    df = pd.read_pickle(os.path.join(position, "output", "tables", "trajectories_targets.pkl"))
    entries = neighbor_entries(df, CONTACT_0)
    assert set(entries) == true_contacts(sim)
    presence = running_count(true_contacts(sim))
    assert {k: n["cumulated_presence"] for k, n in entries.items()} == presence

    assert in_contact(pairs) == true_contacts(sim)
    touching = pairs[f"status_{CONTACT_0}"] == 1
    assert np.all(np.abs(pairs.loc[touching, "contact_live_nuclei_channel_mean"] - 1000) < 20)
    assert os.path.exists(os.path.join(position, "log_neighborhood.txt"))


def test_process_distance_neighborhood_and_pairs(position, sim):
    pairs = run_process(position, neighborhood_type="distance_threshold", distance=[30])
    m = centre_distances(sim)
    close = m[m["d"] <= 30]
    expected = set(zip(close["ID_a"], close["ID_b"], close["FRAME"]))

    df = pd.read_pickle(os.path.join(position, "output", "tables", "trajectories_targets.pkl"))
    entries = neighbor_entries(df, CIRCLE_30)
    assert set(entries) == expected
    assert {k: n["cumulated_presence"] for k, n in entries.items()} == running_count(expected)

    rows = pairs[pairs[f"status_{CIRCLE_30}"] == 1]
    assert set(zip(rows["REFERENCE_ID"], rows["NEIGHBOR_ID"], rows["FRAME"])) == expected
