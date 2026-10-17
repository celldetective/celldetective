"""
Every neighborhood configuration checked against an independent oracle.

The oracle rebuilds, from the exact geometry and masks of the synthetic
effector-target movie, what each option should produce: the neighbors of
every cell at every frame, with their distance, status, mask intersection,
attention weight, ``closest`` flag and cumulated presence, the symmetrized
neighborhood written in the neighbor table, the inclusive / exclusive /
intermediate counts split by status, and their means before and after an
event. The configuration grid is then swept at three levels:

- the in-memory functions (``distance_cut_neighborhood`` and
  ``mask_contact_neighborhood``), where every option is observable;
- the position-level functions, which add counts, means, column names and
  the tables written to disk;
- the process run by the GUI, which calls the library with the options the
  GUI writes, reports the progress of the frames and must give the same
  tables as the library.

Targets die after 4 frames of contact (``status`` is 1 once dead). The
effectors get a scripted ``status`` and both tables an event time
``t_event``, so that status options and event means are exercised on both
sides. Track IDs differ from mask labels (targets ``label + 100``,
effectors ``label + 200``), so that mixing up the two fails.
"""

import os
import shutil
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import binary_dilation, distance_transform_edt

from celldetective.neighborhood import (
    compute_contact_neighborhood_at_position,
    compute_neighborhood_at_position,
    distance_cut_neighborhood,
    extract_neighborhood_in_pair_table,
    mask_contact_neighborhood,
    set_live_status,
)
from celldetective.utils.synthetic_interactions import (
    simulate_interactions,
    write_experiment,
)

SHAPE = (160, 160)
OFFSET = {"targets": 100, "effectors": 200}
# Effector event times (-1: no event); status is 1 from the event on.
EFFECTOR_EVENT = {1: 8, 2: -1, 3: 15, 4: 22}
POPULATIONS = [
    ("targets", "effectors"),
    ("effectors", "targets"),
    ("targets", "targets"),
    ("effectors", "effectors"),
]
# (status, not_status_option) as the GUI writes them.
STATUS = {
    "all-alive": ([None, None], [False, False]),
    "reference-alive-until-event": (["status", None], [True, False]),
    "neighbor-alive-until-event": ([None, "status"], [False, True]),
    "both-raw-status": (["status", "status"], [False, False]),
}
# Contact needs a tolerance between targets, which never touch at 0 px; 2 px
# also checks that a distance of 2 does not clash with the "_2_" of the
# two-population column names.
DISTANCE = {"circle": 30, "contact": 2}


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    return simulate_interactions(seed=0, death_after=4)


@pytest.fixture(scope="module")
def tables(sim):
    targets = sim.targets.copy()
    targets["t_event"] = targets["t_death"]
    effectors = sim.effectors.copy()
    effectors["t_event"] = effectors["TRACK_ID"].map(EFFECTOR_EVENT)
    effectors["status"] = (
        (effectors["t_event"] >= 0) & (effectors["FRAME"] >= effectors["t_event"])
    ).astype(int)
    out = {}
    for population, df in (("targets", targets), ("effectors", effectors)):
        df = df[["TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y", "class_id", "status", "t_event"]].copy()
        df["TRACK_ID"] = df["class_id"] + OFFSET[population]
        out[population] = df.reset_index(drop=True)
    return out


@pytest.fixture(scope="module")
def labels(sim):
    return {"targets": sim.labels_targets, "effectors": sim.labels_effectors}


def test_the_data_exercises_every_status(tables, sim):
    """Without dead targets, live effectors and contacts in both states, the
    status options would go untested."""
    for df in tables.values():
        assert set(df["status"]) == {0, 1}
    t_event = tables["targets"].groupby("TRACK_ID")["t_event"].first()
    states = {0 <= t_event[t + OFFSET["targets"]] <= f for t, f in zip(sim.contacts["TARGET_ID"], sim.contacts["FRAME"])}
    assert states == {True, False}


def write_position(folder, tables):
    sim = simulate_interactions(seed=0, death_after=4)
    pos = write_experiment(str(folder), sim, write_labels=True, write_tables=True)
    for population, df in tables.items():
        df.to_csv(os.path.join(pos, "output", "tables", f"trajectories_{population}.csv"), index=False)
    return pos


@pytest.fixture(scope="module")
def template(tmp_path_factory, tables):
    return write_position(tmp_path_factory.mktemp("template") / "SynthExp", tables)


@pytest.fixture
def position(tmp_path, template):
    """A fresh copy of the experiment, so that tests do not share outputs."""
    return copy_position(template, tmp_path / "copy")


def copy_position(src, dest):
    dest = os.path.join(str(dest), "W1", "100")
    shutil.copytree(src, dest)
    return dest + os.sep


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

_contact_cache = {}


def contact_pairs(labels, pop, d):
    """{frame: {(mask A, mask B)}} for masks that, each grown outward by
    ``d`` px, touch or overlap (8-connectivity)."""
    key = (pop, d)
    if key in _contact_cache:
        return _contact_cache[key]
    same = pop[0] == pop[1]
    out = {}
    for f in range(len(labels[pop[0]])):

        def grown(lbl):
            return {int(c): distance_transform_edt(lbl != c) <= d for c in np.unique(lbl) if c > 0}

        grown_a = grown(labels[pop[0]][f])
        grown_b = grown_a if same else grown(labels[pop[1]][f])
        found = set()
        for a, ga in grown_a.items():
            touching = binary_dilation(ga, structure=np.ones((3, 3)))
            for b, gb in grown_b.items():
                if not (same and a == b) and np.any(touching & gb):
                    found.add((a, b))
        out[f] = found
    _contact_cache[key] = out
    return out


def live_status(df, col, not_status):
    """The status the neighborhood should use for each cell: 1 when no
    column is given, the column with 2 read as 1, inverted on request."""
    if col is None:
        return np.ones(len(df))
    values = df[col].to_numpy(float).copy()
    if not_status is None:
        return values
    values[values == 2] = 1
    if not_status:
        values = np.where(np.isin(values, [0, 1]), 1 - values, np.nan)
    return values


def expected_neighborhood(
    tables,
    labels,
    pop,
    kind,
    d,
    status=(None, None),
    not_status=(False, False),
    include_dead_weight=True,
    attention_weight=True,
    compute_cum_sum=False,
):
    """Expected neighbor lists of the reference cells and, as symmetrize
    writes them, of the neighbor cells: {(TRACK_ID, FRAME): [neighbor dict]}."""
    A, B = tables[pop[0]], tables[pop[1]]
    same = pop[0] == pop[1]
    s_A = live_status(A, status[0], None if not_status is None else not_status[0])
    s_B = live_status(B, status[1], None if not_status is None else not_status[1])
    contacts = contact_pairs(labels, pop, d) if kind == "contact" else None
    reference, symmetric = {}, {}
    presence, presence_weighted = {}, {}

    for f in sorted(set(A["FRAME"]) | set(B["FRAME"])):
        ia, ib = np.flatnonzero(A["FRAME"] == f), np.flatnonzero(B["FRAME"] == f)
        a, b = A.iloc[ia], B.iloc[ib]
        sa, sb = s_A[ia], s_B[ib]
        id_a, id_b = a["TRACK_ID"].to_numpy(), b["TRACK_ID"].to_numpy()
        xy_a, xy_b = a[["POSITION_X", "POSITION_Y"]].to_numpy(), b[["POSITION_X", "POSITION_Y"]].to_numpy()
        D = np.hypot(*(xy_a[:, None, :] - xy_b[None, :, :]).transpose(2, 0, 1))

        if kind == "circle":
            M = D <= d
        else:
            M = np.array([[(ma, mb) in contacts[f] for mb in b["class_id"]] for ma in a["class_id"]])
        if same:
            np.fill_diagonal(M, False)

        # The distance used to find each neighbor's closest reference cell:
        # the centre distance, over every reference cell in a circle
        # neighborhood and over those in contact otherwise.
        Dm = D.copy()
        if kind == "contact":
            Dm[~M] = 1.0e6
        if same:
            np.fill_diagonal(Dm, 1.0e6)
        counted = np.ones(len(ia), bool) if include_dead_weight else sa == 1
        n_refs = (M & counted[:, None]).sum(axis=0)
        weight_b = np.where(n_refs > 0, 1.0 / np.maximum(n_refs, 1), np.nan)
        closest_of_b = [
            id_a[counted][np.argmin(Dm[counted, j])] if counted.any() else None for j in range(len(ib))
        ]

        for i in range(len(ia)):
            js = np.flatnonzero(M[i])
            neighbors = []
            for j in js:
                n = {"id": id_b[j], "distance": round(D[i, j], 2), "status": sb[j]}
                if kind == "contact":
                    n["intersection"] = (
                        0
                        if same
                        else int(np.sum((labels[pop[0]][f] == a["class_id"].iloc[i]) & (labels[pop[1]][f] == b["class_id"].iloc[j])))
                    )
                if attention_weight:
                    n["weight"] = weight_b[j]
                    n["closest"] = closest_of_b[j] == id_a[i]
                if compute_cum_sum:
                    key = (id_a[i], id_b[j])
                    presence[key] = presence.get(key, 0) + 1
                    n["cumulated_presence"] = presence[key]
                    if attention_weight:
                        presence_weighted[key] = presence_weighted.get(key, 0.0) + weight_b[j]
                        n["cumulated_presence_weighted"] = presence_weighted[key]
                neighbors.append(n)
            reference[(id_a[i], f)] = neighbors

            if len(js) == 0:
                continue
            # Symmetrized entry: reference cell i seen from each of its neighbors.
            n_neighbors = len(js) if include_dead_weight else int(np.sum(sb[js] == 1))
            weight_a = 1.0 / n_neighbors if n_neighbors > 0 else np.nan
            if not include_dead_weight and sa[i] == 0:
                weight_a = 0
            closest_j = js[np.argmin([round(D[i, j], 2) for j in js])]
            for j in js:
                n = {"id": id_a[i], "distance": round(D[i, j], 2), "status": sa[i]}
                if kind == "contact":
                    n["intersection"] = next(x["intersection"] for x in neighbors if x["id"] == id_b[j])
                if attention_weight:
                    n["weight"] = weight_a
                    n["closest"] = j == closest_j
                symmetric.setdefault((id_b[j], f), []).append(n)
    return reference, symmetric


def expected_counts(reference, kind):
    """Inclusive, intermediate and (circle only) exclusive counts of each
    reference cell, overall and split by neighbor status."""
    rows = []
    for (track, frame), neighbors in reference.items():
        status = np.array([n["status"] for n in neighbors], float)
        weight = np.array([n.get("weight", np.nan) for n in neighbors], float)
        closest = np.array([n.get("closest", False) for n in neighbors], bool)
        row = {"TRACK_ID": track, "FRAME": frame}
        row["inclusive"] = len(neighbors)
        row["intermediate"] = np.sum(weight)
        row["exclusive"] = int(np.sum(closest))
        for s in (0, 1):
            row[f"inclusive_s{s}"] = int(np.sum(status == s))
            row[f"intermediate_s{s}"] = np.sum(weight[status == s])
            row[f"exclusive_s{s}"] = int(np.sum(closest & (status == s)))
        rows.append(row)
    out = pd.DataFrame(rows)
    if kind == "contact":
        out = out.drop(columns=[c for c in out.columns if c.startswith("exclusive")])
    return out


def expected_means(counts, events, metrics, with_event):
    """Mean live-neighbor count of each track before its event (over the
    whole track without an event) and after it."""
    rows = []
    for track, group in counts.groupby("TRACK_ID"):
        ev = events[track] if with_event else -1
        if ev < 0:
            ev = group["FRAME"].max() if with_event else counts["FRAME"].max()
        row = {"TRACK_ID": track}
        for m in metrics:
            values = group[f"{m}_s1"]
            before = values[group["FRAME"] <= ev]
            row[f"{m}_before"] = np.nanmean(before) if before.notnull().any() else np.nan
            after = values[group["FRAME"] > ev]
            has_event = with_event and events[track] >= 0
            row[f"{m}_after"] = np.nanmean(after) if has_event and after.notnull().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("TRACK_ID")


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def same_value(got, expected):
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        return got is None or (isinstance(got, (float, np.floating)) and np.isnan(got))
    if isinstance(expected, (bool, np.bool_)):
        return bool(got) == bool(expected)
    return got == pytest.approx(expected, abs=1e-6)


def assert_same_neighbors(got, expected, where):
    got = sorted(got, key=lambda n: n["id"])
    expected = sorted(expected, key=lambda n: n["id"])
    assert [int(n["id"]) for n in got] == [int(n["id"]) for n in expected], where
    for g, e in zip(got, expected):
        assert set(g) == set(e), f"{where}: keys {sorted(g)} != {sorted(e)}"
        for key in e:
            if key == "distance":
                assert g[key] == pytest.approx(e[key], abs=0.011), f"{where}: distance"
            else:
                assert same_value(g[key], e[key]), f"{where} neighbor {e['id']}: {key} {g[key]!r} != {e[key]!r}"


def neighbor_column(df, col):
    return {(int(t), int(f)): v for t, f, v in zip(df["TRACK_ID"], df["FRAME"], df[col])}


def assert_column(df, col, expected, empty_is_nan=False):
    """Every row of ``df`` holds the expected neighbors; rows without an
    expected entry are NaN when ``empty_is_nan`` (symmetrized side)."""
    got = neighbor_column(df, col)
    for key, value in got.items():
        if key in expected:
            assert isinstance(value, list), f"{key}: {value!r}"
            assert_same_neighbors(value, expected[key], key)
        elif empty_is_nan:
            assert not isinstance(value, list), f"{key} should have no neighbor, got {value!r}"
        else:
            raise AssertionError(f"unexpected row {key}")
    if not empty_is_nan:
        assert set(got) == set(expected)


def options(status_case="all-alive", **overrides):
    status, not_status = STATUS[status_case]
    kw = {
        "status": list(status),
        "not_status_option": list(not_status),
        "include_dead_weight": True,
        "compute_cum_sum": False,
        "attention_weight": True,
        "symmetrize": False,
    }
    kw.update(overrides)
    return kw


def oracle_kwargs(kw):
    return {
        "status": kw["status"],
        "not_status": kw["not_status_option"],
        "include_dead_weight": kw["include_dead_weight"],
        "attention_weight": kw["attention_weight"],
        "compute_cum_sum": kw["compute_cum_sum"],
    }


def run_in_memory(tables, labels, kind, pop, d, kw):
    mode = "self" if pop[0] == pop[1] else "two-pop"
    A, B = tables[pop[0]].copy(), tables[pop[1]].copy()
    if kind == "circle":
        return distance_cut_neighborhood(A, B, d, mode=mode, **kw)
    lb = None if mode == "self" else list(labels[pop[1]])
    return mask_contact_neighborhood(A, B, list(labels[pop[0]]), lb, d, mode=mode, **kw)


def raw_column(kind, pop, d):
    mode = "self" if pop[0] == pop[1] else "2"
    return f"neighborhood_{mode}_{kind}_{d}_px"


# ---------------------------------------------------------------------------
# In-memory functions: every option
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status_case", list(STATUS))
@pytest.mark.parametrize("pop", POPULATIONS, ids="-".join)
@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_populations_and_statuses(tables, labels, kind, pop, status_case):
    d = DISTANCE[kind]
    kw = options(status_case, include_dead_weight=False, compute_cum_sum=True, symmetrize=True)
    df_A, df_B = run_in_memory(tables, labels, kind, pop, d, dict(kw))
    reference, symmetric = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(kw))
    if pop != ("targets", "targets"):
        assert any(reference.values()), "the configuration finds no neighbor at all"
    col = raw_column(kind, pop, d)
    assert_column(df_A, col, reference)
    assert_column(df_B, col, symmetric, empty_is_nan=True)


@pytest.mark.parametrize(
    "include_dead_weight, attention_weight, symmetrize, compute_cum_sum",
    list(product([True, False], repeat=4)),
)
@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_weight_symmetrize_and_presence_options(
    tables, labels, kind, include_dead_weight, attention_weight, symmetrize, compute_cum_sum
):
    pop, d = ("targets", "effectors"), DISTANCE[kind]
    kw = options(
        "all-alive",
        status=["status", "status"],
        not_status_option=[True, True],
        include_dead_weight=include_dead_weight,
        attention_weight=attention_weight,
        symmetrize=symmetrize,
        compute_cum_sum=compute_cum_sum,
    )
    df_A, df_B = run_in_memory(tables, labels, kind, pop, d, dict(kw))
    reference, symmetric = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(kw))
    col = raw_column(kind, pop, d)
    assert_column(df_A, col, reference)
    if symmetrize:
        assert_column(df_B, col, symmetric, empty_is_nan=True)
    else:
        assert df_B[col].isnull().all()


@pytest.mark.parametrize("pop", POPULATIONS, ids="-".join)
@pytest.mark.parametrize("kind, distances", [("circle", [15, 45]), ("contact", [0, 1, 4])])
def test_several_distances_in_one_call(tables, labels, kind, distances, pop):
    kw = options("reference-alive-until-event", compute_cum_sum=True)
    df_A, _ = run_in_memory(tables, labels, kind, pop, list(distances), dict(kw))
    sizes = []
    for d in distances:
        reference, _ = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(kw))
        assert_column(df_A, raw_column(kind, pop, d), reference)
        sizes.append(sum(map(len, reference.values())))
    assert sizes == sorted(sizes)


def test_raw_status_values_without_not_status_option(tables, labels):
    """Without ``not_status_option``, the status column is used as is."""
    tabs = {k: v.copy() for k, v in tables.items()}
    tabs["effectors"]["status"] = tabs["effectors"]["status"] * 2  # 0 or 2
    kw = options(status=[None, "status"], not_status_option=None)
    df_A, _ = run_in_memory(tabs, labels, "circle", ("targets", "effectors"), 30, dict(kw))
    reference, _ = expected_neighborhood(tabs, labels, ("targets", "effectors"), "circle", 30, **oracle_kwargs(kw))
    assert {n["status"] for v in reference.values() for n in v} == {0, 2}
    assert_column(df_A, raw_column("circle", ("targets", "effectors"), 30), reference)


def test_untracked_tables_list_neighbors_without_presence(tables, labels):
    tabs = {k: v.rename(columns={"TRACK_ID": "ID"}) for k, v in tables.items()}
    pop = ("targets", "effectors")
    for kind in ("circle", "contact"):
        d = DISTANCE[kind]
        kw = options(compute_cum_sum=True)
        df_A, _ = run_in_memory(tabs, labels, kind, pop, d, dict(kw))
        reference, _ = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(options()))
        assert_column(df_A.rename(columns={"ID": "TRACK_ID"}), raw_column(kind, pop, d), reference)


def test_cells_outside_a_track_are_left_out(tables, labels):
    tabs = {k: v.copy() for k, v in tables.items()}
    lost = tabs["effectors"]["FRAME"] == 10
    tabs["effectors"]["TRACK_ID"] = tabs["effectors"]["TRACK_ID"].where(~lost)
    pop = ("targets", "effectors")
    for kind in ("circle", "contact"):
        d = DISTANCE[kind]
        df_A, df_B = run_in_memory(tabs, labels, kind, pop, d, dict(options()))
        assert not df_B["TRACK_ID"].isnull().any()
        col = raw_column(kind, pop, d)
        # Without a neighbor cell at a frame, the neighborhood is not computed.
        at_10 = df_A.loc[df_A["FRAME"] == 10, col]
        assert len(at_10) > 0 and at_10.isnull().all()
        reference, _ = expected_neighborhood(tables, labels, pop, kind, d)
        expected = {k: v for k, v in reference.items() if k[1] != 10}
        got = df_A[df_A["FRAME"] != 10]
        assert_column(got, col, expected)


def test_presence_counts_past_frames_without_neighbor_cells(tables, labels):
    """Before the neighbor population appears, the reference cells have no
    neighborhood (NaN); the cumulated presence must skip those frames
    instead of failing on them."""
    tabs = {k: v.copy() for k, v in tables.items()}
    tabs["effectors"] = tabs["effectors"][tabs["effectors"]["FRAME"] >= 3]
    pop = ("targets", "effectors")
    for kind in ("circle", "contact"):
        d = DISTANCE[kind]
        df_A, _ = run_in_memory(tabs, labels, kind, pop, d, dict(options(compute_cum_sum=True)))
        col = raw_column(kind, pop, d)
        assert df_A.loc[df_A["FRAME"] < 3, col].isnull().all()
        n_checked = 0
        for _, track in df_A.sort_values("FRAME").groupby("TRACK_ID"):
            seen = {}
            for neighbors in track[col]:
                if not isinstance(neighbors, list):
                    continue
                for n in neighbors:
                    seen[n["id"]] = seen.get(n["id"], 0) + 1
                    assert n["cumulated_presence"] == seen[n["id"]]
                    n_checked += 1
        assert n_checked > 0


@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_an_unknown_mode_returns_no_tables(tables, labels, kind):
    A, B = tables["targets"].copy(), tables["effectors"].copy()
    if kind == "circle":
        result = distance_cut_neighborhood(A, B, 30, mode="typo")
    else:
        result = mask_contact_neighborhood(
            A, B, list(labels["targets"]), list(labels["effectors"]), 2, mode="typo"
        )
    assert result == (None, None)


# ---------------------------------------------------------------------------
# Live status
# ---------------------------------------------------------------------------


def test_live_status_options():
    A = pd.DataFrame({"c": [0, 1, 2, np.nan]})
    B = pd.DataFrame({"c": [1, 0, 2, 0]})

    a, b, s = set_live_status(A.copy(), B.copy(), None, None)
    assert s == ["live_status", "live_status"]
    assert (a["live_status"] == 1).all() and (b["live_status"] == 1).all()

    a, b, s = set_live_status(A.copy(), B.copy(), [None, None], [False, False])
    assert s == ["live_status", "live_status"]

    status = ["c", "c"]
    a, b, s = set_live_status(A.copy(), B.copy(), status, [True, False])
    assert status == ["c", "c"], "the caller's list must stay as given"
    assert s == ["not_c", "c"]
    np.testing.assert_array_equal(a["not_c"].to_numpy(float), [1, 0, 0, np.nan])
    np.testing.assert_array_equal(b["c"].to_numpy(float), [1, 0, 1, 0])

    a, b, s = set_live_status(A.copy(), B.copy(), [None, "c"], [False, True])
    assert s == ["live_status", "not_c"]
    np.testing.assert_array_equal(b["not_c"].to_numpy(float), [0, 1, 0, 1])

    # An option left to None does not invert, but past events still count.
    a, b, s = set_live_status(A.copy(), B.copy(), ["c", None], [None, False])
    assert s == ["c", "live_status"]
    np.testing.assert_array_equal(a["c"].to_numpy(float), [0, 1, 1, np.nan])

    for not_status in ([False, False], [True, False], None):
        with pytest.raises(KeyError, match="not found in set A"):
            set_live_status(A.copy(), B.copy(), ["missing", None], not_status)
    with pytest.raises(ValueError):
        set_live_status(A.copy(), B.copy(), ["c"], [False])


# ---------------------------------------------------------------------------
# Position level: counts, means, names and written tables
# ---------------------------------------------------------------------------


POSITION_CASES = [
    ("all-alive", None),
    ("reference-alive-until-event", "t_event"),
    ("neighbor-alive-until-event", None),
    ("both-raw-status", "t_event"),
]


def at_position(kind, pos, pop, d, kw, event_time_col=None, clear_neigh=True):
    fn = compute_neighborhood_at_position if kind == "circle" else compute_contact_neighborhood_at_position
    return fn(
        pos,
        distance=d,
        population=list(pop),
        img_shape=SHAPE,
        theta_dist=None,
        return_tables=True,
        clear_neigh=clear_neigh,
        event_time_col=event_time_col,
        neighborhood_kwargs={"mode": "two-pop", **kw},
    )


def final_column(kind, pop, d):
    if pop[0] == pop[1]:
        return f"neighborhood_self_{kind}_{d}_px"
    return f"neighborhood_({pop[0]}-{pop[1]})_{kind}_{d}_px"


@pytest.mark.parametrize("status_case, event_time_col", POSITION_CASES)
@pytest.mark.parametrize("pop", POPULATIONS, ids="-".join)
@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_position_counts_means_and_tables(position, tables, labels, kind, pop, status_case, event_time_col):
    d = DISTANCE[kind]
    kw = options(status_case, compute_cum_sum=True)
    df_A, df_B = at_position(kind, position, pop, d, dict(kw), event_time_col)
    col = final_column(kind, pop, d)

    reference, _ = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(kw))
    assert_column(df_A, col, reference)

    metrics = ["inclusive", "intermediate"] + (["exclusive"] if kind == "circle" else [])
    counts = expected_counts(reference, kind)
    got = df_A.set_index(["TRACK_ID", "FRAME"])
    exp = counts.set_index(["TRACK_ID", "FRAME"]).loc[got.index]
    for m in metrics:
        for split in ("", "s0_", "s1_"):
            name = f"{m}_count_{split}{col}"
            assert name in got.columns, name
            np.testing.assert_allclose(got[name].to_numpy(float), exp[f"{m}{'_' + split[:-1] if split else ''}"].to_numpy(float), err_msg=name)
    if kind == "contact":
        assert not any(c.startswith("exclusive_count") for c in df_A.columns)

    events = tables[pop[0]].groupby("TRACK_ID")["t_event"].first()
    means = expected_means(counts, events, metrics, event_time_col is not None)
    suffix = "_before_event" if event_time_col else ""
    per_track = df_A.groupby("TRACK_ID").first()
    for m in metrics:
        name = f"mean_count_{m}_{col}{suffix}"
        assert name in per_track.columns, name
        np.testing.assert_allclose(per_track[name].to_numpy(float), means.loc[per_track.index, f"{m}_before"].to_numpy(float), err_msg=name)
        name = f"mean_count_{m}_{col}_after_event"
        if event_time_col is None:
            assert name not in per_track.columns
        else:
            got_after = per_track[name].to_numpy(float) if name in per_track.columns else np.full(len(per_track), np.nan)
            np.testing.assert_allclose(got_after, means.loc[per_track.index, f"{m}_after"].to_numpy(float), err_msg=name)

    # Written tables: the pickle keeps the neighbor lists, the CSV the rest.
    tables_dir = os.path.join(position, "output", "tables")
    pkl = pd.read_pickle(os.path.join(tables_dir, f"trajectories_{pop[0]}.pkl"))
    assert_column(pkl, col, reference)
    csv = pd.read_csv(os.path.join(tables_dir, f"trajectories_{pop[0]}.csv"))
    assert not any(c.startswith("neighborhood_") for c in csv.columns)
    assert f"inclusive_count_{col}" in csv.columns
    if pop[0] != pop[1]:
        assert not any(c.startswith("neighborhood_") for c in df_B.columns)
        assert os.path.exists(os.path.join(tables_dir, f"trajectories_{pop[1]}.pkl"))


@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_rerunning_without_clearing_replaces_the_neighborhood(position, tables, labels, kind):
    pop, d = ("targets", "effectors"), DISTANCE[kind]
    other = 50 if kind == "circle" else 0
    at_position(kind, position, pop, other, options())
    at_position(kind, position, pop, d, options(), clear_neigh=False)
    kw = options("reference-alive-until-event", compute_cum_sum=True)
    df_A, _ = at_position(kind, position, pop, d, dict(kw), clear_neigh=False)

    assert df_A.columns.is_unique, df_A.columns[df_A.columns.duplicated()].tolist()
    assert len(df_A) == len(tables[pop[0]])
    reference, _ = expected_neighborhood(tables, labels, pop, kind, d, **oracle_kwargs(kw))
    col = final_column(kind, pop, d)
    assert_column(df_A, col, reference)
    counts = expected_counts(reference, kind).set_index(["TRACK_ID", "FRAME"])
    got = df_A.set_index(["TRACK_ID", "FRAME"])
    np.testing.assert_allclose(
        got[f"inclusive_count_s1_{col}"].to_numpy(float),
        counts.loc[got.index, "inclusive_s1"].to_numpy(float),
    )
    # The neighborhood computed first is kept.
    previous, _ = expected_neighborhood(tables, labels, pop, kind, other)
    assert_column(df_A, final_column(kind, pop, other), previous)

    csv = pd.read_csv(os.path.join(position, "output", "tables", "trajectories_targets.csv"))
    assert csv.columns.is_unique
    assert not any(c.endswith(("_x", "_y")) for c in csv.columns)


@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_the_options_of_the_caller_are_left_as_given(position, kind):
    """The mode follows the populations without being written back into the
    caller's options, which are then reused for another pair."""
    kw = {"mode": "two-pop", **options()}
    given = dict(kw)
    fn = compute_neighborhood_at_position if kind == "circle" else compute_contact_neighborhood_at_position
    df_A, _ = fn(position, distance=DISTANCE[kind], population=["targets", "targets"],
                 return_tables=True, clear_neigh=True, neighborhood_kwargs=kw)
    assert kw == given
    assert final_column(kind, ("targets", "targets"), DISTANCE[kind]) in df_A.columns


# ---------------------------------------------------------------------------
# The GUI process gives the same tables as the library
# ---------------------------------------------------------------------------


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        pass


def run_process(pos, kind, pop, distances, kw, event_time_col, measure_pairs=False, channel_names=None):
    from celldetective.processes.compute_neighborhood import NeighborhoodProcess

    protocol = {
        "population": list(pop),
        "neighborhood_type": "distance_threshold" if kind == "circle" else "mask_contact",
        "distance": list(distances),
        "clear_neigh": True,
        "event_time_col": event_time_col,
        "neighborhood_kwargs": {"mode": "self" if pop[0] == pop[1] else "two-pop", **kw},
    }
    if channel_names is not None:
        protocol["channel_names"] = channel_names
    queue = _Queue()
    NeighborhoodProcess(
        queue,
        process_args={"batch_structure": {0: {"positions": [pos]}}, "protocols": [protocol], "img_shape": SHAPE,
                      "measure_pairs": measure_pairs},
    ).run()
    assert queue.items[-1] == "finished"
    return queue


def comparable(df):
    """A table with its neighbor lists made comparable with ``==``."""

    def freeze(v):
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return "nan"
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return round(float(v), 9)
        return v

    df = df.sort_values(["TRACK_ID", "FRAME"]).reset_index(drop=True)
    df = df[sorted(df.columns)]
    for c in df.columns:
        if c.startswith("neighborhood_"):
            df[c] = [
                tuple(sorted(tuple(sorted((k, freeze(x)) for k, x in n.items())) for n in v)) if isinstance(v, list) else "nan"
                for v in df[c]
            ]
    return df


GUI_CASES = [
    ("all-alive", False, None),
    ("reference-alive-until-event", True, "t_event"),
    ("neighbor-alive-until-event", True, None),
    ("both-raw-status", False, "t_event"),
]


@pytest.mark.parametrize("status_case, compute_cum_sum, event_time_col", GUI_CASES)
@pytest.mark.parametrize("pop", POPULATIONS, ids="-".join)
@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_gui_process_matches_the_library(
    tmp_path, template, kind, pop, status_case, compute_cum_sum, event_time_col
):
    # The options the GUI writes: attention weights, no symmetrization.
    kw = options(status_case, compute_cum_sum=compute_cum_sum, symmetrize=False, include_dead_weight=True)
    distances = [DISTANCE[kind], 20] if kind == "circle" else [DISTANCE[kind], 0]
    lib = copy_position(template, tmp_path / "library")
    gui = copy_position(template, tmp_path / "gui")
    at_position(kind, lib, pop, distances, dict(kw), event_time_col)
    run_process(gui, kind, pop, distances, dict(kw), event_time_col)

    for population in set(pop):
        for ext in ("pkl", "csv"):
            path = f"output/tables/trajectories_{population}.{ext}"
            read = pd.read_pickle if ext == "pkl" else pd.read_csv
            expected, got = read(lib + path), read(gui + path)
            pd.testing.assert_frame_equal(comparable(got), comparable(expected), check_dtype=False, obj=path)


@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_gui_process_measures_pairs_of_every_distance(tmp_path, template, tables, labels, kind):
    pop = ("targets", "effectors")
    distances = [DISTANCE[kind], 20] if kind == "circle" else [DISTANCE[kind], 0]
    pos = copy_position(template, tmp_path / "gui")
    run_process(pos, kind, pop, distances, options(), None, measure_pairs=True)
    pairs = pd.read_csv(os.path.join(pos, "output", "tables", "trajectories_pairs.csv"))
    for d in distances:
        reference, _ = expected_neighborhood(tables, labels, pop, kind, d)
        expected = {(t, n["id"], f) for (t, f), v in reference.items() for n in v}
        assert len(expected) > 0
        rows = pairs[pairs[f"status_{final_column(kind, pop, d)}"] == 1]
        assert set(zip(rows["REFERENCE_ID"], rows["NEIGHBOR_ID"], rows["FRAME"])) == expected


@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_gui_process_reports_the_progress_of_every_frame(tmp_path, template, tables, kind):
    """Every frame is reported, with or without neighbor cells: once per
    distance for the circle neighborhoods, once for all the contact ones."""
    distances = [DISTANCE[kind], 20] if kind == "circle" else [DISTANCE[kind], 0]
    pos = copy_position(template, tmp_path / "gui")
    queue = run_process(pos, kind, ("targets", "effectors"), distances, options(), None)
    progress = [m["frame_progress"] for m in queue.items if isinstance(m, dict) and "frame_progress" in m]
    n = len(np.union1d(tables["targets"]["FRAME"], tables["effectors"]["FRAME"]))
    sweep = [100 * (i + 1) / n for i in range(n)]
    np.testing.assert_allclose(progress, sweep * (len(distances) if kind == "circle" else 1))


def test_gui_process_measures_the_contact_site_intensity(tmp_path, template, sim):
    """The channels the GUI writes for a contact neighborhood are measured at
    the contact site, as by the library."""
    pop, distances, channels = ("targets", "effectors"), [DISTANCE["contact"], 0], list(sim.channel_names)
    lib = copy_position(template, tmp_path / "library")
    gui = copy_position(template, tmp_path / "gui")
    # The config names the movie to measure.
    config = os.path.join(template, os.pardir, os.pardir, "config.ini")
    for pos in (lib, gui):
        shutil.copy(config, os.path.join(pos, os.pardir, os.pardir))
    compute_contact_neighborhood_at_position(
        lib, distance=distances, population=list(pop), theta_dist=None, clear_neigh=True,
        channel_names=channels, neighborhood_kwargs=options(),
    )
    run_process(gui, "contact", pop, distances, options(), None, channel_names=channels)

    path = f"output/tables/trajectories_{pop[0]}.csv"
    expected, got = pd.read_csv(lib + path), pd.read_csv(gui + path)
    for ch in channels:
        assert got[f"contact_{ch}_mean"].notnull().any()
    pd.testing.assert_frame_equal(comparable(got), comparable(expected), obj=path)


@pytest.mark.parametrize("pop", POPULATIONS, ids="-".join)
@pytest.mark.parametrize("kind", ["circle", "contact"])
def test_pairs_of_a_neighborhood_are_found_by_distance(pop, kind):
    """Without a key, the pair-table column is named as the neighborhood
    was, self neighborhoods included."""
    col = "status_" + final_column(kind, pop, 30)
    pairs = pd.DataFrame({
        "reference_population": [pop[0]] * 3,
        "neighbor_population": [pop[1]] * 3,
        col: [1, 0, 1],
    })
    got = extract_neighborhood_in_pair_table(
        pairs, distance=30, reference_population=pop[0], neighbor_population=pop[1], mode=kind
    )
    assert list(got.index) == [0, 2]
