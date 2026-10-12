"""
The contact hazard window of the Analysis tab and the collapse of a pair
table into contact episodes in the table UI, on a synthetic experiment whose
targets die from their contacts.
"""

import logging
import os
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from celldetective.relative_measurements import measure_pair_signals_at_position
from celldetective.utils.synthetic_interactions import simulate_interactions
from tests import test_interactions as ti

NB = "neighborhood_(targets-effectors)_contact_0_px"
SHAPE = (320, 320)


@pytest.fixture(autouse=True)
def quiet():
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    logging.disable(logging.NOTSET)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    sim = simulate_interactions(shape=SHAPE, n_frames=60, n_targets=16, n_effectors=16, seed=0,
                                death_probability=0.08)
    pos = ti.write_position(tmp_path_factory.mktemp("hazard") / "Exp", sim)
    for d in (0, 2):
        ti.compute_contact_neighborhood_at_position(
            pos, distance=d, population=["targets", "effectors"], img_shape=SHAPE, theta_dist=0,
            return_tables=False, clear_neigh=d == 0, neighborhood_kwargs=ti.kwargs(),
        )
    pairs = measure_pair_signals_at_position(
        pos, {"reference": "targets", "neighbor": "effectors", "type": "contact", "distance": 0, "description": NB}
    )
    pairs.to_csv(os.path.join(pos, "output", "tables", "trajectories_pairs.csv"), index=False)
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(pos))) + os.sep
    return sim, exp_dir, pairs


def selection(indices):
    return SimpleNamespace(getSelectedIndices=lambda: indices)


@pytest.fixture
def window(qtbot, experiment):
    from celldetective.gui.contact_hazard_ui import ConfigContactHazard

    _, exp_dir, _ = experiment
    control = SimpleNamespace(
        FrameToMin=1.0, populations=["effectors", "targets"], well_list=selection([0]),
        position_list=selection([0]),
    )
    w = ConfigContactHazard(SimpleNamespace(exp_dir=exp_dir, parent_window=control))
    qtbot.addWidget(w)
    shown = {"tables": [], "figures": []}
    w.show_table = lambda df, title: shown["tables"].append((title, df))
    w.show_figure = lambda fig, title: shown["figures"].append((title, fig))
    return w, shown


def test_window_offers_what_the_tables_hold(window):
    w, _ = window
    assert list(w.cols_per_pop) == ["targets"]
    neighborhoods = [w.neighborhood_cb.itemText(i) for i in range(w.neighborhood_cb.count())]
    assert set(neighborhoods) == {NB, NB.replace("_0_px", "_2_px")}
    assert w.selected(w.neighborhood_cb) == [neighborhoods[0]]
    assert w.event_cb.currentText() == "t_death"
    assert set(w.selected(w.covariate_cb)) == {"in_contact", "cumulative_contact"}
    assert w.episodes_check.isEnabled()


def test_window_fits_every_neighborhood_and_the_episodes(window):
    w, shown = window
    w.neighborhood_cb.selectAll()
    w.long_table_check.setChecked(True)
    w.episodes_check.setChecked(True)
    w.align_cb.setCurrentText("start")
    w.run()

    titles = [t for t, _ in shown["tables"]]
    summary = shown["tables"][0][1]
    assert titles[0].startswith("Contact hazard of death")
    assert set(summary["distance"]) == {0.0, 2.0}
    in_contact = summary[summary["covariate"] == "in_contact"]
    assert (in_contact["ci_lower"] > 1).all()

    long_table = dict(shown["tables"])[f"Long table: {w.selected(w.neighborhood_cb)[0]}"]
    assert {"start", "stop", "event", "in_contact"} <= set(long_table.columns)

    episodes = dict(shown["tables"])[f"Contact episodes: {NB}"]
    assert {"duration", "outcome"} <= set(episodes.columns)
    assert [t for t, _ in shown["figures"]] == ["Hazard ratios", "Event after contact episodes"]


def test_table_ui_collapses_pairs_into_episodes(qtbot, experiment):
    from celldetective.gui.tableUI import TableUI

    sim, _, pairs = experiment
    table = TableUI(pairs.assign(position="p"), "pairs", population="pairs", plot_mode="static")
    qtbot.addWidget(table)
    table.collapse_pairs_in_neigh()
    table.neigh_cb.setCurrentText(f"status_{NB}".replace("status_", ""))
    table.groupby_episode_rb.setChecked(True)
    assert not table.contact_only_check.isEnabled()
    table.episode_projection_cb.setCurrentText("max")
    table.prepare_table_at_neighborhood()

    episodes = table.subtable.data
    truth = sim.episodes
    assert set(zip(episodes["REFERENCE_ID"], episodes["NEIGHBOR_ID"], episodes["t_start"])) == set(
        zip(truth["TARGET_ID"] + ti.T_OFF, truth["EFFECTOR_ID"] + ti.E_OFF, truth["t_start"])
    )
    assert np.array_equal(
        episodes.sort_values(["REFERENCE_ID", "t_start"])["duration"].to_numpy(),
        truth.assign(REFERENCE_ID=truth["TARGET_ID"] + ti.T_OFF)
        .sort_values(["REFERENCE_ID", "t_start"])["duration"].to_numpy(),
    )
    table.subtable.close()
