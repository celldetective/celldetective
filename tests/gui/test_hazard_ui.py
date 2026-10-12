"""
The hazard model window of the Analysis tab and the collapse of a pair
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
    from celldetective.gui.hazard_ui import ConfigHazard

    _, exp_dir, _ = experiment
    control = SimpleNamespace(
        FrameToMin=1.0, populations=["effectors", "targets"], well_list=selection([0]),
        position_list=selection([0]),
    )
    w = ConfigHazard(SimpleNamespace(exp_dir=exp_dir, parent_window=control))
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
    assert titles[0].startswith("Hazard of death")
    assert set(summary["distance"]) == {0.0, 2.0}
    in_contact = summary[summary["covariate"] == "in_contact"]
    assert (in_contact["ci_lower"] > 1).all()

    long_table = dict(shown["tables"])[f"Long table: {w.selected(w.neighborhood_cb)[0]}"]
    assert {"start", "stop", "event", "in_contact"} <= set(long_table.columns)

    episodes = dict(shown["tables"])[f"Contact episodes: {NB}"]
    assert {"duration", "outcome"} <= set(episodes.columns)
    assert [t for t, _ in shown["figures"]] == ["Hazard ratios", "Event after contact episodes"]


@pytest.fixture(scope="module")
def dose_experiment(tmp_path_factory):
    """Two wells, without and with the antibody: the same contacts, lethal
    only in the second well."""
    from celldetective.utils.synthetic_interactions import write_experiment

    exp_dir = str(tmp_path_factory.mktemp("dose") / "Dose")
    wells = [
        [simulate_interactions(shape=SHAPE, n_frames=60, n_targets=16, n_effectors=16, seed=10 * k + p,
                               death_probability=p_contact, background_death=0.003) for p in range(2)]
        for k, p_contact in enumerate((0.0, 0.1))
    ]
    positions = write_experiment(exp_dir, wells, write_labels=True, write_tables=True,
                                 well_labels={"concentrations": [0, 100], "antibodies": ["none", "bsAb"]})
    for pos in (p for well in positions for p in well):
        ti.compute_contact_neighborhood_at_position(
            pos, distance=0, population=["targets", "effectors"], img_shape=SHAPE, theta_dist=0,
            return_tables=False, clear_neigh=True, neighborhood_kwargs=ti.kwargs(),
        )
    return exp_dir + os.sep


@pytest.fixture
def dose_window(qtbot, dose_experiment):
    from celldetective.gui.hazard_ui import ConfigHazard

    control = SimpleNamespace(FrameToMin=1.0, populations=["effectors", "targets"],
                              well_list=selection([0, 1]), position_list=selection([0, 1]))
    w = ConfigHazard(SimpleNamespace(exp_dir=dose_experiment, parent_window=control))
    qtbot.addWidget(w)
    shown = {"tables": [], "figures": []}
    w.show_table = lambda df, title: shown["tables"].append((title, df))
    w.show_figure = lambda fig, title: shown["figures"].append((title, fig))
    return w, shown


def check_only(combo, names):
    combo.unselectAll()
    for i in range(combo.count()):
        if combo.itemText(i) in names:
            combo.setCurrentIndex(i)


def test_the_condition_of_the_wells_is_a_covariate(dose_window):
    # The targets die more in the well with the antibody: the concentration,
    # compared with the well without, raises the hazard.
    w, shown = dose_window
    covariates = [w.covariate_cb.itemText(i) for i in range(w.covariate_cb.count())]
    assert "concentrations (each vs 0)" in covariates
    check_only(w.covariate_cb, ["concentrations (each vs 0)"])
    w.run()
    summary = shown["tables"][0][1]
    row = summary[summary["covariate"] == "concentrations = 100"].iloc[0]
    assert row["ci_lower"] > 1
    assert row["effect"] == "higher hazard"


@pytest.mark.parametrize("covariate", ["concentrations", "log10(1 + concentrations)"])
def test_a_numeric_condition_is_a_continuous_covariate(dose_window, covariate):
    # A concentration is a quantity: its ratio is per pM (or per SD, or per
    # log unit), and a text condition is only offered by level.
    w, shown = dose_window
    covariates = [w.covariate_cb.itemText(i) for i in range(w.covariate_cb.count())]
    assert "antibodies (each vs none)" in covariates
    assert "antibodies" not in covariates and "log10(1 + antibodies)" not in covariates
    check_only(w.covariate_cb, [covariate])
    w.run()
    row = shown["tables"][0][1].set_index("covariate").loc[covariate]
    assert row["ci_lower"] > 1


def test_the_window_works_without_a_neighborhood(dose_window, monkeypatch):
    # The effect of the concentration does not need a neighborhood; the
    # contact covariates do.
    import celldetective.gui.hazard_ui as ui

    messages = []
    monkeypatch.setattr(ui, "generic_message", messages.append)
    w, shown = dose_window
    w.neighborhood_cb.unselectAll()
    check_only(w.covariate_cb, ["in_contact"])
    w.run()
    assert not shown["tables"] and "need a neighborhood" in messages[0]
    check_only(w.covariate_cb, ["concentrations (each vs 0)"])
    w.run()
    summary = shown["tables"][0][1]
    assert list(summary["neighborhood"]) == ["none"]
    assert summary.iloc[0]["ci_lower"] > 1


def test_the_fit_splits_by_condition(dose_window):
    # In contact, a target dies more only where there is antibody.
    w, shown = dose_window
    w.split_cb.setCurrentText("concentrations")
    check_only(w.covariate_cb, ["in_contact"])
    w.run()
    summary = shown["tables"][0][1].set_index("condition")
    assert summary.loc["concentrations = 100", "ci_lower"] > 1
    assert summary.loc["concentrations = 0", "effect"] == "no clear effect"


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
