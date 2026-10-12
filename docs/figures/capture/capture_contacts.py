"""
Contacts between two populations: the viewer of the mask contact distance, the
contact hazard window of the Analyze tab with its forest plot and its contact
episodes, and the collapse of a pair table into contact episodes.

Runs on a synthetic effector-target experiment, not on demo_ricm: write it with
make_contacts_experiment.py and set CELLDETECTIVE_DOCS_EXP to it. Its targets
are killed at random during their contacts, it has mask-contact neighborhoods
at 0 and 2 px, and the pairs of the 0 px one are measured.
"""

import os

from PyQt5.QtWidgets import QTabWidget

from harness import *

POS = os.path.join(EXP, "W1", "100") + os.sep
NB = "neighborhood_(targets-effectors)_contact_0_px"

init, cp = open_experiment()
init.hide()
move(cp, 20, 40)

# -- Mask contact viewer ---------------------------------------------------------
from celldetective.gui.viewers.contour_viewer import ContactEdgeVisualizer

viewer = ContactEdgeVisualizer(
    cell_type="targets",
    neighbor_cell_type="effectors",
    edge_range=(0, 30),
    initial_edge=3,
    parent_list_widget=None,
    stack_path=os.path.join(POS, "movie", "synthetic.tif"),
    window_title="Position 100",
    frame_slider=True,
    contrast_slider=True,
    channel_cb=True,
    channel_names=list(cp.exp_channels),
    n_channels=len(cp.exp_channels),
    PxToUm=1,
)
viewer.show()
pump(2)
move(viewer.canvas, 500, 40, 560, 760)
viewer.channel_cb.setCurrentText("brightfield_channel")
pump(1)
viewer.frame_slider.setValue(26)
pump(1.5)
viewer.edge_slider.setValue(-4)
pump(1)
grab(viewer.canvas, "contact_viewer", marks={"edge": viewer.edge_slider, "opacity": viewer.opacity_slider})
viewer.close()
pump(0.5)

# -- Analyze tab and contact hazard ----------------------------------------------
tabs = cp.findChildren(QTabWidget)[0]
tabs.setCurrentIndex(1)
pump(1)
move(cp, 1100, 40, 460, 560)
grab(cp, "analyze_tab_contacts", marks={"hazard": cp.SurvivalBlock.contact_hazard_btn})

panel = cp.SurvivalBlock
panel.configure_contact_hazard()
pump(2)
hz = panel.config_contact_hazard
move(hz, 500, 40)
hz.neighborhood_cb.selectAll()
hz.episodes_check.setChecked(True)
hz.align_cb.setCurrentText("start")
pump(1)
grab(hz, "contact_hazard_options", marks={
    "population": hz.population_cb, "neighborhoods": hz.neighborhood_cb, "event": hz.event_cb,
    "origin": hz.origin_cb, "covariates": hz.covariate_cb, "lag": hz.lag_sb, "unit": hz.unit_cb,
    "strata": hz.strata_cb, "penalty": hz.penalizer_le, "query": hz.query_le,
    "calibration": hz.time_calibration_le, "long": hz.long_table_check, "episodes": hz.episodes_check,
    "window": hz.window_sb, "align": hz.align_cb, "gap": hz.gap_sb, "bins": hz.bins_le,
    "submit": hz.submit_btn,
})
hz.run()
pump(4)
figures = {w.windowTitle(): w for w in hz.windows if w.windowTitle() in ("Hazard ratios", "Event after contact episodes")}
forest = figures["Hazard ratios"]
move(forest, 500, 40, 720, 380)
pump(1)
grab(forest, "contact_hazard_forest")
dose = figures["Event after contact episodes"]
move(dose, 1000, 380, 520, 440)
pump(1)
grab(dose, "contact_episodes_plot")
for w in hz.windows:
    w.close()
hz.close()
pump(0.5)

# -- Pair table collapsed into contact episodes ----------------------------------
import pandas as pd

from celldetective.gui.tableUI import TableUI

pairs = pd.read_csv(os.path.join(POS, "output", "tables", "trajectories_pairs.csv"))
pairs.insert(0, "position", "W1/100")
table = TableUI(pairs, "pairs", population="pairs", plot_mode="static")
table.show()
pump(1)
move(table, 20, 40, 900, 500)
table.collapse_pairs_in_neigh()
pump(1)
dialog = table.selectNeighWidget
dialog.move(940, 40)
table.groupby_episode_rb.setChecked(True)
table.episode_gap_sb.setValue(1)
pump(0.5)
grab(dialog, "pair_collapse_episodes", marks={
    "reference": table.reference_pop_cb, "neighborhood": table.neigh_cb,
    "episode": table.groupby_episode_rb, "gap": table.episode_gap_sb,
    "operation": table.episode_projection_cb, "set": table.apply_neigh_btn,
})
table.prepare_table_at_neighborhood()
pump(2)
episodes = table.subtable
move(episodes, 20, 420, 1050, 420)
header = episodes.table_view.horizontalHeader()
first = list(episodes.data.columns).index("REFERENCE_ID")
episodes.table_view.horizontalScrollBar().setValue(header.sectionPosition(first))
pump(1)
grab(episodes, "contact_episodes_table")
episodes.close()
table.close()
pump(0.5)
cp.close()
