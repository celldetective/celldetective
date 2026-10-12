"""
Write the synthetic experiment of the contact figures (capture_contacts.py):
targets killed at random during their contacts with effectors, mask-contact
neighborhoods at 0 and 2 px, and the pairs of the 0 px one measured.

    python make_contacts_experiment.py <scratch folder>/SyntheticContacts
    set CELLDETECTIVE_DOCS_EXP=<scratch folder>/SyntheticContacts
    python capture_contacts.py
"""

import logging
import os
import sys
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

from celldetective.neighborhood import compute_contact_neighborhood_at_position
from celldetective.relative_measurements import measure_pair_signals_at_position
from celldetective.utils.synthetic_interactions import simulate_interactions, write_experiment

path = sys.argv[1]
shape = (320, 320)
sim = simulate_interactions(shape=shape, n_frames=60, n_targets=16, n_effectors=16, seed=0, death_probability=0.08)
pos = write_experiment(path, sim, write_labels=True, write_tables=True, frametomin=2.0)
for d in (0, 2):
    compute_contact_neighborhood_at_position(
        pos, distance=d, population=["targets", "effectors"], img_shape=shape, theta_dist=0,
        return_tables=False, clear_neigh=d == 0,
        neighborhood_kwargs={"mode": "two-pop", "status": None, "not_status_option": None,
                             "include_dead_weight": True, "compute_cum_sum": False,
                             "attention_weight": True, "symmetrize": True},
    )
nb = "neighborhood_(targets-effectors)_contact_0_px"
pairs = measure_pair_signals_at_position(
    pos, {"reference": "targets", "neighbor": "effectors", "type": "contact", "distance": 0, "description": nb}
)
pairs.to_csv(os.path.join(pos, "output", "tables", "trajectories_pairs.csv"), index=False)
print(pos, len(pairs), "pair rows")
