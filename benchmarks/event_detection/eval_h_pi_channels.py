"""lysis_H_PI: does the event recall depend on which measurement is read?"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from eval_common import HERE, movies, per_track, score, table  # noqa: E402

from celldetective.event_detection_models import SignalDetectionModel  # noqa: E402
from celldetective.signals import detect_events_with_model  # noqa: E402
from celldetective.utils.event_status import resolve_unobserved_events  # noqa: E402

model = SignalDetectionModel(pretrained=os.path.join(HERE, "zenodo_models", "lysis_H_PI", "lysis_H_PI"))
results = {}
pairs = itertools.product(
    ["dead_nuclei_channel_circle_10_mean", "dead_nuclei_channel_mean"],
    ["live_nuclei_channel_circle_10_mean", "live_nuclei_channel_mean"],
)
for columns in pairs:
    collected = []
    by_movie = {}
    for movie, ds in movies("db-si-NucPI"):
        df = table(ds, list(columns))
        resolve_unobserved_events(df, "class_truth", "t_truth")
        out, _, (class_col, time_col, _) = detect_events_with_model(df, model, selected_signals=list(columns))
        arrays = per_track(out, class_col, time_col)
        collected.append(arrays)
        by_movie[movie] = score(*arrays)["recall"][0]
    result = score(*(np.concatenate(a) for a in zip(*collected)))
    result["event_recall_per_movie"] = by_movie
    results["+".join(columns)] = result
    print("+".join(columns), json.dumps(result), flush=True)

with open(os.path.join(HERE, "h_pi_channels.json"), "w") as f:
    json.dump(results, f, indent=1)
