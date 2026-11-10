"""Shipped models through the detection route, against the annotated Zenodo sets."""
import json
import os
import sys

import numpy as np

WORKTREE, LOCAL_MODELS = sys.argv[1], sys.argv[2]
sys.path.insert(0, WORKTREE)
from eval_common import HERE, movies, per_track, score, table  # noqa: E402

from celldetective.event_detection_models import SignalDetectionModel  # noqa: E402
from celldetective.signals import detect_events_with_model  # noqa: E402
from celldetective.utils.event_status import resolve_unobserved_events  # noqa: E402

DOWNLOADED = os.path.join(HERE, "zenodo_models", "lysis_H_PI", "lysis_H_PI")
DEAD, LIVE = "dead_nuclei_channel_circle_10_mean", "live_nuclei_channel_circle_10_mean"
CASES = [
    ("NucCond", os.path.join(LOCAL_MODELS, "NucCond"), "db-si-NucCondensation", ["area"]),
    ("lysis_PI_area", os.path.join(LOCAL_MODELS, "lysis_PI_area"), "db-si-NucPI", [DEAD, "area"]),
    ("lysis_H_PI", DOWNLOADED, "db-si-NucPI", [DEAD, LIVE]),
]

results = {}
for name, path, dataset, columns in CASES:
    model = SignalDetectionModel(pretrained=path)
    model.config = {k: v for k, v in model.config.items() if k != "selected_channels"}
    collected, per_movie = [], {}
    for movie, ds in movies(dataset):
        df = table(ds, columns)
        resolve_unobserved_events(df, "class_truth", "t_truth")
        out, _, (class_col, time_col, _) = detect_events_with_model(df, model, selected_signals=columns)
        arrays = per_track(out, class_col, time_col)
        collected.append(arrays)
        per_movie[movie] = score(*arrays)["accuracy"]
    results[name] = score(*(np.concatenate(a) for a in zip(*collected)))
    results[name]["per_movie_accuracy"] = per_movie
    print(name, json.dumps(results[name]), flush=True)

with open(os.path.join(HERE, "model_results.json"), "w") as f:
    json.dump(results, f, indent=1)
