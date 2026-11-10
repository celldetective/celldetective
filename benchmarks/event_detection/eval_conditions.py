"""Times from conditions: on real annotated traces, and in a noise sweep."""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

WORKTREE = sys.argv[1]
sys.path.insert(0, WORKTREE)
from eval_common import HERE, movies, per_track, score, table  # noqa: E402
from skimage.filters import threshold_otsu  # noqa: E402

from celldetective.measure import classify_irreversible_events, classify_tracks_from_query  # noqa: E402
from celldetective.utils.event_status import resolve_unobserved_events  # noqa: E402

warnings.simplefilter("ignore")
DEAD = "dead_nuclei_channel_circle_10_mean"
CASES = [
    ("PI above threshold", "db-si-NucPI", DEAD, ">", [50, 60, 70, 80, 85, 90, 95, 97, 99]),
    ("area below threshold", "db-si-NucCondensation", "area", "<", [1, 3, 5, 10, 15, 20, 30, 40, 50]),
]


def run(df, column, op, threshold):
    out = classify_tracks_from_query(
        df.copy(), "e", f"`{column}` {op} {threshold:.6g}", irreversible_event=True
    )
    return per_track(out, "class_e", "t_e")


def total(arrays):
    return score(*(np.concatenate(a) for a in zip(*arrays)))


results = {"real": {}, "synthetic": []}
for label, dataset, column, op, percentiles in CASES:
    tables = []
    for _, ds in movies(dataset):
        df = table(ds, [column])
        resolve_unobserved_events(df, "class_truth", "t_truth")
        tables.append(df)

    otsu = [run(df, column, op, threshold_otsu(df[column].dropna().to_numpy())) for df in tables]
    entry = {"otsu_per_movie": total(otsu), "by_percentile": {}}
    for q in percentiles:
        arrays = [run(df, column, op, np.nanpercentile(df[column], q)) for df in tables]
        entry["by_percentile"][q] = total(arrays)
    best = max(entry["by_percentile"], key=lambda q: entry["by_percentile"][q]["event_f1"])
    entry["best_percentile"] = best
    results["real"][label] = entry
    print(label, "otsu", json.dumps(entry["otsu_per_movie"]), flush=True)
    print(label, "best", best, json.dumps(entry["by_percentile"][best]), flush=True)


def synthetic(flicker, width, n=300, n_frames=100, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(20, 80, n)
    frames = np.arange(n_frames)
    rows, none_rows = [], []
    for tid, t in enumerate(truth):
        p = (frames >= t).astype(float) if width == 0 else 1 / (1 + np.exp(-(frames - t) / width))
        status = (rng.random(n_frames) < p).astype(float)
        flips = rng.random(n_frames) < flicker
        status[flips] = 1 - status[flips]
        rows.append(pd.DataFrame({"TRACK_ID": tid, "FRAME": frames, "class_e": status, "status_e": status}))
        negative = (rng.random(n_frames) < flicker).astype(float)
        none_rows.append(pd.DataFrame({"TRACK_ID": tid, "FRAME": frames, "class_e": negative, "status_e": negative}))
    out = classify_irreversible_events(pd.concat(rows, ignore_index=True), "class_e", r2_threshold=0.5)
    first = out.groupby("TRACK_ID")[["class_e", "t_e"]].first()
    called = first["class_e"].to_numpy() == 0
    error = first["t_e"].to_numpy()[called] - truth[called]
    negatives = classify_irreversible_events(pd.concat(none_rows, ignore_index=True), "class_e", r2_threshold=0.5)
    false_events = float((negatives.groupby("TRACK_ID")["class_e"].first() == 0).mean())
    return {
        "flicker": flicker, "width": width, "recall": float(called.mean()),
        "other_class_1": float((first["class_e"] == 1).mean()),
        "other_class_2": float((first["class_e"] == 2).mean()),
        "median_abs_dt": float(np.median(np.abs(error))) if called.any() else None,
        "p90_abs_dt": float(np.percentile(np.abs(error), 90)) if called.any() else None,
        "bias_dt": float(error.mean()) if called.any() else None,
        "false_event_rate": false_events,
    }


for flicker in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
    for width in (0, 2, 5, 10):
        row = synthetic(flicker, width)
        results["synthetic"].append(row)
        print(json.dumps(row), flush=True)

with open(os.path.join(HERE, "condition_results.json"), "w") as f:
    json.dump(results, f, indent=1)
