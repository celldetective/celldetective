"""EVT-2: does the padding after a track end change what the shipped models call?

Annotated tracks all reach the last frame of their movie, so tracks that end early
are simulated by cutting them after their annotated event. The tail is then filled
as in training (zeros) or as inference does today (hold the last value).
"""
import glob
import json
import os
import sys

import numpy as np

here = os.environ.get("CELLDETECTIVE_EVAL_DIR", os.getcwd())  # the data and the results
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_npy  # noqa: E402

WORKTREE = sys.argv[1]
MODELS = sys.argv[2]
sys.path.insert(0, WORKTREE)
from celldetective.event_detection_models import SignalDetectionModel  # noqa: E402

CASES = [
    ("NucCond", "db-si-NucCondensation", ["area"]),
    ("lysis_PI_area", "db-si-NucPI", ["dead_nuclei_channel_mean", "area"]),
    ("lysis_PI_area", "db-si-NucPI", ["dead_nuclei_channel_circle_10_mean", "area"]),
]
MARGIN = 5  # frames kept after the event, and cut before the last frame


def build(ds, columns, last, cuts=None, tail="zeros"):
    """Signals on the frame axis, cut at ``cuts`` (one per track) when given."""

    x = np.zeros((len(ds), last + 2, len(columns)))
    for k, d in enumerate(ds):
        frames = d["FRAME"].astype(int)
        keep = frames <= (cuts[k] if cuts is not None else last)
        frames = frames[keep]
        for c, col in enumerate(columns):
            values = np.nan_to_num(np.asarray(d[col], dtype=float)[keep])
            x[k, frames, c] = values
            if tail == "hold":
                x[k, frames.max():, c] = values[-1]
    return x


def score(cls_true, t_true, cls_pred, t_pred):
    ev = cls_true == 0
    both = ev & (cls_pred == 0)
    err = np.abs(t_pred[both] - t_true[both])
    return {
        "n": int(len(cls_true)),
        "accuracy": float(np.mean(cls_true == cls_pred)),
        "event_recall": float(np.mean(cls_pred[ev] == 0)),
        "no_event_recall": float(np.mean(cls_pred[cls_true == 1] == 1)),
        "false_events": int(np.sum((cls_true == 1) & (cls_pred == 0))),
        "median_abs_dt": float(np.median(err)) if len(err) else None,
        "within_3": float(np.mean(err <= 3)) if len(err) else None,
    }


results = {}
for model_name, dataset, columns in CASES:
    model = SignalDetectionModel(pretrained=os.path.join(MODELS, model_name))
    rng = np.random.default_rng(0)
    collected = {k: ([], [], [], []) for k in ("full", "cut_zeros", "cut_hold")}
    n_cut = 0
    for path in sorted(glob.glob(os.path.join(here, "zenodo_data", dataset, "**", "*.npy"), recursive=True)):
        ds = list(safe_npy.load(path))
        if not all(c in ds[0] for c in columns):
            continue
        last = max(int(d["FRAME"].max()) for d in ds)
        cls_true = np.array([int(d["class"]) for d in ds])
        t_true = np.array([float(d["time_of_interest"]) for d in ds])

        cuts = np.full(len(ds), -1)
        for k, d in enumerate(ds):
            lo = int(max(10, t_true[k] + MARGIN)) if cls_true[k] == 0 else 10
            hi = last - MARGIN
            if lo <= hi:
                cuts[k] = rng.integers(lo, hi + 1)
        chosen = np.flatnonzero(cuts >= 0)
        n_cut += len(chosen)
        sub = [ds[i] for i in chosen]

        variants = {
            "full": (build(ds, columns, last, tail="hold"), np.arange(len(ds))),
            "cut_zeros": (build(sub, columns, last, cuts[chosen], "zeros"), chosen),
            "cut_hold": (build(sub, columns, last, cuts[chosen], "hold"), chosen),
        }
        for name, (x, idx) in variants.items():
            cp = np.asarray(model.predict_class(x))
            tp = np.asarray(model.predict_time_of_interest(x, class_predictions=cp), dtype=float)
            bucket = collected[name]
            bucket[0].append(cls_true[idx]); bucket[1].append(t_true[idx])
            bucket[2].append(cp); bucket[3].append(tp)

    key = f"{model_name} | {dataset} | {'+'.join(columns)}"
    results[key] = {"tracks_cut": n_cut}
    for name, bucket in collected.items():
        if bucket[0]:
            results[key][name] = score(*(np.concatenate(b) for b in bucket))
    print(key, json.dumps(results[key], indent=1), flush=True)

with open(os.path.join(here, "padding_results.json"), "w") as f:
    json.dump(results, f, indent=1)
