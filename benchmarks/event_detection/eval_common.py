"""Shared by the model and condition evaluations."""
import glob
import os

import numpy as np
import pandas as pd

import safe_npy

HERE = os.environ.get("CELLDETECTIVE_EVAL_DIR", os.getcwd())  # the data and the results
EXTRA = ["t_firstdetection", "class_firstdetection", "status_firstdetection"]


def movies(dataset):
    pattern = os.path.join(HERE, "zenodo_data", dataset, "**", "*.npy")
    for path in sorted(glob.glob(pattern, recursive=True)):
        yield os.path.basename(path)[:-4], list(safe_npy.load(path))


def table(ds, columns):
    """One row per frame, with the annotated class and time of each track."""

    parts = []
    for k, d in enumerate(ds):
        frames = np.asarray(d["FRAME"], dtype=float)
        part = {"TRACK_ID": k, "FRAME": frames,
                "POSITION_X": np.asarray(d["POSITION_X"], dtype=float),
                "POSITION_Y": np.asarray(d["POSITION_Y"], dtype=float),
                "class_truth": float(d["class"]), "t_truth": float(d["time_of_interest"])}
        for col in columns:
            part[col] = np.asarray(d[col], dtype=float)
        for col in EXTRA:
            if col in d:
                part[col] = np.asarray(d[col], dtype=float)
        parts.append(pd.DataFrame(part))
    df = pd.concat(parts, ignore_index=True)
    df.loc[df["class_truth"] != 0, "t_truth"] = -1.0
    return df


def per_track(df, class_col, time_col):
    first = df.groupby("TRACK_ID")[["class_truth", "t_truth", class_col, time_col]].first()
    return (first["class_truth"].to_numpy(), first["t_truth"].to_numpy(),
            first[class_col].to_numpy(), first[time_col].to_numpy())


def score(cls_true, t_true, cls_pred, t_pred):
    """Three-class confusion (rows are the annotation), and time errors in frames."""

    labels = [0, 1, 2]
    confusion = [[int(np.sum((cls_true == a) & (cls_pred == b))) for b in labels] for a in labels]
    no_class = int(np.isnan(cls_pred).sum())
    recall, precision = {}, {}
    for c in labels:
        truth, pred = cls_true == c, cls_pred == c
        recall[c] = float((truth & pred).sum() / truth.sum()) if truth.sum() else None
        precision[c] = float((truth & pred).sum() / pred.sum()) if pred.sum() else None
    r, p = recall[0], precision[0]
    both = (cls_true == 0) & (cls_pred == 0)
    error = t_pred[both] - t_true[both]
    absolute = np.abs(error)
    return {
        "n": int(len(cls_true)),
        "confusion": confusion,
        "no_class": no_class,
        "accuracy": float(np.mean(cls_true == cls_pred)),
        "balanced_accuracy": float(np.mean([v for v in recall.values() if v is not None])),
        "recall": recall,
        "precision": precision,
        "event_f1": float(2 * p * r / (p + r)) if p and r else 0.0,
        "n_timed": int(both.sum()),
        "mean_abs_dt": float(absolute.mean()) if len(error) else None,
        "median_abs_dt": float(np.median(absolute)) if len(error) else None,
        "p90_abs_dt": float(np.percentile(absolute, 90)) if len(error) else None,
        "bias_dt": float(error.mean()) if len(error) else None,
        "within_1": float(np.mean(absolute <= 1)) if len(error) else None,
        "within_3": float(np.mean(absolute <= 3)) if len(error) else None,
    }
