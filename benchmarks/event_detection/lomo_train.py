"""Leave one movie out: retrain with the GUI defaults, test on the movie left out.

Each fold runs in its own process (GPU memory is released between folds) and is
skipped when its result exists, so that a run can be resumed.
Usage: lomo_train.py REPOSITORY [MODEL]        every fold, of one model if named, then shipped baselines
       lomo_train.py REPOSITORY MODEL MOVIE    one fold
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.environ.get("CELLDETECTIVE_EVAL_DIR", os.getcwd())  # the data and the results
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORKTREE = sys.argv[1]
sys.path.insert(0, WORKTREE)
OUT = os.path.join(HERE, "lomo")
LOCAL = os.path.join(WORKTREE, "celldetective", "models", "signal_detection")
DEAD = "dead_nuclei_channel_circle_10_mean"
LIVE = "live_nuclei_channel_circle_10_mean"

CASES = {
    "lysis_H_PI": dict(
        dataset="db-si-NucPI", columns=[DEAD, LIVE], options=["dead_nuclei_channel", "live_nuclei_channel"],
        label="lysis", values=[[0.01, 99.999]] * 2,
        shipped=os.path.join(HERE, "zenodo_models", "lysis_H_PI", "lysis_H_PI"),
    ),
    "lysis_PI_area": dict(
        dataset="db-si-NucPI", columns=[DEAD, "area"], options=["dead_nuclei_channel", "area"],
        label="death", values=[[0.1, 99.99]] * 2, shipped=os.path.join(LOCAL, "lysis_PI_area"),
    ),
    "NucCond": dict(
        dataset="db-si-NucCondensation", columns=["area"], options=["area"],
        label="nucShrink", values=[[0.1, 99.99]], shipped=os.path.join(LOCAL, "NucCond"),
    ),
}


def set_paths(dataset):
    return sorted(glob.glob(os.path.join(HERE, "zenodo_data", dataset, "**", "*.npy"), recursive=True))


def name_of(path):
    return os.path.basename(path)[:-4]


def predict(model, ds, columns, destination):
    from eval_common import table
    from celldetective.signals import detect_events_with_model
    from celldetective.utils.event_status import resolve_unobserved_events

    df = table(ds, columns)
    resolve_unobserved_events(df, "class_truth", "t_truth")
    model.config = {k: v for k, v in model.config.items() if k != "selected_channels"}
    out, _, (cls, t, _) = detect_events_with_model(df, model, selected_signals=columns)
    first = out.groupby("TRACK_ID")[["class_truth", "t_truth", cls, t]].first()
    first.columns = ["class_truth", "t_truth", "class_pred", "t_pred"]
    first.to_csv(destination)


def fold(name, movie):
    import safe_npy
    from celldetective.event_detection_models import SignalDetectionModel

    case = CASES[name]
    paths = set_paths(case["dataset"])
    train = [list(safe_npy.load(p)) for p in paths if name_of(p) != movie]
    test = list(safe_npy.load(next(p for p in paths if name_of(p) == movie)))
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    start = time.time()
    n = len(case["options"])
    model = SignalDetectionModel(
        channel_option=case["options"], model_signal_length=128, n_channels=n, label=case["label"]
    )
    model.fit_from_directory(
        train, normalize=True, normalization_percentile=[True] * n, normalization_values=case["values"],
        normalization_clip=[True] * n, channel_option=case["options"], model_name=movie,
        target_directory=os.path.join(OUT, name), augment=True, augmentation_factor=2,
        validation_split=0.25, test_split=0.0, batch_size=64, epochs=300, learning_rate=0.001, show_plots=False,
    )
    trained = SignalDetectionModel(pretrained=os.path.join(OUT, name, movie))
    predict(trained, test, case["columns"], os.path.join(OUT, name, f"{movie}.csv"))
    print(f"DONE {name} {movie} {time.time() - start:.0f}s", flush=True)


def baseline(name):
    import safe_npy
    from celldetective.event_detection_models import SignalDetectionModel

    case = CASES[name]
    model = SignalDetectionModel(pretrained=case["shipped"])
    folder = os.path.join(OUT, name, "shipped")
    os.makedirs(folder, exist_ok=True)
    for path in set_paths(case["dataset"]):
        predict(model, list(safe_npy.load(path)), case["columns"], os.path.join(folder, f"{name_of(path)}.csv"))
    print(f"DONE shipped {name}", flush=True)


if len(sys.argv) == 4 and sys.argv[2] == "baseline":
    baseline(sys.argv[3])
elif len(sys.argv) == 4:
    fold(sys.argv[2], sys.argv[3])
else:
    env = dict(os.environ, MPLBACKEND="Agg")
    for name, case in CASES.items():
        if len(sys.argv) == 3 and name != sys.argv[2]:
            continue
        for path in set_paths(case["dataset"]):
            movie = name_of(path)
            if not os.path.exists(os.path.join(OUT, name, f"{movie}.csv")):
                subprocess.run([sys.executable, os.path.abspath(__file__), WORKTREE, name, movie], env=env)
        if not os.path.isdir(os.path.join(OUT, name, "shipped")):
            subprocess.run([sys.executable, os.path.abspath(__file__), WORKTREE, "baseline", name], env=env)
    print("ALL DONE", flush=True)
