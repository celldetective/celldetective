"""Summarize leave-one-movie-out results against the shipped models."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

here = os.environ.get("CELLDETECTIVE_EVAL_DIR", os.getcwd())  # the data and the results
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import score  # noqa: E402

BINS = [-1, 3, 5, 10, 20, 200]
out = {}
for model in ("lysis_H_PI", "lysis_PI_area", "NucCond"):
    folder = os.path.join(here, "lomo", model)
    for variant, pattern in (("left-out", "*.csv"), ("shipped", os.path.join("shipped", "*.csv"))):
        files = sorted(glob.glob(os.path.join(folder, pattern)))
        if variant == "shipped":  # the same movies as the retrained models
            files = [f for f in files if os.path.exists(os.path.join(folder, os.path.basename(f)))]
        if not files:
            print(model, variant, "no results")
            continue
        parts = [pd.read_csv(f).assign(movie=os.path.basename(f)[:-4]) for f in files]
        d = pd.concat(parts)
        s = score(d.class_truth.to_numpy(), d.t_truth.to_numpy(), d.class_pred.to_numpy(), d.t_pred.to_numpy())
        per = [score(p.class_truth.to_numpy(), p.t_truth.to_numpy(), p.class_pred.to_numpy(), p.t_pred.to_numpy()) for p in parts]
        ev = d[d.class_truth == 0]
        early = ev.assign(b=pd.cut(ev.t_truth, BINS)).groupby("b", observed=True).class_pred
        s["movies"] = len(files)
        s["accuracy_range"] = [min(p["accuracy"] for p in per), max(p["accuracy"] for p in per)]
        s["event_recall_range"] = [min(p["recall"][0] for p in per if p["recall"][0] is not None), max(p["recall"][0] for p in per if p["recall"][0] is not None)]
        s["recall_by_time"] = {str(k): [int(len(v)), round(float((v == 0).mean()), 2), round(float((v == 2).mean()), 2)] for k, v in early}
        out[f"{model} {variant}"] = s
        print(f"{model:14s} {variant:9s} movies={len(files)} n={s['n']} acc={s['accuracy']:.3f} [{s['accuracy_range'][0]:.2f}-{s['accuracy_range'][1]:.2f}] bal={s['balanced_accuracy']:.3f}"
              f" evR={s['recall'][0]:.3f} [{s['event_recall_range'][0]:.2f}-{s['event_recall_range'][1]:.2f}] evP={s['precision'][0]:.3f} F1={s['event_f1']:.3f}"
              f" noEvR={s['recall'][1]:.3f} else R={s['recall'][2]:.3f} P={s['precision'][2]:.3f} medDt={s['median_abs_dt']:.2f} p90={s['p90_abs_dt']:.2f} bias={s['bias_dt']:+.2f}")
        print("    by t_truth [n, event, class2]:", s["recall_by_time"])
        print("    confusion", s["confusion"])
json.dump(out, open(os.path.join(here, "lomo_results.json"), "w"), indent=1)
