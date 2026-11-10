"""lysis_H_PI: what differs between the movies, and the events it misses?"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from eval_common import HERE, movies, table  # noqa: E402

from celldetective.event_detection_models import SignalDetectionModel  # noqa: E402
from celldetective.signals import detect_events_with_model  # noqa: E402
from celldetective.utils.event_status import resolve_unobserved_events  # noqa: E402

DEAD, LIVE = "dead_nuclei_channel_circle_10_mean", "live_nuclei_channel_circle_10_mean"
model = SignalDetectionModel(pretrained=os.path.join(HERE, "zenodo_models", "lysis_H_PI", "lysis_H_PI"))
rows = []
for movie, ds in movies("db-si-NucPI"):
    df = table(ds, [DEAD, LIVE])
    resolve_unobserved_events(df, "class_truth", "t_truth")
    out, _, (cls, t, _) = detect_events_with_model(df.copy(), model, selected_signals=[DEAD, LIVE])
    last = int(df["FRAME"].max())
    for tid, g in out.groupby("TRACK_ID"):
        g = g.sort_values("FRAME")
        truth = g["class_truth"].iloc[0]
        t0 = g["t_truth"].iloc[0]
        dead, live = g[DEAD].to_numpy(), g[LIVE].to_numpy()
        row = {"movie": movie, "n_frames": last + 1, "truth": truth, "pred": g[cls].iloc[0], "t_truth": t0, "t_pred": g[t].iloc[0],
               "dead_start": np.nanmedian(dead[:3]), "live_start": np.nanmedian(live[:3]),
               "dead_end": np.nanmedian(dead[-3:]), "live_end": np.nanmedian(live[-3:])}
        if truth == 0 and t0 >= 0:
            i = int(max(0, min(len(dead) - 1, round(t0))))
            before = dead[max(0, i - 4):max(1, i)]
            after = dead[i + 1:i + 5]
            row["dead_rise"] = np.nanmedian(after) - np.nanmedian(before) if len(after) else np.nan
            row["dead_before"] = np.nanmedian(before)
            row["live_before"] = np.nanmedian(live[max(0, i - 4):max(1, i)])
        rows.append(row)

r = pd.DataFrame(rows)
r.to_csv(os.path.join(HERE, "h_pi_tracks.csv"), index=False)
pd.set_option("display.width", 220, "display.max_columns", 40)
events = r[r.truth == 0]
print("Per movie (all tracks, then annotated events)")
summary = r.groupby("movie").agg(n=("truth", "size"), frames=("n_frames", "first"),
                                 dead_start=("dead_start", "median"), live_start=("live_start", "median"),
                                 live_start_p90=("live_start", lambda v: v.quantile(.9)),
                                 dead_end_p99=("dead_end", lambda v: v.quantile(.99)))
summary["ev_n"] = events.groupby("movie").size()
summary["ev_recall"] = events.groupby("movie")["pred"].apply(lambda v: (v == 0).mean())
summary["ev_t_median"] = events.groupby("movie")["t_truth"].median()
summary["ev_dead_rise"] = events.groupby("movie")["dead_rise"].median()
summary["ev_dead_before"] = events.groupby("movie")["dead_before"].median()
summary["ev_live_before"] = events.groupby("movie")["live_before"].median()
print(summary.round(2).to_string())
print("\nAnnotated events, called event vs class 2")
print(events[events.pred.isin([0, 2])].groupby("pred")[["t_truth", "dead_before", "dead_rise", "live_before", "dead_start", "live_start"]].median().round(2).to_string())
print(events.assign(early=pd.cut(events.t_truth, [-1, 5, 10, 20, 40, 200])).groupby("early", observed=True)["pred"].agg(n="size", recall=lambda v: (v == 0).mean(), to_2=lambda v: (v == 2).mean()).round(2).to_string())
