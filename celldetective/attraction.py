"""
Are effectors drawn to the targets, or do they meet them by chance?

Counting contacts cannot tell: effectors that bump into targets at random and
then stay on them (adhesion, a synapse) pile up on the targets as well. These
tools compare the motion of the effectors with a null model where the same
tracks are blind to the targets:

-   :func:`shift_tracks` moves all the effector tracks of a position by one
    random offset, wrapping around the field (a toroidal shift). The
    effectors keep their speeds, turns, pauses and arrangement; only their
    coupling to the targets is lost.
-   :func:`arrival_test` counts how often a free effector arrives in contact
    with a target, per step of free motion, and compares this arrival rate
    with the rates of shifted tracks: a permutation test, per position and
    pooled over the positions.
-   :func:`approach_bias` measures, by distance to the nearest target, how
    much free effectors head towards it: the mean cosine of the angle between
    their step and the direction of the target, 0 for a blind effector. The
    same statistic on shifted tracks gives the null band.

Both take ``time_bins``, to follow an attraction that builds up during the
movie (a recruitment that starts after the first kills, say): each window is
compared with the shifted tracks over the same window, so a change of the
effectors' motility over time is in the null as well.

An effector is in contact when its centre is within ``contact_distance`` of
the centre of a target (take the sum of the two radii): shifted tracks have
no masks, and the real and shifted tracks must be scored the same way.

Only the steps an effector starts free, in the real data, are scored, for the
real tracks and for the shifted ones alike. The time effectors spend on the
targets thus counts neither way, and a long retention on the targets does not
make effectors look attracted: the test is about how they arrive, not how
long they stay.

The positions, not the cells, are the independent replicates: cells of one
field share its geometry and its history. The pooled test shifts every
position on its own, and the per-position ratios can be compared between
conditions (cognate against non-cognate targets, say).

Examples
--------
>>> result = arrival_test(df_targets, df_effectors, contact_distance=18, shape=(480, 480))
>>> bias = approach_bias(df_targets, df_effectors, contact_distance=18, shape=(480, 480))
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from celldetective import get_logger

logger = get_logger(__name__)

_XY = ["POSITION_X", "POSITION_Y"]


def _group_keys(*tables: pd.DataFrame) -> List[str]:
    return [c for c in ("well", "position") if all(c in t.columns for t in tables)]


def _groups(targets: pd.DataFrame, effectors: pd.DataFrame) -> List[Tuple[tuple, pd.DataFrame, pd.DataFrame]]:
    """``(key, targets, effectors)`` per position, ``key = ()`` without position columns."""
    keys = _group_keys(targets, effectors)
    if not keys:
        return [((), targets, effectors)]
    by_target = dict(iter(targets.groupby(keys)))
    out = []
    for key, eff in effectors.groupby(keys):
        if key in by_target:
            out.append((key if isinstance(key, tuple) else (key,), by_target[key], eff))
    return out


def _field_shape(targets: pd.DataFrame, effectors: pd.DataFrame, shape: Optional[Sequence[float]]) -> np.ndarray:
    """Field size as ``(X, Y)``; from the extent of the cells when ``shape`` is None."""
    if shape is not None:
        return np.array([float(shape[1]), float(shape[0])])
    extent = np.maximum(targets[_XY].max().to_numpy(), effectors[_XY].max().to_numpy())
    logger.warning(
        "No field shape given: using the extent of the cells, %.0f x %.0f px. Pass shape=(Y, X).",
        extent[1], extent[0],
    )
    return np.ceil(extent + 1)


class _Field:
    """The tracks of one position, ready to be scored again and again."""

    def __init__(self, targets: pd.DataFrame, effectors: pd.DataFrame, size: np.ndarray, contact_distance: float):
        eff = effectors.dropna(subset=_XY).sort_values(["TRACK_ID", "FRAME"])
        self.size = size
        self.contact_distance = float(contact_distance)
        self.xy = eff[_XY].to_numpy(float)
        self.frame = eff["FRAME"].to_numpy(int)
        track = eff["TRACK_ID"].to_numpy()
        # The step into row i starts at row i - 1, when it is the previous frame of the same track.
        same = np.r_[False, (track[1:] == track[:-1]) & (np.diff(self.frame) == 1)]
        self.start = np.flatnonzero(same) - 1
        self.end = self.start + 1
        self.rows = {f: np.flatnonzero(self.frame == f) for f in np.unique(self.frame)}
        self.trees = {}
        for f, df in targets.dropna(subset=_XY).groupby("FRAME"):
            pts = df[_XY].to_numpy(float)
            self.trees[int(f)] = (cKDTree(pts), pts)
        self.real = self.nearest(self.xy)

    def nearest(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Distance and vector from each row to the nearest target of its frame."""
        dist = np.full(len(xy), np.inf)
        vec = np.full((len(xy), 2), np.nan)
        for f, rows in self.rows.items():
            if f not in self.trees:
                continue
            tree, pts = self.trees[f]
            d, k = tree.query(xy[rows])
            dist[rows] = d
            vec[rows] = pts[k] - xy[rows]
        return dist, vec

    def shifted(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Positions shifted by one random offset, and which steps crossed the wrap."""
        xy = np.mod(self.xy + rng.uniform(0, self.size), self.size)
        raw = self.xy[self.end] - self.xy[self.start]
        wrapped = xy[self.end] - xy[self.start]
        return xy, np.any(np.abs(wrapped - raw) > 1e-6, axis=1)

    def steps(self, xy: Optional[np.ndarray] = None, seam: Optional[np.ndarray] = None,
              border: float = 0.0) -> Dict[str, np.ndarray]:
        """Scores of the steps started free in the real data, on the real tracks
        (``xy`` None) or on shifted ones; a shifted step must also start free in
        its own geometry and not cross the wrap."""
        real_dist, _ = self.real
        if xy is None:
            xy, (dist, vec) = self.xy, self.real
        else:
            dist, vec = self.nearest(xy)
        s, e = self.start, self.end
        keep = (real_dist[s] > self.contact_distance) & (dist[s] > self.contact_distance) & np.isfinite(dist[s])
        if seam is not None:
            keep &= ~seam
        s, e = s[keep], e[keep]
        step = xy[e] - xy[s]
        length = np.hypot(*step.T)
        to_target = vec[s] / dist[s, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.sum(step * to_target, axis=1) / length
        inside = np.all((xy[s] >= border) & (xy[s] <= self.size - border), axis=1)
        return {"arrival": dist[e] <= self.contact_distance, "distance": dist[s], "cos": cos,
                "length": length, "inside": inside, "frame": self.frame[s]}


def _fields(targets, effectors, contact_distance, shape) -> List[Tuple[tuple, _Field]]:
    size = _field_shape(targets, effectors, shape)
    return [(key, _Field(t, e, size, contact_distance)) for key, t, e in _groups(targets, effectors)]


def shift_tracks(
    effectors: pd.DataFrame, shape: Sequence[float], seed: Optional[int] = None
) -> pd.DataFrame:
    """Shift all the tracks of each position by one random offset, wrapping
    around the field.

    The shifted effectors move exactly as the real ones did, but their
    positions are unrelated to those of the targets: a population blind to
    the targets, to score with any tool in place of the real one.

    Parameters
    ----------
    effectors : DataFrame
        Trajectories with ``TRACK_ID``, ``FRAME``, ``POSITION_X`` and
        ``POSITION_Y``, and ``well`` / ``position`` columns for several
        positions (each gets its own offset).
    shape : tuple
        Field size ``(Y, X)`` in the units of the positions.
    seed : int, optional
        Random seed.

    Returns
    -------
    DataFrame
        A copy with shifted positions and ``shift_seam``, True when the
        step into the row crossed the edge of the field: the jump is an
        artefact of the wrap, not a move.
    """
    rng = np.random.default_rng(seed)
    size = np.array([float(shape[1]), float(shape[0])])
    keys = [c for c in ("well", "position") if c in effectors.columns]
    out = effectors.sort_values(keys + ["TRACK_ID", "FRAME"]).copy()
    tracks = keys + ["TRACK_ID"]
    raw = out.groupby(tracks)[_XY].diff()
    groups = out.groupby(keys).groups.values() if keys else [out.index]
    for idx in groups:
        out.loc[idx, _XY] = np.mod(out.loc[idx, _XY].to_numpy(float) + rng.uniform(0, size), size)
    moved = out.groupby(tracks)[_XY].diff()
    out["shift_seam"] = ((moved - raw).abs() > 1e-6).any(axis=1)
    return out


def _time_edges(time_bins, fields: List[Tuple[tuple, _Field]]) -> np.ndarray:
    """Frame edges of the time windows, ``[first, last + 1]`` when ``time_bins`` is None."""
    first = min(int(f.frame.min()) for _, f in fields)
    last = max(int(f.frame.max()) for _, f in fields)
    if time_bins is None:
        return np.array([first, last + 1])
    if np.ndim(time_bins) == 0:
        return np.unique(np.round(np.linspace(first, last + 1, int(time_bins) + 1)).astype(int))
    return np.asarray(time_bins, int)


def _window_columns(edges: np.ndarray, w: int, time_bins) -> dict:
    return {} if time_bins is None else {"window_start": int(edges[w]), "window_end": int(edges[w + 1])}


def _p_value(observed: float, null: np.ndarray, alternative: str) -> float:
    null = null[np.isfinite(null)]
    n = len(null)
    greater = (1 + np.sum(null >= observed)) / (1 + n)
    less = (1 + np.sum(null <= observed)) / (1 + n)
    if alternative == "greater":
        return float(greater)
    if alternative == "less":
        return float(less)
    return float(min(1.0, 2 * min(greater, less)))


def arrival_test(
    targets: pd.DataFrame,
    effectors: pd.DataFrame,
    contact_distance: float,
    shape: Optional[Sequence[float]] = None,
    n_shifts: int = 199,
    time_bins: Optional[Union[int, Sequence[int]]] = None,
    alternative: str = "greater",
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Do free effectors arrive on targets more often than blind ones would?

    The arrival rate is the fraction of the steps an effector starts free
    that end in contact with a target. It is compared with the rates of the
    same tracks shifted at random (:func:`shift_tracks`), which keep the
    motion of the effectors and lose any link to the targets.

    ``contact_distance`` must reach at least as far as the effectors catch
    the targets: an effector that grabs a target it grazes ends in contact
    from a distance where a shifted track would only pass by, and blind
    effectors would then look attracted. Repeat the test over a few
    distances; a real attraction holds at all of them.

    Parameters
    ----------
    targets, effectors : DataFrame
        Trajectories with ``TRACK_ID``, ``FRAME``, ``POSITION_X`` and
        ``POSITION_Y``; ``well`` and ``position`` columns, when both tables
        have them, split the positions.
    contact_distance : float
        Distance between centres below which an effector touches a target,
        in the units of the positions: the sum of the two radii, or a little
        more.
    shape : tuple, optional
        Field size ``(Y, X)``; from the extent of the cells if None (with a
        warning: the shifts need the true field).
    n_shifts : int
        Number of shifted copies of each position.
    time_bins : int or sequence of int, optional
        Test each time window on its own, for an attraction that builds up
        or fades during the movie: a number of equal windows, or their frame
        edges (a step belongs to the window of its first frame). The whole
        movie if None.
    alternative : {"greater", "less", "two-sided"}
        "greater" tests attraction, "less" avoidance.
    seed : int, optional
        Random seed of the shifts.

    Returns
    -------
    DataFrame
        One row per position, then a pooled row (``all``), for each time
        window (``window_start`` and ``window_end``, in frames, with
        ``time_bins``), with ``arrivals``, ``free_steps``, ``rate`` (arrivals
        per free step), ``null_rate`` (mean over the shifts) with its 95%
        range ``null_low`` / ``null_high``, ``ratio`` (``rate / null_rate``,
        above 1 for attraction) and the permutation ``p_value``.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative is 'greater', 'less' or 'two-sided'.")
    rng = np.random.default_rng(seed)
    fields = _fields(targets, effectors, contact_distance, shape)
    if not fields:
        raise ValueError("No position has both targets and effectors.")
    keys = _group_keys(targets, effectors)
    edges = _time_edges(time_bins, fields)
    n_windows = len(edges) - 1

    def counts(steps: Dict[str, np.ndarray]) -> np.ndarray:
        """Arrivals and free steps per time window."""
        w = np.digitize(steps["frame"], edges) - 1
        ok = (w >= 0) & (w < n_windows)
        return np.stack([np.bincount(w[ok], weights=steps["arrival"][ok], minlength=n_windows),
                         np.bincount(w[ok], minlength=n_windows)], axis=-1)

    observed = np.zeros((len(fields), n_windows, 2))
    null = np.zeros((n_shifts, len(fields), n_windows, 2))
    for g, (_, field) in enumerate(fields):
        observed[g] = counts(field.steps())
        for s in range(n_shifts):
            null[s, g] = counts(field.steps(*field.shifted(rng)))

    def row(obs, nul, label):
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = obs[0] / obs[1]
            rates = nul[:, 0] / nul[:, 1]
        finite = np.isfinite(rates).any()
        null_rate = float(np.nanmean(rates)) if finite else np.nan
        return {**label, "arrivals": int(obs[0]), "free_steps": int(obs[1]), "rate": rate,
                "null_rate": null_rate, "null_low": np.nanpercentile(rates, 2.5) if finite else np.nan,
                "null_high": np.nanpercentile(rates, 97.5) if finite else np.nan,
                "ratio": rate / null_rate if null_rate else np.nan,
                "p_value": _p_value(rate, rates, alternative) if finite and np.isfinite(rate) else np.nan}

    rows = []
    for w in range(n_windows):
        window = _window_columns(edges, w, time_bins)
        rows += [row(observed[g, w], null[:, g, w], {**window, **dict(zip(keys, key))})
                 for g, (key, _) in enumerate(fields)]
        if keys:
            rows.append(row(observed[:, w].sum(axis=0), null[:, :, w].sum(axis=1),
                            {**window, **{k: "all" for k in keys}}))
    return pd.DataFrame(rows)


def approach_bias(
    targets: pd.DataFrame,
    effectors: pd.DataFrame,
    contact_distance: float,
    shape: Optional[Sequence[float]] = None,
    bins: Union[int, Sequence[float]] = 8,
    max_distance: Optional[float] = None,
    border: Optional[float] = None,
    min_step: float = 0.5,
    n_shifts: int = 99,
    time_bins: Optional[Union[int, Sequence[int]]] = None,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """How much free effectors head towards the nearest target, by distance.

    For each step an effector starts and ends free, the cosine of the angle
    between the step and the direction of the nearest target: 1 straight
    towards it, -1 straight away, 0 on average for an effector blind to the
    targets. The step into a contact is left out: a cell that docks or
    spreads onto a target moves its centre towards it, whatever brought it
    there. Shifted tracks (:func:`shift_tracks`) give the null band of the
    mean cosine in each distance bin, which also absorbs the biases of the
    method itself: just outside contact, the steps left are those that did
    not end in contact, and they lean away from the target, for blind
    effectors too.

    Parameters
    ----------
    targets, effectors, contact_distance, shape
        As in :func:`arrival_test`.
    bins : int or sequence of float
        Number of distance bins between ``contact_distance`` and
        ``max_distance``, or the bin edges.
    max_distance : float, optional
        Upper edge of the bins; five times ``contact_distance`` if None.
    border : float, optional
        Steps that start closer than this to an edge of the field are left
        out: the edges bias the direction of the nearest target, which lies
        inwards, and the shifted tracks do not keep this bias.
        ``max_distance`` if None.
    min_step : float
        Steps shorter than this, with no direction to speak of, are left out.
    n_shifts : int
        Number of shifted copies of each position for the null band; 0 for
        none.
    time_bins : int or sequence of int, optional
        Time windows, as in :func:`arrival_test`.
    seed : int, optional
        Random seed of the shifts.

    Returns
    -------
    DataFrame
        One row per distance bin (and time window, with ``window_start`` and
        ``window_end``): ``distance`` (bin centre), ``low``, ``high``, ``n``
        (steps), ``mean_cos`` with its standard error ``sem``, and with
        shifts ``null_cos`` (mean over the shifts) with its 95% range
        ``null_low`` / ``null_high``.
    """
    rng = np.random.default_rng(seed)
    if max_distance is None:
        max_distance = 5 * contact_distance
    if border is None:
        border = max_distance
    edges = np.linspace(contact_distance, max_distance, bins + 1) if np.ndim(bins) == 0 else np.asarray(bins, float)
    fields = _fields(targets, effectors, contact_distance, shape)
    if not fields:
        raise ValueError("No position has both targets and effectors.")
    t_edges = _time_edges(time_bins, fields)
    n_b, n_w = len(edges) - 1, len(t_edges) - 1

    def binned(parts: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Steps, mean cosine and its standard error, shaped (windows, distance bins)."""
        d, c, f = (np.concatenate([p[k] for p in parts]) for k in ("distance", "cos", "frame"))
        ok = np.concatenate([p["inside"] & ~p["arrival"] & (p["length"] >= min_step) for p in parts])
        b = np.digitize(d, edges) - 1
        w = np.digitize(f, t_edges) - 1
        ok &= np.isfinite(c) & (b >= 0) & (b < n_b) & (w >= 0) & (w < n_w)
        cell, c = w[ok] * n_b + b[ok], c[ok]
        n = np.bincount(cell, minlength=n_w * n_b)
        total = np.bincount(cell, weights=c, minlength=n_w * n_b)
        squares = np.bincount(cell, weights=c**2, minlength=n_w * n_b)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / n
            var = np.maximum(squares - n * mean**2, 0) / (n - 1)
            sem = np.where(n > 1, np.sqrt(var / n), np.nan)
        return n.reshape(n_w, n_b), mean.reshape(n_w, n_b), sem.reshape(n_w, n_b)

    n, mean, sem = binned([f.steps(border=border) for _, f in fields])
    if n_shifts:
        null = np.array([binned([f.steps(*f.shifted(rng), border=border) for _, f in fields])[1]
                         for _ in range(n_shifts)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # bins no shift reaches
            null_stats = (np.nanmean(null, axis=0), *np.nanpercentile(null, [2.5, 97.5], axis=0))
    frames = []
    for w in range(n_w):
        out = pd.DataFrame({**_window_columns(t_edges, w, time_bins),
                            "distance": (edges[:-1] + edges[1:]) / 2, "low": edges[:-1], "high": edges[1:],
                            "n": n[w], "mean_cos": mean[w], "sem": sem[w]})
        if n_shifts:
            out["null_cos"], out["null_low"], out["null_high"] = (x[w] for x in null_stats)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)

