"""
What affects the hazard of an event of the cells (their death, say).

These tools relate an event of each cell to quantities that may change its
hazard frame by frame: the contacts it receives (with effectors, say, from
the neighborhood counts and the pair table), its own measurements (area,
intensities...) and the conditions of its well (a concentration...):

-   :func:`~celldetective.utils.data_cleaning.collapse_pairs_by_episode`
    (also in the table UI, as a collapse of the pairs) splits the contacts of
    each pair into episodes with a start, an end and a duration;
    :func:`episode_outcomes` tells, for each episode, whether the reference
    cell had the event within a time window, and
    :func:`binned_event_fraction` how often it did per class of episode
    duration: a dose-response.
-   :func:`time_varying_table` writes the long table of a survival analysis
    with time-varying covariates: one row per cell and frame interval
    ``(start, stop]``, holding the covariates of the cell up to the start of
    the interval (its contact exposure, its measurements...) and whether the
    event happened in it.
-   :func:`fit_hazard` fits a Cox proportional-hazards model to it,
    and :func:`fit_hazard_per_neighborhood` repeats the fit for several
    neighborhoods, e.g. contact distances.

The exposure of an interval only uses what happened before it, so that a cell
which dies early is not counted as little exposed because it died (the
immortal-time bias of comparing survival between groups of cells formed by
their total contact time). An optional lag accounts for the delay between a
lethal hit and the moment the death is seen.

Examples
--------
>>> table = time_varying_table(df_targets, "neighborhood_(targets-effectors)_contact_0_px", "death")
>>> fitter, summary = fit_hazard(table, ["in_contact", "cumulative_contact"])
"""

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from celldetective import get_logger

logger = get_logger(__name__)

#: Covariates computed by :func:`time_varying_table`, with what they hold.
CONTACT_COVARIATES: Dict[str, str] = {
    "in_contact": "1 when the cell has at least one neighbor, else 0",
    "n_neighbors": "number of neighbors",
    "weighted_neighbors": "neighbors, each counted 1 / the number of reference cells it touches",
    "cumulative_contact": "time spent with at least one neighbor so far",
    "ongoing_contact": "time since the current contact began, 0 out of contact",
}

_DURATION_UNITS = {"frame": None, "min": 1.0, "h": 60.0}


def _event_columns(event: str) -> Tuple[str, str]:
    """``(class_<event>, t_<event>)`` from ``death``, ``t_death`` or ``class_death``."""
    for prefix in ("t_", "class_"):
        if event.startswith(prefix):
            event = event[len(prefix):]
    return f"class_{event}", f"t_{event}"


def _neighborhood_name(neighborhood: str) -> str:
    """The neighborhood column from any of its derived column names."""
    for prefix in ("status_", "inclusive_count_", "intermediate_count_", "exclusive_count_"):
        if neighborhood.startswith(prefix):
            return neighborhood[len(prefix):]
    return neighborhood


def neighborhood_distance(neighborhood: str) -> float:
    """The distance in pixels in the name of a neighborhood, NaN if none."""
    from celldetective.relative_measurements import extract_neighborhood_settings

    try:
        return float(extract_neighborhood_settings(_neighborhood_name(neighborhood))["distance"])
    except Exception:
        return np.nan


def _time_scale(frame_to_min: float, unit: str) -> float:
    if unit not in _DURATION_UNITS:
        raise ValueError(f"Unknown duration unit {unit!r}; use one of {list(_DURATION_UNITS)}.")
    minutes = _DURATION_UNITS[unit]
    return 1.0 if minutes is None else frame_to_min / minutes


def _reference_events(reference: pd.DataFrame, event: str) -> pd.DataFrame:
    """Per reference cell: the event time (NaN when it did not happen) and
    the last frame it is seen. Cells whose event happened before they were
    observed (class 2) are left out, as in :func:`time_varying_table`."""
    class_col, time_col = _event_columns(event)
    keys = (["position"] if "position" in reference.columns else []) + ["TRACK_ID"]
    agg = reference.groupby(keys).agg(
        t_event=(time_col, "first"),
        last_frame=("FRAME", "max"),
        **({"class_event": (class_col, "first")} if class_col in reference.columns else {}),
    )
    if "class_event" in agg.columns:
        agg = agg[agg["class_event"] != 2].copy()
        happened = (agg["t_event"] >= 0) & (agg["class_event"].isna() | (agg["class_event"] == 0))
    else:
        happened = agg["t_event"] >= 0
    agg["t_event"] = agg["t_event"].where(happened)
    return agg.reset_index().rename(columns={"TRACK_ID": "REFERENCE_ID"})


def episode_outcomes(
    episodes: pd.DataFrame,
    reference: pd.DataFrame,
    event: str,
    window: int,
    align: str = "end",
) -> pd.DataFrame:
    """
    Whether the reference cell had the event within ``window`` frames of each
    contact episode.

    Parameters
    ----------
    episodes : pandas.DataFrame
        Output of :func:`~celldetective.utils.data_cleaning.collapse_pairs_by_episode`.
    reference : pandas.DataFrame
        Table of the reference population, with ``TRACK_ID``, ``FRAME`` and
        the event columns.
    event : str
        The event, e.g. ``death`` for ``class_death`` and ``t_death``.
    window : int
        Number of frames after the alignment frame in which the event counts.
    align : {"end", "start"}
        ``end`` (default): the event is looked for after the end of the
        episode, among cells alive then; the duration is fully observed but
        a cell dying during the contact is left out. ``start``: the event is
        looked for from the start of the episode, among cells alive then;
        the cells dying during the contact count, but their contact may have
        been cut short by the death, which then shortens long contacts.

    Returns
    -------
    pandas.DataFrame
        The episodes kept (the cell alive at the alignment frame and seen for
        the whole window, or having the event in it), with ``t_event`` and
        ``outcome`` (1 when the event happened in the window).
    """
    if align not in ("end", "start"):
        raise ValueError("align must be 'end' or 'start'.")
    events = _reference_events(reference, event)
    keys = (["position"] if "position" in episodes.columns and "position" in events.columns else []) + ["REFERENCE_ID"]
    merged = episodes.merge(events, on=keys, how="inner")
    t0 = merged["t_end"] if align == "end" else merged["t_start"]
    alive = ~(merged["t_event"] <= t0)
    happened = (merged["t_event"] > t0) & (merged["t_event"] <= t0 + window)
    observed = merged["last_frame"] >= t0 + window
    kept = merged[alive & (happened | observed)].copy()
    kept["outcome"] = happened[kept.index].astype(int)
    return kept.drop(columns=["last_frame", "class_event"], errors="ignore")


def binned_event_fraction(
    outcomes: pd.DataFrame,
    bins: Sequence[float],
    column: str = "duration",
) -> pd.DataFrame:
    """
    Fraction of episodes followed by the event, per class of ``column``, with
    a 95% Wilson interval.

    Parameters
    ----------
    outcomes : pandas.DataFrame
        Output of :func:`episode_outcomes`.
    bins : sequence of float
        Bin edges, intervals closed on the left.
    column : str
        The episode property to bin, ``duration`` by default.

    Returns
    -------
    pandas.DataFrame
        ``bin``, ``n``, ``events``, ``fraction``, ``ci_lower``, ``ci_upper``.
    """
    cut = pd.cut(outcomes[column], bins=list(bins), right=False)
    grouped = outcomes.groupby(cut, observed=False)["outcome"]
    n = grouped.size()
    k = grouped.sum()
    z = 1.959964
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return pd.DataFrame(
        {
            "bin": n.index.astype(str),
            "n": n.to_numpy(),
            "events": k.to_numpy(),
            "fraction": p.to_numpy(),
            "ci_lower": (centre - half).to_numpy(),
            "ci_upper": (centre + half).to_numpy(),
        }
    )


def time_varying_table(
    reference: pd.DataFrame,
    neighborhood: Optional[str],
    event: str,
    origin: Optional[str] = None,
    lag: int = 0,
    frame_to_min: float = 1.0,
    duration_unit: str = "min",
    extra_covariates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    The long table of a survival analysis with time-varying contact exposure.

    Each tracked reference cell is followed frame by frame from its origin
    until its event or the end of its track. The interval ``(t, t + 1]``
    gets one row: the exposure of the cell at frame ``t - lag`` (what it had
    received by then) and ``event = 1`` when the event happened in the
    interval, at most once per cell, on its last row.

    Parameters
    ----------
    reference : pandas.DataFrame
        Table of the reference population, with ``TRACK_ID``, ``FRAME``, the
        ``inclusive_count_`` column of the neighborhood (``intermediate_count_``
        too for ``weighted_neighbors``) and the event columns; ``position``
        and ``well`` are kept when present.
    neighborhood : str or None
        The neighborhood, e.g. ``neighborhood_(targets-effectors)_contact_0_px``.
        None for a population without neighborhood: the long table then only
        carries the ``extra_covariates``, measurements of the cell or
        conditions of its well, and no contact covariate.
    event : str
        The event, e.g. ``death`` for ``class_death`` (0 when it happened,
        1 when it did not, 2 when it happened before the observation) and
        ``t_death``.
    origin : str, optional
        Column of the time each cell starts to be followed (e.g.
        ``t_firstdetection``); by default its first frame.
    lag : int, optional
        Frames between the exposure and the interval it explains. The first
        ``lag`` frames of each cell have no exposure yet and are left out.
    frame_to_min : float, optional
        Duration of a frame in minutes.
    duration_unit : {"min", "h", "frame"}
        Unit of ``start``, ``stop`` and of the contact durations.
    extra_covariates : sequence of str, optional
        Other columns of the reference table to carry as covariates, lagged
        like the others: contact-site intensities (``contact_*``), whose
        missing values (no contact) become 0, or any measurement of the cell
        (area, intensity, ...), whose missing values (a frame where the cell
        was not detected) take the last value measured.

    Returns
    -------
    pandas.DataFrame
        ``cell`` (one id per cell, positions included), the ``position``,
        ``well`` and ``TRACK_ID`` of the cell, ``start``, ``stop``, ``event``
        and the covariates of :data:`CONTACT_COVARIATES` (``weighted_neighbors``
        only with the intermediate counts) and ``extra_covariates``.
    """
    class_col, time_col = _event_columns(event)
    if neighborhood is None:
        count_col = weight_col = None
    else:
        neighborhood = _neighborhood_name(neighborhood)
        count_col = f"inclusive_count_{neighborhood}"
        weight_col = f"intermediate_count_{neighborhood}"
    extra = list(extra_covariates or [])
    for col in [c for c in ("TRACK_ID", "FRAME", count_col, time_col, *extra) if c]:
        if col not in reference.columns:
            raise KeyError(f"Column {col} not found in the reference table.")
    if lag < 0:
        raise ValueError("The lag must be positive.")
    scale = _time_scale(frame_to_min, duration_unit)

    keys = [c for c in ("well", "position") if c in reference.columns] + ["TRACK_ID"]
    per_cell = list(dict.fromkeys([time_col] + [c for c in (class_col, origin) if c and c in reference.columns]))
    per_frame = [c for c in (count_col, weight_col, *extra) if c and c in reference.columns]
    data = (
        reference.dropna(subset=["TRACK_ID", "FRAME"])
        .loc[:, list(dict.fromkeys(keys + ["FRAME"] + per_cell + per_frame))]
        .assign(FRAME=lambda d: d["FRAME"].astype(int))
        .sort_values(keys + ["FRAME"])
        .drop_duplicates(keys + ["FRAME"])
    )
    contact = list(CONTACT_COVARIATES) if count_col else []
    columns = ["cell", *keys, "start", "stop", "event", *contact, *extra]
    if data.empty:
        return pd.DataFrame(columns=columns)

    # One row per cell and frame from its first to its last frame: frames
    # missing from a track count as without contact.
    cells = data.groupby(keys).agg(first=("FRAME", "min"), last=("FRAME", "max"),
                                   **{c: (c, "first") for c in per_cell})
    lengths = (cells["last"] - cells["first"] + 1).to_numpy()
    row_of_cell = np.repeat(np.arange(len(cells)), lengths)
    offsets = np.cumsum(lengths) - lengths
    grid = cells.index.to_frame(index=False).iloc[row_of_cell].reset_index(drop=True)
    grid["FRAME"] = np.repeat(cells["first"].to_numpy(), lengths) + np.arange(len(grid)) - offsets[row_of_cell]
    grid = grid.merge(data[keys + ["FRAME"] + per_frame], on=keys + ["FRAME"], how="left")

    covariates = {}
    if count_col:
        n = grid[count_col].fillna(0).to_numpy(dtype=float)
        touching = n > 0
        cumulative = pd.Series(touching).groupby(row_of_cell).cumsum().to_numpy()
        # Frames since the last frame out of contact, without reaching into
        # the previous cell.
        index = np.arange(len(grid))
        last_out = np.maximum.accumulate(np.where(touching, offsets[row_of_cell] - 1, index))
        ongoing = index - last_out
        covariates.update({
            "in_contact": touching.astype(int),
            "n_neighbors": n,
            "cumulative_contact": cumulative * scale,
            "ongoing_contact": ongoing * scale,
        })
    if weight_col and weight_col in grid.columns:
        covariates["weighted_neighbors"] = grid[weight_col].fillna(0).to_numpy(dtype=float)
    # A contact-site measurement is missing because there is no contact: 0.
    # A cell measurement is missing because the cell was not detected on that
    # frame: it keeps its last value; frames before its first measurement
    # stay missing and are left out of the fit.
    contact_site = [c for c in extra if c.startswith("contact_")]
    measured = [c for c in extra if not c.startswith("contact_")]
    filled = pd.concat(
        [grid[contact_site].astype(float).fillna(0),
         grid[measured].astype(float).groupby(row_of_cell).ffill()],
        axis=1,
    )
    for col in extra:
        covariates[col] = filled[col].to_numpy(dtype=float)

    # When each cell is followed, and whether its event is seen.
    t_event = cells[time_col].to_numpy(dtype=float)
    happened = t_event >= 0
    kept = np.ones(len(cells), dtype=bool)
    if class_col in cells.columns:
        cls = cells[class_col].to_numpy(dtype=float)
        kept &= cls != 2  # the event happened before the cell was observed
        happened &= np.isnan(cls) | (cls == 0)
    start = cells["first"].to_numpy() + lag
    if origin is not None and origin in cells.columns:
        t_origin = cells[origin].to_numpy(dtype=float)
        later = t_origin >= 0
        start[later] = np.maximum(start[later], np.ceil(t_origin[later]).astype(int))
    last = cells["last"].to_numpy()
    with np.errstate(invalid="ignore"):
        end_event = np.ceil(np.where(happened, t_event, 0)).astype(int)
    happened &= end_event <= last + 1  # an event after the track is not seen
    end = np.where(happened, end_event, last)
    kept &= ~(happened & (t_event <= start))  # the event happened before the cell is followed

    # Intervals (t, t + 1] for start <= t < end, with the exposure at t - lag.
    frame = grid["FRAME"].to_numpy()
    rows = kept[row_of_cell] & (frame >= start[row_of_cell]) & (frame < end[row_of_cell])
    exposure = np.flatnonzero(rows) - lag
    table = grid.loc[rows, keys].reset_index(drop=True)
    labels = cells.index.to_frame(index=False).astype(str).agg("/".join, axis=1).to_numpy()
    table.insert(0, "cell", labels[row_of_cell[rows]])
    t = frame[rows]
    table["start"] = t * scale
    table["stop"] = (t + 1) * scale
    table["event"] = (happened[row_of_cell[rows]] & (t == end[row_of_cell[rows]] - 1)).astype(int)
    for name, values in covariates.items():
        table[name] = values[exposure]
    return table


def fit_hazard(
    table: pd.DataFrame,
    covariates: Sequence[str],
    strata: Optional[Sequence[str]] = None,
    penalizer: float = 0.0,
    standardize: bool = False,
):
    """
    Fit a Cox proportional-hazards model with time-varying covariates.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`time_varying_table`.
    covariates : sequence of str
        Covariates of the model. Those that do not vary are left out, with a
        warning.
    strata : sequence of str, optional
        Columns whose levels get their own baseline hazard, e.g. ``well`` to
        compare cells only with cells of the same well.
    penalizer : float, optional
        L2 penalty on the coefficients, 0.5 * penalizer * sum(coef ** 2)
        subtracted from the partial log-likelihood, whatever the number of
        cells and frames. A moderate one (0.5) barely moves a ratio supported by
        the data but keeps it finite when a covariate separates the cells with
        and without the event.
    standardize : bool, optional
        Divide each covariate that is not binary by its standard deviation,
        so that its hazard ratio is per standard deviation and comparable
        between covariates of different units (an area in px² and a
        duration in minutes, say).

    Returns
    -------
    fitter : lifelines.CoxTimeVaryingFitter
        The fitted model.
    summary : pandas.DataFrame
        One row per covariate: ``covariate``, ``coef``, ``hazard_ratio`` and
        its 95% interval ``ci_lower`` and ``ci_upper``, ``p``, ``scale`` (the
        step of the covariate a hazard ratio is per: its standard deviation
        when standardized, else 1), with ``n_cells`` and ``n_events``, and
        the ``effect`` read off the interval (see :func:`effect_of`).
    """
    from lifelines import CoxTimeVaryingFitter
    from lifelines.exceptions import ConvergenceWarning

    covariates = list(covariates)
    missing = [c for c in covariates if c not in table.columns]
    if missing:
        raise KeyError(f"Covariates {missing} are not in the table (weighted_neighbors needs the intermediate counts).")
    varying = [c for c in covariates if table[c].nunique(dropna=True) > 1]
    for c in set(covariates) - set(varying):
        logger.warning(f"Covariate {c} does not vary and is left out of the model.")
    if not varying:
        raise ValueError("No covariate varies: nothing to fit.")
    if table["event"].sum() == 0:
        raise ValueError("No event in the table: nothing to fit.")

    strata = list(strata or [])
    columns = ["cell", "start", "stop", "event", *varying, *strata]
    data = table[columns].dropna(subset=varying)
    if len(data) < len(table):
        logger.warning(f"{len(table) - len(data)} intervals without a value of the covariates are left out.")
    if data["event"].sum() == 0:
        raise ValueError("No event left once the intervals without a value are left out: nothing to fit.")
    constant = [c for c in varying if data[c].nunique() < 2]
    for c in constant:
        logger.warning(f"Covariate {c} does not vary once the intervals without a value are left out.")
    varying = [c for c in varying if c not in constant]
    if not varying:
        raise ValueError("No covariate varies: nothing to fit.")
    data = data.drop(columns=constant)
    scales = {}
    for c in varying:
        binary = data[c].isin((0, 1)).all()
        sd = data[c].std()
        scales[c] = sd if standardize and not binary and sd > 0 else 1.0
    data = data.assign(**{c: data[c] / scales[c] for c in varying})

    # lifelines multiplies the penalty by the number of rows, one per cell and
    # frame here: dividing it back makes the penalty independent of the length
    # of the movies, a weight on the partial log-likelihood of the events.
    fitter = CoxTimeVaryingFitter(penalizer=penalizer / len(data))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitter.fit(
            data,
            id_col="cell",
            event_col="event",
            start_col="start",
            stop_col="stop",
            strata=strata or None,
        )
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        logger.warning(
            "The hazard model converged poorly: a covariate may (almost) fully separate the cells "
            "with and without the event. A penalty keeps the hazard ratios finite."
        )
    s = fitter.summary
    summary = pd.DataFrame(
        {
            "covariate": s.index.to_numpy(),
            "coef": s["coef"].to_numpy(),
            "hazard_ratio": s["exp(coef)"].to_numpy(),
            "ci_lower": s["exp(coef) lower 95%"].to_numpy(),
            "ci_upper": s["exp(coef) upper 95%"].to_numpy(),
            "p": s["p"].to_numpy(),
            "scale": [scales[c] for c in s.index],
        }
    )
    summary["n_cells"] = data["cell"].nunique()
    summary["n_events"] = int(data["event"].sum())
    summary["effect"] = [effect_of(a, b) for a, b in summary[["ci_lower", "ci_upper"]].to_numpy(dtype=float)]
    return fitter, summary


def fit_hazard_per_neighborhood(
    reference: pd.DataFrame,
    neighborhoods: Sequence[str],
    event: str,
    covariates: Sequence[str],
    strata: Optional[Sequence[str]] = None,
    penalizer: float = 0.0,
    standardize: bool = False,
    split_by: Optional[str] = None,
    return_tables: bool = False,
    **table_options,
):
    """
    Fit the same hazard model for several neighborhoods, e.g. contact
    distances, to see whether the conclusion depends on the definition of a
    contact, and optionally once per condition of the wells.

    Parameters
    ----------
    reference : pandas.DataFrame
        Table of the reference population (see :func:`time_varying_table`).
    neighborhoods : sequence of str
        The neighborhoods to compare; empty to fit once without contact
        covariates, for a population without neighborhood (its row and long
        table are named ``none``).
    event, covariates, strata, penalizer, standardize
        As in :func:`time_varying_table` and :func:`fit_hazard`.
    split_by : str, optional
        A column of the reference table, e.g. ``concentrations``: the model
        is fitted once per value, to compare the hazard ratios between
        conditions; the rows of each fit get a ``condition`` column such as
        ``concentrations = 100``.
    return_tables : bool, optional
        Also return the long table of each neighborhood (of the first
        condition when split), to show or save it without building it again.
    **table_options
        Passed to :func:`time_varying_table` (``origin``, ``lag``, ...).

    Returns
    -------
    pandas.DataFrame
        The summaries of :func:`fit_hazard` stacked, with the
        ``neighborhood`` and its ``distance``; a neighborhood that cannot be
        fitted gets one row with its ``error``.
    dict of str to pandas.DataFrame
        With ``return_tables``: the long table of each neighborhood that could
        be built.
    """
    groups = [(None, reference)] if split_by is None else list(reference.groupby(split_by, sort=True))
    summaries, tables = [], {}
    for condition, cells in groups:
        for neighborhood in list(neighborhoods) or [None]:
            neighborhood = None if neighborhood is None else _neighborhood_name(neighborhood)
            name = neighborhood or "none"
            try:
                table = time_varying_table(cells, neighborhood, event, **table_options)
                tables.setdefault(name, table)
                _, summary = fit_hazard(
                    table, covariates, strata=strata, penalizer=penalizer, standardize=standardize
                )
            except Exception as e:
                logger.warning(f"Could not fit the hazard for {name}: {e}")
                summary = pd.DataFrame({"error": [str(e)]})
            summary.insert(0, "distance", np.nan if neighborhood is None else neighborhood_distance(neighborhood))
            summary.insert(0, "neighborhood", name)
            if split_by is not None:
                summary.insert(0, "condition", f"{split_by} = {format_condition(condition)}")
            summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    return (summary, tables) if return_tables else summary


def effect_of(ci_lower: float, ci_upper: float) -> str:
    """
    Read the effect of a covariate off its 95% interval: ``higher hazard``
    when it lies entirely above 1, ``lower hazard`` entirely below 1, ``no
    clear effect`` when it crosses 1 (or could not be computed).
    """
    if ci_lower > 1:
        return "higher hazard"
    if ci_upper < 1:
        return "lower hazard"
    return "no clear effect"


def format_condition(value) -> str:
    """A condition of a well as written in config.ini: 100 rather than 100.0."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def condition_covariates(labels: Dict[str, Sequence]) -> Dict[str, Tuple[str, str, object]]:
    """
    The ways the conditions of the wells can enter a hazard model.

    Only the conditions that differ between wells are offered. A numeric one
    (a concentration) is a continuous covariate on its own scale
    (``concentrations``) or a log scale (``log10(1 + concentrations)``, the
    1 + keeping the wells at 0), or enters by level (``concentrations (each
    vs 0)``), which assumes no shape of the dose-response. A text one (an
    antibody) only by level, each value compared with that of the first well.

    Parameters
    ----------
    labels : dict
        The conditions of the wells, one value per well, as returned by
        :func:`~celldetective.utils.experiment.get_experiment_labels`.

    Returns
    -------
    dict
        Covariate names mapped to ``(kind, column, reference value)``, kind
        being ``linear``, ``log`` or ``levels``; the input of
        :func:`encode_conditions`.
    """
    options = {}
    for key, values in labels.items():
        if len(set(map(str, values))) < 2:
            continue
        numeric = all(isinstance(v, (int, float)) for v in values)
        if numeric:
            options[key] = ("linear", key, None)
            if min(values) >= 0:
                options[f"log10(1 + {key})"] = ("log", key, None)
        options[f"{key} (each vs {format_condition(values[0])})"] = ("levels", key, values[0])
    return options


def encode_conditions(
    reference: pd.DataFrame, covariates: Sequence[str], options: Dict[str, Tuple[str, str, object]]
) -> Tuple[pd.DataFrame, list]:
    """
    Give the conditions of the wells among the covariates their columns in
    the reference table: the value, or its log, for a continuous covariate;
    by level, one indicator per other value (``concentrations = 100`` is 1 in
    the wells at 100, 0 elsewhere, its hazard ratio comparing them with the
    wells of reference).

    Parameters
    ----------
    reference : pandas.DataFrame
        The reference table, with one column per condition (as loaded with
        :func:`~celldetective.utils.data_loaders.load_experiment_tables`).
    covariates : sequence of str
        The covariates of the model; those not in ``options`` are kept as
        they are.
    options : dict
        Output of :func:`condition_covariates`.

    Returns
    -------
    pandas.DataFrame
        The reference table with the new columns.
    list of str
        The covariates, each condition replaced by its columns.
    """
    columns, encoded = {}, []
    for covariate in covariates:
        if covariate not in options:
            encoded.append(covariate)
            continue
        kind, key, ref = options[covariate]
        if key not in reference.columns:
            raise KeyError(f"The table has no {key} column.")
        values = reference[key]
        if kind == "linear":
            columns[covariate] = values.astype(float)
        elif kind == "log":
            columns[covariate] = np.log10(1 + values.astype(float))
        else:
            levels = sorted(set(values.dropna()), key=lambda v: (str(type(v)), v))
            if levels and ref not in levels:
                # The wells of reference are not in the table (not selected):
                # one indicator per level would sum to 1 for every cell, the
                # baseline hazard, and could not be estimated.
                logger.warning(f"No cell with {key} = {format_condition(ref)}: the levels are compared with "
                               f"{format_condition(levels[0])} instead.")
                ref = levels[0]
            for value in (v for v in levels if v != ref):
                columns[f"{key} = {format_condition(value)}"] = (values == value).astype(int)
        encoded.extend(c for c in columns if c not in encoded)
    return reference.assign(**columns), encoded
