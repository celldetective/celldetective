"""
Hazard model: relate the hazard of an event of the cells (their death, say)
to what may affect it frame by frame, with a Cox model with time-varying
covariates: the contacts they receive, for one or several neighborhoods,
their own measurements and the conditions of their wells. The contact
episodes can also be related to the event that follows them.
"""

import os
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from celldetective import get_logger
from celldetective.hazard import CONTACT_COVARIATES, condition_covariates, effect_of, encode_conditions
from celldetective.gui.base.collapsible import CollapsibleFrame
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    QCheckableComboBox,
    QHSeperationLine,
    generic_message,
)
from celldetective.gui.base.styles import CELLDETECTIVE_BLUE, DANGER_COLOR, MUTED_INK
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import NumberLineEdit
from celldetective.utils.data_cleaning import TABLE_METADATA_COLUMNS, extract_cols_from_table_list

logger = get_logger(__name__)

_DEFAULT_COVARIATES = ("in_contact", "cumulative_contact")

# Columns of a trajectory table that are bookkeeping, events or neighborhood
# counts rather than measurements of the cell, besides the metadata columns
# shared by every table and the conditions of the wells.
_NOT_FEATURES = {*TABLE_METADATA_COLUMNS, "TRACK_ID", "FRAME", "ID", "label", "generation", "root",
                 "parent", "state", "dummy", "index", "POSITION_X", "POSITION_Y", "x", "y"}
_NOT_FEATURE_PREFIXES = ("t_", "class_", "status_", "inclusive_count_", "intermediate_count_",
                         "exclusive_count_", "mean_count_", "neighborhood_", "contact_", "Unnamed")


def _cell_features(table: str, exclude=()) -> List[str]:
    """The numeric measurements of the cells in a trajectory table (area,
    intensities, shape...), to offer as covariates; the columns of
    ``exclude`` (the conditions of the wells) are left out."""
    try:
        head = pd.read_csv(table, nrows=200)
    except Exception:
        return []
    numeric = head.select_dtypes(include="number").columns
    return sorted(
        c for c in numeric
        if c not in _NOT_FEATURES and c not in exclude and not c.startswith(_NOT_FEATURE_PREFIXES)
    )


class _PenaltyEdit(NumberLineEdit):
    INVALID = "The penalty must be a positive number, or empty for none."

    def _parse(self, text: str) -> float:
        value = float(text)
        if value < 0:
            raise ValueError(self.INVALID)
        return value


class _FrameToMinEdit(NumberLineEdit):
    INVALID = "The time calibration must be a positive number of minutes per frame."
    OPTIONAL = False
    LOWER_BOUND = 0.0


def _row(label: str, widget, tooltip: Optional[str] = None) -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    row.addWidget(lbl, 33)
    row.addWidget(widget, 66)
    return row


def _log_ticks(lo: float, hi: float) -> List[float]:
    """Readable ticks of a log axis from ``lo`` to ``hi``: 1-2-5 steps, finer
    ones around 1 when the range is narrow."""
    steps = sorted({m * 10.0**e for e in range(-4, 5) for m in (1, 2, 5)})
    ticks = [t for t in steps if lo <= t <= hi]
    if len(ticks) < 3:
        fine = sorted(set(steps) | {0.8, 0.9, 1.1, 1.2, 1.5})
        ticks = [t for t in fine if lo <= t <= hi]
    # Labels closer than a sixth of the axis overlap: keep 1, thin the others.
    gap = np.log(hi / lo) / 6
    kept = [1.0] if lo <= 1 <= hi else []
    for t in sorted(ticks, key=lambda t: abs(np.log(t))):
        if all(abs(np.log(t / k)) >= gap for k in kept):
            kept.append(t)
    return sorted(kept)


_EFFECTS = {
    "higher hazard": DANGER_COLOR,
    "lower hazard": CELLDETECTIVE_BLUE,
    "no clear effect": MUTED_INK,
}
# The effect drawn with open markers and thin lines: no conclusion.
_UNCLEAR = "no clear effect"


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.2g}"


def forest_plot(summary: pd.DataFrame):
    """Hazard ratios and their 95% intervals on a single log axis, one
    category per covariate, the neighborhoods side by side within it with a
    marker each. Each estimate is coloured by its effect: red when the
    interval lies above 1, blue below 1, an open grey marker when it crosses
    1."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fitted = summary.dropna(subset=["hazard_ratio"])
    covariates = list(dict.fromkeys(fitted["covariate"]))
    # One series per neighborhood, and per condition when the fit was split.
    short = fitted["neighborhood"].str.replace("neighborhood_", "", regex=False)
    if "condition" in fitted.columns:
        conditions = fitted["condition"].astype(str)
        short = conditions if short.nunique() == 1 else short + " | " + conditions
    fitted = fitted.assign(series=short)
    series = list(dict.fromkeys(fitted["series"]))
    markers = ["o", "s", "D", "^", "v", "P", "X"]

    # A poorly converged fit can give a bound of 0 or inf: the log axis spans
    # the finite values, the others are drawn at its ends.
    values = fitted[["hazard_ratio", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    lo = min(values.min(initial=1.0), 1.0) / 1.3
    hi = max(values.max(initial=1.0), 1.0) * 1.3

    width = 0.6
    step = width / max(1, len(series))
    fig, ax = plt.subplots(
        figsize=(3.6 + len(covariates) * (0.8 + 0.7 * len(series)), 4.2), layout="constrained"
    )

    # The two sides of 1 tinted, so that where a point falls reads at a glance.
    ax.axhspan(lo, 1, color=CELLDETECTIVE_BLUE, alpha=0.06, lw=0, zorder=0)
    ax.axhspan(1, hi, color=DANGER_COLOR, alpha=0.06, lw=0, zorder=0)
    ax.axhline(1, color=MUTED_INK, lw=1, ls="--", zorder=1)
    ax.text(0.01, 0.98, "↑ higher hazard", transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=DANGER_COLOR)
    ax.text(0.01, 0.02, "↓ lower hazard", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color=CELLDETECTIVE_BLUE)

    for xi, covariate in enumerate(covariates):
        rows = fitted[fitted["covariate"] == covariate].set_index("series")
        for j, name in enumerate(series):
            if name not in rows.index:
                continue
            row = rows.loc[name]
            hr, a, b = (float(row[c]) for c in ("hazard_ratio", "ci_lower", "ci_upper"))
            if not np.isfinite(hr):
                continue
            effect = effect_of(a, b)
            color = _EFFECTS[effect]
            clear = effect != _UNCLEAR
            x = xi - width / 2 + step * (j + 0.5)
            ax.vlines(x, np.clip(a, lo, hi), np.clip(b, lo, hi), color=color, lw=2.5 if clear else 1.5, zorder=2)
            ax.plot(x, np.clip(hr, lo, hi), markers[j % len(markers)], ms=9, zorder=3, color=color,
                    markerfacecolor=color if clear else "white",
                    markeredgecolor="white" if clear else color, markeredgewidth=1.5)
            label = f"×{hr:.2g}"
            p = _format_p(float(row["p"])) if "p" in row.index else ""
            if p:
                label += f"\n{p}"
            ax.annotate(label, (x, np.clip(hr, lo, hi)), xytext=(7, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7, color=color,
                        fontweight="bold" if clear else "normal")

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(FixedLocator(_log_ticks(lo, hi)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("hazard ratio (95% CI)")
    ax.set_xticks(np.arange(len(covariates)))
    # Say what step of the covariate a hazard ratio is per when it is not 1.
    ticks = []
    for covariate in covariates:
        scales = fitted.loc[fitted["covariate"] == covariate, "scale"] if "scale" in fitted.columns else []
        scales = set(np.round(np.asarray(scales, dtype=float), 6))
        if not scales or scales == {1.0}:
            ticks.append(covariate)
        elif len(scales) == 1:
            ticks.append(f"{covariate}\nper SD ({scales.pop():.3g})")
        else:  # the SD of a contact covariate differs between neighborhoods
            ticks.append(f"{covariate}\nper SD")
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_xlim(-0.6, len(covariates) - 0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", which="major", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)

    handles = [
        Line2D([], [], marker="o", ls="-", color=color, ms=7,
               markerfacecolor=color if effect != _UNCLEAR else "white", label=effect)
        for effect, color in _EFFECTS.items()
    ]
    if len(series) > 1:
        handles.append(Line2D([], [], ls="none", label=" "))
        handles += [
            Line2D([], [], marker=markers[j % len(markers)], ls="none", color=MUTED_INK, ms=7,
                   label=n)
            for j, n in enumerate(series)
        ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
              fontsize=8, labelspacing=0.9, handlelength=1.6, borderaxespad=0)
    return fig


def dose_response_plot(binned: pd.DataFrame, event: str, window: int, align: str):
    """Fraction of episodes followed by the event per class of duration."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.arange(len(binned))
    shown = binned["n"] > 0
    ax.errorbar(
        x[shown], binned.loc[shown, "fraction"],
        yerr=[binned.loc[shown, "fraction"] - binned.loc[shown, "ci_lower"],
              binned.loc[shown, "ci_upper"] - binned.loc[shown, "fraction"]],
        fmt="o", color=CELLDETECTIVE_BLUE, ms=8, lw=2, capsize=0,
        markeredgecolor="white", markeredgewidth=1.5,
    )
    for xi, (n, k) in zip(x[shown], binned.loc[shown, ["n", "events"]].to_numpy()):
        ax.annotate(f"{int(k)}/{int(n)}", (xi, 1.02), ha="center", va="bottom", fontsize=8, color=MUTED_INK)
    ax.set_xticks(x)
    ax.set_xticklabels(binned["bin"], fontsize=8)
    ax.set_xlabel("episode duration [frames]")
    ax.set_ylabel(f"fraction with {event}\nwithin {window} frames of the {align}")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.5, len(binned) - 0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


class ConfigHazard(CelldetectiveWidget):
    """
    Set up a hazard analysis on the wells and positions selected in
    the control panel.
    """

    def __init__(self, parent_window: Optional[QMainWindow] = None) -> None:
        super().__init__()
        self.parent_window = parent_window
        self.control = parent_window.parent_window
        self.exp_dir = parent_window.exp_dir
        self.FrameToMin = float(self.control.FrameToMin)
        self.setWindowTitle("Hazard model")
        self.setMinimumWidth(460)
        self.windows: List[CelldetectiveWidget] = []

        self.cols_per_pop = {}
        self.tables_per_pop = {}
        for population in self.control.populations:
            tables = self.tables_of(population)
            if tables:
                cols = list(extract_cols_from_table_list(tables))
                # Any population with an event can be studied; a neighborhood
                # only adds the contact covariates.
                if any(c.startswith("t_") for c in cols):
                    self.cols_per_pop[population] = cols
                    self.tables_per_pop[population] = tables[0]
        # The measurements of each population, read when it is first shown.
        self.features_per_pop = {}
        self.has_pairs = len(self.tables_of("pairs")) > 0
        # The conditions of the wells ([Labels] of config.ini), one column
        # each in the loaded tables, and the ways they can enter the model.
        try:
            from celldetective.utils.experiment import get_experiment_labels

            self.well_labels = get_experiment_labels(self.exp_dir)
        except Exception:
            self.well_labels = {}
        self.condition_options = condition_covariates(self.well_labels)
        self.populate_widget()

    def tables_of(self, population: str) -> List[str]:
        return glob(self.exp_dir + os.sep.join(["W*", "*", "output", "tables", f"trajectories_{population}.csv"]))

    def populate_widget(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        layout.addWidget(self.section("Model"), alignment=Qt.AlignCenter)

        self.population_cb = QComboBox()
        self.population_cb.addItems(list(self.cols_per_pop))
        layout.addLayout(_row("reference\npopulation: ", self.population_cb,
                              "Cells whose event is studied. With a neighborhood, their contacts can be\n"
                              "covariates; without, their measurements and the conditions of the wells."))

        self.neighborhood_cb = QCheckableComboBox(obj="neighborhood")
        layout.addLayout(_row("neighborhoods: ", self.neighborhood_cb,
                              "The model is fitted for each neighborhood checked: several contact\n"
                              "distances tell whether the result depends on the definition of a contact."))

        self.event_cb = QComboBox()
        layout.addLayout(_row("event: ", self.event_cb,
                              "Time of the event of the reference cells, with its class column\n"
                              "(0: happened, 1: did not, 2: happened before the observation)."))

        self.covariate_cb = QCheckableComboBox(obj="covariate")
        layout.addLayout(_row("covariates: ", self.covariate_cb,
                              "\n".join(f"{k}: {v}" for k, v in CONTACT_COVARIATES.items())
                              + "\ncontact_*: contact-site intensities, 0 out of contact"
                              + "\nthen the measurements of the cell (area, intensities, ...),\n"
                              "taking their last value on the frames where the cell is not detected"))

        # Only the conditions that differ between wells are worth splitting by.
        varying = list(dict.fromkeys(key for _, key, _ in self.condition_options.values()))
        self.split_cb = QComboBox()
        self.split_cb.addItems(["none", *varying])
        layout.addLayout(_row("fit separately\nper: ", self.split_cb,
                              "Fit the model once per value of a condition of the wells (config.ini),\n"
                              "e.g. per concentration, to compare the hazard ratios between conditions:\n"
                              "does a contact kill more at a higher concentration?"))

        self.query_le = QLineEdit()
        layout.addLayout(_row("select cells\nwith query: ", self.query_le))

        layout.addWidget(self.advanced_frame())

        layout.addWidget(QHSeperationLine())
        layout.addWidget(self.section("Contact episodes"), alignment=Qt.AlignCenter)

        self.episodes_check = QCheckBox("event after each contact episode")
        self.episodes_check.setToolTip(
            "From the pair table: the fraction of contact episodes followed by the event,\n"
            "per class of episode duration. Descriptive: episodes of a cell are not independent."
        )
        self.episodes_check.setEnabled(self.has_pairs)
        if not self.has_pairs:
            self.episodes_check.setToolTip("Measure the pairs of a neighborhood first.")
        layout.addWidget(self.episodes_check)

        # The options of the episodes are only shown once they are asked for.
        self.episode_options = QWidget()
        options = QVBoxLayout(self.episode_options)
        options.setContentsMargins(0, 0, 0, 0)

        self.window_sb = QSpinBox()
        self.window_sb.setRange(1, 10000)
        self.window_sb.setValue(10)
        options.addLayout(_row("window [frames]: ", self.window_sb,
                               "The event counts when it happens within this many frames."))

        self.align_cb = QComboBox()
        self.align_cb.addItems(["end", "start"])
        options.addLayout(_row("after the: ", self.align_cb,
                               "end: among cells alive at the end of the episode, its duration fully seen,\n"
                               "but the cells dying during the contact are left out.\n"
                               "start: among cells alive at its start; a death can cut the contact short."))

        self.gap_sb = QSpinBox()
        self.gap_sb.setRange(0, 100)
        options.addLayout(_row("max gap [frames]: ", self.gap_sb,
                               "Contacts interrupted for up to this many frames make one episode."))

        self.bins_le = QLineEdit("1, 3, 6, 12, inf")
        options.addLayout(_row("duration bins: ", self.bins_le,
                               "Edges of the classes of episode duration, in frames."))

        self.episode_options.setVisible(False)
        self.episodes_check.toggled.connect(self.show_episode_options)
        layout.addWidget(self.episode_options)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.run)
        layout.addWidget(self.submit_btn)

        self.population_cb.currentIndexChanged.connect(self.fill_options)
        self.fill_options()

    def advanced_frame(self) -> CollapsibleFrame:
        """The options of the model that rarely need changing, folded by
        default."""
        frame = CollapsibleFrame("ADVANCED PARAMETERS")
        content = QWidget()
        advanced = QVBoxLayout(content)

        self.origin_cb = QComboBox()
        advanced.addLayout(_row("follow from: ", self.origin_cb,
                                "When each cell starts to be at risk; by default its first frame."))

        self.lag_sb = QSpinBox()
        self.lag_sb.setRange(0, 1000)
        advanced.addLayout(_row("lag [frames]: ", self.lag_sb,
                                "Frames between the exposure and the event it explains: the delay\n"
                                "between a lethal hit and the moment the death is seen."))

        self.strata_cb = QComboBox()
        self.strata_cb.addItems(["none", "well", "position"])
        advanced.addLayout(_row("baseline per: ", self.strata_cb,
                                "Give each well or position its own baseline hazard, so that cells are\n"
                                "only compared with cells of the same well or position."))

        self.penalizer_le = _PenaltyEdit("0.5")
        advanced.addLayout(_row("penalty: ", self.penalizer_le,
                                "L2 penalty on the coefficients. A small one keeps the fit finite when\n"
                                "every event follows a contact (complete separation); larger ones pull\n"
                                "the hazard ratios towards 1."))

        self.unit_cb = QComboBox()
        self.unit_cb.addItems(["min", "h", "frame"])
        advanced.addLayout(_row("duration unit: ", self.unit_cb,
                                "Unit of the contact durations: a hazard ratio of a duration is per unit."))

        self.time_calibration_le = _FrameToMinEdit(str(self.FrameToMin))
        advanced.addLayout(_row("time calibration\n(frame to min): ", self.time_calibration_le,
                                "Read from the experiment; change it only if it is wrong there."))

        self.standardize_check = QCheckBox("hazard ratio per standard deviation")
        self.standardize_check.setChecked(True)
        self.standardize_check.setToolTip(
            "Express the hazard ratio of each covariate that is not binary per standard\n"
            "deviation, so that covariates of different units (an area in px², a duration\n"
            "in minutes) can be compared. Unchecked: per unit, e.g. per px²."
        )
        self.unit_cb.setEnabled(False)
        self.standardize_check.toggled.connect(lambda checked: self.unit_cb.setEnabled(not checked))
        advanced.addWidget(self.standardize_check)

        self.long_table_check = QCheckBox("show the long table")
        self.long_table_check.setToolTip(
            "One row per cell and frame interval (start, stop]: the exposure of the cell\n"
            "up to the start of the interval and whether the event happened in it."
        )
        advanced.addWidget(self.long_table_check)

        frame.set_content(content)
        frame.animation_finished.connect(lambda _: self.adjustSize())
        return frame

    def show_episode_options(self, shown: bool):
        self.episode_options.setVisible(shown)
        self.adjustSize()

    def section(self, title: str) -> QLabel:
        label = QLabel(title)
        label.setStyleSheet(self.block_title)
        return label

    def fill_options(self):
        cols = self.cols_per_pop.get(self.population_cb.currentText(), [])
        neighborhoods = [c[len("inclusive_count_"):] for c in cols if c.startswith("inclusive_count_neighborhood")]
        self.neighborhood_cb.clear()
        self.neighborhood_cb.addItems(neighborhoods)
        if neighborhoods:
            self.neighborhood_cb.setCurrentIndex(0)

        times = [c for c in cols if c.startswith("t_")]
        self.event_cb.clear()
        self.event_cb.addItems(times)
        for preferred in ("t_death", "t_lysis"):
            if preferred in times:
                self.event_cb.setCurrentText(preferred)
        self.origin_cb.clear()
        self.origin_cb.addItems(["first frame"] + times)

        covariates = []
        if neighborhoods:
            has_intermediate = any(c.startswith("intermediate_count_neighborhood") for c in cols)
            covariates += [c for c in CONTACT_COVARIATES if c != "weighted_neighbors" or has_intermediate]
            covariates += [c for c in cols if c.startswith("contact_") and c.endswith("_mean")]
        self.neighborhood_cb.setEnabled(bool(neighborhoods))
        self.neighborhood_cb.setToolTip(
            "" if neighborhoods else "No neighborhood for this population: only the measurements of the cells\n"
            "and the conditions of the wells can be covariates."
        )
        population = self.population_cb.currentText()
        if population in self.tables_per_pop and population not in self.features_per_pop:
            self.features_per_pop[population] = _cell_features(
                self.tables_per_pop[population], exclude=self.well_labels
            )
        covariates += self.features_per_pop.get(population, [])
        # The conditions of the wells, e.g. the concentration.
        covariates += list(self.condition_options)
        self.covariate_cb.clear()
        self.covariate_cb.addItems(covariates)
        for i, c in enumerate(covariates):
            if c in _DEFAULT_COVARIATES:
                self.covariate_cb.setCurrentIndex(i)

    def selected(self, combo: QCheckableComboBox) -> List[str]:
        return [combo.itemText(i) for i in combo.getSelectedIndices()]

    def load(self, population: str) -> Optional[pd.DataFrame]:
        from celldetective.utils.data_loaders import load_experiment_tables

        return load_experiment_tables(
            self.exp_dir,
            population=population,
            well_option=self.control.well_list.getSelectedIndices(),
            position_option=self.control.position_list.getSelectedIndices(),
        )

    def run(self):
        from celldetective.hazard import fit_hazard_per_neighborhood

        population = self.population_cb.currentText()
        neighborhoods = self.selected(self.neighborhood_cb)
        covariates = self.selected(self.covariate_cb)
        event = self.event_cb.currentText()
        if not population or not covariates or not event:
            generic_message("Choose a population, a covariate and an event.")
            return
        contact = [c for c in covariates if c in CONTACT_COVARIATES or c.startswith("contact_")]
        if contact and not neighborhoods:
            generic_message(f"The covariates {', '.join(contact)} need a neighborhood: check one.")
            return
        conditions = [self.condition_options[c][1] for c in covariates if c in self.condition_options]
        repeated = sorted({k for k in conditions if conditions.count(k) > 1})
        if repeated:
            generic_message(f"Choose one way for {', '.join(repeated)} to enter the model: its encodings "
                            "carry the same information and cannot be estimated together.")
            return
        if self.episodes_check.isChecked() and not neighborhoods:
            generic_message("The contact episodes need a neighborhood: check one.")
            return
        valid_calibration, frame_to_min = self.time_calibration_le.value_or_warn()
        valid_penalty, penalizer = self.penalizer_le.value_or_warn()
        if not (valid_calibration and valid_penalty):
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            reference = self.load(population)
            if reference is None:
                raise ValueError("No table could be loaded for the selection.")
            query = self.query_le.text().strip()
            if query:
                reference = reference.query(query)
            reference, covariates = encode_conditions(reference, covariates, self.condition_options)
            extra = [c for c in covariates if c not in CONTACT_COVARIATES]
            options = {
                "origin": None if self.origin_cb.currentText() == "first frame" else self.origin_cb.currentText(),
                "lag": self.lag_sb.value(),
                "frame_to_min": frame_to_min,
                "duration_unit": self.unit_cb.currentText(),
                "extra_covariates": extra,
            }
            strata = None if self.strata_cb.currentText() == "none" else [self.strata_cb.currentText()]

            split = self.split_cb.currentText()
            summary, tables = fit_hazard_per_neighborhood(
                reference, neighborhoods, event, covariates, strata=strata, penalizer=penalizer or 0.0,
                standardize=self.standardize_check.isChecked(),
                split_by=None if split == "none" else split,
                return_tables=True, **options,
            )
            self.show_table(summary, f"Hazard of {event.replace('t_', '')}: {population}")
            if "hazard_ratio" in summary.columns and summary["hazard_ratio"].notna().any():
                self.show_figure(forest_plot(summary), "Hazard ratios")
            else:
                generic_message("No model could be fitted: see the error column of the table.")

            first = neighborhoods[0] if neighborhoods else "none"
            if self.long_table_check.isChecked() and first in tables:
                self.show_table(tables[first], f"Long table: {first}")

            if self.episodes_check.isChecked():
                self.run_episodes(reference, population, neighborhoods[0], event)
        except Exception as e:
            logger.exception("Hazard analysis failed.")
            generic_message(f"The analysis failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def run_episodes(self, reference: pd.DataFrame, population: str, neighborhood: str, event: str):
        from celldetective.hazard import binned_event_fraction, episode_outcomes
        from celldetective.utils.data_cleaning import collapse_pairs_by_episode

        pairs = self.load("pairs")
        if pairs is None or f"status_{neighborhood}" not in pairs.columns:
            generic_message(f"The pairs of {neighborhood} were not measured.")
            return
        pairs = pairs[pairs["reference_population"] == population]
        episodes = collapse_pairs_by_episode(pairs, neighborhood, max_gap=self.gap_sb.value())
        window, align = self.window_sb.value(), self.align_cb.currentText()
        outcomes = episode_outcomes(episodes, reference, event, window=window, align=align)
        bins = [float(b) for b in self.bins_le.text().replace(";", ",").split(",") if b.strip()]
        binned = binned_event_fraction(outcomes, bins)
        self.show_table(outcomes, f"Contact episodes: {neighborhood}")
        self.show_figure(
            dose_response_plot(binned, event.replace("t_", ""), window, align), "Event after contact episodes"
        )

    def show_table(self, df: pd.DataFrame, title: str):
        from celldetective.gui.tableUI import TableUI

        table = TableUI(df, title, plot_mode="static", collapse_tracks_option=False)
        table.show()
        self.windows.append(table)

    def show_figure(self, fig, title: str):
        from celldetective.gui.base.figure_canvas import FigureCanvas
        from celldetective.gui.gui_utils import ExportPlotBtn

        canvas = FigureCanvas(fig, title=title, interactive=True)
        canvas.layout.addWidget(ExportPlotBtn(fig, export_dir=self.exp_dir), alignment=Qt.AlignRight)
        canvas.show()
        center_window(canvas)
        self.windows.append(canvas)
