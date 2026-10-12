"""
Contact hazard: relate the contacts a reference cell receives to the hazard
of one of its events (its death, say) with a Cox model with time-varying
covariates, for one or several neighborhoods, and the contact episodes to the
event that follows them.
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
)

from celldetective import get_logger
from celldetective.contact_hazard import CONTACT_COVARIATES
from celldetective.gui.base.components import (
    CelldetectiveWidget,
    QCheckableComboBox,
    QHSeperationLine,
    generic_message,
)
from celldetective.gui.base.styles import CELLDETECTIVE_BLUE, MUTED_INK
from celldetective.gui.base.utils import center_window
from celldetective.gui.gui_utils import NumberLineEdit
from celldetective.utils.data_cleaning import extract_cols_from_table_list

logger = get_logger(__name__)

_DEFAULT_COVARIATES = ("in_contact", "cumulative_contact")


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
    return ticks


def forest_plot(summary: pd.DataFrame):
    """Hazard ratios and their 95% intervals, one panel per covariate, one
    line per neighborhood."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fitted = summary.dropna(subset=["hazard_ratio"])
    covariates = list(dict.fromkeys(fitted["covariate"]))
    fig, axes = plt.subplots(
        1, max(1, len(covariates)), figsize=(3.2 * max(1, len(covariates)), 0.5 + 0.45 * fitted["neighborhood"].nunique() + 1),
        sharey=True, squeeze=False,
    )
    labels = list(dict.fromkeys(fitted["neighborhood"]))
    short = [n.replace("neighborhood_", "") for n in labels]
    for ax, covariate in zip(axes[0], covariates):
        rows = fitted[fitted["covariate"] == covariate].set_index("neighborhood").reindex(labels)
        # A poorly converged fit can give a bound of 0 or inf: the log axis
        # spans the finite values, the others are drawn at its ends.
        values = rows[["hazard_ratio", "ci_lower", "ci_upper"]].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        lo = min(values.min(initial=1.0), 1.0) / 1.15
        hi = max(values.max(initial=1.0), 1.0) * 1.15
        rows = rows[["hazard_ratio", "ci_lower", "ci_upper"]].clip(lower=lo, upper=hi)
        y = np.arange(len(labels))
        ax.axvline(1, color=MUTED_INK, lw=1, ls="--", zorder=0)
        ax.hlines(y, rows["ci_lower"], rows["ci_upper"], color=CELLDETECTIVE_BLUE, lw=2)
        ax.plot(rows["hazard_ratio"], y, "o", color=CELLDETECTIVE_BLUE, ms=8,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(FixedLocator(_log_ticks(lo, hi)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(covariate, fontsize=10)
        ax.set_xlabel("hazard ratio (95% CI)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", which="major", color="#e6e6e6", lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_yticks(np.arange(len(labels)))
    axes[0][0].set_yticklabels(short, fontsize=8)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
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


class ConfigContactHazard(CelldetectiveWidget):
    """
    Set up a contact hazard analysis on the wells and positions selected in
    the control panel.
    """

    def __init__(self, parent_window: Optional[QMainWindow] = None) -> None:
        super().__init__()
        self.parent_window = parent_window
        self.control = parent_window.parent_window
        self.exp_dir = parent_window.exp_dir
        self.FrameToMin = float(self.control.FrameToMin)
        self.setWindowTitle("Contact hazard")
        self.setMinimumWidth(460)
        self.windows: List[CelldetectiveWidget] = []

        self.cols_per_pop = {}
        for population in self.control.populations:
            tables = self.tables_of(population)
            if tables:
                cols = list(extract_cols_from_table_list(tables))
                if any(c.startswith("inclusive_count_neighborhood") for c in cols):
                    self.cols_per_pop[population] = cols
        self.has_pairs = len(self.tables_of("pairs")) > 0
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
                              "Cells whose event is studied, with a neighborhood computed."))

        self.neighborhood_cb = QCheckableComboBox(obj="neighborhood")
        layout.addLayout(_row("neighborhoods: ", self.neighborhood_cb,
                              "The model is fitted for each neighborhood checked: several contact\n"
                              "distances tell whether the result depends on the definition of a contact."))

        self.event_cb = QComboBox()
        layout.addLayout(_row("event: ", self.event_cb,
                              "Time of the event of the reference cells, with its class column\n"
                              "(0: happened, 1: did not, 2: happened before the observation)."))

        self.origin_cb = QComboBox()
        layout.addLayout(_row("follow from: ", self.origin_cb,
                              "When each cell starts to be at risk; by default its first frame."))

        self.covariate_cb = QCheckableComboBox(obj="covariate")
        layout.addLayout(_row("covariates: ", self.covariate_cb,
                              "\n".join(f"{k}: {v}" for k, v in CONTACT_COVARIATES.items())
                              + "\ncontact_*: contact-site intensities, 0 out of contact"))

        self.lag_sb = QSpinBox()
        self.lag_sb.setRange(0, 1000)
        layout.addLayout(_row("lag [frames]: ", self.lag_sb,
                              "Frames between the exposure and the event it explains: the delay\n"
                              "between a lethal hit and the moment the death is seen."))

        self.unit_cb = QComboBox()
        self.unit_cb.addItems(["min", "h", "frame"])
        layout.addLayout(_row("duration unit: ", self.unit_cb,
                              "Unit of the contact durations: a hazard ratio of a duration is per unit."))

        self.strata_cb = QComboBox()
        self.strata_cb.addItems(["none", "well", "position"])
        layout.addLayout(_row("baseline per: ", self.strata_cb,
                              "Give each well or position its own baseline hazard, so that cells are\n"
                              "only compared with cells of the same well or position."))

        self.penalizer_le = _PenaltyEdit("0.01")
        layout.addLayout(_row("penalty: ", self.penalizer_le,
                              "L2 penalty on the coefficients. A small one keeps the fit finite when\n"
                              "every event follows a contact (complete separation); larger ones pull\n"
                              "the hazard ratios towards 1."))

        self.query_le = QLineEdit()
        layout.addLayout(_row("select cells\nwith query: ", self.query_le))

        self.time_calibration_le = _FrameToMinEdit(str(self.FrameToMin))
        layout.addLayout(_row("time calibration\n(frame to min): ", self.time_calibration_le))

        self.long_table_check = QCheckBox("show the long table")
        self.long_table_check.setToolTip(
            "One row per cell and frame interval (start, stop]: the exposure of the cell\n"
            "up to the start of the interval and whether the event happened in it."
        )
        layout.addWidget(self.long_table_check)

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

        self.window_sb = QSpinBox()
        self.window_sb.setRange(1, 10000)
        self.window_sb.setValue(10)
        layout.addLayout(_row("window [frames]: ", self.window_sb,
                              "The event counts when it happens within this many frames."))

        self.align_cb = QComboBox()
        self.align_cb.addItems(["end", "start"])
        layout.addLayout(_row("after the: ", self.align_cb,
                              "end: among cells alive at the end of the episode, its duration fully seen,\n"
                              "but the cells dying during the contact are left out.\n"
                              "start: among cells alive at its start; a death can cut the contact short."))

        self.gap_sb = QSpinBox()
        self.gap_sb.setRange(0, 100)
        layout.addLayout(_row("max gap [frames]: ", self.gap_sb,
                              "Contacts interrupted for up to this many frames make one episode."))

        self.bins_le = QLineEdit("1, 3, 6, 12, inf")
        layout.addLayout(_row("duration bins: ", self.bins_le,
                              "Edges of the classes of episode duration, in frames."))

        for w in (self.window_sb, self.align_cb, self.gap_sb, self.bins_le):
            w.setEnabled(False)
            self.episodes_check.toggled.connect(w.setEnabled)

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.setStyleSheet(self.button_style_sheet)
        self.submit_btn.clicked.connect(self.run)
        layout.addWidget(self.submit_btn)

        self.population_cb.currentIndexChanged.connect(self.fill_options)
        self.fill_options()

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

        has_intermediate = any(c.startswith("intermediate_count_neighborhood") for c in cols)
        covariates = [c for c in CONTACT_COVARIATES if c != "weighted_neighbors" or has_intermediate]
        covariates += [c for c in cols if c.startswith("contact_") and c.endswith("_mean")]
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
        from celldetective.contact_hazard import contact_hazard_sensitivity

        population = self.population_cb.currentText()
        neighborhoods = self.selected(self.neighborhood_cb)
        covariates = self.selected(self.covariate_cb)
        event = self.event_cb.currentText()
        if not population or not neighborhoods or not covariates or not event:
            generic_message("Choose a population, a neighborhood, a covariate and an event.")
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
            extra = [c for c in covariates if c not in CONTACT_COVARIATES]
            options = {
                "origin": None if self.origin_cb.currentText() == "first frame" else self.origin_cb.currentText(),
                "lag": self.lag_sb.value(),
                "frame_to_min": frame_to_min,
                "duration_unit": self.unit_cb.currentText(),
                "extra_covariates": extra,
            }
            strata = None if self.strata_cb.currentText() == "none" else [self.strata_cb.currentText()]

            summary, tables = contact_hazard_sensitivity(
                reference, neighborhoods, event, covariates, strata=strata, penalizer=penalizer or 0.0,
                return_tables=True, **options,
            )
            self.show_table(summary, f"Contact hazard of {event.replace('t_', '')}: {population}")
            if "hazard_ratio" in summary.columns and summary["hazard_ratio"].notna().any():
                self.show_figure(forest_plot(summary), "Hazard ratios")
            else:
                generic_message("No model could be fitted: see the error column of the table.")

            if self.long_table_check.isChecked() and neighborhoods[0] in tables:
                self.show_table(tables[neighborhoods[0]], f"Long table: {neighborhoods[0]}")

            if self.episodes_check.isChecked():
                self.run_episodes(reference, population, neighborhoods[0], event)
        except Exception as e:
            logger.exception("Contact hazard analysis failed.")
            generic_message(f"The analysis failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def run_episodes(self, reference: pd.DataFrame, population: str, neighborhood: str, event: str):
        from celldetective.contact_hazard import binned_event_fraction, episode_outcomes
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
