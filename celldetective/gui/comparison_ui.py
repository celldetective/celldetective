"""Tests between the curves of the survival and signal plots.

The computations are in :mod:`celldetective.comparisons`; the results are
shown like the tests of the table explorer, as pairwise matrices
(:class:`celldetective.gui.tableUI.PivotTableUI`).
"""

from typing import Any, List

import numpy as np
import pandas as pd
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout

from celldetective.gui.base.components import CelldetectiveWidget, generic_message, hint_label
from celldetective.gui.base.utils import center_window

WELLS_NOTE = (
    "The cells of a well are pooled, so these tests compare wells. With one well per "
    "condition, a difference may come from the well rather than its condition: repeat the "
    "experiment to conclude on a condition."
)
LOGRANK_GUIDE = (
    "<b>How to read:</b> each cell tests whether the <b>row</b> and <b>column</b> wells have "
    "the same survival (log-rank test on their cells pooled, Holm-adjusted for the number of "
    "pairs). Read the direction on the curves."
)


def show_survival_comparison(plot_window: Any) -> List[CelldetectiveWidget]:
    """Compare the survival of the wells shown with a matrix of log-rank
    p-values. Returns the window, in a list (empty if nothing to compare)."""
    from celldetective.comparisons import pairwise_logrank
    from celldetective.gui.tableUI import PivotTableUI

    info = plot_window.df_well_info
    info = info.loc[info["select"].astype(bool)] if "select" in info.columns else info
    fits = {
        name: fit
        for name, fit in zip(info["well_name"], info.get("survival_fit", []))
        if hasattr(fit, "durations")
    }
    if len(fits) < 2:
        generic_message("Show at least two wells with a survival curve to compare them.")
        return []

    pvalues = PivotTableUI(pairwise_logrank(fits), title="Log-rank p-value (Holm-adjusted)", mode="pvalue")
    pvalues.set_reading_guide(LOGRANK_GUIDE + "<br>" + WELLS_NOTE)
    pvalues.show()
    return [pvalues]


class SignalComparisonWidget(CelldetectiveWidget):
    """Compare synchronised signals by the response of each cell: the mean
    over a window after the synchronising time minus the mean over a baseline
    window. The responses are tested between wells as in the table explorer
    (KS test and Cliff's delta)."""

    def __init__(self, plot_window: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.plot_window = plot_window
        self.frame_to_min = plot_window.parent_window.FrameToMin
        self.setWindowTitle("Compare signals")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        validator = QDoubleValidator()

        self.windows = {}
        for name, (lo, hi) in (("baseline", (-10, -1)), ("response", (0, 9))):
            hbox = QHBoxLayout()
            hbox.addWidget(QLabel(f"{name} [min]: "), 30)
            fields = []
            for value in (lo, hi):
                field = QLineEdit(f"{value * self.frame_to_min:g}")
                field.setValidator(validator)
                field.setToolTip(f"The {name} window relative to the synchronising time, both ends included.")
                hbox.addWidget(field, 35)
                fields.append(field)
            self.windows[name] = fields
            layout.addLayout(hbox)

        layout.addWidget(hint_label(WELLS_NOTE))

        self.compare_btn = QPushButton("compare")
        self.compare_btn.setStyleSheet(self.button_style_sheet)
        self.compare_btn.clicked.connect(self.run)
        layout.addWidget(self.compare_btn)
        center_window(self)

    def window_in_frames(self, name: str) -> tuple:
        lo, hi = (float(f.text().replace(",", ".")) / self.frame_to_min for f in self.windows[name])
        if hi < lo:
            raise ValueError(f"The {name} window ends before it starts.")
        return lo, hi

    def alignment_problem(self) -> str:
        """Cells without the event are drawn from the start of the movie, not
        from an event: a window around the synchronising time means nothing
        for them unless all cells share an absolute time."""
        parent = self.plot_window.parent_window
        absolute = hasattr(parent, "abs_time_checkbox") and parent.abs_time_checkbox.isChecked()
        if not absolute and list(self.plot_window.target_class) != [0]:
            return ("Cells without the event have no event time and are aligned on the start of the "
                    "movie. Show the class 'event', or synchronise on an absolute time, to compare responses.")
        return ""

    def responses(self) -> pd.DataFrame:
        """One row per cell of the positions shown, with its response."""
        from celldetective.comparisons import window_responses

        _, _, key = self.plot_window.signal_keys()
        baseline, response = self.window_in_frames("baseline"), self.window_in_frames("response")
        info = self.plot_window.df_pos_info
        tables = []
        for _, row in info.loc[info["select"].astype(bool)].iterrows():
            signal = row["signal"]
            if not isinstance(signal, dict):
                continue
            values = window_responses(signal[key], signal["timeline"], baseline, response)
            values = values[np.isfinite(values)]  # cells not observed enough, and empty rows
            tables.append(pd.DataFrame(dict(well_name=row["well_name"], pos_name=row["pos_name"], response=values)))
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def run(self) -> None:
        from celldetective.gui.tableUI import two_sample_test_table

        problem = self.alignment_problem()
        if problem:
            generic_message(problem)
            return None
        try:
            cells = self.responses()
        except ValueError as e:
            generic_message(str(e))
            return None
        if cells.empty or cells["well_name"].nunique() < 2:
            generic_message("Not enough cells observed over both windows in at least two wells.")
            return None

        self.cells = cells
        self.result_windows = [
            two_sample_test_table(cells, "response", ["well_name"], method) for method in ("ks_2samp", "cliffs_delta")
        ]
        for window in self.result_windows:
            window.show()
